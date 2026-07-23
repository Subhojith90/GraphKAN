from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import yaml

from stage4f1b_tx import (
    MANDATORY_PAYLOADS,
    Stage4F1BRegistry,
    expected_runs,
    hash_artifacts_before_complete,
    load_config,
    reconcile_campaign,
    sha256_file,
    validate_run_bundle_semantic,
    write_json,
    write_valid_synthetic_bundle,
)


def init_cmd(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    root = Path(cfg["campaign"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    registry = Stage4F1BRegistry(root / "registry.sqlite")
    manifest = registry.register_campaign(cfg_path, cfg)
    write_json(root / "campaign_manifest.json", manifest)
    runs = expected_runs(cfg)
    ledger = root / "expected_runs.jsonl"
    ledger.write_text("".join(json.dumps(run.to_dict(), sort_keys=True) + "\n" for run in runs))
    output = {
        "manifest": manifest,
        "expected_runs_ledger": str(ledger),
        "expected_runs_ledger_sha256": sha256_file(ledger),
        "registry_summary": registry.summary(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


def _rehash_bundle(bundle: Path) -> None:
    write_json(bundle / "artifact_manifest.json", {"schema_version": "stage4f-native-v1c", "files": hash_artifacts_before_complete(bundle)})
    complete = json.loads((bundle / "COMPLETE.json").read_text())
    complete["artifact_manifest_sha256"] = sha256_file(bundle / "artifact_manifest.json")
    time.sleep(0.002)
    write_json(bundle / "COMPLETE.json", complete)


def semantic_test_cmd(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    run = expected_runs(cfg)[0]
    root = Path(args.output)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    valid = write_valid_synthetic_bundle(root, run)
    expected = run.to_dict()
    ok, errors = validate_run_bundle_semantic(valid, expected_spec=expected)
    if not ok:
        raise SystemExit(f"valid bundle rejected: {errors}")
    cases: dict[str, dict] = {}

    def mutate_case(name, mutator, rehash=True):
        case = root / f"case_{name}"
        shutil.copytree(valid, case)
        mutator(case)
        if rehash:
            _rehash_bundle(case)
        accepted, case_errors = validate_run_bundle_semantic(case, expected_spec=expected)
        cases[name] = {"accepted": accepted, "errors": case_errors}
        if accepted:
            raise RuntimeError(f"invalid bundle accepted: {name}")

    mutate_case("metrics_only_run_id", lambda p: (p / "metrics.csv").write_text(f"run_id\n{run.run_id}\n"))
    mutate_case("nonnumeric_physics", lambda p: (p / "metrics.csv").write_text(
        f"run_id,status,scientific_validity,model,boundary,delta,c6,seed,energy_mean,energy_error\n"
        f"{run.run_id},COMPLETED,VALID,{run.model},{run.boundary},{run.delta},{run.c6},{run.seed},banana,nope\n"
    ))
    mutate_case("invalid_status", lambda p: (p / "metrics.csv").write_text(
        f"run_id,status,scientific_validity,model,boundary,delta,c6,seed,energy_mean,energy_error\n"
        f"{run.run_id},MAGIC,VALID,{run.model},{run.boundary},{run.delta},{run.c6},{run.seed},-1,0.1\n"
    ))
    mutate_case("one_row_history", lambda p: (p / "history.csv").write_text(f"run_id,step,energy_mean,energy_error\n{run.run_id},1,-1,0.1\n"))
    mutate_case("wrong_resolved_identity", lambda p: (p / "resolved_config.yaml").write_text("transaction_identity:\n  run_id: WRONG\n"))
    mutate_case("corrupt_checkpoint_rehashed", lambda p: (p / "checkpoint.npz").write_bytes(b"not-a-checkpoint"))
    mutate_case("false_complete_manifest_hash", lambda p: write_json(p / "COMPLETE.json", {
        "schema_version": "stage4f-native-v1c", "run_id": run.run_id, "attempt_id": "synthetic-attempt",
        "artifact_validation": "PASS", "artifact_manifest_sha256": "0" * 64,
    }), rehash=False)
    mutate_case("manifest_omits_mandatory", lambda p: write_json(p / "artifact_manifest.json", {
        "schema_version": "stage4f-native-v1c", "files": {k: v for k, v in hash_artifacts_before_complete(p).items() if k != "metrics.csv"}
    }), rehash=False)
    mutate_case("foreign_relabelled_bundle", lambda p: write_json(p / "run_manifest.json", {
        **json.loads((p / "run_manifest.json").read_text()), "run_id": "FOREIGN-RUN"
    }))
    mutate_case("path_traversal_manifest", lambda p: write_json(p / "artifact_manifest.json", {
        "schema_version": "stage4f-native-v1c", "files": {**hash_artifacts_before_complete(p), "../escape": "0" * 64}
    }), rehash=False)

    output = {"semantic_validator": "PASS", "valid_bundle": True, "false_accepts": 0, "invalid_cases": cases}
    write_json(root / "semantic_validator_summary.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


def lease_test_cmd(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    root = Path(args.output)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    cfg["campaign"]["output_root"] = str(root)
    cfg["scheduler"]["max_attempts"] = 2
    local_cfg = root / "campaign.yaml"
    local_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    registry = Stage4F1BRegistry(root / "registry.sqlite")
    registry.register_campaign(local_cfg, cfg)
    first = registry.claim_next("core", "worker-A", lease_seconds=1)
    if first is None:
        raise RuntimeError("failed to claim first run")
    time.sleep(float(args.wait))
    second = registry.claim_next("core", "worker-B", lease_seconds=2)
    if second is None:
        raise RuntimeError("stale lease was not reclaimed")
    stale_heartbeat = registry.mark_heartbeat(first["run_id"], "worker-A", first["attempt_id"], first["claim_token"], 2)
    stale_complete_blocked = False
    try:
        registry.mark_complete(first["run_id"], "worker-A", first["attempt_id"], first["claim_token"], root / "late", "0" * 64)
    except RuntimeError:
        stale_complete_blocked = True
    if stale_heartbeat or not stale_complete_blocked:
        raise RuntimeError("stale claim fencing failed")
    output = {
        "lease_retry": "PASS",
        "stale_heartbeat_rejected": not stale_heartbeat,
        "stale_completion_rejected": stale_complete_blocked,
        "first_attempt_id": first["attempt_id"],
        "second_attempt_id": second["attempt_id"],
        "registry_summary": registry.summary(),
    }
    write_json(root / "lease_retry_summary.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


def validate_cmd(args: argparse.Namespace) -> None:
    root = Path(args.campaign)
    registry_path = root / "registry.sqlite"
    if not registry_path.is_file():
        raise SystemExit(f"missing registry: {registry_path}")
    registry = Stage4F1BRegistry(registry_path)
    registry.init()
    reconciliation = reconcile_campaign(root, registry)
    summary = registry.summary()
    status_counts = summary["status_counts"]
    terminal = sum(status_counts.get(name, 0) for name in ("COMPLETE", "FAILED_FINAL", "QUARANTINED"))
    errors = list(reconciliation["registry_filesystem_errors"])
    if args.require_terminal and terminal != summary["total_expected"]:
        errors.append({"error": f"campaign is not terminal: {terminal}/{summary['total_expected']}"})
    if args.require_complete and status_counts.get("COMPLETE", 0) != summary["total_expected"]:
        errors.append({"error": f"campaign is not completely successful: {status_counts.get('COMPLETE', 0)}/{summary['total_expected']}"})
    output = {
        "schema_version": "stage4f-native-v1c",
        "campaign": str(root),
        "registry_summary": summary,
        **reconciliation,
        "validation": "PASS" if not errors else "FAIL",
        "blocking_errors": errors,
    }
    if not args.no_write:
        destination = Path(args.output) if args.output else root / "stage4f1c_validation_summary.json"
        write_json(destination, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4F.1c schema-fenced campaign CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    command = sub.add_parser("init")
    command.add_argument("--config", required=True)
    command.set_defaults(func=init_cmd)
    command = sub.add_parser("semantic-test")
    command.add_argument("--config", default="configs/stage4f1c_campaign.yaml")
    command.add_argument("--output", default="results_stage4f1c/semantic_tests")
    command.set_defaults(func=semantic_test_cmd)
    command = sub.add_parser("lease-test")
    command.add_argument("--config", default="configs/stage4f1c_campaign.yaml")
    command.add_argument("--output", default="results_stage4f1c/lease_tests")
    command.add_argument("--wait", default="1.2")
    command.set_defaults(func=lease_test_cmd)
    command = sub.add_parser("validate")
    command.add_argument("--campaign", required=True)
    command.add_argument("--strict-integrity", action="store_true", help="Compatibility flag; strict checks are always enabled.")
    command.add_argument("--reconcile-registry", action="store_true", help="Compatibility flag; reconciliation is always enabled.")
    command.add_argument("--require-terminal", action="store_true")
    command.add_argument("--require-complete", action="store_true")
    command.add_argument("--no-write", action="store_true", help="Validate without modifying the campaign tree.")
    command.add_argument("--output", help="Write the validation record to this path instead of the campaign tree.")
    command.set_defaults(func=validate_cmd)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from stage4f1b_tx import (
    Stage4F1BRegistry,
    atomic_promote_no_overwrite,
    expected_runs,
    load_config,
    make_campaign_id,
    protocol_hash_from_config,
    sha256_file,
    validate_run_bundle_semantic,
    write_json,
    write_release_manifest,
    write_valid_synthetic_bundle,
)


def init_cmd(args):
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    root = Path(cfg["campaign"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)

    reg = Stage4F1BRegistry(root / "registry.sqlite")
    manifest = reg.register_campaign(cfg_path, cfg)
    write_json(root / "campaign_manifest.json", manifest)
    out = {"manifest": manifest, "registry_summary": reg.summary()}
    print(json.dumps(out, indent=2, sort_keys=True))


def semantic_test_cmd(args):
    cfg = load_config(Path(args.config))
    runs = expected_runs(cfg)
    run = runs[0]

    root = Path(args.output)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    valid = write_valid_synthetic_bundle(root, run)
    ok, errors = validate_run_bundle_semantic(valid, run.run_id)
    if not ok:
        raise SystemExit(f"valid bundle rejected: {errors}")

    cases = {}

    def mutate_case(name, mutator):
        case = root / f"case_{name}"
        shutil.copytree(valid, case)
        mutator(case)
        ok2, err2 = validate_run_bundle_semantic(case, run.run_id)
        cases[name] = {"valid": ok2, "errors": err2}
        if ok2:
            raise RuntimeError(f"invalid bundle accepted: {name}")

    mutate_case("tampered_metrics", lambda p: (p / "metrics.csv").write_text(f"run_id,status,energy_error\n{run.run_id},VALID,999.0\n"))
    mutate_case("false_artifact_hash", lambda p: (p / "artifact_manifest.json").write_text('{"files":{"metrics.csv":"0000"}}\n'))
    mutate_case("missing_checkpoint_payload", lambda p: (p / "checkpoint.npz").unlink())
    mutate_case("corrupt_checkpoint_manifest", lambda p: (p / "checkpoint_manifest.json").write_text("{bad json"))
    mutate_case("checkpoint_unreadable", lambda p: (p / "checkpoint_manifest.json").write_text(json.dumps({"run_id": run.run_id, "checkpoint": "checkpoint.npz", "sha256": sha256_file(p / "checkpoint.npz"), "readable": False}) + "\n"))
    mutate_case("wrong_metrics_run_id", lambda p: (p / "metrics.csv").write_text("run_id,status,energy_error\nWRONG,VALID,0.0\n"))
    mutate_case("wrong_complete_run_id", lambda p: (p / "COMPLETE.json").write_text(json.dumps({"run_id": "WRONG", "artifact_validation": "PASS"}) + "\n"))
    mutate_case("complete_not_last", lambda p: ((p / "metrics.csv").write_text(f"run_id,status,energy_error\n{run.run_id},VALID,0.0\n")))

    out = {"semantic_validator": "PASS", "valid_bundle": True, "invalid_cases": cases}
    write_json(root / "semantic_validator_summary.json", out)
    print(json.dumps(out, indent=2, sort_keys=True))


def lease_test_cmd(args):
    cfg = load_config(Path(args.config))
    root = Path(args.output)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    cfg["campaign"]["output_root"] = str(root)
    local_cfg = root / "campaign.yaml"
    import yaml
    local_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

    reg = Stage4F1BRegistry(root / "registry.sqlite")
    reg.register_campaign(local_cfg, cfg)

    first = reg.claim_next("core", "worker-A", lease_seconds=1)
    if first is None:
        raise RuntimeError("failed to claim first run")
    time_wait = float(args.wait)
    import time
    time.sleep(time_wait)

    second = reg.claim_next("core", "worker-B", lease_seconds=1)
    if second is None:
        raise RuntimeError("stale lease was not reclaimed")

    third = reg.claim_next("core", "worker-C", lease_seconds=1)
    if third is None:
        raise RuntimeError("pending work starved after stale reclaim")

    out = {
        "lease_retry": "PASS",
        "first_claim": first["run_id"],
        "second_claim_after_stale": second["run_id"],
        "third_claim_pending_progress": third["run_id"],
        "registry_summary": reg.summary(),
    }
    write_json(root / "lease_retry_summary.json", out)
    print(json.dumps(out, indent=2, sort_keys=True))


def validate_cmd(args):
    root = Path(args.campaign)
    reg = Stage4F1BRegistry(root / "registry.sqlite")
    invalid = []
    complete_checked = 0

    for d in sorted((root / "runs").glob("*")) if (root / "runs").exists() else []:
        if d.is_dir() and (d / "COMPLETE.json").exists():
            complete_checked += 1
            ok, errors = validate_run_bundle_semantic(d)
            if not ok:
                invalid.append({"bundle": str(d), "errors": errors})

    out = {
        "campaign": str(root),
        "registry_summary": reg.summary(),
        "complete_bundles_checked": complete_checked,
        "invalid_complete_bundles": invalid,
    }
    write_json(root / "stage4f1b_validation_summary.json", out)
    print(json.dumps(out, indent=2, sort_keys=True))
    if invalid:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("init")
    q.add_argument("--config", required=True)
    q.set_defaults(func=init_cmd)

    q = sub.add_parser("semantic-test")
    q.add_argument("--config", default="configs/stage4f1b_campaign.yaml")
    q.add_argument("--output", default="results_stage4f1b/semantic_tests")
    q.set_defaults(func=semantic_test_cmd)

    q = sub.add_parser("lease-test")
    q.add_argument("--config", default="configs/stage4f1b_campaign.yaml")
    q.add_argument("--output", default="results_stage4f1b/lease_tests")
    q.add_argument("--wait", default="1.2")
    q.set_defaults(func=lease_test_cmd)

    q = sub.add_parser("validate")
    q.add_argument("--campaign", required=True)
    q.set_defaults(func=validate_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

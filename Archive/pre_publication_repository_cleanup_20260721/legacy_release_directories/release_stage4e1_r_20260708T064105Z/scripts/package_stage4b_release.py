#!/usr/bin/env python3
"""Create a deterministic, auditable Stage 4B release archive after completed controls."""
from __future__ import annotations
import argparse, hashlib, json, shutil, tarfile
from pathlib import Path

REQUIRED = [
    "clean_replay/stage4b_clean_replay.log",
    "n8_component_replay/raw/stage4b_metrics.csv",
    "n8_controls_core/raw/stage4b_metrics.csv",
    "n8_capacity_compute/raw/stage4b_metrics.csv",
    "evidence_bundle/batch_index.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results_stage4b")
    p.add_argument("--output", default="release_stage4b")
    a = p.parse_args()
    results = Path(a.results_root)
    missing = [x for x in REQUIRED if not (results / x).exists()]
    if missing:
        raise SystemExit("Release blocked; missing required artifacts: " + ", ".join(missing))
    output = Path(a.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(results, output / "results_stage4b")
    for name in ["README.md", "STAGE_4B_README.md", "REPRODUCIBILITY.md", "environment-nqs-py312-macos-arm64-stage4b.yml"]:
        shutil.copy2(name, output / name)
    (output / "configs").mkdir()
    for cfg in Path("configs").glob("netket_stage4b_*.yaml"):
        shutil.copy2(cfg, output / "configs" / cfg.name)
    (output / "scripts").mkdir()
    for name in ["run_stage4b_controls.py", "summarize_stage4b.py", "summarize_stage4b_all.py", "run_stage4b_clean_replay.sh", "package_stage4b_release.py", "_common.py", "preflight_nqs.py", "check_netket_conventions.py"]:
        shutil.copy2(Path("scripts") / name, output / "scripts" / name)
    shutil.copytree("src", output / "src")
    shutil.copytree("tests", output / "tests")
    manifest = {}
    for file in sorted(p for p in output.rglob("*") if p.is_file()):
        manifest[str(file.relative_to(output))] = {"sha256": sha256(file), "bytes": file.stat().st_size}
    (output / "release_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tar = output.with_suffix(".tar.gz")
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(output, arcname=output.name)
    tar.with_suffix(tar.suffix + ".sha256").write_text(f"{sha256(tar)}  {tar.name}\n")
    print("Wrote", tar)

if __name__ == "__main__":
    main()

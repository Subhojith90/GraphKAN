from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from stage4f_tx import sha256_file, write_release_root_manifest


def main():
    parser = argparse.ArgumentParser(description="Package Stage 4F.1 repair handoff")
    parser.add_argument("--name", default=None)
    parser.add_argument("--campaign", default="results_stage4f1/transactional_campaign")
    args = parser.parse_args()

    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.name or f"release_stage4f1_transactional_scheduler_{stamp}"
    root = Path(name)

    if root.exists():
        shutil.rmtree(root)

    root.mkdir(parents=True)

    for path in [
        "docs/tracking/stage4f/README_STAGE4F1.md",
        "configs/stage4f_campaign.yaml",
        "configs/stage4f_fault_injection.yaml",
        "configs/stage4f_runtime_calibration.yaml",
        "scripts/stage4f_tx.py",
        "scripts/stage4f_campaign.py",
        "scripts/stage4f_worker.py",
        "scripts/summarize_stage4f.py",
        "scripts/package_stage4f_release.py",
        "tests/test_stage4f1_transactional_scheduler.py",
        "reproducibility/environment_stage4f_portable.yml",
    ]:
        p = Path(path)
        if p.exists():
            target = root / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)

    campaign = Path(args.campaign)
    if campaign.exists():
        shutil.copytree(campaign, root / "results_stage4f1" / campaign.name)

    write_release_root_manifest(root)

    zip_path = Path(f"{name}.zip")
    sidecar = Path(f"{name}.zip.sha256")
    if zip_path.exists():
        zip_path.unlink()
    if sidecar.exists():
        sidecar.unlink()

    subprocess.check_call(["zip", "-qr", str(zip_path), str(root)])
    sidecar.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n")

    print(zip_path.resolve())
    print(sidecar.resolve())


if __name__ == "__main__":
    main()

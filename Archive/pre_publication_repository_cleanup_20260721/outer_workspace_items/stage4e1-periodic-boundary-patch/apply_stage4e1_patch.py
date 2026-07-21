from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Stage 4E.1 periodic-boundary patch")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    src_root = here / "files"
    project = Path(args.project_root).resolve()
    if not (project / "src" / "graphkan_amo_nqs").exists():
        raise SystemExit(f"Not a GraphKAN project root: {project}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = project / ".stage4e1_backup" / stamp
    changed = []
    for source in sorted(src_root.rglob("*")):
        if source.is_dir():
            continue
        rel = source.relative_to(src_root)
        target = project / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup_target = backup / rel
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_target)
        shutil.copy2(source, target)
        changed.append(str(rel))
    print(f"Applied {len(changed)} files.")
    print(f"Backup directory: {backup}")
    for item in changed:
        print(item)


if __name__ == "__main__":
    main()

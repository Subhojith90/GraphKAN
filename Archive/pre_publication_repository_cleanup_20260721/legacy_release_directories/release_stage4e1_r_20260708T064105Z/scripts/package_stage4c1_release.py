#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4c-results-root", required=True)
    ap.add_argument("--output", default="release_stage4c1")
    args = ap.parse_args()
    rr = Path(args.stage4c_results_root).expanduser().resolve()
    out = Path(args.output)
    if out.exists(): shutil.rmtree(out)
    req = [rr / "backups/stage4c_metrics_final_raw.csv", rr / "backups/stage4c_metrics_final_raw.csv.sha256", rr / "backups/repaired/aggregated_repaired/stage4c_decision_gate_repaired.csv", rr / "backups/repaired/aggregated_repaired/stage4c_cell_gate.csv"]
    missing = [str(x) for x in req if not x.exists()]
    if missing: raise SystemExit("Missing explicit Stage 4C.1 inputs:\n" + "\n".join(missing))
    out.mkdir(parents=True)
    cp(req[0], out / "raw/stage4c_metrics_final_raw.csv")
    for q in (rr / "backups/repaired/aggregated_repaired").glob("*"):
        if q.is_file(): cp(q, out / "repaired" / q.name)
    logdir = rr / "backups/logs"
    if logdir.exists():
        for q in logdir.glob("*.log"): cp(q, out / "logs" / q.name)
    test = """from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parents[1]
def d(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def test_layout():
 assert (ROOT/'release_manifest.json').exists()
 assert (ROOT/'raw/stage4c_metrics_final_raw.csv').exists()
def test_manifest():
 m=json.loads((ROOT/'release_manifest.json').read_text())
 for rel, expected in m['files'].items(): assert d(ROOT/rel)==expected
"""
    path = out / "release_tests/test_release_artifact.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(test)
    files = {}
    for q in sorted(out.rglob("*")):
        if q.is_file() and q.name not in {"release_manifest.json", "MANIFEST.sha256"}:
            files[str(q.relative_to(out))] = digest(q)
    (out / "release_manifest.json").write_text(json.dumps({"schema":"stage4c1-release-v2", "files":files}, indent=2, sort_keys=True) + "\n")
    lines=[]
    for q in sorted(out.rglob("*")):
        if q.is_file() and q.name != "MANIFEST.sha256": lines.append(f"{digest(q)}  {q.relative_to(out)}")
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print("Created", out)
if __name__ == "__main__": main()

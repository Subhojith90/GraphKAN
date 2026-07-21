#!/usr/bin/env python3
"""Build a self-testing, portable Stage 4C.1 evidence release from frozen Stage 4C results."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

def sha(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,dst)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-root',default='results_stage4c'); ap.add_argument('--output',default='release_stage4c1'); a=ap.parse_args()
    rr=Path(a.results_root); out=Path(a.output)
    if out.exists(): shutil.rmtree(out)
    out.mkdir()
    required=[
      rr/'backups/stage4c_metrics_final_raw.csv',
      rr/'backups/stage4c_metrics_final_raw.csv.sha256',
      rr/'backups/repaired/aggregated_repaired/stage4c_decision_gate_repaired.csv',
      rr/'backups/repaired/aggregated_repaired/stage4c_cell_gate.csv',
      Path('scripts/summarize_stage4c_repaired.py'),
      Path('tests/test_stage4c_repaired_pairing.py'),
    ]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit('Missing required Stage 4C.1 inputs:\n'+'\n'.join(missing))
    cp(rr/'backups/stage4c_metrics_final_raw.csv',out/'raw/stage4c_metrics_final_raw.csv')
    # Generate portable sidecar rooted inside release.
    raw=out/'raw/stage4c_metrics_final_raw.csv'
    (out/'raw/stage4c_metrics_final_raw.csv.sha256').write_text(f'{sha(raw)}  raw/stage4c_metrics_final_raw.csv\n')
    for p in (rr/'backups/repaired/aggregated_repaired').glob('*'):
        if p.is_file(): cp(p,out/'repaired'/p.name)
    cp(Path('scripts/summarize_stage4c_repaired.py'),out/'scripts/summarize_stage4c_repaired.py')
    cp(Path('tests/test_stage4c_repaired_pairing.py'),out/'release_tests/test_stage4c_repaired_pairing.py')
    # portable test checks artifact layout and hashes, no source imports.
    (out/'release_tests/test_release_artifact.py').write_text('''from pathlib import Path\nimport hashlib\nROOT=Path(__file__).resolve().parents[1]\ndef digest(p):\n h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()\ndef test_release_manifest_and_layout():\n m=ROOT/"release_manifest.json"; assert m.exists()\n assert (ROOT/"scripts/summarize_stage4c_repaired.py").exists()\n assert (ROOT/"raw/stage4c_metrics_final_raw.csv").exists()\ndef test_portable_raw_sidecar():\n line=(ROOT/"raw/stage4c_metrics_final_raw.csv.sha256").read_text().strip()\n expected, rel=line.split(None,1)\n assert digest(ROOT/rel.strip())==expected\n''')
    # preserve logs if available
    for p in (rr/'backups/logs').glob('*.log') if (rr/'backups/logs').exists() else []:
        cp(p,out/'logs'/p.name)
    manifest={}
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in {'MANIFEST.sha256','release_manifest.json'}: manifest[str(p.relative_to(out))]=sha(p)
    (out/'release_manifest.json').write_text(json.dumps({'schema':'stage4c1-release-v1','files':manifest},indent=2,sort_keys=True)+'\n')
    # manifest includes itself after generation only via manifest sha separate.
    lines=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='MANIFEST.sha256': lines.append(f'{sha(p)}  {p.relative_to(out)}')
    (out/'MANIFEST.sha256').write_text('\n'.join(lines)+'\n')
    print('Created',out)
    print('Run: cd',out,'&& python -m pytest -q release_tests && shasum -a 256 -c MANIFEST.sha256')
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Package a trimmed, self-testable Stage 4C release without inherited historical tests."""
from __future__ import annotations
import argparse, hashlib, json, shutil, tarfile
from pathlib import Path
REQUIRED = ["clean_replay/stage4c_clean_replay.log", "hard_patch/raw/stage4c_metrics.csv", "evidence_bundle/batch_index.csv"]
def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--results-root", default="results_stage4c"); p.add_argument("--output", default="release_stage4c"); a=p.parse_args()
    results=Path(a.results_root); missing=[x for x in REQUIRED if not (results/x).exists()]
    if missing: raise SystemExit("Release blocked; missing required artifacts: "+", ".join(missing))
    out=Path(a.output)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True); shutil.copytree(results,out/'results_stage4c')
    documents = {
        'README.md': 'README.md',
        'REPRODUCIBILITY.md': 'REPRODUCIBILITY.md',
        'docs/tracking/stage_history/STAGE_4C_README.md': 'STAGE_4C_README.md',
        'environment-nqs-py312-macos-arm64-stage4c.yml': 'environment-nqs-py312-macos-arm64-stage4c.yml',
    }
    for source, destination in documents.items(): shutil.copy2(source,out/destination)
    (out/'configs').mkdir(); shutil.copy2('configs/netket_stage4c_hard_patch.yaml',out/'configs/netket_stage4c_hard_patch.yaml')
    (out/'scripts').mkdir()
    for n in ['run_stage4c_patch.py','summarize_stage4c.py','run_stage4c_batch.sh','run_stage4c_clean_replay.sh','package_stage4c_release.py','_common.py','preflight_nqs.py','check_netket_conventions.py']: shutil.copy2(Path('scripts')/n,out/'scripts'/n)
    shutil.copytree('src',out/'src'); shutil.copytree('release_tests',out/'release_tests')
    manifest={}
    for f in sorted(x for x in out.rglob('*') if x.is_file()): manifest[str(f.relative_to(out))]={'sha256':sha256(f),'bytes':f.stat().st_size}
    (out/'release_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    tar=out.with_suffix('.tar.gz')
    with tarfile.open(tar,'w:gz') as tf: tf.add(out,arcname=out.name)
    tar.with_suffix(tar.suffix+'.sha256').write_text(f'{sha256(tar)}  {tar.name}\n')
    print('Wrote',tar)
if __name__=='__main__': main()

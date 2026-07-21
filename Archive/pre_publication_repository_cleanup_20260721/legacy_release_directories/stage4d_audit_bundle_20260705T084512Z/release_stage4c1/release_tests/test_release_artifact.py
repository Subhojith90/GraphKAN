from pathlib import Path
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

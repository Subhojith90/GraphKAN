# Static regression sentinel: repaired aggregator must use full cell+seed pairing and fail gate below validity threshold.
from pathlib import Path
p=Path(__file__).parents[1]/"scripts"/"summarize_stage4c_repaired.py"
s=p.read_text()
assert 'CELL + ["seed"]' in s
assert 'candidate_rate>=args.min_candidate_exact_valid_rate' in s
assert 'status="PASS"' in s
assert 'Incomplete patch cells present' in s

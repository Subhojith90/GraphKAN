#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
from graphkan_amo_nqs.training.nqs_preflight import preflight_dict

p=argparse.ArgumentParser(description='Check the NetKet/Flax runtime before any VMC launch.')
p.add_argument('--output', default='results/environment/nqs_preflight.json')
p.add_argument('--allow-missing', action='store_true', help='exit 0 while reporting missing optional NQS dependencies')
a=p.parse_args(); report=preflight_dict(require_netket=True)
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['ok'] or a.allow_missing else 2)

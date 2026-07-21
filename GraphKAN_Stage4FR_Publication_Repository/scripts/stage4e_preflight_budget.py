#!/usr/bin/env python3
"""Estimate Stage 4E run count; prevents accidental full-grid launches."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--max-runs',type=int,default=100)
    a=ap.parse_args(); c=yaml.safe_load(Path(a.config).read_text())
    n=len(c['chains']['n_sites'])*len(c['parameters']['delta'])*len(c['parameters']['c6'])*len(c['training']['seeds'])*len(c['model_specs'])
    print({'profile':c['profile'],'registered_runs':n,'max_runs':a.max_runs,'ok':n<=a.max_runs})
    if n>a.max_runs: raise SystemExit('Budget guard blocked this launch. Use a smaller registered profile or explicitly increase --max-runs.')
if __name__=='__main__': main()

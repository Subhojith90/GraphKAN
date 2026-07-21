#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from _common import load_config, hash_dict, write_status_not_run


def main(default_profile: str):
    ap = argparse.ArgumentParser(description=f"Record {default_profile} as not executed in this CPU artifact.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); profile = cfg.get("profile", default_profile)
    reason = cfg.get("reason", "Profile not run in CPU artifact.")
    write_status_not_run(profile, reason, hash_dict(cfg), Path(cfg.get("output_root", "results")))
    print(f"{profile}: not run - {reason}")

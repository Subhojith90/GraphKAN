#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import json
from graphkan_amo_nqs.environment import collect_environment
from graphkan_amo_nqs.manifests import atomic_write_json, hash_file


def main():
    ap=argparse.ArgumentParser(description="Create release manifest with file hashes and environment status.")
    ap.parse_args()
    root=Path('.')
    files=[]
    for pattern in [
        "*.md", "docs/*.md", "configs/*.yaml",
        "src/**/*.py", "scripts/*.py", "tests/*.py",
        "results/**/*.csv", "results/**/*.npz",
        "figures/*.pdf", "tables/*.tex", "tables/*.csv",
        "paper/production_analysis_report.tex", "paper/production_analysis_report.pdf", "paper/references.bib",
        "pyproject.toml", "LICENSE", "CITATION.cff",
    ]:
        files.extend(root.glob(pattern))
    manifest={
        "artifact":"graphkan-amo-nqs-real",
        "version":"0.1.0",
        "environment":collect_environment(),
        "files":{str(f):hash_file(f) for f in files if f.exists()},
        "unexecuted_profiles":["rydberg_pra_core","rydberg_geometry_transfer","graphkan_full","robustness_core","bose_hubbard_optional","ibm_optional_sanity"],
        "main_claims_validated":["CPU Hamiltonian validation","exact-fullsum pilot plumbing"],
        "main_claims_not_validated":["KAN superiority","Graph-KAN value","PRA readiness","Quantum/PRX readiness","hardware validation"],
    }
    atomic_write_json(root/"release"/"manifest.json", manifest)
    print("release manifest written")

if __name__=="__main__":
    main()

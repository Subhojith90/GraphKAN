#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from graphkan_amo_nqs.reporting.readiness import readiness_summary
from graphkan_amo_nqs.manifests import atomic_write_dataframe


def main():
    ap=argparse.ArgumentParser(description="Check top-journal readiness gates.")
    ap.parse_args()
    gates=pd.read_csv("results/aggregated/readiness_gates.csv") if Path("results/aggregated/readiness_gates.csv").exists() else pd.DataFrame()
    df=readiness_summary(gates)
    atomic_write_dataframe("results/aggregated/top_journal_readiness.csv", df)
    print("top-journal readiness written")

if __name__=="__main__":
    main()

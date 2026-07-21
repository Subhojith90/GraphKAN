#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from graphkan_amo_nqs.reporting.claim_audit import validate_claims
from graphkan_amo_nqs.manifests import atomic_write_dataframe


def main():
    ap=argparse.ArgumentParser(description="Validate safe/unsafe claim ladder against produced evidence.")
    ap.parse_args()
    df=validate_claims("results")
    atomic_write_dataframe("results/aggregated/claim_validation.csv", df)
    print("claim validation written")

if __name__=="__main__":
    main()

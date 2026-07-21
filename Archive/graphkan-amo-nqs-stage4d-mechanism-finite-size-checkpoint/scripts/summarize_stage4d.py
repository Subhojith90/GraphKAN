#!/usr/bin/env python3
"""Stage 4D descriptive mechanism and finite-size summary.

This script deliberately makes no universal-superiority decision. It reports
model/cell health, exact metrics, edge-vs-bulk density error, and endpoint
feature trade-offs from native Stage 4D raw outputs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def median_or_nan(x):
    return float(x.median()) if len(x) else float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    args = ap.parse_args()
    root = Path(args.results)
    raw = root / 'raw'
    metrics_path = raw / 'stage4d_metrics.csv'
    site_path = raw / 'stage4d_site_observables.csv'
    if not metrics_path.exists():
        raise SystemExit(f'Missing {metrics_path}')
    df = pd.read_csv(metrics_path)
    if 'status' in df:
        df = df[df.status == 'COMPLETED'].copy()
    out = root / 'aggregated'
    out.mkdir(parents=True, exist_ok=True)
    group = ['n_sites','delta','c6','model_label','model_kind','basis_family','n_parameters']
    summary = (df.groupby(group, dropna=False)
        .agg(runs=('run_id','size'),
             exact_valid=('scientific_validity_exact', lambda x:(x=='VALID').sum()),
             sample_valid=('scientific_validity_sample', lambda x:(x=='VALID').sum()),
             median_exact_energy=('exact_model_abs_energy_error_to_ed','median'),
             median_exact_density=('exact_model_density_mae_to_ed','median'),
             median_exact_corr=('exact_model_connected_correlation_mae_to_ed','median'),
             median_wall_seconds=('wall_seconds','median'))
        .reset_index())
    summary['exact_valid_rate'] = summary.exact_valid / summary.runs
    summary['sample_valid_rate'] = summary.sample_valid / summary.runs
    summary.to_csv(out/'stage4d_model_cell_summary.csv',index=False)

    if site_path.exists():
        site = pd.read_csv(site_path)
        site['region'] = 'bulk'
        site.loc[site.site.isin([0]) | (site.site == site.n_sites - 1), 'region'] = 'edge'
        reg = (site.groupby(['n_sites','delta','c6','model_label','region'])
               .agg(median_exact_abs_density_error=('exact_model_abs_error','median'),
                    mean_exact_abs_density_error=('exact_model_abs_error','mean'),
                    rows=('run_id','size')).reset_index())
        reg.to_csv(out/'stage4d_site_region_density_errors.csv',index=False)
    else:
        reg = pd.DataFrame()

    # Feature-focused comparison against sum, preserving per-cell medians.
    focus = summary[summary.model_label.str.startswith('rbf_endpoint_', na=False)].copy()
    focus.to_csv(out/'stage4d_endpoint_feature_summary.csv', index=False)

    note = '''Stage 4D summaries are descriptive diagnostics. They do not constitute a
universal performance gate. Interpret exact energy, density, correlation, health,
runtime, and edge/bulk errors jointly. Nominal parameter labels may vary with N;
use n_parameters in this table for exact counts.\n'''
    (out/'README_STAGE4D_SUMMARY.txt').write_text(note)
    print('Wrote', out)
    print(summary.to_string(index=False))
    if len(reg):
        print('\nSite-region density diagnostic:')
        print(reg.to_string(index=False))

if __name__ == '__main__':
    main()

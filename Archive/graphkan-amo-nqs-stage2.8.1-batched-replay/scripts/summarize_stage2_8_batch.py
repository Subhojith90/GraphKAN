#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--results',required=True); a=p.parse_args()
f=Path(a.results)/'raw'/'stage2_7_metrics.csv'
df=pd.read_csv(f)
cols=[c for c in ['model_label','seed','scientific_validity','abs_energy_error_to_ed','exact_model_abs_energy_error_to_ed','sample_density_mae_to_ed','exact_model_density_mae_to_ed','sample_connected_correlation_mae_to_ed','exact_model_connected_correlation_mae_to_ed','wall_seconds','health_flags'] if c in df.columns]
print(df[cols].to_string(index=False))
print('\nCounts:')
print(df.groupby(['model_label','scientific_validity']).size().rename('runs').to_string())

#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--input', default='results_stage2_4/raw/netket_vmc_stabilized_metrics.csv')
ap.add_argument('--output', default='results_stage2_4/aggregated/stage2_4_health_summary.csv')
args=ap.parse_args()
df=pd.read_csv(args.input)
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
summary=(df.groupby(['model','scientific_validity'],dropna=False)
         .size().reset_index(name='runs'))
summary.to_csv(args.output,index=False)
print(summary.to_string(index=False))
print('\nValid-only ED errors:')
valid=df[df['scientific_validity'].eq('VALID')]
if valid.empty:
    print('No scientifically valid run. Do not compare architectures.')
else:
    print(valid.groupby('model')['abs_energy_error_to_ed'].agg(['count','median','mean','max']).to_string())

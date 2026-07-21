#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--input',default='results_stage2_6/raw/netket_vmc_stabilized_metrics.csv'); ap.add_argument('--output',default='results_stage2_6/aggregated/stage2_6_summary.csv'); args=ap.parse_args()
df=pd.read_csv(args.input); Path(args.output).parent.mkdir(parents=True,exist_ok=True)
summary=(df.groupby(['model','scientific_validity']).size().reset_index(name='runs')); summary.to_csv(args.output,index=False); print(summary.to_string(index=False))
valid=df[df.scientific_validity.eq('VALID')].copy()
cols=['abs_energy_error_to_ed','density_mae_to_ed','connected_correlation_mae_to_ed','wall_seconds']
if valid.empty: print('No valid rows. Do not compare architectures.'); raise SystemExit(1)
print('\nValid-only aggregate metrics:'); print(valid.groupby('model')[cols].median().to_string())
paired=[]
for baseline in ['smooth_jastrow','mlp_pair','rbm']:
 a=valid[valid.model.eq('kan_jastrow')].set_index(['instance_id','seed'])
 b=valid[valid.model.eq(baseline)].set_index(['instance_id','seed'])
 x=a.join(b,lsuffix='_kan',rsuffix='_base',how='inner')
 if len(x):
  paired.append({'comparison':f'kan_minus_{baseline}','n_pairs':len(x),'median_energy_error_delta':float((x.abs_energy_error_to_ed_kan-x.abs_energy_error_to_ed_base).median()),'median_density_mae_delta':float((x.density_mae_to_ed_kan-x.density_mae_to_ed_base).median()),'median_corr_mae_delta':float((x.connected_correlation_mae_to_ed_kan-x.connected_correlation_mae_to_ed_base).median())})
pd.DataFrame(paired).to_csv(Path(args.output).with_name('stage2_6_paired_kan_deltas.csv'),index=False)
print('\nPaired KAN deltas:'); print(pd.DataFrame(paired).to_string(index=False))

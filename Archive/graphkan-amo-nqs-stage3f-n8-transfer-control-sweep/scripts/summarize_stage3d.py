#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--results',required=True); a=p.parse_args()
root=Path(a.results); raw=root/'raw'; out=root/'aggregated'; out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(raw/'stage3c_metrics.csv')
metrics=['exact_model_abs_energy_error_to_ed','exact_model_density_mae_to_ed','exact_model_connected_correlation_mae_to_ed','wall_seconds']
identity=['model_label','model_kind','basis_family','n_parameters']
summary=m.groupby(identity,dropna=False).agg(
    runs=('run_id','count'), valid=('scientific_validity',lambda x:(x=='VALID').sum()),
    **{f'median_{k}':(k,'median') for k in metrics}
).reset_index()
summary['valid_rate']=summary['valid']/summary['runs']
summary.to_csv(out/'model_summary.csv',index=False)
# Factorial contrasts only where labels exist.
labels=set(m.model_label)
comparisons=[]
for base,treated,question in [
 ('sine_radial_19p','sine_endpoint_19p','endpoint effect under sine radial basis'),
 ('rbf_radial_19p','rbf_endpoint_19p','endpoint effect under RBF radial basis'),
 ('sine_radial_19p','rbf_radial_19p','radial-basis effect without endpoints'),
 ('sine_endpoint_19p','rbf_endpoint_19p','radial-basis effect with endpoints'),
]:
 if base in labels and treated in labels:
  a_df=m[m.model_label==base].set_index('seed'); b_df=m[m.model_label==treated].set_index('seed')
  common=a_df.index.intersection(b_df.index)
  for metric in metrics[:3]:
   d=b_df.loc[common,metric]-a_df.loc[common,metric]
   comparisons.append({'comparison':f'{treated} minus {base}','question':question,'metric':metric,'pairs':len(common),'median_delta':d.median(),'mean_delta':d.mean(),'treated_lower_fraction':float((d<0).mean())})
pd.DataFrame(comparisons).to_csv(out/'factorial_contrasts.csv',index=False)
site=pd.read_csv(raw/'stage3c_site_observables.csv')
site_summary=site.groupby(['model_label','basis_family','site'])['exact_model_abs_error'].median().reset_index()
site_summary.to_csv(out/'site_density_error_summary.csv',index=False)
print('\nModel summary:\n',summary.to_string(index=False))
print('\nFactorial contrasts:\n',pd.DataFrame(comparisons).to_string(index=False))
print('\nWrote:',out)

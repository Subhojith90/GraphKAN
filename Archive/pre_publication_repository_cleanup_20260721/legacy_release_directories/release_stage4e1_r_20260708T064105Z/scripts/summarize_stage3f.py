#!/usr/bin/env python
"""Stage 3F summaries with a joint energy-density-correlation transfer gate."""
from pathlib import Path
import argparse
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--results',required=True); a=p.parse_args()
root=Path(a.results); raw=root/'raw'; out=root/'aggregated'; out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(raw/'stage3f_metrics.csv')
metrics=['exact_model_abs_energy_error_to_ed','exact_model_density_mae_to_ed','exact_model_connected_correlation_mae_to_ed','wall_seconds']
identity=['model_label','model_kind','basis_family','n_parameters']
summary=m.groupby(identity,dropna=False).agg(runs=('run_id','count'),valid=('scientific_validity',lambda x:(x=='VALID').sum()),**{f'median_{k}':(k,'median') for k in metrics}).reset_index()
summary['valid_rate']=summary['valid']/summary['runs']
summary.to_csv(out/'model_summary.csv',index=False)
summary[['model_label','n_parameters']].to_csv(out/'parameter_audit.csv',index=False)
# Generic paired contrasts among labels that are present.
pairs=[]
for base,treated,question in [
 ('sine_radial_21p','sine_endpoint_21p','endpoint effect under sine basis at N=8'),
 ('rbf_radial_21p','rbf_endpoint_21p','endpoint effect under RBF basis at N=8'),
 ('sine_radial_19p','sine_endpoint_full_19p','endpoint effect under sine basis at N=6'),
 ('rbf_radial_19p','rbf_endpoint_full_19p','endpoint-component effect under RBF basis at N=6'),
]:
 a_df=m[m.model_label==base].set_index('seed'); b_df=m[m.model_label==treated].set_index('seed'); common=a_df.index.intersection(b_df.index)
 if len(common):
  for metric in metrics:
   d=b_df.loc[common,metric]-a_df.loc[common,metric]
   pairs.append({'comparison':f'{treated} minus {base}','question':question,'metric':metric,'pairs':len(common),'median_delta':d.median(),'mean_delta':d.mean(),'treated_lower_fraction':float((d<0).mean())})
pd.DataFrame(pairs).to_csv(out/'paired_contrasts.csv',index=False)
site=pd.read_csv(raw/'stage3f_site_observables.csv')
site.groupby(['model_label','basis_family','site'])['exact_model_abs_error'].median().reset_index().to_csv(out/'site_density_error_summary.csv',index=False)
# Joint N=8 gate; conservative and only calculated where the designated models exist.
gate={'status':'NOT_APPLICABLE','reason':'Required N=8 RBF comparison missing'}
rad=m[m.model_label=='rbf_radial_21p'].set_index('seed'); end=m[m.model_label=='rbf_endpoint_21p'].set_index('seed'); common=rad.index.intersection(end.index)
if len(common):
 dE=end.loc[common,'exact_model_abs_energy_error_to_ed']-rad.loc[common,'exact_model_abs_energy_error_to_ed']
 dD=end.loc[common,'exact_model_density_mae_to_ed']-rad.loc[common,'exact_model_density_mae_to_ed']
 dC=end.loc[common,'exact_model_connected_correlation_mae_to_ed']-rad.loc[common,'exact_model_connected_correlation_mae_to_ed']
 corr_base=rad.loc[common,'exact_model_connected_correlation_mae_to_ed']
 valid_end=(end.loc[common,'scientific_validity']=='VALID').all()
 corr_ok=bool((dC<=0.01).all() and (dC <= 0.25*corr_base).all())
 gate={'status':'PASS' if valid_end and (dE<=0).all() and (dD<=0).all() and corr_ok else 'FAIL',
       'pairs':int(len(common)),'endpoint_all_valid':bool(valid_end),'endpoint_energy_not_worse_all_seeds':bool((dE<=0).all()),
       'endpoint_density_not_worse_all_seeds':bool((dD<=0).all()),'correlation_penalty_within_abs_0_01_and_rel_25pct':corr_ok,
       'median_energy_delta':float(dE.median()),'median_density_delta':float(dD.median()),'median_correlation_delta':float(dC.median())}
pd.DataFrame([gate]).to_csv(out/'n8_joint_gate.csv',index=False)
print('\nModel summary:\n',summary.to_string(index=False)); print('\nJoint gate:\n',pd.DataFrame([gate]).to_string(index=False)); print('\nWrote:',out)

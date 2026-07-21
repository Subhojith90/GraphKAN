#!/usr/bin/env python
from pathlib import Path
import pandas as pd, numpy as np
root=Path('results_stage2_7'); raw=root/'raw'/'stage2_7_metrics.csv'; df=pd.read_csv(raw); out=root/'aggregated'; out.mkdir(parents=True,exist_ok=True)
print(df.groupby(['profile','model_label','scientific_validity']).size().reset_index(name='runs').to_string(index=False))
valid=df.query("scientific_validity=='VALID'").copy()
metrics=['abs_energy_error_to_ed','exact_model_abs_energy_error_to_ed','sample_density_mae_to_ed','exact_model_density_mae_to_ed','sample_connected_correlation_mae_to_ed','exact_model_connected_correlation_mae_to_ed','wall_seconds','n_parameters']
summary=valid.groupby(['profile','model_label'])[metrics].median().reset_index(); summary.to_csv(out/'stage2_7_summary.csv',index=False)
print('\nValid median metrics:'); print(summary.to_string(index=False))
# paired fairness only
fair=valid[valid.profile.str.contains('fairness')]
rows=[]
for label in ['kan_matched','mlp_matched','rbf_matched']:
 x=fair[fair.model_label.eq(label)].set_index(['instance_id','seed'])
 if label!='kan_matched':
  k=fair[fair.model_label.eq('kan_matched')].set_index(['instance_id','seed']); j=k.join(x,lsuffix='_kan',rsuffix='_base',how='inner')
  if len(j): rows.append({'comparison':'kan_minus_'+label,'pairs':len(j),'median_exact_energy_delta':float((j.exact_model_abs_energy_error_to_ed_kan-j.exact_model_abs_energy_error_to_ed_base).median()),'median_exact_density_delta':float((j.exact_model_density_mae_to_ed_kan-j.exact_model_density_mae_to_ed_base).median()),'median_exact_corr_delta':float((j.exact_model_connected_correlation_mae_to_ed_kan-j.exact_model_connected_correlation_mae_to_ed_base).median()),'median_wall_delta':float((j.wall_seconds_kan-j.wall_seconds_base).median())})
pd.DataFrame(rows).to_csv(out/'stage2_7_paired_fairness.csv',index=False); print('\nFairness paired deltas:'); print(pd.DataFrame(rows).to_string(index=False))
# hard regime pass map
hard=df[(df.delta==2)&(df.c6==4)].groupby(['model_label','n_samples','diag_shift','scientific_validity']).size().reset_index(name='runs'); hard.to_csv(out/'stage2_7_hard_regime_pass_map.csv',index=False); print('\nHard-regime pass map:'); print(hard.to_string(index=False))

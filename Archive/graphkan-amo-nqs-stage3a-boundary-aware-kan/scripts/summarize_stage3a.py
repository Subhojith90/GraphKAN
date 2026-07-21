#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--results', default='results_stage3a')
a=ap.parse_args()
root=Path(a.results)
m=pd.read_csv(root/'raw'/'stage3a_metrics.csv')
cols=['model_label','n_parameters','scientific_validity','exact_model_abs_energy_error_to_ed','exact_model_density_mae_to_ed','exact_model_connected_correlation_mae_to_ed','wall_seconds','health_flags']
print(m[cols].to_string(index=False))
print('\nModel medians:')
print(m.groupby('model_label')[['exact_model_abs_energy_error_to_ed','exact_model_density_mae_to_ed','exact_model_connected_correlation_mae_to_ed','wall_seconds']].median().to_string())
s=pd.read_csv(root/'raw'/'stage3a_site_observables.csv')
site=s.groupby(['model_label','site'])['exact_model_abs_error'].median().reset_index()
site.to_csv(root/'aggregated_stage3a_site_density_error.csv',index=False)
print('\nSite-resolved median density errors:')
print(site.to_string(index=False))

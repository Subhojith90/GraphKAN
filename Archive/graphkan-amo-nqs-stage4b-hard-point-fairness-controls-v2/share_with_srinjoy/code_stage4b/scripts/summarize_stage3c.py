#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd

ap=argparse.ArgumentParser(); ap.add_argument("--results", required=True); a=ap.parse_args()
root=Path(a.results); m=pd.read_csv(root/"raw"/"stage3c_metrics.csv")
metrics=["exact_model_abs_energy_error_to_ed","exact_model_density_mae_to_ed","exact_model_connected_correlation_mae_to_ed","wall_seconds"]
cols=["model_label","model_kind","basis_family","n_parameters","seed","scientific_validity",*metrics,"health_flags"]
print(m[cols].to_string(index=False))
print("\nModel medians:")
print(m.groupby(["model_label","model_kind","basis_family"])[metrics].median().to_string())
print("\nValidity:")
print(m.groupby(["model_label","scientific_validity"]).size().to_string())
s=pd.read_csv(root/"raw"/"stage3c_site_observables.csv")
site=s.groupby(["model_label","site"])["exact_model_abs_error"].median().reset_index()
(root/"aggregated").mkdir(exist_ok=True)
site.to_csv(root/"aggregated"/"site_density_error_summary.csv",index=False)
print("\nSite-resolved density errors:")
print(site.to_string(index=False))

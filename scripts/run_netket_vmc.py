#!/usr/bin/env python
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

from _common import load_config, build_geometries_from_config, parameter_grid, hash_dict
from graphkan_amo_nqs.hamiltonians.rydberg import RydbergParams, instance_id
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.models.flax_nqs import build_flax_model
from graphkan_amo_nqs.training.nqs_preflight import preflight_dict


def append_rows(path: Path, rows: list[dict]):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame=pd.DataFrame(rows)
    if path.exists(): frame.to_csv(path, mode='a', index=False, header=False)
    else: frame.to_csv(path,index=False)


def count_params(model, n_sites: int, seed: int):
    import jax, jax.numpy as jnp
    variables=model.init(jax.random.PRNGKey(seed), jnp.ones((1,n_sites)))
    leaves=jax.tree_util.tree_leaves(variables['params'])
    return int(sum(np.asarray(x).size for x in leaves))


def main():
    ap=argparse.ArgumentParser(description='Run registered NetKet/Flax VMC benchmark. No CPU fallback is used.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args=ap.parse_args(); cfg=load_config(args.config); root=Path(cfg.get('output_root','results'))
    preflight=preflight_dict(True); (root/'environment').mkdir(parents=True,exist_ok=True)
    (root/'environment'/'nqs_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    if not preflight['ok']:
        append_rows(root/'raw'/'failed_runs.csv',[{'profile':cfg.get('profile','netket_vmc'),'status':'NOT_RUN_DEPENDENCY','reason':'; '.join(preflight['errors']),'timestamp_utc':pd.Timestamp.utcnow().isoformat()}])
        raise SystemExit('Preflight failed. See results/environment/nqs_preflight.json. No VMC result has been generated.')
    if args.dry_run:
        print('Preflight passed. Dry run intentionally launched no training.'); return
    import jax, optax, netket as nk
    profile=cfg.get('profile','netket_vmc'); config_hash=hash_dict(cfg); metrics=[]; history=[]
    training=cfg.get('training',{}); n_samples=int(training.get('n_samples',1024)); n_iter=int(training.get('n_iter',300)); chunk=int(training.get('chunk_size',25))
    models=cfg.get('models',['product','smooth_jastrow','mlp_pair','kan_jastrow']); seeds=training.get('seeds',[0,1,2,3,4])
    for geometry in build_geometries_from_config(cfg):
      for params in parameter_grid(cfg):
       for model_name in models:
        for seed in seeds:
         started=time.perf_counter(); run_id=f'{profile}:{instance_id(geometry,params)}:{model_name}:seed{seed}'
         try:
          hi,H=build_netket_rydberg_operator(geometry.positions,params)
          r_scale=params.r_omega if np.isfinite(params.r_omega) else 1.0
          model=build_flax_model(model_name,geometry.positions,n_basis=int(training.get('n_basis',8)),hidden=int(training.get('hidden',16)),r_scale=r_scale)
          n_params=count_params(model,geometry.n_sites,int(seed))
          sampler=nk.sampler.MetropolisLocal(hi, n_chains_per_rank=int(training.get('n_chains',16)))
          vstate=nk.vqs.MCState(sampler,model,n_samples=n_samples,seed=int(seed))
          optimizer=optax.adam(float(training.get('learning_rate',0.01)))
          driver=nk.driver.VMC(H,optimizer,variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=float(training.get('diag_shift',0.01))))
          for step in range(0,n_iter,chunk):
            current=min(chunk,n_iter-step); driver.run(n_iter=current, out=None, show_progress=False)
            energy=driver.energy
            history.append({'run_id':run_id,'step':step+current,'energy_mean':float(np.real(energy.mean)),'energy_error':float(energy.error_of_mean),'profile':profile,'model':model_name,'seed':seed,'config_hash':config_hash})
          energy=driver.energy; elapsed=time.perf_counter()-started
          metrics.append({'run_id':run_id,'profile':profile,'status':'COMPLETED','model':model_name,'seed':seed,'n_sites':geometry.n_sites,'instance_id':instance_id(geometry,params),'omega':params.omega,'delta':params.delta,'c6':params.c6,'energy_mean':float(np.real(energy.mean)),'energy_error':float(energy.error_of_mean),'n_parameters':n_params,'n_samples':n_samples,'n_iter':n_iter,'wall_seconds':elapsed,'config_hash':config_hash,'backend':'netket_flax'})
          print(f'completed {run_id}: E={float(np.real(energy.mean)):.8f} ± {float(energy.error_of_mean):.3g}')
         except Exception as exc:
          metrics.append({'run_id':run_id,'profile':profile,'status':'FAILED','model':model_name,'seed':seed,'n_sites':geometry.n_sites,'instance_id':instance_id(geometry,params),'omega':params.omega,'delta':params.delta,'c6':params.c6,'error_type':type(exc).__name__,'error_message':str(exc),'traceback':traceback.format_exc(),'config_hash':config_hash,'backend':'netket_flax'})
          print(f'FAILED {run_id}: {type(exc).__name__}: {exc}')
    append_rows(root/'raw'/'netket_vmc_metrics.csv',metrics); append_rows(root/'raw'/'netket_vmc_history.csv',history)
    if any(r['status']=='FAILED' for r in metrics): raise SystemExit('One or more VMC runs failed; successful and failed rows were retained.')

if __name__=='__main__': main()

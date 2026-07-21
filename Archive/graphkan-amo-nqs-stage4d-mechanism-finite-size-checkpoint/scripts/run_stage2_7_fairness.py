#!/usr/bin/env python
"""Stage 2.7 hard-regime and fairness runner.

Adds exact all-state post-training observables for N=6,8, site-resolved error
exports, variant-level sample/SR diagnostics, and narrow parameter-matched
comparisons. The runner deliberately refuses to describe outcomes as Stage 3.
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from _common import load_config, build_geometries_from_config, parameter_grid, hash_dict
from graphkan_amo_nqs.hamiltonians.rydberg import RydbergParams, instance_id, build_dense_rydberg
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.models.flax_nqs import build_flax_model
from graphkan_amo_nqs.training.nqs_preflight import preflight_dict
from graphkan_amo_nqs.hamiltonians.basis import enumerate_binary_states
from graphkan_amo_nqs.observables.density import density_from_state
from graphkan_amo_nqs.observables.correlations import connected_correlations_from_state, correlation_mae

def append(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(path, mode='a' if path.exists() else 'w', header=not path.exists(), index=False)

def exact_reference(positions, params):
    H=np.asarray(build_dense_rydberg(positions,params),dtype=np.complex128)
    vals,vecs=np.linalg.eigh(H); return float(vals[0]),H,vecs[:,0]

def operator_check(positions, params, h):
    e,H,psi=exact_reference(positions,params); hn=np.asarray(h.to_dense(),dtype=np.complex128)
    return e,H,psi,float(np.max(np.abs(H-hn))),float(np.max(np.abs(np.linalg.eigvalsh(H)-np.linalg.eigvalsh(hn))))

def physical_sigma_from_binary(states, inverted):
    # state basis uses physical occupation n in {0,1}
    return 2.0*states-1.0 if inverted else 1.0-2.0*states

def exact_model_observables(vstate, states, H, inverted):
    import jax.numpy as jnp
    sigma=physical_sigma_from_binary(states,inverted)
    logs=np.asarray(vstate.model.apply({'params':vstate.parameters},jnp.asarray(sigma,dtype=jnp.float64)),dtype=np.complex128)
    logs=logs-np.max(logs.real)
    psi=np.exp(logs); psi=psi/np.linalg.norm(psi)
    density=density_from_state(psi,states)
    corr=connected_correlations_from_state(psi,states)
    energy=float(np.real(np.vdot(psi,H@psi)))
    return psi,energy,density,corr

def sample_observables(vstate,n,inverted):
    s=np.asarray(vstate.samples).reshape((-1,n)); occ=0.5*(s+1) if inverted else 0.5*(1-s)
    d=occ.mean(0); c=(occ.T@occ)/len(occ)-np.outer(d,d)
    unique=np.unique(s,axis=0).shape[0]
    return d,c,{'sample_count_observed':int(len(s)),'unique_sampled_states':int(unique),'unique_state_fraction':float(unique/max(1,len(s)))}

def classify(e, est_err, exact_e, sm, ex, cfg):
    flags=[]
    if not (np.isfinite(e) and np.isfinite(est_err)): flags.append('NONFINITE_ESTIMATOR')
    if est_err<=cfg.get('min_energy_error',1e-10): flags.append('NEAR_ZERO_ERROR')
    if abs(e-exact_e)>cfg.get('max_abs_energy_error_to_ed',0.5): flags.append('LARGE_ED_ERROR')
    if sm['density_mae']>cfg.get('max_density_mae_to_ed',0.08): flags.append('LARGE_DENSITY_ERROR')
    if sm['corr_mae']>cfg.get('max_connected_correlation_mae_to_ed',0.08): flags.append('LARGE_CORRELATION_ERROR')
    if sm['unique_state_fraction']<cfg.get('min_unique_state_fraction',0.005): flags.append('LOW_SAMPLE_DIVERSITY')
    return ('VALID' if not flags else 'FLAGGED'), flags

def nparams(model,n,seed):
    import jax, jax.numpy as jnp
    vs=model.init(jax.random.PRNGKey(seed),jnp.ones((1,n),dtype=jnp.float64))
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(vs['params'])))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args(); cfg=load_config(a.config)
 root=Path(cfg.get('output_root','results_stage2_7')); prof=cfg['profile']; raw=root/'raw'; env=root/'environment'; env.mkdir(parents=True,exist_ok=True)
 pf=preflight_dict(True); (env/'nqs_preflight.json').write_text(json.dumps(pf,indent=2)+'\n')
 if not pf['ok']: raise SystemExit('Preflight failed; no Stage 2.7 run generated.')
 try: (env/'pip_freeze_minimal.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze','--exclude-editable'],text=True))
 except Exception: pass
 (env/'run_manifest.json').write_text(json.dumps({'profile':prof,'config_hash':hash_dict(cfg),'stage':'2.7','purpose':'hard-regime diagnostic and matched-fairness gate'},indent=2)+'\n')
 if a.overwrite:
  for p in [raw/'stage2_7_metrics.csv',raw/'stage2_7_history.csv',raw/'stage2_7_site_observables.csv',raw/'stage2_7_corr_observables.csv']:
   if p.exists(): p.unlink()
 import netket as nk, optax
 inv=bool(cfg.get('netket',{}).get('inverted_ordering',False)); health=cfg.get('health_gate',{}); tr=cfg['training']; models=cfg['model_specs']; config_hash=hash_dict(cfg)
 rows=[]; hist=[]; site_rows=[]; corr_rows=[]
 for geo in build_geometries_from_config(cfg):
  for par in parameter_grid(cfg):
   hi,h=build_netket_rydberg_operator(geo.positions,par,inverted_ordering=inv)
   e0,H,psi0,md,sd=operator_check(geo.positions,par,h)
   if md>1e-10 or sd>1e-10: raise RuntimeError('operator check failed')
   states=enumerate_binary_states(geo.n_sites); edd=density_from_state(psi0,states); edc=connected_correlations_from_state(psi0,states)
   for label,spec in models.items():
    for nsamp in spec.get('n_samples',[tr['n_samples']]):
     for shift in spec.get('diag_shifts',[tr['diag_shift']]):
      for seed in tr['seeds']:
       rid=f"{prof}:{instance_id(geo,par)}:{label}:samples{nsamp}:shift{shift}:seed{seed}"; start=time.perf_counter()
       try:
        model=build_flax_model(spec['kind'],geo.positions,n_basis=int(spec.get('n_basis',tr['n_basis'])),hidden=int(spec.get('hidden',tr['hidden'])),r_scale=par.r_omega if np.isfinite(par.r_omega) else 1.0,omega=par.omega,inverted_ordering=inv)
        sampler=nk.sampler.MetropolisLocal(hi,n_chains_per_rank=int(tr['n_chains']))
        vs=nk.vqs.MCState(sampler,model,n_samples=int(nsamp),n_discard_per_chain=int(tr['n_discard_per_chain']),seed=int(seed))
        opt=optax.chain(optax.clip_by_global_norm(float(tr['grad_clip_norm'])),optax.adam(float(tr['learning_rate'])))
        dr=nk.driver.VMC(h,opt,variational_state=vs,preconditioner=nk.optimizer.SR(diag_shift=float(shift)))
        for step in range(0,int(tr['n_iter']),int(tr['chunk_size'])):
         dr.run(n_iter=min(int(tr['chunk_size']),int(tr['n_iter'])-step),out=None,show_progress=False); est=dr.energy
         hist.append({'run_id':rid,'step':step+min(int(tr['chunk_size']),int(tr['n_iter'])-step),'energy_mean':float(np.real(est.mean)),'energy_error':float(est.error_of_mean),'profile':prof,'model_label':label,'seed':seed,'n_samples':nsamp,'diag_shift':shift})
        est=dr.energy; smd,smc,sdia=sample_observables(vs,geo.n_sites,inv); psi,all_e,all_d,all_c=exact_model_observables(vs,states,H,inv)
        sm={'density_mae':float(np.mean(np.abs(smd-edd))),'corr_mae':correlation_mae(smc,edc),**sdia}
        ex={'density_mae':float(np.mean(np.abs(all_d-edd))),'corr_mae':correlation_mae(all_c,edc)}
        val,flags=classify(float(np.real(est.mean)),float(est.error_of_mean),e0,sm,ex,health)
        base={'run_id':rid,'profile':prof,'status':'COMPLETED','scientific_validity':val,'health_flags':';'.join(flags),'model_label':label,'model_kind':spec['kind'],'seed':seed,'n_sites':geo.n_sites,'instance_id':instance_id(geo,par),'omega':par.omega,'delta':par.delta,'c6':par.c6,'n_samples':nsamp,'diag_shift':shift,'n_basis':spec.get('n_basis',tr['n_basis']),'hidden':spec.get('hidden',tr['hidden']),'n_parameters':nparams(model,geo.n_sites,seed),'energy_mean':float(np.real(est.mean)),'energy_error':float(est.error_of_mean),'abs_energy_error_to_ed':abs(float(np.real(est.mean))-e0),'exact_model_energy':all_e,'exact_model_abs_energy_error_to_ed':abs(all_e-e0),'sample_density_mae_to_ed':sm['density_mae'],'sample_connected_correlation_mae_to_ed':sm['corr_mae'],'exact_model_density_mae_to_ed':ex['density_mae'],'exact_model_connected_correlation_mae_to_ed':ex['corr_mae'],'exact_ground_energy':e0,'operator_matrix_max_abs_diff':md,'operator_spectrum_max_abs_diff':sd,'wall_seconds':time.perf_counter()-start,'config_hash':config_hash,**sdia}
        rows.append(base)
        for i in range(geo.n_sites): site_rows.append({**{k:base[k] for k in ['run_id','profile','model_label','model_kind','seed','n_sites','instance_id','omega','delta','c6','n_samples','diag_shift','scientific_validity']},'site':i,'ed_density':edd[i],'sample_density':smd[i],'exact_model_density':all_d[i],'sample_abs_error':abs(smd[i]-edd[i]),'exact_model_abs_error':abs(all_d[i]-edd[i])})
        for i in range(geo.n_sites):
         for j in range(geo.n_sites): corr_rows.append({**{k:base[k] for k in ['run_id','profile','model_label','model_kind','seed','n_sites','instance_id','omega','delta','c6','n_samples','diag_shift','scientific_validity']},'i':i,'j':j,'ed_connected':edc[i,j],'sample_connected':smc[i,j],'exact_model_connected':all_c[i,j],'sample_abs_error':abs(smc[i,j]-edc[i,j]),'exact_model_abs_error':abs(all_c[i,j]-edc[i,j])})
        print(f"{val.lower()} {rid}: sample|E-E0|={base['abs_energy_error_to_ed']:.4g}, exact|E-E0|={base['exact_model_abs_energy_error_to_ed']:.4g}, flags={base['health_flags'] or 'none'}")
       except Exception as exc:
        rows.append({'run_id':rid,'profile':prof,'status':'FAILED','scientific_validity':'INVALID','health_flags':'EXECUTION_FAILURE','model_label':label,'model_kind':spec['kind'],'seed':seed,'n_sites':geo.n_sites,'instance_id':instance_id(geo,par),'omega':par.omega,'delta':par.delta,'c6':par.c6,'n_samples':nsamp,'diag_shift':shift,'error_type':type(exc).__name__,'error_message':str(exc),'traceback':traceback.format_exc(),'config_hash':config_hash})
        print('FAILED',rid,type(exc).__name__,exc)
 append(raw/'stage2_7_metrics.csv',rows); append(raw/'stage2_7_history.csv',hist); append(raw/'stage2_7_site_observables.csv',site_rows); append(raw/'stage2_7_corr_observables.csv',corr_rows)
 pd.DataFrame(rows).to_csv(raw/'stage2_7_metrics_snapshot.csv',index=False)
 print(f"Stage 2.7 complete: {sum(r.get('scientific_validity')=='VALID' for r in rows)}/{len(rows)} valid.")
if __name__=='__main__': main()

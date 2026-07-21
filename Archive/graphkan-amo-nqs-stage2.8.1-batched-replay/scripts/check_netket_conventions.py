#!/usr/bin/env python
"""Lock and test NetKet Spin local-state mapping against the physical Rydberg occupation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from _common import load_config, build_geometries_from_config, parameter_grid
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.hamiltonians.rydberg import build_dense_rydberg

def occ_from_sigma(states, inverted):
    return 0.5*(states+1.0) if inverted else 0.5*(1.0-states)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    cfg=load_config(args.config); net=cfg.get('netket',{}); inverted=bool(net.get('inverted_ordering',False))
    output=Path(cfg.get('output_root','results_stage2_6'))/'environment'/'netket_convention_check.json'; output.parent.mkdir(parents=True,exist_ok=True)
    checks=[]
    for geo in build_geometries_from_config(cfg):
      for params in parameter_grid(cfg):
        hi,h=build_netket_rydberg_operator(geo.positions,params,inverted_ordering=inverted)
        states=np.asarray(hi.all_states(),dtype=float)
        numbers=np.asarray(hi.states_to_numbers(states),dtype=int)
        occ=occ_from_sigma(states,inverted)
        # state numbers must span whole finite basis and physical occupation remains binary
        dense=np.asarray(build_dense_rydberg(geo.positions,params),dtype=np.complex128)
        nkdense=np.asarray(h.to_dense(),dtype=np.complex128)
        checks.append({"n_sites":geo.n_sites,"omega":params.omega,"delta":params.delta,"c6":params.c6,
          "inverted_ordering":inverted,"numbering_is_permutation":bool(np.array_equal(np.sort(numbers),np.arange(2**geo.n_sites))),
          "occupation_binary":bool(np.all(np.isin(occ,[0.,1.]))),"occupation_formula":"(1-sigma)/2" if not inverted else "(sigma+1)/2",
          "matrix_max_abs_diff":float(np.max(np.abs(dense-nkdense))),
          "operator_valid":bool(np.max(np.abs(dense-nkdense))<=1e-10)})
    ok=all(c['numbering_is_permutation'] and c['occupation_binary'] and c['operator_valid'] for c in checks)
    payload={"ok":ok,"convention":{"inverted_ordering":inverted,"physical_occupation":"(1-sigma)/2" if not inverted else "(sigma+1)/2"},"checks":checks}
    output.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2));
    if not ok: raise SystemExit('Convention check failed')
if __name__=='__main__': main()

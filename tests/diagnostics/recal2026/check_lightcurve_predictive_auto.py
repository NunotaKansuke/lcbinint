"""Exercise the default light-curve auto entry, including status propagation."""
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'tests/diagnostics/results/recal2026/predictive-route-20261015'

def main():
    for k in list(os.environ):
        if k.startswith('LCBININT_'):del os.environ[k]
    os.sched_setaffinity(0,{0})
    # Match the established 1400-phase calibration during direct parity checks.
    ordinary_mode='--ordinary' in sys.argv
    if not ordinary_mode:
        os.environ['LCBININT_DIAGNOSTIC_DISABLE_AUTO_CAUSTIC_BINS']='1'
    # The installed editable finder otherwise overrides sys.path and loads
    # the unrelated installed binary instead of this branch's build.
    sys.meta_path=[f for f in sys.meta_path if 'editable' not in getattr(type(f),'__module__','')]
    sys.path.insert(0,str(ROOT/'build-adaptive-production'))
    import lcbinint as lc
    assert ROOT/'build-adaptive-production' in Path(lc._native.__file__).resolve().parents
    import numpy as np
    assert lc.Options().inverse_ray_grid=='auto'
    assert lc.Options().nbin=='auto'
    assert lc.Options().max_source_bins==512
    rows=list(csv.DictReader((ROOT/'figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv').open(),delimiter='\t'))
    byid={r['geom_id']:r for r in rows}
    refs={r['geom_id']:float(r['reference'])for r in rows}
    for r in csv.DictReader((ROOT/'figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv').open(),delimiter='\t'):
        refs[r['geom_id']]=float(r['adjudicated_reference'])
    saved=json.loads((OUT/'measurements.json').read_text())
    ids=[]
    for profile in ['uniform','linear']:
        ids+=sorted({r['id']for r in saved if r['split']=='test'and r['profile']==profile},key=int)[::2][:48]
    records=[]
    def evaluate(r,target,**overrides):
        opts=dict(coordinates='center_of_mass',finite_source_tol=0.,finite_source_reltol=target,
            point_source_threshold=0.,hexadecapole_threshold=0.,adaptive_hex_threshold=0.)
        opts.update(overrides)
        curve=lc.LightCurve(lens='binary',options=lc.Options(**opts))
        info=curve.info(np.array([0.,0.]),t0=-float(r['time']),tE=1.,u0=float(r['y']),alpha=0.,
            s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),limb_darkening_c=float(r['c']))
        a,b=info.finite_source_magnifications
        assert a==b or (math.isnan(a) and math.isnan(b)), ('cached repeat differs',r['geom_id'],target)
        assert abs(info.source_x[0]-float(r['time']))<1e-12
        return info
    if ordinary_mode:
        checks=[]
        for key in ids[::12]:
            info=evaluate(byid[key],1e-3)
            checks.append(dict(id=key,method=info.finite_source_method_names[0],status=info.statuses[0]))
        (OUT/'lightcurve-default-caustic-validation.json').write_text(json.dumps(checks,indent=2)+'\n')
        return
    for key in ids:
        r=byid[key]
        for target in [1e-3,1e-4]:
            kw=dict(s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),
                limb_darkening=lc.LimbDarkening(float(r['c']),0),relative_tolerance=target,
                absolute_tolerance=sys.float_info.min)
            direct=lc.binary_adaptive_auto_flux(float(r['time']),float(r['y']),**kw)
            info=evaluate(r,target)
            assert info.finite_source_method_names[0]=='inverse_ray_'+direct['selected_method'],(key,info.finite_source_method_names)
            value=info.finite_source_magnifications[0]
            # The diagnostic Polar wrapper reflects y before solving, whereas
            # LightCurve uses its trajectory frame. Compare at requested
            # numerical accuracy and also check the saved reference below.
            assert abs(value/direct['magnification']-1)<=target,(key,target,value,direct['magnification'])
            # The LC layer also enforces the final dimensional budget and
            # grazing-caustic floor; it may reject a direct-path acceptance.
            if info.finite_source_converged[0]:
                assert direct['estimated_target_met']
                assert info.finite_source_error_estimates[0]<=target*max(abs(value),1.)
                assert abs(value/refs[key]-1)<=target,(key,target,'reference mismatch')
            records.append(dict(id=key,target=target,method=direct['selected_method'],
                converged=info.finite_source_converged[0],status=info.statuses[0],
                reference_error=abs(value/refs[key]-1),relative_direct_difference=abs(value/direct['magnification']-1)))
    # Default shortcut routing, and explicitly fixed grids, still execute.
    far=lc.LightCurve(lens='binary').info(np.array([50.]),t0=0.,tE=1.,u0=.1,alpha=0.,s=1.2,q=.1,rho=.001)
    assert far.finite_source_method_names[0]in ['point_source','hexadecapole'],far.finite_source_method_names
    fixed=[]
    for grid in ['cartesian','polar']:
        info=evaluate(byid[ids[0]],1e-3,nbin=16,inverse_ray_grid=grid)
        assert info.finite_source_method_names[0]=='inverse_ray_'+grid
        fixed.append(grid)
    implicit=evaluate(byid[ids[0]],0.)
    assert implicit.finite_source_converged[0]
    absolute=evaluate(byid['3288'],0.,finite_source_tol=1e-3)
    assert absolute.finite_source_method_names[0]=='inverse_ray_polar'
    if absolute.finite_source_converged[0]:
        assert absolute.finite_source_error_estimates[0]<=1e-3
    # An explicitly restricted Cartesian cap is honored, not silently raised
    # to the new default. This known case needs the 512 refinement level.
    capped=evaluate(byid['3288'],1e-4,max_source_bins=400)
    assert not capped.finite_source_converged[0]
    assert capped.statuses[0]=='numerical_error'
    # The coarse-caustic switch is cached at first use: verify default settings
    # in a fresh process instead of merely deleting the environment variable.
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--ordinary'],check=True,
        env={k:v for k,v in os.environ.items()if not k.startswith('LCBININT_')})
    ordinary=json.loads((OUT/'lightcurve-default-caustic-validation.json').read_text())
    result=dict(conditions=len(records),records=records,default_caustic_checks=ordinary,
        implicit_tolerance_check=True,absolute_tolerance_check=True,explicit_cap_check=True,
        default_far_method=far.finite_source_method_names[0],fixed_grid_checks=fixed,
        native_sha256=hashlib.sha256(next((ROOT/'build-adaptive-production').glob('_lcbinint*.so')).read_bytes()).hexdigest())
    (OUT/'lightcurve-auto-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items()if k!='records'})

if __name__=='__main__':main()

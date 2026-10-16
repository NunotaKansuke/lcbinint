"""Validate the current default close-root gradient around the historical -0.7 d case."""
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
np,lc,VB=load()
p=dict(t0=0.,tE=10.,u0=-1e-5,alpha=.5,s=.6,q=1e-4,rho=.05)
times=np.array([-.704,-.700,-.696])
analytic=[]
for tol in (1e-3,1e-5,1e-6):
    curve=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(.5))
    r=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol)
    order=list(r['param_order'])
    analytic.append({'tol':tol,'gradient':{key:[float(r['grad_mu'][i,order.index(key)]) for i in range(len(times))] for key in ('u0','alpha','s')},'value_converged':[bool(x) for x in r['value_converged']],'support_certified':[bool(x) for x in r['support_certified']],'gradient_valid':[bool(all(row)) for row in r['gradient_valid']]})

def vbm_value(pp):
    v=VB.VBMicrolensing();v.Tol=v.RelTol=1e-10;v.a1=.5;v.a2=0.
    args=[np.log(pp['s']),np.log(pp['q']),pp['u0'],pp['alpha'],np.log(pp['rho']),np.log(pp['tE']),pp['t0']]
    return np.asarray(v.BinaryLightCurve(args,times.tolist())[0],dtype=float)
fd=[]
for key,steps in {'alpha':[1e-4,3e-5,1e-5]}.items():
    for h in steps:
        plus,minus=dict(p),dict(p);plus[key]+=h;minus[key]-=h
        values=(vbm_value(plus)-vbm_value(minus))/(2*h)
        fd.append({'parameter':key,'step':h,'fd':[float(x) for x in values]})
module=Path(__import__('_lcbinint_holonomic_fused').__file__).resolve()
native=Path(lc._native.__file__).resolve()
result={'case':'historical_close_root_case','params':p,'limb_darkening_linear':.5,'times':times.tolist(),'analytic_sweep':analytic,'vbm_central_differences':fd,'vbm_settings':{'Tol':1e-10,'RelTol':1e-10,'a1':.5,'a2':0.0},'gradient_module':str(module),'gradient_module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'native_module':str(native),'native_module_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'existing_vbm_reference_csv':str(ROOT/'evidence/holonomic/gradient_geometry_20261015/comparison.csv')}
(OUT/'alpha_vbm_fd.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

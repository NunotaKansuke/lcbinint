from pathlib import Path
from importlib.util import module_from_spec,spec_from_file_location
import os,sys,time,json,csv
import numpy as np
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
variant=sys.argv[1]
if variant=='screen_off':os.environ['LCBININT_GEOMETRY_SCREEN']='0'
paths={'original':'build-algebraic-boundary/fused-gradient/python','full':'build-algebraic-boundary/gradient-probe/geometry-before-screen-python','screen':'build-algebraic-boundary/gradient-probe/geometry-python','screen_off':'build-algebraic-boundary/gradient-probe/geometry-python'}
sys.path.insert(0,str(root/paths[variant]))
for f in tuple(sys.meta_path):
 if type(f).__module__=='_lcbinint_editable':sys.meta_path.remove(f)
spec=spec_from_file_location('lcbinint',root/'python/lcbinint/__init__.py',submodule_search_locations=[str(root/'python/lcbinint'),str(root/'build-algebraic-boundary/python-api/lcbinint')]);l=module_from_spec(spec);sys.modules['lcbinint']=l;spec.loader.exec_module(l)
p=dict(t0=0.,tE=10.,u0=-1e-5,alpha=.5,s=.6,q=1e-4,rho=.05)
curve=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=1e-3,finite_source_reltol=1e-3,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(.5))
scan=np.unique(np.r_[np.linspace(-.706,-.69,65),np.linspace(.775,.785,21),[-.72,-.68,.1]])
r=curve.holonomic_gradient(scan,p,mu_atol=1e-3,mu_rtol=1e-3)
np.savez(out/f'scan_{variant}.npz',times=scan,**{k:np.asarray(r[k]) for k in ['mu','grad_mu','gradient_valid','value_converged']})
report=[]
for case,q,rho in [('historical',1e-4,.05),('current',1e-3,5e-4),('ordinary',.01,.001)]:
 params={**p,'q':q,'rho':rho};times=np.linspace(-.8,.8,401)
 if case=='ordinary':params.update(s=1.2,u0=.2);times=np.linspace(-10,10,401)
 curve.holonomic_gradient(times,params,mu_atol=1e-3,mu_rtol=1e-3)
 elapsed=[]
 for rep in range(5):
  start=time.perf_counter();r=curve.holonomic_gradient(times,params,mu_atol=1e-3,mu_rtol=1e-3);elapsed.append((time.perf_counter()-start)*1000)
 np.savez(out/f'curve_{case}_{variant}.npz',times=times,mu=r['mu'],grad_mu=r['grad_mu'],valid=r['gradient_valid'],value_converged=r['value_converged'])
 report.append(dict(case=case,ms=elapsed,median_ms=float(np.median(elapsed)),finite=int(np.isfinite(r['grad_mu']).sum()),value_converged=int(np.sum(r['value_converged']))))
 print(variant,report[-1],flush=True)
(out/f'timing_{variant}.json').write_text(json.dumps(report,indent=2)+'\n')

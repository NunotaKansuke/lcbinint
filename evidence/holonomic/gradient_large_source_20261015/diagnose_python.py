from pathlib import Path
from importlib.util import module_from_spec,spec_from_file_location
import os,sys,time,json,csv
import numpy as np
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
variant='screen'
if variant=='screen_off':os.environ['LCBININT_GEOMETRY_SCREEN']='0'
paths={'original':'build-algebraic-boundary/fused-gradient/python','full':'build-algebraic-boundary/gradient-probe/geometry-before-screen-python','screen':'build-algebraic-boundary/fused-gradient/python','screen_off':'build-algebraic-boundary/gradient-probe/geometry-python'}
sys.path.insert(0,str(root/paths[variant]))
for f in tuple(sys.meta_path):
 if type(f).__module__=='_lcbinint_editable':sys.meta_path.remove(f)
spec=spec_from_file_location('lcbinint',root/'python/lcbinint/__init__.py',submodule_search_locations=[str(root/'python/lcbinint'),str(root/'build-algebraic-boundary/python-api/lcbinint')]);l=module_from_spec(spec);sys.modules['lcbinint']=l;spec.loader.exec_module(l)
import VBMicrolensing
p=dict(t0=0.,tE=10.,u0=-1e-5,alpha=.5,s=.6,q=1e-3,rho=.5)
times=np.array([-.484,-.48,-.476]);rows=[]
for tol in [1e-3,1e-4,1e-5,1e-6]:
 c=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(.5))
 r=c.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol)
 np.savez(out/f'tol_{tol}.npz',times=times,mu=r['mu'],grad_mu=r['grad_mu'],value_converged=r['value_converged'])
 print('tol',tol,'s',r['grad_mu'][:,6],flush=True)
c=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=1e-9,finite_source_reltol=1e-9,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(.5))
for name,h0 in [('s',1e-2),('u0',1e-2),('q',1e-5),('rho',1e-3),('t0',1e-2)]:
 for factor in [1.,.3,.1]:
  h=h0*factor;plus={**p,name:p[name]+h};minus={**p,name:p[name]-h};ip=c.info(times,plus);im=c.info(times,minus)
  for i,t in enumerate(times):rows.append(dict(t=t,param=name,h=h,fd=(ip['mu'][i]-im['mu'][i])/(2*h),plus_converged=bool(ip['value_converged'][i]),minus_converged=bool(im['value_converged'][i]),plus_error=ip.get('error_estimate',[None]*3)[i],minus_error=im.get('error_estimate',[None]*3)[i]))
 print('native fd',name,flush=True)
with (out/'native_reference.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)

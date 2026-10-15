from pathlib import Path
from importlib.util import module_from_spec,spec_from_file_location
import os,sys,time,json,csv
import numpy as np
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
variant=sys.argv[1] if len(sys.argv)>1 else 'previous'
if variant=='screen_off':os.environ['LCBININT_GEOMETRY_SCREEN']='0'
paths={'previous':'build-algebraic-boundary/gradient-probe/geometry-python','original':'build-algebraic-boundary/fused-gradient/python','full':'build-algebraic-boundary/gradient-probe/geometry-before-screen-python','screen':'build-algebraic-boundary/fused-gradient/python','screen_off':'build-algebraic-boundary/gradient-probe/geometry-python'}
sys.path.insert(0,str(root/paths[variant]))
for f in tuple(sys.meta_path):
 if type(f).__module__=='_lcbinint_editable':sys.meta_path.remove(f)
spec=spec_from_file_location('lcbinint',root/'python/lcbinint/__init__.py',submodule_search_locations=[str(root/'python/lcbinint'),str(root/'build-algebraic-boundary/python-api/lcbinint')]);l=module_from_spec(spec);sys.modules['lcbinint']=l;spec.loader.exec_module(l)
import VBMicrolensing
p=dict(t0=0.,tE=10.,u0=-1e-5,alpha=.5,s=.6,q=1e-3,rho=.5)
times=np.linspace(-.8,.8,401)
c=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=1e-3,finite_source_reltol=1e-3,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(.5))
start=time.perf_counter();r=c.holonomic_gradient(times,p,mu_atol=1e-3,mu_rtol=1e-3);print('gradient seconds',time.perf_counter()-start,flush=True)
np.savez(out/('baseline.npz' if variant=='previous' else 'reproduced_fixed.npz'),times=times,**{k:np.asarray(r[k]) for k in ['mu','grad_mu','gradient_valid','value_converged']})
g=c.finite_source_geometry(times,p)
with (out/'inputs.tsv').open('w') as f:
 for i,t in enumerate(times):f.write('\t'.join(map(str,[t,g.source_x[i],g.source_y[i],p['rho'],g.mass_ratio[i],p['s'],.5]))+'\n')
times=np.array([-.52,-.5,-.484,-.48,-.476,-.46,-.44])
indices=np.rint((times+.8)/.004).astype(int)
def vbm(params):
 v=VBMicrolensing.VBMicrolensing();v.Tol=1e-9;v.RelTol=1e-9;v.a1=.5;v.a2=0.
 a=[np.log(params['s']),np.log(params['q']),params['u0'],params['alpha'],np.log(params['rho']),np.log(params['tE']),params['t0']]
 return np.asarray(v.BinaryLightCurve(a,times.tolist())[0])
refs=[];names=['t0','u0','tE','alpha','rho','q','s'];steps=[1e-3,1e-4,1e-3,1e-4,1e-4,1e-6,1e-4]
for j,(name,h) in enumerate(zip(names,steps)):
 plus={**p,name:p[name]+h};minus={**p,name:p[name]-h};fd=(vbm(plus)-vbm(minus))/(2*h);refs.append(fd)
 err=abs(r['grad_mu'][indices,j]-fd);idx=np.nanargmax(err);print(name,'worst',times[idx],r['grad_mu'][indices[idx],j],fd[idx],err[idx],flush=True)
np.savez(out/'reference_scan.npz',times=times,fd=np.array(refs).T)
(out/'config.json').write_text(json.dumps(dict(params=p,tol=1e-3,reltol=1e-3,vbm_tol=1e-9,steps=dict(zip(names,steps)),names=names),indent=2)+'\n')

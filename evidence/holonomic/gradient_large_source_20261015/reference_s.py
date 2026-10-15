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
def vbm_value(params):
 v=VBMicrolensing.VBMicrolensing();v.Tol=v.RelTol=1e-10;v.a1=.5;v.a2=0.
 a=[np.log(params['s']),np.log(params['q']),params['u0'],params['alpha'],np.log(params['rho']),np.log(params['tE']),params['t0']]
 return np.asarray(v.BinaryLightCurve(a,times.tolist())[0])
for h in [.02,.01,.003,.001]:
 vp=vbm_value({**p,'s':p['s']+h});vm=vbm_value({**p,'s':p['s']-h})
 for i,t in enumerate(times):rows.append(dict(t=t,h=h,fd=(vp[i]-vm[i])/(2*h),vp=vp[i],vm=vm[i]))
 print('VBM s',h,(vp-vm)/(2*h),flush=True)
with (out/'vbm_s_reference.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)

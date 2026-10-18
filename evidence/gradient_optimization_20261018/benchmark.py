from pathlib import Path
import argparse,os,sys,json,hashlib,time,statistics
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--build',default='build-route-unified');parser.add_argument('--output',required=True);a=parser.parse_args()
sys.meta_path[:]=[f for f in sys.meta_path if 'editable' not in getattr(f,'__module__','')];sys.path.insert(0,str(ROOT/a.build))
import numpy as np,lcbinint as lc
assert Path(lc.__file__).is_relative_to(ROOT/a.build)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
nb=dict(t0=0.,tE=10.,u0=-.01,alpha=180.,s=1.5,q=.01,rho=.001)
cases=[('notebook',nb,np.linspace(-.8,.8,401),.5,1e-4),('notebook_1e-3',nb,np.linspace(-.8,.8,401),.5,1e-3),('notebook_uniform',nb,np.linspace(-.8,.8,401),0.,1e-4),('crossing_ld',dict(t0=0.,u0=.03,tE=1.,alpha=-.08,rho=.003,q=.01,s=1.2),np.linspace(-.2,.8,101),.5,1e-3),('point',dict(t0=0.,u0=.3,tE=1.,alpha=0.,rho=.003,q=.01,s=1.2),np.linspace(-1,1,201),0.,1e-3)]
output=dict(native=lc._native.__file__,sha256=hashlib.sha256(Path(lc._native.__file__).read_bytes()).hexdigest(),cases=[])
for name,p,t,limb,tol in cases:
 c=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(limb))
 # Warm quadrature code on another lens; target geometry is rebuilt in each call.
 c(np.linspace(-.05,.05,9),dict(p,s=1.3,u0=.02),with_gradient=True)
 samples={False:[],True:[]}
 results={}
 for rep in range(7):
  for grad in ((False,True) if rep%2==0 else (True,False)):
   start=time.perf_counter();r=c(t,p,with_gradient=grad,with_diagnostics=True);duration=(time.perf_counter()-start)*1000
   sec=np.asarray(r.diagnostics['seconds'])*1000;m=np.asarray(r.diagnostics['integrator_used'],bool)
   samples[grad].append(dict(ms=duration,integral_ms=float(sec[m].sum()),smooth_ms=float(sec[~m].sum()),nodes=int(np.asarray(r.diagnostics['node_evaluations'])[m].sum())))
   results[grad]=r
 row=dict(case=name,parameters=p,n=len(t),limb=limb,tolerance=tol,measurements={})
 for grad in (False,True):row['measurements'][str(grad)]={k:statistics.median(s[k] for s in samples[grad]) for k in samples[grad][0]}
 row['ratio']=row['measurements']['True']['ms']/row['measurements']['False']['ms']
 row['max_value_fused_relative_difference']=float(np.max(abs(results[False].magnifications-results[True].magnifications)/np.maximum(1,abs(results[False].magnifications))))
 output['cases'].append(row);print(name,row['measurements'],flush=True)
(ROOT/a.output).write_text(json.dumps(output,indent=2)+'\n')

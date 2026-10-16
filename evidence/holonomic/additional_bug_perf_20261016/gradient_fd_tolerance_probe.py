"""Tolerance and finite-difference step study for one near-resonant candidate."""
import hashlib,json,os,sys,subprocess,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
import numpy as np
np,lc,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
case=next(c for c in json.loads((OUT/'cases.json').read_text()) if c['id']=='stratified_00')
p=case['params'];limb=case['limb'];times=np.asarray([-.025,0.,.025])
def make(tol):
 return lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(limb))
steps={'q':[1e-8,3e-9,1e-9,3e-10,1e-10],'s':[1e-5,3e-6,1e-6,3e-7,1e-7,3e-8,1e-8]}
records=[]
for tol in (1e-3,1e-5,1e-7):
 curve=make(tol);g=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol)
 order=list(g['param_order']);ga=np.asarray(g['grad_mu'],dtype=float);analytic={key:ga[:,order.index(key)].tolist() for key in steps}
 group={'tol':tol,'times_days':times.tolist(),'analytic_q_s':analytic,'analytic_valid':np.asarray(g['gradient_valid'],dtype=bool)[:,[order.index('q'),order.index('s')]].tolist(),'finite_differences':[]}
 for key,hs in steps.items():
  for h in hs:
   pp=dict(p);pm=dict(p);pp[key]+=h;pm[key]-=h
   ip=make(tol).info(times,pp);im=make(tol).info(times,pm)
   vp=np.asarray(ip['mu'],dtype=float);vm=np.asarray(im['mu'],dtype=float);fd=(vp-vm)/(2*h)
   errp=np.asarray(ip['error_estimate'],dtype=float);errm=np.asarray(im['error_estimate'],dtype=float)
   err=(errp+errm)/(2*h)
   delta=fd-np.asarray(analytic[key])
   group['finite_differences'].append({'parameter':key,'step':h,'fd':fd.tolist(),'analytic':analytic[key],
    'fd_minus_analytic':delta.tolist(),'max_abs_difference':float(np.max(np.abs(delta))),
    'estimated_fd_error':err.tolist(),'plus_value_converged':np.asarray(ip['value_converged'],dtype=bool).tolist(),
    'minus_value_converged':np.asarray(im['value_converged'],dtype=bool).tolist(),
    'plus_fully_converged':np.asarray(ip['fully_converged'],dtype=bool).tolist(),
    'minus_fully_converged':np.asarray(im['fully_converged'],dtype=bool).tolist()})
  print('tol',tol,key,'done',flush=True)
 records.append(group)
source=Path(lc._native.__file__).resolve()
out={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'native_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'case_id':case['id'],'params':p,'limb':limb,'times_days':times.tolist(),'steps':steps,'records':records}
(OUT/'gradient_fd_tolerance_probe.json').write_text(json.dumps(out,indent=2)+'\n')

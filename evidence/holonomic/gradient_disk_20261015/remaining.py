from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,_=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for i,j in [(58,16),(81,12),(109,24)]:
 c=json.loads((ROOT/f'evidence/holonomic/stress_20261015/case_{i:03}.json').read_text());p=c['params'];t=np.linspace(-c['span'],c['span'],33)[j]*p['tE'];curve=l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(c['limb']))
 for tol in [1e-3,1e-4,1e-5]:
  g=curve.holonomic_gradient([t],p,mu_atol=tol,mu_rtol=tol);print(i,float(t),tol,np.asarray(g['grad_mu']).tolist(),g['value_converged'],flush=True)

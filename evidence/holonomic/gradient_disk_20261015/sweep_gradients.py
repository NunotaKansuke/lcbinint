from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,_=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
for c in json.loads((ROOT/'evidence/holonomic/stress_20261015/cases.json').read_text()):
 p=c['params'];times=np.linspace(-c['span'],c['span'],33)*p['tE'];curve=l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=1e-3,finite_source_reltol=1e-3),limb_darkening=l.LimbDarkening.linear(c['limb']))
 start=time.perf_counter();g=curve.holonomic_gradient(times,p,mu_atol=1e-3,mu_rtol=1e-3);a=np.asarray(g['grad_mu']);r=dict(id=c['id'],finite=int(np.isfinite(a).sum()),nan_indices=np.where(~np.isfinite(a).all(axis=1))[0].tolist(),ms=(time.perf_counter()-start)*1000);rows.append(r);np.savez_compressed(OUT/(c['id']+'_gradient.npz'),times=times,gradient=a,mu=g['mu'],valid=g['gradient_valid'],converged=g['value_converged']);print(r,flush=True)
 (OUT/'sweep_gradients.json').write_text(json.dumps(rows,indent=2)+'\n')

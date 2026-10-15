from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,_=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
for i in [2,8,33,64]:
 c=json.loads((OUT/f'case_{i:03}.json').read_text());p=c['params'];times=np.load(OUT/(c['id']+'.npz'))['times'][[8,16,24]]
 for tol in [1e-3,1e-4,1e-5,1e-7]:
  curve=l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=tol,finite_source_reltol=tol),limb_darkening=l.LimbDarkening.linear(c['limb']));g=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol)
  r=dict(case=c['id'],tol=tol,mu=np.asarray(g['mu']).tolist(),gradient=np.asarray(g['grad_mu']).tolist());rows.append(r);print(c['id'],tol,'rho',np.asarray(g['grad_mu'])[:,4],flush=True)
 (OUT/'gradient_ladder.json').write_text(json.dumps(rows,indent=2)+'\n')

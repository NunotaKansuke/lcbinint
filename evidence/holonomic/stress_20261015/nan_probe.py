from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,_=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for i in [4,10,49]:
 c=json.loads((OUT/f'case_{i:03}.json').read_text());p=c['params'];curve=l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=1e-3,finite_source_reltol=1e-3),limb_darkening=l.LimbDarkening.linear(c['limb']))
 for t in [0.,1e-12,1e-9,1e-6]:
  g=curve.holonomic_gradient([t],p,mu_atol=1e-3,mu_rtol=1e-3);print(i,t,{k:np.asarray(g[k]).tolist() for k in ['mu','grad_mu','stop','seed_support_proven']},flush=True)

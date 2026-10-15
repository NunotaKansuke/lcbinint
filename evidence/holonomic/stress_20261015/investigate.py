from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
rows=[]
for name in ['case_033','case_013']:
 c=json.loads((OUT/(name+'.json')).read_text());p=c['params'];old=np.load(OUT/(name+'.npz'));j=c['worst'];t=float(old['times'][j]);v=VB.VBMicrolensing()
 for limb in [0.,c['limb']]:
  for tol in [1e-3,1e-5,1e-7]:
   curve=l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=tol,finite_source_reltol=tol),limb_darkening=l.LimbDarkening.linear(limb));info=curve.info([t],p)
   v.Tol=v.RelTol=tol;v.a1=limb;v.a2=0;vp=[np.log(p['s']),np.log(p['q']),p['u0'],p['alpha'],np.log(p['rho']),np.log(p['tE']),p['t0']]
   vb=float(v.BinaryLightCurve(vp,[t])[0][0]);r=dict(case=name,time=t,limb=limb,tol=tol,lc=float(info['mu'][0]),vbm=vb,converged=bool(info['value_converged'][0]),nodes=int(info['node_evaluations'][0]));rows.append(r);print(r,flush=True)
 (OUT/'investigate.json').write_text(json.dumps(rows,indent=2)+'\n')

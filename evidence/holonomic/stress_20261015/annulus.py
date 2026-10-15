from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
for i in [13,33,64,89,117,118]:
 c=json.loads((OUT/f'case_{i:03}.json').read_text());p=c['params'];old=np.load(OUT/(c['id']+'.npz'));j=c['worst'];t=float(old['times'][j]);v=VB.VBMicrolensing();v.Tol=v.RelTol=1e-3;v.a1=v.a2=0.;args=[np.log(p['s']),np.log(p['q']),p['u0'],p['alpha'],np.log(p['rho']),np.log(p['tE']),p['t0']];outer=float(v.BinaryLightCurve(args,[t])[0][0])
 for n in [32,64,128]:
  x,w=np.polynomial.legendre.leggauss(n);z=(x+1)/2;mu=[]
  for zz in z:
   a=args.copy();a[4]=np.log(p['rho']*np.sqrt(1-zz*zz));mu.append(float(v.BinaryLightCurve(a,[t])[0][0]))
  ld=c['limb'];value=((1-ld)*outer+ld*np.dot(w/2*(1-z*z),mu))/(1-ld/3);r=dict(case=c['id'],n=n,time=t,lc=float(old['mu'][j]),vbm=float(old['vbm'][j]),annulus=float(value));rows.append(r);print(r,flush=True);(OUT/'annulus.json').write_text(json.dumps(rows,indent=2)+'\n')

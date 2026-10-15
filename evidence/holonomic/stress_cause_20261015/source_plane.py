from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
np,l,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});c=json.loads((ROOT/'evidence/holonomic/stress_20261015/case_033.json').read_text());p=c['params'];tau=-.15/p['tE'];x=-tau*np.cos(p['alpha']);y=-tau*np.sin(p['alpha']);v=VB.VBMicrolensing();rows=[]
print('center point',v.BinaryMag0(p['s'],p['q'],x,y),flush=True)
for ld in [0.,.5,.9]:
 for n in [8,16]:
  # z=sqrt(1-r^2), so brightness is polynomial and disk measure r dr=-z dz.
  a,w=np.polynomial.legendre.leggauss(n);z=(a+1)/2;w=w/2;theta=np.arange(4*n)*2*np.pi/(4*n)
  def magnification(rho):
   result=0.
   for zi,wi in zip(z,w):
    rr=rho*np.sqrt(1-zi*zi);values=[v.BinaryMag0(p['s'],p['q'],float(x+rr*np.cos(t)),float(y+rr*np.sin(t))) for t in theta]
    result+=2*zi*wi*(1-ld+ld*zi)*np.mean(values)/(1-ld/3)
   return result
  for frac in [0.05,.02,.01]:
   h=p['rho']*frac;a=magnification(p['rho']+h);b=magnification(p['rho']-h);r=dict(ld=ld,n=n,fraction=frac,plus=a,minus=b,derivative=(a-b)/(2*h));rows.append(r);print(r,flush=True)
(OUT/'source_plane.json').write_text(json.dumps(rows,indent=2)+'\n')

from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
import numpy as np
from scipy.integrate import quad
rows=[]
# Source-centred rings: integrate analytic point-lens magnification.
# At u=rho use lens-centred polar coordinates instead: boundary R=2*rho*cos(theta).
for i in [33,64,89,117]:
 c=json.loads((OUT/f'case_{i:03}.json').read_text());d=np.load(OUT/(c['id']+'.npz'));t=float(d['times'][c['worst']]);p=c['params'];rho=p['rho'];u=np.hypot(t/p['tE'],p['u0']);ld=c['limb']
 def angle(theta):
  projection=u*np.cos(theta);disc=rho*rho-u*u*np.sin(theta)**2
  if disc<=0:return 0.
  a=max(0.,projection-np.sqrt(disc));b=projection+np.sqrt(disc)
  if b<=a:return 0.
  def radial(r):
   rsrc2=r*r+u*u-2*r*projection;brightness=1-ld+ld*np.sqrt(max(0.,1-rsrc2/rho**2))
   return brightness*(r*r+2)/np.sqrt(r*r+4)
  return quad(radial,a,b,epsabs=1e-10,epsrel=1e-9,limit=200)[0]
 limit=np.pi if u<=rho else np.arcsin(rho/u)
 mu=2*quad(angle,0,limit,epsabs=1e-10,epsrel=1e-8,limit=200)[0]/(np.pi*rho*rho*(1-ld/3));r=dict(case=c['id'],single_lens_limit=mu,lc=float(d['mu'][c['worst']]),vbm=float(d['vbm'][c['worst']]),u=float(u),rho=rho,q=p['q']);rows.append(r);print(r,flush=True)
(OUT/'single_lens.json').write_text(json.dumps(rows,indent=2)+'\n')

from pathlib import Path
import numpy as np,pandas as pd,json,math
from numpy.polynomial import Polynomial as P
r=Path(__file__).parent
x=np.cos(np.pi*np.arange(1,8)/8)
w=[];v=[];hs=[];ks=[];ls=[]
integ=lambda p:p.integ()(1)-p.integ()(-1)
for i,t in enumerate(x):
 L=P([1.])
 for j,z in enumerate(x):
  if i!=j:L*=P([-z,1])/(t-z)
 ls.append(L);hs.append((P([1])-2*L.deriv()(t)*P([-t,1]))*L*L);ks.append(P([-t,1])*L*L)
 w.append(integ((P([1])-2*L.deriv()(t)*P([-t,1]))*L*L));v.append(integ(P([-t,1])*L*L))
w=np.array(w);v=np.array(v)
assert max(abs(np.dot(w,x**k)+np.dot(v,k*x**(k-1) if k else x*0)-(0 if k%2 else 2/(k+1))) for k in range(14))<1e-10
fw=np.array([4*np.sin(np.pi*k/8)/8*sum(np.sin(j*np.pi*k/8)/j for j in range(1,8,2)) for k in range(1,8)])
d=pd.read_csv(r/'raw.tsv',sep=r'\s+');out=[]
for (name,pan),g in d.groupby(['name','panel'],sort=False):
 assert len(g)==7
 f=g.g.to_numpy();der=g.dg.to_numpy();q7=fw@f;qh=w@f+v@der;ref=g.ref128.iloc[0];unc=abs(g.ref64.iloc[0]-ref)
 hp=sum((hs[i]*f[i]+ks[i]*der[i] for i in range(7)),P([0.]));lp=sum((ls[i]*f[i] for i in range(7)),P([0.]));delta=hp-lp
 bound=sum(abs(c)*2/(i+1) for i,c in enumerate(delta.coef) if i%2==0)+sum(abs(c)*2/(i+1) for i,c in enumerate(delta.coef) if i%2)
 gx,gw=np.polynomial.legendre.leggauss(64);l1=float(gw@abs(delta(gx)))
 out.append(dict(polynomial_l1_sample=l1,polynomial_l1_upper=bound,name=name,panel=int(pan),kind=int(g.kind.iloc[0]),q7=q7,hermite=qh,q15=g.q15.iloc[0],reference=ref,reference_uncertainty=unc,estimate=abs(qh-q7),error7=abs(q7-ref),errorH=abs(qh-ref),error15=abs(g.q15.iloc[0]-ref),fd_max_scaled=float(np.nanmax(abs(der-g.fd)/(1+abs(g.fd)))),base_us=g.base_us.sum(),derivative_us=g.derivative_us.sum()))
p=pd.DataFrame(out);p.to_csv(r/'panels.tsv',sep='\t',index=False)
valid=p[np.isfinite(p.reference)&np.isfinite(p.hermite)&(p.reference_uncertainty<1e-7*np.maximum(1,abs(p.reference)))]
bump=P.fromroots(x)**2
assert max(abs(bump(x)))<1e-12 and max(abs(bump.deriv()(x)))<1e-11
assert integ(bump)>0
summary={'invisible_bump_integral':float(integ(bump)),'panels':len(p),'usable':len(valid),'hermite_better_than_7':int((valid.errorH<valid.error7).sum()),'hermite_better_than_15':int((valid.errorH<valid.error15).sum()),'median_error7':float(valid.error7.median()),'median_errorH':float(valid.errorH.median()),'median_error15':float(valid.error15.median()),'derivative_to_base_ratio_median':float((valid.derivative_us/valid.base_us).median()),'fd_scaled_max':float(valid.fd_max_scaled.max())}
for safety in [1,2,4]:
 summary['underestimate_'+str(safety)]=int((valid.error7>safety*valid.estimate+valid.reference_uncertainty+1e-10).sum())
summary['polynomial_envelope_underestimates']=int((valid.error7>valid.polynomial_l1_upper+valid.reference_uncertainty+1e-10).sum())
summary['sampled_l1_underestimates']=int((valid.error7>2*valid.polynomial_l1_sample+valid.reference_uncertainty+1e-10).sum())
summary['sampled_l1_median']=float(valid.polynomial_l1_sample.median())
summary['polynomial_envelope_median']=float(valid.polynomial_l1_upper.median())
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)

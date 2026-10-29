"""Independent original-equation high precision source-disk validation only."""
import os,sys,json,math,time,resource
from pathlib import Path
import mpmath as mp
P=Path(__file__).resolve().parent;O=P.parent/'triple_event_precision_20261029'
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,305));mp.mp.dps=70
case=int(sys.argv[1]);c=json.load(open(O/'inputs'/f'{case:04d}.json'));p=c['params'];Q=mp.mpf;C=mp.mpc
# Freeze the actual binary64 physical input; evaluate its original equation in 70 digits.
den=1+p['q']+p['q2'];off=p['q']*p['s']/(1+p['q']);l=[C(off),C(-p['s']/(1+p['q'])),C(off-p['sep2']*math.cos(p['ang']),-p['sep2']*math.sin(p['ang']))];m=list(map(Q,[1/den,p['q']/den,p['q2']/den]));maxres=Q(0)
def field(z,w):
 f=z-w;k=C(0)
 for mass,lens in zip(m,l):
  d=mp.conj(z-lens)
  if not d:return None
  f-=mass/d;k+=mass/d**2
 return f,k

def solve(z,w):
 for _ in range(100):
  a=field(z,w)
  if a is None:return None
  f,k=a;det=1-abs(k)**2
  if not det:return None
  if abs(f)<Q('1e-52'):return z,det,abs(f)
  z-=(f-k*mp.conj(f))/det
 return None

def roots_at(w):
 seeds=[w];roots=[]
 for j,lens in enumerate(l):
  h=lens-w-sum(m[k]/mp.conj(lens-l[k])for k in range(3)if k!=j)
  if h:seeds +=[lens+v*m[j]/mp.conj(h)for v in [Q('.5'),1,2,-1]]
 for lens,mass in zip(l,m):
  for k in range(16):seeds.append(lens+mp.sqrt(mass)*mp.exp(2j*mp.pi*k/16))
 for z in seeds:
  a=solve(z,w)
  if a is not None and all(abs(a[0]-old)>Q('1e-35')for old in roots):roots.append(a[0])
 assert len(roots)==4,(case,len(roots),w)
 return roots

def disk(w,rho,limb,roots,n,nt):
 global maxres
 nodes,weights=mp.gauss_quadrature(n,'legendre');total=Q(0);samples=[]
 for a,b in zip(nodes,weights):
  th=(a+1)*mp.pi/4;r=rho*mp.sin(th);weight=b*mp.pi/4*2*mp.sin(th)*mp.cos(th)*(1-limb+limb*mp.cos(th))/(1-limb/3)
  for k in range(nt):
   source=w+r*mp.exp(2j*mp.pi*k/nt);images=[solve(z,source)for z in roots];assert all(z is not None for z in images)
   assert len({mp.nstr(z[0],40)for z in images})==4
   assert sum(z[1]>0 for z in images)==1
   maxres=max(maxres,max(z[2]for z in images));value=sum(1/abs(z[1])for z in images);total+=weight*value/nt;samples.append(mp.nstr(value,65))
 return total,samples
out=dict(case=case,dps=mp.mp.dps,physical_lenses=[[str(z.real),str(z.imag)]for z in l],physical_masses=list(map(str,m)),rows=[]);start=time.thread_time();indices=range(5)if case in [467,515]else[2]
for j in indices:
 t=c['epochs'][j];tau=(t-p['t0'])/p['tE'];w=C(tau*math.cos(p['alpha'])-p['u0']*math.sin(p['alpha']),tau*math.sin(p['alpha'])+p['u0']*math.cos(p['alpha']));roots=roots_at(w)
 for n,nt in [(4,16),(8,32),(16,64)]:
  value,samples=disk(w,Q(p['rho']),Q(c['limb']),roots,n,nt);out['rows'].append(dict(index=j,n=n,angles=nt,mu=mp.nstr(value,65),uniform_samples=samples,images=[[str(z.real),str(z.imag)]for z in roots]));out.update(maximum_equation_residual=str(maxres),cpu_seconds=time.thread_time()-start);(P/f'{case:04d}-mp-reference.json').write_text(json.dumps(out,indent=2));print(case,j,n,mp.nstr(value,24),flush=True)

"""Validation only: 80-digit original-equation image solves and source-disk quadrature."""
import json,math,resource,time
from pathlib import Path
import mpmath as mp
P=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,125));mp.mp.dps=80
c=json.load(open(P/'inputs/0514.json'));p=c['params'];den=1+p['q']+p['q2'];off=p['q']*p['s']/(1+p['q']);lenses=[mp.mpc(off,0),mp.mpc(-p['s']/(1+p['q']),0),mp.mpc(off-p['sep2']*math.cos(p['ang']),-p['sep2']*math.sin(p['ang']))];m=[mp.mpf(x)for x in [1/den,p['q']/den,p['q2']/den]]
def field(z,w):
 f=z-w;k=mp.mpc(0)
 for mass,lens in zip(m,lenses):
  d=mp.conj(z-lens)
  if not d:return None
  f-=mass/d;k+=mass/d**2
 return f,k

def solve(z,w):
 for _ in range(90):
  a=field(z,w)
  if a is None:return None
  f,k=a;det=1-abs(k)**2
  if not det:return None
  if abs(f)<mp.mpf('1e-60'):return z,det,abs(f)
  z-=(f-k*mp.conj(f))/det
 return None
out=dict(case=514,dps=80,rows=[],epochs=[]);maxres=mp.mpf(0);start=time.thread_time()
for j,t in enumerate(c['epochs']):
 tau=(t-p['t0'])/p['tE'];w=mp.mpc(tau*math.cos(p['alpha'])-p['u0']*math.sin(p['alpha']),tau*math.sin(p['alpha'])+p['u0']*math.cos(p['alpha']));roots=[];seeds=[w]
 for k,l in enumerate(lenses):
  h=l-w-sum(m[a]/mp.conj(l-lenses[a])for a in range(3)if a!=k);d=m[k]/mp.conj(h)
  seeds += [l+scale*d for scale in [mp.mpf('.5'),1,2,-1]]
 seeds += [mp.mpc(x,y)for x in [-100,-50,0,50,100]for y in [-50,-1,0,1,50]]
 for z in seeds:
  a=solve(z,w)
  if a is not None and all(abs(a[0]-old)>mp.mpf('1e-30')for old in roots):roots.append(a[0])
 assert len(roots)==4,(j,len(roots))
 point=sum(1/abs(solve(z,w)[1])for z in roots);out['epochs'].append(dict(index=j,source_x=mp.nstr(w.real,80),source_y=mp.nstr(w.imag,80),point_mu=mp.nstr(point,80),images=[[mp.nstr(z.real,80),mp.nstr(z.imag,80)]for z in roots]))
 for n,ntheta in [(2,8),(4,16),(8,32)]:
  nodes,weights=mp.gauss_quadrature(n,'legendre');total=mp.mpf(0);samples=[]
  for a,b in zip(nodes,weights):
   radius=mp.mpf(p['rho'])*mp.sqrt((a+1)/2)
   for k in range(ntheta):
    source=w+radius*mp.exp(2j*mp.pi*k/ntheta);images=[solve(z,source)for z in roots];assert all(z is not None for z in images);maxres=max(maxres,max(z[2]for z in images));mu=sum(1/abs(z[1])for z in images);total+=b*mu/(2*ntheta);samples.append(mp.nstr(mu,80))
  out['rows'].append(dict(index=j,n=n,angles=ntheta,mu=mp.nstr(total,80),uniform_samples=samples));out.update(maximum_equation_residual=mp.nstr(maxres,80),cpu_seconds=time.thread_time()-start);(P/'independent-far-disk.json').write_text(json.dumps(out,indent=2));print(j,n,mp.nstr(total,30),flush=True)

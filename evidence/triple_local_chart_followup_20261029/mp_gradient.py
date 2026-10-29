"""70-digit original-equation finite-source central differences; validation only."""
import sys,json,time,math
from pathlib import Path
P=Path(__file__).resolve().parent
prefix=(P/'mp_reference.py').read_text().split('out=dict(case=case,')[0]
ns={'__file__':str(P/'mp_reference.py'),'__name__':'mp_validation'}
exec(compile(prefix,str(P/'mp_reference.py'),'exec'),ns)
mp=ns['mp'];Q=mp.mpf;C=mp.mpc;case=int(sys.argv[1]);c=ns['c'];p={k:Q(v)for k,v in c['params'].items()};t=Q(c['epochs'][2]);start=time.thread_time()
def geometry(v):
 den=1+v['q']+v['q2'];off=v['q']*v['s']/(1+v['q']);return [C(off),C(-v['s']/(1+v['q'])),C(off-v['sep2']*mp.cos(v['ang']),-v['sep2']*mp.sin(v['ang']))],[1/den,v['q']/den,v['q2']/den]
def source(v):
 tau=(t-v['t0'])/v['tE'];return C(tau*mp.cos(v['alpha'])-v['u0']*mp.sin(v['alpha']),tau*mp.sin(v['alpha'])+v['u0']*mp.cos(v['alpha']))
ns['l'],ns['m']=geometry(p);base_roots=ns['roots_at'](source(p))
def value(v):
 ns['l'],ns['m']=geometry(v)
 roots=ns['roots_at'](source(v))
 return ns['disk'](source(v),v['rho'],Q(c['limb']),roots,8,32)[0]
steps={'t0':min(abs(p['u0']),p['rho'])*Q('.01'),'u0':abs(p['u0'])*Q('.001'),'tE':p['tE']*Q('.001'),'alpha':Q('.0001'),'rho':p['rho']*Q('.001'),'s':p['s']*Q('.00001'),'q':p['q']*Q('.0001'),'sep2':p['sep2']*Q('.00001'),'q2':p['q2']*Q('.001'),'ang':Q('.000001')}
out=dict(case=case,index=2,dps=70,radial_nodes=8,angles=32,rows=[])
existing=P/f'{case:04d}-mp-gradient-v2.json'
if existing.exists():out=json.loads(existing.read_text())
for key,h in steps.items():
 if any(r['parameter']==key for r in out['rows']):continue
 estimates=[];samples=[]
 for step in [h,h/2]:
  a=dict(p);b=dict(p);a[key]+=step;b[key]-=step
  va=value(a);vb=value(b);estimates.append((va-vb)/(2*step));samples.append(dict(h=str(step),plus=str(va),minus=str(vb)))
 rich=(4*estimates[1]-estimates[0])/3
 out['rows'].append(dict(parameter=key,estimates=list(map(str,estimates)),richardson=str(rich),samples=samples));out.update(cpu_seconds=time.thread_time()-start,maximum_equation_residual=str(ns['maxres']));(P/f'{case:04d}-mp-gradient-v2.json').write_text(json.dumps(out,indent=2));print(case,key,mp.nstr(rich,20),flush=True)

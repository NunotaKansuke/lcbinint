import json,os,sys,subprocess,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
s=(P/'audit.py').read_text().split("if '--worker'in sys.argv:")[0];exec(compile(s,str(P/'audit.py'),'exec'))
os.sched_setaffinity(0,{0})
if '--child'in sys.argv:
 if '--before'in sys.argv:os.environ['AUDIT_BASELINE']='1'
 lc=native();report=[]
 for c in json.loads((P/'cases.json').read_text()):
  if c['name']not in ['crossing','equal','close','wide','planetary','resonant','secondary_dominant']:continue
  model=lc.LightCurve(lens='triple',param_type='vbm',options=lc.Options(mode=0));p=parameters(c);p['rho']=0.
  rng=np.random.default_rng(173);xs=rng.uniform(-2.5,2.5,64)
  for y in [-.6,.03,.15,.7]:
   p['u0']=-y;ts=(p['q']/(1+p['q'])*p['s']-xs).tolist();out=model.magnification(ts,p)
   report.append(dict(name=c['name'],params=dict(p),times=ts,mu=out.magnifications.tolist(),status=out.status.tolist()))
 print(json.dumps(report,allow_nan=False));sys.exit()
results={}
for mode in ['before','after']:
 z=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',*(['--before']if mode=='before'else[])],capture_output=True,text=True,check=True,timeout=60,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));results[mode]=json.loads(z.stdout)
import VBMicrolensing as V
checks=[]
for a,b in zip(results['before'],results['after']):
 assert a['params']==b['params'];p=b['params'];off=p['q']*p['s']/(1+p['q']);total=1+p['q']+p['q2'];m=[1/total,p['q']/total,p['q2']/total];third=complex(off,0)-p['sep2']*complex(math.cos(p['ang']),math.sin(p['ang']));geo=[off,0,m[0],-p['s']/(1+p['q']),0,m[1],third.real,third.imag,m[2]];refs={}
 for method in ['Singlepoly','Multipoly','Nopoly']:
  v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,method));v.SetLensGeometry(geo);refs[method]=[v.MultiMag0(x,p['u0'])for x in b['times']]
 a['references']=refs;b['references']=refs
 gap=float(np.max(np.abs(np.array(a['mu'])/b['mu']-1)));errs={k:float(np.max(np.abs(np.array(b['mu'])/z-1)))for k,z in refs.items()};checks.append(dict(name=b['name'],u0=p['u0'],before_after_max=gap,reference_errors=errs,nonzero_status=sum(b['status'])))
report=dict(native=results,checks=checks,protocol='1792 point-source epochs per native implementation, seven lens configurations, four source latitudes and 64 discontinuous randomized horizontal positions. Ordinary mode=0/rho=0 native API vs direct VBM MultiMag0 with reconstructed identical physical lens coordinates. Each VBM method resets its geometry before its complete source group. This is a point-root diagnostic, not finite-source timing.')
(P/'point-check.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('max before/after',max(c['before_after_max']for c in checks),'max reference',max(max(c['reference_errors'].values())for c in checks),'nonzero status',sum(c['nonzero_status']for c in checks))

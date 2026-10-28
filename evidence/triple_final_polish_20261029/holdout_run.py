"""Fresh fixed-seed geometries, ordinary API and all three VBM methods."""
import os,sys,json,math,random,hashlib,time,subprocess,concurrent.futures,resource
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'holdout';D.mkdir(exist_ok=True)
if '--worker'in sys.argv:
 import worker_annulus as w
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 c=json.loads(Path(sys.argv[2]).read_text());engine=sys.argv[3]
 if engine.endswith('-tight')and c['params']['rho']>0:
  method=0 if engine.startswith('MultiPoly')else 1;p=c['params'];vp=[math.log(p[k])if k in ['s','q','rho','tE','sep2','q2']else p[k]for k in ['s','q','u0','alpha','rho','tE','t0','sep2','q2','ang']];probe=P.parent/'triple_stress_expanded_20261028/vbm-audit/probe';z=dict(engine=engine,classification='ok',mu=[],ref_diagnostics=[],reference_rtol=1e-7)
  for t in c['epochs']:
   r=None
   data=' '.join(map(repr,vp+[t]))+'\n'
   try:
    r=subprocess.run([str(probe),'1e-7',str(c['limb']),'100',str(method)],input=data,text=True,capture_output=True,timeout=8)
    if r.returncode:raise RuntimeError(dict(returncode=r.returncode,stderr=r.stderr[-1000:]))
    v=json.loads(r.stdout);z['ref_diagnostics'].append(v);z['mu'].append(v['mu'])
   except Exception as exc:
    z.update(classification='reference_failure',error=repr(exc))
    if r is not None:z['last_reference_stdout']=r.stdout[-4000:]
    break
  if any(not math.isfinite(x)or x<=0 for x in z['mu']):z['classification']='invalid_value'
 else:
  cc=json.loads(json.dumps(c))
  if engine.endswith('-tight'):cc['rtol']=1e-7
  z=w.run(cc,engine.removesuffix('-tight'));z['engine']=engine
 print(json.dumps(w.clean(z),allow_nan=False),flush=True);sys.exit()
seed=2026102907;rng=random.Random(seed);path=D/'cases.json'
if path.exists():cases=json.loads(path.read_text())
else:
 cases=[]
 for i in range(240):
  family='moderate'if i<128 else'wide'if i<208 else'near_collinear'if i<224 else'point'
  lower=-3 if family=='moderate'else-6
  p=dict(t0=rng.uniform(-2,2),u0=rng.uniform(-1.5,1.5),tE=10**rng.uniform(-.5,1),alpha=rng.uniform(-math.pi,math.pi),s=10**rng.uniform(-.6,.7),q=10**rng.uniform(lower,1),q2=10**rng.uniform(lower,.7),sep2=10**rng.uniform(-.6,.7),ang=rng.uniform(-math.pi,math.pi),rho=10**rng.uniform(-4,-1))
  if family=='near_collinear':p['ang']=rng.choice([-1,1])*10**rng.uniform(-9,-3)
  if family=='point':p['rho']=0.
  cases.append(dict(name=f'holdout-{family}-{i:04d}',family=family,params=p,epochs=[p['t0']+a*p['tE']for a in [-1,-.25,0,.25,1]],limb=0. if family=='point'else [0.,.5,1.][i%3],rtol=[1e-3,1e-4][i%2]))
 path.write_text(json.dumps(cases,indent=2))
I=D/'inputs';I.mkdir(exist_ok=True);R=D/'rows';R.mkdir(exist_ok=True)
for i,c in enumerate(cases):(I/f'{i:04d}.json').write_text(json.dumps(c))
so=Path(os.environ['TRIPLE_BUILD'])/'lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so';identity=hashlib.sha256(so.read_bytes()).hexdigest();start=time.time();engines=['value','gradient','SinglePoly','MultiPoly','NoPoly','MultiPoly-tight','NoPoly-tight']
(D/'provenance.json').write_text(json.dumps(dict(seed=seed,cases_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native_sha256=identity,workers=6,cpu_limit=20,wall_limit=50,reference_point_wall_limit=8,ram_per_worker_gib=2,geometry_families=dict(moderate=128,wide=80,near_collinear=16,point=16)),indent=2))
def task(job):
 i,e=job;f=R/f'{i:04d}-{e}.json';inp=I/f'{i:04d}.json';ih=hashlib.sha256(inp.read_bytes()).hexdigest()
 if f.exists():
  z=json.loads(f.read_text());assert z['input_sha256']==ih
  if e in ['value','gradient']and'native_sha256'in z:assert z['native_sha256']==identity
  return z
 try:
  r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(inp),e],text=True,capture_output=True,timeout=50)
  z=json.loads(r.stdout)if not r.returncode else dict(engine=e,classification='worker_exit',returncode=r.returncode,stderr=r.stderr[-2000:])
 except Exception as exc:z=dict(engine=e,classification='harness_failure',error=repr(exc))
 z.update(case=i,input_sha256=ih);f.write_text(json.dumps(z,indent=2));return z
counts={};done=0
with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:
 jobs=[pool.submit(task,(i,e))for i in range(len(cases))for e in engines]
 for future in concurrent.futures.as_completed(jobs):
  z=future.result();done+=1;counts[z['classification']]=counts.get(z['classification'],0)+1
  state=dict(state='RUNNING',pid=os.getpid(),completed=done,total=len(jobs),counts=counts,native_sha256=identity,elapsed=time.time()-start,updated=time.time());(D/'status.json').write_text(json.dumps(state,indent=2))
  if done%100==0 or z['classification']!='ok':print(done,z['case'],z['engine'],z['classification'],flush=True)
state.update(state='COMPLETE_WITH_FINDINGS'if set(counts)-{'ok'}else'COMPLETE');(D/'status.json').write_text(json.dumps(state,indent=2))

import os,sys,json,math,random,subprocess,time,hashlib,concurrent.futures,resource
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent/'triple_event_precision_20261029';D=P/'holdout';D.mkdir(exist_ok=True);I=D/'inputs';I.mkdir(exist_ok=True);R=D/'rows';R.mkdir(exist_ok=True)
seed=2026102911;rng=random.Random(seed);cases=[]
for base in [400,403,433,460,467,484,515]:
 c=json.load(open(O/'inputs'/f'{base:04d}.json'))
 for k in range(8):
  p=dict(c['params']);p['q']*=10**rng.uniform(-.3,.3);p['q2']*=10**rng.uniform(-.4,.4);p['u0']=10**rng.uniform(-3.4,-.6);p['rho']=min(10**rng.uniform(-6,-3.4),p['u0']*.025);p['alpha']=rng.uniform(-math.pi,math.pi)
  p['ang']+=rng.uniform(-1e-9,1e-9);cases.append(dict(name=f'critical-progress-holdout-{base}-{k}',family=f'near-{base}',params=p,epochs=[-1.,-.2,0.,.2,1.],limb=[0.,.5,1.][k%3],rtol=[1e-3,1e-4][k%2]))
f=D/'cases.json'
if f.exists():assert json.load(open(f))==cases
else:f.write_text(json.dumps(cases,indent=2))
for i,c in enumerate(cases):(I/f'{i:04d}.json').write_text(json.dumps(c))
# Exact repaired fixtures get VBM runs too; they are not counted as new cases.
for i,base in enumerate([400,403,433,460,467,484,515],len(cases)):(I/f'{i:04d}.json').write_bytes((O/'inputs'/f'{base:04d}.json').read_bytes())
worker=(O/'worker.py').read_text().replace('resource.setrlimit(resource.RLIMIT_CPU,(60,62))',"resource.setrlimit(resource.RLIMIT_CPU,(60,62)if sys.argv[2]in ['value','gradient']else(8,10))")
(P/'holdout_worker.py').write_text(worker)
probe=P.parent/'triple_stress_expanded_20261028/vbm-audit/probe';start=time.time();counts={};jobs=[(i,e)for i in range(len(cases)+7)for e in ['value','gradient','SinglePoly','MultiPoly','NoPoly','MultiPoly-tight','NoPoly-tight']if i<len(cases)or e not in ['value','gradient']]
def ref_limit():resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(8,10));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
def task(job):
 i,e=job;f=R/f'{i:04d}-{e}.json';inp=I/f'{i:04d}.json';inputhash=hashlib.sha256(inp.read_bytes()).hexdigest()
 if f.exists():
  z=json.load(open(f));assert z['input_sha256']==inputhash;return z
 try:
  if not e.endswith('-tight'):
   r=subprocess.run([sys.executable,str(P/'holdout_worker.py'),str(inp),e],capture_output=True,text=True,timeout=90 if e in ['value','gradient']else 15,env=dict(os.environ,TRIPLE_BUILD=str(P/'early-package'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
   z=json.loads(r.stdout)if r.returncode==0 else dict(classification='worker_exit',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
  else:
   c=json.load(open(inp));p=c['params'];vp=[math.log(p[k])if k in ['s','q','rho','tE','sep2','q2']else p[k]for k in ['s','q','u0','alpha','rho','tE','t0','sep2','q2','ang']];zs=[];mu=[]
   for t in c['epochs']:
    r=subprocess.run([str(probe),'1e-7',str(c['limb']),'100','1'if e=='NoPoly-tight'else'0'],input=' '.join(map(str,vp+[t]))+'\n',capture_output=True,text=True,timeout=15,preexec_fn=ref_limit,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    if r.returncode:raise RuntimeError(dict(returncode=r.returncode,stderr=r.stderr,stdout=r.stdout))
    row=json.loads(r.stdout);zs.append(row);mu.append(row['mu'])
   z=dict(classification='ok',mu=mu,ref_diagnostics=zs,reference_rtol=1e-7)
 except Exception as exc:z=dict(classification='harness_failure',error=repr(exc))
 z.update(case=i,engine=e,input_sha256=inputhash);tmp=f.with_suffix('.writing');tmp.write_text(json.dumps(z,indent=2,allow_nan=False));tmp.replace(f);return z
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
 for n,z in enumerate(pool.map(task,jobs),1):
  counts[z['engine']+':'+z['classification']]=counts.get(z['engine']+':'+z['classification'],0)+1
  status=dict(state='RUNNING',completed=n,total=len(jobs),counts=counts,elapsed=time.time()-start,seed=seed,workers=4,reference_cpu_seconds=8,pid=os.getpid());f=D/'status.json';tmp=f.with_suffix('.writing');tmp.write_text(json.dumps(status,indent=2));tmp.replace(f)
  if n%50==0 or z['classification']!='ok':print(n,z['case'],z['engine'],z['classification'],flush=True)
status['state']='COMPLETE_WITH_FINDINGS';f.write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)

import os,sys,json,math,random,hashlib,time,subprocess,concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'followup';D.mkdir(exist_ok=True);I=D/'inputs';I.mkdir(exist_ok=True);R=D/'ready-rows';R.mkdir(exist_ok=True)
seed=2026102908;rng=random.Random(seed);cases=[]
for i in range(32):
 s=10**rng.uniform(1.7,2.3);first=i<16;p=dict(t0=0.,u0=rng.uniform(.2,1.)if first else 10**rng.uniform(-3,-1),tE=1.,alpha=rng.uniform(-math.pi,math.pi),s=s,q=10**rng.uniform(-9,-6)if first else 10**rng.uniform(-.3,.3),q2=10**rng.uniform(-.3,.3)if first else 10**rng.uniform(-9,-6),sep2=s*(1+10**rng.uniform(-4,-1)),ang=(math.pi if first else 0.)+rng.uniform(-1e-10,1e-10),rho=10**rng.uniform(-1.5,-.7)if first else 10**rng.uniform(-5,-3))
 cases.append(dict(name=f'wide-faint-holdout-{i:03d}',family='wide_tiny_secondary'if first else'wide_tiny_third',params=p,epochs=[-1.,-.2,0.,.2,1.],limb=[0.,.5,1.][i%3],rtol=[1e-3,1e-4][i%2]))
f=D/'cases.json'
if f.exists():assert json.loads(f.read_text())==cases
else:f.write_text(json.dumps(cases,indent=2))
for i,c in enumerate(cases):(I/f'{i:04d}.json').write_text(json.dumps(c))
identity=hashlib.sha256((P/'ready-package/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so').read_bytes()).hexdigest()
(D/'ready-provenance.json').write_text(json.dumps(dict(seed=seed,native_sha256=identity,cases_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),workers=6,native_cpu_seconds=60,reference_cpu_seconds=20,wall_seconds=120,ram_per_worker_gib=2),indent=2))
engines=['value','gradient'];start=time.time();counts={}
def task(job):
 i,e=job;f=R/f'{i:04d}-{e}.json';inp=I/f'{i:04d}.json'
 if f.exists():return json.loads(f.read_text())
 args=[str(P/'worker.py'),str(inp),e]if e in ['value','gradient']else[str(P.parent/'triple_final_polish_20261029/holdout_run.py'),'--worker',str(inp),e]
 try:
  r=subprocess.run([sys.executable,*args],capture_output=True,text=True,timeout=120,env=dict(os.environ,TRIPLE_BUILD=str(P/'ready-package'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));z=json.loads(r.stdout)if r.returncode==0 else dict(classification='worker_exit',returncode=r.returncode,stderr=r.stderr[-2000:])
 except Exception as exc:z=dict(classification='harness_failure',error=repr(exc))
 z.update(case=i,engine=e,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest());f.write_text(json.dumps(z,indent=2));return z
with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:
 for n,z in enumerate(pool.map(task,[(i,e)for i in range(len(cases))for e in engines]),1):
  counts[z['classification']]=counts.get(z['classification'],0)+1;status=dict(state='RUNNING',completed=n,total=len(cases)*len(engines),counts=counts,elapsed=time.time()-start,pid=os.getpid());tmp=D/'ready-status.writing';tmp.write_text(json.dumps(status));tmp.replace(D/'ready-status.json');print(n,z['case'],z['engine'],z['classification'],flush=True)
status['state']='COMPLETE_WITH_FINDINGS'if set(counts)-{'ok'}else'COMPLETE';(D/'ready-status.json').write_text(json.dumps(status,indent=2))

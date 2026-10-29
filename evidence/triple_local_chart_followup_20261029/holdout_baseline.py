import os,sys,json,hashlib,subprocess,time,concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent/'triple_event_precision_20261029';D=P/'holdout/before-rows';D.mkdir(exist_ok=True);start=time.time()
def task(job):
 i,e=job;inp=P/'holdout/inputs'/f'{i:04d}.json';f=D/f'{i:04d}-{e}.json'
 if f.exists():return json.load(open(f))
 try:
  r=subprocess.run([sys.executable,str(O/'worker.py'),str(inp),e],capture_output=True,text=True,timeout=120,env=dict(os.environ,TRIPLE_BUILD=str(O/'polished-package'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));z=json.loads(r.stdout)if r.returncode==0 else dict(classification='worker_exit',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as exc:z=dict(classification='harness_failure',error=repr(exc))
 z.update(case=i,engine=e,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest());f.write_text(json.dumps(z,indent=2));return z
counts={}
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
 for n,z in enumerate(pool.map(task,[(i,e)for i in range(56)for e in ['value','gradient']]),1):
  key=z['engine']+':'+z['classification'];counts[key]=counts.get(key,0)+1
  if n%20==0:print(n,counts,flush=True)
(P/'holdout/before-status.json').write_text(json.dumps(dict(completed=n,counts=counts,elapsed=time.time()-start),indent=2))

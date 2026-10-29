import os,sys,json,hashlib,time,subprocess,concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent;O=P.parent/'triple_event_precision_20261029';package=P/os.environ.get('VARIANT','parent-package');tag=os.environ.get('TAG','pilot');D=P/(tag+'-rows');D.mkdir(exist_ok=True)
ids=[400,403,433,439,460,467,484,499,515]if '--full'not in sys.argv else range(1768)
jobs=[('main',i,e,O/'inputs'/f'{i:04d}.json')for i in ids for e in ['value','gradient']]
if '--full'in sys.argv:jobs +=[('followup',i,e,O/'followup/inputs'/f'{i:04d}.json')for i in range(32)for e in ['value','gradient']]
if '--full'in sys.argv and os.environ.get('INCLUDE_HOLDOUT')=='1':jobs +=[('holdout',i,e,P/'holdout/inputs'/f'{i:04d}.json')for i in range(56)for e in ['value','gradient']]
identity=hashlib.sha256((package/'lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so').read_bytes()).hexdigest();start=time.time()
def task(job):
 group,i,e,inp=job;f=D/f'{group}-{i:04d}-{e}.json';input_hash=hashlib.sha256(inp.read_bytes()).hexdigest()
 if f.exists():
  z=json.load(open(f));assert z['input_sha256']==input_hash;assert z.get('native_sha256',identity)==identity;return z
 try:
  r=subprocess.run([sys.executable,str(O/'worker.py'),str(inp),e],capture_output=True,text=True,timeout=120,env=dict(os.environ,TRIPLE_BUILD=str(package),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  z=json.loads(r.stdout)if r.returncode==0 else dict(classification='worker_exit',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as exc:z=dict(classification='harness_error',error=repr(exc))
 z.update(group=group,case=i,engine=e,input_sha256=input_hash);tmp=f.with_suffix('.writing');tmp.write_text(json.dumps(z,indent=2,allow_nan=False));tmp.replace(f);return z
counts={}
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('WORKERS','3')))as pool:
 for n,z in enumerate(pool.map(task,jobs),1):
  counts[z['classification']]=counts.get(z['classification'],0)+1;s=dict(state='RUNNING',completed=n,total=len(jobs),counts=counts,elapsed=time.time()-start,pid=os.getpid(),native_sha256=identity);f=P/(tag+'-status.json');tmp=f.with_suffix('.writing');tmp.write_text(json.dumps(s,indent=2));tmp.replace(f)
  if z['classification']!='ok'or n%100==0:print(n,z['group'],z['case'],z['engine'],z['classification'],flush=True)
s['state']='COMPLETE_WITH_FINDINGS'if set(counts)-{'ok'}else'COMPLETE';f.write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True)

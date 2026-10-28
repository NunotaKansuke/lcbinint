import os,sys,json,time,subprocess,hashlib,concurrent.futures,resource
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent
engines=['value','gradient'];cases=json.loads((P/'cases.json').read_text());D=P/os.environ.get('EXPANDED_ROWS','rows');D.mkdir(exist_ok=True);I=P/'inputs';I.mkdir(exist_ok=True)
for i,c in enumerate(cases):
 f=I/(f'{i:04d}'+'.json')
 if not f.exists():f.write_text(json.dumps(c))
selected=list(range(len(cases)))
if '--pilot'in sys.argv:selected=list(range(0,400,50))+list(range(400,568,21))+list(range(568,len(cases),120))
expected=hashlib.sha256((Path(os.environ.get('TRIPLE_BUILD',str(P/'local-package')))/'lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so').read_bytes()).hexdigest()
def task(i,e):
 f=D/(f'{i:04d}-{e}.json')
 if f.exists():
  z=json.loads(f.read_text());assert z.get('input_sha256')==hashlib.sha256((I/(f'{i:04d}.json')).read_bytes()).hexdigest()
  if e in ['value','gradient']and 'native_sha256'in z:assert z['native_sha256']==expected
  return i,e,z['classification']
 start=time.time();inp=I/(f'{i:04d}.json');z=dict(engine=e)
 try:
  r=subprocess.run([sys.executable,str(P/'worker.py'),str(inp),e],capture_output=True,text=True,timeout=120,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  if r.returncode:z.update(classification='worker_exit',returncode=r.returncode,stderr=r.stderr[-16000:],stdout=r.stdout[-4000:])
  else:z=json.loads(r.stdout)
 except subprocess.TimeoutExpired as exc:z.update(classification='wall_timeout',stderr=(exc.stderr or b'').decode(errors='replace')[-16000:])
 except Exception as exc:z.update(classification='harness_error',error=repr(exc))
 z.update(case=i,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),wall_seconds=time.time()-start)
 tmp=f.with_suffix('.writing');tmp.write_text(json.dumps(z,indent=2,allow_nan=False));tmp.replace(f);return i,e,z['classification']
def save_status(path,status):
 tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(status,indent=2));tmp.replace(path)
start=time.time();counts={};done=0
with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:
 jobs=[pool.submit(task,i,e)for i in selected for e in engines]
 for future in concurrent.futures.as_completed(jobs):
  i,e,state=future.result();done+=1;counts[state]=counts.get(state,0)+1
  status=dict(state='RUNNING',pid=os.getpid(),completed=done,total=len(jobs),counts=counts,updated=time.time(),elapsed=time.time()-start,native_sha256=expected)
  save_status(P/('pilot-status.json'if '--pilot'in sys.argv else 'deferred-status.json'),status)
  if state!='ok'or done%50==0:print(done,i,e,state,flush=True)
status['state']='COMPLETE_WITH_FINDINGS'if set(counts)-{'ok'}else'COMPLETE';save_status(P/('pilot-status.json'if '--pilot'in sys.argv else 'deferred-status.json'),status);print(json.dumps(status),flush=True)

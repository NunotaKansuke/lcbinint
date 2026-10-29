import os,sys,json,subprocess,time,concurrent.futures,resource,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent/'triple_final_polish_20261029';D=P/'new-references';D.mkdir(exist_ok=True)
ids=[504,514]
engines=['SinglePoly','MultiPoly','NoPoly','MultiPoly-tight','NoPoly-tight'];start=time.time()
def task(job):
 i,e=job;f=D/f'{i:04d}-{e}.json'
 if f.exists():return json.loads(f.read_text())
 try:
  r=subprocess.run([sys.executable,str(B/'holdout_run.py'),'--worker',str(P/'inputs'/f'{i:04d}.json'),e],capture_output=True,text=True,timeout=50,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
  z=json.loads(r.stdout)if r.returncode==0 else dict(classification='worker_exit',returncode=r.returncode,stderr=r.stderr[-2000:])
 except Exception as exc:z=dict(classification='harness_failure',error=repr(exc))
 z.update(case=i,engine=e);f.write_text(json.dumps(z,indent=2));return z
with concurrent.futures.ThreadPoolExecutor(max_workers=1)as pool:
 for n,z in enumerate(pool.map(task,[(i,e)for i in ids for e in engines]),1):
  (P/'new-reference-status.json').write_text(json.dumps(dict(state='RUNNING',pid=os.getpid(),completed=n,total=len(ids)*len(engines),elapsed=time.time()-start)))
  print(n,z['case'],z['engine'],z['classification'],flush=True)
(P/'new-reference-status.json').write_text(json.dumps(dict(state='COMPLETE',completed=n,total=n,elapsed=time.time()-start)))

import os,sys,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'vbm-controlled-timing';D.mkdir(exist_ok=True);rows=[];jobs=[(i,e)for i in [21,514,483,940,972,1508]for e in ['SinglePoly','MultiPoly','NoPoly']];start=time.time()
for n,(i,e)in enumerate(jobs,1):
 f=D/f'{i:04d}-{e}.json'
 if f.exists():z=json.load(open(f))
 else:
  try:
   r=subprocess.run([sys.executable,str(P/'polished_timing.py'),'--worker',str(i),e],capture_output=True,text=True,timeout=75,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
   if r.returncode:z=dict(case=i,engine=e,classification='worker_exit',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   else:
    try:z=json.loads(r.stdout)
    except Exception as exc:z=dict(case=i,engine=e,classification='invalid_worker_output',error=repr(exc),stdout=r.stdout,stderr=r.stderr)
  except Exception as exc:z=dict(case=i,engine=e,classification='harness_failure',error=repr(exc))
  f.write_text(json.dumps(z,indent=2))
 rows.append(z);status=dict(state='RUNNING',completed=n,total=len(jobs),elapsed=time.time()-start,pid=os.getpid());tmp=P/'vbm-timing-status.writing';tmp.write_text(json.dumps(status));tmp.replace(P/'vbm-timing-status.json');print(i,e,z.get('warm',z.get('error',z.get('returncode'))),flush=True)
(P/'vbm-controlled-timing.json').write_text(json.dumps(rows,indent=2));status['state']='COMPLETE_WITH_FINDINGS'if any('warm'not in r for r in rows)else'COMPLETE';(P/'vbm-timing-status.json').write_text(json.dumps(status,indent=2))

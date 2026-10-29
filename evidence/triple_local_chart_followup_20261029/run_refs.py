import os,sys,json,subprocess,time,concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent
jobs=[400,403,433,460,467,484,515]
def run(i):
 with open(P/f'{i:04d}-mp-reference.log','w')as f:
  t=time.time();r=subprocess.run([sys.executable,str(P/'mp_reference.py'),str(i)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),timeout=360)
 z=dict(case=i,returncode=r.returncode,wall_seconds=time.time()-t);(P/f'{i:04d}-mp-reference-status.json').write_text(json.dumps(z));print(z,flush=True);return z
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:results=list(pool.map(run,jobs))
(P/'mp-reference-status.json').write_text(json.dumps(results,indent=2))

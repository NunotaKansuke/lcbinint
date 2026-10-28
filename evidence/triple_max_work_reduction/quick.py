import os,json,sys,subprocess,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;cases={c['name']:c for c in json.loads((P/'cases.json').read_text())};rows=[]
for name in ['far','miss','grazing','close','large']:
 workers={};out={}
 try:
  for e in ['before_value','value','before_gradient','gradient']:
   c=dict(cases[name],limb=.5,rtol=1e-3,engine=e);w=subprocess.Popen([sys.executable,str(P/'audit.py'),'--worker',json.dumps(c)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(P/f'quick-{name}-{e}.log','w'),text=True,bufsize=1,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));workers[e]=w
   assert json.loads(w.stdout.readline())['ready'];out[e]=[]
  for i in range(10):
   keys=list(workers)
   for e in (keys if i%2 else keys[::-1]):
    w=workers[e];w.stdin.write('run\n');w.stdin.flush();z=json.loads(w.stdout.readline())
    if i>=2:out[e].append(z['cpu_ms'])
  row=dict(name=name,times={e:statistics.median(z)for e,z in out.items()});rows.append(row);print(row,flush=True)
 finally:
  for w in workers.values():w.stdin.write('stop\n');w.stdin.flush();w.wait()
(P/'quick.json').write_text(json.dumps(rows,indent=2)+'\n')

import sys,json,time,statistics,os
from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'audit.py').read_text().split("if '--worker'in sys.argv:")[0];exec(compile(s,str(P/'audit.py'),'exec'))
os.sched_setaffinity(0,{0});cases={c['name']:c for c in json.loads((P/'cases.json').read_text())};report=[]
# Each size/mode is a separate process so a previous mode cannot populate its
# native caches. Internal thread CPU timing excludes subprocess overhead.
if '--child'in sys.argv:
 c=json.loads(sys.argv[-1]);mode=c.pop('engine');model,p,times,call,constructor=setup(c,mode);samples=[]
 for i in range(23):
  a=time.thread_time_ns();out=call(p);elapsed=(time.thread_time_ns()-a)/1e6
  if i==0:first=elapsed
  if i>=3:samples.append(elapsed)
 print(json.dumps(dict(condition=c,engine=mode,constructor_ms=constructor,first_ms=first,median_ms=statistics.median(samples),samples_ms=samples)),flush=True);sys.exit()
for n in [1,17,65,257]:
 for mode in ['value','gradient','point_base']:
  c=dict(cases['far'],n=n,limb=0.,rtol=1e-3,engine=mode)
  p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',json.dumps(c)],capture_output=True,text=True,check=True,timeout=30,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
  report.append(json.loads(p.stdout));print(n,mode,report[-1]['median_ms'],flush=True)
(P/'scaling.json').write_text(json.dumps(report,indent=2)+'\n')

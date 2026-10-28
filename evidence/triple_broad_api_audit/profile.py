"""Enable perf only around an already-warm ordinary API workload."""
import os,sys,json,time
from pathlib import Path
s=Path(__file__).with_name('audit.py').read_text().split("if '--worker'in sys.argv:")[0]
exec(compile(s,str(Path(__file__).with_name('audit.py')),'exec'))
os.sched_setaffinity(0,{0});c=json.loads(sys.argv[-1]);model,p,times,call,_=setup(c,c['engine'])
for _ in range(3):call(p)
with open(c['control'],'w',buffering=1)as ctl:
 ctl.write('enable\n');a=time.thread_time();calls=0;checksum=0.
 while time.thread_time()-a<3.:
  q=dict(p,s=p['s']*(1.+(1+calls%2)*1e-4))if c.get('change')else p
  out=call(q);checksum+=float(out.magnifications[0]);calls+=1
 ctl.write('disable\n')
print(json.dumps(dict(calls=calls,cpu_seconds=time.thread_time()-a,checksum=checksum)),flush=True)

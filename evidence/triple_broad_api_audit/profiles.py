import os,sys,json,subprocess,signal
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'profiles';D.mkdir(exist_ok=True)
cases={c['name']:c for c in json.loads((P/'cases.json').read_text())}
configs=[]
for name in cases:
 for mode in ['value','gradient']:configs.append((name,mode,False))
configs += [('far','value',True),('grazing','value',True)]
report=[]
for name,mode,changed in configs:
 key=f'{name}-{mode}-'+('changed'if changed else'warm');ctl=D/(key+'.ctl');data=D/(key+'.data')
 c=dict(cases[name],limb=0. if name in ['far','miss','planetary']else .5,rtol=1e-3,engine=mode,change=changed,control=str(ctl))
 os.mkfifo(ctl)
 cmd=['perf','--buildid-dir',str(D/'perf-cache'),'record','--no-buildid-cache','-e','cpu-clock:u','-F','199','--call-graph','dwarf,8192','-D','-1','--control','fifo:'+str(ctl),'-o',str(data),'--',sys.executable,str(P/'profile.py'),json.dumps(c)]
 try:
  with open(D/(key+'.log'),'w')as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=log,text=True,start_new_session=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
  r=dict(key=key,condition=c,returncode=p.returncode,command=cmd);report.append(r)
  if p.returncode==0:
   with open(D/(key+'.report'),'w')as out:subprocess.run(['perf','report','--stdio','--no-children','--percent-limit','0.5','-i',str(data)],stdout=out,stderr=subprocess.STDOUT,check=True)
   with open(D/(key+'.stacks'),'w')as out:subprocess.run(['perf','script','-i',str(data)],stdout=out,stderr=subprocess.DEVNULL,check=True)
  (P/'profiles.json').write_text(json.dumps(report,indent=2)+'\n');print(key,p.returncode,flush=True)
 finally:ctl.unlink(missing_ok=True)

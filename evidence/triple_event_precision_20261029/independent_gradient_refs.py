import sys,os,json,math,time,resource,subprocess,concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent
if '--worker'in sys.argv:
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,125));os.sched_setaffinity(0,{7})
 import numpy as np,VBMicrolensing as V
 i=int(sys.argv[2]);name=sys.argv[3];method=sys.argv[4];c=json.loads((P/'cases.json').read_text())[i];p=c['params'];index={504:2}[i];t=c['epochs'][index];D=P/'reframed-independent-gradient';D.mkdir(exist_ok=True);f=D/f'{i:04d}-{name}-{method}.json';out=dict(case=i,index=index,parameter=name,method=method,rows=[])
 if f.exists():out=json.loads(f.read_text())
 h=(p['tE']*p['rho']*.002 if name=='t0'else abs(p[name])*.002)
 def value(par,n):
  v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,method));v.Tol=0.;v.RelTol=1e-9;v.a1=0.;v.a2=0.;vp=[math.log(par[k])if k in ['s','q','rho','tE','sep2','q2']else par[k]for k in ['s','q','u0','alpha','rho','tE','t0','sep2','q2','ang']];o=v.TripleLightCurve(vp,[t]);uniform=float(o[0][0]);x,y=float(o[1][0]),float(o[2][0]);a,w=np.polynomial.legendre.leggauss(n);th=(a+1)*math.pi/4;weights=w*math.pi/4;mu=[float(v.MultiMag(x,y,float(par['rho']*np.sin(theta))))for theta in th];integral=float(np.dot(weights,np.sin(th)**3*np.asarray(mu)));return ((1-c['limb'])*uniform+c['limb']*integral)/(1-c['limb']/3)
 for n in [64,128,256]:
  for step in [h,h*.5]:
   if any(r['n']==n and r['h']==step for r in out['rows']):continue
   a=time.thread_time();plus=value(dict(p,**{name:p[name]+step}),n);minus=value(dict(p,**{name:p[name]-step}),n);out['rows'].append(dict(n=n,h=step,plus=plus,minus=minus,derivative=(plus-minus)/(2*step),cpu_seconds=time.thread_time()-a));f.write_text(json.dumps(out,indent=2));print(i,name,method,n,step,out['rows'][-1]['derivative'],flush=True)
 sys.exit()
def task(job):
 i,name,method=job
 try:r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(i),name,method],capture_output=True,text=True,timeout=240,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));z=dict(case=i,parameter=name,method=method,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as exc:z=dict(case=i,parameter=name,method=method,error=repr(exc))
 d=P/'reframed-independent-gradient';d.mkdir(exist_ok=True);(d/f'{i:04d}-{name}-{method}-status.json').write_text(json.dumps(z));print(i,name,method,z.get('returncode',z.get('error')),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=1)as pool:list(pool.map(task,[(i,name,e)for i,name in [(504,'t0'),(504,'rho')]for e in ['Multipoly','Nopoly']]))
(P/'reframed-gradient-complete.json').write_text(json.dumps(dict(finished=time.time())))

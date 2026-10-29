import os,sys,json,time,math,resource,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'independent-values';D.mkdir(exist_ok=True)
if '--worker'in sys.argv:
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,125));os.sched_setaffinity(0,{6})
 import numpy as np,VBMicrolensing as V
 method=sys.argv[2];c=json.load(open(P/'inputs/0483.json'));p=c['params'];assert c['limb']==1.;out=dict(case=483,method=method,uniform_rtol=1e-9,rows=[]);f=D/f'0483-{method}.json'
 if f.exists():out=json.load(open(f))
 for n in [128,256,512]:
  a,w=np.polynomial.legendre.leggauss(n);theta=(a+1)*math.pi/4;weights=w*math.pi/4
  for j,t in enumerate(c['epochs']):
   if any(r['n']==n and r['index']==j for r in out['rows']):continue
   v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,method));v.Tol=0.;v.RelTol=1e-9;v.a1=0.;v.a2=0.;vp=[math.log(p[k])if k in ['s','q','rho','tE','sep2','q2']else p[k]for k in ['s','q','u0','alpha','rho','tE','t0','sep2','q2','ang']];start=time.thread_time();z=v.TripleLightCurve(vp,[t]);x,y=float(z[1][0]),float(z[2][0]);u=[float(v.MultiMag(x,y,float(p['rho']*np.sin(th))))for th in theta];mu=float(np.dot(weights,np.sin(theta)**3*np.asarray(u)))/(1-c['limb']/3);out['rows'].append(dict(n=n,index=j,source_x=x,source_y=y,uniform_values=u,mu=mu,cpu_seconds=time.thread_time()-start));f.write_text(json.dumps(out,indent=2));print(method,n,j,mu,flush=True)
 sys.exit()
for method in ['Multipoly','Nopoly']:
 try:r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',method],capture_output=True,text=True,timeout=180,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));z=dict(method=method,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as exc:z=dict(method=method,error=repr(exc))
 (D/f'0483-{method}-status.json').write_text(json.dumps(z,indent=2));print(method,z.get('returncode',z.get('error')),flush=True)

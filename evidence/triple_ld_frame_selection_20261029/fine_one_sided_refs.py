import os,sys,json,math,subprocess,concurrent.futures,resource,time
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'one-sided-fine-references';D.mkdir(exist_ok=True)
if '--worker'in sys.argv:
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(120,125));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));os.sched_setaffinity(0,{6,7})
 import numpy as np,VBMicrolensing as V
 i=int(sys.argv[2]);method=sys.argv[3];c=json.loads((P/'cases.json').read_text())[i];par=c['params'];name='alpha'if i==595 else'rho';h=1e-4 if name=='alpha'else par['rho']*.00125;offsets=sorted({0.}|{sign*h*scale for sign in [-1,1]for scale in [2,1,.5,.25,.125]});f=D/f'{i}-{method}.json';out=dict(case=i,method=method,parameter=name,base_step=h,epoch=0.,limb=c['limb'],uniform_rtol=1e-10,rows=[])
 if f.exists():out=json.loads(f.read_text())
 for n in [128,256]:
  a,w=np.polynomial.legendre.leggauss(n);theta=(a+1)*math.pi/4;weights=w*math.pi/4*np.sin(theta)**3
  for offset in offsets:
   if any(r['n']==n and r['offset']==offset for r in out['rows']):continue
   p=dict(par);p[name]+=offset;v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,method));v.Tol=0.;v.RelTol=1e-10;v.a1=0.;v.a2=0.;vp=[math.log(p[k])if k in ['s','q','rho','tE','sep2','q2']else p[k]for k in ['s','q','u0','alpha','rho','tE','t0','sep2','q2','ang']];start=time.thread_time();o=v.TripleLightCurve(vp,[0.]);uniform=float(o[0][0]);x,y=float(o[1][0]),float(o[2][0]);vals=[float(v.MultiMag(x,y,float(p['rho']*np.sin(t))))for t in theta]
   if any(not math.isfinite(x)or x<=0 for x in vals+[uniform]):raise RuntimeError('invalid independent uniform-disk value')
   mu=((1-c['limb'])*uniform+c['limb']*float(np.dot(weights,vals)))/(1-c['limb']/3);out['rows'].append(dict(n=n,offset=offset,mu=mu,uniform=uniform,uniform_disks=vals,cpu_seconds=time.thread_time()-start));f.write_text(json.dumps(out,indent=2))
 print(i,method,'COMPLETE',flush=True);sys.exit()
def task(job):
 i,m=job
 try:
  r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(i),m],capture_output=True,text=True,timeout=240,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));z=dict(case=i,method=m,returncode=r.returncode,stderr=r.stderr,stdout=r.stdout)
 except Exception as exc:z=dict(case=i,method=m,error=repr(exc))
 (D/f'{i}-{m}-status.json').write_text(json.dumps(z));print(i,m,z.get('returncode',z.get('error')),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=1)as pool:list(pool.map(task,[(1104,'Multipoly')]))
(P/'one-sided-fine-reference-complete.json').write_text(json.dumps(dict(finished=time.time())))

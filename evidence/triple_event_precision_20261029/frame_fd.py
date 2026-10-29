import os,sys,json,math,subprocess,resource
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'frame-fd';D.mkdir(exist_ok=True)
if '--worker'in sys.argv:
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(30,32))
 import VBMicrolensing as V
 method=sys.argv[2];c=json.load(open(P/'inputs/0321.json'));p=c['params'];t=c['epochs'][4];rows=[]
 def value(par):
  v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,method));v.Tol=0.;v.RelTol=1e-10;v.a1=0.;v.a2=0.;vp=[math.log(par[k])if k in ['s','q','rho','tE','sep2','q2']else par[k]for k in ['s','q','u0','alpha','rho','tE','t0','sep2','q2','ang']];return float(v.TripleLightCurve(vp,[t])[0][0])
 for h in [1e-3,5e-4,2e-4,1e-4,5e-5]:
  a=value(dict(p,alpha=p['alpha']+h));b=value(dict(p,alpha=p['alpha']-h));rows.append(dict(h=h,plus=a,minus=b,derivative=(a-b)/(2*h)));(D/f'0321-{method}.json').write_text(json.dumps(dict(case=321,index=4,method=method,rows=rows),indent=2))
 sys.exit()
for method in ['Multipoly','Nopoly']:
 try:r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',method],capture_output=True,text=True,timeout=60,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));z=dict(method=method,returncode=r.returncode,stderr=r.stderr)
 except Exception as exc:z=dict(method=method,error=repr(exc))
 (D/f'0321-{method}-status.json').write_text(json.dumps(z,indent=2));print(z,flush=True)

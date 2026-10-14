import json,time
from pathlib import Path
import VBMicrolensing
D=Path(__file__).resolve().parent
out=[]
for line in (D/'input.tsv').read_text().splitlines():
 if line.startswith('#'):continue
 p=line.split();cid,profile,epoch=p[0],p[2],p[4];s,q,rho,tt,y,u=map(float,[p[5],p[6],p[7],p[10],p[9],p[11]])
 for tol in [1e-4,1e-5,1e-6,1e-7,1e-8]:
  v=VBMicrolensing.VBMicrolensing();v.Tol=1e-12;v.RelTol=tol;v.a1=u;v.a2=0.;v.SetLDprofile(v.LDlinear)
  start=time.monotonic()
  mu=v.BinaryMagDark(s,q,-tt,y,rho,1e-12) if u else v.BinaryMag(s,q,-tt,y,rho,1e-12)
  r=dict(case_id=cid,profile=profile,epoch=epoch,reltol=tol,mu=mu,seconds=time.monotonic()-start)
  out.append(r);print(json.dumps(r),flush=True)
  (D/'vbm.json').write_text(json.dumps(out,indent=2)+'\n')

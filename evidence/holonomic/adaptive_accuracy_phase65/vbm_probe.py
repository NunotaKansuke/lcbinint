import pandas as pd,VBMicrolensing,time,json
from pathlib import Path
p=Path(__file__).resolve().parent
r=pd.read_csv(p/'input.tsv',sep=r'\s+',comment='#',header=None)
out=[]
for a in r.itertuples(index=False,name=None):
 id,cid,profile,db,ep,s,q,rho,x,y,t,u,X,ref=a
 if id!=115 or db!=2 or ep not in (7,15):continue
 for tol in [1e-4,1e-6,1e-7,1e-8]:
  v=VBMicrolensing.VBMicrolensing();v.RelTol=tol;v.Tol=1e-12
  v.a1=u;v.SetLDprofile(v.LDlinear)
  start=time.perf_counter()
  z=v.BinaryMagDark(s,q,-t,y,rho,1e-12) if u else v.BinaryMag(s,q,-t,y,rho,1e-12)
  o=dict(case_id=id,profile=profile,epoch=ep,rtol=tol,mu=z,seconds=time.perf_counter()-start,saved_reference=ref)
  out.append(o);print(o,flush=True)
  (p/'vbm_probe.json').write_text(json.dumps(out,indent=2)+'\n')

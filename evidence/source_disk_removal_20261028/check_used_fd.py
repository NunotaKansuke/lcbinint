import ctypes as C, json, sys
from pathlib import Path
lib=C.CDLL(str(Path(sys.argv[1]).resolve()),mode=C.RTLD_LOCAL)
ptr=C.POINTER(C.c_double)
lib.create.argtypes=[C.c_double,C.c_double];lib.create.restype=C.c_void_p
lib.evaluate.argtypes=[C.c_void_p,ptr,C.c_double,C.c_double,C.c_int,ptr]
lib.reason.restype=C.c_char_p
lib.destroy.argtypes=[C.c_void_p]

import math
cases=[]
raw=json.loads(Path('evidence/source_disk_removal_20261028/random-before.json').read_text())
for name in ['random_20','random_61']:
 row=next(r for r in raw if r['case']==name);p=row['p'];x,y,r,q,a=p;cases.append((name,x+q/(1+q)*a,y,r,q,a))
rows=[]
for name,X,Y,r,q,s in cases:
 p=[X-q/(1+q)*s,Y,r,q,s]
 for u in [0.]:
  state=lib.create(u,1e-9)
  def call(x,tol,mode):
   out=(C.c_double*20)();lib.evaluate(state,(C.c_double*5)(*x),u,tol,mode,out);return list(out)
  refs=[]
  for j in range(5):
   h=1e-5 if j in [0,1,4] else (r*1e-4 if j==2 else q*1e-4)
   slopes=[];valid=True
   for step in [h,h/2]:
    vals=[]
    for k in [-2,-1,1,2]:
     x=p.copy();x[j]+=k*step
     v=call(x,1e-9,0);valid &= bool(v[2]);vals.append(v[0])
    slopes.append((vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*step))
   refs.append(dict(reference=slopes[1],uncertainty=abs(slopes[1]-slopes[0]),converged=valid,step=h/2))
  for tol in [1e-3,1e-4]:
   lib.destroy(state);state=lib.create(u,tol);result=call(p,tol,1)
   for j,ref in enumerate(refs):
    gate=1e-4+2e-4*abs(ref['reference']);delta=abs(result[10+j]-ref['reference'])
    row=dict(case=name,u=u,tol=tol,parameter=j,gradient=result[10+j],**ref,gate=gate,delta=delta,pass_gate=bool(result[2] and result[3] and ref['converged'] and delta+ref['uncertainty']<=gate))
    rows.append(row)
   print(name,u,tol,'passed',sum(r['pass_gate'] for r in rows[-5:]),flush=True)
  lib.destroy(state)
  Path(sys.argv[2]).write_text(json.dumps(rows,indent=2))

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
for t in [-1.7455113096375089e-6,0,1.7455113096375089e-6]:
 tau=t/10;alpha=.5;u0=9.846481110931334e-7
 cases.append(("stratified09_"+str(t),[tau*math.cos(alpha)-u0*math.sin(alpha),tau*math.sin(alpha)+u0*math.cos(alpha),1e-6,1e7,.95],.5))
rows=[]
for name,p,u in cases:
 for tol in [1e-3,1e-4,1e-7]:
  state=lib.create(u,tol)
  for mode in [0,1]:
   out=(C.c_double*20)();lib.evaluate(state,(C.c_double*5)(*p),u,tol,mode,out)
   row=dict(case=name,p=p,u=u,tol=tol,mode=mode,out=list(out),reason=lib.reason().decode());rows.append(row)
   print(name,tol,mode,list(out)[:7],flush=True)
   Path(sys.argv[2]).write_text(json.dumps(rows,indent=2))
  lib.destroy(state)

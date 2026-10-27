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
import random
rng=random.Random(20261028)
for i in range(64):
 q=10**rng.uniform(-4,4);sep=rng.uniform(.6,1.6);r=10**rng.uniform(-6,-.4)
 x=rng.uniform(-.15,.15);y=rng.choice([0.,rng.uniform(-.1,.1)])
 cases.append(("random_%02d"%i,[x,y,r,q,sep],rng.choice([0.,.5])))
rows=[]
for name,p,u in cases:
 for tol in [1e-3,1e-4]:
  state=lib.create(u,tol)
  for mode in [0,1]:
   out=(C.c_double*20)();lib.evaluate(state,(C.c_double*5)(*p),u,tol,mode,out)
   row=dict(case=name,p=p,u=u,tol=tol,mode=mode,out=list(out),reason=lib.reason().decode());rows.append(row)
   print(name,tol,mode,list(out)[:7],flush=True)
   Path(sys.argv[2]).write_text(json.dumps(rows,indent=2))
  lib.destroy(state)

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
for t in [-.275,-.25,-.225,-.2,-.075,-.025,0,.025]:
 alpha=-1.8890494698555882; tau=t/10;u0=-1e-5
 cases.append(("stratified02_"+str(t),[tau*math.cos(alpha)-u0*math.sin(alpha),tau*math.sin(alpha)+u0*math.cos(alpha),1e-6,100.,.95],.9))
for sign in [-1,1]:
 for rf in [.999,1,1.001]:
  cases.append(("edge_%d_%g"%(sign,rf),[-6.3359522138426355e-05,sign*1.9809194818738246e-09,.00031754130897469404*rf,704.16939274402819,.63244064975758063],.6584))
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

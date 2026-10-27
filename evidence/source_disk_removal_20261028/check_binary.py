import ctypes as C, json, sys
from pathlib import Path
lib=C.CDLL(str(Path(sys.argv[1]).resolve()),mode=C.RTLD_LOCAL)
ptr=C.POINTER(C.c_double)
lib.create.argtypes=[C.c_double,C.c_double];lib.create.restype=C.c_void_p
lib.evaluate.argtypes=[C.c_void_p,ptr,C.c_double,C.c_double,C.c_int,ptr]
lib.reason.restype=C.c_char_p
lib.destroy.argtypes=[C.c_void_p]
rows=[]
for name,X,Y,r,q,s in [('axis',2,0,.02,.1,1),('tiny',.5,.4,.0001,.1,1),('ordinary',.5,.4,.03,.1,1),('equal',.5,.4,.03,1,1),('planet',.1,.2,.03,.001,1.2),('contact',.2,.1,.2,.1,1)]:
 for u in [0,.5]:
  for tol in [1e-3,1e-4]:
   p=[X-q/(1+q)*s,Y,r,q,s];state=lib.create(u,tol)
   for mode in [0,1,2,3]:
    out=(C.c_double*20)();lib.evaluate(state,(C.c_double*5)(*p),u,tol,mode,out)
    row=dict(case=name,p=p,u=u,tol=tol,mode=mode,out=list(out),reason=lib.reason().decode());rows.append(row)
    print(name,u,tol,mode,list(out)[:7],flush=True)
   lib.destroy(state)
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2))

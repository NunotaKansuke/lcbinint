import ctypes as C,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent.parent;os.sched_setaffinity(0,{0});dl=C.CDLL('libdl.so.2');dl.dlmopen.argtypes=[C.c_long,C.c_char_p,C.c_int];dl.dlmopen.restype=C.c_void_p;libs={}
for key in['adopted','no-geometry']:
 p=D/f'{key}.so';h=dl.dlmopen(-1,os.fsencode(p),os.RTLD_NOW|os.RTLD_LOCAL);assert h;l=C.CDLL(str(p),handle=h);l.binary_create.argtypes=[C.POINTER(C.c_double)];l.binary_create.restype=C.c_void_p;l.binary_evaluate.argtypes=[C.c_void_p,C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)];l.binary_destroy.argtypes=[C.c_void_p];libs[key]=l
screen=C.CDLL(str(R/'evidence/binary_triple_broad_cost/binary-router.so'));screen.screen.argtypes=[C.POINTER(C.c_double),C.c_double,C.c_double,C.POINTER(C.c_double)];rows=[]
lines=(R/'evidence/easy_gradient_20261018/notebook-geometry.txt').read_text().splitlines()[1:]
for i,line in enumerate(lines):
 s,q,x,y,rho,mu=map(float,line.split());p=[x,y,rho,q,s];route=(C.c_double*32)();screen.screen((C.c_double*5)(*p),.5,1e-4,route)
 if route[0]!=7:continue
 p[0]+=q/(1+q)*s;out={}
 for key,l in libs.items():
  h=l.binary_create((C.c_double*5)(*p))
  if not h:out[key]=None;continue
  o=(C.c_double*16)();l.binary_evaluate(h,.5,1e-4,1,o);l.binary_destroy(h);out[key]=list(o)
 delta=max(abs(a-b)/(1e-4+2e-4*abs(a))for a,b in zip(out['adopted'][10:15],out['no-geometry'][10:15]))if all(out.values())else None
 rows.append(dict(index=i,p=p,limb=.5,rtol=1e-4,outputs=out,gradient_budget_delta=delta));(D/'ablation-notebook.json').write_text(json.dumps(rows,indent=2))
 print(i,delta,flush=True)

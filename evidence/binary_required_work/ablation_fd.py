import ctypes as C,json,math,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent.parent;os.sched_setaffinity(0,{0});target=D/'ablation-independent-fd.json';case=next(r for r in json.loads((D/'ablation-notebook.json').read_text())if r['index']==221);p=case['p'];l=C.CDLL(str(D/'adopted.so'));l.binary_create.argtypes=[C.POINTER(C.c_double)];l.binary_create.restype=C.c_void_p;l.binary_evaluate.argtypes=[C.c_void_p,C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)];l.binary_destroy.argtypes=[C.c_void_p];raw=[];rows=[];cache={}
def save():target.write_text(json.dumps(dict(case=case,raw=raw,rows=rows),indent=2))
for j in[1,2]:
 sweep=[]
 for h in[2e-5,1e-5,5e-6]:
  vals=[]
  for k in[-2,-1,1,2]:
   x=p.copy();x[j]+=h*k;key=tuple(x)
   if key not in cache:
    ptr=l.binary_create((C.c_double*5)(*x));assert ptr;o=(C.c_double*16)();l.binary_evaluate(ptr,.5,1e-10,0,o);l.binary_destroy(ptr);cache[key]=list(o);raw.append(dict(p=x,limb=.5,rtol=1e-10,mode='value-only',output=[v if math.isfinite(v)else None for v in o]));save()
   vals.append(cache[key])
  fd=sum(w*v[0]for w,v in zip([1,-8,8,-1],vals))/(12*h);err=sum(abs(w)*v[1]for w,v in zip([1,-8,8,-1],vals))/(12*h);ok=all(v[2]and v[9]and math.isfinite(v[1])for v in vals);sweep.append(dict(step=h,fd=fd,error=err if math.isfinite(err)else None,converged=ok))
 ref=sweep[-1]['fd'];budget=1e-4+2e-4*abs(ref);unc=sum(abs(sweep[i]['fd']-sweep[i-1]['fd'])for i in[1,2])+sum(s['error']if s['error']is not None else math.inf for s in sweep);resolved=all(s['converged']for s in sweep)and unc<budget/2
 rows.append(dict(parameter=j,reference=ref,uncertainty=unc if math.isfinite(unc)else None,budget=budget,resolved=resolved,sweep=sweep,ratios={mode:(abs(o[10+j]-ref)+unc)/budget if math.isfinite(unc)else None for mode,o in case['outputs'].items()}));save();print(rows[-1],flush=True)

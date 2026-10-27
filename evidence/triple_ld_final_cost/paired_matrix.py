import ctypes as C,json,os,time,sys,statistics as S,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent;os.sched_setaffinity(0,{0});paths={'before':R/'evidence/triple_coupled_optimization/final-verified.so','after':P/sys.argv[1]}
dl=C.CDLL('libdl.so.2');dl.dlmopen.argtypes=[C.c_long,C.c_char_p,C.c_int];dl.dlmopen.restype=C.c_void_p;libs={}
for k,path in paths.items():
 h=dl.dlmopen(-1,os.fsencode(path),os.RTLD_NOW|os.RTLD_LOCAL);assert h;lib=C.CDLL(str(path),handle=h);lib.create.restype=C.c_void_p;lib.destroy.argtypes=[C.c_void_p];lib.evaluate.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)];libs[k]=lib
configs=[('before_value','before',0),('after_gradient','after',2),('after_value','after',0),('before_gradient','before',2)];rows=[]
for ref in json.loads((R/'evidence/triple_speed_comparison_20261028/results.json').read_text())['rows']:
 p=(C.c_double*8)(*ref['source'],ref['rho'],.2,.08,1,-.4,.6);states={k:libs[l].create()for k,l,m in configs};times={k:[]for k,l,m in configs};outputs={}
 for i in range(20):
  order=configs[i%4:]+configs[:i%4]
  if i%8>=4:order=order[::-1]
  for k,l,mode in order:
   o=(C.c_double*24)();t=time.thread_time_ns();libs[l].evaluate(states[k],p,ref['limb'],ref['rtol'],mode,o);dt=(time.thread_time_ns()-t)/1e6
   assert o[2]and o[3]and(mode==0 or o[4]),(ref['source'],ref['rho'],ref['limb'],ref['rtol'],k,list(o))
   assert all(abs(o[0]-v)<=ref['rtol']*abs(v)for v in ref['references'].values()),(k,list(o),ref)
   if i>=4:times[k].append(dt)
   outputs[k]=list(o)
 for k,l,m in configs:libs[l].destroy(states[k])
 med={k:S.median(t)for k,t in times.items()};rows.append(dict(source=ref['source'],rho=ref['rho'],limb=ref['limb'],rtol=ref['rtol'],median_ms=med,raw_ms=times,outputs=outputs))
result=dict(libraries={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for k,p in paths.items()},protocol='CPU0, 4warm+16rotated/reversed thread CPU samples; matched before/after direct support+integral',rows=rows)
(P/sys.argv[2]).write_text(json.dumps(result,indent=2))
for u in [0.,.5]:
 for tol in [1e-3,1e-4]:
  rr=[r for r in rows if r['limb']==u and r['rtol']==tol]
  print(u,tol,'time reduction %',{mode:round(S.median(100*(1-r['median_ms']['after_'+mode]/r['median_ms']['before_'+mode])for r in rr),2)for mode in ['value','gradient']},flush=True)

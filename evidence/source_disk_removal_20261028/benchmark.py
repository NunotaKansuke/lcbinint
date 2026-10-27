import ctypes as C,json,os,time,statistics as S,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent.parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
dl=C.CDLL('libdl.so.2');dl.dlmopen.argtypes=[C.c_long,C.c_char_p,C.c_int];dl.dlmopen.restype=C.c_void_p
ptr=C.POINTER(C.c_double)
def load(path,binary):
 h=dl.dlmopen(-1,os.fsencode(path),os.RTLD_NOW|os.RTLD_LOCAL);assert h
 l=C.CDLL(str(path),handle=h);l.create.restype=C.c_void_p;l.create.argtypes=[C.c_double,C.c_double] if binary else [];l.destroy.argtypes=[C.c_void_p];l.evaluate.argtypes=[C.c_void_p,ptr,C.c_double,C.c_double,C.c_int,ptr];return l
rows=[];hashes={}
for binary in [True,False]:
 paths={'before':D/'binary-before.so' if binary else R/'evidence/triple_ld_final_cost/final.so','after':D/('binary-final.so' if binary else 'triple-final.so')}
 libs={k:load(p,binary) for k,p in paths.items()};hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths.values()})
 if binary:
  configs=[dict(case=r['case'],p=r['p'],u=r['u'],tol=r['tol'],mode=r['mode']) for r in json.loads((D/'random-before.json').read_text()) if r['out'][5]==7]
  configs += [dict(case='contact',p=[.2-.1/1.1,.1,.2,.1,1],u=u,tol=t,mode=m)for u in [0.,.5]for t in [1e-3,1e-4]for m in [0,1]]
 else:
  refs=json.loads((R/'evidence/triple_speed_comparison_ld_20261028/results.json').read_text())['rows']
  configs=[dict(case='natural',p=[*r['source'],r['rho'],.2,.08,1,-.4,.6],u=r['limb'],tol=r['rtol'],mode=m)for r in refs for m in [0,2]]
  configs += [dict(case='strict_smooth',p=[.5,.4,.03,.2,.08,1,-.4,.6],u=u,tol=t,mode=1)for u in [0.,.5]for t in [1e-3,1e-4]]
 for c in configs:
  states={k:l.create(c['u'],c['tol'])if binary else l.create()for k,l in libs.items()};times={k:[]for k in libs};out={}
  for i in range(20):
   for k in (['before','after']if i%2 else ['after','before']):
    o=(C.c_double*(20 if binary else 24))();p=(C.c_double*len(c['p']))(*c['p']);start=time.thread_time_ns();libs[k].evaluate(states[k],p,c['u'],c['tol'],c['mode'],o);ms=(time.thread_time_ns()-start)/1e6
    if i>=4:times[k].append(ms)
    out[k]=list(o)
  for k,l in libs.items():l.destroy(states[k])
  rows.append(dict(engine='binary'if binary else 'triple',**c,median_ms={k:S.median(x)for k,x in times.items()},raw_ms=times,out=out))
  (D/'timings.json').write_text(json.dumps(dict(protocol='One pinned CPU; isolated ELF namespaces; 4 warmup +16 alternating thread CPU timings; setup excluded, integral and support included; no concurrent builds or checks',hashes=hashes,rows=rows),indent=2))
 print('DONE',binary,len(rows),flush=True)

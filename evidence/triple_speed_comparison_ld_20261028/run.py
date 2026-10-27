import ctypes as C, json, os, time, statistics as S, hashlib, platform, subprocess
from pathlib import Path
import VBMicrolensing
import importlib.metadata
P=Path(__file__).resolve().parent; R=P.parent.parent; E=R/'evidence/triple_coupled_optimization'
os.sched_setaffinity(0,{0})
class Out(C.Structure):
 _fields_=[('mu',C.c_double),('error',C.c_double),('seconds',C.c_double),('samples',C.c_int),('annuli',C.c_int),('converged',C.c_int)]
vpath=R/'evidence/triple_natural_integral_benchmark/kernel.so'; v=C.CDLL(str(vpath))
v.vbm_create.argtypes=[C.c_int,C.POINTER(C.c_double)];v.vbm_create.restype=C.c_void_p
v.vbm_point.argtypes=[C.c_void_p]+[C.c_double]*5+[C.c_int,C.POINTER(Out)];v.vbm_destroy.argtypes=[C.c_void_p]
dl=C.CDLL('libdl.so.2');dl.dlmopen.argtypes=[C.c_long,C.c_char_p,C.c_int];dl.dlmopen.restype=C.c_void_p
apath=R/'evidence/triple_ld_final_cost/final.so'; handle=dl.dlmopen(-1,os.fsencode(apath),os.RTLD_NOW|os.RTLD_LOCAL);assert handle
a=C.CDLL(str(apath),handle=handle);a.create.restype=C.c_void_p;a.destroy.argtypes=[C.c_void_p];a.evaluate.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.c_double,C.c_double,C.c_int,C.POINTER(C.c_double)]
r=C.CDLL(str(P/'router.so'));r.create.argtypes=[C.POINTER(C.c_double),C.c_double,C.c_double];r.create.restype=C.c_void_p;r.evaluate.argtypes=[C.c_void_p]+[C.c_double]*3+[C.POINTER(C.c_double)];r.destroy.argtypes=[C.c_void_p];r.reason.restype=C.c_char_p
geometry=(C.c_double*9)(0,0,.72,1,0,.2,-.4,.6,.08)
engines=['value','gradient','SinglePoly','MultiPoly','NoPoly']; ids={'SinglePoly':0,'MultiPoly':1,'NoPoly':2}
old=json.loads((R/'evidence/triple_natural_integral_benchmark/results.json').read_text())
report=dict(status='RUNNING',protocol='CPU0 thread_time_ns; 4 warmups and 16 rotating/reversed timed calls. Persistent separate state per condition and engine; geometry setup excluded; lcbinint includes support preparation and integral; VBM direct MultiMag/MultiMagDark Tol=0 a2=0. Fresh normal-router screen before timing. No light-curve, hex or source-disc timing. Tight references are saved prior independent VBM runs, diagnostic only.',rows=[],excluded=[],provenance=dict(vbm_version=importlib.metadata.version('VBMicrolensing'),host=platform.node(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [apath,vpath,P/'router.so',P/'run.py',Path(VBMicrolensing.__file__).parent/'VBMicrolensing.so']}))
def save(): (P/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
for rho in [.075,.15,.3]:
 for x,y in [(.07,.15),(.1,.15),(.13,.15),(.1,.18)]:
  for u in [0.,.5]:
   ref=next(z for z in old['references'] if z['source']==[x,y] and z['rho']==rho and z['limb']==u)
   refs={k:z[-1]['mu'] for k,z in ref['results'].items() if k!='algebraic' and z[-1]['mu'] is not None}
   for tol in [1e-3,1e-4]:
    sh=r.create(geometry,u,tol);o=(C.c_double*32)();r.evaluate(sh,x,y,rho,o);route=dict(method=int(o[0]),reason=r.reason().decode(),mu=o[1]);r.destroy(sh)
    if route['method'] not in [2,3,4]: report['excluded'].append(dict(x=x,y=y,rho=rho,u=u,tol=tol,route=route));save();continue
    states={k:a.create() if k in ['value','gradient'] else v.vbm_create(ids[k],geometry) for k in engines}
    p=(C.c_double*8)(x,y,rho,.2,.08,1,-.4,.6); samples={k:[] for k in engines};first={}
    for i in range(20):
     order=engines[i%5:]+engines[:i%5]
     if i%2:order=order[::-1]
     for k in order:
      start=time.thread_time_ns()
      if k in ['value','gradient']:
       out=(C.c_double*24)();a.evaluate(states[k],p,u,tol,2 if k=='gradient' else 0,out)
       elapsed=(time.thread_time_ns()-start)/1e6
       z=dict(cpu_ms=elapsed,mu=out[0],error=out[1],converged=bool(out[2]),support=bool(out[3]),gradient_valid=bool(out[4]),gradient=list(out[12:20]) if k=='gradient' else [],annuli=None)
      else:
       out=Out();v.vbm_point(states[k],x,y,rho,u,tol,int(u!=0),C.byref(out));elapsed=(time.thread_time_ns()-start)/1e6
       z=dict(cpu_ms=elapsed,mu=out.mu,error=out.error if out.error==out.error else None,annuli=out.annuli,samples=out.samples)
      if i==0:first[k]=z
      if i>=4:samples[k].append(z)
    for k in engines:
     (a.destroy if k in ['value','gradient'] else v.vbm_destroy)(states[k])
    summary={}
    for k,vals in samples.items():
     z=vals[-1];summary[k]=dict(median_ms=S.median(t['cpu_ms'] for t in vals),mu=z['mu'],annuli=sorted(set(t['annuli'] for t in vals),key=str),relative_differences={j:abs(z['mu']-q)/abs(q) for j,q in refs.items()},converged=z.get('converged'),gradient_valid=z.get('gradient_valid'))
    row=dict(source=[x,y],rho=rho,limb=u,rtol=tol,route=route,first=first,samples=samples,summary=summary,references=refs);report['rows'].append(row);save()
    print(len(report['rows']),[x,y],rho,u,tol,{k:round(z['median_ms'],3) for k,z in summary.items()},flush=True)
report['status']='COMPLETE';save()

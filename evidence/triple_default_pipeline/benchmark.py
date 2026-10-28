"""Public triple lightcurve APIs, process-isolated engine/condition timings."""
import os,sys,json,time,math,statistics,subprocess,hashlib,platform
from pathlib import Path
P=Path(__file__).resolve().parent; R=P.parent.parent
ENGINES=['value','gradient','SinglePoly','MultiPoly','NoPoly']
def worker(c):
 os.sched_setaffinity(0,{0})
 import numpy as np
 sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable' not in str(f)]
 sys.path.insert(0,str(R/'build-triple-native'))
 import lcbinint as lc
 import VBMicrolensing as V
 assert Path(lc.__file__).resolve().is_relative_to(R/'build-triple-native')
 assert Path(lc._native.__file__).resolve().is_relative_to(R/'build-triple-native')
 params=dict(t0=0.,u0=-c['y'],tE=1.,alpha=0.,s=1.,q=.2/.72,q2=.08/.72,sep2=math.hypot(.4,.6),ang=math.atan2(.6,-.4),rho=c['rho'])
 xs=np.linspace(c['xmin'],c['xmax'],c['n']);times=(.2/.92-xs).tolist()
 engine=c['engine'];start=time.thread_time_ns()
 if engine in ('value','gradient'):
  obj=lc.LightCurve(lens='triple',param_type='vbm',options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']),ld_c=c['limb'])
  def call(p):return obj.magnification(times,p,with_gradient=engine=='gradient')
 else:
  obj=V.VBMicrolensing();obj.SetMethod(getattr(obj.Method,{'SinglePoly':'Singlepoly','MultiPoly':'Multipoly','NoPoly':'Nopoly'}[engine]));obj.Tol=0.;obj.RelTol=c['rtol'];obj.a1=c['limb'];obj.a2=0.;obj.SetLDprofile(obj.LDlinear)
  def call(p):
   v=[math.log(p['s']),math.log(p['q']),p['u0'],p['alpha'],math.log(p['rho']),math.log(p['tE']),p['t0'],math.log(p['sep2']),math.log(p['q2']),p['ang']]
   return obj.TripleLightCurve(v,times)
 constructor=(time.thread_time_ns()-start)/1e6
 samples=[];wall=[];first=None;last=None
 for i in range(7):
  a=time.thread_time_ns();b=time.perf_counter_ns();out=call(params)
  dt=(time.thread_time_ns()-a)/1e6;dw=(time.perf_counter_ns()-b)/1e6
  if i==0:first=dt
  if i>=2:samples.append(dt);wall.append(dw)
  last=out
 def unpack(out):
  if engine in ('value','gradient'):
   d=out.diagnostics or {}
   return dict(mu=np.asarray(out.magnifications).tolist(),status=np.asarray(out.status).tolist(),routes=np.asarray(d.get('finite_source_method',[])).tolist(),route_names=list(d.get('finite_source_method_name',[])),gradient_valid=np.asarray(out.gradient_valid).tolist() if engine=='gradient' else None,gradients=np.asarray(out.gradients).tolist() if engine=='gradient' else None)
  return dict(mu=list(out[0]),source_x=list(out[1]),source_y=list(out[2]),last_annuli=getattr(obj,"nannuli",None),last_samples=getattr(obj,"NPS",None))
 output=unpack(last)
 if engine in ('value','gradient'):
  assert obj.options.finite_source_backend=='algebraic_boundary'
  diagnostic=obj.info(times,params,with_gradient=engine=='gradient')
  output['routes']=list(map(int,diagnostic.diagnostics['finite_source_method']))
  output['diagnostic_value_max_diff']=float(np.max(np.abs(np.asarray(output['mu'])-diagnostic.magnifications)))
 changed=dict(params,s=1.0001)
 a=time.thread_time_ns();co=call(changed);changed_ms=(time.thread_time_ns()-a)/1e6
 return dict(condition=c,constructor_ms=constructor,cold_api_ms=first,cold_total_ms=constructor+first,warm_cpu_ms=samples,warm_wall_ms=wall,warm_median_ms=statistics.median(samples),output=output,changed_geometry_ms=changed_ms,changed_output=unpack(co),package=lc.__file__,extension=lc._native.__file__)
if '--worker' in sys.argv:
 print(json.dumps(worker(json.loads(sys.argv[-1])),allow_nan=False))
else:
 report=dict(status='RUNNING',protocol='Public ordinary LightCurve.magnification (no backend flag) and VBM TripleLightCurve. Physical trajectory x=-.5..1.1,y=.15,rho=.03; N=129. Absolute tolerance0, relative1e-3/1e-4, linear LD0/.5. Fresh subprocess per engine and condition, CPU0 thread CPU milliseconds. Constructor and first call are cold; second warmup, final five samples median. Changed s=1.0001 separately. Imports/IO excluded; actual caustic construction and scalar routing included. Serial rotated engine order, no builds/tests concurrently.',rows=[],failures=[])
 def save():(P/'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 for limb in (0.,.5):
  for tol in (1e-3,1e-4):
   cfg=dict(name='mixed',n=129,y=.15,rho=.03,xmin=-.5,xmax=1.1,limb=limb,rtol=tol);row=dict(condition=cfg,engines={})
   shift=len(report['rows']);order=ENGINES[shift:]+ENGINES[:shift]
   for k in order:
    cp=subprocess.run([sys.executable,str(P/'benchmark.py'),'--worker',json.dumps(dict(cfg,engine=k))],capture_output=True,text=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),timeout=180)
    if cp.returncode:report['failures'].append(dict(condition=cfg,engine=k,returncode=cp.returncode,stderr=cp.stderr[-4000:]));save();continue
    row['engines'][k]=json.loads(cp.stdout)
   report['rows'].append(row);save();print(cfg,{k:round(v['warm_median_ms'],3)for k,v in row['engines'].items()},flush=True)
 report['status']='COMPLETE';save()

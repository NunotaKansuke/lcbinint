"""Independent persistent engine processes; serial round-robin timing."""
import os,sys,json,time,math,statistics,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent
ENGINES=['value','gradient','SinglePoly','MultiPoly','NoPoly']
if '--worker'in sys.argv:
 os.sched_setaffinity(0,{0});c=json.loads(sys.argv[-1])
 import numpy as np
 sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)];sys.path.insert(0,str(R/'build-triple-native'))
 import lcbinint as lc
 import VBMicrolensing as V
 assert Path(lc._native.__file__).resolve().is_relative_to(R/'build-triple-native')
 params=dict(t0=0.,u0=-c['y'],tE=1.,alpha=0.,s=1.,q=.2/.72,q2=.08/.72,sep2=math.hypot(.4,.6),ang=math.atan2(.6,-.4),rho=c['rho'])
 times=(.2/.92-np.linspace(c['xmin'],c['xmax'],c['n'])).tolist();engine=c['engine'];a=time.thread_time_ns()
 if engine in ['value','gradient']:
  model=lc.LightCurve(lens='triple',param_type='vbm',ld_c=c['limb'],options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']))
  assert model.options.finite_source_backend=='algebraic_boundary'
  def call(p):return model.magnification(times,p,with_gradient=engine=='gradient')
 else:
  model=V.VBMicrolensing();model.SetMethod(getattr(model.Method,dict(SinglePoly='Singlepoly',MultiPoly='Multipoly',NoPoly='Nopoly')[engine]));model.Tol=0.;model.RelTol=c['rtol'];model.a1=c['limb'];model.a2=0.;model.SetLDprofile(model.LDlinear)
  def call(p):return model.TripleLightCurve([math.log(p['s']),math.log(p['q']),p['u0'],p['alpha'],math.log(p['rho']),math.log(p['tE']),p['t0'],math.log(p['sep2']),math.log(p['q2']),p['ang']],times)
 constructor=(time.thread_time_ns()-a)/1e6
 print(json.dumps(dict(ready=True,constructor_ms=constructor)),flush=True)
 for line in sys.stdin:
  command=line.strip()
  if command=='stop':break
  p=dict(params,s=1.0001)if command=='changed'else params
  a=time.thread_time_ns();b=time.perf_counter_ns();out=call(p);cpu=(time.thread_time_ns()-a)/1e6;wall=(time.perf_counter_ns()-b)/1e6
  if engine in ['value','gradient']:
   z=dict(mu=out.magnifications.tolist(),status=out.status.tolist(),integral_mask=out.support_required.tolist(),gradient_valid=out.gradient_valid.tolist()if engine=='gradient'else None)
  else:z=dict(mu=list(out[0]),source_x=list(out[1]),source_y=list(out[2]))
  if command=='diagnostic'and engine in ['value','gradient']:
   d=model.info(times,p,with_gradient=engine=='gradient')
   z['routes']=d.diagnostics['finite_source_method'].tolist()
  print(json.dumps(dict(cpu_ms=cpu,wall_ms=wall,output=z),allow_nan=False),flush=True)
else:
 report=dict(status='RUNNING',protocol='Five independent persistent subprocesses, all CPU0, serial rotated/reversed round-robin requests. No backend flag in ordinary lcbinint magnification(). VBM TripleLightCurve SinglePoly/MultiPoly/NoPoly. Each constructor and first API call are cold. Two warmups and eight timed calls per engine; cold and changed-geometry times separate. Thread CPU milliseconds inside each worker exclude IPC/imports/serialization. Calls include actual lazy caustic-tree construction, routing, value/tangent and API conversion. No builds or validation concurrently. CPU frequency and external load uncontrolled. VBM Tol=0/RelTol=request, so its current corrquad<Tol point gate cannot pass; this is the requested relative-only contract, not VBM default Tol.',rows=[],failures=[])
 wide='--wide'in sys.argv
 output_name='interleaved-wide.json'if wide else'interleaved.json'
 def save():(P/output_name).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 for limb in [0.,.5]:
  for tol in [1e-3,1e-4]:
   c=dict(n=129,y=.15,rho=.03,xmin=-2. if wide else -.5,xmax=2. if wide else 1.1,limb=limb,rtol=tol);row=dict(condition=c,engines={});workers={};logs={}
   try:
    for e in ENGINES:
     logs[e]=open(P/f'interleaved-{wide}-{limb}-{tol}-{e}.log','w')
     w=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(dict(c,engine=e))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs[e],text=True,bufsize=1,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
     workers[e]=w;ready=json.loads(w.stdout.readline());row['engines'][e]=dict(constructor_ms=ready['constructor_ms'],samples=[])
    def ask(e,command):
     w=workers[e];w.stdin.write(command+'\n');w.stdin.flush();line=w.stdout.readline()
     if not line:raise RuntimeError(f'{e} worker exited {w.poll()}')
     return json.loads(line)
    for i in range(10):
     order=ENGINES[i%5:]+ENGINES[:i%5]
     if i%2:order=order[::-1]
     for e in order:
      z=ask(e,'run');d=row['engines'][e]
      if i==0:d['first']=z
      if i>=2:d['samples'].append(z)
    for e in ENGINES:
     d=row['engines'][e];d['warm_median_ms']=statistics.median(z['cpu_ms']for z in d['samples']);d['cold_total_ms']=d['constructor_ms']+d['first']['cpu_ms'];d['changed']=ask(e,'changed');d['diagnostic']=ask(e,'diagnostic')
    report['rows'].append(row);save();print(limb,tol,{e:round(z['warm_median_ms'],3)for e,z in row['engines'].items()},flush=True)
   except Exception as ex:
    report['failures'].append(dict(condition=c,error=str(ex)));save();raise
   finally:
    for w in workers.values():
     if w.poll()is None:
      w.stdin.write('stop\n');w.stdin.flush();w.stdin.close();w.wait(timeout=10)
    for f in logs.values():f.close()
 report['status']='COMPLETE';save()

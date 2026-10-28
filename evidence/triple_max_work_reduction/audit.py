"""Bounded, serial, route-aware triple public API before/after optimization audit."""
import os,sys,time,json,math,statistics,subprocess,select,resource
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent
ENGINES=['before_value','value','before_gradient','gradient','point_base','SinglePoly','MultiPoly','NoPoly']
PYTHON=sys.executable

def native():
 sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)]
 sys.path.insert(0,str(R/'build-triple-native'))
 if os.environ.get('AUDIT_BASELINE'):
  import importlib.util
  spec=importlib.util.spec_from_file_location('lcbinint',P/'before-package'/'__init__.py',submodule_search_locations=[str(P/'before-package')]);lc=importlib.util.module_from_spec(spec);sys.modules['lcbinint']=lc;spec.loader.exec_module(lc)
 else:
  import lcbinint as lc
 assert Path(lc._native.__file__).resolve().is_relative_to(P/'before-package' if os.environ.get('AUDIT_BASELINE') else R/'build-triple-native')
 return lc

def parameters(c):
 m=c['masses'];b=complex(*c['b'])
 return dict(t0=0.,u0=-c['y'],tE=1.,alpha=0.,s=c['s'],q=m[1]/m[0],q2=m[2]/m[0],sep2=abs(b),ang=math.atan2(b.imag,b.real),rho=c['rho'])
def vparams(p):
 return [math.log(p['s']),math.log(p['q']),p['u0'],p['alpha'],math.log(p['rho']),math.log(p['tE']),p['t0'],math.log(p['sep2']),math.log(p['q2']),p['ang']]
def clean(v):
 if hasattr(v,'tolist'):return clean(v.tolist())
 if isinstance(v,list):return [clean(x)for x in v]
 if isinstance(v,dict):return {k:clean(x)for k,x in v.items()}
 if isinstance(v,float)and not math.isfinite(v):return None
 return v

def setup(c,engine):
 import numpy as np
 lc=native();import VBMicrolensing as V
 p=parameters(c);xs=np.linspace(c['xmin'],c['xmax'],c['n']);times=(p['q']/(1+p['q'])*p['s']-xs).tolist()
 start=time.thread_time_ns()
 if engine in ['value','gradient','point_base']:
  model=lc.LightCurve(lens='triple',param_type='vbm',ld_c=c['limb'],options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol'],**({'mode':0}if engine=='point_base'else{})))
  assert model.options.finite_source_backend=='algebraic_boundary'
  def call(q):return model.magnification(times,q,with_gradient=engine=='gradient')
 else:
  model=V.VBMicrolensing();model.SetMethod(getattr(model.Method,dict(SinglePoly='Singlepoly',MultiPoly='Multipoly',NoPoly='Nopoly')[engine]));model.Tol=0.;model.RelTol=c['rtol'];model.a1=c['limb'];model.a2=0.;model.SetLDprofile(model.LDlinear)
  def call(q):return model.TripleLightCurve(vparams(q),times)
 constructor=(time.thread_time_ns()-start)/1e6
 return model,p,times,call,constructor

def packed(out,engine):
 if engine in ['value','gradient','point_base']:
  return dict(mu=out.magnifications.tolist(),status=out.status.tolist(),integral_mask=out.support_required.tolist(),gradient_valid=out.gradient_valid.tolist()if engine=='gradient'else None)
 return dict(mu=list(out[0]),source_x=list(out[1]),source_y=list(out[2]))

if '--worker'in sys.argv:
 os.sched_setaffinity(0,{0});resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 c=json.loads(sys.argv[-1]);engine=c['engine']
 if engine.startswith('before_'):os.environ['AUDIT_BASELINE']='1';engine=engine.removeprefix('before_')
 model,p,times,call,constructor=setup(c,engine)
 print(json.dumps(dict(ready=True,constructor_ms=constructor)),flush=True)
 for line in sys.stdin:
  command=line.strip()
  if command=='stop':break
  q=dict(p,s=p['s']*(1.0002 if command=='changed2'else 1.0001))if command.startswith('changed')else p
  start=time.thread_time_ns();wall=time.perf_counter_ns();out=call(q);cpu=(time.thread_time_ns()-start)/1e6;elapsed=(time.perf_counter_ns()-wall)/1e6
  z=packed(out,engine)
  if command=='diagnostic'and engine in ['value','gradient','point_base']:
   d=model.info(times,q,with_gradient=engine=='gradient')
   z['diagnostics']=clean({k:v for k,v in d.diagnostics.items()if any(s in k for s in ['finite_source','root_','image_count','support'])})
   z['routes']=d.diagnostics['finite_source_method'].tolist()
   if engine=='gradient':z['gradients']=clean(d.gradients)
  print(json.dumps(dict(cpu_ms=cpu,wall_ms=elapsed,output=z),allow_nan=False),flush=True)
 sys.exit()

if '--profile'in sys.argv:
 c=json.loads(sys.argv[-1]);engine=c.pop('engine');model,p,times,call,_=setup(c,engine)
 # All inputs/loops recorded. Import is visible to perf but timed loop marker
 # is used to choose a sufficiently long native-dominated sample.
 for _ in range(3):call(p)
 a=time.thread_time();calls=0;checksum=0.
 while time.thread_time()-a<3.:
  q=dict(p,s=p['s']*(1.+(1+calls%2)*1e-4))if c.get('change')else p
  out=call(q);checksum+=float(out.magnifications[0]);calls+=1
 print(json.dumps(dict(calls=calls,cpu_seconds=time.thread_time()-a,checksum=checksum)),flush=True);sys.exit()

if '--cases'in sys.argv:
 import numpy as np
 lc=native();base=dict(s=1.,b=[-.4,.6],masses=[.72,.2,.08],n=65,rho=.03,y=.15,xmin=-.5,xmax=1.1)
 cases=[]
 def add(name,**kw):cases.append(dict(base,name=name,**kw))
 # Plotting samples are only for independent descriptive geometry labels;
 # numerical routing always uses the continuous native caustic enclosures.
 c=lc.LightCurve(lens='triple',param_type='vbm').caustics(parameters(base));offset=.2/.92
 points=np.column_stack((offset-np.concatenate(c.x),-np.concatenate(c.y)))
 peak=points[np.argmax(points[:,1])];ymax=float(peak[1]);xpeak=float(peak[0]);np.save(P/'baseline-caustic-samples.npy',points)
 add('far',y=ymax+2.,rho=.005,xmin=-3.,xmax=3.)
 add('miss',y=ymax+.38,xmin=float(points[:,0].min())-.2,xmax=float(points[:,0].max())+.2)
 add('near_mixed',y=.3,rho=.01,xmin=.3,xmax=.65)
 add('grazing',y=ymax+1.001*.03,xmin=xpeak-.15,xmax=xpeak+.15)
 add('crossing')
 add('large',rho=.15,xmin=-.4,xmax=.8)
 add('equal',masses=[1/3,1/3,1/3],b=[-.5,.6],xmin=-.8,xmax=1.5)
 add('close',s=.4,b=[-.3,.2],masses=[.6,.25,.15],rho=.015,y=.03,xmin=-.6,xmax=.6)
 add('wide',s=2.,b=[-1.,.7],masses=[.7,.2,.1],y=.1,xmin=-2.5,xmax=2.5)
 add('planetary',s=1.2,b=[.5,.8],masses=[1.,.01,.002],rho=.003,y=.02,xmin=-.3,xmax=1.5)
 add('resonant',s=.9,b=[.8,.2],masses=[.45,.35,.2],y=.12,xmin=-.8,xmax=1.5)
 add('secondary_dominant',s=1.1,b=[-.7,.8],masses=[.15,.6,.25],xmin=-1.,xmax=2.)
 for c in cases:
  if c['name']in ['far','miss','near_mixed','grazing','crossing','large']:
   xs=np.linspace(c['xmin'],c['xmax'],c['n']);d=np.sqrt(((np.column_stack((xs,np.full(len(xs),c['y'])))[:,None,:]-points[None,:,:])**2).sum(2)).min(1)
   c['sampled_min_caustic_distance']=float(d.min());c['sampled_edge_intersections']=int(np.count_nonzero(d<c['rho']))
 (P/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');print([(c['name'],c.get('sampled_min_caustic_distance'))for c in cases]);sys.exit()

cases=json.loads((P/'cases.json').read_text());rows=P/'rows';rows.mkdir(exist_ok=True)
report=dict(status='RUNNING',protocol='Eight independent persistent engine processes, CPU0, one thread, serial rotated/reversed requests. 65 uniformly spaced epochs. Constructor+first API call cold; two warmups, eight measured warm calls; four geometry-change calls alternating s*(1.0001/1.0002). Thread CPU ms inside worker exclude IPC/imports/serialization. Ordinary default native LightCurve, no backend flag. VBM TripleLightCurve SinglePoly/MultiPoly/NoPoly, Tol=0/RelTol=request, linear LD; Tol=0 disables its corrquad<Tol point shortcut, this is the requested relative-only contract, not VBM defaults. point_base is an intentionally point-only native mode=0 ablation, not an equivalent finite-source competitor or stage timer. No builds, references or profiling concurrent with timing. CPU frequency/external load uncontrolled.',rows=[],failures=[])
def save():(P/'status.json').write_text(json.dumps(report,indent=2)+'\n')
for case in cases:
 for limb in [0.,.5]:
  for tol in [1e-3,1e-4]:
   c=dict(case,limb=limb,rtol=tol);key=f"{case['name']}-u{limb}-r{tol}";path=rows/(key+'.json')
   if path.exists():report['rows'].append(key);continue
   row=dict(condition=c,engines={});workers={};logs={};start=time.monotonic()
   try:
    for e in ENGINES:
     logs[e]=open(rows/(key+'-'+e+'.log'),'w')
     w=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--worker',json.dumps(dict(c,engine=e))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs[e],text=True,bufsize=1,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
     workers[e]=w
     if not select.select([w.stdout],[],[],60)[0]:raise TimeoutError('worker import '+e)
     line=w.stdout.readline()
     if not line:raise RuntimeError('worker startup '+e)
     ready=json.loads(line);row['engines'][e]=dict(constructor_ms=ready['constructor_ms'],samples=[],changed_samples=[])
    def ask(e,command):
     w=workers[e];w.stdin.write(command+'\n');w.stdin.flush()
     if not select.select([w.stdout],[],[],60)[0]:raise TimeoutError(e+' '+command)
     line=w.stdout.readline()
     if not line:raise RuntimeError(e+' exited '+str(w.poll()))
     return json.loads(line)
    for i in range(10):
     order=ENGINES[i%len(ENGINES):]+ENGINES[:i%len(ENGINES)]
     for e in (order[::-1]if i%2 else order):
      z=ask(e,'run');d=row['engines'][e]
      if i==0:d['first']=z
      if i>=2:d['samples'].append(z)
    for e in ENGINES:row['engines'][e]['diagnostic']=ask(e,'diagnostic')
    for i in range(4):
     for e in (ENGINES[::-1]if i%2 else ENGINES):row['engines'][e]['changed_samples'].append(ask(e,'changed2'if i%2 else'changed'))
    for e,d in row['engines'].items():
     d['warm_median_ms']=statistics.median(z['cpu_ms']for z in d['samples']);d['changed_median_ms']=statistics.median(z['cpu_ms']for z in d['changed_samples']);d['cold_total_ms']=d['constructor_ms']+d['first']['cpu_ms']
    row['wall_seconds']=time.monotonic()-start;path.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');report['rows'].append(key);save()
    print(key,{e:round(d['warm_median_ms'],3)for e,d in row['engines'].items()},flush=True)
   except Exception as exc:
    row['failure']=str(exc);path.with_suffix('.failed.json').write_text(json.dumps(row,indent=2)+'\n');report['failures'].append(dict(key=key,error=str(exc)));save();print('FAILED',key,str(exc),flush=True)
   finally:
    for w in workers.values():
     if w.poll()is None:
      try:w.stdin.write('stop\n');w.stdin.flush();w.stdin.close();w.wait(timeout=5)
      except Exception:w.kill();w.wait()
    for f in logs.values():f.close()
report['status']='COMPLETE_WITH_FAILURES'if report['failures']else'COMPLETE';save()

import os,sys,json,time,statistics,subprocess,resource,math,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;Q=P.parent/'triple_stress_expanded_20261028'
if '--worker'in sys.argv:
 os.sched_setaffinity(0,{0});resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(30,32));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
 i=int(sys.argv[2]);engine=sys.argv[3];c=json.loads((P/'timing-cases.json').read_text())[i];p=c['params'];meta=dict(case=i,engine=engine,rtol=c['rtol'])
 if engine.startswith('native'):
  sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)];build=P/'adopted-package'if 'before'in engine else P/'ready-package';sys.path.insert(0,str(build));import lcbinint as lc
  meta['sha256']=hashlib.sha256(Path(lc._native.__file__).read_bytes()).hexdigest();m=lc.LightCurve(lens='triple',param_type='vbm',ld_c=c['limb'],options=lc.Options(finite_source_tol=0.,finite_source_reltol=c['rtol']))
  def evaluate():
   o=m.magnification(c['epochs'],p,with_gradient='gradient'in engine)
   return dict(mu=o.magnifications.tolist(),status=o.status.tolist(),gradient_valid=o.gradient_valid.tolist()if 'gradient'in engine else None)
 else:
  import VBMicrolensing as V
  v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,dict(SinglePoly='Singlepoly',MultiPoly='Multipoly',NoPoly='Nopoly')[engine]));v.Tol=0.;v.RelTol=c['rtol'];v.a1=c['limb'];v.a2=0.;v.SetLDprofile(v.LDlinear)
  vp=[math.log(p[k])if k in ['s','q','rho','tE','sep2','q2']else p[k]for k in ['s','q','u0','alpha','rho','tE','t0','sep2','q2','ang']]
  def evaluate():return dict(mu=list(v.TripleLightCurve(vp,c['epochs'])[0]))
 try:
  a=time.thread_time();out=evaluate();meta['first']=time.thread_time()-a
  for _ in range(2):evaluate()
  samples=[]
  for _ in range(9):
   a=time.thread_time();out=evaluate();samples.append(time.thread_time()-a)
  meta.update(out,samples=samples,warm=statistics.median(samples))
 except Exception as exc:meta['error']=repr(exc)
 print(json.dumps(meta),flush=True);sys.exit()
D=P/'ready-controlled-timing';D.mkdir(exist_ok=True);rows=[]
engines=['native-before-value','native-after-value','native-before-gradient','native-after-gradient']
for k,i in enumerate([21,504,514,940,972,1508,1777]):
 for engine in (engines if k%2 else list(reversed(engines))):
  f=D/f'{i:04d}-{engine}.json'
  if f.exists():z=json.loads(f.read_text())
  else:
   try:
    r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(i),engine],capture_output=True,text=True,timeout=75,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'));z=json.loads(r.stdout)if r.returncode==0 else dict(case=i,engine=engine,returncode=r.returncode,stderr=r.stderr)
   except Exception as exc:z=dict(case=i,engine=engine,error=repr(exc))
   f.write_text(json.dumps(z,indent=2))
  rows.append(z);print(i,engine,z.get('warm',z.get('error',z.get('returncode'))),flush=True)
(P/'ready-controlled-timing.json').write_text(json.dumps(rows,indent=2));(P/'ready-timing-complete.json').write_text(json.dumps(dict(rows=len(rows),finished=time.time(),protocol='CPU0 serial; 2 warmups, 9 thread CPU samples; imports and constructors excluded; no concurrent campaigns/builds/references. External load/frequency uncontrolled.')))

"""Isolated ordinary triple API / VBM evaluation with raw failure diagnostics."""
import os,sys,json,math,time,hashlib,resource,signal
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent.parent
os.sched_setaffinity(0,{1,2,3,4,5,6})
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(60,62)if sys.argv[2]in ['value','gradient']else(8,10))
def clean(x):
 if hasattr(x,'tolist'):return clean(x.tolist())
 if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v)for v in x]
 if isinstance(x,float)and not math.isfinite(x):return None
 return x

def geometry(p):
 off=p['q']*p['s']/(1+p['q']);z=[complex(off,0),complex(-p['s']/(1+p['q']),0),complex(off,0)-p['sep2']*complex(math.cos(p['ang']),math.sin(p['ang']))];total=1+p['q']+p['q2'];m=[1/total,p['q']/total,p['q2']/total];return z,m

def run(c,e):
 import numpy as np
 p=c['params'];ts=c['epochs'];tol=c['rtol'];limb=c['limb'];z=dict(engine=e,classification='ok')
 if e in ['value','gradient']:
  sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable'not in str(f)];build=Path(os.environ.get('TRIPLE_BUILD',str(P/'local-package'))).resolve();sys.path.insert(0,str(build));import lcbinint as lc
  assert Path(lc._native.__file__).resolve().is_relative_to(build)
  z['native_sha256']=hashlib.sha256(Path(lc._native.__file__).read_bytes()).hexdigest()
  m=lc.LightCurve(lens='triple',param_type=c.get('param_type','vbm'),ld_c=limb,options=lc.Options(finite_source_tol=0.,finite_source_reltol=tol));assert m.options.finite_source_backend=='algebraic_boundary'
  try:
   a=time.thread_time();o=m.magnification(ts,p,with_gradient=e=='gradient');z['cpu_seconds']=time.thread_time()-a
   d=m.info(ts,p,with_gradient=e=='gradient');z.update(mu=o.magnifications,status=o.status,info={k:v for k,v in d.diagnostics.items()if k in ['finite_source_method','stop','error_estimate','support_certified','value_converged','node_evaluations','center_image_count']},info_mu=d.magnifications)
   if e=='gradient':z.update(gradients=o.gradients,gradient_valid=o.gradient_valid,gradient_parameters=list(o.gradient_parameters))
   if np.any(o.status):z['classification']='nonconverged'
   if e=='gradient'and not np.all(o.gradient_valid):z['classification']='invalid_gradient'
  except Exception as exc:
   z.update(classification='exception',error=repr(exc))
   try:z['raw_native']=m._native._evaluate_algebraic_boundary(np.asarray(ts),p,0.,tol,e=='gradient',True)
   except Exception as other:z['raw_error']=repr(other)
 else:
  import VBMicrolensing as V
  v=V.VBMicrolensing();v.SetMethod(getattr(v.Method,dict(SinglePoly='Singlepoly',MultiPoly='Multipoly',NoPoly='Nopoly')[e]));v.Tol=0.;v.RelTol=tol;v.a1=limb;v.a2=0.;v.SetLDprofile(v.LDlinear)
  a=time.thread_time()
  if p['rho']==0:
   ls,ms=geometry(p);v.SetLensGeometry([coord for zz,mm in zip(ls,ms) for coord in [zz.real,zz.imag,mm]])
   ca,sa=math.cos(p['alpha']),math.sin(p['alpha']);xs=[(t-p['t0'])/p['tE']*ca-p['u0']*sa for t in ts];ys=[(t-p['t0'])/p['tE']*sa+p['u0']*ca for t in ts];z.update(mu=[v.MultiMag0(x,y)for x,y in zip(xs,ys)],source_x=xs,source_y=ys)
  else:
   vp=[math.log(p[k])if k in ['s','q','rho','tE','sep2','q2']else p[k]for k in ['s','q','u0','alpha','rho','tE','t0','sep2','q2','ang']];o=v.TripleLightCurve(vp,ts);z.update(mu=list(o[0]),source_x=list(o[1]),source_y=list(o[2]))
  z['cpu_seconds']=time.thread_time()-a
  if not all(math.isfinite(x)and x>0 for x in z['mu']):z['classification']='invalid_value'
  z['vbm_version']='5.5'
 z['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return clean(z)
if __name__=='__main__':
 c=json.loads(Path(sys.argv[1]).read_text());print(json.dumps(run(c,sys.argv[2]),allow_nan=False),flush=True)

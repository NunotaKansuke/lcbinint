"""Bounded public-API curve comparison; run pilot, then matrix. No kernel calls."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback,resource,hashlib,platform
from collections import Counter
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
CASES=[
 ('ordinary',1.2,.01,.003,.3,0.,-1.,1.),
 ('wide_caustic',1.2,.01,.003,.03,-.08,-.2,.8),
 ('resonant',.95,.01,.005,.03,.5,-.15,.15),
 ('equal_mass',1.,1.,.01,.02,.7,-.3,.3),
 ('close_binary',.6,.1,.005,.05,1.2,-.3,.3),
 ('close_planet',.6,.001,.0005,-1e-5,.5,-.08,.08),
 ('broad_source',.6,.0001,.05,-1e-5,.5,-.08,.08),
 ('large_source',.6,.001,.5,-1e-5,.5,-.08,.08),
 ('tiny_source',1.5,.0001,1e-5,.2,-.2,-.5,.5),
 ('notebook_alpha180_rad',1.5,.01,.001,-.01,180.,-.08,.08),
]
def configs():
 result=[]
 for name,s,q,rho,u0,alpha,lo,hi in CASES:
  for limb in [0.,.5,.9]:
   for tol in [1e-3,1e-4]:
    result.append(dict(id=f'{name}_ld{limb:g}_tol{tol:g}',case=name,params=dict(s=s,q=q,rho=rho,t0=0.,tE=10.,u0=u0,alpha=alpha),tau_min=lo,tau_max=hi,limb=limb,tol=tol,n_times=401))
 return result

def load():
 import numpy as np
 from importlib.util import module_from_spec,spec_from_file_location
 import VBMicrolensing
 for f in tuple(sys.meta_path):
  if type(f).__module__=='_lcbinint_editable':sys.meta_path.remove(f)
 sys.path.insert(0,str(ROOT/'build-algebraic-boundary/fused-gradient/python'))
 source=ROOT/'python/lcbinint';native=ROOT/'build-algebraic-boundary/python-api/lcbinint'
 spec=spec_from_file_location('lcbinint',source/'__init__.py',submodule_search_locations=[str(source),str(native)])
 l=module_from_spec(spec);sys.modules['lcbinint']=l;spec.loader.exec_module(l)
 assert Path(l._native.__file__).resolve().is_relative_to(native.resolve())
 return np,l,VBMicrolensing

def worker(config):
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 np,l,VB=load();p=config['params'];tol=config['tol'];limb=config['limb'];times=p['t0']+p['tE']*np.linspace(config['tau_min'],config['tau_max'],config['n_times'])
 start=time.perf_counter()
 # c=0 is exactly a uniform source in both public APIs.
 curve=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(limb))
 v=VB.VBMicrolensing();v.Tol=v.RelTol=tol;v.a1=limb;v.a2=0.
 vp=[np.log(p['s']),np.log(p['q']),p['u0'],p['alpha'],np.log(p['rho']),np.log(p['tE']),p['t0']];vt=times.tolist();setup_ms=1e3*(time.perf_counter()-start)
 def lc():return curve.info(times,p)
 def vb():return np.asarray(v.BinaryLightCurve(vp,vt)[0],dtype=float)
 cold={};warm={};outputs={}
 for name,fn in [('lc',lc),('vbm',vb)]:
  start=time.perf_counter();outputs[name]=fn();cold[name]=1e3*(time.perf_counter()-start);warm[name]=[]
 for repeat in range(3):
  for name,fn in ([('lc',lc),('vbm',vb)] if repeat%2==0 else [('vbm',vb),('lc',lc)]):
   start=time.perf_counter();outputs[name]=fn();warm[name].append(1e3*(time.perf_counter()-start))
 info=outputs['lc'];a=np.asarray(info['mu']);b=outputs['vbm'];finite=np.isfinite(a)&np.isfinite(b)&(b>0);rel=np.full(len(times),np.nan);rel[finite]=abs(a[finite]-b[finite])/abs(b[finite]);scaled=np.full(len(times),np.nan);scaled[finite]=abs(a[finite]-b[finite])/np.maximum(tol,tol*abs(b[finite]));worst=int(np.nanargmax(rel)) if finite.any() else None
 integral=np.asarray(info['integrator_used'],dtype=bool);converged=np.asarray(info['value_converged'],dtype=bool);support=np.asarray(info['support_certified'],dtype=bool);fully=np.asarray(info['fully_converged'],dtype=bool)
 stop=Counter(str(s) for s in info['stop']);routes=Counter(str(s) for s in info['finite_source_method_name'])
 # An ordinary value call is checked separately, without mixing its time into info().
 start=time.perf_counter();plain=np.asarray(curve(times,p));plain_ms=1e3*(time.perf_counter()-start)
 start=time.perf_counter();g=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol);gradient_ms=1e3*(time.perf_counter()-start)
 grads=np.asarray(g['grad_mu']);gradient_valid=np.asarray(g['gradient_valid'])
 data=dict(times=times,lc=a,vbm=b,relative_difference=rel,scaled_difference=scaled,converged=converged,support_certified=support,fully_converged=fully,integral=integral,nodes=np.asarray(info['node_evaluations']),epoch_seconds=np.asarray(info['seconds']),gradient=grads,gradient_valid=gradient_valid,gradient_value_converged=np.asarray(g['value_converged']))
 np.savez_compressed(OUT/(config['id']+'.npz'),**data)
 result={**config,'status':'ok','setup_ms':setup_ms,'cold_ms':cold,'warm_ms':warm,'lc_ms':float(np.median(warm['lc'])),'vbm_ms':float(np.median(warm['vbm'])),'speed_vbm_over_lc':float(np.median(warm['vbm'])/np.median(warm['lc'])),'plain_call_ms':plain_ms,'plain_matches_info':bool(np.array_equal(a,plain,equal_nan=True)),'gradient_ms_one_call':gradient_ms,'finite_gradient_components':int(np.isfinite(grads).sum()),'valid_gradient_components':int(gradient_valid.sum()),'gradient_value_converged':int(np.asarray(g['value_converged']).sum()),'finite_values':int(finite.sum()),'value_converged':int(converged.sum()),'fully_converged':int(fully.sum()),'integral_epochs':int(integral.sum()),'integral_support_certified':int((support&integral).sum()),'routes':dict(routes),'stops':dict(stop),'nodes_total':int(np.sum(data['nodes'])),'nodes_max':int(np.max(data['nodes'])),'relative_difference_max':float(np.nanmax(rel)) if finite.any() else None,'relative_difference_p99':float(np.nanpercentile(rel,99)) if finite.any() else None,'relative_difference_median':float(np.nanmedian(rel)) if finite.any() else None,'scaled_difference_max':float(np.nanmax(scaled)) if finite.any() else None,'scaled_difference_over_one':int(np.sum(scaled>1)),'worst_index':worst,'worst_time':float(times[worst]) if worst is not None else None,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'native_file':l._native.__file__,'info_keys':list(info.keys())}
 (OUT/(config['id']+'.json')).write_text(json.dumps(result,indent=2)+'\n');return result

def run(mode):
 allconfigs=configs();(OUT/'cases.json').write_text(json.dumps(allconfigs,indent=2)+'\n')
 selected=[c for c in allconfigs if (c['case'] in ['ordinary','wide_caustic','large_source'] and c['limb']==.5 and c['tol']==1e-3)] if mode=='pilot' else allconfigs
 started=time.perf_counter()
 for c in selected:
  path=OUT/(c['id']+'.json')
  if mode!='pilot' and path.exists():continue
  try:
   r=subprocess.run([sys.executable,str(Path(__file__)), 'worker',json.dumps(c)],cwd=ROOT,env=os.environ,capture_output=True,text=True,timeout=40)
   if r.returncode:raise RuntimeError(r.stderr[-4000:] or r.stdout[-4000:])
   result=json.loads(path.read_text());print(c['id'],f"lc={result['lc_ms']:.2f} vb={result['vbm_ms']:.2f} ms",f"diff={result['relative_difference_max']:.3g} conv={result['value_converged']}/401",flush=True)
  except Exception as exc:
   result={**c,'status':'failed','error':str(exc)};path.write_text(json.dumps(result,indent=2)+'\n');print(c['id'],'FAILED',str(exc)[-300:],flush=True)
 (OUT/(mode+'_run.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-started,cases=len(selected),worker_timeout_seconds=40,cpu='one pinned allowed CPU; sequential workers',repeats=3),indent=2)+'\n')
if __name__=='__main__':
 if sys.argv[1]=='worker':worker(json.loads(sys.argv[2]))
 else:run(sys.argv[1])

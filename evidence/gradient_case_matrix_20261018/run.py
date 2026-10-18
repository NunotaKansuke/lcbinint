from pathlib import Path
import os,sys,json,time,hashlib,subprocess,statistics,argparse
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent/'tol_1e-4';O.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--build');p.add_argument('--index',type=int);p.add_argument('--rep',type=int);a=p.parse_args()
base=dict(t0=0.,tE=10.,u0=.2,alpha=.5,s=1.2,q=.01,rho=.001)
cases=[]
for name,change in [('ordinary',{}),('close_planet',dict(s=.7,q=.001,u0=.1)),('wide_planet',dict(s=1.8,q=.001,u0=.12)),('resonant',dict(s=1.,q=.001,u0=.05)),('equal_mass',dict(s=1.2,q=1.,u0=.4)),('large_source',dict(s=1.2,q=.01,rho=.02,u0=.1))]:
 for limb in [0.,.5]:cases.append(dict(name=name+('_uniform' if limb==0 else '_LD'),params=dict(base,**change),limb=limb,n=401,start=-10.,end=10.))
saved=json.loads((R/'evidence/holonomic/stress_20261015/cases.json').read_text())
for i in [33,81,107,117]:
 c=saved[i];span=c['span']*c['params']['tE'];cases.append(dict(name=c['id'],params=c['params'],limb=c['limb'],n=33,start=-span,end=span))
if a.worker:
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 sys.meta_path[:]=[f for f in sys.meta_path if 'editable' not in getattr(f,'__module__','')];sys.path.insert(0,str(R/a.build))
 import numpy as np,lcbinint as lc
 assert Path(lc.__file__).is_relative_to(R/a.build)
 cfg=lc.Options(param_type='vbm',nbin='auto',finite_source_backend='algebraic_boundary',finite_source_tol=1e-4,finite_source_reltol=1e-4)
 warm=lc.LightCurve(options=cfg,limb_darkening=lc.LimbDarkening.linear(.5));warm(np.linspace(-.05,.05,9),dict(base,s=1.3,u0=.0,rho=.003),with_gradient=True)
 c=cases[a.index];t=np.linspace(c['start'],c['end'],c['n']);row=dict(case=c['name'],rep=a.rep,measurements={});values={}
 for g in ([False,True] if a.rep%2==0 else [True,False]):
  curve=lc.LightCurve(options=cfg,limb_darkening=lc.LimbDarkening.linear(c['limb']))
  start=time.perf_counter();v=curve(t,c['params'],with_gradient=g,with_diagnostics=True);ms=1000*(time.perf_counter()-start)
  methods=np.asarray(v.diagnostics['finite_source_method']);sec=np.asarray(v.diagnostics['seconds'])*1000
  row['measurements'][str(g)]=dict(ms=ms,method_counts={str(int(m)):int(sum(methods==m)) for m in np.unique(methods)},method_ms={str(int(m)):float(sec[methods==m].sum()) for m in np.unique(methods)},finite=bool(np.isfinite(v.magnifications).all() and (not g or np.isfinite(v.gradients).all())),converged=int(sum(v.value_converged)),gradient_valid=int(sum(v.gradient_valid)) if g else None)
  values[g]=v.magnifications
 row['value_fused_max_scaled_difference']=float(np.max(abs(values[False]-values[True])/np.maximum(1,abs(values[False]))))
 print(json.dumps(row));sys.exit()
(O/'inputs.json').write_text(json.dumps(cases,indent=2)+'\n')
builds={'before':'evidence/easy_gradient_20261018/before-package','current':'build-route-unified'}
summary={}
for label,build in builds.items():
 rows=[];log=O/(label+'.jsonl')
 with log.open('w') as f:
  for i,c in enumerate(cases):
   samples=[]
   for rep in range(3):
    result=subprocess.run([sys.executable,str(__file__),'--worker','--build',build,'--index',str(i),'--rep',str(rep)],text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr)
    row=json.loads(result.stdout);f.write(json.dumps(row)+'\n');f.flush();samples.append(row)
   row=dict(c,measurements={})
   for g in ['False','True']:
    m=samples[-1]['measurements'][g].copy();m['ms']=statistics.median(s['measurements'][g]['ms'] for s in samples);m['samples_ms']=[s['measurements'][g]['ms'] for s in samples];row['measurements'][g]=m
   row['ratio']=row['measurements']['True']['ms']/row['measurements']['False']['ms'];row['max_value_fused_difference']=max(s['value_fused_max_scaled_difference'] for s in samples)
   rows.append(row);print(label,c['name'],round(row['measurements']['False']['ms'],3),round(row['measurements']['True']['ms'],3),round(row['ratio'],3),flush=True)
 summary[label]=rows
 (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

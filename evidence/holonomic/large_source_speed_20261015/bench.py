from pathlib import Path
import sys,os,json,time,subprocess
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def worker(c,variant):
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 import numpy as np
 from importlib.util import module_from_spec,spec_from_file_location
 for f in tuple(sys.meta_path):
  if type(f).__module__=='_lcbinint_editable':sys.meta_path.remove(f)
 native=ROOT/('build-algebraic-boundary/speed-before/lcbinint' if variant=='before' else 'build-algebraic-boundary/python-api/lcbinint');source=ROOT/'python/lcbinint'
 spec=spec_from_file_location('lcbinint',source/'__init__.py',submodule_search_locations=[str(source),str(native)]);l=module_from_spec(spec);sys.modules['lcbinint']=l;spec.loader.exec_module(l)
 p=c['params'];tol=c['tol'];times=np.asarray(c['times']) if 'times' in c else p['t0']+p['tE']*np.linspace(c['tau_min'],c['tau_max'],c['n_times']);curve=l.LightCurve(options=l.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=l.LimbDarkening.linear(c['limb']))
 curve.info(times,p);elapsed=[]
 for k in range(5):
  start=time.perf_counter();r=curve.info(times,p);elapsed.append(1e3*(time.perf_counter()-start))
 fields=['mu','value_converged','fully_converged','seed_support_proven','support_certified','center_image_count','augmented_seed_count','node_evaluations','integrator_used']
 np.savez_compressed(OUT/f"{c['id']}_{variant}.npz",times=times,**{k:np.asarray(r[k]) for k in fields})
 result={**c,'variant':variant,'ms':elapsed,'median_ms':float(np.median(elapsed)),'native':l._native.__file__};(OUT/f"{c['id']}_{variant}.json").write_text(json.dumps(result,indent=2)+'\n');print(c['id'],variant,result['median_ms'],flush=True)
if __name__=='__main__':
 if sys.argv[1]=='worker':worker(json.loads(sys.argv[2]),sys.argv[3])
 else:
  mode=sys.argv[1];cases=json.loads((ROOT/'evidence/holonomic/lightcurve_matrix_20261015/cases.json').read_text())
  if mode=='pilot':cases=[c for c in cases if c['case']=='large_source' and c['limb'] in [0,.5] and c['tol']==1e-3]
  elif mode=='large':cases=[c for c in cases if c['case'] in ['large_source','broad_source']]
  for c in cases:
   for variant in ([mode] if mode in ['scan','final'] else ['before','after']):
    if (OUT/f"{c['id']}_{variant}.json").exists():continue
    r=subprocess.run([sys.executable,str(Path(__file__)), 'worker',json.dumps(c),variant],cwd=ROOT,env=os.environ,capture_output=True,text=True,timeout=40)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    print(r.stdout.strip(),flush=True)

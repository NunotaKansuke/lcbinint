from pathlib import Path
import os,sys,json,time,subprocess,importlib.util
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
if len(sys.argv)>1:
 c=json.loads(sys.argv[2]);variant=sys.argv[1];os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 if variant=='before':
  spec=importlib.util.spec_from_file_location('_lcbinint_holonomic_fused',OUT/'before.so');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'));from benchmark import load
 np,l,_=load();p=c['params'];tol=c['tol'];times=p['tE']*np.linspace(c['tau_min'],c['tau_max'],401);curve=l.LightCurve(options=l.Options(coordinates='vbm',finite_source_backend='algebraic_boundary',finite_source_tol=tol,finite_source_reltol=tol),limb_darkening=l.LimbDarkening.linear(c['limb']))
 curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol);ms=[]
 for k in range(3):
  start=time.perf_counter();g=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol);ms.append((time.perf_counter()-start)*1000)
 np.savez_compressed(OUT/(c['id']+'_'+variant+'.npz'),**{k:np.asarray(g[k]) for k in ['mu','grad_mu','gradient_valid','value_converged','support_certified']})
 r=dict(id=c['id'],variant=variant,ms=ms,median_ms=float(np.median(ms)),finite=int(np.isfinite(g['grad_mu']).sum()),converged=int(np.asarray(g['value_converged']).sum()));(OUT/(c['id']+'_'+variant+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
else:
 cs=json.loads((ROOT/'evidence/holonomic/lightcurve_matrix_20261015/cases.json').read_text())
 for c in cs:
  if c['limb']!=.5 or c['tol']!=1e-3:continue
  for variant in ['before','after']:
   r=subprocess.run([sys.executable,str(Path(__file__)),variant,json.dumps(c)],capture_output=True,text=True,timeout=40);print(r.stdout or r.stderr,flush=True)

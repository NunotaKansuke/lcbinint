"""Repeated public gradient timing for controlled candidate comparison."""
import hashlib,json,os,sys,subprocess,time,statistics
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
import numpy as np
np,lc,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import _lcbinint_holonomic_fused as fused
cases={c['id']:c for c in json.loads((OUT/'cases.json').read_text())}
records=[]
for name in ('stratified_00','stratified_12','stratified_13','stratified_04'):
 c=cases[name];times=np.load(OUT/(name+'.npz'))['times'];p=c['params'];tol=1e-3
 curve=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(c['limb']))
 elapsed=[];reference=None;matches=[]
 for _ in range(6):
  start=time.perf_counter();r=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol);elapsed.append(1e3*(time.perf_counter()-start))
  data={'mu':np.asarray(r['mu']),'grad_mu':np.asarray(r['grad_mu']),'gradient_valid':np.asarray(r['gradient_valid']),'stop':np.asarray(r['stop'])}
  if reference is None:reference=data
  else:matches.append(all(np.array_equal(reference[k],data[k],equal_nan=(data[k].dtype.kind in 'fc')) for k in reference))
 rec={'id':name,'n_epochs':int(len(times)),'calls_ms':elapsed,'warm_median_ms':float(statistics.median(elapsed[1:])),'warm_min_ms':float(min(elapsed[1:])),'within_process_output_stable':bool(all(matches)),'valid_components':int(reference['gradient_valid'].sum()),'all_components':int(reference['gradient_valid'].size),'mu':reference['mu'].tolist(),'grad_mu':reference['grad_mu'].tolist(),'gradient_valid':reference['gradient_valid'].tolist(),'stop':reference['stop'].tolist()}
 records.append(rec);print(name,'calls_ms',[round(z,2) for z in elapsed],'warm_median',round(rec['warm_median_ms'],2),flush=True)
module=Path(fused.__file__).resolve()
out={'label':sys.argv[1],'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':1,'tol':1e-3,'calls_per_case':6,'records':records}
(OUT/(sys.argv[1]+'_gradient_repeat.json')).write_text(json.dumps(out,indent=2)+'\n')

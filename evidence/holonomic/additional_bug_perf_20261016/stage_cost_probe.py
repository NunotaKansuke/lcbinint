"""Separate value and gradient cost at exact-axis degeneracies."""
import hashlib,json,os,sys,subprocess,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
import numpy as np
np,lc,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import _lcbinint_holonomic_fused as fused
cases={c['id']:c for c in json.loads((OUT/'cases.json').read_text())}
points=[('stratified_00',0,'exact_axis',0.0),('stratified_00',0,'off_axis',1e-6),('stratified_12',13,'exact_axis',0.0),('stratified_12',13,'off_axis',1e-9),('stratified_13',13,'exact_axis',0.0),('stratified_13',13,'off_axis',1e-8)]
rows=[];tol=1e-3
for name,index,label,y_override in points:
 c=cases[name];p=c['params'];ts=np.load(OUT/(name+'.npz'))['times'];t=float(ts[index])
 curve=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(c['limb']))
 geom=curve.finite_source_geometry(np.asarray([t]),p);gx=float(np.asarray(geom.source_x).reshape(-1)[0]);gy=float(np.asarray(geom.source_y).reshape(-1)[0]);rho=float(np.asarray(geom.source_radius).reshape(-1)[0])
 x=gx;y=(gy if label=='exact_axis' and name=='stratified_13' else y_override)
 item={'case':name,'index':index,'point':label,'time_days':t,'source_x':x,'source_y':y,'trajectory_source_y':gy,'rho':rho,'routes':{}}
 for route in ('evaluate_value','evaluate'):
  evaluator=fused.BinarySeededFusedGradient(float(np.asarray(geom.separation).reshape(-1)[0]),float(np.asarray(geom.mass_ratio).reshape(-1)[0]),rho,1400)
  calls=[];last=None
  for _ in range(2):
   start=time.perf_counter();last=getattr(evaluator,route)(x,y,float(c['limb']),tol,tol,True);calls.append(1e3*(time.perf_counter()-start))
  item['routes'][route]={'call_ms':calls,'warm_call_ms':calls[-1],'stop':last['stop'],'value_converged':bool(last['value_converged']),'node_evaluations':int(last['node_evaluations']),'center_image_count':int(last['center_image_count']),'augmented_seed_count':int(last['augmented_seed_count']),'support_certified':bool(last['support_certified'])}
  if route=='evaluate':
   item['routes'][route]['gradient_valid_count']=int(np.asarray(last['gradient_valid'],dtype=bool).sum())
   item['routes'][route]['mu']=float(last['mu'])
   item['routes'][route]['grad_mu']=np.asarray(last['grad_mu'],dtype=float).tolist()
   item['routes'][route]['gradient_valid']=np.asarray(last['gradient_valid'],dtype=bool).tolist()
 print(name,label,'value',item['routes']['evaluate_value']['call_ms'],'gradient',item['routes']['evaluate']['call_ms'],flush=True)
 rows.append(item)
source=Path(lc._native.__file__).resolve();mod=Path(fused.__file__).resolve()
out={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'native_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'gradient_sha256':hashlib.sha256(mod.read_bytes()).hexdigest(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'tol':tol,'warmup_repetitions':1,'rows':rows}
(OUT/'stage_cost_probe.json').write_text(json.dumps(out,indent=2)+'\n')

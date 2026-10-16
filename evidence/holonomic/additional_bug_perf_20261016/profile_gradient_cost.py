"""Split slow public gradient calls into geometry, caustic setup, and epoch costs."""
import hashlib,json,os,sys,time,subprocess
from collections import Counter
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
import numpy as np
np,lc,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import _lcbinint_holonomic_fused as fused
base=OUT
reference=OUT/'post_patch'
case_rows=json.loads((base/'cases.json').read_text())
targets=['stratified_00','stratified_03','stratified_12','stratified_13']
records=[]
for name in targets:
 c=next(x for x in case_rows if x['id']==name);saved=np.load(base/(name+'.npz'));times=saved['times'];p=c['params'];limb=c['limb'];tol=1e-3
 curve=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(limb))
 t=time.perf_counter();geometry=curve.finite_source_geometry(times,p);geometry_ms=1e3*(time.perf_counter()-t)
 t=time.perf_counter();seeder=fused.BinarySeededFusedGradient(float(np.asarray(geometry.separation)[0]),float(np.asarray(geometry.mass_ratio)[0]),float(np.asarray(geometry.source_radius)[0]),1400);caustic_setup_ms=1e3*(time.perf_counter()-t)
 ids=sorted(set([0,len(times)//4,len(times)//2,3*len(times)//4,len(times)-1]))
 epochs=[]
 for i in ids:
  t=time.perf_counter();r=seeder.evaluate(float(geometry.source_x[i]),float(geometry.source_y[i]),float(limb),tol,tol,True);elapsed=1e3*(time.perf_counter()-t)
  epochs.append({'index':int(i),'time_days':float(times[i]),'source_x':float(geometry.source_x[i]),'source_y':float(geometry.source_y[i]),'elapsed_ms':elapsed,'stop':r['stop'],'node_evaluations':int(r['node_evaluations']),'center_image_count':int(r['center_image_count']),'augmented_seed_count':int(r['augmented_seed_count']),'seed_support_proven':bool(r['seed_support_proven']),'support_certified':bool(r['support_certified']),'gradient_valid_count':int(np.asarray(r['gradient_valid'],dtype=bool).sum())})
 t=time.perf_counter();result=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol);public_ms=1e3*(time.perf_counter()-t)
 t=time.perf_counter();result_small=curve.holonomic_gradient(times[ids],p,mu_atol=tol,mu_rtol=tol);subset_ms=1e3*(time.perf_counter()-t)
 record={'id':name,'params':p,'limb':limb,'n_epochs':int(len(times)),'screen_value_ms':float(json.loads((reference/(name+'.json')).read_text())['info_ms']),'screen_gradient_ms':float(json.loads((reference/(name+'.json')).read_text())['gradient_ms']),'geometry_preparation_ms':geometry_ms,'caustic_seed_object_setup_ms':caustic_setup_ms,'five_sample_evaluations':epochs,'public_all_epoch_gradient_ms':public_ms,'public_five_epoch_gradient_ms':subset_ms,'public_gradient_finite_components':int(np.isfinite(result['grad_mu']).sum()),'public_gradient_valid_components':int(np.asarray(result['gradient_valid'],dtype=bool).sum()),'public_gradient_stop_counts':dict(Counter(result['stop']))}
 records.append(record)
 print(name,'geom_ms',round(geometry_ms,3),'caustic_setup_ms',round(caustic_setup_ms,3),'subset_ms',round(subset_ms,3),'public_all_ms',round(public_ms,3),'epoch_samples_ms',[round(x['elapsed_ms'],3) for x in epochs],flush=True)
source=Path(lc._native.__file__).resolve();module=Path(fused.__file__).resolve()
out={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'native_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'gradient_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'tol':tol,'profiles':records}
(OUT/'gradient_cost_profile.json').write_text(json.dumps(out,indent=2)+'\n')

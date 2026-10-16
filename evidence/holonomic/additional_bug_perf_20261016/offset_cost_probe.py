"""Bounded one-CPU probe of gradient cost near source-axis/center geometry."""
import hashlib, json, os, sys, subprocess, time, statistics
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
import numpy as np
np, lc, VB=load()
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import _lcbinint_holonomic_fused as fused
cases=json.loads((OUT/'cases.json').read_text())
by_id={c['id']:c for c in cases}
probes=[('stratified_00',0),('stratified_00',20),('stratified_12',13),('stratified_13',13)]
offsets=[-0.1,-0.04,-0.01,-0.001,0.0,0.001,0.01,0.04,0.1]
records=[]
for name,index in probes:
 c=by_id[name];p=c['params'];rho=float(p['rho']);tol=1e-3
 curve=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(c['limb']))
 times=np.load(OUT/(name+'.npz'))['times']
 geom=curve.finite_source_geometry(np.asarray([times[index]]),p)
 x=float(np.asarray(geom.source_x).reshape(-1)[0]); y=float(np.asarray(geom.source_y).reshape(-1)[0])
 row={'case':name,'epoch_index':index,'time_days':float(times[index]),'source_x':x,'source_y':y,'rho':rho,'points':[]}
 for off in offsets:
  yy=y+off*rho
  ev=fused.BinarySeededFusedGradient(float(np.asarray(geom.separation).reshape(-1)[0]),float(np.asarray(geom.mass_ratio).reshape(-1)[0]),float(np.asarray(geom.source_radius).reshape(-1)[0]),1400)
  calls=[]; last=None
  for repeat in range(2):
   t=time.perf_counter();last=ev.evaluate(x,yy,float(c['limb']),tol,tol,True);calls.append(1e3*(time.perf_counter()-t))
  row['points'].append({'y_offset_over_rho':off,'source_y':yy,'call_ms':calls,'warm_call_ms':calls[-1],
   'stop':last['stop'],'value_converged':bool(last['value_converged']),
   'gradient_valid_count':int(np.asarray(last['gradient_valid'],dtype=bool).sum()),
   'node_evaluations':int(last['node_evaluations']),'center_image_count':int(last['center_image_count']),
   'augmented_seed_count':int(last['augmented_seed_count']),'support_certified':bool(last['support_certified'])})
  print(name,index,off,'ms',','.join(f'{z:.3f}' for z in calls),'nodes',last['node_evaluations'],'stop',last['stop'],flush=True)
 records.append(row)
source=Path(lc._native.__file__).resolve();module=Path(fused.__file__).resolve()
result={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'native_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'gradient_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
 'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':1,'tol':tol,'warmup_repetitions':1,
 'offsets_over_rho':offsets,'records':records}
(OUT/'offset_cost_probe.json').write_text(json.dumps(result,indent=2)+'\n')

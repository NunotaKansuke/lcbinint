"""Tight-tolerance triage for new invalid-gradient and symmetry candidates."""
import hashlib,json,os,sys,subprocess,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
import numpy as np
np,lc,VB=load();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
configs=json.loads((OUT/'cases.json').read_text())
def config(name):return next(c for c in configs if c['id']==name)
def curve_for(c,tol):
 return lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(c['limb']))
records=[]
for name in ('stratified_02','stratified_09'):
 c=config(name);saved=np.load(OUT/(name+'.npz'));all_t=saved['times']
 if name=='stratified_02':times=np.asarray([-0.275,-0.25,-0.225,-0.2,-0.075,-0.025,0.,0.025])
 else:times=np.asarray([-1.7455113096375089e-6,0.,1.7455113096375089e-6])
 by_tol=[]
 for tol in (1e-3,1e-5,1e-7):
  curve=curve_for(c,tol)
  info=curve.info(times,c['params'])
  grad=curve.holonomic_gradient(times,c['params'],mu_atol=tol,mu_rtol=tol)
  by_tol.append({'tol':tol,'times_days':times.tolist(),'mu':[float(x) for x in info['mu']],
   'value_converged':[bool(x) for x in info['value_converged']],
   'fully_converged':[bool(x) for x in info['fully_converged']],
   'stop':list(grad['stop']),'gradient_valid':np.asarray(grad['gradient_valid'],dtype=bool).tolist(),
   'gradient_finite':np.isfinite(np.asarray(grad['grad_mu'],dtype=float)).tolist(),
   'support_certified':[bool(x) for x in grad['support_certified']],
   'seed_support_proven':[bool(x) for x in grad['seed_support_proven']],
   'center_image_count':np.asarray(grad['center_image_count'],dtype=int).tolist(),
   'augmented_seed_count':np.asarray(grad['augmented_seed_count'],dtype=int).tolist()})
 records.append({'id':name,'params':c['params'],'limb':c['limb'],'tolerance_sweep':by_tol})

# The close-resonant, on-axis candidate also has large q/s gradient sensitivity.
c=config('stratified_00');times=np.asarray([-.025,0.,.025]);basep=c['params'];limb=c['limb']
base_curve=curve_for(c,1e-5);base_grad=base_curve.holonomic_gradient(times,basep,mu_atol=1e-5,mu_rtol=1e-5)
order=list(base_grad['param_order']);grad=np.asarray(base_grad['grad_mu'],dtype=float)
fd_records=[]
for key,steps in [('q',[1e-8,3e-9,1e-9]),('s',[1e-5,3e-6,1e-6,3e-7])]:
 for h in steps:
  plus=dict(basep);minus=dict(basep);plus[key]+=h;minus[key]-=h
  vp=np.asarray(base_curve(times,plus),dtype=float);vm=np.asarray(base_curve(times,minus),dtype=float)
  fd=(vp-vm)/(2*h);j=order.index(key)
  fd_records.append({'parameter':key,'step':h,'finite_difference':[float(x) for x in fd],
                     'analytic_tol1e5':[float(x) for x in grad[:,j]],
                     'max_abs_difference':float(np.max(np.abs(fd-grad[:,j]))),
                     'max_relative_difference':float(np.max(np.abs(fd-grad[:,j])/np.maximum(np.abs(fd),1e-8)))})
prov={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'native_sha256':hashlib.sha256(Path(lc._native.__file__).read_bytes()).hexdigest(),
      'gradient_sha256':hashlib.sha256(Path(__import__('_lcbinint_holonomic_fused').__file__).read_bytes()).hexdigest(),
      'invalid_gradient_cases':records,
      'highly_sensitive_case':{'id':'stratified_00','params':basep,'limb':limb,'times_days':times.tolist(),
       'analytic_tolerance1e5':grad.tolist(),'param_order':order,'finite_difference_step_study':fd_records}}
(OUT/'followup.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps({'invalid_cases':[(r['id'],[(x['tol'],sum(not all(v) for v in x['gradient_valid']),sum(not v for v in x['value_converged']),x['stop']) for x in r['tolerance_sweep']]) for r in records],
                  'fd':fd_records},indent=2))

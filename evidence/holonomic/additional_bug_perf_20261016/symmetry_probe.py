"""Check the exact reflection symmetry of source-plane gradients on the lens axis."""
import hashlib,json,os,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(ROOT/'evidence/holonomic/lightcurve_matrix_20261015'))
from benchmark import load
np,lc,VB=load()
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import _lcbinint_holonomic_fused as fused
p=dict(s=1.000001,q=1e-6,rho=1e-3,t0=0.,tE=10.,u0=0.,alpha=0.)
times=np.array([-.025,-.01,0.,.01,.025])
limb=0.;order=('t0','u0','tE','alpha','rho','q','s')
def make(tol):
 return lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=tol,finite_source_reltol=tol,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(limb))
analytic=[]
for tol in (1e-3,1e-5,1e-7):
 curve=make(tol);r=curve.holonomic_gradient(times,p,mu_atol=tol,mu_rtol=tol)
 j=order.index('u0')
 analytic.append({'tol':tol,'mu':[float(x) for x in r['mu']],
                  'du0':[float(x) for x in np.asarray(r['grad_mu'])[:,j]],
                  'du0_valid':[bool(x) for x in np.asarray(r['gradient_valid'])[:,j]],
                  'value_converged':[bool(x) for x in r['value_converged']],
                  'stop':list(r['stop']),'support_certified':[bool(x) for x in r['support_certified']]})
curve=make(1e-5);base=np.asarray(curve(times,p),dtype=float)
fd=[]
for h in (1e-4,3e-5,1e-5,3e-6,1e-6):
 plus=dict(p);minus=dict(p);plus['u0']=h;minus['u0']=-h
 vp=np.asarray(curve(times,plus),dtype=float);vm=np.asarray(curve(times,minus),dtype=float)
 fd.append({'step':h,'central':[float(x) for x in (vp-vm)/(2*h)],
            'right':[float(x) for x in (vp-base)/h],
            'left':[float(x) for x in (base-vm)/h],
            'relative_reflection_error':[float(x) for x in (vp-vm)/np.maximum(np.abs(vp),1e-12)]})
geom=curve.finite_source_geometry(times,p)
seed=fused.BinarySeededFusedGradient(float(np.asarray(geom.separation)[0]),float(np.asarray(geom.mass_ratio)[0]),p['rho'],1400)
raw=[]
for i,(x,y) in enumerate(zip(geom.source_x,geom.source_y)):
 r=seed.evaluate(float(x),float(y),limb,1e-3,1e-3,True)
 raw.append({'time':float(times[i]),'source_x':float(x),'source_y':float(y),
             'mu':float(r['mu']),'raw_source_y_gradient':float(r['grad_mu'][1]),
             'gradient_valid':[bool(x) for x in r['gradient_valid']],
             'value_converged':bool(r['value_converged']),'stop':r['stop'],
             'node_evaluations':int(r['node_evaluations']),
             'seed_support_proven':bool(r['seed_support_proven']),
             'support_certified':bool(r['support_certified']),
             'center_image_count':int(r['center_image_count']),
             'augmented_seed_count':int(r['augmented_seed_count'])})
module=Path(lc._native.__file__).resolve()
prov={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'native_module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
      'gradient_module_sha256':hashlib.sha256(Path(fused.__file__).read_bytes()).hexdigest(),
      'parameters':p,'times_days':times.tolist(),'limb_darkening_linear':limb,
      'symmetry':'For a binary lens on the real axis, mu(xs,ys)=mu(xs,-ys); at ys=0, dmu/dys=0 if differentiable.',
      'analytic_tolerance_sweep':analytic,'same_engine_value_finite_differences':fd,
      'raw_fused_source_plane_gradients':raw}
out=Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'symmetry_before.json'
out.write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps(prov,indent=2))

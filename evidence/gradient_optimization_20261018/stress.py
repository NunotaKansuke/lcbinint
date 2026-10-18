from pathlib import Path
import json,os,sys,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.meta_path[:]=[f for f in sys.meta_path if 'editable' not in getattr(f,'__module__','')];sys.path.insert(0,str(ROOT/'build-route-unified'))
import numpy as np,lcbinint as lc
assert Path(lc.__file__).is_relative_to(ROOT/'build-route-unified')
cases=json.loads((ROOT/'evidence/holonomic/stress_20261015/cases.json').read_text());records=[]
for case in cases:
 p=case['params'];t=np.linspace(-case['span'],case['span'],33)*p['tE'];c=lc.LightCurve(options=lc.Options(coordinates='vbm',nbin='auto',finite_source_tol=1e-3,finite_source_reltol=1e-3,finite_source_backend='algebraic_boundary'),limb_darkening=lc.LimbDarkening.linear(case['limb']))
 value=c(t,p,with_diagnostics=True);grad=c(t,p,with_gradient=True,with_diagnostics=True)
 before=np.load(ROOT/'evidence/backend_cleanup_20261018/baseline'/(case['id']+'.npz'))
 assert np.array_equal(value.magnifications,before['value']),case['id']+' value-only changed'
 assert np.all(np.isfinite(grad.gradients)) and np.all(np.isfinite(grad.magnifications)),case['id']
 delta=float(np.max(np.abs(grad.magnifications-value.magnifications)/np.maximum(1,np.abs(value.magnifications))))
 assert delta<2e-3,(case['id'],delta)
 metrics=dict(case=case['id'],value_fused_max_relative=delta,converged=int(sum(grad.value_converged)),valid=int(sum(grad.gradient_valid)),changed_valid=int(np.sum(grad.gradient_valid!=before['gradient_valid'])),gradient_max_scaled_change=float(np.max(np.abs(grad.gradients-before['gradients'])/np.maximum(1,np.abs(before['gradients'])))))
 records.append(metrics)
 (OUT/'stress-results').mkdir(exist_ok=True)
 np.savez(OUT/'stress-results'/(case['id']+'.npz'),value=value.magnifications,gradient=grad.magnifications,gradients=grad.gradients,gradient_valid=grad.gradient_valid)
 if len(records)%10==0:print(len(records),'/',len(cases),flush=True)
 (OUT/'stress.json').write_text(json.dumps(dict(native_sha256=hashlib.sha256(Path(lc._native.__file__).read_bytes()).hexdigest(),cases=records),indent=2)+'\n')
print('All 120 completed; value-only bitwise unchanged; finite value/tangent candidates.')

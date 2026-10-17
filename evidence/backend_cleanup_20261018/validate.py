"""Compare all saved 120 configurations with the pre-cleanup unified engine."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
ROOT=Path(__file__).resolve().parents[2]
BUILD=ROOT/'build-route-unified'
sys.meta_path[:]=[f for f in sys.meta_path if 'editable' not in getattr(f,'__module__','')]
sys.path.insert(0,str(BUILD))
import numpy as np
import lcbinint as lc
assert Path(lc.__file__).is_relative_to(BUILD)
assert Path(lc._native.__file__).is_relative_to(BUILD)
if hasattr(os,'sched_setaffinity'): os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
OUT=Path(__file__).resolve().parent
cases=json.loads((ROOT/'evidence/holonomic/stress_20261015/cases.json').read_text())
records=[]
for case in cases:
 p=case['params']; t=np.linspace(-case['span'],case['span'],33)*p['tE']
 model=lc.LightCurve(lens='binary',options=lc.Options(param_type='vbm',nbin='auto',
     finite_source_backend='algebraic_boundary',finite_source_tol=1e-3,finite_source_reltol=1e-3),
     limb_darkening=lc.LimbDarkening.linear(case['limb']))
 result={}; durations=[]
 for grad in (False,True):
  start=time.perf_counter(); result[grad]=model(t,p,with_gradient=grad,with_diagnostics=True)
  durations.append((time.perf_counter()-start)*1000)
 before=np.load(OUT/'baseline'/(case['id']+'.npz'))
 diffs={}
 for name,now in [('value',result[False].magnifications),('gradient',result[True].magnifications),
                  ('gradients',result[True].gradients),('gradient_valid',result[True].gradient_valid)]:
  now=np.asarray(now)
  assert np.all(np.isfinite(now)),(case['id'],name)
  diffs[name]=dict(exact=bool(np.array_equal(now,before[name])),
      max_abs=float(np.max(np.abs(now.astype(float)-before[name].astype(float)))))
 records.append(dict(case=case['id'],differences=diffs,value_ms=durations[0],
     gradient_ms=durations[1],value_converged=int(np.sum(result[False].value_converged)),
     gradient_valid=int(np.sum(result[True].gradient_valid))))
 if len(records)%10==0:print(len(records),'/',len(cases),flush=True)
 (OUT/'numerical_parity.json').write_text(json.dumps(dict(native=str(lc._native.__file__),
     native_sha256=hashlib.sha256(Path(lc._native.__file__).read_bytes()).hexdigest(),
     cases=records),indent=2)+'\n')
assert all(d['exact'] for r in records for d in r['differences'].values()),'cleanup changed numerical candidates'
print('All 120 cases: values, fused values, derivatives and validity flags match exactly.')

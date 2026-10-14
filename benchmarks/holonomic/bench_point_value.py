"""Scalar-only point-curve comparison with explicit native build provenance."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import VBMicrolensing

parser=argparse.ArgumentParser()
parser.add_argument('--native-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--repetitions',type=int,default=101)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
for finder in list(sys.meta_path):
    if any(x=='lcbinint' or x.startswith('lcbinint.') for x in getattr(finder,'known_source_files',{})):
        sys.meta_path.remove(finder)
sys.path.insert(0,str(args.native_dir.resolve()))
native=importlib.import_module('_lcbinint')
sys.modules['lcbinint._lcbinint']=native
package=root/'python/lcbinint'
spec=importlib.util.spec_from_file_location('lcbinint',package/'__init__.py',submodule_search_locations=[str(package)])
lcbinint=importlib.util.module_from_spec(spec)
sys.modules['lcbinint']=lcbinint
spec.loader.exec_module(lcbinint)
rows=[]
for count in [201,1001,2001]:
 for ld in [0.,.5]:
  params=dict(t0=0.,u0=.3,tE=1.,alpha=0.,rho=.003,q=.01,s=1.2)
  times=np.linspace(-1.,1.,count)
  curve=lcbinint.LightCurve(lens='binary',ld_c=ld,options=lcbinint.Options(param_type='vbm',nbin='auto',finite_source_tol=.001,finite_source_reltol=.001))
  vb=VBMicrolensing.VBMicrolensing(); vb.Tol=vb.RelTol=.001; vb.a1=ld
  vp=[math.log(1.2),math.log(.01),.3,0.,math.log(.003),0.,0.]; vt=times.tolist()
  diagnostic=curve._native._evaluate_holonomic(times,params,.001,.001)
  assert set(diagnostic['finite_source_method_name'])=={'point_source'}
  calls=dict(native=lambda:curve._native._magnification_holonomic(times,params,.001,.001),public=lambda:curve.holonomic_magnification(times,params),vbm=lambda:vb.BinaryLightCurve(vp,vt)[0])
  values={name:call() for name,call in calls.items()}
  np.testing.assert_allclose(values['native'],values['vbm'],rtol=2e-12,atol=2e-12)
  np.testing.assert_array_equal(values['native'],values['public'])
  samples={name:[] for name in calls}
  for k in range(args.repetitions):
   order=list(calls); order=order[k%3:]+order[:k%3]
   for name in order:
    start=time.perf_counter(); calls[name](); samples[name].append(time.perf_counter()-start)
  row=dict(epochs=count,ld=ld,params=params,atol=.001,rtol=.001,samples_seconds=samples,median_ms={name:1000*statistics.median(x) for name,x in samples.items()},max_relative_vbm_difference=float(np.max(abs(np.array(values['native'])-values['vbm'])/np.maximum(abs(np.array(values['vbm'])),1e-12))))
  rows.append(row); print(count,ld,row['median_ms'],flush=True)
report=dict(native=native.__file__,native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),repetitions=args.repetitions,threads=1,vbm_version=VBMicrolensing.__version__,no_gradients=True,fresh_lc_preparation_each_call=True,construction_and_warmup_excluded=True,rows=rows)
args.output.write_text(json.dumps(report,indent=2)+'\n')

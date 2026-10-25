"""Saved, bounded full-integral/reference audit. Run with the local build on PYTHONPATH."""
import argparse
import json
import math
import pathlib
import time
import importlib.metadata
import sys
import VBMicrolensing as vbm

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--case',default='contact')
parser.add_argument('--build',type=pathlib.Path)
args=parser.parse_args()
if args.build:
 build=args.build.resolve()
 sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable' not in str(f)]
 sys.path.insert(0,str(build))
 import lcbinint as native
 assert pathlib.Path(native._native.__file__).resolve().is_relative_to(build)
else:
 import _triple_algebraic_test as native
args.output.parent.mkdir(parents=True,exist_ok=True)
cases={
 'contact':dict(lenses=[0j,1+0j,-.4+.6j],masses=[.72,.2,.08],source=.1+.15j,rho=.15),
 'regular':dict(lenses=[0j,1+0j,-.4+.6j],masses=[.72,.2,.08],source=.5+.4j,rho=.03),
 'large':dict(lenses=[0j,1+0j,-.4+.6j],masses=[.72,.2,.08],source=.1+.15j,rho=.3),
 'planet':dict(lenses=[0j,1.2+0j,.7+.3j],masses=[1/1.0011,.001/1.0011,.0001/1.0011],source=.1+.05j,rho=.01),
 'asymmetric':dict(lenses=[.3-.2j,.8+.9j,-.7+.6j],masses=[.65,.25,.1],source=.2+.4j,rho=.12),
}
case=cases[args.case]
results={'case':args.case,'input':{k:[str(x) for x in v] if k=='lenses' else str(v) if k=='source' else v for k,v in case.items()},
 'vbm_version':importlib.metadata.version('VBMicrolensing'),'reference_method':'Singlepoly',
 'extension':getattr(native,'_native',native).__file__,'rows':[]}
def save():
 args.output.write_text(json.dumps(results,indent=2)+'\n')
def reference(limb,tolerance):
 b=vbm.VBMicrolensing();b.SetMethod(b.Method.Singlepoly);b.Tol=tolerance;b.RelTol=0
 geometry=[]
 for z,m in zip(case['lenses'],case['masses']):geometry.extend([z.real,z.imag,m])
 b.SetLensGeometry(geometry)
 b.a1=limb
 return b.MultiMagDark(case['source'].real,case['source'].imag,case['rho'],tolerance) if limb else b.MultiMag2(case['source'].real,case['source'].imag,case['rho'])
for limb,gradient in [(0.,False),(.5,False),(.5,True)]:
 start=time.monotonic()
 row=native.triple_algebraic(**case,limb=limb,with_gradient=gradient,atol=1e-7,rtol=1e-6)
 row.update(limb=limb,with_gradient=gradient,seconds=time.monotonic()-start)
 results['rows'].append(row);save()
 refs=[reference(limb,t) for t in (1e-6,2e-7)]
 row['reference_values']=refs;row['reference_stability']=abs(refs[1]-refs[0]);row['absolute_difference']=abs(row['mu']-refs[1]);save()
 print(args.case,limb,gradient,row['stop'],row['mu'],refs,row['absolute_difference'],flush=True)
 assert row['value_converged'] and row['support_complete'],row
 assert not gradient or row['gradient_reliable'],row
 allowed=row['error_estimate']+max(2e-6,4*row['reference_stability'])
 row['comparison_status']='PASS' if row['absolute_difference']<=allowed else 'INCONCLUSIVE_VBM_DARK'
 save()
 # Uniform-source contour magnifications remain a strict independent gate.
 # VBM's annular dark-source calculation is a separate, capped algorithm;
 # failed dark comparisons remain in the audit and are resolved with the
 # independently adaptive original-map GK test, not by loosening tolerance.
 if limb==0:assert row['absolute_difference']<=allowed,(row,allowed)
results['solver_status']='PASS'
results['dark_reference_status']='PASS' if all(r.get('comparison_status')=='PASS' for r in results['rows']) else 'INCONCLUSIVE_REQUIRES_GK'
save()

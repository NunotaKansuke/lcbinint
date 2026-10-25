"""Public API/derivative audit, with an explicitly verified active-checkout build."""
import argparse
import importlib
import json
import math
import pathlib
import sys
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--build',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--param-type',default='vbm')
args=parser.parse_args()
build=args.build.resolve()
sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable' not in str(f)]
sys.path.insert(0,str(build))
import lcbinint
assert pathlib.Path(lcbinint.__file__).resolve().is_relative_to(build)
assert pathlib.Path(lcbinint._native.__file__).resolve().is_relative_to(build)
params=dict(t0=-(.2/.92-.1),u0=-.15,tE=1.,alpha=0.,s=1.,q=.2/.72,q2=.08/.72,
            sep2=math.hypot(.4,.6),ang=math.atan2(.6,-.4),rho=.15)
if args.param_type!='vbm':params.update(t0=-.1,u0=.15,alpha=.2)
options=lcbinint.Options(finite_source_backend='algebraic_boundary',finite_source_tol=1e-7,finite_source_reltol=1e-6)
curve=lcbinint.LightCurve(lens='triple',param_type=args.param_type,options=options,ld_c=.5)
result=curve.info([0.],params,with_gradient=True)
assert result.status.tolist()==[0] and result.gradient_valid.tolist()==[True],result
assert len(result.gradient_parameters)==10
assert result.gradients.shape==(1,10)
value_curve=lcbinint.LightCurve(lens='triple',param_type=args.param_type,
    options=lcbinint.Options(finite_source_backend='algebraic_boundary',finite_source_tol=1e-9,finite_source_reltol=1e-9),ld_c=.5)
report={'package':lcbinint.__file__,'extension':lcbinint._native.__file__,'param_type':args.param_type,
        'input':params,'mu':result.magnifications.tolist(),'gradient':result.gradients.tolist(),'rows':[]}
args.output.parent.mkdir(parents=True,exist_ok=True)
def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
save()
for j,name in enumerate(result.gradient_parameters):
 row={'parameter':name,'analytic':float(result.gradients[0,j]),'steps':[]};report['rows'].append(row)
 for h in [3e-4,1e-4]:
  plus=dict(params);minus=dict(params);plus[name]+=h;minus[name]-=h
  vp=value_curve.info([0.],plus);vm=value_curve.info([0.],minus)
  assert vp.status.tolist()==[0] and vm.status.tolist()==[0],(name,vp,vm)
  fd=float((vp.magnifications[0]-vm.magnifications[0])/(2*h))
  row['steps'].append({'h':h,'finite_difference':fd,'absolute_difference':abs(fd-row['analytic'])})
  print(args.param_type,name,h,fd,row['analytic'],abs(fd-row['analytic']),flush=True)
  save()
  assert abs(fd-row['analytic'])<1e-4+2e-4*abs(fd),row
report['status']='PASS';save()

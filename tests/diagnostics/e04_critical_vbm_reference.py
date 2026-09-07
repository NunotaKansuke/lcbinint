"""Independent VBM 5.x value and step-study evidence; JSON to stdout.

Run in an explicitly selected native environment with x64, CPU and one thread.
No integration environment or result files are modified by this script.
"""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import VBMicrolensing
from lcbinint_jax import binary_algebraic_boundary_batch_value_jacobian_ffi as native

cases = [
    ('regular_1',[-.1737444200273406,.15560487302576048,.01,10.,1.2]),
    ('regular_2',[-.11642423067980426,.17333608542544085,.01,10.,1.2]),
    ('regular_3',[-.05910404133226791,.19106729782512122,.01,10.,1.2]),
    ('regular_4',[-.0017838519847315562,.2087985102248016,.01,10.,1.2]),
    ('regular_5',[.0555363373628048,.22652972262448196,.01,10.,1.2]),
    ('q_low',[.037,-.021,.007,.02,.9]),
    ('q_high',[-.037,-.021,.007,50.,.9]),
    ('q_low_second',[-.11,.04,.007,.02,.9]),
    ('q_high_second',[.11,.04,.007,50.,.9]),
    ('ld_uniform',[.08,.03,.01,.001,1.1]),
    ('ld_linear',[.08,.03,.01,.001,1.1]),
    ('ld_square_root',[.08,.03,.01,.001,1.1]),
    ('near_caustic',[.1,.05,.01,.2,1.]),
    ('low_q_close',[.002,.0007,.001,.001,1.1]),
]
for i,t in enumerate([-.12,0.,.12]):
    cases.append((f'signed_u0_{i+1}',[t*np.cos(.3)+.2*np.sin(.3),
        t*np.sin(.3)-.2*np.cos(.3),.01,10.,1.2]))

parser=argparse.ArgumentParser()
parser.add_argument('--regular-uniform-only',action='store_true',
    help='The other three cases use e04_fixed_disc_reference.py with a stable reference stencil.')
args=parser.parse_args()
if args.regular_uniform_only:
    cases=[entry for entry in cases if entry[0] not in {'ld_linear','ld_square_root','low_q_close'}]

package=Path(VBMicrolensing.__file__).resolve()
so=package.parent/'VBMicrolensing.so'
print(json.dumps({'reference':str(so),'sha256':hashlib.sha256(so.read_bytes()).hexdigest(),
                  'version':VBMicrolensing.__version__}),flush=True)
failures=0
for name,p in cases:
    p=np.array(p)
    c=.4 if name=='ld_linear' else 0.
    d=.3 if name=='ld_square_root' else 0.
    vbm=VBMicrolensing.VBMicrolensing()
    vbm.Tol=1e-12;vbm.RelTol=1e-12
    vbm.a1=c;vbm.a2=d
    vbm.SetLDprofile(vbm.LDsquareroot if d else vbm.LDlinear)
    def value(p):
        # Public -> native sign and reciprocal-q transformation is inverted.
        args=(float(p[4]),float(1/p[3]),float(-p[0]),float(-p[1]),float(p[2]),1e-12)
        return float(vbm.BinaryMagDark(*args) if c or d else vbm.BinaryMag(*args))
    result=native(jnp.array(p[0:1]),jnp.array(p[1:2]),jnp.array(p[4:5]),p[2],p[3],
        limb_c=c,limb_d=d,refinement_policy_id='algebraic-critical-events-v2')
    actual=float(result.magnification[0]); jac=np.asarray(result.jacobian[0])
    reference=value(p)
    steps=[]; fd=[]
    for fraction in [1e-3,5e-4,2.5e-4,1.25e-4]:
        h=fraction*np.array([p[2],p[2],p[2],min(p[3],1.),p[2]])
        steps.append(h.tolist())
        fd.append([(value(p+np.eye(5)[i]*h[i])-value(p-np.eye(5)[i]*h[i]))/(2*h[i]) for i in range(5)])
    fd=np.array(fd)
    richardson=(4*fd[1:]-fd[:-1])/3
    errors=np.abs(richardson-jac)
    # All step estimates are retained. This diagnostic reports non-acceptance
    # when even the best extrapolate exceeds the declared absolute J budget.
    ok=int(result.status[0])==0 and abs(actual-reference)<=1e-4+1e-4*max(1,abs(reference)) and np.all(errors.min(axis=0)<=3e-4)
    failures+=not ok
    print(json.dumps({'case':name,'native_parameters':p.tolist(),'ld':[c,d],
        'status':int(result.status[0]),'A':actual,'reference_A':reference,
        'value_difference':actual-reference,'J':jac.tolist(),'steps':steps,
        'reference_fd':fd.tolist(),'richardson_abs_errors':errors.tolist(),'pass':bool(ok)}),flush=True)
print(json.dumps({'failures':int(failures),'count':len(cases)}),flush=True)
raise SystemExit(bool(failures))

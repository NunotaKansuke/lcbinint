"""Fixed-node VBM uniform-disc oracle, independent of native image quadrature.

Integration by parts gives the linear moment integral (1-u^2) A(rho sqrt(1-u^2))
and, with mu=u^2, square-root moment (1-u^4) A(rho sqrt(1-u^4)).
This avoids differentiating VBM's adaptively selected LD annuli.
"""
import json
import numpy as np
import jax.numpy as jnp
import VBMicrolensing
from lcbinint_jax import binary_algebraic_boundary_batch_value_jacobian_ffi as native

vbm=VBMicrolensing.VBMicrolensing()
vbm.Tol=1e-12;vbm.RelTol=1e-12
failures=0

def disc(p):
    return float(vbm.BinaryMag(float(p[4]),float(1/p[3]),float(-p[0]),float(-p[1]),float(p[2]),1e-12))

def value(p,c,d,n):
    total=(1-c-d)*disc(p)
    if not(c or d):return total
    nodes,weights=np.polynomial.legendre.leggauss(n)
    for u,w in zip((nodes+1)/2,weights/2):
        power=2 if c else 4
        fraction=1-u**power
        pp=p.copy();pp[2]*=np.sqrt(fraction)
        total+=(c or d)*w*fraction*disc(pp)
    return total/(1-c/3-d/5)

for name,p,c,d in [('linear',[.08,.03,.01,.001,1.1],.4,0.),
                    ('square_root',[.08,.03,.01,.001,1.1],0.,.3),
                    ('lowq',[.002,.0007,.001,.001,1.1],0.,0.)]:
    p=np.array(p)
    result=native(jnp.array(p[0:1]),jnp.array(p[1:2]),jnp.array(p[4:5]),p[2],p[3],
        limb_c=c,limb_d=d,refinement_policy_id='algebraic-critical-events-v2')
    J=np.asarray(result.jacobian[0]);A=float(result.magnification[0])
    for n in ([16,32] if c or d else [0]):
        ref=value(p,c,d,n)
        for factor in [1.,.5]:
            h=factor*np.array([.01*p[2],.01*p[2],.01*p[2],.01*p[3],1e-4])
            estimates=[]
            for j in range(5):
                delta=np.eye(5)[j]*h[j]
                # Sixth-order centered stencil; retain both step scales.
                f={k:value(p+k*delta,c,d,n) for k in [-3,-2,-1,1,2,3]}
                estimates.append((-f[-3]+9*f[-2]-45*f[-1]+45*f[1]-9*f[2]+f[3])/(60*h[j]))
            print(json.dumps({'case':name,'order':n,'step_factor':factor,'steps':h.tolist(),
                'status':int(result.status[0]),'A':A,'reference_A':ref,'value_difference':A-ref,
                'J':J.tolist(),'reference_J':estimates,'errors':(np.array(estimates)-J).tolist()}),flush=True)
            failures += (int(result.status[0]) != 0 or
                         abs(A-ref) > 1e-4+1e-4*max(1.,abs(ref)) or
                         np.max(np.abs(np.array(estimates)-J)) > 3e-4)
print(json.dumps({'failures':int(failures)}),flush=True)
raise SystemExit(bool(failures))

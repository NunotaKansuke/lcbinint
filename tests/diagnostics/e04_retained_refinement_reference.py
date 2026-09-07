"""Independent fixed-disc VBM oracle for a retained, uncertified Jet estimate."""
import json
from functools import lru_cache
import numpy as np
import jax.numpy as jnp
import VBMicrolensing
from lcbinint_jax import binary_algebraic_boundary_batch_value_jacobian_ffi as native

p=np.array([.05135221895347432,.22523542312781708,.01,10.,1.2])
solver=VBMicrolensing.VBMicrolensing();solver.Tol=1e-12;solver.RelTol=1e-12
@lru_cache(maxsize=2)
def nodes(n):
    x,w=np.polynomial.legendre.leggauss(n)
    return (x+1)/2,w/2
def reference(v,n):
    def disc(rho):
        return float(solver.BinaryMag(float(v[4]),float(1/v[3]),float(-v[0]),float(-v[1]),float(rho),1e-12))
    total=.6*disc(v[2])
    x,w=nodes(n)
    for u,weight in zip(x,w):
        fraction=1-u*u
        total+=.4*weight*fraction*disc(v[2]*np.sqrt(fraction))
    return total/(1-.4/3)
r=native(jnp.array(p[:1]),jnp.array(p[1:2]),jnp.array(p[4:5]),p[2],p[3],
    limb_c=.4,value_atol=1e-8,value_rtol=1e-8,gradient_atol=1e-5,
    refinement_policy_id='algebraic-critical-events-v2')
assert int(r.status[0])==3  # The independent oracle must not relabel this certified.
for order in [16,32]:
    value=reference(p,order)
    for factor in [1.,.5]:
        steps=factor*np.array([1e-4,1e-4,1e-4,.1,1e-4])
        gradient=[]
        for axis,h in enumerate(steps):
            delta=np.eye(5)[axis]*h
            f={k:reference(p+k*delta,order) for k in [-3,-2,-1,1,2,3]}
            gradient.append((-f[-3]+9*f[-2]-45*f[-1]+45*f[1]-9*f[2]+f[3])/(60*h))
        difference=np.asarray(gradient)-np.asarray(r.jacobian[0])
        print(json.dumps({'order':order,'step_factor':factor,'status':int(r.status[0]),
            'A':float(r.magnification[0]),'reference_A':value,'J':np.asarray(r.jacobian[0]).tolist(),
            'reference_J':gradient,'max_absolute_gradient_difference':float(abs(difference).max())}),flush=True)
        assert abs(float(r.magnification[0])-value) < 1e-8+1e-8*max(1.,abs(value))
        assert np.all(abs(difference)<3e-4)

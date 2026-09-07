"""Opt-in critical-event policy: strict native results and transform closure."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lcbinint_jax import (
    binary_algebraic_boundary_batch_ffi as forward,
    binary_algebraic_boundary_batch_value_jacobian_ffi as explicit,
    reset_algebraic_boundary_batch_call_counters as reset,
    algebraic_boundary_batch_call_counters as counters,
)

POLICY = 'algebraic-critical-events-v2'

import lcbinint._lcbinint as extension
pytestmark = pytest.mark.skipif(
    not getattr(extension._jax_ir, 'algebraic_critical_events_enabled', lambda: False)(),
    reason='critical-event policy is opt-in at build time',
)

def call(p, *, jac=False, **kw):
    return (explicit if jac else forward)(
        p[0:1], p[1:2], p[4:5], p[2], p[3],
        refinement_policy_id=POLICY, **kw)

@pytest.mark.parametrize('p', [
    [.0555363373628048,.22652972262448196,.01,10.,1.2],
    [.037,-.021,.007,.02,.9],
    [-.037,-.021,.007,50.,.9],
    [.08,.03,.01,.001,1.1],
    [.002,.0007,.001,.001,1.1],
])
def test_strict_value_jacobian_and_jvp_closure(p):
    p = jnp.array(p)
    result = call(p, jac=True)
    assert result.status[0] == 0
    assert np.isfinite(result.jacobian).all()
    ordinary = call(p).magnification
    compiled = jax.jit(lambda v: call(v).magnification)
    primal, tangent = jax.jvp(compiled, (p,), (jnp.ones_like(p),))
    np.testing.assert_array_equal(ordinary, primal)
    np.testing.assert_array_equal(result.magnification, primal)
    np.testing.assert_allclose(tangent, result.jacobian.sum(axis=1), rtol=1e-14)
    value, gradient = jax.value_and_grad(lambda v: call(v).magnification.sum())(p)
    np.testing.assert_array_equal(value, ordinary[0])
    np.testing.assert_array_equal(gradient, result.jacobian[0])

@pytest.mark.parametrize('count', [1,5,257])
def test_one_full_curve_call(count):
    x = jnp.linspace(.05,.06,count)
    fn = jax.jit(lambda x: forward(x,jnp.full_like(x,.2265),
        jnp.full_like(x,1.2),.01,10.,refinement_policy_id=POLICY).magnification)
    assert np.isfinite(fn(x)).all()
    reset()
    fn(x).block_until_ready()
    assert counters() == {'forward':1,'jacobian':0}
    reset()
    jax.jvp(fn,(x,),(jnp.ones_like(x),))[1].block_until_ready()
    assert counters() == {'forward':0,'jacobian':1}
    assert str(fn.lower(x).compiler_ir(dialect='stablehlo')).count('custom_call') == 1

@pytest.mark.parametrize('p', [
    [0.,0.,.01,1.,1.], [.05,0.,.01,1.,1.],
    [.05,.2,0.,.1,1.], [.05,.2,.01,0.,1.],
    [.05,.2,.01,.1,-1.], [np.nan,.2,.01,.1,1.],
])
def test_degenerate_and_invalid_fail_closed(p):
    result = call(jnp.array(p),jac=True)
    assert result.status[0] != 0
    assert np.isnan(result.magnification).all()
    assert np.isnan(result.jacobian).all()

def test_fixed_mask_and_strict_budget():
    args=(jnp.array([.055,.055]),jnp.array([.2265,.2265]),jnp.array([1.2,1.2]),.01,10.)
    result=explicit(*args, finite_mask=jnp.array([True,False]),refinement_policy_id=POLICY)
    np.testing.assert_array_equal(result.status,[0,0])
    np.testing.assert_array_equal(result.method,[2,1])
    failed=explicit(*args,gradient_atol=0.,refinement_policy_id=POLICY)
    assert np.all(failed.status != 0)
    assert np.isnan(failed.jacobian).all()

@pytest.mark.parametrize('p,c,d', [
    ([.08,.03,.01,.001,1.1],.4,0.),
    ([.08,.03,.01,.001,1.1],0.,.3),
    ([.002,.0007,.001,.001,1.1],0.,0.),
])
def test_returned_value_finite_difference_step_study(p,c,d):
    p=np.array(p)
    def evaluate(v):
        result=call(jnp.array(v),jac=True,limb_c=c,limb_d=d)
        assert result.status[0] == 0
        return result
    base=evaluate(p)
    estimates=[]
    for fraction in [1e-3,5e-4,2.5e-4]:
        steps=fraction*np.array([p[2],p[2],p[2],p[3],p[2]])
        estimates.append([(float(evaluate(p+np.eye(5)[i]*h).magnification[0])-
                           float(evaluate(p-np.eye(5)[i]*h).magnification[0]))/(2*h)
                          for i,h in enumerate(steps)])
    estimates=np.array(estimates)
    extrapolates=(4*estimates[1:]-estimates[:-1])/3
    # Both extrapolates must agree, not just a selected best step.
    np.testing.assert_allclose(extrapolates,np.broadcast_to(base.jacobian[0],(2,5)),
                               rtol=0.,atol=3e-4)

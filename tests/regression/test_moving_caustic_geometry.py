"""Moving geometry must retain fold and enclosed-caustic image components."""
import numpy as np
import pytest


@pytest.mark.parametrize("q", [.001, 1., 1000.])
@pytest.mark.parametrize("limb", [0., .5])
def test_moving_seeds_match_individual_geometry(q, limb):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from lcbinint_jax import binary_algebraic_boundary_batch_ffi as call
    from test_binary_caustic_enclosure import _independent_caustic

    separations = np.array([.7, .700001, 1., 1.000001, 2., 2.000001])
    sources = []
    for i, s in enumerate(separations):
        caustic = _independent_caustic(s, q)
        # Alternate a generic fold and a cusp; the source straddles the
        # caustic, so centre images alone need not cover all image components.
        z = caustic[173] if i % 2 == 0 else caustic[np.argmax(caustic.real)]
        sources.append(z + .0003 + .0001j)
    x = jnp.asarray(np.real(sources)); y = jnp.asarray(np.imag(sources))
    s = jnp.asarray(separations)
    def evaluate(xx, yy, ss):
        return call(xx, yy, ss, q, .003, limb_c=limb,
                    absolute_tolerance=1e-6, relative_tolerance=1e-6)
    moving = np.asarray(evaluate(x, y, s))
    individual = np.concatenate([np.asarray(evaluate(x[i:i+1], y[i:i+1], s[i]))
                                 for i in range(len(s))])
    assert np.isfinite(moving).all()
    np.testing.assert_allclose(moving, individual, rtol=5e-6, atol=5e-6)
    reverse = np.asarray(jax.grad(lambda xx: jnp.sum(evaluate(xx, y, s)))(x))
    separate = np.concatenate([np.asarray(jax.grad(
        lambda xx: jnp.sum(evaluate(xx, y[i:i+1], s[i])))(x[i:i+1]))
        for i in range(len(s))])
    assert np.isfinite(reverse).all()
    np.testing.assert_allclose(reverse, separate, rtol=5e-4, atol=5e-4)

import jax
import jax.numpy as jnp
import numpy as np

from lcbinint_jax.limb_darkening import combine_limb_darkening_moments


def test_moment_combination_matches_direct_formula():
    moments = jnp.asarray([3.1, 2.4, 2.7])
    rho = 0.2
    c = 0.4
    d = 0.1
    expected = ((1.0 - c - d) * moments[0] + c * moments[1] + d * moments[2]) / (
        np.pi * rho * rho * (1.0 - c / 3.0 - d / 5.0)
    )
    actual = combine_limb_darkening_moments(moments, rho, c, d)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14)


def test_limb_coefficients_and_radius_are_differentiable():
    moments = jnp.asarray([3.1, 2.4, 2.7])

    def value(parameters):
        rho, c, d = parameters
        return combine_limb_darkening_moments(moments, rho, c, d)

    parameters = jnp.asarray([0.2, 0.4, 0.1])
    gradient = jax.grad(value)(parameters)
    assert jnp.all(jnp.isfinite(gradient))

    step = 1.0e-6
    for index in range(parameters.size):
        direction = jnp.zeros_like(parameters).at[index].set(step)
        finite_difference = (
            value(parameters + direction) - value(parameters - direction)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            gradient[index], finite_difference, rtol=2.0e-8, atol=2.0e-8
        )

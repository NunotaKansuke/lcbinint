import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from lcbinint_jax.cell_moments import affine_cell_moments


@pytest.mark.parametrize("power_index,power", [(0, 0.0), (1, 0.5), (2, 0.25)])
@pytest.mark.parametrize(
    "phi_centre,gradient_x,gradient_y,cell_size",
    [
        (0.1, 1.0, 0.5, 0.2),
        (-0.03, -0.7, 1.3, 0.15),
        (0.02, 1.0e-12, 0.8, 0.1),
        (0.02, -0.6, 1.0e-12, 0.1),
    ],
)
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_affine_cell_moment_matches_numerical_quadrature(
    power_index, power, phi_centre, gradient_x, gradient_y, cell_size
):
    def integrand(y, x):
        phi = phi_centre + gradient_x * x + gradient_y * y
        return max(phi, 0.0) ** power if phi > 0.0 else 0.0

    half = 0.5 * cell_size
    expected = integrate.dblquad(
        integrand,
        -half,
        half,
        lambda _: -half,
        lambda _: half,
        epsabs=2.0e-10,
        epsrel=2.0e-10,
    )[0]
    actual = affine_cell_moments(phi_centre, gradient_x, gradient_y, cell_size)[
        power_index
    ]
    tolerance = 2.0e-5 if power == 0.0 else 2.0e-7
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_affine_cell_jvp_matches_finite_difference():
    def value(parameters):
        phi, gradient_x, gradient_y = parameters
        return affine_cell_moments(phi, gradient_x, gradient_y, 0.12)

    parameters = jnp.asarray([0.015, 0.9, -0.35])
    tangent = jnp.asarray([0.2, -0.1, 0.3])
    _, jvp = jax.jvp(value, (parameters,), (tangent,))
    step = 1.0e-6
    finite_difference = (
        value(parameters + step * tangent) - value(parameters - step * tangent)
    ) / (2.0 * step)
    np.testing.assert_allclose(jvp, finite_difference, rtol=5.0e-8, atol=5.0e-8)


def test_affine_cell_gradient_is_finite_at_axis_aligned_boundary():
    for gradients in ((0.0, 1.0), (1.0, 0.0)):
        derivative = jax.jacrev(
            lambda phi: affine_cell_moments(phi, gradients[0], gradients[1], 0.1)
        )(jnp.asarray(0.01))
        assert jnp.all(jnp.isfinite(derivative))

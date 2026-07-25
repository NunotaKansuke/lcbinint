import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import binary_inverse_ray_fixed_support


def _covering_tiles(cell_size=0.05, tile_size=8, extent=2.0):
    tile_width = cell_size * tile_size
    starts = np.arange(-extent, extent, tile_width)
    origins = np.asarray([(x, y) for y in starts for x in starts])
    return jnp.asarray(origins), jnp.ones(origins.shape[0], dtype=bool)


@pytest.fixture(scope="module")
def support():
    return _covering_tiles()


def _evaluate(parameters, support, kernel="real"):
    source_x, source_y, separation, mass_ratio, rho, c, d = parameters
    origins, mask = support
    return binary_inverse_ray_fixed_support(
        origins,
        mask,
        0.05,
        source_x,
        source_y,
        separation,
        mass_ratio,
        rho,
        c,
        d,
        tile_size=8,
        kernel=kernel,
    )


def test_real_and_complex_hot_kernels_agree(support):
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
    real_result = _evaluate(parameters, support, "real")
    complex_result = _evaluate(parameters, support, "complex")
    np.testing.assert_allclose(
        real_result.moments,
        complex_result.moments,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        real_result.magnification,
        complex_result.magnification,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert int(real_result.boundary_cells) > 0
    assert int(real_result.active_cells) > int(real_result.boundary_cells)


def test_value_and_grad_matches_directional_finite_difference(support):
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])

    def value(active_parameters):
        return _evaluate(active_parameters, support).magnification

    value_at_parameters, gradient = jax.value_and_grad(value)(parameters)
    assert jnp.isfinite(value_at_parameters)
    assert jnp.all(jnp.isfinite(gradient))

    tangent = jnp.asarray([0.2, -0.1, 0.05, 0.02, -0.03, 0.1, -0.05])
    reverse_directional = jnp.vdot(gradient, tangent)
    _, forward_directional = jax.jvp(value, (parameters,), (tangent,))
    np.testing.assert_allclose(
        reverse_directional,
        forward_directional,
        rtol=2.0e-11,
        atol=2.0e-11,
    )

    step = 1.0e-5
    finite_difference = (
        value(parameters + step * tangent) - value(parameters - step * tangent)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        forward_directional,
        finite_difference,
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_masked_tiles_do_not_contribute():
    origins, mask = _covering_tiles()
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.0, 0.0])
    full = _evaluate(parameters, (origins, mask))
    empty = _evaluate(parameters, (origins, jnp.zeros_like(mask)))
    assert full.magnification > 0.0
    assert empty.magnification == 0.0
    np.testing.assert_array_equal(empty.moments, jnp.zeros(3))

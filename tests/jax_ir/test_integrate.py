import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import (
    binary_inverse_ray_fixed_support,
    binary_inverse_ray_fixed_support_cpp,
    binary_inverse_ray_fixed_support_ffi,
)
from lcbinint_jax.integrate import (
    _phi_gradient_laplacian_complex,
    _phi_gradient_laplacian_real,
)


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


def _require_compiled_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "fixed_support_forward_ffi"):
        pytest.skip("lcbinint was built without JAX FFI headers")


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


def test_analytic_phi_laplacian_matches_complex_autodiff():
    arguments = (0.37, -0.42, 0.11, -0.07, 0.93, 0.013, 0.04)
    real = _phi_gradient_laplacian_real(*arguments)
    complex_autodiff = _phi_gradient_laplacian_complex(*arguments)
    np.testing.assert_allclose(
        real,
        complex_autodiff,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_boundary_capacity_must_cover_a_whole_tile(support):
    origins, mask = support
    with pytest.raises(ValueError, match="boundary_capacity"):
        binary_inverse_ray_fixed_support(
            origins,
            mask,
            0.05,
            0.2,
            0.1,
            1.2,
            0.1,
            0.2,
            tile_size=8,
            boundary_capacity=32,
        )


def test_adaptive_boundary_rule_tracks_full_four_by_four_rule(support):
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.3, 0.2])
    origins, mask = support
    adaptive = binary_inverse_ray_fixed_support(
        origins,
        mask,
        0.05,
        *parameters,
        tile_size=8,
        boundary_subdivision=0,
    )
    full = binary_inverse_ray_fixed_support(
        origins,
        mask,
        0.05,
        *parameters,
        tile_size=8,
        boundary_subdivision=4,
    )
    np.testing.assert_allclose(
        adaptive.magnification,
        full.magnification,
        rtol=3.0e-4,
        atol=3.0e-4,
    )


def test_adaptive_boundary_threshold_must_be_positive(support):
    origins, mask = support
    with pytest.raises(ValueError, match="boundary_adaptive_threshold"):
        binary_inverse_ray_fixed_support(
            origins,
            mask,
            0.05,
            0.2,
            0.1,
            1.2,
            0.1,
            0.2,
            tile_size=8,
            boundary_adaptive_threshold=0.0,
        )


@pytest.mark.parametrize(
    ("moment_mode", "boundary_subdivision"),
    (("uniform", 3), ("linear", 3), ("two_coefficient", 4)),
)
def test_cpp_forward_matches_jax_fixed_support(
    support, moment_mode, boundary_subdivision
):
    parameters = np.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
    origins, mask = support
    try:
        cpp = binary_inverse_ray_fixed_support_cpp(
            origins,
            mask,
            0.05,
            *parameters,
            tile_size=8,
            moment_mode=moment_mode,
            boundary_subdivision=boundary_subdivision,
        )
    except RuntimeError as error:
        pytest.skip(str(error))
    jax_result = binary_inverse_ray_fixed_support(
        origins,
        mask,
        0.05,
        *parameters,
        tile_size=8,
        moment_mode=moment_mode,
        boundary_subdivision=boundary_subdivision,
    )
    np.testing.assert_allclose(
        cpp.moments,
        jax_result.moments,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        cpp.magnification,
        jax_result.magnification,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert cpp.boundary_cells == int(jax_result.boundary_cells)
    assert cpp.active_cells == int(jax_result.active_cells)


@pytest.mark.parametrize(
    ("moment_mode", "boundary_subdivision"),
    (("uniform", 3), ("linear", 3), ("two_coefficient", 4)),
)
def test_ffi_forward_matches_jax_under_jit(support, moment_mode, boundary_subdivision):
    _require_compiled_ffi()
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
    origins, mask = support

    @jax.jit
    def evaluate(active_parameters):
        return binary_inverse_ray_fixed_support_ffi(
            origins,
            mask,
            0.05,
            *active_parameters,
            tile_size=8,
            moment_mode=moment_mode,
            boundary_subdivision=boundary_subdivision,
        )

    ffi_result = evaluate(parameters)
    jax_result = binary_inverse_ray_fixed_support(
        origins,
        mask,
        0.05,
        *parameters,
        tile_size=8,
        moment_mode=moment_mode,
        boundary_subdivision=boundary_subdivision,
    )
    np.testing.assert_allclose(
        ffi_result.moments,
        jax_result.moments,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        ffi_result.magnification,
        jax_result.magnification,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert int(ffi_result.boundary_cells) == int(jax_result.boundary_cells)
    assert int(ffi_result.active_cells) == int(jax_result.active_cells)


@pytest.mark.parametrize(
    ("moment_mode", "boundary_subdivision"),
    (("uniform", 3), ("linear", 3), ("two_coefficient", 4)),
)
def test_ffi_analytic_jvp_and_grad_match_jax(
    support, moment_mode, boundary_subdivision
):
    _require_compiled_ffi()
    origins, mask = support
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
    tangent = jnp.asarray([0.2, -0.1, 0.05, 0.02, -0.03, 0.1, -0.05])

    def evaluate(function, active_parameters):
        result = function(
            origins,
            mask,
            0.05,
            *active_parameters,
            tile_size=8,
            moment_mode=moment_mode,
            boundary_subdivision=boundary_subdivision,
        )
        return jnp.concatenate(
            (jnp.reshape(result.magnification, (1,)), result.moments)
        )

    jax_value, jax_tangent = jax.jvp(
        lambda active: evaluate(binary_inverse_ray_fixed_support, active),
        (parameters,),
        (tangent,),
    )
    ffi_value, ffi_tangent = jax.jit(
        lambda active, direction: jax.jvp(
            lambda values: evaluate(binary_inverse_ray_fixed_support_ffi, values),
            (active,),
            (direction,),
        )
    )(parameters, tangent)
    np.testing.assert_allclose(
        ffi_value,
        jax_value,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        ffi_tangent,
        jax_tangent,
        rtol=2.0e-9,
        atol=2.0e-9,
    )

    jax_gradient = jax.grad(
        lambda active: evaluate(binary_inverse_ray_fixed_support, active)[0]
    )(parameters)
    ffi_gradient = jax.jit(
        jax.grad(
            lambda active: evaluate(binary_inverse_ray_fixed_support_ffi, active)[0]
        )
    )(parameters)
    np.testing.assert_allclose(
        ffi_gradient,
        jax_gradient,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_ffi_forward_supports_sequential_vmap(support):
    _require_compiled_ffi()
    origins, mask = support

    def evaluate(source_x):
        return binary_inverse_ray_fixed_support_ffi(
            origins,
            mask,
            0.05,
            source_x,
            0.1,
            1.2,
            0.1,
            0.2,
            0.4,
            0.0,
            moment_mode="linear",
            boundary_subdivision=3,
        ).magnification

    source_x = jnp.asarray([0.19, 0.21])
    ffi_values = jax.jit(jax.vmap(evaluate))(source_x)
    jax_values = jax.vmap(
        lambda x: binary_inverse_ray_fixed_support(
            origins,
            mask,
            0.05,
            x,
            0.1,
            1.2,
            0.1,
            0.2,
            0.4,
            0.0,
            moment_mode="linear",
            boundary_subdivision=3,
        ).magnification
    )(source_x)
    np.testing.assert_allclose(
        ffi_values,
        jax_values,
        rtol=2.0e-11,
        atol=2.0e-11,
    )

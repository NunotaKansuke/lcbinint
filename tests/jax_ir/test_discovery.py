"""Regression tests for the FFI-only binary Cartesian inverse-ray path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import (
    binary_inverse_ray,
    binary_inverse_ray_linear,
    binary_inverse_ray_uniform,
    discover_binary_macro_tiles_ffi,
)


def _require_cartesian_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "cartesian_epoch_forward_ffi"):
        pytest.skip("lcbinint was built without Cartesian epoch FFI support")


def test_public_cartesian_inverse_ray_uses_ffi_for_legacy_backend_spellings():
    _require_cartesian_ffi()
    parameters = (0.2, 0.1, 1.2, 0.1, 0.05, 0.4, 0.0)
    options = dict(resolution=16, tile_size=8, tile_capacity=512, limb_samples=8)
    automatic = binary_inverse_ray(*parameters, **options)
    legacy = binary_inverse_ray(
        *parameters,
        cartesian_backend="jax",
        root_backend="jax",
        **options,
    )
    np.testing.assert_allclose(legacy.magnification, automatic.magnification, rtol=0.0)
    assert automatic.magnification.dtype == jnp.float64
    assert bool(automatic.support_valid)


def test_public_cartesian_value_and_gradient_are_finite():
    _require_cartesian_ffi()

    def evaluate(source_x):
        return binary_inverse_ray(
            source_x,
            0.02,
            1.0,
            0.2,
            0.01,
            0.5,
            0.0,
            resolution=24,
            tile_size=8,
            tile_capacity=256,
            limb_samples=8,
        ).magnification

    value, gradient = jax.value_and_grad(evaluate)(jnp.asarray(0.08))
    step = 2.0e-6
    finite_difference = (evaluate(0.08 + step) - evaluate(0.08 - step)) / (2.0 * step)
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, rtol=6.0e-2, atol=2.0e-2)


def test_uniform_and_linear_specializations_match_general_kernel():
    _require_cartesian_ffi()
    options = dict(resolution=16, tile_size=8, tile_capacity=128, limb_samples=8)
    uniform = binary_inverse_ray_uniform(0.2, 0.1, 1.2, 0.1, 0.05, **options)
    uniform_general = binary_inverse_ray(0.2, 0.1, 1.2, 0.1, 0.05, **options)
    linear = binary_inverse_ray_linear(0.2, 0.1, 1.2, 0.1, 0.05, 0.4, **options)
    linear_general = binary_inverse_ray(0.2, 0.1, 1.2, 0.1, 0.05, 0.4, **options)
    np.testing.assert_allclose(uniform.magnification, uniform_general.magnification)
    np.testing.assert_allclose(linear.magnification, linear_general.magnification)


def test_low_level_discovery_is_also_seeded_by_ffi_roots():
    _require_cartesian_ffi()
    result = discover_binary_macro_tiles_ffi(
        0.2,
        0.1,
        1.2,
        0.1,
        0.05,
        0.05 / 16,
        tile_size=8,
        tile_capacity=128,
        limb_samples=8,
        root_backend="jax",  # compatibility spelling; implementation is FFI-only
    )
    assert not bool(result.root_failure)
    assert int(result.active_count) > 0

"""Focused tests for the retained low-level fixed-support FFI primitive."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import binary_inverse_ray_fixed_support_ffi


def _require_fixed_support_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "fixed_support_forward_ffi"):
        pytest.skip("lcbinint was built without fixed-support FFI support")


def _support():
    starts = np.arange(-2.0, 2.0, 0.4)
    origins = jnp.asarray([(x, y) for y in starts for x in starts])
    return origins, jnp.ones(origins.shape[0], dtype=bool)


def test_fixed_support_ffi_value_and_gradient_are_finite():
    _require_fixed_support_ffi()
    origins, mask = _support()

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
            tile_size=8,
        ).magnification

    value, gradient = jax.value_and_grad(evaluate)(jnp.asarray(0.2))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_fixed_support_ffi_supports_sequential_vmap():
    _require_fixed_support_ffi()
    origins, mask = _support()

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
            tile_size=8,
            moment_mode="uniform",
        ).magnification

    result = jax.jit(jax.vmap(evaluate))(jnp.asarray([0.19, 0.2, 0.21]))
    assert result.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(result)))

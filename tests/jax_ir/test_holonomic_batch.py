"""Opt-in D14/holonomic batch FFI checks.

The normal LCBININT dispatcher is intentionally outside these tests.  The
route is exposed as an explicit research ABI until a trajectory-state ABI and
whole-epoch parity benchmark justify a production router change.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lcbinint_jax.cpp_backend import (
    binary_holonomic_batch_value_jacobian_ffi,
    cpp_holonomic_batch_ffi_available,
)


jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    not cpp_holonomic_batch_ffi_available(),
    reason="opt-in D14/holonomic batch FFI is not built",
)


def _inputs():
    return (
        jnp.asarray((0.12, 0.21), dtype=jnp.float64),
        jnp.asarray((-0.08, -0.03), dtype=jnp.float64),
        jnp.asarray((1.2, 1.2), dtype=jnp.float64),
        jnp.asarray(0.01, dtype=jnp.float64),
        jnp.asarray(0.28, dtype=jnp.float64),
    )


def test_explicit_d14_batch_returns_value_jacobian_and_support():
    result = binary_holonomic_batch_value_jacobian_ffi(*_inputs(), n_r=16)
    assert np.asarray(result.support_valid).all()
    assert np.isfinite(np.asarray(result.magnification)).all()
    assert np.isfinite(np.asarray(result.jacobian)).all()
    assert np.asarray(result.jacobian).shape == (2, 5)


def test_explicit_d14_custom_jvp_uses_the_fused_geometry_jacobian():
    source_x, source_y, separation, rho, q = _inputs()

    def value(x):
        return binary_holonomic_batch_value_jacobian_ffi(
            x, source_y, separation, rho, q, n_r=16
        ).magnification

    result = binary_holonomic_batch_value_jacobian_ffi(
        source_x, source_y, separation, rho, q, n_r=16
    )
    _, tangent = jax.jvp(value, (source_x,), (jnp.ones_like(source_x),))
    np.testing.assert_allclose(
        np.asarray(tangent),
        np.asarray(result.jacobian)[:, 0],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_quadratic_limb_darkening_is_fail_closed():
    result = binary_holonomic_batch_value_jacobian_ffi(
        *_inputs(), limb_d=0.1, n_r=16
    )
    assert not np.asarray(result.support_valid).any()
    assert not np.isfinite(np.asarray(result.magnification)).any()
    assert not np.isfinite(np.asarray(result.jacobian)).any()

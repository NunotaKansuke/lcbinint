import math

import jax
import jax.numpy as jnp
import numpy as np

from lcbinint_jax import (
    binary_hexadecapole,
    binary_magnification_auto,
    binary_point_source_magnification,
)


jax.config.update("jax_enable_x64", True)


def test_point_source_implicit_gradient_matches_finite_difference():
    def magnification(source_x):
        return binary_point_source_magnification(
            source_x, 0.4, 1.4, 1.0e-3
        ).magnification

    source_x = 0.3
    step = 1.0e-6
    finite_difference = (
        magnification(source_x + step) - magnification(source_x - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        jax.grad(magnification)(source_x),
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_hexadecapole_matches_native_formula_and_has_stable_gradient():
    parameters = (0.3, 0.4, 1.4, 1.0e-3, 0.01)

    def magnification(source_x):
        return binary_hexadecapole(
            source_x,
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
            0.4,
            0.0,
        ).magnification

    result = binary_hexadecapole(*parameters, 0.4, 0.0)
    assert bool(result.topology_stable)
    assert not bool(result.root_failure)
    np.testing.assert_allclose(result.magnification, 2.17750899, rtol=2.0e-9)

    step = 1.0e-6
    finite_difference = (
        magnification(parameters[0] + step)
        - magnification(parameters[0] - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        jax.grad(magnification)(parameters[0]),
        finite_difference,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_hybrid_uses_multipole_far_away_and_inverse_rays_at_cusp():
    far = binary_magnification_auto(
        0.3,
        0.4,
        1.4,
        1.0e-3,
        0.01,
        0.4,
        0.0,
        moment_mode="linear",
    )
    assert int(far.method) == 0
    assert bool(far.used_multipole)
    assert not bool(far.used_polar)

    cusp = binary_magnification_auto(
        0.653,
        0.0,
        1.2,
        0.1,
        0.02,
        0.4,
        0.0,
        resolution=64,
        tile_capacity=1024,
        moment_mode="linear",
    )
    assert int(cusp.method) == 1
    assert not bool(cusp.used_multipole)
    assert not bool(cusp.used_polar)
    assert bool(cusp.support_valid)


def test_hybrid_far_field_gradient_is_finite():
    def magnification(parameters):
        return binary_magnification_auto(
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            0.01,
            0.4,
            0.0,
            moment_mode="linear",
        ).magnification

    gradient = jax.grad(magnification)(
        jnp.asarray((0.3, 0.4, 1.4, 1.0e-3))
    )
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_hybrid_does_not_silently_fallback_to_polar_on_overflow():
    result = binary_magnification_auto(
        0.653,
        0.0,
        1.2,
        0.1,
        0.02,
        0.4,
        0.0,
        resolution=64,
        tile_capacity=1,
        moment_mode="linear",
    )
    assert int(result.method) == 1
    assert not bool(result.support_valid)
    assert math.isnan(float(result.magnification))


def test_hybrid_keeps_calibrated_tiny_high_magnification_polar_path():
    separation = 0.95
    mass_ratio = 0.01
    source_radius = 0.005
    u0 = -0.001
    alpha = 0.5
    epoch = 0.004
    source_x = epoch * math.cos(alpha) - u0 * math.sin(alpha)
    source_y = epoch * math.sin(alpha) + u0 * math.cos(alpha)
    result = binary_magnification_auto(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        0.4,
        0.0,
        polar_resolution=64,
        polar_angular_bins=4096,
        polar_radial_capacity=256,
        moment_mode="linear",
    )
    assert int(result.method) == 2
    assert bool(result.support_valid)
    np.testing.assert_allclose(result.magnification, 95.43373464, rtol=1.0e-9)

import math

import jax
import numpy as np
import pytest

from lcbinint_jax import (
    binary_inverse_ray,
    binary_inverse_ray_auto,
    binary_inverse_ray_polar,
    binary_inverse_ray_polar_ffi,
)


def test_polar_value_and_gradient_converge_to_cartesian():
    polar_options = dict(
        resolution=128,
        angular_bins=4096,
        radial_capacity=512,
        band_capacity=4,
        limb_samples=64,
        angular_chunk_size=32,
    )

    def polar(source_x):
        return binary_inverse_ray_polar(
            source_x,
            0.1,
            1.2,
            0.1,
            0.2,
            0.4,
            0.1,
            **polar_options,
        ).magnification

    def cartesian(source_x):
        return binary_inverse_ray(
            source_x,
            0.1,
            1.2,
            0.1,
            0.2,
            0.4,
            0.1,
            resolution=128,
            tile_size=16,
            tile_capacity=4096,
            limb_samples=64,
        ).magnification

    np.testing.assert_allclose(polar(0.2), cartesian(0.2), rtol=8.0e-4)
    np.testing.assert_allclose(
        jax.grad(polar)(0.2),
        jax.grad(cartesian)(0.2),
        rtol=1.0e-3,
        atol=1.0e-3,
    )


def test_auto_dispatches_regular_to_cartesian_and_tiny_high_mag_to_polar():
    common = dict(
        resolution=32,
        tile_capacity=512,
        limb_samples=32,
        polar_resolution=32,
        polar_angular_bins=256,
        polar_radial_capacity=128,
    )
    regular = binary_inverse_ray_auto(
        0.2,
        0.1,
        1.2,
        0.1,
        0.2,
        0.4,
        0.0,
        moment_mode="linear",
        **common,
    )
    assert not bool(regular.used_polar)
    assert bool(regular.support_valid)

    separation = 0.95
    mass_ratio = 0.01
    source_radius = 0.005
    u0 = -0.001
    alpha = 0.5
    time = 0.004
    source_x = time * math.cos(alpha) - u0 * math.sin(alpha)
    source_y = time * math.sin(alpha) + u0 * math.cos(alpha)
    high = binary_inverse_ray_auto(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        0.4,
        0.0,
        moment_mode="linear",
        **common,
    )
    assert bool(high.used_polar)
    assert bool(high.support_valid)
    assert math.isfinite(float(high.magnification))


def test_auto_uses_polar_when_cartesian_discovery_overflows():
    result = binary_inverse_ray_auto(
        0.2,
        0.1,
        1.2,
        0.1,
        0.2,
        0.4,
        0.0,
        resolution=32,
        tile_capacity=1,
        limb_samples=32,
        polar_resolution=32,
        polar_angular_bins=256,
        polar_radial_capacity=128,
        polar_magnification_threshold=1.0e12,
        moment_mode="linear",
    )

    assert bool(result.used_polar)
    assert bool(result.support_valid)


def test_polar_ffi_accepts_repeated_root_at_exact_fold():
    result = binary_inverse_ray_polar_ffi(
        0.06611188225495068,
        0.1549319030240759,
        0.9,
        0.1,
        0.01,
        resolution=64,
        angular_bins=2048,
        radial_capacity=512,
        band_capacity=8,
        limb_samples=64,
        boundary_capacity=8192,
    )

    assert bool(result.support_valid)
    assert not bool(result.root_failure)
    assert np.isfinite(float(result.magnification))


def test_polar_high_magnification_matches_native_lcbinint():
    lcbinint = pytest.importorskip("lcbinint", exc_type=ImportError)
    separation = 0.95
    mass_ratio = 0.01
    source_radius = 0.005
    u0 = -0.001
    alpha = 0.5
    time = 0.004
    source_x = time * math.cos(alpha) - u0 * math.sin(alpha)
    source_y = time * math.sin(alpha) + u0 * math.cos(alpha)
    native = lcbinint.binary_ray_shooting(
        source_x,
        source_y,
        s=separation,
        q=mass_ratio,
        rho=source_radius,
        limb_darkening=lcbinint.LimbDarkening.linear(0.4),
        options=lcbinint.Options(
            nbin=256,
            inverse_ray_grid="polar",
            coordinates="center_of_mass",
        ),
    )
    result = binary_inverse_ray_polar(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        0.4,
        0.0,
        resolution=64,
        angular_bins=2048,
        radial_capacity=256,
        band_capacity=4,
        limb_samples=64,
        angular_chunk_size=32,
        moment_mode="linear",
    )

    assert bool(result.support_valid)
    np.testing.assert_allclose(result.magnification, native, rtol=5.0e-4)


def test_polar_flood_ffi_converges_to_native_lcbinint():
    lcbinint = pytest.importorskip("lcbinint", exc_type=ImportError)
    separation = 0.95
    mass_ratio = 0.01
    source_radius = 0.005
    u0 = -0.001
    alpha = 0.5
    time = 0.004
    source_x = time * math.cos(alpha) - u0 * math.sin(alpha)
    source_y = time * math.sin(alpha) + u0 * math.cos(alpha)
    native = lcbinint.binary_ray_shooting(
        source_x,
        source_y,
        s=separation,
        q=mass_ratio,
        rho=source_radius,
        limb_darkening=lcbinint.LimbDarkening.linear(0.4),
        options=lcbinint.Options(
            nbin=256,
            inverse_ray_grid="polar",
            coordinates="center_of_mass",
        ),
    )

    def evaluate(resolution, angular_bins):
        return binary_inverse_ray_polar_ffi(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            0.4,
            0.0,
            resolution=resolution,
            angular_bins=angular_bins,
            radial_capacity=512,
            band_capacity=8,
            limb_samples=64,
            angular_chunk_size=256,
            boundary_capacity=8192,
            boundary_subdivision=2,
            moment_mode="linear",
        )

    coarse = evaluate(64, 2048)
    fine = evaluate(256, 8192)
    assert bool(coarse.support_valid)
    assert bool(fine.support_valid)
    assert abs(float(fine.magnification) - native) < abs(
        float(coarse.magnification) - native
    )
    np.testing.assert_allclose(fine.magnification, native, rtol=3.0e-5)


def test_public_polar_is_ffi_and_has_all_parameter_gradients():
    separation = 0.95
    mass_ratio = 0.01
    source_radius = 0.005
    u0 = -0.001
    alpha = 0.5
    time = 0.004
    parameters = np.asarray(
        (
            time * math.cos(alpha) - u0 * math.sin(alpha),
            time * math.sin(alpha) + u0 * math.cos(alpha),
            separation,
            mass_ratio,
            source_radius,
            0.3,
            0.2,
        )
    )
    options = {
        "resolution": 64,
        "angular_bins": 1024,
        "radial_capacity": 128,
        "band_capacity": 4,
        "limb_samples": 32,
        "angular_chunk_size": 256,
        "boundary_capacity": 1024,
        "boundary_subdivision": 2,
        "moment_mode": "two_coefficient",
    }

    assert binary_inverse_ray_polar is binary_inverse_ray_polar_ffi

    def evaluate(active):
        result = binary_inverse_ray_polar(*active, **options)
        return result.magnification, (
            result.moments,
            result.boundary_cells,
            result.active_cells,
            result.tile_count,
            result.support_valid,
        )

    (value, aux), gradient = jax.value_and_grad(
        evaluate,
        has_aux=True,
    )(parameters)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert bool(aux[4])
    assert int(aux[3]) > 0
    assert gradient[5] != 0.0
    assert gradient[6] != 0.0
    step = 1.0e-7
    plus = parameters.copy()
    minus = parameters.copy()
    plus[0] += step
    minus[0] -= step
    finite_difference = (
        float(evaluate(plus)[0])
        - float(evaluate(minus)[0])
    ) / (2.0 * step)
    np.testing.assert_allclose(
        gradient[0], finite_difference, rtol=2.0e-3
    )

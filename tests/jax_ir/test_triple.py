import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcbinint
from lcbinint_jax import (
    cpp_triple_cartesian_epoch_ffi_available,
    cpp_triple_hexadecapole_batch_ffi_available,
    cpp_triple_point_batch_ffi_available,
    triple_hexadecapole_batch_ffi,
    triple_caustic_distance_batch_ffi,
    triple_inverse_ray_adaptive,
    triple_inverse_ray_batch,
    triple_inverse_ray_dense,
    triple_inverse_ray_polar_batch_ffi,
    triple_lens_geometry,
    triple_lens_map_and_derivatives_real,
    triple_magnification_auto,
    triple_magnification_batch,
    triple_point_source_batch_ffi,
    triple_source_plane_quadrature,
)
from lcbinint_jax.lens import binary_lens_map_and_derivatives_real

jax.config.update("jax_enable_x64", True)


def test_triple_geometry_is_center_of_mass_centered():
    geometry = triple_lens_geometry(1.0, 0.1, 0.03, 0.7, 0.8)
    np.testing.assert_allclose(jnp.sum(geometry.masses), 1.0, atol=1.0e-15)
    np.testing.assert_allclose(
        jnp.sum(geometry.positions * geometry.masses[:, None], axis=0),
        0.0,
        atol=1.0e-15,
    )


def test_zero_tertiary_mass_reduces_to_binary_lens_map():
    image_x = jnp.asarray((-0.8, 0.2, 1.1))
    image_y = jnp.asarray((0.4, -0.7, 0.3))
    triple_geometry = triple_lens_geometry(1.2, 0.1, 0.0, 0.7, 0.8)
    triple = triple_lens_map_and_derivatives_real(
        image_x, image_y, triple_geometry
    )
    binary = binary_lens_map_and_derivatives_real(
        image_x, image_y, 1.2, 0.1
    )
    for triple_value, binary_value in zip(triple, binary):
        np.testing.assert_allclose(triple_value, binary_value, atol=3.0e-15)


def test_dense_triple_inverse_ray_matches_native_uniform_source():
    parameters = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.3,
        "alpha": 0.0,
        "s": 1.0,
        "q": 0.1,
        "q2": 0.03,
        "sep2": 0.7,
        "ang": 0.8,
        "rho": 0.2,
    }
    native = lcbinint.LightCurve(
        model=lcbinint.Model(lens="triple"),
        options=lcbinint.Options(
            coordinates="original",
            nbin=200,
            inverse_ray_grid="cartesian",
        ),
    )([0.2], parameters)[0]
    actual = triple_inverse_ray_dense(
        0.2,
        0.3,
        1.0,
        0.1,
        0.03,
        0.7,
        0.8,
        0.2,
        resolution=512,
        image_extent=3.0,
        moment_mode="uniform",
    )
    assert bool(actual.support_valid)
    assert int(actual.contributing_cells) > 0
    assert int(actual.boundary_cells) > 0
    np.testing.assert_allclose(actual.magnification, native, rtol=3.0e-3)


def test_dense_triple_inverse_ray_has_parameter_gradients():
    def loss(source_x, tertiary_mass_ratio, tertiary_angle):
        return triple_inverse_ray_dense(
            source_x,
            0.3,
            1.0,
            0.1,
            tertiary_mass_ratio,
            0.7,
            tertiary_angle,
            0.2,
            resolution=512,
            image_extent=3.0,
            moment_mode="uniform",
        ).magnification

    value, gradients = jax.value_and_grad(loss, argnums=(0, 1, 2))(
        0.2, 0.03, 0.8
    )
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(gradients))))
    step = 3.0e-5
    finite_difference = (
        loss(0.2 + step, 0.03, 0.8) - loss(0.2 - step, 0.03, 0.8)
    ) / (2.0 * step)
    np.testing.assert_allclose(gradients[0], finite_difference, rtol=1.0e-3)


def test_dense_triple_inverse_ray_rejects_insufficient_support():
    result = triple_inverse_ray_dense(
        0.2,
        0.3,
        1.0,
        0.1,
        0.03,
        0.7,
        0.8,
        0.2,
        resolution=128,
        image_extent=0.5,
        moment_mode="uniform",
    )
    assert not bool(result.support_valid)


@pytest.mark.skipif(
    not cpp_triple_cartesian_epoch_ffi_available(),
    reason="triple Cartesian FFI is unavailable",
)
def test_adaptive_triple_matches_native_uniform_source():
    parameters = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": 0.3,
        "alpha": 0.0,
        "s": 1.0,
        "q": 0.1,
        "q2": 0.03,
        "sep2": 0.7,
        "ang": 0.8,
        "rho": 0.2,
    }
    native = lcbinint.LightCurve(
        model=lcbinint.Model(lens="triple"),
        options=lcbinint.Options(
            coordinates="original",
            nbin=400,
            inverse_ray_grid="cartesian",
        ),
    )([0.2], parameters)[0]
    actual = triple_inverse_ray_adaptive(
        0.2,
        0.3,
        1.0,
        0.1,
        0.03,
        0.7,
        0.8,
        0.2,
        moment_mode="uniform",
    )
    assert bool(actual.support_valid)
    assert not bool(actual.root_failure)
    assert int(actual.visited_tiles) > 0
    np.testing.assert_allclose(actual.magnification, native, rtol=2.0e-4)


@pytest.mark.skipif(
    not cpp_triple_cartesian_epoch_ffi_available(),
    reason="triple Cartesian FFI is unavailable",
)
def test_adaptive_triple_analytic_jacobian_matches_finite_difference():
    parameters = jnp.asarray(
        (0.2, 0.3, 1.0, 0.1, 0.03, 0.7, 0.8, 0.2, 0.3, 0.2)
    )

    def magnification(values):
        return triple_inverse_ray_adaptive(
            *values,
            resolution=96,
            moment_mode="two_coefficient",
        ).magnification

    value, gradient = jax.value_and_grad(magnification)(parameters)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    # The custom JVP intentionally freezes discovered support and cell size.
    # A sufficiently local difference therefore checks its smooth branch;
    # rho is excluded because changing rho also changes the stopped grid size.
    step = 3.0e-7
    for index in (0, 2, 4, 6, 8, 9):
        plus = parameters.at[index].add(step)
        minus = parameters.at[index].add(-step)
        finite_difference = (
            magnification(plus) - magnification(minus)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            gradient[index], finite_difference, rtol=6.0e-3, atol=2.0e-5
        )


@pytest.mark.skipif(
    not cpp_triple_cartesian_epoch_ffi_available(),
    reason="triple Cartesian FFI is unavailable",
)
def test_triple_batch_matches_scalar_values_and_gradients():
    source_x = jnp.linspace(-0.25, 0.25, 8)
    source_y = jnp.full_like(source_x, 0.3)

    def batch_loss(separation):
        return jnp.sum(
            triple_inverse_ray_batch(
                source_x,
                source_y,
                separation,
                0.1,
                0.03,
                0.7,
                0.8,
                0.08,
                moment_mode="uniform",
                resolution=48,
            ).magnification
        )

    def scalar_loss(separation):
        values = jax.vmap(
            lambda x, y: triple_inverse_ray_adaptive(
                x,
                y,
                separation,
                0.1,
                0.03,
                0.7,
                0.8,
                0.08,
                moment_mode="uniform",
                resolution=48,
            ).magnification
        )(source_x, source_y)
        return jnp.sum(values)

    batch_value, batch_gradient = jax.value_and_grad(batch_loss)(1.0)
    scalar_value, scalar_gradient = jax.value_and_grad(scalar_loss)(1.0)
    np.testing.assert_allclose(batch_value, scalar_value, rtol=2.0e-13)
    np.testing.assert_allclose(batch_gradient, scalar_gradient, rtol=2.0e-12)


@pytest.mark.skipif(
    not cpp_triple_point_batch_ffi_available(),
    reason="triple point-source FFI is unavailable",
)
def test_triple_point_source_analytic_jacobian_matches_finite_difference():
    parameters = jnp.asarray((3.0, 2.0, 1.0, 0.1, 0.03, 0.7, 0.8))

    def magnification(values):
        return triple_point_source_batch_ffi(
            values[:1], values[1:2], *values[2:]
        ).magnification[0]

    value, gradient = jax.value_and_grad(magnification)(parameters)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    for index in range(parameters.size):
        step = 1.0e-5
        finite_difference = (
            magnification(parameters.at[index].add(step))
            - magnification(parameters.at[index].add(-step))
        ) / (2.0 * step)
        np.testing.assert_allclose(
            gradient[index], finite_difference, rtol=2.0e-5, atol=2.0e-8
        )


@pytest.mark.skipif(
    not cpp_triple_hexadecapole_batch_ffi_available(),
    reason="triple hexadecapole FFI is unavailable",
)
def test_triple_hexadecapole_analytic_jacobian_matches_finite_difference():
    parameters = jnp.asarray(
        (-0.8, 0.6, 1.0, 0.1, 0.03, 0.7, 0.8, 0.03, 0.3, 0.2)
    )

    def magnification(values):
        return triple_hexadecapole_batch_ffi(
            values[:1], values[1:2], *values[2:]
        ).magnification[0]

    value, gradient = jax.value_and_grad(magnification)(parameters)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    for index in range(parameters.size):
        step = 1.0e-5
        finite_difference = (
            magnification(parameters.at[index].add(step))
            - magnification(parameters.at[index].add(-step))
        ) / (2.0 * step)
        np.testing.assert_allclose(
            gradient[index], finite_difference, rtol=2.0e-4, atol=2.0e-7
        )


@pytest.mark.skipif(
    not (
        cpp_triple_point_batch_ffi_available()
        and cpp_triple_hexadecapole_batch_ffi_available()
        and cpp_triple_cartesian_epoch_ffi_available()
    ),
    reason="the complete triple dispatcher backend is unavailable",
)
def test_triple_dispatcher_uses_point_hexadecapole_and_inverse_ray():
    result = triple_magnification_batch(
        jnp.asarray((3.0, -0.8, 0.1)),
        jnp.asarray((2.0, 0.6, 0.3)),
        1.0,
        0.1,
        0.03,
        0.7,
        0.8,
        0.03,
        moment_mode="uniform",
    )
    np.testing.assert_array_equal(result.method, (0, 1, 4))
    assert bool(jnp.all(jnp.isfinite(result.magnification)))
    assert bool(jnp.all(result.support_valid))


@pytest.mark.skipif(
    not cpp_triple_cartesian_epoch_ffi_available(),
    reason="triple Cartesian FFI is unavailable",
)
def test_triple_dispatcher_uses_caustic_clear_high_magnification_polar():
    result = triple_magnification_auto(
        -0.046038516588439035,
        0.025585304221408988,
        1.0,
        0.1,
        1.0e-5,
        1.0,
        np.pi / 2.0,
        1.0e-4,
        moment_mode="uniform",
    )
    assert int(result.method) == 3
    assert not bool(result.used_multipole)
    assert bool(result.support_valid)


def test_triple_polar_matches_native_and_has_all_parameter_gradients():
    parameters = jnp.asarray(
        (
            -0.046038516588439035,
            0.025585304221408988,
            1.0,
            0.1,
            1.0e-5,
            1.0,
            np.pi / 2.0,
            1.0e-4,
            0.5,
            0.0,
        )
    )

    def magnification(values):
        return triple_inverse_ray_polar_batch_ffi(
            values[:1],
            values[1:2],
            *values[2:],
            resolution=64,
            limb_samples=64,
        ).magnification[0]

    actual, gradient = jax.value_and_grad(magnification)(parameters)
    assert float(actual) == pytest.approx(152.7134857937064, rel=1.0e-4)
    reference_gradient = np.asarray(
        (
            -2.01793265e5,
            5.84961956e4,
            -1.84630516e4,
            -9.65335234e4,
            -1.01124116e5,
            -5.67472490e-1,
            -1.15738797,
            2.07718742e4,
            -2.15022440e-1,
            -1.40940954e-1,
        )
    )
    np.testing.assert_allclose(
        gradient, reference_gradient, rtol=3.0e-3, atol=3.0e-4
    )


def test_triple_crossing_polar_uses_image_plane_boundary_gradient():
    parameters = jnp.asarray(
        (
            -0.04665563087,
            0.07266183991,
            0.9,
            0.1,
            0.003,
            1.5,
            1.0,
            0.01,
            0.45,
            0.0,
        )
    )

    def magnification(values):
        return triple_magnification_auto(
            *values, moment_mode="linear"
        ).magnification

    result = triple_magnification_auto(
        *parameters, moment_mode="linear"
    )
    actual, gradient = jax.value_and_grad(magnification)(parameters)
    assert int(result.method) == 3
    assert bool(result.support_valid)
    assert int(result.gradient_resolution) == 256
    assert not bool(result.gradient_extrapolated)
    assert float(actual) == pytest.approx(20.49171, rel=2.0e-4)
    # Independent resolution-256 Cartesian image-plane Jacobian.  The former
    # fixed source-plane point-source JVP missed the caustic/topology boundary
    # term here and was wrong by up to four orders of magnitude.
    reference_gradient = np.asarray(
        (
            1098.873,
            288.655,
            73.107,
            30.060,
            2630.001,
            7.537,
            -17.589,
            -702.800,
            1.00928,
            0.0,
        )
    )
    np.testing.assert_allclose(
        gradient, reference_gradient, rtol=1.5e-2, atol=2.0e-3
    )


def test_triple_extreme_polar_gradient_uses_capacity_safe_extrapolation():
    parameters = jnp.asarray(
        (
            -7.367712711025054e-6,
            -1.1825060697610411e-5,
            0.3,
            1.0e-5,
            1.0e-6,
            0.08,
            0.0,
            1.0e-5,
            0.8,
            0.0,
        )
    )

    def magnification(values):
        return triple_inverse_ray_polar_batch_ffi(
            values[:1],
            values[1:2],
            *values[2:],
            resolution=64,
            moment_mode="linear",
            gradient_backend="image_plane",
        ).magnification[0]

    actual, gradient = jax.value_and_grad(magnification)(parameters)
    dispatched = triple_magnification_auto(
        *parameters, moment_mode="linear"
    )
    assert float(actual) == pytest.approx(79478.09735, rel=2.0e-5)
    assert int(dispatched.method) == 3
    assert int(dispatched.gradient_resolution) == 64
    assert bool(dispatched.gradient_extrapolated)
    # Independent native polar finite differences.  A resolution-128
    # derivative would exceed the 50M-cell support bound at this epoch; the
    # production JVP instead extrapolates the valid 48/64 Jacobian pair.
    indices = np.asarray((0, 1, 2, 3, 4, 7))
    reference_gradient = np.asarray(
        (
            3.5779071955e9,
            5.3756113189e9,
            2.5986868166e4,
            2.1680823340e8,
            2.5844708737e8,
            9.5006708611e8,
        )
    )
    np.testing.assert_allclose(
        np.asarray(gradient)[indices],
        reference_gradient,
        rtol=5.0e-2,
        atol=1.0e-3,
    )


@pytest.mark.parametrize(
    ("parameters", "reference_source_x_gradient"),
    (
        (
            (
                8.9854485232332e-6,
                1.721829384078528e-5,
                0.3,
                1.0e-5,
                1.0e-6,
                0.08,
                0.0,
                1.0e-5,
            ),
            -1.4116476900e9,
        ),
        (
            (
                -3.0175997497641854e-5,
                -1.2758675656857664e-5,
                1.3,
                1.0e-5,
                1.0e-5,
                0.08,
                np.pi / 2.0,
                1.0e-5,
            ),
            5.8951009015e8,
        ),
    ),
)
def test_triple_extreme_polar_held_out_source_gradient(
    parameters, reference_source_x_gradient
):
    values = tuple(jnp.asarray(value) for value in parameters)

    def magnification(source_x):
        return triple_inverse_ray_polar_batch_ffi(
            jnp.reshape(source_x, (1,)),
            jnp.reshape(values[1], (1,)),
            *values[2:],
            0.0,
            0.0,
            resolution=64,
            moment_mode="uniform",
            gradient_backend="image_plane",
        ).magnification[0]

    gradient = jax.grad(magnification)(values[0])
    assert float(gradient) == pytest.approx(
        reference_source_x_gradient, rel=3.0e-2
    )


def test_triple_caustic_distance_routes_known_polar_case():
    distance = triple_caustic_distance_batch_ffi(
        jnp.asarray((-0.046038516588439035,)),
        jnp.asarray((0.025585304221408988,)),
        1.0,
        0.1,
        1.0e-5,
        1.0,
        np.pi / 2.0,
        1.0e-4,
    )[0]
    assert float(distance / 1.0e-4) == pytest.approx(3.5509, rel=2.0e-3)


def test_triple_dispatcher_handles_extreme_mass_ratio_caustic_anchor():
    result = triple_magnification_auto(
        -1.0615335967922759e-6,
        -1.1148168901668208e-7,
        0.3,
        1.0e-5,
        1.0e-6,
        0.08,
        0.0,
        1.0e-5,
        0.5,
        0.0,
    )
    assert int(result.method) == 3
    assert bool(result.support_valid)
    assert not bool(result.discovery_overflow)
    assert float(result.magnification) == pytest.approx(
        212521.81362031004, rel=1.0e-4
    )


def test_triple_dispatcher_keeps_zero_radius_on_point_source_path():
    result = triple_magnification_auto(
        -0.046038516588439035,
        0.025585304221408988,
        1.0,
        0.1,
        1.0e-5,
        1.0,
        np.pi / 2.0,
        0.0,
    )
    assert int(result.method) == 0
    assert bool(result.support_valid)
    assert float(result.magnification) > 100.0


def test_triple_source_plane_chord_converges_and_differentiates():
    parameters = jnp.asarray(
        (-0.8, 0.6, 1.0, 0.1, 0.03, 0.7, 0.8, 0.03, 0.3, 0.2)
    )

    def magnification(values):
        return triple_source_plane_quadrature(
            *values,
            rule="chord",
            coarse_order=16,
            fine_order=32,
        ).magnification

    result = triple_source_plane_quadrature(
        *parameters,
        rule="chord",
        coarse_order=16,
        fine_order=32,
    )
    assert bool(result.converged)
    assert not bool(result.root_failure)
    np.testing.assert_allclose(result.magnification, 1.27518316, rtol=1.0e-8)

    value, gradient = jax.value_and_grad(magnification)(parameters)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    for index in range(parameters.size):
        step = 1.0e-5
        finite_difference = (
            magnification(parameters.at[index].add(step))
            - magnification(parameters.at[index].add(-step))
        ) / (2.0 * step)
        np.testing.assert_allclose(
            gradient[index], finite_difference, rtol=3.0e-4, atol=3.0e-7
        )

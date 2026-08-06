import jax
import jax.numpy as jnp
import numpy as np

from lcbinint_jax import binary_magnification_calibrated

jax.config.update("jax_enable_x64", True)


def test_calibrated_dispatcher_reports_native_resolution_and_has_finite_ad():
    parameters = jnp.asarray((0.3, 0.4, 1.4, 1.0e-3, 0.01, 0.4, 0.0))

    def value(active):
        result = binary_magnification_calibrated(
            *active,
            maximum_source_bins=128,
            moment_mode="linear",
        )
        return result.magnification, result

    magnitude, result = value(parameters)
    gradient = jax.grad(lambda active: value(active)[0])(parameters)
    assert bool(result.support_valid)
    assert int(result.selected_source_bins) == 128
    assert int(result.comparison_resolution) == 100
    assert int(result.executed_resolution) == 128
    assert int(result.tile_capacity) == 4194304
    assert not bool(result.prefer_polar)
    assert np.isfinite(float(magnitude))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(result.value_converged)
    assert float(result.value_error) <= float(result.value_budget)


def test_calibrated_dispatcher_keeps_routing_stopped_across_source_position():
    def diagnostics(source_x):
        result = binary_magnification_calibrated(
            source_x,
            0.4,
            1.4,
            1.0e-3,
            0.01,
            maximum_source_bins=128,
            moment_mode="uniform",
        )
        return result.caustic_distance, result.selected_source_bins

    distance_gradient = jax.grad(lambda x: diagnostics(x)[0])(0.3)
    assert float(distance_gradient) == 0.0


def test_calibrated_dispatcher_uses_inverse_ray_without_grid_overflow():
    result = binary_magnification_calibrated(
        -0.003300193370754633,
        0.014763662146897852,
        0.9,
        1.0e-3,
        0.003,
        maximum_source_bins=400,
        moment_mode="uniform",
    )
    assert int(result.method) == 1
    assert int(result.executed_resolution) == 160
    # The coarse bucket carries the comparison: with the bounded frontier and
    # the capacities it needs, 128 no longer overflows into the polar route, so
    # the dispatcher no longer has to reach past 160 for a second opinion.
    assert int(result.comparison_resolution) == 128
    assert bool(result.support_valid)
    assert bool(result.value_converged)
    assert abs(float(result.magnification) - 67.80033446861006) < 6.9e-3

    cross_method = binary_magnification_calibrated(
        -0.003300193370754633,
        0.014763662146897852,
        0.9,
        1.0e-3,
        0.003,
        0.4,
        0.0,
        maximum_source_bins=400,
        moment_mode="linear",
    )
    # The limb-darkened arm used to be pushed onto the polar route by the same
    # overflow; with the capacities the bounded frontier needs it stays on the
    # Cartesian one.  Polar routing is covered by the high-magnification test
    # below, where no capacity in the table is enough.
    assert int(cross_method.method) == 1
    assert bool(cross_method.support_valid)
    assert bool(cross_method.value_converged)


def test_calibrated_dispatcher_high_magnification_uses_polar_inverse_ray():
    result = binary_magnification_calibrated(
        -0.0007041237762303424,
        3.954526794514226e-05,
        1.0,
        1.0e-3,
        1.0e-4,
        maximum_source_bins=400,
        moment_mode="uniform",
    )
    assert int(result.method) == 2
    assert bool(result.used_polar)
    assert bool(result.support_valid)
    assert bool(result.value_converged)
    assert float(result.value_error) <= float(result.value_budget)


def test_calibrated_dispatcher_polar_angular_floor_covers_grazing_cusp():
    result = binary_magnification_calibrated(
        -0.0011744718711112381,
        -0.0017214156233429664,
        1.0,
        1.0e-3,
        1.0e-4,
        maximum_source_bins=400,
        moment_mode="uniform",
    )
    assert bool(result.support_valid)
    assert bool(result.value_converged)
    assert abs(float(result.magnification) - 241.85340579542768) < 0.0243


def test_calibrated_dispatcher_routes_native_safe_point_source():
    parameters = (0.2, 0.1, 1.0, 1.0e-3, 1.0e-4)
    result = binary_magnification_calibrated(
        *parameters,
        maximum_source_bins=400,
        moment_mode="uniform",
    )
    derivative = jax.grad(
        lambda x: binary_magnification_calibrated(
            x,
            *parameters[1:],
            maximum_source_bins=400,
            moment_mode="uniform",
        ).magnification
    )(parameters[0])
    assert int(result.method) == 4
    assert bool(result.point_safe)
    assert bool(result.support_valid)
    assert bool(result.value_converged)
    assert np.isfinite(float(derivative))


def test_calibrated_dispatcher_routes_grazing_source_to_source_plane():
    result = binary_magnification_calibrated(
        -0.0011744718711112381,
        -0.0017214156233429664,
        1.0,
        1.0e-3,
        1.0e-4,
        maximum_source_bins=400,
        moment_mode="uniform",
    )
    assert int(result.method) == 3
    assert bool(result.used_source_plane)
    assert bool(result.chord_band)
    assert bool(result.support_valid)
    assert bool(result.value_converged)
    assert abs(float(result.magnification) - 241.85340579542768) < 2.43e-2


def test_calibrated_source_direct_root_failure_uses_source_plane():
    result = binary_magnification_calibrated(
        -0.03322385260958629,
        -0.0017355510794855819,
        0.98,
        1.0e-5,
        3.0e-4,
        maximum_source_bins=400,
        moment_mode="uniform",
    )
    assert bool(result.chord_band)
    assert int(result.method) == 3
    assert bool(result.used_source_plane)
    assert bool(result.support_valid)
    assert bool(result.value_converged)
    assert abs(float(result.magnification) - 32.75951274809496) < 3.38e-3

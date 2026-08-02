import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import (
    binary_magnification_auto,
    binary_magnification_calibrated,
    binary_magnification_native_pipeline_trajectory,
    binary_magnification_trajectory,
)

jax.config.update("jax_enable_x64", True)


def test_native_pipeline_trajectory_adds_point_route_and_gradient():
    source_x = jnp.asarray((0.2, 0.21))
    source_y = jnp.asarray((0.1, 0.11))
    result = binary_magnification_native_pipeline_trajectory(
        source_x,
        source_y,
        1.0,
        1.0e-3,
        1.0e-4,
        0.0,
        0.0,
    )
    assert np.all(np.asarray(result.method) == 4)
    assert np.all(np.asarray(result.support_valid))
    gradient = jax.grad(
        lambda q: jnp.sum(
            binary_magnification_native_pipeline_trajectory(
                source_x,
                source_y,
                1.0,
                q,
                1.0e-4,
                0.0,
                0.0,
            ).magnification
        )
    )(1.0e-3)
    assert jnp.isfinite(gradient)


def test_native_pipeline_trajectory_adds_converged_grazing_source_route():
    result = binary_magnification_native_pipeline_trajectory(
        jnp.asarray((-0.0011744718711112381,)),
        jnp.asarray((-0.0017214156233429664,)),
        1.0,
        1.0e-3,
        1.0e-4,
        0.0,
        0.0,
    )
    assert int(result.method[0]) == 3
    assert bool(result.used_source_plane[0])
    assert bool(result.support_valid[0])
    np.testing.assert_allclose(
        result.magnification[0],
        241.85340579542768,
        rtol=0.0,
        atol=0.0243,
    )


def test_native_pipeline_trajectory_uses_calibrated_inverse_ray_value():
    result = binary_magnification_native_pipeline_trajectory(
        jnp.asarray((0.2,)),
        jnp.asarray((0.1,)),
        1.2,
        0.1,
        0.2,
        0.4,
        0.0,
        moment_mode="linear",
    )
    reference = binary_magnification_calibrated(
        0.2,
        0.1,
        1.2,
        0.1,
        0.2,
        0.4,
        0.0,
        moment_mode="linear",
    )
    assert int(result.method[0]) == 1
    assert bool(result.support_valid[0])
    np.testing.assert_allclose(
        result.magnification[0],
        reference.magnification,
        rtol=0.0,
        atol=1.0e-12,
    )
    trajectory_gradient = jax.grad(
        lambda x: binary_magnification_native_pipeline_trajectory(
            jnp.atleast_1d(x),
            jnp.asarray((0.1,)),
            1.2,
            0.1,
            0.2,
            0.4,
            0.0,
            moment_mode="linear",
        ).magnification[0]
    )(0.2)
    scalar_gradient = jax.grad(
        lambda x: binary_magnification_calibrated(
            x,
            0.1,
            1.2,
            0.1,
            0.2,
            0.4,
            0.0,
            moment_mode="linear",
        ).magnification
    )(0.2)
    np.testing.assert_allclose(trajectory_gradient, scalar_gradient, rtol=0.0, atol=0.0)


def test_native_pipeline_tight_inverse_ray_value_fails_closed():
    # The bounded frontier reaches 3.8649626 here against a 3.8649464 ladder
    # limit, so the coarse/fine pair honestly agrees to 8e-6 and anything
    # looser than about 3e-6 is now met.  Ask for 1e-6, with no absolute floor
    # to swamp it, to keep exercising the closed door.
    result = binary_magnification_native_pipeline_trajectory(
        jnp.asarray((0.653,)),
        jnp.asarray((0.02,)),
        1.2,
        0.1,
        0.02,
        0.4,
        0.0,
        absolute_tolerance=0.0,
        relative_tolerance=1.0e-6,
        maximum_source_bins=400,
        moment_mode="linear",
    )

    assert int(result.method[0]) == 1
    assert bool(result.support_valid[0])
    assert not bool(result.value_converged[0])
    assert bool(jnp.isfinite(result.estimated_error[0]))


def test_native_pipeline_loose_inverse_ray_bucket_converges():
    result = binary_magnification_native_pipeline_trajectory(
        jnp.asarray((0.653,)),
        jnp.asarray((0.02,)),
        1.2,
        0.1,
        0.02,
        0.4,
        0.0,
        absolute_tolerance=1.0e-4,
        relative_tolerance=1.0e-3,
        maximum_source_bins=400,
        moment_mode="linear",
    )

    assert bool(result.support_valid[0])
    assert bool(result.value_converged[0])
    np.testing.assert_allclose(
        result.magnification[0], 3.8652407702, rtol=0.0, atol=1.0e-8
    )


def test_native_pipeline_trajectory_uses_high_magnification_polar_value():
    result = binary_magnification_native_pipeline_trajectory(
        jnp.asarray((0.7276663,)),
        jnp.asarray((0.0,)),
        1.4,
        1.0e-3,
        0.005,
        0.0,
        0.0,
        moment_mode="uniform",
    )
    reference = binary_magnification_calibrated(
        0.7276663,
        0.0,
        1.4,
        1.0e-3,
        0.005,
        0.0,
        0.0,
        moment_mode="uniform",
    )
    assert int(result.method[0]) == 2
    assert bool(result.support_valid[0])
    np.testing.assert_allclose(
        result.magnification[0],
        reference.magnification,
        rtol=0.0,
        atol=1.0e-12,
    )
    trajectory_gradient = jax.grad(
        lambda x: binary_magnification_native_pipeline_trajectory(
            jnp.atleast_1d(x),
            jnp.asarray((0.0,)),
            1.4,
            1.0e-3,
            0.005,
            0.0,
            0.0,
            moment_mode="uniform",
        ).magnification[0]
    )(0.7276663)
    scalar_gradient = jax.grad(
        lambda x: binary_magnification_calibrated(
            x,
            0.0,
            1.4,
            1.0e-3,
            0.005,
            0.0,
            0.0,
            moment_mode="uniform",
        ).magnification
    )(0.7276663)
    np.testing.assert_allclose(trajectory_gradient, scalar_gradient, rtol=0.0, atol=0.0)


def _require_compiled_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "fixed_support_forward_ffi"):
        pytest.skip("lcbinint was built without JAX FFI support")


def test_bucketed_trajectory_matches_scalar_dispatcher():
    source_x = jnp.asarray((0.3, 0.65, 0.4))
    source_y = jnp.asarray((0.4, 0.0, 0.5))
    trajectory = binary_magnification_trajectory(
        source_x,
        source_y,
        1.2,
        0.1,
        0.02,
        0.4,
        0.0,
        resolution=64,
        tile_capacity=1024,
        limb_samples=16,
        source_plane_fallback=False,
        moment_mode="linear",
    )
    scalar = [
        binary_magnification_auto(
            x,
            y,
            1.2,
            0.1,
            0.02,
            0.4,
            0.0,
            resolution=64,
            tile_capacity=1024,
            limb_samples=16,
            source_plane_fallback=False,
            moment_mode="linear",
        )
        for x, y in zip(source_x, source_y)
    ]
    np.testing.assert_allclose(
        trajectory.magnification,
        jnp.asarray([result.magnification for result in scalar]),
        rtol=1.0e-12,
    )
    np.testing.assert_array_equal(
        trajectory.method,
        jnp.asarray([result.method for result in scalar]),
    )
    np.testing.assert_array_equal(trajectory.attempted_counts, (2, 1, 0, 0))


def test_mixed_trajectory_gradient_matches_finite_difference():
    source_x = jnp.asarray((0.3, 0.65, 0.4))
    source_y = jnp.asarray((0.4, 0.0, 0.5))

    def loss(separation):
        result = binary_magnification_trajectory(
            source_x,
            source_y,
            separation,
            0.1,
            0.02,
            0.4,
            0.0,
            resolution=64,
            tile_capacity=1024,
            limb_samples=16,
            source_plane_fallback=False,
            moment_mode="linear",
        )
        return jnp.sum(result.magnification)

    step = 1.0e-6
    finite_difference = (loss(1.2 + step) - loss(1.2 - step)) / (2.0 * step)
    np.testing.assert_allclose(
        jax.grad(loss)(1.2),
        finite_difference,
        rtol=1.0e-3,
        atol=1.0e-7,
    )


def test_source_plane_bucket_rescues_only_converged_epoch():
    result = binary_magnification_trajectory(
        jnp.asarray((0.3, -0.12659234106123154, 0.04532088349390295)),
        jnp.asarray((0.4, -0.0033574016897174154, 1.5733427409150726)),
        0.55,
        1.0,
        0.003,
        0.4,
        0.0,
        tile_capacity=1,
        moment_mode="linear",
    )
    np.testing.assert_array_equal(result.method, (0, 3, 3))
    np.testing.assert_array_equal(result.support_valid, (True, True, False))
    np.testing.assert_allclose(result.magnification[1], 22.46300856034185, rtol=1.0e-4)
    assert bool(jnp.isnan(result.magnification[2]))
    np.testing.assert_array_equal(result.attempted_counts, (1, 2, 2, 0))


def test_expanded_cartesian_attempt_is_reported():
    result = binary_magnification_trajectory(
        jnp.asarray((-0.04040947298588264,)),
        jnp.asarray((0.004139830325865027,)),
        0.98,
        1.0e-5,
        3.0e-4,
        0.4,
        0.0,
        expanded_cartesian_fallback=True,
        moment_mode="linear",
    )
    np.testing.assert_array_equal(result.support_valid, (True,))
    np.testing.assert_array_equal(result.used_expanded_cartesian, (True,))
    np.testing.assert_array_equal(result.attempted_counts, (0, 1, 1, 1))


def test_curved_boundary_gradients_match_local_vbm_references():
    source_x = jnp.asarray((0.5096774193548388, 0.6387096774193548, 0.7161290322580645))
    source_y = jnp.full(source_x.shape, 0.04)

    def loss(active_x):
        return jnp.sum(
            binary_magnification_trajectory(
                active_x,
                source_y,
                1.2,
                0.1,
                0.02,
                0.4,
                0.0,
                source_plane_fallback=False,
                moment_mode="linear",
            ).magnification
        )

    gradient = jax.grad(loss)(source_x)
    reference = jnp.asarray((149.61, 7.4339348, 0.1912601))
    budget = 1.0e-3 + 5.0e-3 * jnp.maximum(jnp.abs(reference), 1.0)
    assert bool(jnp.all(jnp.abs(gradient - reference) <= budget))


def test_square_root_limb_gradient_uses_detailed_inside_band():
    source_x = jnp.asarray((0.5096774193548388, 0.6387096774193548, 0.7161290322580645))
    source_y = jnp.full(source_x.shape, 0.04)

    def loss(active_x):
        return jnp.sum(
            binary_magnification_trajectory(
                active_x,
                source_y,
                1.2,
                0.1,
                0.02,
                0.3,
                0.2,
                source_plane_fallback=False,
                moment_mode="two_coefficient",
            ).magnification
        )

    gradient = jax.grad(loss)(source_x)
    native_reference = jnp.asarray((149.9868, 7.429045, 0.1992056))
    budget = 1.0e-3 + 5.0e-3 * jnp.maximum(jnp.abs(native_reference), 1.0)
    assert bool(jnp.all(jnp.abs(gradient - native_reference) <= budget))


def test_trajectory_ffi_matches_jax_dispatch_value_and_gradient():
    _require_compiled_ffi()
    source_x = jnp.asarray((0.3, 0.65, 0.4))
    source_y = jnp.asarray((0.4, 0.0, 0.5))

    def evaluate(separation, backend):
        result = binary_magnification_trajectory(
            source_x,
            source_y,
            separation,
            0.1,
            0.02,
            0.4,
            0.0,
            resolution=64,
            tile_capacity=1024,
            limb_samples=16,
            source_plane_fallback=False,
            moment_mode="linear",
            cartesian_backend=backend,
        )
        return jnp.sum(result.magnification), (
            result.magnification,
            result.method,
            result.support_valid,
            result.attempted_counts,
        )

    (jax_loss, jax_aux), jax_gradient = jax.value_and_grad(evaluate, has_aux=True)(
        1.2, "jax"
    )
    (ffi_loss, ffi_aux), ffi_gradient = jax.value_and_grad(evaluate, has_aux=True)(
        1.2, "ffi"
    )

    np.testing.assert_allclose(ffi_loss, jax_loss, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(ffi_aux[0], jax_aux[0], rtol=2.0e-11, atol=2.0e-11)
    for ffi_diagnostic, jax_diagnostic in zip(ffi_aux[1:], jax_aux[1:]):
        np.testing.assert_array_equal(ffi_diagnostic, jax_diagnostic)
    np.testing.assert_allclose(
        ffi_gradient,
        jax_gradient,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_trajectory_ffi_accepts_epoch_dependent_separation_and_gradient():
    _require_compiled_ffi()
    source_x = jnp.asarray((0.3, 0.65, 0.4))
    source_y = jnp.asarray((0.4, 0.0, 0.5))
    separation = jnp.asarray((1.15, 1.2, 1.25))

    def loss(active_separation, backend="auto"):
        return jnp.sum(
            binary_magnification_trajectory(
                source_x,
                source_y,
                active_separation,
                0.1,
                0.02,
                0.4,
                0.0,
                resolution=64,
                tile_capacity=1024,
                limb_samples=16,
                source_plane_fallback=False,
                moment_mode="linear",
                cartesian_backend=backend,
            ).magnification
        )

    gradient = jax.grad(loss)(separation)
    reference = jax.grad(loss)(separation, "jax")
    np.testing.assert_allclose(
        gradient,
        reference,
        rtol=2.0e-9,
        atol=2.0e-9,
    )

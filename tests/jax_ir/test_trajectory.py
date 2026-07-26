import jax
import jax.numpy as jnp
import numpy as np

from lcbinint_jax import (
    binary_magnification_auto,
    binary_magnification_trajectory,
)

jax.config.update("jax_enable_x64", True)


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

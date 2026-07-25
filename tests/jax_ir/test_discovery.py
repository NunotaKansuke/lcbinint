import jax
import jax.numpy as jnp
import numpy as np

from lcbinint_jax import (
    binary_inverse_ray,
    binary_inverse_ray_linear,
    discover_binary_macro_tiles,
)
from lcbinint_jax.discovery import binary_image_seed_points


def test_source_centre_and_limb_seed_shapes_are_static():
    seeds = binary_image_seed_points(0.2, 0.1, 1.2, 0.1, 0.2, limb_samples=16)
    assert seeds.roots.shape == (17 * 5,)
    assert seeds.physical.shape == seeds.roots.shape
    assert int(jnp.sum(seeds.physical)) >= 17 * 3
    assert not bool(seeds.root_failure)


def test_macro_tile_discovery_builds_halo_without_overflow():
    discovery = discover_binary_macro_tiles(
        0.2,
        0.1,
        1.2,
        0.1,
        0.2,
        0.2 / 16.0,
        tile_size=16,
        tile_capacity=512,
        limb_samples=16,
    )
    assert discovery.tile_indices.shape == (512, 2)
    assert discovery.tile_origins.shape == (512, 2)
    assert discovery.tile_mask.shape == (512,)
    assert not bool(discovery.overflow)
    assert not bool(discovery.root_failure)
    assert int(discovery.visited_count) > int(discovery.active_count)
    assert int(discovery.active_count) > 0


def test_macro_tile_capacity_overflow_is_explicit():
    result = binary_inverse_ray(
        0.2,
        0.1,
        1.2,
        0.1,
        0.2,
        0.4,
        0.1,
        resolution=32,
        tile_size=16,
        tile_capacity=32,
        limb_samples=16,
    )
    assert bool(result.discovery_overflow)
    assert not bool(result.support_valid)


def test_automatic_inverse_ray_converges_with_resolution():
    values = []
    for resolution, capacity in ((16, 512), (32, 1024), (64, 2048)):
        result = binary_inverse_ray(
            0.2,
            0.1,
            1.2,
            0.1,
            0.2,
            0.4,
            0.1,
            resolution=resolution,
            tile_size=16,
            tile_capacity=capacity,
            limb_samples=32,
        )
        assert bool(result.support_valid)
        values.append(float(result.magnification))
    assert abs(values[2] - values[1]) < abs(values[1] - values[0])


def test_automatic_path_jvp_matches_local_finite_difference():
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
    # Keep rho fixed here. Its forward value selects the numerical grid spacing,
    # which is intentionally stopped-gradient; rho itself is tested through a
    # fixed grid in test_integrate.py.
    tangent = jnp.asarray([0.2, -0.1, 0.05, 0.02, 0.0, 0.1, -0.05])

    def value(active_parameters):
        return binary_inverse_ray(
            *active_parameters,
            resolution=32,
            tile_size=16,
            tile_capacity=1024,
            limb_samples=32,
        ).magnification

    _, directional_jvp = jax.jvp(value, (parameters,), (tangent,))
    step = 1.0e-6
    finite_difference = (
        value(parameters + step * tangent) - value(parameters - step * tangent)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        directional_jvp,
        finite_difference,
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_linear_specialization_matches_general_value_and_gradient():
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4])

    def general(active_parameters):
        x, y, separation, mass_ratio, radius, limb_c = active_parameters
        return binary_inverse_ray(
            x,
            y,
            separation,
            mass_ratio,
            radius,
            limb_c,
            0.0,
            resolution=32,
            tile_size=16,
            tile_capacity=512,
            limb_samples=16,
        ).magnification

    def specialized(active_parameters):
        return binary_inverse_ray_linear(
            *active_parameters,
            resolution=32,
            tile_size=16,
            tile_capacity=512,
            limb_samples=16,
        ).magnification

    general_value, general_gradient = jax.value_and_grad(general)(parameters)
    specialized_value, specialized_gradient = jax.value_and_grad(specialized)(
        parameters
    )
    np.testing.assert_allclose(
        specialized_value,
        general_value,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        specialized_gradient,
        general_gradient,
        rtol=5.0e-8,
        atol=5.0e-8,
    )

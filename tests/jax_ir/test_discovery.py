import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import (
    binary_inverse_ray,
    binary_inverse_ray_cartesian_batch_ffi,
    binary_inverse_ray_cartesian_ffi,
    binary_inverse_ray_fixed_support_ffi,
    binary_inverse_ray_linear,
    binary_inverse_ray_uniform,
    discover_binary_macro_tiles,
    discover_binary_macro_tiles_ffi,
)
from lcbinint_jax.discovery import binary_image_seed_points
from lcbinint_jax.cpp_backend import cpp_binary_image_roots_ffi_available


def _require_compiled_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "fixed_support_forward_ffi"):
        pytest.skip("lcbinint was built without JAX FFI support")


def _require_discovery_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "macro_tile_discovery_ffi"):
        pytest.skip("lcbinint was built without macro-tile discovery FFI support")


def _require_cartesian_epoch_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "cartesian_epoch_forward_ffi"):
        pytest.skip("lcbinint was built without fused Cartesian epoch FFI support")


def _require_cartesian_batch_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "cartesian_batch_forward_ffi"):
        pytest.skip("lcbinint was built without Cartesian batch FFI support")


def test_source_centre_and_limb_seed_shapes_are_static():
    seeds = binary_image_seed_points(0.2, 0.1, 1.2, 0.1, 0.2, limb_samples=16)
    assert seeds.roots.shape == (17 * 5,)
    assert seeds.physical.shape == seeds.roots.shape
    assert int(jnp.sum(seeds.physical)) >= 17 * 3
    assert not bool(seeds.root_failure)


def test_public_pure_inverse_ray_promotes_float32_inputs_to_float64():
    parameters = tuple(
        jnp.asarray(value, dtype=jnp.float32)
        for value in (0.2, 0.1, 1.2, 0.1, 0.05, 0.4, 0.0)
    )
    result = binary_inverse_ray(
        *parameters,
        resolution=8,
        tile_size=8,
        tile_capacity=64,
        limb_samples=8,
        cartesian_backend="jax",
        root_backend="jax",
    )
    assert result.magnification.dtype == jnp.float64
    assert result.moments.dtype == jnp.float64


def test_exact_caustic_repeated_root_is_valid_finite_source_support():
    if not cpp_binary_image_roots_ffi_available():
        pytest.skip("lcbinint was built without binary image-root FFI support")

    # A smooth fold of the s=0.9, q=0.1 caustic.  The source-centre solve keeps
    # five algebraic branch slots: the appearing/disappearing pair is a
    # repeated root exactly on the caustic.  Keeping its multiplicity is
    # required by algebraic root diagnostics even though there are only four distinct
    # image coordinates at that point.
    seeds = binary_image_seed_points(
        0.06611188225495068,
        0.1549319030240759,
        0.9,
        0.1,
        0.01,
        limb_samples=64,
        root_backend="ffi",
    )
    physical_counts = jnp.sum(seeds.physical.reshape(65, 5), axis=1)

    assert int(physical_counts[0]) == 5
    assert set(map(int, physical_counts.tolist())) == {3, 5}
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


@pytest.mark.parametrize("tile_capacity", (32, 512))
def test_macro_tile_discovery_ffi_matches_jax_exactly(tile_capacity):
    _require_discovery_ffi()
    arguments = (0.2, 0.1, 1.2, 0.1, 0.2, 0.2 / 32.0)
    options = {
        "tile_size": 16,
        "tile_capacity": tile_capacity,
        "limb_samples": 16,
    }
    pure = discover_binary_macro_tiles(*arguments, **options)
    ffi = discover_binary_macro_tiles_ffi(*arguments, **options)
    for field in pure._fields:
        np.testing.assert_array_equal(getattr(ffi, field), getattr(pure, field))


def test_macro_tile_discovery_ffi_is_stopped_gradient():
    _require_discovery_ffi()

    def support_origin_sum(source_x):
        result = discover_binary_macro_tiles_ffi(
            source_x,
            0.1,
            1.2,
            0.1,
            0.2,
            0.2 / 32.0,
            tile_size=16,
            tile_capacity=512,
            limb_samples=16,
        )
        return jnp.sum(result.tile_origins)

    assert float(jax.grad(support_origin_sum)(0.2)) == 0.0


def test_extreme_planetary_limb_roots_use_robust_fallback():
    seeds = binary_image_seed_points(
        -0.04040947298588264,
        0.004139830325865027,
        0.98,
        1.0e-5,
        3.0e-4,
        limb_samples=16,
    )
    assert not bool(seeds.root_failure)
    assert int(jnp.sum(seeds.physical)) >= 3 * 17


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


def test_component_certificate_is_independent_of_limb_seed_count():
    parameters = (
        -0.32501429088718237,
        0.1657421063761565,
        0.8889281178512844,
        0.03138754742696359,
        0.045762863498365475,
    )

    def value(limb_samples):
        return binary_inverse_ray(
            *parameters,
            0.4,
            0.0,
            resolution=32,
            tile_size=16,
            tile_capacity=256,
            limb_samples=limb_samples,
        ).magnification

    eight = value(8)
    sixteen = value(16)
    thirty_two = value(32)

    np.testing.assert_allclose(eight, thirty_two, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(sixteen, thirty_two, rtol=1.0e-12, atol=1.0e-12)


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


def test_uniform_specialization_matches_general_value_and_gradient():
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2])

    def general(active_parameters):
        return binary_inverse_ray(
            *active_parameters,
            0.0,
            0.0,
            resolution=32,
            tile_size=16,
            tile_capacity=512,
            limb_samples=16,
        ).magnification

    def specialized(active_parameters):
        return binary_inverse_ray_uniform(
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


def test_public_inverse_ray_ffi_matches_jax_value_and_gradient():
    _require_compiled_ffi()
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])

    def evaluate(active_parameters, backend):
        result = binary_inverse_ray(
            *active_parameters,
            resolution=32,
            tile_size=16,
            tile_capacity=1024,
            limb_samples=32,
            cartesian_backend=backend,
        )
        return result.magnification, (
            result.moments,
            result.boundary_cells,
            result.active_cells,
            result.support_valid,
        )

    (jax_value, jax_aux), jax_gradient = jax.value_and_grad(evaluate, has_aux=True)(
        parameters, "jax"
    )
    (ffi_value, ffi_aux), ffi_gradient = jax.value_and_grad(evaluate, has_aux=True)(
        parameters, "ffi"
    )

    np.testing.assert_allclose(ffi_value, jax_value, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(ffi_aux[0], jax_aux[0], rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_array_equal(ffi_aux[1:], jax_aux[1:])
    np.testing.assert_allclose(
        ffi_gradient,
        jax_gradient,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


@pytest.mark.parametrize(
    "moment_mode,limb_c,limb_d,boundary_subdivision",
    (
        ("uniform", 0.0, 0.0, 3),
        ("linear", 0.4, 0.0, 3),
        ("two_coefficient", 0.3, 0.2, 4),
    ),
)
def test_fused_cartesian_epoch_matches_staged_ffi_value_and_gradient(
    moment_mode,
    limb_c,
    limb_d,
    boundary_subdivision,
):
    _require_cartesian_epoch_ffi()
    parameters = jnp.asarray([0.5, 0.02, 1.0, 0.1, 0.03, limb_c, limb_d])
    cell_size = parameters[4] / 64.0
    discovery = discover_binary_macro_tiles_ffi(
        *parameters[:5],
        cell_size,
        tile_size=16,
        tile_capacity=2048,
        limb_samples=16,
        root_backend="ffi",
    )

    def staged(active):
        result = binary_inverse_ray_fixed_support_ffi(
            discovery.tile_origins,
            discovery.tile_mask,
            cell_size,
            *active,
            tile_size=16,
            moment_mode=moment_mode,
            boundary_subdivision=boundary_subdivision,
        )
        return result.magnification

    def fused(active):
        result = binary_inverse_ray_cartesian_ffi(
            *active,
            cell_size=cell_size,
            tile_size=16,
            tile_capacity=2048,
            limb_samples=16,
            moment_mode=moment_mode,
            boundary_subdivision=boundary_subdivision,
        )
        return result.magnification, (
            result.moments,
            result.boundary_cells,
            result.active_cells,
            result.tile_count,
            result.discovery_overflow,
            result.root_failure,
        )

    staged_value, staged_gradient = jax.value_and_grad(staged)(parameters)
    (fused_value, fused_aux), fused_gradient = jax.value_and_grad(fused, has_aux=True)(
        parameters
    )
    np.testing.assert_allclose(fused_value, staged_value, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        fused_gradient,
        staged_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert int(fused_aux[3]) == int(discovery.visited_count)
    assert not bool(fused_aux[4])
    assert not bool(fused_aux[5])


@pytest.mark.parametrize(
    "moment_mode,limb_c,limb_d,boundary_subdivision",
    (
        ("uniform", 0.0, 0.0, 3),
        ("linear", 0.4, 0.0, 3),
        ("two_coefficient", 0.3, 0.2, 4),
    ),
)
def test_masked_cartesian_batch_ffi_matches_scalar_values_and_gradient(
    moment_mode,
    limb_c,
    limb_d,
    boundary_subdivision,
):
    _require_cartesian_batch_ffi()
    source_x = jnp.asarray((-0.02, 0.0, 0.035, 0.08))
    source_y = jnp.asarray((0.01, 0.015, -0.01, 0.03))
    active = jnp.asarray((True, False, True, True))
    shared = jnp.asarray((1.0, 0.1, 0.03, limb_c, limb_d))
    options = {
        "cell_size": 5.0e-4,
        "tile_size": 16,
        "tile_capacity": 4096,
        "limb_samples": 24,
        "moment_mode": moment_mode,
        "boundary_subdivision": boundary_subdivision,
    }

    def batched(active_shared):
        result = binary_inverse_ray_cartesian_batch_ffi(
            source_x,
            source_y,
            *active_shared,
            active=active,
            **options,
        )
        return jnp.sum(result.magnification), result

    (batch_loss, batch), batch_gradient = jax.value_and_grad(
        batched,
        has_aux=True,
    )(shared)
    scalar_results = [
        binary_inverse_ray_cartesian_ffi(
            source_x[index],
            source_y[index],
            *shared,
            **options,
        )
        for index in (0, 2, 3)
    ]
    expected_magnification = jnp.asarray(
        (
            scalar_results[0].magnification,
            0.0,
            scalar_results[1].magnification,
            scalar_results[2].magnification,
        )
    )

    def scalar_loss(active_shared):
        return sum(
            binary_inverse_ray_cartesian_ffi(
                source_x[index],
                source_y[index],
                *active_shared,
                **options,
            ).magnification
            for index in (0, 2, 3)
        )

    scalar_gradient = jax.grad(scalar_loss)(shared)
    np.testing.assert_allclose(
        batch.magnification,
        expected_magnification,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(batch_loss, jnp.sum(expected_magnification))
    np.testing.assert_array_equal(batch.support_valid, active)
    np.testing.assert_allclose(
        batch_gradient,
        scalar_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )


# Caustic nearly tangent to the source limb: the five-image cap is 8.2e-6 deep
# against rho = 0.02 and subtends 0.088 rad of the limb, so an 8-, 16- or
# 32-sample limb raster finds or misses it by luck.  The reference is
# VBMicrolensing BinaryMag2 at Tol=1e-9.
_TANGENT_CUSP = (0.653, 0.020, 1.2, 0.1, 0.020)
_TANGENT_CUSP_REFERENCE = 3.960888498085


def _tangent_cusp_magnification(resolution, limb_samples):
    return binary_inverse_ray(
        *_TANGENT_CUSP,
        0.0,
        0.0,
        resolution=resolution,
        tile_size=16,
        tile_capacity=8192,
        limb_samples=limb_samples,
    )


# Resolution 32 is excluded on purpose: one macro tile is then wider than the
# cap is deep, so which tiles the flood marks active is decided by tile
# granularity rather than by the support, and extra limb seeds still move the
# answer.  From resolution 64 up the certificate alone fixes the component set.
@pytest.mark.parametrize("resolution", (64, 128, 256))
def test_tangent_cusp_support_is_independent_of_limb_samples(resolution):
    results = [_tangent_cusp_magnification(resolution, n) for n in (8, 16, 32)]
    for result in results:
        assert bool(result.support_valid)
    for result in results[1:]:
        # Identical component set; only the moment summation order changes with
        # the number of redundant limb seeds.
        np.testing.assert_allclose(
            result.magnification,
            results[0].magnification,
            rtol=1.0e-12,
            atol=0.0,
        )


def test_tangent_cusp_converges_to_the_reference():
    values = [
        float(_tangent_cusp_magnification(resolution, 16).magnification)
        for resolution in (64, 128, 256)
    ]
    errors = [
        abs(value / _TANGENT_CUSP_REFERENCE - 1.0) for value in values
    ]
    assert errors == sorted(errors, reverse=True)
    # Losing the extra image pair costs 3.1e-3 relative and does not shrink
    # with resolution; staying inside 1e-3 at the finest resolution means the
    # pair is integrated, not merely approached.
    assert errors[-1] < 1.0e-3


def test_tangent_cusp_value_and_jvp_share_one_support():
    """The Jet kernel certifies from the primal values, so both paths agree.

    ``discover_cartesian_support`` is handed ``scalar_value`` of each Jet, so
    the derivative evaluation cannot select a different set of components from
    the value evaluation.  Any drift would show up here as a limb-sample
    dependence that the primal alone does not have.
    """

    def magnification(source_y, limb_samples):
        return binary_inverse_ray(
            _TANGENT_CUSP[0],
            source_y,
            *_TANGENT_CUSP[2:],
            0.0,
            0.0,
            resolution=128,
            tile_size=16,
            tile_capacity=8192,
            limb_samples=limb_samples,
        ).magnification

    reference = None
    for limb_samples in (8, 16, 32):
        value, gradient = jax.value_and_grad(magnification)(
            _TANGENT_CUSP[1], limb_samples)
        assert np.isfinite(float(value))
        assert np.isfinite(float(gradient))
        if reference is None:
            reference = (float(value), float(gradient))
            continue
        np.testing.assert_allclose(
            float(value), reference[0], rtol=1.0e-12, atol=0.0)
        np.testing.assert_allclose(
            float(gradient), reference[1], rtol=1.0e-9, atol=0.0)

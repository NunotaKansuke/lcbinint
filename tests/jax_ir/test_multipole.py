import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcbinint_jax import (
    binary_hexadecapole,
    binary_hexadecapole_batch_ffi,
    binary_magnification_auto,
    binary_point_source_magnification,
)

jax.config.update("jax_enable_x64", True)


def _require_root_ffi():
    from lcbinint import _native

    if not hasattr(_native._jax_ir, "binary_image_roots_ffi"):
        pytest.skip("lcbinint was built without binary-image root FFI support")


def test_point_source_root_ffi_matches_jax_value_and_gradient():
    _require_root_ffi()
    parameters = jnp.asarray((0.2, 0.1, 1.2, 0.1))

    def evaluate(active, backend):
        return binary_point_source_magnification(
            *active,
            root_backend=backend,
        ).magnification

    pure_value, pure_gradient = jax.value_and_grad(evaluate)(parameters, "jax")
    ffi_value, ffi_gradient = jax.value_and_grad(evaluate)(parameters, "ffi")
    np.testing.assert_allclose(ffi_value, pure_value, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        ffi_gradient,
        pure_gradient,
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_hexadecapole_root_ffi_matches_jax_value_and_gradient():
    _require_root_ffi()
    parameters = jnp.asarray((0.3, 0.4, 1.4, 1.0e-3, 0.01, 0.4, 0.1))

    def evaluate(active, backend):
        return binary_hexadecapole(
            *active,
            root_backend=backend,
        ).magnification

    pure_value, pure_gradient = jax.value_and_grad(evaluate)(parameters, "jax")
    ffi_value, ffi_gradient = jax.value_and_grad(evaluate)(parameters, "ffi")
    np.testing.assert_allclose(ffi_value, pure_value, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(
        ffi_gradient,
        pure_gradient,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_batched_hexadecapole_ffi_matches_scalar_value_and_gradient():
    _require_root_ffi()
    source_x = jnp.asarray((0.2, 0.5, 0.8))
    source_y = jnp.asarray((0.04, 0.04, 0.04))
    shared = jnp.asarray((1.2, 0.1, 0.02, 0.4, 0.1))

    def scalar(active):
        return jnp.sum(
            jax.lax.map(
                lambda position: binary_hexadecapole(
                    position[0],
                    position[1],
                    *active,
                    root_backend="ffi",
                ).magnification,
                (source_x, source_y),
            )
        )

    def batched(active):
        return jnp.sum(
            binary_hexadecapole_batch_ffi(
                source_x,
                source_y,
                *active,
            ).magnification
        )

    scalar_value, scalar_gradient = jax.value_and_grad(scalar)(shared)
    batch_value, batch_gradient = jax.value_and_grad(batched)(shared)
    np.testing.assert_allclose(batch_value, scalar_value, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        batch_gradient,
        scalar_gradient,
        rtol=0.0,
        atol=5.0e-9,
    )


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


def test_pure_root_rule_allows_first_order_and_rejects_nested_ad():
    def magnification(source_x):
        return binary_point_source_magnification(
            source_x,
            0.1,
            1.2,
            0.1,
            root_backend="jax",
        ).magnification

    assert jnp.isfinite(jax.jit(jax.grad(magnification))(0.2))
    for transform in (jax.hessian(magnification), jax.jit(jax.hessian(magnification))):
        with pytest.raises(NotImplementedError, match="second and higher derivatives"):
            transform(0.2)


def test_ffi_and_auto_root_rules_reject_nested_ad_consistently():
    _require_root_ffi()

    for backend in ("ffi", "auto"):
        def magnification(source_x):
            return binary_point_source_magnification(
                source_x,
                0.1,
                1.2,
                0.1,
                root_backend=backend,
            ).magnification

        for transform in (
            jax.hessian(magnification),
            jax.jit(jax.hessian(magnification)),
        ):
            with pytest.raises(
                NotImplementedError, match="second and higher derivatives"
            ):
                transform(0.2)


def test_pure_point_source_promotes_float32_inputs_to_float64():
    result = binary_point_source_magnification(
        jnp.asarray(0.2, dtype=jnp.float32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(1.2, dtype=jnp.float32),
        jnp.asarray(0.1, dtype=jnp.float32),
        root_backend="jax",
    )
    assert result.magnification.dtype == jnp.float64


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
        magnification(parameters[0] + step) - magnification(parameters[0] - step)
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

    gradient = jax.grad(magnification)(jnp.asarray((0.3, 0.4, 1.4, 1.0e-3)))
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
    assert not bool(result.used_source_plane)
    assert not bool(result.used_polar)
    assert math.isnan(float(result.magnification))


def test_hybrid_source_plane_fallback_is_disabled_for_smooth_overflow():
    result = binary_magnification_auto(
        -0.12659234106123154,
        -0.0033574016897174154,
        0.55,
        1.0,
        0.003,
        0.4,
        0.0,
        tile_capacity=1,
        moment_mode="linear",
    )
    assert int(result.method) == 1
    assert not bool(result.support_valid)
    assert not bool(result.used_source_plane)
    assert math.isnan(float(result.magnification))


def test_hybrid_expanded_cartesian_retry_rescues_source_plane_rejection():
    result = binary_magnification_auto(
        -0.04040947298588264,
        0.004139830325865027,
        0.98,
        1.0e-5,
        3.0e-4,
        0.4,
        0.0,
        expanded_cartesian_fallback=True,
        moment_mode="linear",
    )
    assert int(result.method) == 1
    assert bool(result.support_valid)
    assert not bool(result.used_source_plane)
    np.testing.assert_allclose(result.magnification, 27.687565570160192, rtol=1.0e-4)


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

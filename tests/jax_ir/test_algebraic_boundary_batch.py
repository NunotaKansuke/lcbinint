"""M1 full-curve ABI checks for the explicit algebraic-boundary backend."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lcbinint_jax import (
    algebraic_boundary_batch_call_counters,
    binary_algebraic_boundary_batch_ffi,
    binary_algebraic_boundary_batch_value_jacobian_ffi,
    binary_algebraic_boundary_polar_ffi,
    cpp_algebraic_boundary_batch_ffi_available,
    reset_algebraic_boundary_batch_call_counters,
)


pytestmark = pytest.mark.skipif(
    not cpp_algebraic_boundary_batch_ffi_available(),
    reason="algebraic-boundary M1 batch FFI not built",
)


def _curve(count, *, c=0.4, d=0.2, mask=None):
    x = jnp.linspace(0.19, 0.23, count)
    y = jnp.linspace(0.13, 0.17, count)
    return binary_algebraic_boundary_batch_ffi(
        x,
        y,
        jnp.full((count,), 1.35),
        0.05,
        0.28,
        finite_mask=mask,
        limb_c=c,
        limb_d=d,
        value_atol=1.0e-4,
        value_rtol=1.0e-4,
        gradient_atol=3.0e-4,
    )


@pytest.mark.parametrize("count", [1, 5, 257])
def test_one_native_call_per_full_curve_and_lowered_ir(count):
    def forward(x):
        return binary_algebraic_boundary_batch_ffi(
            x,
            jnp.full((count,), 0.15),
            jnp.full((count,), 1.35),
            0.05,
            0.28,
            limb_c=0.4,
            limb_d=0.2,
        ).magnification

    compiled = jax.jit(forward)
    x = jnp.linspace(0.19, 0.23, count)
    compiled(x).block_until_ready()  # compilation/warmup is not counted
    reset_algebraic_boundary_batch_call_counters()
    compiled(x).block_until_ready()
    assert algebraic_boundary_batch_call_counters() == {"forward": 1, "jacobian": 0}
    reset_algebraic_boundary_batch_call_counters()
    jax.jvp(compiled, (x,), (jnp.ones_like(x),))[0].block_until_ready()
    assert algebraic_boundary_batch_call_counters() == {"forward": 0, "jacobian": 1}
    assert str(compiled.lower(x).compiler_ir(dialect="stablehlo")).count("custom_call") == 1


def test_scalar_batch_values_jacobian_status_and_order_are_identical():
    count = 5
    result = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.linspace(0.19, 0.23, count),
        jnp.linspace(0.13, 0.17, count),
        jnp.full((count,), 1.35),
        0.05,
        0.28,
        limb_c=0.4,
        limb_d=0.2,
    )
    x = np.linspace(0.19, 0.23, count)
    y = np.linspace(0.13, 0.17, count)

    def scalar(parameters):
        return binary_algebraic_boundary_polar_ffi(
            source_x=parameters[0],
            source_y=parameters[1],
            source_radius=parameters[2],
            mass_ratio=parameters[3],
            separation=parameters[4],
            limb_c=0.4,
            limb_d=0.2,
            grad_tol=3.0e-4,
            relative_tolerance=1.0e-4,
        ).magnification

    scalar_values = []
    scalar_rows = []
    for xi, yi in zip(x, y):
        parameters = jnp.array([xi, yi, 0.05, 0.28, 1.35], dtype=jnp.float64)
        scalar_values.append(scalar(parameters))
        scalar_rows.append(jax.jacfwd(scalar)(parameters))
    np.testing.assert_allclose(result.magnification, scalar_values, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(result.jacobian, scalar_rows, rtol=0.0, atol=2.0e-12, equal_nan=True)
    assert np.all(np.isin(result.status, [0, 3]))
    assert np.all(result.method == 2)  # finite algebraic route


@pytest.mark.parametrize("c,d", [(0.5, 0.0), (0.3, 0.2)])
def test_fixed_limb_darkening_has_one_forward_jvp_primal(c, d):
    def value(x):
        return binary_algebraic_boundary_batch_ffi(
            x, jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28, limb_c=c, limb_d=d
        ).magnification

    x = jnp.array([0.22])
    ordinary = value(x)
    primal, tangent = jax.jvp(value, (x,), (jnp.ones_like(x),))
    np.testing.assert_array_equal(ordinary, primal)
    assert np.all(np.isfinite(tangent))


def test_nonzero_limb_tangent_is_refused():
    with pytest.raises(NotImplementedError, match="limb-darkening tangents"):
        jax.jvp(
            lambda c: binary_algebraic_boundary_batch_ffi(
                jnp.array([0.22]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28, limb_c=c
            ).magnification,
            (jnp.array(0.4),),
            (jnp.array(1.0),),
        )


def test_mask_keeps_point_and_finite_rows_in_one_native_call():
    mask = jnp.array([True, False, True, False, True])
    result = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.linspace(0.19, 0.23, 5),
        jnp.linspace(0.13, 0.17, 5),
        jnp.full((5,), 1.35),
        0.05,
        0.28,
        finite_mask=mask,
        limb_c=0.4,
        limb_d=0.2,
    )
    assert np.array_equal(result.method, np.array([2, 1, 2, 1, 2], dtype=np.int32))
    assert np.all(np.isin(result.status[mask], [0, 3]))
    assert np.all(result.status[~mask] == 0)
    assert np.all(np.isfinite(result.jacobian[~mask]))
    assert np.all(result.root_solve_count[~mask] > 0)
    assert np.all(result.lens_evaluation_count[~mask] > 0)


def test_forced_finite_mask_never_reports_router_shortcut_method():
    result = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.array([10.0]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28,
        limb_c=0.4, limb_d=0.2,
    )
    assert np.array_equal(result.method, np.array([2], dtype=np.int32))
    assert np.all(result.root_solve_count >= 0)
    assert np.all(result.lens_evaluation_count >= 0)


def test_zero_value_budget_is_fail_closed():
    result = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.array([0.22]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28,
        limb_c=0.4, limb_d=0.2, value_atol=0.0, value_rtol=0.0,
    )
    assert np.array_equal(result.status, np.array([2], dtype=np.int32))
    assert np.all(np.isnan(result.magnification))
    assert np.all(np.isnan(result.jacobian))
    # Fail closed must not erase the measured absolute error or work evidence.
    assert np.all(np.isfinite(result.value_error))
    assert np.all(result.value_error > 0.0)
    assert np.all(np.isfinite(result.gradient_error))
    assert np.all(result.root_solve_count > 0)
    assert np.all(result.lens_evaluation_count > 0)
    assert np.all(result.refinement_level >= 0)


def test_nonstrict_keeps_structurally_valid_jacobian_after_grad_budget_failure():
    result = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.array([0.22]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28,
        limb_c=0.4, limb_d=0.2, gradient_atol=0.0, strict=False,
    )
    assert np.array_equal(result.status, np.array([3], dtype=np.int32))
    assert np.all(np.isfinite(result.gradient_error))
    assert np.all(np.isfinite(result.jacobian))


def test_point_rows_report_measured_not_sentinel_diagnostics():
    result = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.array([0.22]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28,
        finite_mask=jnp.array([False]), limb_c=0.4, limb_d=0.2,
    )
    assert np.array_equal(result.status, np.array([0], dtype=np.int32))
    assert np.all(result.method == 1)
    assert np.all(np.isfinite(result.value_error))
    assert np.all(np.isfinite(result.gradient_error))
    assert np.all(result.root_solve_count > 0)
    assert np.all(result.lens_evaluation_count > 0)


@pytest.mark.parametrize("strict", [True, False])
def test_point_rows_obey_gradient_budget_and_strict_masking(strict):
    result = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.array([0.22]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28,
        finite_mask=jnp.array([False]), limb_c=0.4, limb_d=0.2,
        gradient_atol=0.0, strict=strict,
    )
    assert np.array_equal(result.status, np.array([3], dtype=np.int32))
    assert np.all(np.isfinite(result.gradient_error))
    assert np.all(np.isnan(result.jacobian) if strict else np.isfinite(result.jacobian))


def test_unknown_policy_is_rejected_before_ffi_lowering():
    with pytest.raises(ValueError, match="refinement policy"):
        binary_algebraic_boundary_batch_ffi(
            jnp.array([0.22]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28,
            refinement_policy_id="adaptive-unimplemented",
        )
    with pytest.raises(ValueError, match="fallback policy"):
        binary_algebraic_boundary_batch_value_jacobian_ffi(
            jnp.array([0.22]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28,
            fallback_policy_id="point",
        )


def test_q_exchange_preserves_value_and_canonical_geometry_jacobian():
    left = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.array([0.037]), jnp.array([-0.021]), jnp.array([0.9]), 0.007, 0.02
    )
    right = binary_algebraic_boundary_batch_value_jacobian_ffi(
        jnp.array([-0.037]), jnp.array([-0.021]), jnp.array([0.9]), 0.007, 50.0
    )
    np.testing.assert_allclose(left.magnification, right.magnification, rtol=3.0e-4, atol=3.0e-4)
    np.testing.assert_allclose(left.jacobian[:, 0], -right.jacobian[:, 0], rtol=3.0e-4, atol=3.0e-4)
    np.testing.assert_allclose(left.jacobian[:, 1:3], right.jacobian[:, 1:3], rtol=3.0e-4, atol=3.0e-4)
    np.testing.assert_allclose(left.jacobian[:, 4], right.jacobian[:, 4], rtol=3.0e-4, atol=3.0e-4)
    np.testing.assert_allclose(
        left.jacobian[:, 3], -right.jacobian[:, 3] / 0.02**2, rtol=3.0e-4, atol=3.0e-4
    )


def test_multiple_tolerances_keep_the_same_smooth_value_jacobian_target():
    results = [
        binary_algebraic_boundary_batch_value_jacobian_ffi(
            jnp.array([0.22]), jnp.array([0.15]), jnp.array([1.35]), 0.05, 0.28,
            limb_c=0.4, limb_d=0.2, value_rtol=rtol, gradient_atol=3.0e-4,
        )
        for rtol in (1.0e-4, 3.0e-5, 1.0e-5)
    ]
    for result in results:
        assert np.array_equal(result.status, np.array([0], dtype=np.int32))
        assert np.all(np.isfinite(result.jacobian))
    for result in results[1:]:
        np.testing.assert_allclose(result.magnification, results[0].magnification, rtol=0.0, atol=2.0e-12)
        np.testing.assert_allclose(result.jacobian, results[0].jacobian, rtol=0.0, atol=2.0e-12)

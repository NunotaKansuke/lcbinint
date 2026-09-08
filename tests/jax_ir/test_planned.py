"""Geometry-only fixed route execution plans."""

import jax
import jax.numpy as jnp
import numpy as np

from lcbinint_jax.planned import make_result_function


jax.config.update("jax_enable_x64", True)


def test_point_plan_groups_epochs_and_remains_differentiable():
    plan = make_result_function(
        ("point_source", "point_source", "point_source"), (0, 0, 0), "uniform"
    )

    def evaluate(source_x):
        return plan(
            source_x,
            jnp.asarray((-0.1, 0.05, 0.2)),
            jnp.asarray(1.2),
            jnp.asarray(0.1),
            jnp.asarray(0.01),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        ).magnification

    source_x = jnp.asarray((-0.2, 0.0, 0.2))
    result = plan(
        source_x,
        jnp.asarray((-0.1, 0.05, 0.2)),
        jnp.asarray(1.2),
        jnp.asarray(0.1),
        jnp.asarray(0.01),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    assert np.asarray(result.support_valid).all()
    assert np.isfinite(np.asarray(result.magnification)).all()
    gradient = jax.jacfwd(evaluate)(source_x)
    assert np.isfinite(np.asarray(gradient)).all()


def test_native_point_batch_ffi_matches_reference_value_and_jvp():
    """Exercise the optional grouped native point route when it is built."""

    from lcbinint_jax.cpp_backend import (
        binary_point_source_batch_ffi,
        cpp_point_source_batch_ffi_available,
    )
    from lcbinint_jax.multipole import binary_point_source_magnification

    if not cpp_point_source_batch_ffi_available():
        import pytest

        pytest.skip("native point-source FFI is not built in this checkout")

    source_x = jnp.asarray((0.2, 0.9, 1.3))
    source_y = jnp.asarray((0.1, -0.2, 0.4))
    separation = jnp.asarray(1.2)
    mass_ratio = jnp.asarray(0.1)

    native = binary_point_source_batch_ffi(
        source_x, source_y, separation, mass_ratio
    )
    reference = jax.vmap(
        lambda x, y: binary_point_source_magnification(
            x, y, separation, mass_ratio, root_backend="jax"
        )
    )(source_x, source_y)
    np.testing.assert_allclose(
        np.asarray(native.magnification),
        np.asarray(reference.magnification),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    assert np.asarray(native.image_count).tolist() == [5, 3, 3]
    assert not np.asarray(native.root_failure).any()

    native_jacobian = jax.jacfwd(
        lambda x: binary_point_source_batch_ffi(
            x, source_y, separation, mass_ratio
        ).magnification
    )(source_x)
    reference_jacobian = jax.jacfwd(
        lambda x: jax.vmap(
            lambda xx, yy: binary_point_source_magnification(
                xx, yy, separation, mass_ratio, root_backend="jax"
            )
        )(x, source_y).magnification
    )(source_x)
    np.testing.assert_allclose(
        np.asarray(native_jacobian),
        np.asarray(reference_jacobian),
        rtol=2.0e-9,
        atol=2.0e-10,
    )


def test_dynamic_separation_plans_keep_one_differentiable_batch():
    """Orbit-like separation arrays do not fall back to scalar calls."""

    point_plan = make_result_function(
        ("point_source", "point_source", "point_source"), (0, 0, 0), "uniform"
    )
    source_x = jnp.asarray((0.2, 0.9, 1.3))
    source_y = jnp.asarray((0.1, -0.2, 0.4))
    separation = jnp.asarray((1.18, 1.2, 1.22))

    def point_value(current_separation):
        return point_plan(
            source_x,
            source_y,
            current_separation,
            jnp.asarray(0.1),
            jnp.asarray(0.01),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        ).magnification

    point_result = point_plan(
        source_x,
        source_y,
        separation,
        jnp.asarray(0.1),
        jnp.asarray(0.01),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    assert np.asarray(point_result.support_valid).all()
    assert np.isfinite(np.asarray(point_result.magnification)).all()
    assert np.isfinite(np.asarray(jax.jacfwd(point_value)(separation))).all()

    from lcbinint_jax.cpp_backend import cpp_trajectory_ffi_available

    if not cpp_trajectory_ffi_available():
        import pytest

        pytest.skip("trajectory FFI is not built in this checkout")
    trajectory_plan = make_result_function(
        ("trajectory_dispatch",) * 3, (32, 32, 32), "uniform"
    )
    trajectory_result = trajectory_plan(
        source_x,
        source_y,
        separation,
        jnp.asarray(0.1),
        jnp.asarray(0.01),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    assert np.asarray(trajectory_result.support_valid).all()
    assert np.isfinite(np.asarray(trajectory_result.magnification)).all()
    assert np.isfinite(
        np.asarray(
            jax.jacfwd(
                lambda current_separation: trajectory_plan(
                    source_x,
                    source_y,
                    current_separation,
                    jnp.asarray(0.1),
                    jnp.asarray(0.01),
                    jnp.asarray(0.0),
                    jnp.asarray(0.0),
                ).magnification
            )(separation)
        )
    ).all()

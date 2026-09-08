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


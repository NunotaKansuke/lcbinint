"""Tests for the parallax ephemeris interpolation kernels."""

import jax
import jax.numpy as jnp
import numpy as np
from lcbinint_jax import higher_order

import lcbinint

jax.config.update("jax_enable_x64", True)


def test_cubic_hermite_reproduces_position_and_velocity():
    times = jnp.asarray((0.0, 0.7, 2.1, 3.4))
    positions = jnp.stack(
        (
            0.3 * times**3 - 0.4 * times**2 + 0.2 * times + 1.0,
            -0.2 * times**3 + 0.5 * times**2 - 0.1 * times + 2.0,
            0.1 * times**3 + 0.3 * times**2 + 0.4,
        ),
        axis=-1,
    )
    velocities = jnp.stack(
        (
            0.9 * times**2 - 0.8 * times + 0.2,
            -0.6 * times**2 + times - 0.1,
            0.3 * times**2 + 0.6 * times,
        ),
        axis=-1,
    )
    query = jnp.asarray((0.05, 0.35, 0.9, 1.5, 2.9, 3.35))

    actual, actual_velocity = higher_order._hermite_interpolate(
        times, positions, velocities, query
    )
    expected = jnp.stack(
        (
            0.3 * query**3 - 0.4 * query**2 + 0.2 * query + 1.0,
            -0.2 * query**3 + 0.5 * query**2 - 0.1 * query + 2.0,
            0.1 * query**3 + 0.3 * query**2 + 0.4,
        ),
        axis=-1,
    )
    expected_velocity = jnp.stack(
        (
            0.9 * query**2 - 0.8 * query + 0.2,
            -0.6 * query**2 + query - 0.1,
            0.3 * query**2 + 0.6 * query,
        ),
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        actual_velocity, expected_velocity, rtol=0.0, atol=2.0e-14
    )


def test_space_table_velocity_is_shared_by_native_and_jax():
    table_times = np.arange(2458998.0, 2459003.0)
    reduced = table_times - 2459000.0
    distance = 1.0 + 0.01 * reduced + 0.002 * reduced**2 + 0.0003 * reduced**3
    table = np.column_stack((table_times, np.zeros((5, 2)), distance))
    site = lcbinint.obs.Site("space", table)

    expected_velocity = np.asarray(
        higher_order._estimate_ephemeris_velocity(
            jnp.asarray(site.ephemeris_time),
            jnp.asarray(site.ephemeris_position),
        )
    )
    np.testing.assert_allclose(
        site.ephemeris_velocity, expected_velocity, rtol=0.0, atol=2.0e-15
    )

    model = lcbinint.Model(
        parallax=True,
        sky=lcbinint.obs.SkyCoord(270.0, -30.0),
        t_ref=9000.0,
    )
    parameters = {
        "t0": 9000.0,
        "tE": 20.0,
        "u0": 0.1,
        "alpha": 0.2,
        "s": 1.1,
        "q": 10.0,
        "rho": 0.0,
        "piEN": 0.1,
        "piEE": -0.05,
    }
    times = np.asarray((8998.25, 8999.4, 9000.5, 9001.75))
    native = lcbinint.LightCurve(
        model=model,
        site=site,
        options=lcbinint.Options(coordinates="vbm"),
    ).source_trajectory(times, parameters)
    jax_curve = lcbinint.LightCurve(
        model=model,
        site=site,
        options=lcbinint.Options(jax=True, coordinates="vbm"),
    )
    actual = jax_curve.source_trajectory(jnp.asarray(times), parameters)
    np.testing.assert_allclose(actual.x, native.x, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(actual.y, native.y, rtol=0.0, atol=3.0e-14)

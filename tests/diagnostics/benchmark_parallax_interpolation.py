"""Benchmark linear and Hermite parallax-table interpolation.

Run this from a build that contains the package, for example::

    PYTHONPATH=build python tests/diagnostics/benchmark_parallax_interpolation.py

The speed comparison isolates the JAX geometry functions.  The physical-table
accuracy check uses a local degree-7 Lagrange reconstruction of the Earth table
and an analytic smooth spacecraft trajectory; neither reference is used by the
production interpolator.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from lcbinint_jax import (
    annual_parallax_offsets,
    higher_order,
    load_earth_ephemeris,
    space_parallax_offsets,
)

jax.config.update("jax_enable_x64", True)


def linear_interpolate(times, values, query):
    return np.stack(
        tuple(np.interp(query, times, values[:, axis]) for axis in range(values.shape[1])),
        axis=-1,
    )


def hermite_interpolate(times, values, derivatives, query):
    upper = np.searchsorted(times, query, side="right")
    upper = np.clip(upper, 1, len(times) - 1)
    lower = upper - 1
    query = np.clip(query, times[0], times[-1])
    t0 = times[lower]
    t1 = times[upper]
    interval = t1 - t0
    u = (query - t0) / interval
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return (
        h00[..., None] * values[lower]
        + h10[..., None] * interval[..., None] * derivatives[lower]
        + h01[..., None] * values[upper]
        + h11[..., None] * interval[..., None] * derivatives[upper]
    )


def local_lagrange_reference(times, values, query, order=8):
    """Evaluate a local high-order polynomial without SciPy dependencies."""

    result = []
    for point in np.asarray(query).flat:
        insertion = int(np.searchsorted(times, point, side="left"))
        start = min(max(insertion - order // 2, 0), len(times) - order)
        nodes = times[start : start + order]
        samples = values[start : start + order]
        if np.any(point == nodes):
            result.append(samples[np.flatnonzero(point == nodes)[0]])
            continue
        weights = np.ones(order)
        for index in range(order):
            weights[index] = 1.0 / np.prod(
                nodes[index] - np.delete(nodes, index)
            )
        barycentric = weights / (point - nodes)
        result.append(np.sum(barycentric[:, None] * samples, axis=0) / np.sum(barycentric))
    return np.asarray(result).reshape(np.asarray(query).shape + (values.shape[1],))


def three_point_derivatives(times, values):
    def derivative(x0, x1, x2, y0, y1, y2, x):
        w0 = (2.0 * x - x1 - x2) / ((x0 - x1) * (x0 - x2))
        w1 = (2.0 * x - x0 - x2) / ((x1 - x0) * (x1 - x2))
        w2 = (2.0 * x - x0 - x1) / ((x2 - x0) * (x2 - x1))
        return w0 * y0 + w1 * y1 + w2 * y2

    if len(times) == 2:
        slope = (values[1] - values[0]) / (times[1] - times[0])
        return np.stack((slope, slope), axis=0)
    result = np.empty_like(values)
    result[0] = derivative(*times[:3], *values[:3], times[0])
    for index in range(1, len(times) - 1):
        result[index] = derivative(
            times[index - 1], times[index], times[index + 1],
            values[index - 1], values[index], values[index + 1], times[index],
        )
    result[-1] = derivative(*times[-3:], *values[-3:], times[-1])
    return result


def old_linear_annual(time, earth):
    time = jnp.asarray(time)
    offset = jnp.where(9000.0 < higher_order._RJD_ORIGIN, higher_order._RJD_ORIGIN, 0.0)
    earth_time = time + offset
    reference_time = 9000.0 + offset
    event, north, east = higher_order._sky_basis(270.0, -30.0)

    def interpolate(query):
        position = jnp.stack(
            tuple(jnp.interp(query, earth.time, earth.position[:, axis]) for axis in range(3)),
            axis=-1,
        )
        velocity = jnp.stack(
            tuple(jnp.interp(query, earth.time, earth.velocity[:, axis]) for axis in range(3)),
            axis=-1,
        )
        return position, velocity

    def light_time(query):
        emit = query
        for _ in range(5):
            position, _ = interpolate(emit)
            emit = query - jnp.sum(position * event, axis=-1) * higher_order._AU_LIGHT_TRAVEL_DAYS
        return emit

    position, _ = interpolate(light_time(earth_time))
    reference_position, reference_velocity = interpolate(light_time(reference_time))
    projected = jnp.stack(
        (-jnp.sum(position * north, axis=-1), -jnp.sum(position * east, axis=-1)),
        axis=-1,
    )
    reference = jnp.asarray(
        (-jnp.sum(reference_position * north), -jnp.sum(reference_position * east))
    )
    velocity = jnp.asarray(
        (-jnp.sum(reference_velocity * north), -jnp.sum(reference_velocity * east))
    )
    displacement = projected - reference - (earth_time - reference_time)[..., None] * velocity
    return (0.12 * displacement[..., 0] - 0.05 * displacement[..., 1],
            0.05 * displacement[..., 0] + 0.12 * displacement[..., 1])


def old_linear_space(time, ephemeris_time, position):
    time = jnp.asarray(time)
    query = time + jnp.where(time < higher_order._RJD_ORIGIN, higher_order._RJD_ORIGIN, 0.0)
    interpolated = jnp.stack(
        tuple(jnp.interp(query, ephemeris_time, position[:, axis]) for axis in range(3)),
        axis=-1,
    )
    cos_obliquity = 0.9174820003578725
    sin_obliquity = 0.3977772982704228
    ra = 270.0 * higher_order._DEGREE_TO_RADIAN
    dec = -30.0 * higher_order._DEGREE_TO_RADIAN
    object_direction = jnp.asarray(
        (
            jnp.cos(ra) * jnp.cos(dec),
            jnp.sin(ra) * jnp.cos(dec) * cos_obliquity
            + jnp.sin(dec) * sin_obliquity,
            -jnp.sin(ra) * jnp.cos(dec) * sin_obliquity
            + jnp.sin(dec) * cos_obliquity,
        )
    )
    north_2000 = jnp.asarray((0.0, sin_obliquity, cos_obliquity))
    radial = higher_order._normalize(
        -north_2000 + object_direction * jnp.sum(north_2000 * object_direction)
    )
    tangential = jnp.cross(radial, object_direction)
    radial_projection = jnp.sum(interpolated * radial, axis=-1)
    tangential_projection = jnp.sum(interpolated * tangential, axis=-1)
    return (0.12 * radial_projection + 0.05 * tangential_projection,
            0.12 * tangential_projection - 0.05 * radial_projection)


def measure(function, arguments, repeat):
    compiled = jax.jit(function)
    for _ in range(3):
        output = compiled(*arguments)
        (output[0] + output[1]).block_until_ready()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        output = compiled(*arguments)
        (output[0] + output[1]).block_until_ready()
        samples.append(time.perf_counter() - start)
    return {
        "median_ms": float(np.median(samples) * 1.0e3),
        "min_ms": float(np.min(samples) * 1.0e3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=8)
    args = parser.parse_args()

    cubic_times = np.asarray((0.0, 0.7, 2.1, 3.4))
    cubic_values = np.stack((
        0.3 * cubic_times**3 - 0.4 * cubic_times**2 + 0.2 * cubic_times + 1.0,
        -0.2 * cubic_times**3 + 0.5 * cubic_times**2 - 0.1 * cubic_times + 2.0,
        0.1 * cubic_times**3 + 0.3 * cubic_times**2 + 0.4,
    ), axis=-1)
    cubic_velocity = np.stack((
        0.9 * cubic_times**2 - 0.8 * cubic_times + 0.2,
        -0.6 * cubic_times**2 + cubic_times - 0.1,
        0.3 * cubic_times**2 + 0.6 * cubic_times,
    ), axis=-1)
    cubic_query = np.linspace(0.001, 3.399, 1001)
    cubic_truth = np.stack((
        0.3 * cubic_query**3 - 0.4 * cubic_query**2 + 0.2 * cubic_query + 1.0,
        -0.2 * cubic_query**3 + 0.5 * cubic_query**2 - 0.1 * cubic_query + 2.0,
        0.1 * cubic_query**3 + 0.3 * cubic_query**2 + 0.4,
    ), axis=-1)
    print("cubic_position_max_abs_error", {
        "linear": float(np.max(np.abs(linear_interpolate(cubic_times, cubic_values, cubic_query) - cubic_truth))),
        "hermite": float(np.max(np.abs(hermite_interpolate(cubic_times, cubic_values, cubic_velocity, cubic_query) - cubic_truth))),
    })

    earth = load_earth_ephemeris()
    earth_times = np.asarray(earth.time)
    earth_positions = np.asarray(earth.position)
    center = int(np.searchsorted(earth_times, 2459000.0))
    earth_times = earth_times[center - 8 : center + 9]
    earth_positions = earth_positions[center - 8 : center + 9]
    earth_velocity = np.asarray(earth.velocity)[center - 8 : center + 9]
    earth_query = (earth_times[:-1] + earth_times[1:]) / 2.0
    earth_reference = local_lagrange_reference(earth_times, earth_positions, earth_query)
    earth_linear = linear_interpolate(earth_times, earth_positions, earth_query)
    earth_hermite = hermite_interpolate(earth_times, earth_positions, earth_velocity, earth_query)
    print("earth_position_max_abs_error_AU", {
        "linear_vs_degree7": float(np.max(np.abs(earth_linear - earth_reference))),
        "hermite_vs_degree7": float(np.max(np.abs(earth_hermite - earth_reference))),
    })

    satellite_times = np.arange(-10.0, 11.0)
    satellite_values = np.stack((
        0.02 * np.sin(0.3 * satellite_times),
        0.015 * np.cos(0.2 * satellite_times),
        0.005 * np.sin(0.11 * satellite_times + 0.3),
    ), axis=-1)
    satellite_query = np.linspace(-9.999, 9.999, 5001)
    satellite_truth = np.stack((
        0.02 * np.sin(0.3 * satellite_query),
        0.015 * np.cos(0.2 * satellite_query),
        0.005 * np.sin(0.11 * satellite_query + 0.3),
    ), axis=-1)
    satellite_hermite = hermite_interpolate(
        satellite_times,
        satellite_values,
        three_point_derivatives(satellite_times, satellite_values),
        satellite_query,
    )
    print("spacecraft_position_max_abs_error_AU", {
        "linear": float(np.max(np.abs(linear_interpolate(satellite_times, satellite_values, satellite_query) - satellite_truth))),
        "hermite_estimated_tangent": float(np.max(np.abs(satellite_hermite - satellite_truth))),
    })

    benchmark_times = jnp.linspace(8990.001, 9009.999, args.points)
    earth_full = load_earth_ephemeris()
    annual_linear = jax.jit(lambda t: old_linear_annual(t, earth_full))
    annual_hermite = jax.jit(lambda t: annual_parallax_offsets(
        t, 0.12, -0.05, 270.0, -30.0, 9000.0, earth_full
    ))
    space_times = jnp.linspace(2458990.0, 2459010.0, 21)
    space_position = jnp.stack((
        0.01 * jnp.sin((space_times - 2459000.0) / 4.0),
        0.01 * jnp.cos((space_times - 2459000.0) / 5.0),
        0.002 * (space_times - 2459000.0),
    ), axis=-1)
    space_linear = jax.jit(lambda t: old_linear_space(t, space_times, space_position))
    space_hermite = jax.jit(lambda t: space_parallax_offsets(
        t, 0.12, -0.05, 270.0, -30.0, space_times, space_position
    ))
    annual_linear_time = measure(annual_linear, (benchmark_times,), args.repeat)
    annual_hermite_time = measure(annual_hermite, (benchmark_times,), args.repeat)
    space_linear_time = measure(space_linear, (benchmark_times,), args.repeat)
    space_hermite_time = measure(space_hermite, (benchmark_times,), args.repeat)
    print("jax_speed_ms", {
        "annual_linear": annual_linear_time,
        "annual_hermite": annual_hermite_time,
        "annual_hermite_over_linear": annual_hermite_time["median_ms"] / annual_linear_time["median_ms"],
        "space_linear": space_linear_time,
        "space_hermite": space_hermite_time,
        "space_hermite_over_linear": space_hermite_time["median_ms"] / space_linear_time["median_ms"],
    })


if __name__ == "__main__":
    main()

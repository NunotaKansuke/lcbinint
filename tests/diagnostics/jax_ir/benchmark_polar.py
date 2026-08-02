#!/usr/bin/env python3
"""Benchmark the experimental JAX polar path at a tiny high-mag epoch."""

import math
import time

import jax
import jax.numpy as jnp
import lcbinint
import numpy as np
from lcbinint_jax import binary_inverse_ray, binary_inverse_ray_polar
from microlux.basic_function import to_lowmass
from microlux.limb_darkening import LinearLimbDarkening
from microlux.trajectory_model import extended_light_curve_from_trajectory_l

jax.config.update("jax_enable_x64", True)


def timed(function, arguments, repeat=5):
    first = function(*arguments)
    jax.block_until_ready(first)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function(*arguments)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return float(first), float(np.median(samples))


def timed_only(function, arguments, repeat=5):
    first = function(*arguments)
    jax.block_until_ready(first)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function(*arguments)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main():
    separation = 0.95
    mass_ratio = 0.01
    source_radius = 0.005
    limb_c = 0.4
    u0 = -0.001
    alpha = 0.5
    epoch = 0.004
    source_x = epoch * math.cos(alpha) - u0 * math.sin(alpha)
    source_y = epoch * math.sin(alpha) + u0 * math.cos(alpha)
    parameters = jnp.asarray(
        (source_x, source_y, separation, mass_ratio, source_radius)
    )

    native = lcbinint.binary_ray_shooting(
        source_x,
        source_y,
        s=separation,
        q=mass_ratio,
        rho=source_radius,
        limb_darkening=lcbinint.LimbDarkening.linear(limb_c),
        options=lcbinint.Options(
            nbin=512,
            inverse_ray_grid="polar",
            coordinates="center_of_mass",
        ),
    )
    budget = 1.0e-4 + 1.0e-4 * max(abs(native), 1.0)
    print(f"native reference={native:.12f}, budget={budget:.6g}")

    for resolution, angular_bins, radial_capacity in (
        (64, 2048, 128),
        (64, 4096, 128),
        (128, 4096, 256),
        (128, 8192, 256),
    ):

        @jax.jit
        def polar(active):
            return binary_inverse_ray_polar(
                *active,
                limb_c,
                0.0,
                resolution=resolution,
                angular_bins=angular_bins,
                radial_capacity=radial_capacity,
                band_capacity=4,
                limb_samples=32,
                angular_chunk_size=min(1024, angular_bins),
                boundary_capacity=2048,
                boundary_subdivision=2,
                moment_mode="linear",
            ).magnification

        value, seconds = timed(polar, (parameters,))
        print(
            "polar",
            resolution,
            angular_bins,
            f"value={value:.12f}",
            f"error={abs(value - native):.6g}",
            f"passes={abs(value - native) <= budget}",
            f"time={1e3 * seconds:.3f} ms",
        )

    @jax.jit
    def cartesian(active):
        return binary_inverse_ray(
            *active,
            limb_c,
            0.0,
            resolution=64,
            tile_capacity=4096,
            limb_samples=64,
        )

    cartesian_result = cartesian(parameters)
    jax.block_until_ready(cartesian_result)
    print(
        "cartesian",
        f"value={float(cartesian_result.magnification):.12f}",
        f"support_valid={bool(cartesian_result.support_valid)}",
    )

    @jax.jit
    def microlux(active):
        x, y, s, q, rho = active
        trajectory = to_lowmass(s, q, jnp.atleast_1d(x + 1j * y))
        return extended_light_curve_from_trajectory_l(
            trajectory,
            s,
            q,
            rho,
            tol=1.0e-4,
            retol=1.0e-4,
            default_strategy=(30, 30, 60, 120, 240),
            analytic=True,
            limb_darkening=LinearLimbDarkening(limb_c),
            n_annuli=80,
        )[0]

    value, seconds = timed(microlux, (parameters,))
    print(
        "microLUX",
        f"value={value:.12f}",
        f"error={abs(value - native):.6g}",
        f"passes={abs(value - native) <= budget}",
        f"time={1e3 * seconds:.3f} ms",
    )

    direction = jnp.asarray((0.2, -0.1, 0.05, 0.02, 0.0))

    @jax.jit
    def selected_polar(active):
        return binary_inverse_ray_polar(
            *active,
            limb_c,
            0.0,
            resolution=64,
            angular_bins=4096,
            radial_capacity=128,
            band_capacity=4,
            limb_samples=32,
            angular_chunk_size=1024,
            boundary_capacity=1024,
            boundary_subdivision=2,
            moment_mode="linear",
        ).magnification

    polar_jvp = jax.jit(
        lambda active, tangent: jax.jvp(selected_polar, (active,), (tangent,))[1]
    )
    polar_grad = jax.jit(jax.grad(selected_polar))
    micro_jvp = jax.jit(
        lambda active, tangent: jax.jvp(microlux, (active,), (tangent,))[1]
    )
    micro_grad = jax.jit(jax.grad(microlux))
    print(
        "polar AD",
        f"jvp={1e3 * timed_only(polar_jvp, (parameters, direction)):.3f} ms",
        f"grad={1e3 * timed_only(polar_grad, (parameters,)):.3f} ms",
    )
    print(
        "microLUX AD",
        f"jvp={1e3 * timed_only(micro_jvp, (parameters, direction)):.3f} ms",
        f"grad={1e3 * timed_only(micro_grad, (parameters,)):.3f} ms",
    )


if __name__ == "__main__":
    main()

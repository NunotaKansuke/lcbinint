#!/usr/bin/env python3
"""Benchmark the fixed-support JAX inverse-ray hot kernel on CPU."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from lcbinint_jax import (  # noqa: E402
    binary_inverse_ray,
    binary_inverse_ray_fixed_support,
)


def covering_tiles(cell_size, tile_size, extent):
    tile_width = cell_size * tile_size
    starts = np.arange(-extent, extent, tile_width, dtype=np.float64)
    origins = np.asarray([(x, y) for y in starts for x in starts])
    return jnp.asarray(origins), jnp.ones(origins.shape[0], dtype=bool)


def timed_call(function, *args, repeat):
    compile_start = time.perf_counter()
    first = function(*args)
    jax.block_until_ready(first)
    compile_seconds = time.perf_counter() - compile_start

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function(*args)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return {
        "compile_and_first_seconds": compile_seconds,
        "median_seconds": float(np.median(samples)),
        "minimum_seconds": float(np.min(samples)),
        "samples_seconds": samples,
    }


def benchmark_configuration(tile_size, kernel, cell_size, extent, repeat):
    tile_origins, tile_mask = covering_tiles(cell_size, tile_size, extent)
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
    tangent = jnp.asarray([0.2, -0.1, 0.05, 0.02, -0.03, 0.1, -0.05])

    def value(active_parameters):
        source_x, source_y, separation, mass_ratio, rho, limb_c, limb_d = (
            active_parameters
        )
        return binary_inverse_ray_fixed_support(
            tile_origins,
            tile_mask,
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            rho,
            limb_c,
            limb_d,
            tile_size=tile_size,
            kernel=kernel,
        ).magnification

    forward = jax.jit(value)
    directional_jvp = jax.jit(
        lambda active_parameters, active_tangent: jax.jvp(
            value, (active_parameters,), (active_tangent,)
        )[1]
    )
    value_and_grad = jax.jit(jax.value_and_grad(value))

    forward_result = timed_call(forward, parameters, repeat=repeat)
    jvp_result = timed_call(directional_jvp, parameters, tangent, repeat=repeat)
    value_and_grad_result = timed_call(value_and_grad, parameters, repeat=repeat)
    ray_count = int(tile_origins.shape[0]) * tile_size * tile_size
    forward_result["million_rays_per_second"] = (
        ray_count / forward_result["median_seconds"] / 1.0e6
    )
    return {
        "tile_size": tile_size,
        "kernel": kernel,
        "cell_size": cell_size,
        "extent": extent,
        "tile_count": int(tile_origins.shape[0]),
        "ray_count": ray_count,
        "forward": forward_result,
        "directional_jvp": jvp_result,
        "value_and_grad": value_and_grad_result,
    }


def benchmark_automatic(resolution, tile_capacity, repeat):
    parameters = jnp.asarray([0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1])
    tangent = jnp.asarray([0.2, -0.1, 0.05, 0.02, 0.0, 0.1, -0.05])

    def value(active_parameters):
        return binary_inverse_ray(
            *active_parameters,
            resolution=resolution,
            tile_size=16,
            tile_capacity=tile_capacity,
            limb_samples=32,
            kernel="real",
        ).magnification

    forward = jax.jit(value)
    directional_jvp = jax.jit(
        lambda active_parameters, active_tangent: jax.jvp(
            value, (active_parameters,), (active_tangent,)
        )[1]
    )
    value_and_grad = jax.jit(jax.value_and_grad(value))
    diagnostics = binary_inverse_ray(
        *parameters,
        resolution=resolution,
        tile_size=16,
        tile_capacity=tile_capacity,
        limb_samples=32,
        kernel="real",
    )
    jax.block_until_ready(diagnostics.magnification)
    return {
        "resolution": resolution,
        "tile_size": 16,
        "tile_capacity": tile_capacity,
        "limb_samples": 32,
        "tile_count": int(diagnostics.tile_count),
        "boundary_cells": int(diagnostics.boundary_cells),
        "contributing_cells": int(diagnostics.active_cells),
        "support_valid": bool(diagnostics.support_valid),
        "forward": timed_call(forward, parameters, repeat=repeat),
        "directional_jvp": timed_call(
            directional_jvp, parameters, tangent, repeat=repeat
        ),
        "value_and_grad": timed_call(value_and_grad, parameters, repeat=repeat),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-size", type=float, default=0.02)
    parser.add_argument("--extent", type=float, default=2.0)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--skip-automatic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        },
        "configurations": [],
        "automatic": [],
    }
    for tile_size in (8, 16):
        for kernel in ("real", "complex"):
            report["configurations"].append(
                benchmark_configuration(
                    tile_size,
                    kernel,
                    args.cell_size,
                    args.extent,
                    args.repeat,
                )
            )
    if not args.skip_automatic:
        for resolution, capacity in ((16, 128), (32, 256), (64, 512)):
            report["automatic"].append(
                benchmark_automatic(resolution, capacity, args.repeat)
            )

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

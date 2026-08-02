#!/usr/bin/env python3
"""Compare the raw C++ and JAX fixed-support forward kernels on CPU."""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from lcbinint_jax import (  # noqa: E402
    binary_inverse_ray_fixed_support,
    binary_inverse_ray_fixed_support_cpp,
    binary_inverse_ray_fixed_support_ffi,
    discover_binary_macro_tiles,
)


def timed(function, repeat, block=False):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        if block:
            jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": float(np.median(samples)),
        "minimum_seconds": float(np.min(samples)),
        "samples_seconds": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()

    source_x = 0.5
    source_y = 0.0
    separation = 1.0
    mass_ratio = 0.1
    source_radius = 0.03
    limb_c = 0.4
    limb_d = 0.0
    resolution = 96
    tile_size = 16
    cell_size = source_radius / resolution

    discovery = discover_binary_macro_tiles(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        cell_size,
        tile_size=tile_size,
        tile_capacity=4096,
        limb_samples=24,
    )
    jax.block_until_ready(discovery.tile_origins)
    origins = np.asarray(discovery.tile_origins)
    mask = np.asarray(discovery.tile_mask)
    parameters = jnp.asarray(
        [
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        ]
    )
    tangent = jnp.asarray([0.2, -0.1, 0.05, 0.02, -0.03, 0.1, -0.05])

    def parameterized_value(function, active_parameters):
        return function(
            origins,
            mask,
            cell_size,
            *active_parameters,
            tile_size=tile_size,
            moment_mode="linear",
            boundary_subdivision=3,
        ).magnification

    def jax_forward():
        return binary_inverse_ray_fixed_support(
            jnp.asarray(origins),
            jnp.asarray(mask),
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            tile_size=tile_size,
            moment_mode="linear",
            boundary_subdivision=3,
        )

    def cpp_forward():
        return binary_inverse_ray_fixed_support_cpp(
            origins,
            mask,
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            tile_size=tile_size,
            moment_mode="linear",
            boundary_subdivision=3,
        )

    ffi_forward = jax.jit(
        lambda: binary_inverse_ray_fixed_support_ffi(
            origins,
            mask,
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            tile_size=tile_size,
            moment_mode="linear",
            boundary_subdivision=3,
        )
    )
    jax_jvp = jax.jit(
        lambda active, direction: jax.jvp(
            lambda values: parameterized_value(
                binary_inverse_ray_fixed_support, values
            ),
            (active,),
            (direction,),
        )[1]
    )
    ffi_jvp = jax.jit(
        lambda active, direction: jax.jvp(
            lambda values: parameterized_value(
                binary_inverse_ray_fixed_support_ffi, values
            ),
            (active,),
            (direction,),
        )[1]
    )
    jax_value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda active: parameterized_value(binary_inverse_ray_fixed_support, active)
        )
    )
    ffi_value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda active: parameterized_value(
                binary_inverse_ray_fixed_support_ffi, active
            )
        )
    )

    jax_reference = jax_forward()
    jax.block_until_ready(jax_reference)
    cpp_reference = cpp_forward()
    ffi_reference = ffi_forward()
    jax.block_until_ready(ffi_reference)
    jax_timing = timed(jax_forward, args.repeat, block=True)
    cpp_timing = timed(cpp_forward, args.repeat)
    ffi_timing = timed(ffi_forward, args.repeat, block=True)
    jax_jvp(parameters, tangent).block_until_ready()
    ffi_jvp(parameters, tangent).block_until_ready()
    jax.block_until_ready(jax_value_and_grad(parameters))
    jax.block_until_ready(ffi_value_and_grad(parameters))
    jax_jvp_timing = timed(
        lambda: jax_jvp(parameters, tangent), args.repeat, block=True
    )
    ffi_jvp_timing = timed(
        lambda: ffi_jvp(parameters, tangent), args.repeat, block=True
    )
    jax_grad_timing = timed(
        lambda: jax_value_and_grad(parameters), args.repeat, block=True
    )
    ffi_grad_timing = timed(
        lambda: ffi_value_and_grad(parameters), args.repeat, block=True
    )
    jax_jvp_value = jax_jvp(parameters, tangent)
    ffi_jvp_value = ffi_jvp(parameters, tangent)
    jax_gradient = jax_value_and_grad(parameters)[1]
    ffi_gradient = ffi_value_and_grad(parameters)[1]

    report = {
        "configuration": {
            "resolution": resolution,
            "tile_size": tile_size,
            "tile_capacity": 4096,
            "active_tiles": int(np.sum(mask)),
            "visited_tiles": int(discovery.visited_count),
            "boundary_cells": int(jax_reference.boundary_cells),
            "active_cells": int(jax_reference.active_cells),
        },
        "agreement": {
            "magnification_absolute_error": abs(
                float(jax_reference.magnification) - cpp_reference.magnification
            ),
            "moment_max_absolute_error": float(
                np.max(
                    np.abs(np.asarray(jax_reference.moments) - cpp_reference.moments)
                )
            ),
            "diagnostics_equal": (
                cpp_reference.boundary_cells == int(jax_reference.boundary_cells)
                and cpp_reference.active_cells == int(jax_reference.active_cells)
            ),
            "ffi_magnification_absolute_error": abs(
                float(jax_reference.magnification) - float(ffi_reference.magnification)
            ),
            "ffi_moment_max_absolute_error": float(
                np.max(
                    np.abs(
                        np.asarray(jax_reference.moments)
                        - np.asarray(ffi_reference.moments)
                    )
                )
            ),
            "ffi_jvp_absolute_error": abs(float(jax_jvp_value) - float(ffi_jvp_value)),
            "ffi_gradient_max_absolute_error": float(
                np.max(np.abs(np.asarray(jax_gradient - ffi_gradient)))
            ),
        },
        "jax": jax_timing,
        "cpp": cpp_timing,
        "ffi": ffi_timing,
        "cpp_speedup": (jax_timing["median_seconds"] / cpp_timing["median_seconds"]),
        "ffi_speedup": (jax_timing["median_seconds"] / ffi_timing["median_seconds"]),
        "jvp": {
            "jax": jax_jvp_timing,
            "ffi": ffi_jvp_timing,
            "ffi_speedup": (
                jax_jvp_timing["median_seconds"] / ffi_jvp_timing["median_seconds"]
            ),
        },
        "value_and_grad": {
            "jax": jax_grad_timing,
            "ffi": ffi_grad_timing,
            "ffi_speedup": (
                jax_grad_timing["median_seconds"] / ffi_grad_timing["median_seconds"]
            ),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

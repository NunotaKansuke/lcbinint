#!/usr/bin/env python3
"""Benchmark JAX and typed-FFI binary-image roots and hexadecapole."""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from lcbinint_jax import binary_hexadecapole, binary_images_ffi  # noqa: E402
from lcbinint_jax.multipole import (  # noqa: E402
    _differentiable_binary_image_roots,
)


def timed(function, repeat):
    compiled = jax.jit(function)
    result = compiled()
    jax.block_until_ready(result)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = compiled()
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "samples_seconds": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root_parameters = jnp.asarray((0.2, 0.1, 1.2, 0.1))
    root_tangent = jnp.asarray((0.2, -0.1, 0.05, 0.02))

    def pure_root(parameters):
        return _differentiable_binary_image_roots(
            parameters[0] + 1j * parameters[1],
            parameters[2],
            parameters[3],
        )

    def ffi_root(parameters):
        return binary_images_ffi(
            parameters[0] + 1j * parameters[1],
            parameters[2],
            parameters[3],
        ).roots

    angles = 2.0 * jnp.pi * jnp.arange(25) / 25
    sources = 0.2 + 0.1j + 0.02 * jnp.exp(1j * angles)
    hex_parameters = jnp.asarray((0.3, 0.4, 1.4, 1.0e-3, 0.01, 0.4, 0.0))
    hex_tangent = jnp.ones((7,))

    def hex_value(parameters, backend):
        return binary_hexadecapole(
            *parameters,
            root_backend=backend,
        ).magnification

    timings = {
        "jax_root_forward": timed(
            lambda: pure_root(root_parameters),
            args.repeat,
        ),
        "ffi_root_forward": timed(
            lambda: ffi_root(root_parameters),
            args.repeat,
        ),
        "jax_root_jvp": timed(
            lambda: jax.jvp(
                pure_root,
                (root_parameters,),
                (root_tangent,),
            ),
            args.repeat,
        ),
        "ffi_root_jvp": timed(
            lambda: jax.jvp(
                ffi_root,
                (root_parameters,),
                (root_tangent,),
            ),
            args.repeat,
        ),
        "jax_roots_25": timed(
            lambda: jax.vmap(
                lambda source: _differentiable_binary_image_roots(
                    source,
                    1.2,
                    0.1,
                )
            )(sources),
            args.repeat,
        ),
        "ffi_roots_25": timed(
            lambda: jax.vmap(lambda source: binary_images_ffi(source, 1.2, 0.1).roots)(
                sources
            ),
            args.repeat,
        ),
    }
    for backend in ("jax", "ffi"):
        timings[f"{backend}_hex_forward"] = timed(
            lambda backend=backend: hex_value(hex_parameters, backend),
            args.repeat,
        )
        timings[f"{backend}_hex_jvp"] = timed(
            lambda backend=backend: jax.jvp(
                lambda active: hex_value(active, backend),
                (hex_parameters,),
                (hex_tangent,),
            ),
            args.repeat,
        )
        timings[f"{backend}_hex_value_and_grad"] = timed(
            lambda backend=backend: jax.value_and_grad(
                lambda active: hex_value(active, backend)
            )(hex_parameters),
            args.repeat,
        )

    output = {"configuration": vars(args), "timings": timings}
    rendered = json.dumps(output, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

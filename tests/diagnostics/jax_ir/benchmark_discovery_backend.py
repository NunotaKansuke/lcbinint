#!/usr/bin/env python3
"""Benchmark JAX and typed-FFI Cartesian macro-tile discovery on CPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from lcbinint_jax import (  # noqa: E402
    discover_binary_macro_tiles,
    discover_binary_macro_tiles_ffi,
)


def timed(function, repeat):
    result = function()
    jax.block_until_ready(result.tile_origins)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        jax.block_until_ready(result.tile_origins)
        samples.append(time.perf_counter() - start)
    return result, {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "samples_seconds": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    parameters = (0.5, 0.0, 1.0, 0.1, 0.03)
    cell_size = parameters[4] / 96
    options = {
        "tile_size": 16,
        "tile_capacity": 4096,
        "limb_samples": 24,
    }
    pure_function = jax.jit(
        lambda: discover_binary_macro_tiles(
            *parameters,
            cell_size,
            **options,
        )
    )
    ffi_function = jax.jit(
        lambda: discover_binary_macro_tiles_ffi(
            *parameters,
            cell_size,
            **options,
        )
    )
    pure, pure_timing = timed(pure_function, args.repeat)
    ffi, ffi_timing = timed(ffi_function, args.repeat)
    exact = all(
        bool(jax.numpy.array_equal(getattr(pure, field), getattr(ffi, field)))
        for field in pure._fields
    )
    output = {
        "configuration": {
            "repeat": args.repeat,
            "parameters": parameters,
            "cell_size": cell_size,
            **options,
        },
        "timings": {
            "jax": pure_timing,
            "ffi": ffi_timing,
            "speedup": (pure_timing["median_seconds"] / ffi_timing["median_seconds"]),
        },
        "support": {
            "exact": exact,
            "visited_count": int(ffi.visited_count),
            "active_count": int(ffi.active_count),
            "seed_count": int(ffi.seed_count),
            "overflow": bool(ffi.overflow),
            "root_failure": bool(ffi.root_failure),
        },
    }
    rendered = json.dumps(output, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return int(not exact)


if __name__ == "__main__":
    raise SystemExit(main())

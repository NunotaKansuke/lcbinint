"""Separate the JAX pipeline's wrapper cost from its integrator cost.

The campaign's ext sweep reports the public ``Options(jax=True)`` light-curve
call against the native one and finds a large ratio.  That measurement is
correct about what a user pays and silent about why, because the two backends
run the *same* compiled Cartesian kernel: ``Options(jax=True)`` with
``nbin="auto"`` reaches ``binary_magnification_native_pipeline_trajectory``,
which calls ``binary_inverse_ray_cartesian_batch_ffi``.

This module measures the four quantities that decompose the ratio:

``kernel``
    One FFI call against native forced Cartesian on identical source positions
    at identical resolution, with routing disabled on both sides.  This is the
    integrator-against-integrator number.
``ladder``
    The same FFI resolution, evaluated the way the pipeline evaluates it: once
    per calibrated bucket under a data-dependent mask.  The mask must come from
    a traced argument, or XLA folds it away and deletes thirteen of the
    fourteen calls.
``blocks``
    The public call at several block lengths.  A per-call constant and a
    per-epoch cost are indistinguishable at one block length and obvious across
    several.
``jit``
    The pipeline as shipped against the same function wrapped in ``jax.jit``.

Everything is single-threaded on purpose: ``sweep_ext.py`` pins
``OMP_NUM_THREADS=1`` and one worker per core, and no native path threads
across the epochs of one trajectory (``batch_thread_count`` returns 1), so a
threaded FFI batch would be measuring the harness rather than the code.

    OMP_NUM_THREADS=1 taskset -c 57 python -m \\
        tests.diagnostics.recal2026.jax_kernel_audit --out audit.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

SEPARATION = 1.2
MASS_RATIO = 0.1
SOURCE_RADIUS = 0.02
IMPACT = 0.04
FIRST, LAST = 0.20, 0.36


def _positions(epochs):
    return np.linspace(FIRST, LAST, epochs)


def _median(call, repeat):
    call()
    samples = []
    for _ in range(max(repeat, 1)):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _block(call, repeat):
    """``_median`` for a JAX value, blocking on the result before stopping."""
    import jax

    def wrapped():
        jax.block_until_ready(call())

    return _median(wrapped, repeat)


def native_forced(nbin, limb_c, epochs, repeat):
    """Native forced Cartesian through the cached light-curve entry point.

    ``LightCurve`` is used rather than the scalar ``binary_ray_shooting``
    because the scalar entry rebuilds the lens geometry on every epoch.  The
    FFI batch rebuilds once for the whole block, so the cached path is the one
    that compares kernel against kernel; the scalar path is reported separately
    by ``--include-scalar`` because the design-era benchmark used it.
    """
    import lcbinint

    times = _positions(epochs)
    curve = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        nbin=nbin, inverse_ray_grid="cartesian", coordinates="center_of_mass",
        point_source_threshold=0.0, hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0))

    def call():
        return curve.info(
            times, t0=0.0, tE=1.0, u0=IMPACT, alpha=0.0, s=SEPARATION,
            q=MASS_RATIO, rho=SOURCE_RADIUS, limb_darkening_c=limb_c)

    values = np.asarray(call().finite_source_magnifications, float).ravel()
    return _median(call, repeat) / epochs, values


def native_scalar(nbin, limb_c, epochs, repeat):
    """The same grid through the per-epoch scalar entry point."""
    import lcbinint

    options = lcbinint.Options(
        nbin=nbin, inverse_ray_grid="cartesian", coordinates="center_of_mass",
        point_source_threshold=0.0, hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0)
    limb = lcbinint.LimbDarkening(c=limb_c, d=0.0)

    def call():
        return np.asarray([
            lcbinint.binary_ray_shooting(
                float(x), IMPACT, s=SEPARATION, q=MASS_RATIO,
                rho=SOURCE_RADIUS, limb_darkening=limb, options=options)
            for x in _positions(epochs)])

    return _median(call, repeat) / epochs, call()


def ffi_single(nbin, limb_c, epochs, repeat):
    """One masked Cartesian FFI call over the whole block."""
    import jax
    import jax.numpy as jnp
    from lcbinint_jax.cpp_backend import binary_inverse_ray_cartesian_batch_ffi
    from lcbinint_jax.trajectory import _CALIBRATED_EXECUTION_BUCKETS

    capacity = dict(_CALIBRATED_EXECUTION_BUCKETS)[nbin]
    x = jnp.asarray(_positions(epochs))
    y = jnp.full((epochs,), IMPACT)
    active = jnp.ones((epochs,), dtype=jnp.bool_)

    @jax.jit
    def run():
        return binary_inverse_ray_cartesian_batch_ffi(
            x, y, SEPARATION, MASS_RATIO, SOURCE_RADIUS, limb_c, 0.0,
            active=active, cell_size=SOURCE_RADIUS / nbin, tile_size=16,
            tile_capacity=capacity, limb_samples=32,
            moment_mode="uniform" if limb_c == 0.0 else "linear",
            boundary_subdivision=4).magnification

    values = np.asarray(run(), float)
    return _block(run, repeat) / epochs, values


def ffi_ladder(nbin, limb_c, epochs, repeat):
    """The same resolution, evaluated once per calibrated bucket.

    ``bucket_index`` is passed in rather than closed over so the mask stays
    data-dependent, as it is in the pipeline.  A Python constant here would let
    XLA drop every unselected call and report the ladder as free.
    """
    import jax
    import jax.numpy as jnp
    from lcbinint_jax.cpp_backend import binary_inverse_ray_cartesian_batch_ffi
    from lcbinint_jax.trajectory import _CALIBRATED_EXECUTION_BUCKETS

    target = [index for index, (resolution, _)
              in enumerate(_CALIBRATED_EXECUTION_BUCKETS)
              if resolution == nbin][0]
    x = jnp.asarray(_positions(epochs))
    y = jnp.full((epochs,), IMPACT)
    bucket_index = jnp.full((epochs,), target, dtype=jnp.int32)
    mode = "uniform" if limb_c == 0.0 else "linear"

    @jax.jit
    def run(bucket_index):
        results = []
        for index, (resolution, capacity) in enumerate(
                _CALIBRATED_EXECUTION_BUCKETS):
            needed = ((bucket_index == index) | (bucket_index == index - 1)
                      | (bucket_index == index + 1))
            results.append(binary_inverse_ray_cartesian_batch_ffi(
                x, y, SEPARATION, MASS_RATIO, SOURCE_RADIUS, limb_c, 0.0,
                active=needed, cell_size=SOURCE_RADIUS / resolution,
                tile_size=16, tile_capacity=capacity, limb_samples=32,
                moment_mode=mode, boundary_subdivision=4))
        stacked = jnp.stack(tuple(item.magnification for item in results))
        return jnp.take_along_axis(stacked, bucket_index[None, :], axis=0)[0]

    values = np.asarray(run(bucket_index), float)
    return _block(lambda: run(bucket_index), repeat) / epochs, values


def public_call(epochs, reltol, limb_c, repeat, *, use_jax):
    """The user-facing light-curve call on either backend."""
    import lcbinint

    times = _positions(epochs)
    options = dict(coordinates="vbm", nbin="auto", caustic_bins=1400,
                   max_source_bins=400, reltol=reltol)
    if use_jax:
        options["jax"] = True
    curve = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(**options))

    def call():
        return np.asarray(curve.magnification(
            times, t0=0.0, tE=1.0, u0=IMPACT, alpha=0.0, s=SEPARATION,
            q=MASS_RATIO, rho=SOURCE_RADIUS, limb_darkening_c=limb_c), float)

    started = time.perf_counter()
    call()
    first = time.perf_counter() - started
    return _median(call, repeat), first, call()


def jit_comparison(epochs, reltol, limb_c, repeat):
    """The pipeline as shipped against the identical function under ``jax.jit``.

    ``binary_magnification_native_pipeline_trajectory`` carries no ``jax.jit``
    and ``jax_backend`` calls it directly, so each call re-traces fourteen FFI
    calls, two ``lax.map`` bodies and a fourteen-way ``lax.switch`` and
    dispatches them one primitive at a time.
    """
    import jax
    import jax.numpy as jnp
    from lcbinint_jax.trajectory import (
        binary_magnification_native_pipeline_trajectory)

    x = jnp.asarray(_positions(epochs))
    y = jnp.full((epochs,), IMPACT)

    def call(x, y):
        return binary_magnification_native_pipeline_trajectory(
            x, y, SEPARATION, MASS_RATIO, SOURCE_RADIUS, limb_c, 0.0,
            absolute_tolerance=reltol, relative_tolerance=reltol,
            caustic_bins=1400, maximum_source_bins=400).magnification

    jitted = jax.jit(call)
    eager = _block(lambda: call(x, y), repeat)
    started = time.perf_counter()
    jax.block_until_ready(jitted(x, y))
    compile_seconds = time.perf_counter() - started
    fast = _block(lambda: jitted(x, y), repeat)
    eager_values = np.asarray(call(x, y), float)
    jit_values = np.asarray(jitted(x, y), float)
    gap = float(np.nanmax(np.abs(jit_values - eager_values)
                          / np.maximum(np.abs(eager_values), 1.0)))
    return {
        "eager_seconds": eager,
        "jit_seconds": fast,
        "speedup": eager / fast,
        "compile_seconds": compile_seconds,
        "value_gap": gap,
    }


def _relative(left, right):
    return float(np.nanmax(np.abs(left - right) / np.maximum(np.abs(right), 1.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limb-c", type=float, default=0.0)
    parser.add_argument("--reltol", type=float, default=1.0e-3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=24,
                        help="block length for the kernel and ladder tables")
    parser.add_argument("--resolutions", default="16,32,64,128")
    parser.add_argument("--block-lengths", default="1,6,24,96,384,1536")
    parser.add_argument("--include-scalar", action="store_true",
                        help="also time the per-epoch native entry point")
    parser.add_argument("--out")
    arguments = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    report = {"configuration": vars(arguments), "kernel": {}, "blocks": {},
              "jit": {}}

    for nbin in (int(part) for part in arguments.resolutions.split(",")):
        native, native_values = native_forced(
            nbin, arguments.limb_c, arguments.epochs, arguments.repeat)
        single, single_values = ffi_single(
            nbin, arguments.limb_c, arguments.epochs, arguments.repeat)
        ladder, ladder_values = ffi_ladder(
            nbin, arguments.limb_c, arguments.epochs, arguments.repeat)
        entry = {
            "native_seconds_per_epoch": native,
            "ffi_seconds_per_epoch": single,
            "ladder_seconds_per_epoch": ladder,
            "ffi_over_native": single / native,
            "ladder_over_ffi": ladder / single,
            "ffi_native_value_gap": _relative(single_values, native_values),
            "ladder_ffi_value_gap": _relative(ladder_values, single_values),
        }
        if arguments.include_scalar:
            scalar, _ = native_scalar(
                nbin, arguments.limb_c, arguments.epochs, arguments.repeat)
            entry["native_scalar_seconds_per_epoch"] = scalar
            entry["scalar_over_cached"] = scalar / native
        report["kernel"][nbin] = entry
        print(f"kernel nbin={nbin}: {json.dumps(entry)}", flush=True)

    for epochs in (int(part) for part in arguments.block_lengths.split(",")):
        native, _, native_values = public_call(
            epochs, arguments.reltol, arguments.limb_c, arguments.repeat,
            use_jax=False)
        jax_seconds, first, jax_values = public_call(
            epochs, arguments.reltol, arguments.limb_c, arguments.repeat,
            use_jax=True)
        entry = {
            "native_seconds": native,
            "jax_seconds": jax_seconds,
            "jax_seconds_per_epoch": jax_seconds / epochs,
            "ratio": jax_seconds / native,
            "jax_first_call_seconds": first,
            "value_gap": _relative(jax_values, native_values),
        }
        report["blocks"][epochs] = entry
        print(f"block epochs={epochs}: {json.dumps(entry)}", flush=True)

    for epochs in (1, arguments.epochs, 96, 384):
        entry = jit_comparison(
            epochs, arguments.reltol, arguments.limb_c, arguments.repeat)
        report["jit"][epochs] = entry
        print(f"jit epochs={epochs}: {json.dumps(entry)}", flush=True)

    text = json.dumps(report, indent=2)
    if arguments.out:
        with open(arguments.out, "w") as handle:
            handle.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()

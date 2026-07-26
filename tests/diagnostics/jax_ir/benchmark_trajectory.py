#!/usr/bin/env python3
"""Benchmark the conditional JAX trajectory dispatcher on CPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import VBMicrolensing  # noqa: E402
import lcbinint  # noqa: E402
from lcbinint_jax import binary_magnification_trajectory  # noqa: E402
from microlux.basic_function import to_lowmass  # noqa: E402
from microlux.limb_darkening import LinearLimbDarkening  # noqa: E402
from microlux.trajectory_model import (  # noqa: E402
    extended_light_curve_from_trajectory_l,
)

SEPARATION = 1.2
MASS_RATIO = 0.1
SOURCE_RADIUS = 0.02
LIMB_C = 0.4


def timed_jax(function, repeat):
    start = time.perf_counter()
    first = function()
    jax.block_until_ready(first)
    compile_and_first = time.perf_counter() - start
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return first, {
        "compile_and_first_seconds": compile_and_first,
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
    }


def timed_python(function, repeat):
    function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, {
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
    }


def positions(count):
    return (
        jnp.linspace(0.2, 1.0, count),
        jnp.full((count,), 0.04),
    )


def jax_curve(source_x, source_y, separation=SEPARATION, backend="jax"):
    return binary_magnification_trajectory(
        source_x,
        source_y,
        separation,
        MASS_RATIO,
        SOURCE_RADIUS,
        LIMB_C,
        0.0,
        moment_mode="linear",
        cartesian_backend=backend,
    )


def microlux_curve(source_x, source_y, separation=SEPARATION):
    trajectory = to_lowmass(separation, MASS_RATIO, source_x + 1j * source_y)
    return extended_light_curve_from_trajectory_l(
        trajectory,
        separation,
        MASS_RATIO,
        SOURCE_RADIUS,
        tol=1.0e-4,
        retol=1.0e-4,
        default_strategy=(30, 30, 60, 120, 240),
        analytic=True,
        limb_darkening=LinearLimbDarkening(LIMB_C),
        n_annuli=80,
    )


def accuracy_record(values, reference):
    errors = np.abs(np.asarray(values) - reference)
    budgets = 1.0e-4 + 1.0e-4 * np.maximum(np.abs(reference), 1.0)
    return {
        "finite": int(np.count_nonzero(np.isfinite(values))),
        "failures": int(np.count_nonzero(errors > budgets)),
        "median_budget_ratio": float(np.nanmedian(errors / budgets)),
        "max_budget_ratio": float(np.nanmax(errors / budgets)),
        "max_absolute_error": float(np.nanmax(errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--ad-epochs", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_x, source_y = positions(args.epochs)
    jax_result, jax_forward = timed_jax(
        lambda: jax_curve(source_x, source_y), args.repeat
    )
    ffi_result, ffi_forward = timed_jax(
        lambda: jax_curve(source_x, source_y, backend="ffi"), args.repeat
    )
    microlux_result, microlux_forward = timed_jax(
        lambda: microlux_curve(source_x, source_y), args.repeat
    )

    ad_x, ad_y = positions(args.ad_epochs)
    _, jax_jvp = timed_jax(
        lambda: jax.jvp(
            lambda separation: jax_curve(ad_x, ad_y, separation).magnification,
            (SEPARATION,),
            (1.0,),
        ),
        args.repeat,
    )
    _, jax_gradient = timed_jax(
        lambda: jax.value_and_grad(
            lambda separation: jnp.nansum(
                jax_curve(ad_x, ad_y, separation).magnification
            )
        )(SEPARATION),
        args.repeat,
    )
    _, ffi_jvp = timed_jax(
        lambda: jax.jvp(
            lambda separation: jax_curve(
                ad_x, ad_y, separation, backend="ffi"
            ).magnification,
            (SEPARATION,),
            (1.0,),
        ),
        args.repeat,
    )
    _, ffi_gradient = timed_jax(
        lambda: jax.value_and_grad(
            lambda separation: jnp.nansum(
                jax_curve(ad_x, ad_y, separation, backend="ffi").magnification
            )
        )(SEPARATION),
        args.repeat,
    )
    _, microlux_jvp = timed_jax(
        lambda: jax.jvp(
            lambda separation: microlux_curve(ad_x, ad_y, separation),
            (SEPARATION,),
            (1.0,),
        ),
        1,
    )
    _, microlux_gradient = timed_jax(
        lambda: jax.value_and_grad(
            lambda separation: jnp.sum(microlux_curve(ad_x, ad_y, separation))
        )(SEPARATION),
        1,
    )

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1.0e-7
    vbm.a1 = LIMB_C

    def vbm_curve():
        return np.asarray(
            [
                vbm.BinaryMagDark(
                    SEPARATION,
                    MASS_RATIO,
                    float(x),
                    float(y),
                    SOURCE_RADIUS,
                    vbm.Tol,
                )
                for x, y in zip(source_x, source_y)
            ]
        )

    reference, vbm_forward = timed_python(vbm_curve, args.repeat)

    native_options = lcbinint.Options(
        nbin=64,
        inverse_ray_grid="cartesian",
        coordinates="center_of_mass",
    )
    limb_darkening = lcbinint.LimbDarkening(c=LIMB_C, d=0.0)

    def native_curve():
        return np.asarray(
            [
                lcbinint.binary_ray_shooting(
                    float(x),
                    float(y),
                    s=SEPARATION,
                    q=MASS_RATIO,
                    rho=SOURCE_RADIUS,
                    limb_darkening=limb_darkening,
                    options=native_options,
                )
                for x, y in zip(source_x, source_y)
            ]
        )

    native_result, native_forward = timed_python(native_curve, args.repeat)

    output = {
        "configuration": vars(args)
        | {
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "separation": SEPARATION,
            "mass_ratio": MASS_RATIO,
            "source_radius": SOURCE_RADIUS,
            "limb_c": LIMB_C,
        },
        "timings": {
            "jax_forward": jax_forward,
            "jax_jvp": jax_jvp,
            "jax_value_and_grad": jax_gradient,
            "ffi_forward": ffi_forward,
            "ffi_jvp": ffi_jvp,
            "ffi_value_and_grad": ffi_gradient,
            "microlux_forward": microlux_forward,
            "microlux_jvp": microlux_jvp,
            "microlux_value_and_grad": microlux_gradient,
            "native_forward": native_forward,
            "vbm_forward": vbm_forward,
        },
        "accuracy": {
            "jax": accuracy_record(jax_result.magnification, reference),
            "ffi": accuracy_record(ffi_result.magnification, reference),
            "microlux": accuracy_record(microlux_result, reference),
            "native": accuracy_record(native_result, reference),
        },
        "jax_dispatch": {
            "attempted_counts": np.asarray(jax_result.attempted_counts).tolist(),
            "method_counts": {
                str(method): int(jnp.sum(jax_result.method == method))
                for method in range(4)
            },
            "invalid": int(jnp.sum(~jax_result.support_valid)),
        },
        "ffi_dispatch": {
            "attempted_counts": np.asarray(ffi_result.attempted_counts).tolist(),
            "method_counts": {
                str(method): int(jnp.sum(ffi_result.method == method))
                for method in range(4)
            },
            "invalid": int(jnp.sum(~ffi_result.support_valid)),
        },
    }
    text = json.dumps(output, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

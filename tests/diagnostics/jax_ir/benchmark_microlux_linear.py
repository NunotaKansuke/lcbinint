#!/usr/bin/env python3
"""Matched-accuracy CPU benchmark against microLUX linear limb darkening."""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import VBMicrolensing  # noqa: E402
import microlux  # noqa: E402
from lcbinint_jax import binary_inverse_ray_linear  # noqa: E402
from microlux.basic_function import to_lowmass  # noqa: E402
from microlux.limb_darkening import LinearLimbDarkening  # noqa: E402
from microlux.trajectory_model import (  # noqa: E402
    extended_light_curve_from_trajectory_l,
)

DIRECTION = jnp.asarray((0.2, -0.1, 0.05, 0.02, 0.0))
LIMB_C = 0.4
CASES = {
    "regular": {
        "parameters": (0.2, 0.1, 1.2, 0.1, 0.2),
        "jax_configurations": (
            (16, 128),
            (32, 256),
            (64, 512),
            (128, 2048),
        ),
        "microlux_annuli": 10,
        "reference": "vbmicrolensing",
    },
    "resonant_cusp": {
        "parameters": (0.653, 0.0, 1.2, 0.1, 0.02),
        "jax_configurations": (
            (16, 256),
            (32, 512),
            (64, 1024),
            (128, 4096),
        ),
        "microlux_annuli": 80,
        "reference": "native-consensus",
        "reference_value": 9.124999559851442,
    },
}


def timed_jax(function, arguments, repeat):
    start = time.perf_counter()
    first = function(*arguments)
    jax.block_until_ready(first)
    compile_and_first = time.perf_counter() - start
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function(*arguments)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return {
        "compile_and_first_seconds": compile_and_first,
        "median_seconds": float(np.median(samples)),
        "minimum_seconds": float(np.min(samples)),
        "samples_seconds": samples,
    }


def reference_magnification(case, parameters):
    if case["reference"] == "native-consensus":
        return (
            "native lcbinint, 768 bins; JAX/microLUX convergence consensus",
            case["reference_value"],
        )
    x, y, separation, mass_ratio, radius = map(float, parameters)
    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = 1.0e-8
    engine.a1 = LIMB_C
    return (
        "VBMicrolensing BinaryMagDark, requested accuracy 1e-8",
        engine.BinaryMagDark(
            separation,
            mass_ratio,
            x,
            y,
            radius,
            engine.Tol,
        ),
    )


def inverse_ray_value(parameters, resolution, capacity):
    x, y, separation, mass_ratio, radius = parameters
    return binary_inverse_ray_linear(
        x,
        y,
        separation,
        mass_ratio,
        radius,
        LIMB_C,
        resolution=resolution,
        tile_size=16,
        tile_capacity=capacity,
        limb_samples=32,
    ).magnification


def microlux_value(parameters, analytic, n_annuli):
    x, y, separation, mass_ratio, radius = parameters
    trajectory_low_mass = to_lowmass(
        separation,
        mass_ratio,
        jnp.atleast_1d(x + 1j * y),
    )
    return extended_light_curve_from_trajectory_l(
        trajectory_low_mass,
        separation,
        mass_ratio,
        radius,
        tol=1.0e-4,
        retol=1.0e-4,
        default_strategy=(30, 30, 60, 120, 240),
        analytic=analytic,
        limb_darkening=LinearLimbDarkening(LIMB_C),
        n_annuli=n_annuli,
    )[0]


def checkout_commit(module):
    repository = Path(module.__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), default="regular")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    case = CASES[args.case]
    parameters = jnp.asarray(case["parameters"])
    reference_engine, reference = reference_magnification(case, parameters)
    budget = args.atol + args.rtol * max(abs(reference), 1.0)
    calibration = []
    selected = None
    for resolution, capacity in case["jax_configurations"]:
        value_function = jax.jit(
            lambda parameters, resolution=resolution, capacity=capacity: (
                inverse_ray_value(parameters, resolution, capacity)
            )
        )
        value = value_function(parameters)
        jax.block_until_ready(value)
        row = {
            "resolution": resolution,
            "tile_capacity": capacity,
            "value": float(value),
            "absolute_error": abs(float(value) - reference),
        }
        row["passes"] = row["absolute_error"] <= budget
        calibration.append(row)
        if selected is None and row["passes"]:
            selected = row
    if selected is None:
        raise RuntimeError("no inverse-ray configuration met the error budget")

    resolution = selected["resolution"]
    capacity = selected["tile_capacity"]

    def inverse(parameters):
        return inverse_ray_value(parameters, resolution, capacity)

    def micro_analytic(parameters):
        return microlux_value(parameters, True, case["microlux_annuli"])

    def micro_forward_only(parameters):
        return microlux_value(parameters, False, case["microlux_annuli"])

    inverse_forward = jax.jit(inverse)
    inverse_jvp = jax.jit(
        lambda parameters, direction: jax.jvp(
            inverse,
            (parameters,),
            (direction,),
        )[1]
    )
    inverse_gradient = jax.jit(jax.value_and_grad(inverse))
    micro_forward = jax.jit(micro_analytic)
    micro_forward_fast = jax.jit(micro_forward_only)
    micro_jvp = jax.jit(
        lambda parameters, direction: jax.jvp(
            micro_analytic,
            (parameters,),
            (direction,),
        )[1]
    )
    micro_gradient = jax.jit(jax.value_and_grad(micro_analytic))

    micro_forward_timing = timed_jax(
        micro_forward,
        (parameters,),
        args.repeat,
    )
    micro_value = micro_forward(parameters)
    jax.block_until_ready(micro_value)
    micro_error = abs(float(micro_value) - reference)
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
            "microlux_path": str(Path(microlux.__file__).resolve()),
            "microlux_commit": checkout_commit(microlux),
        },
        "case": args.case,
        "parameters": {
            "source_x": float(parameters[0]),
            "source_y": float(parameters[1]),
            "separation": float(parameters[2]),
            "mass_ratio": float(parameters[3]),
            "source_radius": float(parameters[4]),
            "linear_limb_c": LIMB_C,
        },
        "reference": {
            "engine": reference_engine,
            "magnification": reference,
        },
        "error_budget": budget,
        "inverse_ray_calibration": calibration,
        "inverse_ray_selected": {
            "resolution": resolution,
            "tile_capacity": capacity,
            "forward": timed_jax(inverse_forward, (parameters,), args.repeat),
            "directional_jvp": timed_jax(
                inverse_jvp,
                (parameters, DIRECTION),
                args.repeat,
            ),
            "value_and_grad": timed_jax(
                inverse_gradient,
                (parameters,),
                args.repeat,
            ),
        },
        "microlux": {
            "tol": 1.0e-4,
            "retol": 1.0e-4,
            "n_annuli": case["microlux_annuli"],
            "value": float(micro_value),
            "absolute_error": micro_error,
            "passes": micro_error <= budget,
            "forward_analytic": micro_forward_timing,
            "forward_only_nonanalytic": timed_jax(
                micro_forward_fast,
                (parameters,),
                args.repeat,
            ),
            "directional_jvp": timed_jax(
                micro_jvp,
                (parameters, DIRECTION),
                args.repeat,
            ),
            "value_and_grad": timed_jax(
                micro_gradient,
                (parameters,),
                args.repeat,
            ),
        },
    }
    inverse_timings = report["inverse_ray_selected"]
    micro_timings = report["microlux"]
    report["inverse_ray_speedup_over_microlux"] = {
        "forward_vs_analytic": (
            micro_timings["forward_analytic"]["median_seconds"]
            / inverse_timings["forward"]["median_seconds"]
        ),
        "forward_vs_nonanalytic": (
            micro_timings["forward_only_nonanalytic"]["median_seconds"]
            / inverse_timings["forward"]["median_seconds"]
        ),
        "directional_jvp": (
            micro_timings["directional_jvp"]["median_seconds"]
            / inverse_timings["directional_jvp"]["median_seconds"]
        ),
        "value_and_grad": (
            micro_timings["value_and_grad"]["median_seconds"]
            / inverse_timings["value_and_grad"]["median_seconds"]
        ),
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

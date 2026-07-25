#!/usr/bin/env python3
"""Position the JAX inverse-ray solver against the main CPU alternatives.

The benchmark first calibrates each tunable inverse-ray implementation against
the same absolute error budget. Timings are warm scalar CPU calls and exclude
JIT compilation. Native lcbinint and VBMicrolensing are forward-only entries;
JAX and microLUX additionally report exact-program JVP and reverse-mode costs.
"""

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
import lcbinint  # noqa: E402
import microlux  # noqa: E402
from lcbinint_jax import (  # noqa: E402
    binary_inverse_ray_linear,
    binary_inverse_ray_uniform,
)
from microlux.basic_function import to_lowmass  # noqa: E402
from microlux.limb_darkening import LinearLimbDarkening  # noqa: E402
from microlux.trajectory_model import (  # noqa: E402
    extended_light_curve_from_trajectory_l,
)

LIMB_C = 0.4
DIRECTION = jnp.asarray((0.2, -0.1, 0.05, 0.02, 0.0))
JAX_CONFIGURATIONS = (
    (16, 256),
    (32, 512),
    (64, 1024),
    (128, 4096),
)
NATIVE_BINS = (16, 32, 48, 64, 96, 128, 192, 256, 512)
CASES = {
    "regular": {
        "parameters": (0.2, 0.1, 1.2, 0.1, 0.2),
        "microlux_annuli": 10,
    },
    "resonant_cusp": {
        "parameters": (0.653, 0.0, 1.2, 0.1, 0.02),
        "microlux_annuli": 80,
        "linear_reference": 9.124998890920624,
        "linear_reference_engine": (
            "native lcbinint 1536 bins, supported by JAX/microLUX convergence"
        ),
    },
    "planetary_far": {
        "parameters": (0.3, 0.4, 1.4, 1.0e-3, 0.01),
        "microlux_annuli": 10,
    },
    "planetary_cusp": {
        "parameters": (0.7276663, 0.0, 1.4, 1.0e-3, 0.005),
        "microlux_annuli": 80,
        "linear_reference": 4.996458808527914,
        "linear_reference_engine": (
            "native lcbinint 1536 bins, supported by JAX convergence"
        ),
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


def timed_python(function, repeat, inner):
    function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        samples.append((time.perf_counter() - start) / inner)
    return {
        "median_seconds": float(np.median(samples)),
        "minimum_seconds": float(np.min(samples)),
        "samples_seconds": samples,
    }


def reference_value(case, profile):
    if profile == "linear" and "linear_reference" in case:
        return case["linear_reference_engine"], case["linear_reference"]

    x, y, separation, mass_ratio, radius = case["parameters"]
    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = 1.0e-9
    if profile == "uniform":
        engine.a1 = 0.0
        value = engine.BinaryMag2(separation, mass_ratio, x, y, radius)
        label = "VBMicrolensing BinaryMag2, requested accuracy 1e-9"
    else:
        engine.a1 = LIMB_C
        value = engine.BinaryMagDark(separation, mass_ratio, x, y, radius, engine.Tol)
        label = "VBMicrolensing BinaryMagDark, requested accuracy 1e-9"
    return label, value


def inverse_ray_value(parameters, profile, resolution, capacity):
    if profile == "uniform":
        return binary_inverse_ray_uniform(
            *parameters,
            resolution=resolution,
            tile_size=16,
            tile_capacity=capacity,
            limb_samples=32,
        ).magnification
    return binary_inverse_ray_linear(
        *parameters,
        LIMB_C,
        resolution=resolution,
        tile_size=16,
        tile_capacity=capacity,
        limb_samples=32,
    ).magnification


def native_function(parameters, profile, source_bins):
    options = lcbinint.Options(
        nbin=source_bins,
        inverse_ray_grid="cartesian",
        coordinates="center_of_mass",
    )
    limb_c = 0.0 if profile == "uniform" else LIMB_C
    limb_darkening = lcbinint.LimbDarkening(c=limb_c, d=0.0)
    x, y, separation, mass_ratio, radius = map(float, parameters)
    return lambda: lcbinint.binary_ray_shooting(
        x,
        y,
        s=separation,
        q=mass_ratio,
        rho=radius,
        limb_darkening=limb_darkening,
        options=options,
    )


def vbm_function(parameters, profile, budget):
    x, y, separation, mass_ratio, radius = map(float, parameters)
    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = budget
    if profile == "uniform":
        engine.a1 = 0.0
        return lambda: engine.BinaryMag2(separation, mass_ratio, x, y, radius)
    engine.a1 = LIMB_C
    return lambda: engine.BinaryMagDark(separation, mass_ratio, x, y, radius, budget)


def microlux_value(parameters, profile, analytic, n_annuli):
    x, y, separation, mass_ratio, radius = parameters
    trajectory = to_lowmass(
        separation,
        mass_ratio,
        jnp.atleast_1d(x + 1j * y),
    )
    limb_darkening = None if profile == "uniform" else LinearLimbDarkening(LIMB_C)
    return extended_light_curve_from_trajectory_l(
        trajectory,
        separation,
        mass_ratio,
        radius,
        tol=1.0e-4,
        retol=1.0e-4,
        default_strategy=(30, 30, 60, 120, 240),
        analytic=analytic,
        limb_darkening=limb_darkening,
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


def select_first_passing(rows):
    return next((row for row in rows if row["passes"]), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), default="regular")
    parser.add_argument("--profile", choices=("uniform", "linear"), default="uniform")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    case = CASES[args.case]
    parameters = jnp.asarray(case["parameters"])
    reference_engine, reference = reference_value(case, args.profile)
    budget = args.atol + args.rtol * max(abs(reference), 1.0)

    jax_calibration = []
    for resolution, capacity in JAX_CONFIGURATIONS:
        value = inverse_ray_value(parameters, args.profile, resolution, capacity)
        jax.block_until_ready(value)
        error = abs(float(value) - reference)
        jax_calibration.append(
            {
                "resolution": resolution,
                "tile_capacity": capacity,
                "value": float(value),
                "absolute_error": error,
                "passes": error <= budget,
            }
        )
    selected_jax = select_first_passing(jax_calibration)
    if selected_jax is None:
        raise RuntimeError("no JAX inverse-ray configuration met the error budget")

    native_calibration = []
    for source_bins in NATIVE_BINS:
        value = native_function(parameters, args.profile, source_bins)()
        error = abs(value - reference)
        native_calibration.append(
            {
                "source_bins": source_bins,
                "value": value,
                "absolute_error": error,
                "passes": error <= budget,
            }
        )
    selected_native = select_first_passing(native_calibration)

    resolution = selected_jax["resolution"]
    capacity = selected_jax["tile_capacity"]

    def inverse(active):
        return inverse_ray_value(active, args.profile, resolution, capacity)

    inverse_forward = jax.jit(inverse)
    inverse_jvp = jax.jit(
        lambda active, direction: jax.jvp(inverse, (active,), (direction,))[1]
    )
    inverse_gradient = jax.jit(jax.value_and_grad(inverse))

    n_annuli = case["microlux_annuli"]

    def micro(active):
        return microlux_value(active, args.profile, True, n_annuli)

    def micro_fast(active):
        return microlux_value(active, args.profile, False, n_annuli)

    micro_forward = jax.jit(micro)
    micro_forward_fast = jax.jit(micro_fast)
    micro_jvp = jax.jit(
        lambda active, direction: jax.jvp(micro, (active,), (direction,))[1]
    )
    micro_gradient = jax.jit(jax.value_and_grad(micro))
    micro_forward_timing = timed_jax(micro_forward, (parameters,), args.repeat)
    micro_value = micro_forward(parameters)
    jax.block_until_ready(micro_value)
    micro_error = abs(float(micro_value) - reference)

    vbm = vbm_function(parameters, args.profile, budget)
    vbm_value = vbm()
    vbm_error = abs(vbm_value - reference)

    native_report = {
        "calibration": native_calibration,
        "selected_source_bins": None,
        "forward": None,
    }
    if selected_native is not None:
        native = native_function(
            parameters, args.profile, selected_native["source_bins"]
        )
        native_report["selected_source_bins"] = selected_native["source_bins"]
        native_report["forward"] = timed_python(native, args.repeat, args.inner)

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
        "profile": args.profile,
        "linear_limb_c": LIMB_C if args.profile == "linear" else None,
        "parameters": list(map(float, parameters)),
        "reference": {"engine": reference_engine, "value": reference},
        "error_budget": budget,
        "selection_rule": (
            "first increasing-resolution/bin candidate satisfying the common "
            "absolute error budget"
        ),
        "jax_inverse_ray": {
            "calibration": jax_calibration,
            "selected_resolution": resolution,
            "selected_tile_capacity": capacity,
            "forward": timed_jax(inverse_forward, (parameters,), args.repeat),
            "directional_jvp": timed_jax(
                inverse_jvp, (parameters, DIRECTION), args.repeat
            ),
            "value_and_grad": timed_jax(inverse_gradient, (parameters,), args.repeat),
        },
        "microlux": {
            "tol": 1.0e-4,
            "retol": 1.0e-4,
            "n_annuli": n_annuli if args.profile == "linear" else None,
            "value": float(micro_value),
            "absolute_error": micro_error,
            "passes": micro_error <= budget,
            "forward_analytic": micro_forward_timing,
            "forward_only_nonanalytic": timed_jax(
                micro_forward_fast, (parameters,), args.repeat
            ),
            "directional_jvp": timed_jax(
                micro_jvp, (parameters, DIRECTION), args.repeat
            ),
            "value_and_grad": timed_jax(micro_gradient, (parameters,), args.repeat),
        },
        "native_lcbinint": native_report,
        "vbmicrolensing": {
            "requested_accuracy": budget,
            "value": vbm_value,
            "absolute_error": vbm_error,
            "passes": vbm_error <= budget,
            "forward": timed_python(vbm, args.repeat, args.inner),
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

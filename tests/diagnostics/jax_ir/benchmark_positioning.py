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
from lcbinint_jax import (  # noqa: E402
    binary_magnification_auto,
    binary_magnification_calibrated,
    binary_inverse_ray_linear,
    binary_inverse_ray_uniform,
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


def hybrid_value(parameters, profile, resolution, capacity):
    limb_c = 0.0 if profile == "uniform" else LIMB_C
    return binary_magnification_auto(
        *parameters,
        limb_c,
        0.0,
        absolute_tolerance=1.0e-4,
        relative_tolerance=1.0e-4,
        resolution=resolution,
        tile_size=16,
        tile_capacity=capacity,
        limb_samples=16,
        moment_mode=profile,
    )


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


def native_auto_function(parameters, profile, atol, rtol):
    options = lcbinint.Options(
        nbin="auto",
        inverse_ray_grid="auto",
        coordinates="center_of_mass",
        finite_source_tol=atol,
        finite_source_reltol=rtol,
    )
    limb_c = 0.0 if profile == "uniform" else LIMB_C
    limb_darkening = lcbinint.LimbDarkening(c=limb_c, d=0.0)
    x, y, separation, mass_ratio, radius = map(float, parameters)

    def evaluate():
        return lcbinint.binary_ray_shooting(
            x,
            y,
            s=separation,
            q=mass_ratio,
            rho=radius,
            limb_darkening=limb_darkening,
            options=options,
        )

    diagnostics_curve = lcbinint.LightCurve(
        options=options,
        limb_darkening=limb_darkening,
    )
    diagnostics = diagnostics_curve.info(
        [x],
        t0=0.0,
        tE=1.0,
        u0=y,
        alpha=0.0,
        s=separation,
        q=mass_ratio,
        rho=radius,
    )
    return evaluate, {
        "value": float(diagnostics.magnifications[0]),
        "method": diagnostics.finite_source_method_names[0],
        "converged": bool(diagnostics.finite_source_converged[0]),
        "error_estimate": float(diagnostics.finite_source_error_estimates[0]),
        "caustic_distance": float(diagnostics.caustic_distances[0]),
    }


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
    # The stress runner invokes this path in a dedicated process.  Keep the
    # optional dependency local so its import cannot affect core-only work.
    from microlux.basic_function import to_lowmass
    from microlux.limb_darkening import LinearLimbDarkening
    from microlux.trajectory_model import extended_light_curve_from_trajectory_l

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
    """Return the first resolution whose complete tested tail passes.

    A single lucky grid-phase crossing does not establish convergence.  This
    matches the native calibration definition of the minimum reliable nbin.
    """

    for index, row in enumerate(rows):
        if all(candidate["passes"] for candidate in rows[index:]):
            return row
    return None


def report_header(args, parameters, reference_engine, reference, budget):
    """Fields shared by complete and engine-isolated benchmark reports."""

    return {
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
        "engine": args.engine,
        "case": args.case,
        "profile": args.profile,
        "linear_limb_c": LIMB_C if args.profile == "linear" else None,
        "parameters": list(map(float, parameters)),
        "reference": {"engine": reference_engine, "value": reference},
        "error_budget": budget,
        "selection_rule": (
            "first increasing-resolution/bin candidate whose complete tested "
            "higher-resolution tail satisfies the common absolute error budget"
        ),
    }


def write_report(report, output):
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


def isolated_microlux_report(args, case, parameters, reference_engine, reference, budget):
    """Run only microLUX, for a timeout boundary owned by the stress runner."""

    import microlux

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

    report = report_header(args, parameters, reference_engine, reference, budget)
    report["system"].update(
        {
            "microlux_path": str(Path(microlux.__file__).resolve()),
            "microlux_commit": checkout_commit(microlux),
        }
    )
    report["microlux"] = {
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
        "directional_jvp": timed_jax(micro_jvp, (parameters, DIRECTION), args.repeat),
        "value_and_grad": timed_jax(micro_gradient, (parameters,), args.repeat),
    }
    return report


def isolated_core_report(
    args,
    parameters,
    reference_engine,
    reference,
    budget,
    jax_calibration,
    selected_jax,
    native_calibration,
    selected_native,
    inverse_forward,
    inverse_jvp,
    inverse_gradient,
    hybrid_result,
    hybrid_error,
    hybrid_forward,
    hybrid_jvp,
    hybrid_gradient,
    calibrated_report,
):
    """Render JAX, native lcbinint, and VBM without importing microLUX."""

    vbm = vbm_function(parameters, args.profile, budget)
    vbm_value = vbm()
    vbm_error = abs(vbm_value - reference)
    native_report = {
        "calibration": native_calibration,
        "selected_source_bins": None,
        "forward": None,
    }
    if selected_native is not None:
        native = native_function(parameters, args.profile, selected_native["source_bins"])
        native_report["selected_source_bins"] = selected_native["source_bins"]
        native_report["forward"] = timed_python(native, args.repeat, args.inner)
    native_auto, native_auto_report = native_auto_function(
        parameters, args.profile, args.atol, args.rtol
    )
    native_auto_report["absolute_error"] = abs(
        native_auto_report["value"] - reference
    )
    native_auto_report["passes"] = (
        native_auto_report["absolute_error"] <= budget
    )
    native_auto_report["forward"] = timed_python(
        native_auto, args.repeat, args.inner
    )
    native_report["automatic"] = native_auto_report

    report = report_header(args, parameters, reference_engine, reference, budget)
    report.update(
        {
            "jax_inverse_ray": {
                "calibration": jax_calibration,
                "selected_resolution": selected_jax["resolution"],
                "selected_tile_capacity": selected_jax["tile_capacity"],
                "forward": timed_jax(inverse_forward, (parameters,), args.repeat),
                "directional_jvp": timed_jax(
                    inverse_jvp, (parameters, DIRECTION), args.repeat
                ),
                "value_and_grad": timed_jax(
                    inverse_gradient, (parameters,), args.repeat
                ),
            },
            "jax_hybrid": {
                "value": float(hybrid_result.magnification),
                "absolute_error": hybrid_error,
                "passes": hybrid_error <= budget,
                "method": int(hybrid_result.method),
                "method_names": {
                    "0": "hexadecapole",
                    "1": "cartesian",
                    "2": "polar",
                },
                "estimated_error": float(hybrid_result.estimated_error),
                "forward": timed_jax(hybrid_forward, (parameters,), args.repeat),
                "directional_jvp": timed_jax(
                    hybrid_jvp, (parameters, DIRECTION), args.repeat
                ),
                "value_and_grad": timed_jax(
                    hybrid_gradient, (parameters,), args.repeat
                ),
            },
            "jax_calibrated": calibrated_report,
            "native_lcbinint": native_report,
            "vbmicrolensing": {
                "requested_accuracy": budget,
                "value": vbm_value,
                "absolute_error": vbm_error,
                "passes": vbm_error <= budget,
                "forward": timed_python(vbm, args.repeat, args.inner),
            },
        }
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), default="regular")
    parser.add_argument("--profile", choices=("uniform", "linear"), default="uniform")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument(
        "--calibrated-max-source-bins",
        type=int,
        choices=(16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400),
        default=400,
    )
    parser.add_argument(
        "--engine",
        choices=("all", "core", "microlux"),
        default="all",
        help=(
            "all entries (default), or isolate core JAX/native/VBM work from "
            "microLUX"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    case = CASES[args.case]
    parameters = jnp.asarray(case["parameters"])
    reference_engine, reference = reference_value(case, args.profile)
    budget = args.atol + args.rtol * max(abs(reference), 1.0)
    if args.engine == "microlux":
        write_report(
            isolated_microlux_report(
                args, case, parameters, reference_engine, reference, budget
            ),
            args.output,
        )
        return

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

    def hybrid(active):
        return hybrid_value(active, args.profile, resolution, capacity).magnification

    hybrid_forward = jax.jit(hybrid)
    hybrid_jvp = jax.jit(
        lambda active, direction: jax.jvp(hybrid, (active,), (direction,))[1]
    )
    hybrid_gradient = jax.jit(jax.value_and_grad(hybrid))
    hybrid_result = hybrid_value(parameters, args.profile, resolution, capacity)
    jax.block_until_ready(hybrid_result)
    hybrid_error = abs(float(hybrid_result.magnification) - reference)

    def calibrated(active):
        return binary_magnification_calibrated(
            *active,
            limb_c=LIMB_C if args.profile == "linear" else 0.0,
            absolute_tolerance=args.atol,
            relative_tolerance=args.rtol,
            maximum_source_bins=args.calibrated_max_source_bins,
            moment_mode=args.profile,
        ).magnification

    calibrated_forward = jax.jit(calibrated)
    calibrated_jvp = jax.jit(
        lambda active, direction: jax.jvp(
            calibrated, (active,), (direction,)
        )[1]
    )
    calibrated_gradient = jax.jit(jax.value_and_grad(calibrated))
    calibrated_result = binary_magnification_calibrated(
        *parameters,
        limb_c=LIMB_C if args.profile == "linear" else 0.0,
        absolute_tolerance=args.atol,
        relative_tolerance=args.rtol,
        maximum_source_bins=args.calibrated_max_source_bins,
        moment_mode=args.profile,
    )
    jax.block_until_ready(calibrated_result)
    calibrated_error = abs(float(calibrated_result.magnification) - reference)
    calibrated_report = {
        "value": float(calibrated_result.magnification),
        "absolute_error": calibrated_error,
        "passes": calibrated_error <= budget,
        "method": int(calibrated_result.method),
        "support_valid": bool(calibrated_result.support_valid),
        "selected_source_bins": int(calibrated_result.selected_source_bins),
        "comparison_resolution": int(calibrated_result.comparison_resolution),
        "executed_resolution": int(calibrated_result.executed_resolution),
        "tile_capacity": int(calibrated_result.tile_capacity),
        "caustic_distance": float(calibrated_result.caustic_distance),
        "prefer_polar": bool(calibrated_result.prefer_polar),
        "point_safe": bool(calibrated_result.point_safe),
        "chord_band": bool(calibrated_result.chord_band),
        "tangent_band": bool(calibrated_result.tangent_band),
        "grazing_ring_band": bool(calibrated_result.grazing_ring_band),
        "value_error": float(calibrated_result.value_error),
        "value_budget": float(calibrated_result.value_budget),
        "value_converged": bool(calibrated_result.value_converged),
        "forward": timed_jax(calibrated_forward, (parameters,), args.repeat),
        "directional_jvp": timed_jax(
            calibrated_jvp, (parameters, DIRECTION), args.repeat
        ),
        "value_and_grad": timed_jax(
            calibrated_gradient, (parameters,), args.repeat
        ),
    }
    if args.engine == "core":
        write_report(
            isolated_core_report(
                args,
                parameters,
                reference_engine,
                reference,
                budget,
                jax_calibration,
                selected_jax,
                native_calibration,
                selected_native,
                inverse_forward,
                inverse_jvp,
                inverse_gradient,
                hybrid_result,
                hybrid_error,
                hybrid_forward,
                hybrid_jvp,
                hybrid_gradient,
                calibrated_report,
            ),
            args.output,
        )
        return

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
    native_auto, native_auto_report = native_auto_function(
        parameters, args.profile, args.atol, args.rtol
    )
    native_auto_report["absolute_error"] = abs(
        native_auto_report["value"] - reference
    )
    native_auto_report["passes"] = (
        native_auto_report["absolute_error"] <= budget
    )
    native_auto_report["forward"] = timed_python(
        native_auto, args.repeat, args.inner
    )
    native_report["automatic"] = native_auto_report

    import microlux

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
        "engine": args.engine,
        "case": args.case,
        "profile": args.profile,
        "linear_limb_c": LIMB_C if args.profile == "linear" else None,
        "parameters": list(map(float, parameters)),
        "reference": {"engine": reference_engine, "value": reference},
        "error_budget": budget,
        "selection_rule": (
            "first increasing-resolution/bin candidate whose complete tested "
            "higher-resolution tail satisfies the common absolute error budget"
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
        "jax_hybrid": {
            "value": float(hybrid_result.magnification),
            "absolute_error": hybrid_error,
            "passes": hybrid_error <= budget,
            "method": int(hybrid_result.method),
            "method_names": {
                "0": "hexadecapole",
                "1": "cartesian",
                "2": "polar",
            },
            "estimated_error": float(hybrid_result.estimated_error),
            "forward": timed_jax(hybrid_forward, (parameters,), args.repeat),
            "directional_jvp": timed_jax(
                hybrid_jvp, (parameters, DIRECTION), args.repeat
            ),
            "value_and_grad": timed_jax(hybrid_gradient, (parameters,), args.repeat),
        },
        "jax_calibrated": calibrated_report,
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

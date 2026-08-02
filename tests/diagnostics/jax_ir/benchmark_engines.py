#!/usr/bin/env python3
"""Compare JAX, native lcbinint, and VBMicrolensing at matched accuracy."""

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

import VBMicrolensing  # noqa: E402
import lcbinint  # noqa: E402
from lcbinint_jax import (  # noqa: E402
    binary_inverse_ray,
    binary_inverse_ray_fixed_support,
    discover_binary_macro_tiles,
)

CASES = (
    {
        "name": "regular_uniform",
        "parameters": (0.2, 0.1, 1.2, 0.1, 0.2, 0.0, 0.0),
        "jax_capacities": {16: 128, 32: 256, 64: 512, 128: 2048},
    },
    {
        "name": "regular_square_root_limb",
        "parameters": (0.2, 0.1, 1.2, 0.1, 0.2, 0.4, 0.1),
        "jax_capacities": {16: 128, 32: 256, 64: 512, 128: 2048},
    },
    {
        "name": "resonant_cusp_uniform",
        "parameters": (0.653, 0.0, 1.2, 0.1, 0.02, 0.0, 0.0),
        "jax_capacities": {16: 256, 32: 512, 64: 1024, 128: 4096},
    },
)
NATIVE_BINS = (16, 32, 48, 64, 96, 128, 192, 256)
JAX_RESOLUTIONS = (16, 32, 64, 128)
DIRECTION = np.asarray((0.2, -0.1, 0.05, 0.02, 0.0, 0.1, -0.05))


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


def timed_jax(function, arguments, repeat):
    compile_start = time.perf_counter()
    first = function(*arguments)
    jax.block_until_ready(first)
    compile_and_first = time.perf_counter() - compile_start
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


def reference_value(parameters):
    x, y, separation, mass_ratio, radius, limb_c, limb_d = parameters
    if limb_c != 0.0 or limb_d != 0.0:
        value = native_function(parameters, 768)()
        return "native lcbinint, 768 source bins", value
    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = 1.0e-8
    return (
        "VBMicrolensing, requested accuracy 1e-8",
        engine.BinaryMag2(separation, mass_ratio, x, y, radius),
    )


def native_function(parameters, source_bins):
    x, y, separation, mass_ratio, radius, limb_c, limb_d = parameters
    options = lcbinint.Options(
        nbin=source_bins,
        inverse_ray_grid="cartesian",
        coordinates="center_of_mass",
    )
    limb_darkening = lcbinint.LimbDarkening(c=limb_c, d=limb_d)
    return lambda: lcbinint.binary_ray_shooting(
        x,
        y,
        s=separation,
        q=mass_ratio,
        rho=radius,
        limb_darkening=limb_darkening,
        options=options,
    )


def native_parameter_function(source_bins):
    options = lcbinint.Options(
        nbin=source_bins,
        inverse_ray_grid="cartesian",
        coordinates="center_of_mass",
    )

    def evaluate(parameters):
        x, y, separation, mass_ratio, radius, limb_c, limb_d = map(float, parameters)
        return lcbinint.binary_ray_shooting(
            x,
            y,
            s=separation,
            q=mass_ratio,
            rho=radius,
            limb_darkening=lcbinint.LimbDarkening(c=limb_c, d=limb_d),
            options=options,
        )

    return evaluate


def vbm_function(parameters, accuracy):
    x, y, separation, mass_ratio, radius, limb_c, limb_d = parameters
    if limb_c != 0.0 or limb_d != 0.0:
        raise ValueError("VBMicrolensing timing is limited to uniform sources")
    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = accuracy
    return lambda: engine.BinaryMag2(separation, mass_ratio, x, y, radius)


def select_first_passing(rows):
    for row in rows:
        if row["passes"]:
            return row
    raise RuntimeError("no candidate met the requested error budget")


def benchmark_case(case, repeat, inner, atol, rtol):
    parameters = np.asarray(case["parameters"], dtype=np.float64)
    reference_engine, reference = reference_value(parameters)
    budget = atol + rtol * max(abs(reference), 1.0)

    native_calibration = []
    for source_bins in NATIVE_BINS:
        value = native_function(parameters, source_bins)()
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
    native_evaluate = native_parameter_function(selected_native["source_bins"])
    difference_step = 3.0e-3

    def native_forward():
        return native_evaluate(parameters)

    def native_jvp():
        return (
            native_evaluate(parameters + difference_step * DIRECTION)
            - native_evaluate(parameters - difference_step * DIRECTION)
        ) / (2.0 * difference_step)

    def native_gradient():
        gradient = np.empty(7)
        for index in range(7):
            direction = np.zeros(7)
            direction[index] = 1.0
            gradient[index] = (
                native_evaluate(parameters + difference_step * direction)
                - native_evaluate(parameters - difference_step * direction)
            ) / (2.0 * difference_step)
        return gradient

    jax_parameters = jnp.asarray(parameters)
    jax_direction = jnp.asarray(DIRECTION)
    jax_calibration = []
    jax_functions = {}
    for resolution in JAX_RESOLUTIONS:
        capacity = case["jax_capacities"][resolution]

        def value(active_parameters, resolution=resolution, capacity=capacity):
            return binary_inverse_ray(
                *active_parameters,
                resolution=resolution,
                tile_size=16,
                tile_capacity=capacity,
                limb_samples=32,
            ).magnification

        forward = jax.jit(value)
        start = time.perf_counter()
        result = forward(jax_parameters)
        jax.block_until_ready(result)
        compile_and_first = time.perf_counter() - start
        numeric_value = float(result)
        error = abs(numeric_value - reference)
        jax_calibration.append(
            {
                "resolution": resolution,
                "tile_capacity": capacity,
                "value": numeric_value,
                "absolute_error": error,
                "passes": error <= budget,
                "compile_and_first_seconds": compile_and_first,
            }
        )
        jax_functions[resolution] = value

    selected_jax = select_first_passing(jax_calibration)
    selected_resolution = selected_jax["resolution"]
    selected_capacity = selected_jax["tile_capacity"]
    selected_value = jax_functions[selected_jax["resolution"]]
    jax_forward = jax.jit(selected_value)
    jax_jvp = jax.jit(
        lambda active_parameters, active_direction: jax.jvp(
            selected_value,
            (active_parameters,),
            (active_direction,),
        )[1]
    )
    jax_value_and_grad = jax.jit(jax.value_and_grad(selected_value))

    def discover(active_parameters):
        cell_size = jax.lax.stop_gradient(active_parameters[4] / selected_resolution)
        return discover_binary_macro_tiles(
            *active_parameters[:5],
            cell_size,
            tile_size=16,
            tile_capacity=selected_capacity,
            limb_samples=32,
        )

    jax_discover = jax.jit(discover)
    selected_support = jax_discover(jax_parameters)
    jax.block_until_ready(selected_support)

    def integrate_fixed_support(active_parameters):
        cell_size = jax.lax.stop_gradient(active_parameters[4] / selected_resolution)
        return binary_inverse_ray_fixed_support(
            selected_support.tile_origins,
            selected_support.tile_mask,
            cell_size,
            *active_parameters,
            tile_size=16,
        ).magnification

    jax_fixed_support = jax.jit(integrate_fixed_support)

    limb_darkened = bool(parameters[5] != 0.0 or parameters[6] != 0.0)
    if limb_darkened:
        vbm_report = {
            "comparable": False,
            "reason": (
                "the installed Python API did not reproduce lcbinint's "
                "two-coefficient square-root-law convention"
            ),
        }
    else:
        vbm = vbm_function(parameters, budget)
        vbm_value = vbm()
        vbm_error = abs(vbm_value - reference)
        vbm_report = {
            "comparable": True,
            "requested_accuracy": budget,
            "value": vbm_value,
            "absolute_error": vbm_error,
            "passes": vbm_error <= budget,
            "forward": timed_python(vbm, repeat, inner),
        }

    return {
        "name": case["name"],
        "parameters": parameters.tolist(),
        "reference": {
            "engine": reference_engine,
            "value": reference,
        },
        "error_budget": budget,
        "native_lcbinint": {
            "calibration": native_calibration,
            "selected_source_bins": selected_native["source_bins"],
            "forward": timed_python(native_forward, repeat, inner),
            "directional_central_difference": timed_python(
                native_jvp, repeat, max(1, inner // 2)
            ),
            "seven_parameter_central_difference": timed_python(
                native_gradient, repeat, 1
            ),
        },
        "jax": {
            "calibration": jax_calibration,
            "selected_resolution": selected_resolution,
            "selected_tile_capacity": selected_capacity,
            "selected_tile_count": int(selected_support.visited_count),
            "forward": timed_jax(jax_forward, (jax_parameters,), repeat),
            "forward_breakdown": {
                "discovery": timed_jax(
                    jax_discover,
                    (jax_parameters,),
                    repeat,
                ),
                "fixed_support_integration": timed_jax(
                    jax_fixed_support,
                    (jax_parameters,),
                    repeat,
                ),
            },
            "directional_jvp": timed_jax(
                jax_jvp,
                (jax_parameters, jax_direction),
                repeat,
            ),
            "value_and_grad": timed_jax(
                jax_value_and_grad,
                (jax_parameters,),
                repeat,
            ),
        },
        "vbmicrolensing": vbm_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
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
        "selection_rule": (
            "lowest tested setting whose absolute error is no larger than "
            "atol + rtol * max(abs(reference), 1)"
        ),
        "atol": args.atol,
        "rtol": args.rtol,
        "cases": [
            benchmark_case(
                case,
                args.repeat,
                args.inner,
                args.atol,
                args.rtol,
            )
            for case in CASES
        ],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

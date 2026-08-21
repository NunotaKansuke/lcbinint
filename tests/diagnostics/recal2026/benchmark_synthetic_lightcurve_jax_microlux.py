#!/usr/bin/env python3
"""Compare the six paper-facing synthetic binary light curves.

The six binary cases and time windows are copied from the final
``synthetic_lightcurve_benchmark_narrow_windows_20260816`` run.  This driver
keeps that physical time grid and compares native lcbinint, the public JAX
LightCurve path, microLUX, and VBMicrolensing at the light-curve level.

The measured lanes run in isolated child processes.  JAX has two lanes: both
compile the callable, while only the warm lane performs the native-style
route/nbin calibration and seed/support-cache priming.  The differentiable
metric is the full parameter-gradient light curve, not a time derivative:
``G[i, k] = dA(t_i) / dtheta[k]``.  microLUX uses the VBMicrolensing-selected
nannuli for each epoch, grouped by equal counts in one compiled callable per
equal-count group in an isolated worker.
The benchmark also measures a data-free reverse-mode model pullback
``J.T @ w`` in sampler coordinates using a deterministic dense cotangent
probe; this is the HMC-like model-gradient metric.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
DEFAULT_OUTPUT = (
    ROOT
    / "tests/diagnostics/results/recal2026/"
    / "synthetic_lightcurve_jax_vjp_20260820"
)
REFERENCE_RUN = (
    ROOT
    / "tests/diagnostics/results/recal2026/"
    / "synthetic_lightcurve_benchmark_narrow_windows_20260816/benchmark.json"
)
TOLERANCE = 1.0e-3
LIMB_DARKENING_C = 0.5
REPEATS = 5
MICROLUX_STRATEGY = (30, 30, 60, 120, 240)
WORKER_TIMEOUT_SECONDS = 900.0
MICROLUX_GROUP_PARALLELISM = 4
GRADIENT_PARAMETER_KEYS = ("s", "q", "rho", "u0", "alpha", "t0", "tE")
# Sampler coordinates used by the VJP probe.  This is the same positive-scale
# log parameterization used when passing the cases to VBMicrolensing, but it
# is still entirely model-side: no observations, uncertainties, or residuals
# enter the probe.
SAMPLER_PARAMETER_KEYS = (
    "log_s", "log_q", "u0", "alpha", "log_rho", "log_tE", "t0"
)

# Fixed paper-facing order from HANDOFF_synthetic_lightcurve_benchmark_paper.
CASES = (
    {
        "name": "resonant_high_mag",
        "params": {
            "s": 0.95, "q": 1.0e-2, "rho": 5.0e-3,
            "u0": -1.0e-3, "alpha": 0.5, "t0": 0.0, "tE": 1.0,
        },
        "t_min": -0.4, "t_max": 0.4, "n_times": 240,
    },
    {
        "name": "resonant_large_source",
        "params": {
            "s": 0.95, "q": 1.0e-2, "rho": 2.0e-2,
            "u0": -1.0e-2, "alpha": 0.5, "t0": 0.0, "tE": 1.0,
        },
        "t_min": -0.4, "t_max": 0.4, "n_times": 240,
    },
    {
        "name": "close_binary",
        "params": {
            "s": 0.65, "q": 5.0e-3, "rho": 3.0e-3,
            "u0": 3.0e-2, "alpha": 1.1, "t0": 0.0, "tE": 1.0,
        },
        "t_min": -0.5, "t_max": 0.5, "n_times": 240,
    },
    {
        "name": "high_q",
        "params": {
            "s": 1.0, "q": 1.0e-1, "rho": 1.0e-2,
            "u0": 5.0e-2, "alpha": 1.3, "t0": 0.0, "tE": 1.0,
        },
        "t_min": -1.0, "t_max": 1.0, "n_times": 240,
    },
    {
        "name": "wide_planet",
        "params": {
            "s": 2.5, "q": 1.0e-2, "rho": 2.0e-3,
            "u0": 0.294, "alpha": 3.0, "t0": -2.07, "tE": 1.0,
        },
        "t_min": -2.8, "t_max": 0.8, "n_times": 600,
    },
    {
        "name": "close_secondary_caustics",
        "params": {
            "s": 0.65, "q": 2.0e-2, "rho": 4.0e-3,
            "u0": 0.214, "alpha": 3.75, "t0": 0.0, "tE": 1.0,
        },
        "t_min": -1.2, "t_max": 1.2, "n_times": 400,
    },
)


def _load_benchmark_backend():
    """Load the in-tree build before importing the public Python wrapper."""

    path = SCRIPT.parent / "bench_jax_microlux_12800.py"
    spec = importlib.util.spec_from_file_location(
        "bench_jax_microlux_12800", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_nannuli_helper():
    path = SCRIPT.parent / "rerun_microlux_vbm_event_annuli.py"
    spec = importlib.util.spec_from_file_location(
        "rerun_microlux_vbm_event_annuli", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_times(case):
    return np.linspace(
        float(case["t_min"]), float(case["t_max"]), int(case["n_times"])
    )


def _vbm_parameters(params):
    return [
        math.log(params["s"]),
        math.log(params["q"]),
        params["u0"],
        params["alpha"],
        math.log(params["rho"]),
        math.log(params["tE"]),
        params["t0"],
    ]


def _source_xy(times, params):
    tau = (np.asarray(times, dtype=float) - params["t0"]) / params["tE"]
    alpha = params["alpha"]
    return (
        tau * math.cos(alpha) - params["u0"] * math.sin(alpha),
        tau * math.sin(alpha) + params["u0"] * math.cos(alpha),
    )


def _gradient_parameter_vector(params):
    """Return the fixed physical-parameter order used by the gradient curve."""

    return np.asarray(
        [float(params[key]) for key in GRADIENT_PARAMETER_KEYS],
        dtype=float,
    )


def _parameters_from_gradient_vector(base_params, active_parameters):
    return {
        **base_params,
        **{
            key: active_parameters[index]
            for index, key in enumerate(GRADIENT_PARAMETER_KEYS)
        },
    }


def _sampler_parameter_vector(params):
    """Return the unconstrained coordinates used for the model VJP probe."""

    return np.asarray(
        [
            math.log(float(params["s"])),
            math.log(float(params["q"])),
            float(params["u0"]),
            float(params["alpha"]),
            math.log(float(params["rho"])),
            math.log(float(params["tE"])),
            float(params["t0"]),
        ],
        dtype=float,
    )


def _cotangent_probe(times):
    """Build a deterministic dense model-space cotangent, independent of data."""

    count = int(np.asarray(times).size)
    phase = 2.0 * np.pi * (np.arange(count, dtype=float) + 0.5) / count
    probe = (
        np.sin(phase)
        + 0.37 * np.cos(3.0 * phase)
        + 0.19 * np.sin(7.0 * phase)
    )
    return probe / np.linalg.norm(probe)


def _accuracy(reference, values):
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(reference) & np.isfinite(values)
    invalid_epochs = int(reference.size - np.count_nonzero(finite))
    if not np.any(finite):
        return {
            "finite_epochs": 0, "invalid_epochs": invalid_epochs,
            "max_relative": None,
            "p99_relative": None, "median_relative": None,
            "rms_relative": None, "within_tolerance": False,
        }
    relative = np.abs(values[finite] - reference[finite]) / np.maximum(
        np.abs(reference[finite]), 1.0e-12
    )
    return {
        "finite_epochs": int(np.count_nonzero(finite)),
        "invalid_epochs": invalid_epochs,
        "max_relative": float(np.max(relative)),
        "p99_relative": float(np.percentile(relative, 99.0)),
        "median_relative": float(np.median(relative)),
        "rms_relative": float(np.sqrt(np.mean(relative * relative))),
        "within_tolerance": bool(
            invalid_epochs == 0 and np.max(relative) <= TOLERANCE
        ),
    }


def _sync(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return np.asarray(value, dtype=float)


def _timed(evaluate, repeats=REPEATS):
    started = time.perf_counter()
    first = _sync(evaluate())
    cold = time.perf_counter() - started
    samples = []
    value = first
    for _ in range(repeats):
        started = time.perf_counter()
        value = _sync(evaluate())
        samples.append(time.perf_counter() - started)
    return {
        "cold_seconds": float(cold),
        "steady_samples_seconds": [float(item) for item in samples],
        "steady_seconds": float(statistics.median(samples)),
        "value": value,
    }


def _sync_value_and_grad(result):
    """Synchronize and convert a scalar value plus its parameter gradient."""

    value, gradient = result
    return float(_sync(value)), _sync(gradient)


def _timed_value_and_grad(evaluate, repeats=REPEATS):
    """Time one scalar ``value_and_grad`` call and return both outputs."""

    started = time.perf_counter()
    first_value, first_gradient = _sync_value_and_grad(evaluate())
    cold = time.perf_counter() - started
    samples = []
    value = first_value
    gradient = first_gradient
    for _ in range(repeats):
        started = time.perf_counter()
        value, gradient = _sync_value_and_grad(evaluate())
        samples.append(time.perf_counter() - started)
    return {
        "cold_seconds": float(cold),
        "steady_samples_seconds": [float(item) for item in samples],
        "steady_seconds": float(statistics.median(samples)),
        "value": float(value),
        "gradient": gradient.tolist(),
    }


def _linear_nannuli_shim():
    helper = _load_nannuli_helper()
    path = helper._build_vbm_shim(helper.DEFAULT_SHIM)
    library = ctypes.CDLL(str(path))
    library.vbm_nannuli_create.argtypes = [ctypes.c_double, ctypes.c_double]
    library.vbm_nannuli_create.restype = ctypes.c_void_p
    library.vbm_nannuli_destroy.argtypes = [ctypes.c_void_p]
    library.vbm_nannuli_destroy.restype = None
    library.vbm_nannuli_binary_mag_dark.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.vbm_nannuli_binary_mag_dark.restype = ctypes.c_double
    return library


def _vbm_nannuli(times, params, profile_c):
    if profile_c == 0.0:
        return None, None
    library = _linear_nannuli_shim()
    handle = library.vbm_nannuli_create(TOLERANCE, profile_c)
    xs, ys = _source_xy(times, params)
    counts = []
    try:
        for x, y in zip(xs, ys):
            count = ctypes.c_int()
            library.vbm_nannuli_binary_mag_dark(
                handle,
                params["s"], params["q"], float(x), float(y),
                params["rho"], 1.0e-12, ctypes.byref(count),
            )
            counts.append(int(count.value))
    finally:
        library.vbm_nannuli_destroy(handle)
    return counts, max(counts)


def _measure_vbm(times, params, profile_c):
    import VBMicrolensing

    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = TOLERANCE
    engine.RelTol = TOLERANCE
    engine.a1 = profile_c
    engine.a2 = 0.0
    arguments = _vbm_parameters(params)

    def evaluate():
        return np.asarray(
            engine.BinaryLightCurve(arguments, times.tolist())[0],
            dtype=float,
        )

    return _timed(evaluate)


def _make_native_curve(lcbinint, profile_c):
    return lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="vbm", nbin="auto", tol=TOLERANCE, reltol=TOLERANCE
        ),
        limb_darkening=lcbinint.LimbDarkening.linear(profile_c)
        if profile_c
        else lcbinint.LimbDarkening.none(),
    )


def _measure_native(lcbinint, times, params, profile_c):
    native_params = dict(params)
    no_curve = _make_native_curve(lcbinint, profile_c)
    no_timing = _timed(lambda: no_curve(times, native_params))

    warm_curve = _make_native_curve(lcbinint, profile_c)
    warm_started = time.perf_counter()
    warm_error = None
    warm_report = None
    try:
        warm_report = warm_curve.warmup(
            times, native_params, grid_timing_repeats=1
        )
    except Exception as error:  # retain the failure in the report
        warm_error = f"{type(error).__name__}: {error}"
    warm_seconds = time.perf_counter() - warm_started
    warm_timing = None
    if warm_report is not None:
        warm_timing = _timed(lambda: warm_curve(times, native_params))
        warm_timing["warmup_seconds"] = float(warm_seconds)

    return {
        "no_warmup": {
            key: value for key, value in no_timing.items() if key != "value"
        },
        "warmup": (
            None
            if warm_timing is None
            else {
                key: value
                for key, value in warm_timing.items()
                if key != "value"
            }
        ),
        "no_values": no_timing["value"].tolist(),
        "warm_values": (
            None if warm_timing is None else warm_timing["value"].tolist()
        ),
        "warmup_error": warm_error,
        "warmup_all_calibrated": (
            None if warm_report is None else bool(warm_report.all_calibrated)
        ),
    }


def _worker_micro_function(times, params, profile_c, n_annuli_per_epoch):
    import jax
    import jax.numpy as jnp
    from microlux.basic_function import to_lowmass
    from microlux.limb_darkening import LinearLimbDarkening
    from microlux.trajectory import LinearTrajectory
    from microlux.trajectory_model import extended_light_curve_from_trajectory_l

    limb_darkening = (
        LinearLimbDarkening(profile_c) if profile_c else None
    )
    times = jnp.asarray(times, dtype=jnp.float64)
    parameter_vector = jnp.asarray(
        _gradient_parameter_vector(params), dtype=jnp.float64
    )

    if profile_c == 0.0 or n_annuli_per_epoch is None:
        n_annuli_per_epoch = [10] * int(times.size)
    n_annuli_per_epoch = tuple(int(value or 10) for value in n_annuli_per_epoch)
    if len(n_annuli_per_epoch) != int(times.size):
        raise ValueError("microLUX n_annuli plan must match the time grid")
    groups = tuple(
        (
            int(n_annuli),
            tuple(
                index
                for index, value in enumerate(n_annuli_per_epoch)
                if value == n_annuli
            ),
        )
        for n_annuli in sorted(set(n_annuli_per_epoch))
    )

    def evaluate(active_times, active_parameters):
        separation, mass_ratio, rho, u0, alpha, t0, tE = active_parameters
        alpha_deg = -alpha * (180.0 / jnp.pi)
        tau = -((active_times - t0) / tE)[::-1]
        trajectory = LinearTrajectory(0.0, u0, 1.0, alpha_deg)(tau)
        trajectory_l = to_lowmass(separation, mass_ratio, trajectory)
        values = jnp.zeros(active_times.shape, dtype=active_times.dtype)
        reverse_indices = jnp.arange(
            active_times.size - 1, -1, -1, dtype=jnp.int32
        )
        for n_annuli, indices in groups:
            original_indices = jnp.asarray(indices, dtype=jnp.int32)
            reversed_indices = reverse_indices[original_indices]
            group_values = extended_light_curve_from_trajectory_l(
                trajectory_l[reversed_indices],
                separation,
                mass_ratio,
                rho,
                tol=TOLERANCE,
                retol=TOLERANCE,
                default_strategy=MICROLUX_STRATEGY,
                limb_darkening=limb_darkening,
                n_annuli=n_annuli,
            )
            values = values.at[original_indices].set(group_values)
        return values

    # Keep one static executable per annulus group.  Besides making the
    # measured callable explicit, the outer jit gives the persistent XLA
    # cache a stable entry that can be loaded by the timing worker.
    return jax.jit(evaluate), times, parameter_vector


def _worker_jax_functions(times, params, profile_c):
    import jax
    import jax.numpy as jnp
    import lcbinint

    curve = lcbinint.LightCurve(
        lens="binary",
        options=lcbinint.Options(
            coordinates="vbm", nbin="auto", caustic_bins=1400,
            max_source_bins=400, reltol=TOLERANCE, jax=True,
        ),
    )
    jax_params = dict(params, limb_darkening_c=float(profile_c))
    time_array = jnp.asarray(times, dtype=jnp.float64)
    parameter_vector = jnp.asarray(
        _gradient_parameter_vector(params), dtype=jnp.float64
    )
    sampler_vector = jnp.asarray(
        _sampler_parameter_vector(params), dtype=jnp.float64
    )
    cotangent = jnp.asarray(_cotangent_probe(times), dtype=jnp.float64)

    def model(active_parameters):
        return curve(
            time_array,
            _parameters_from_gradient_vector(jax_params, active_parameters),
        )

    gradient_function = jax.jit(jax.jacfwd(model))

    def values():
        return model(parameter_vector)

    def gradient():
        return gradient_function(parameter_vector)

    def sampler_model(active_parameters):
        return curve(
            time_array,
            {
                **jax_params,
                "s": jnp.exp(active_parameters[0]),
                "q": jnp.exp(active_parameters[1]),
                "u0": active_parameters[2],
                "alpha": active_parameters[3],
                "rho": jnp.exp(active_parameters[4]),
                "tE": jnp.exp(active_parameters[5]),
                "t0": active_parameters[6],
            },
        )

    vjp_function = jax.jit(
        jax.grad(
            lambda active_parameters: jnp.vdot(
                cotangent, sampler_model(active_parameters)
            )
        )
    )

    value_and_grad_function = jax.jit(
        jax.value_and_grad(
            lambda active_parameters: jnp.vdot(
                cotangent, sampler_model(active_parameters)
            )
        )
    )

    def vjp():
        return vjp_function(sampler_vector)

    def value_and_grad():
        return value_and_grad_function(sampler_vector)

    return curve, values, gradient, vjp, value_and_grad


def _run_worker(payload):
    # This import also validates that the current build exports every FFI
    # required by the public trajectory path.
    _load_benchmark_backend()
    import jax
    import jax.numpy as jnp

    cache = Path(payload["jax_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    case = payload["case"]
    params = case["params"]
    times = np.asarray(payload["times"], dtype=float)
    profile_c = float(payload["profile_c"])
    n_annuli_per_epoch = payload.get("n_annuli_per_epoch")
    jax_mode = str(payload.get("jax_mode", "warmup"))
    include_micro = bool(payload.get("include_microlux", True))
    measure_micro = bool(payload.get("measure_microlux", True))

    jax_warmup = None
    jax_value_timing = jax_gradient_timing = None
    jax_gradient_compile_seconds = None
    jax_vjp_timing = None
    jax_vjp_compile_seconds = None
    jax_value_and_grad_timing = None
    jax_value_and_grad_compile_seconds = None
    if jax_mode != "micro_only":
        (
            jax_curve,
            jax_values,
            jax_gradient,
            jax_vjp,
            jax_value_and_grad,
        ) = _worker_jax_functions(
            times, params, profile_c
        )
        if jax_mode == "warmup":
            warm_started = time.perf_counter()
            jax_warmup = jax_curve.warmup(
                times, dict(params, limb_darkening_c=profile_c)
            )
            gradient_started = time.perf_counter()
            _sync(jax_gradient())
            jax_gradient_compile_seconds = time.perf_counter() - gradient_started
            vjp_started = time.perf_counter()
            _sync(jax_vjp())
            jax_vjp_compile_seconds = time.perf_counter() - vjp_started
            value_and_grad_started = time.perf_counter()
            _sync_value_and_grad(jax_value_and_grad())
            jax_value_and_grad_compile_seconds = (
                time.perf_counter() - value_and_grad_started
            )
            jax_warmup_seconds = time.perf_counter() - warm_started
        elif jax_mode == "no_warmup":
            jax_warmup_seconds = 0.0
        else:
            raise ValueError(f"unknown JAX benchmark mode: {jax_mode!r}")
        jax_value_timing = _timed(jax_values)
        jax_gradient_timing = _timed(jax_gradient)
        jax_vjp_timing = _timed(jax_vjp)
        jax_value_and_grad_timing = _timed_value_and_grad(jax_value_and_grad)

    micro_values = micro_times = micro_parameters = None
    micro_warnings = []
    micro_value_timing = micro_gradient_timing = micro_vjp_timing = None
    micro_value_and_grad_timing = None
    micro_compile_seconds = None
    micro_server_ready = payload.get("micro_server_ready")
    micro_server_release = payload.get("micro_server_release")
    if include_micro:
        micro_values, micro_times, micro_parameters = _worker_micro_function(
            times, params, profile_c, n_annuli_per_epoch
        )
        micro_gradient_function = jax.jit(
            jax.jacfwd(
                lambda active_parameters: micro_values(
                    micro_times, active_parameters
                )
            )
        )
        micro_sampler_parameters = jnp.asarray(
            _sampler_parameter_vector(params), dtype=jnp.float64
        )
        cotangent_payload = payload.get("cotangent")
        if cotangent_payload is None:
            cotangent_payload = _cotangent_probe(
                np.asarray(micro_times, dtype=float)
            )
        micro_cotangent = jnp.asarray(cotangent_payload, dtype=jnp.float64)
        if int(micro_cotangent.size) != int(micro_times.size):
            raise ValueError("microLUX cotangent must match its time subset")

        def micro_sampler_model(active_parameters):
            physical_parameters = jnp.stack(
                (
                    jnp.exp(active_parameters[0]),
                    jnp.exp(active_parameters[1]),
                    jnp.exp(active_parameters[4]),
                    active_parameters[2],
                    active_parameters[3],
                    active_parameters[6],
                    jnp.exp(active_parameters[5]),
                )
            )
            return micro_values(micro_times, physical_parameters)

        micro_vjp_function = jax.jit(
            jax.grad(
                lambda active_parameters: jnp.vdot(
                    micro_cotangent,
                    micro_sampler_model(active_parameters),
                )
            )
        )
        micro_value_and_grad_function = jax.jit(
            jax.value_and_grad(
                lambda active_parameters: jnp.vdot(
                    micro_cotangent,
                    micro_sampler_model(active_parameters),
                )
            )
        )

        def measure_micro_outputs():
            nonlocal micro_value_timing, micro_gradient_timing
            nonlocal micro_vjp_timing, micro_value_and_grad_timing
            micro_value_timing = _timed(
                lambda: micro_values(micro_times, micro_parameters)
            )
            micro_gradient_timing = _timed(
                lambda: micro_gradient_function(micro_parameters)
            )
            micro_vjp_timing = _timed(
                lambda: micro_vjp_function(micro_sampler_parameters)
            )
            micro_value_and_grad_timing = _timed_value_and_grad(
                lambda: micro_value_and_grad_function(micro_sampler_parameters)
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if measure_micro:
                measure_micro_outputs()
            else:
                compile_started = time.perf_counter()
                _sync(micro_values(micro_times, micro_parameters))
                _sync(micro_gradient_function(micro_parameters))
                _sync(micro_vjp_function(micro_sampler_parameters))
                _sync_value_and_grad(
                    micro_value_and_grad_function(micro_sampler_parameters)
                )
                micro_compile_seconds = time.perf_counter() - compile_started
                if micro_server_ready is not None:
                    ready_path = Path(micro_server_ready)
                    ready_path.write_text(
                        json.dumps({
                            "status": "ready",
                            "compile_seconds": micro_compile_seconds,
                            "warnings": [str(item.message) for item in caught],
                        }) + "\n",
                        encoding="utf-8",
                    )
                    release_path = Path(micro_server_release)
                    wait_started = time.perf_counter()
                    while not release_path.exists():
                        if time.perf_counter() - wait_started > WORKER_TIMEOUT_SECONDS:
                            raise TimeoutError(
                                "microLUX server was not released by the parent"
                            )
                        time.sleep(0.05)
                    measure_micro = True
                    measure_micro_outputs()
            micro_warnings = [str(item.message) for item in caught]

    def pack(timing):
        return {
            key: value
            for key, value in timing.items()
            if key != "value"
        } | {"values": timing["value"].tolist()}

    jax_lane = None
    if jax_mode != "micro_only":
        jax_lane = {
            "mode": jax_mode,
            "warmup_seconds": (
                None if jax_mode == "no_warmup" else float(jax_warmup_seconds)
            ),
            "warmup_all_calibrated": (
                None if jax_warmup is None else bool(jax_warmup.all_calibrated)
            ),
            "jax_compile_seconds": (
                None
                if jax_warmup is None
                else float(jax_warmup.jax_compile_seconds)
            ),
            "seed_cache_primed": (
                None
                if jax_warmup is None
                else bool(jax_warmup.seed_cache_primed)
            ),
            "warmup_methods": (
                None if jax_warmup is None else list(jax_warmup.methods)
            ),
            "warmup_resolutions": (
                None
                if jax_warmup is None
                else np.asarray(jax_warmup.resolutions, dtype=int).tolist()
            ),
            "value": pack(jax_value_timing),
            "gradient_compile_seconds": jax_gradient_compile_seconds,
            "gradient": pack(jax_gradient_timing),
            "vjp_compile_seconds": jax_vjp_compile_seconds,
            "vjp": pack(jax_vjp_timing),
            "value_and_grad_compile_seconds": jax_value_and_grad_compile_seconds,
            "value_and_grad": jax_value_and_grad_timing,
        }
    output = {
        "status": "completed",
        "case": case["name"],
        "profile_c": profile_c,
        "jax_mode": jax_mode,
        "n_annuli_per_epoch": n_annuli_per_epoch,
        "jax": jax_lane,
        "microlux": (
            None
            if not include_micro
            else (
                {
                    "value": pack(micro_value_timing),
                    "gradient": pack(micro_gradient_timing),
                    "vjp": pack(micro_vjp_timing),
                    "value_and_grad": micro_value_and_grad_timing,
                    "compile_seconds": micro_compile_seconds,
                    "warnings": micro_warnings,
                }
                if measure_micro
                else {
                    "compile_seconds": float(micro_compile_seconds),
                    "warnings": micro_warnings,
                }
            )
        ),
    }
    Path(payload["output"]).write_text(
        json.dumps(output, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _run_worker_subprocess(
    output_dir,
    case,
    profile_c,
    times,
    n_annuli_per_epoch,
    cache,
    jax_mode,
    include_micro,
    tag=None,
    measure_micro=True,
):
    tag_suffix = "" if tag is None else f"_{tag}"
    stem = (
        f"{case['name']}_{'linear' if profile_c else 'uniform'}"
        f"_{jax_mode}{tag_suffix}"
    )
    input_path = output_dir / "workers" / f"{stem}.input.json"
    output_path = output_dir / "workers" / f"{stem}.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    # Never accidentally consume a result left by an earlier interrupted run.
    if output_path.exists():
        output_path.unlink()
    payload = {
        "case": case,
        "profile_c": profile_c,
        "times": np.asarray(times, dtype=float).tolist(),
        "n_annuli_per_epoch": n_annuli_per_epoch,
        "jax_mode": jax_mode,
        "include_microlux": bool(include_micro),
        "measure_microlux": bool(measure_micro),
        "jax_cache": str(cache / jax_mode / stem),
        "output": str(output_path),
    }
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    started = time.perf_counter()
    try:
        subprocess.run(
            [sys.executable, str(SCRIPT), "--worker-input", str(input_path)],
            check=True,
            cwd=str(ROOT),
            env=environment,
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except Exception as error:  # keep one failed lane auditable
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started,
        }
    if not output_path.is_file():
        return {
            "status": "failed",
            "error": "worker exited without result JSON",
            "elapsed_seconds": time.perf_counter() - started,
        }
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["worker_elapsed_seconds"] = time.perf_counter() - started
    return result


def _start_micro_server(
    output_dir,
    case,
    profile_c,
    times,
    n_annuli,
    cotangent,
    cache,
    tag,
):
    """Start one compiled microLUX group and keep it alive for timing.

    JAX's nested microLUX kernels do not reliably persist their inner
    executables to the cross-process cache.  A short-lived server process is
    therefore used: it compiles one static group, announces readiness, then
    waits for the parent to release it for uncontended value/Jacobian/VJP/
    value-and-gradient timing.
    """

    stem = (
        f"{case['name']}_{'linear' if profile_c else 'uniform'}"
        f"_micro_only_{tag}"
    )
    workers_dir = output_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    input_path = workers_dir / f"{stem}.input.json"
    output_path = workers_dir / f"{stem}.json"
    ready_path = workers_dir / f"{stem}.ready.json"
    release_path = workers_dir / f"{stem}.release"
    for path in (output_path, ready_path, release_path):
        if path.exists():
            path.unlink()

    payload = {
        "case": case,
        "profile_c": profile_c,
        "times": np.asarray(times, dtype=float).tolist(),
        "cotangent": np.asarray(cotangent, dtype=float).tolist(),
        "n_annuli_per_epoch": [int(n_annuli)] * int(np.asarray(times).size),
        "jax_mode": "micro_only",
        "include_microlux": True,
        "measure_microlux": False,
        "micro_server_ready": str(ready_path),
        "micro_server_release": str(release_path),
        "jax_cache": str(cache / "micro_only" / stem),
        "output": str(output_path),
    }
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--worker-input", str(input_path)],
        cwd=str(ROOT),
        env=environment,
    )
    return {
        "process": process,
        "output_path": output_path,
        "ready_path": ready_path,
        "release_path": release_path,
        "started": time.perf_counter(),
    }


def _micro_annulus_plan(record):
    plan = record.get("vbm_nannuli_per_epoch")
    if plan is None:
        return [10] * int(record["n_times"])
    return [int(value or 10) for value in plan]


def _combine_micro_group_timing(group_results, field):
    timings = [result["microlux"][field] for result in group_results]
    samples = list(zip(*(timing["steady_samples_seconds"] for timing in timings)))
    summed_samples = [float(sum(sample)) for sample in samples]
    return {
        "cold_seconds": float(sum(timing["cold_seconds"] for timing in timings)),
        "steady_samples_seconds": summed_samples,
        "steady_seconds": float(statistics.median(summed_samples)),
    }


def _run_micro_annulus_groups(output_dir, cache, record, case, profile_c):
    """Run one static microLUX executable per annulus count.

    A single JAX function containing every linear-LD annulus branch can retain
    several large LLVM executables at once.  Keeping each equal-annulus group
    in its own child process bounds peak memory while preserving the requested
    event-specific plan.  The reported steady samples are the sum of the
    group timings, i.e. the cost of evaluating the complete light curve.
    """

    plan = _micro_annulus_plan(record)
    group_specs = []
    for n_annuli in sorted(set(plan)):
        indices = np.asarray(
            [index for index, value in enumerate(plan) if value == n_annuli],
            dtype=int,
        )
        print(
            f"  microLUX group n_annuli={n_annuli} "
            f"epochs={indices.size}/{len(plan)}",
            flush=True,
        )
        group_specs.append({
            "n_annuli": int(n_annuli),
            "indices": indices.tolist(),
        })

    groups = []
    full_times = np.asarray(record["times"], dtype=float)
    full_cotangent = _cotangent_probe(full_times)

    def run_batch(batch):
        servers = []
        for spec in batch:
            indices = np.asarray(spec["indices"], dtype=int)
            server = _start_micro_server(
                output_dir,
                case,
                profile_c,
                full_times[indices],
                spec["n_annuli"],
                full_cotangent[indices],
                cache,
                tag=f"n{spec['n_annuli']}",
            )
            servers.append({**spec, "server": server})

        def failure(message):
            return [
                {
                    **{key: value for key, value in item.items() if key != "server"},
                    "compile_worker": {
                        "status": "failed",
                        "error": message,
                    },
                    "worker": {
                        "status": "failed",
                        "error": message,
                    },
                }
                for item in servers
            ]

        try:
            deadline = time.perf_counter() + WORKER_TIMEOUT_SECONDS
            ready = set()
            while len(ready) != len(servers):
                for index, item in enumerate(servers):
                    server = item["server"]
                    if index in ready:
                        continue
                    if server["ready_path"].is_file():
                        ready.add(index)
                    elif server["process"].poll() is not None:
                        raise RuntimeError(
                            f"microLUX group n_annuli={item['n_annuli']} "
                            "exited before compilation readiness"
                        )
                if len(ready) != len(servers):
                    if time.perf_counter() > deadline:
                        raise TimeoutError(
                            "microLUX groups did not all reach compile readiness"
                        )
                    time.sleep(0.05)

            completed_batch = []
            for item in servers:
                server = item["server"]
                server["release_path"].touch()
                try:
                    return_code = server["process"].wait(
                        timeout=WORKER_TIMEOUT_SECONDS
                    )
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError(
                        f"microLUX group n_annuli={item['n_annuli']} "
                        "timing worker exceeded its deadline"
                    ) from error
                if return_code != 0 or not server["output_path"].is_file():
                    raise RuntimeError(
                        f"microLUX group n_annuli={item['n_annuli']} "
                        f"timing worker failed (return code {return_code})"
                    )
                timing_worker = json.loads(
                    server["output_path"].read_text(encoding="utf-8")
                )
                timing_worker["worker_elapsed_seconds"] = (
                    time.perf_counter() - server["started"]
                )
                compile_info = json.loads(
                    server["ready_path"].read_text(encoding="utf-8")
                )
                compile_worker = {
                    "status": "completed",
                    "jax_mode": "micro_only",
                    "microlux": compile_info,
                    "worker_elapsed_seconds": (
                        time.perf_counter() - server["started"]
                    ),
                }
                serializable_item = {
                    key: value
                    for key, value in item.items()
                    if key != "server"
                }
                completed_batch.append({
                    **serializable_item,
                    "compile_worker": compile_worker,
                    "worker": timing_worker,
                })
            return completed_batch
        except Exception as error:
            for item in servers:
                process = item["server"]["process"]
                if process.poll() is None:
                    process.terminate()
            for item in servers:
                process = item["server"]["process"]
                if process.poll() is None:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            return failure(f"{type(error).__name__}: {error}")

    for start in range(0, len(group_specs), MICROLUX_GROUP_PARALLELISM):
        groups.extend(
            run_batch(group_specs[start:start + MICROLUX_GROUP_PARALLELISM])
        )

    completed = [
        group for group in groups
        if group["worker"].get("status") == "completed"
        and group["worker"].get("microlux") is not None
    ]
    if len(completed) != len(groups):
        return None, groups

    n_times = int(record["n_times"])
    full_values = np.full(n_times, np.nan, dtype=float)
    full_gradients = np.full(
        (n_times, len(GRADIENT_PARAMETER_KEYS)), np.nan, dtype=float
    )
    full_vjp = np.zeros(len(SAMPLER_PARAMETER_KEYS), dtype=float)
    full_value_and_grad_value = 0.0
    full_value_and_grad_gradient = np.zeros(
        len(SAMPLER_PARAMETER_KEYS), dtype=float
    )
    for group in completed:
        worker_lane = group["worker"]["microlux"]
        indices = np.asarray(group["indices"], dtype=int)
        full_values[indices] = np.asarray(
            worker_lane["value"]["values"], dtype=float
        )
        full_gradients[indices] = np.asarray(
            worker_lane["gradient"]["values"], dtype=float
        )
        full_vjp += np.asarray(worker_lane["vjp"]["values"], dtype=float)
        full_value_and_grad_value += float(
            worker_lane["value_and_grad"]["value"]
        )
        full_value_and_grad_gradient += np.asarray(
            worker_lane["value_and_grad"]["gradient"], dtype=float
        )

    warnings_seen = []
    for group in completed:
        for warning in group["worker"]["microlux"].get("warnings", []):
            if warning not in warnings_seen:
                warnings_seen.append(warning)

    lane = {
        "mode": "event_specific_grouped",
        "n_annuli_per_epoch": plan,
        "n_annuli_groups": [group["n_annuli"] for group in groups],
        "group_epoch_counts": [len(group["indices"]) for group in groups],
        "value": _combine_micro_group_timing(
            [group["worker"] for group in completed], "value"
        ) | {"values": full_values.tolist()},
        "gradient": _combine_micro_group_timing(
            [group["worker"] for group in completed], "gradient"
        ) | {"values": full_gradients.tolist()},
        "vjp": _combine_micro_group_timing(
            [group["worker"] for group in completed], "vjp"
        ) | {"values": full_vjp.tolist()},
        "value_and_grad": _combine_micro_group_timing(
            [group["worker"] for group in completed], "value_and_grad"
        ) | {
            "value": float(full_value_and_grad_value),
            "gradient": full_value_and_grad_gradient.tolist(),
        },
        "warnings": warnings_seen,
    }
    lane["accuracy_vs_vbm"] = _accuracy(
        np.asarray(record["vbm_values"], dtype=float), full_values
    )
    return lane, groups


def _build_baseline(lcbinint, case, profile_c):
    times = _case_times(case)
    params = dict(case["params"])
    vbm = _measure_vbm(times, params, profile_c)
    native = _measure_native(lcbinint, times, params, profile_c)
    nannuli_per_epoch, nannuli = _vbm_nannuli(times, params, profile_c)
    reference = vbm["value"]
    return {
        "case": case["name"],
        "profile": "C1_linear_ld" if profile_c else "C0_uniform",
        "profile_c": profile_c,
        "n_times": int(times.size),
        "parameters": params,
        "time_range": [float(times[0]), float(times[-1])],
        "times": times.tolist(),
        "vbm": {
            key: value for key, value in vbm.items() if key != "value"
        },
        "vbm_values": reference.tolist(),
        "native": native,
        "vbm_nannuli_per_epoch": nannuli_per_epoch,
        "vbm_event_n_annuli": nannuli,
    }


def _ratio(left, right):
    if left is None or right is None or right == 0.0:
        return None
    return float(left) / float(right)


def _augment_record(record, worker):
    return _augment_lane(record, worker, "jax_warmup")


def _augment_lane(record, worker, key):
    record.setdefault("workers", {})[key] = worker
    reference = np.asarray(record["vbm_values"], dtype=float)
    completed = worker.get("status") == "completed"
    jax_lane = worker.get("jax") if completed else None
    if jax_lane is None:
        record[key] = None
    else:
        values = np.asarray(jax_lane["value"]["values"], dtype=float)
        record[key] = {
            **jax_lane,
            "accuracy_vs_vbm": _accuracy(reference, values),
        }
    micro_lane = worker.get("microlux") if completed else None
    if micro_lane is not None:
        values = np.asarray(micro_lane["value"]["values"], dtype=float)
        record["microlux"] = {
            **micro_lane,
            "accuracy_vs_vbm": _accuracy(reference, values),
        }
    return record


def _stats(records, engine, field="steady_seconds"):
    values = []
    for record in records:
        lane = record.get(engine)
        if lane and lane.get("value", {}).get(field) is not None:
            values.append(float(lane["value"][field]))
    if not values:
        return None
    return float(statistics.median(values))


def _write_report(path, payload):
    records = payload["records"]
    lines = [
        "# Six synthetic light-curve comparison: native/JAX/microLUX/VBM",
        "",
        "This run reuses the six paper-facing binary cases and their exact time "
        "windows from the VBM/native synthetic benchmark.",
        "",
        f"- tolerance and retol: `{TOLERANCE:g}`",
        f"- steady timing: median of `{REPEATS}` calls after first-call compilation/warm-up",
        "- JAX and microLUX compilation is reported separately and excluded from steady times",
        "- linear microLUX uses the maximum VBM `nannuli` over the full curve as one static annulus count",
        "- no independent VBM gradient reference is available; T_J is timing-only",
        "",
        "## Steady-state light-curve timing",
        "",
        "Times are milliseconds per epoch. `R_X/VBM` is the per-row block-time "
        "ratio; values below one mean that X is faster than VBM.",
        "",
        "| case | profile | epochs | native warm | JAX | microLUX | VBM | JAX/VBM | microLUX/VBM | finite JAX | finite microLUX | max rel. err JAX | max rel. err microLUX | VBM nannuli |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        native = record["native"]["warmup"]
        jax = record.get("jax")
        micro = record.get("microlux")
        vbm = record["vbm"]
        native_ms = None if native is None else 1e3 * native["steady_seconds"] / record["n_times"]
        jax_ms = None if jax is None else 1e3 * jax["value"]["steady_seconds"] / record["n_times"]
        micro_ms = None if micro is None else 1e3 * micro["value"]["steady_seconds"] / record["n_times"]
        vbm_ms = 1e3 * vbm["steady_seconds"] / record["n_times"]
        jax_ratio = None if jax is None else jax["value"]["steady_seconds"] / vbm["steady_seconds"]
        micro_ratio = None if micro is None else micro["value"]["steady_seconds"] / vbm["steady_seconds"]
        jax_error = None if jax is None else jax["accuracy_vs_vbm"]["max_relative"]
        micro_error = None if micro is None else micro["accuracy_vs_vbm"]["max_relative"]
        jax_finite = None if jax is None else jax["accuracy_vs_vbm"]["finite_epochs"]
        micro_finite = None if micro is None else micro["accuracy_vs_vbm"]["finite_epochs"]
        jax_finite_text = (
            "n/a" if jax_finite is None else f"{jax_finite}/{record['n_times']}"
        )
        micro_finite_text = (
            "n/a" if micro_finite is None else f"{micro_finite}/{record['n_times']}"
        )
        lines.append(
            f"| `{record['case']}` | {record['profile']} | {record['n_times']} | "
            f"{native_ms:.5g} | {('n/a' if jax_ms is None else format(jax_ms, '.5g'))} | "
            f"{('n/a' if micro_ms is None else format(micro_ms, '.5g'))} | {vbm_ms:.5g} | "
            f"{('n/a' if jax_ratio is None else format(jax_ratio, '.4g'))} | "
            f"{('n/a' if micro_ratio is None else format(micro_ratio, '.4g'))} | "
            f"{jax_finite_text} | {micro_finite_text} | "
            f"{('n/a' if jax_error is None else format(jax_error, '.3e'))} | "
            f"{('n/a' if micro_error is None else format(micro_error, '.3e'))} | "
            f"{('n/a' if record['vbm_event_n_annuli'] is None else record['vbm_event_n_annuli'])} |"
        )
    lines.extend([
        "",
        "## T_J steady timing",
        "",
        "No independent VBM gradient curve is included in the original six-case corpus, so these are timing comparisons only.",
        "",
        "| case | profile | epochs | JAX T_J [ms/curve] | microLUX T_J [ms/curve] | microLUX/JAX |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for record in records:
        jax = record.get("jax")
        micro = record.get("microlux")
        jax_d = None if jax is None else jax["gradient"]["steady_seconds"]
        micro_d = None if micro is None else micro["gradient"]["steady_seconds"]
        lines.append(
            f"| `{record['case']}` | {record['profile']} | {record['n_times']} | "
            f"{('n/a' if jax_d is None else format(1e3 * jax_d, '.5g'))} | "
            f"{('n/a' if micro_d is None else format(1e3 * micro_d, '.5g'))} | "
            f"{('n/a' if jax_d is None or micro_d is None else format(micro_d / jax_d, '.4g'))} |"
        )
    lines.extend([
        "",
        "## Aggregate light-curve ratios",
        "",
        "The win fraction is computed over the six curves within each profile; `R` is the median of per-curve block-time ratios.",
        "",
        "| profile | JAX faster than microLUX | median microLUX/JAX | JAX faster than native warm | median native/JAX | JAX faster than VBM | median VBM/JAX |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for profile in ("C0_uniform", "C1_linear_ld"):
        subset = [r for r in records if r["profile"] == profile]
        pairs = {
            "micro": [(r["jax"]["value"]["steady_seconds"], r["microlux"]["value"]["steady_seconds"]) for r in subset if r.get("jax") and r.get("microlux")],
            "native": [(r["jax"]["value"]["steady_seconds"], r["native"]["warmup"]["steady_seconds"]) for r in subset if r.get("jax") and r["native"].get("warmup")],
            "vbm": [(r["jax"]["value"]["steady_seconds"], r["vbm"]["steady_seconds"]) for r in subset if r.get("jax")],
        }
        def aggregate(key):
            values = pairs[key]
            return (
                f"{sum(left < right for left, right in values)}/{len(values)} "
                f"({sum(left < right for left, right in values) / len(values):.1%})",
                statistics.median(right / left for left, right in values),
            )
        micro_win, micro_r = aggregate("micro")
        native_win, native_r = aggregate("native")
        vbm_win, vbm_r = aggregate("vbm")
        lines.append(
            f"| {profile} | {micro_win} | {micro_r:.4g} | {native_win} | {native_r:.4g} | {vbm_win} | {vbm_r:.4g} |"
        )
    lines.extend([
        "",
        "## Aggregate T_J ratios",
        "",
        "`R_micro/JAX` is the median of the per-curve steady parameter-gradient-time ratios.",
        "",
        "| profile | JAX faster than microLUX | median microLUX/JAX |",
        "|---|---:|---:|",
    ])
    for profile in ("C0_uniform", "C1_linear_ld"):
        pairs = [
            (
                record["jax"]["gradient"]["steady_seconds"],
                record["microlux"]["gradient"]["steady_seconds"],
            )
            for record in records
            if record["profile"] == profile
            and record.get("jax")
            and record.get("microlux")
        ]
        wins = sum(jax < micro for jax, micro in pairs)
        ratio = statistics.median(micro / jax for jax, micro in pairs)
        lines.append(
            f"| {profile} | {wins}/{len(pairs)} ({wins / len(pairs):.1%}) | {ratio:.4g} |"
        )
    lines.extend([
        "",
        "## Accuracy audit against VBM",
        "",
        f"The requested cross-code tolerance is `{TOLERANCE:g}`. A pass requires every epoch to be finite and within that tolerance; speed win rates above are timing statistics and do not treat a numerical failure as a pass.",
        "",
        "| profile | JAX full-curve passes | microLUX full-curve passes | JAX invalid epochs | microLUX invalid epochs |",
        "|---|---:|---:|---:|---:|",
    ])
    for profile in ("C0_uniform", "C1_linear_ld"):
        subset = [r for r in records if r["profile"] == profile]
        jax_lanes = [r["jax"] for r in subset if r.get("jax")]
        micro_lanes = [r["microlux"] for r in subset if r.get("microlux")]
        jax_passes = sum(
            lane["accuracy_vs_vbm"].get("within_tolerance", False)
            for lane in jax_lanes
        )
        micro_passes = sum(
            lane["accuracy_vs_vbm"].get("within_tolerance", False)
            for lane in micro_lanes
        )
        jax_invalid = sum(
            lane["accuracy_vs_vbm"].get("invalid_epochs", 0)
            for lane in jax_lanes
        )
        micro_invalid = sum(
            lane["accuracy_vs_vbm"].get("invalid_epochs", 0)
            for lane in micro_lanes
        )
        lines.append(
            f"| {profile} | {jax_passes}/{len(jax_lanes)} | "
            f"{micro_passes}/{len(micro_lanes)} | {jax_invalid} | {micro_invalid} |"
        )
    lines.extend([
        "",
        "## Reproduction artifacts",
        "",
        f"- input baseline: `{REFERENCE_RUN}`",
        "- the JSON stores the full time grid, four engine curves, parameter-gradient curves, timing samples, and VBM annulus counts",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report_v2(path, payload):
    records = payload["records"]

    def get(mapping, *keys):
        for key in keys:
            if mapping is None:
                return None
            mapping = mapping.get(key)
        return mapping

    def fmt(value, digits=5):
        return "n/a" if value is None else format(float(value), f".{digits}g")

    def ratio(left, right):
        if left is None or right in (None, 0):
            return None
        return float(left) / float(right)

    def aggregate(pairs):
        if not pairs:
            return "n/a", None
        wins = sum(left < right for left, right in pairs)
        return (
            f"{wins}/{len(pairs)} ({wins / len(pairs):.1%})",
            statistics.median(right / left for left, right in pairs if left > 0),
        )

    lines = [
        "# Six synthetic light-curve comparison: JAX warm-up/no-warm-up/microLUX",
        "",
        "The JAX no-warm-up lane still includes XLA compilation in its cold call.",
        "The JAX warm-up lane uses the JAX automatic route/bin proposal, checks it",
        "against the native self-converged reference, then compiles the fixed plan",
        "and primes the FFI support cache before steady timing.",
        "microLUX uses the VBM-selected nannuli for each epoch; each equal-count "
        "group is compiled in an isolated worker and group costs are summed.",
        f"T_J is the full parameter-gradient curve G[i,k] with P={len(GRADIENT_PARAMETER_KEYS)} "
        f"and parameter order {', '.join(GRADIENT_PARAMETER_KEYS)}.",
        "T_VG is a model-only value-and-gradient call for the scalar "
        "S(eta)=w^T A(eta); it returns S and grad S together in sampler "
        "coordinates.",
        f"T_VJP is a model-only reverse-mode pullback in sampler coordinates "
        f"{', '.join(SAMPLER_PARAMETER_KEYS)} using a deterministic dense cotangent probe; "
        "it contains no observed data or likelihood evaluation.",
        "",
        f"- tolerance and retol: {TOLERANCE:g}",
        f"- steady repeats: {REPEATS}",
        "",
        "## Forward steady timing [ms/epoch]",
        "",
        "| case | profile | native warm | JAX no warm | JAX warm | microLUX | VBM | no/VBM | warm/VBM | micro/VBM | warm/no | nannuli max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        native = record["native"].get("warmup")
        no_lane = record.get("jax_no_warmup")
        warm_lane = record.get("jax_warmup")
        micro = record.get("microlux")
        vbm = record["vbm"]
        native_ms = None if native is None else 1e3 * native["steady_seconds"] / record["n_times"]
        no_ms = None if no_lane is None else 1e3 * no_lane["value"]["steady_seconds"] / record["n_times"]
        warm_ms = None if warm_lane is None else 1e3 * warm_lane["value"]["steady_seconds"] / record["n_times"]
        micro_ms = None if micro is None else 1e3 * micro["value"]["steady_seconds"] / record["n_times"]
        vbm_ms = 1e3 * vbm["steady_seconds"] / record["n_times"]
        lines.append(
            f"| {record['case']} | {record['profile']} | {fmt(native_ms)} | "
            f"{fmt(no_ms)} | {fmt(warm_ms)} | {fmt(micro_ms)} | {fmt(vbm_ms)} | "
            f"{fmt(ratio(no_ms, vbm_ms), 4)} | {fmt(ratio(warm_ms, vbm_ms), 4)} | "
            f"{fmt(ratio(micro_ms, vbm_ms), 4)} | {fmt(ratio(warm_ms, no_ms), 4)} | "
            f"{record.get('vbm_event_n_annuli') or 10} |"
        )

    lines.extend([
        "",
        "## Warm-up/cold accounting [s]",
        "",
        "| case | profile | JAX no-warm value cold | JAX no-warm T_J cold | JAX no-warm T_VJP cold | JAX no-warm T_VG cold | JAX warm-up total | fixed-plan compile | Jacobian compile | VJP compile | value+grad compile | seed cache primed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for record in records:
        no_lane = record.get("jax_no_warmup")
        warm_lane = record.get("jax_warmup")
        lines.append(
            f"| {record['case']} | {record['profile']} | "
            f"{fmt(get(no_lane, 'value', 'cold_seconds'))} | "
            f"{fmt(get(no_lane, 'gradient', 'cold_seconds'))} | "
            f"{fmt(get(no_lane, 'vjp', 'cold_seconds'))} | "
            f"{fmt(get(no_lane, 'value_and_grad', 'cold_seconds'))} | "
            f"{fmt(get(warm_lane, 'warmup_seconds'))} | "
            f"{fmt(get(warm_lane, 'jax_compile_seconds'))} | "
            f"{fmt(get(warm_lane, 'gradient_compile_seconds'))} | "
            f"{fmt(get(warm_lane, 'vjp_compile_seconds'))} | "
            f"{fmt(get(warm_lane, 'value_and_grad_compile_seconds'))} | "
            f"{get(warm_lane, 'seed_cache_primed')} |"
        )

    lines.extend([
        "",
        "## Parameter-gradient steady timing: T_J [ms/curve]",
        "",
        "Each timing is for one complete Jacobian curve, not one epoch. The returned array has shape `(n_times, 7)`. ",
        "| case | profile | JAX no warm | JAX warm | microLUX | micro/no | micro/warm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        no_lane = record.get("jax_no_warmup")
        warm_lane = record.get("jax_warmup")
        micro = record.get("microlux")
        no_j = None if no_lane is None else 1e3 * no_lane["gradient"]["steady_seconds"]
        warm_j = None if warm_lane is None else 1e3 * warm_lane["gradient"]["steady_seconds"]
        micro_j = None if micro is None else 1e3 * micro["gradient"]["steady_seconds"]
        lines.append(
            f"| {record['case']} | {record['profile']} | {fmt(no_j)} | "
            f"{fmt(warm_j)} | {fmt(micro_j)} | {fmt(ratio(micro_j, no_j), 4)} | "
            f"{fmt(ratio(micro_j, warm_j), 4)} |"
        )

    lines.extend([
        "",
        "## Model-only value-and-gradient timing: T_VG [ms/curve]",
        "",
        "The timed callable returns both the scalar model contraction "
        "S(eta)=w^T A(eta) and its sampler-coordinate gradient in one "
        "`jax.value_and_grad` transform. The cotangent probe is generated "
        "from the time-grid shape only and is outside the timed region; this "
        "is still a model-only proxy, not a likelihood evaluation.",
        "| case | profile | JAX no warm | JAX warm | microLUX | micro/no | micro/warm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        no_lane = record.get("jax_no_warmup")
        warm_lane = record.get("jax_warmup")
        micro = record.get("microlux")
        no_vg = None if no_lane is None else 1e3 * no_lane["value_and_grad"]["steady_seconds"]
        warm_vg = None if warm_lane is None else 1e3 * warm_lane["value_and_grad"]["steady_seconds"]
        micro_vg = None if micro is None else 1e3 * micro["value_and_grad"]["steady_seconds"]
        lines.append(
            f"| {record['case']} | {record['profile']} | {fmt(no_vg)} | "
            f"{fmt(warm_vg)} | {fmt(micro_vg)} | {fmt(ratio(micro_vg, no_vg), 4)} | "
            f"{fmt(ratio(micro_vg, warm_vg), 4)} |"
        )

    lines.extend([
        "",
        "## Diagnostic reverse-mode timing: T_VJP [ms/curve]",
        "",
        "This legacy diagnostic times the reverse-mode pullback J^T w alone; "
        "the scalar value is not returned. It is retained for implementation "
        "diagnostics, while T_VG is the primary value-plus-gradient comparison.",
        "| case | profile | JAX no warm | JAX warm | microLUX | micro/no | micro/warm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        no_lane = record.get("jax_no_warmup")
        warm_lane = record.get("jax_warmup")
        micro = record.get("microlux")
        no_v = None if no_lane is None else 1e3 * no_lane["vjp"]["steady_seconds"]
        warm_v = None if warm_lane is None else 1e3 * warm_lane["vjp"]["steady_seconds"]
        micro_v = None if micro is None else 1e3 * micro["vjp"]["steady_seconds"]
        lines.append(
            f"| {record['case']} | {record['profile']} | {fmt(no_v)} | "
            f"{fmt(warm_v)} | {fmt(micro_v)} | {fmt(ratio(micro_v, no_v), 4)} | "
            f"{fmt(ratio(micro_v, warm_v), 4)} |"
        )

    lines.extend([
        "",
        "## Aggregate forward win rates",
        "",
        "| profile | no-warm faster micro | warm faster micro | warm faster no-warm | warm faster native | median micro/no | median micro/warm | median no/warm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for profile in ("C0_uniform", "C1_linear_ld"):
        subset = [record for record in records if record["profile"] == profile]
        no_micro = [
            (r["jax_no_warmup"]["value"]["steady_seconds"], r["microlux"]["value"]["steady_seconds"])
            for r in subset if r.get("jax_no_warmup") and r.get("microlux")
        ]
        warm_micro = [
            (r["jax_warmup"]["value"]["steady_seconds"], r["microlux"]["value"]["steady_seconds"])
            for r in subset if r.get("jax_warmup") and r.get("microlux")
        ]
        warm_no = [
            (r["jax_warmup"]["value"]["steady_seconds"], r["jax_no_warmup"]["value"]["steady_seconds"])
            for r in subset if r.get("jax_warmup") and r.get("jax_no_warmup")
        ]
        warm_native = [
            (r["jax_warmup"]["value"]["steady_seconds"], r["native"]["warmup"]["steady_seconds"])
            for r in subset if r.get("jax_warmup") and r["native"].get("warmup")
        ]
        no_win, no_ratio = aggregate(no_micro)
        warm_win, warm_ratio = aggregate(warm_micro)
        warm_no_win, warm_no_ratio = aggregate(warm_no)
        native_win, _ = aggregate(warm_native)
        lines.append(
            f"| {profile} | {no_win} | {warm_win} | {warm_no_win} | {native_win} | "
            f"{fmt(no_ratio, 4)} | {fmt(warm_ratio, 4)} | {fmt(warm_no_ratio, 4)} |"
        )

    lines.extend([
        "",
        "## Aggregate T_J win rates",
        "",
        "| profile | no-warm faster micro | warm faster micro | median micro/no | median micro/warm |",
        "|---|---:|---:|---:|---:|",
    ])
    for profile in ("C0_uniform", "C1_linear_ld"):
        no_pairs = [
            (r["jax_no_warmup"]["gradient"]["steady_seconds"], r["microlux"]["gradient"]["steady_seconds"])
            for r in records if r["profile"] == profile and r.get("jax_no_warmup") and r.get("microlux")
        ]
        warm_pairs = [
            (r["jax_warmup"]["gradient"]["steady_seconds"], r["microlux"]["gradient"]["steady_seconds"])
            for r in records if r["profile"] == profile and r.get("jax_warmup") and r.get("microlux")
        ]
        no_win, no_ratio = aggregate(no_pairs)
        warm_win, warm_ratio = aggregate(warm_pairs)
        lines.append(
            f"| {profile} | {no_win} | {warm_win} | {fmt(no_ratio, 4)} | {fmt(warm_ratio, 4)} |"
        )

    lines.extend([
        "",
        "## Aggregate T_VG win rates",
        "",
        "`R_micro/JAX` is the median per-curve steady-time ratio for the model-only value-and-gradient callable.",
        "",
        "| profile | no-warm faster micro | warm faster micro | median micro/no | median micro/warm |",
        "|---|---:|---:|---:|---:|",
    ])
    for profile in ("C0_uniform", "C1_linear_ld"):
        no_pairs = [
            (r["jax_no_warmup"]["value_and_grad"]["steady_seconds"], r["microlux"]["value_and_grad"]["steady_seconds"])
            for r in records if r["profile"] == profile and r.get("jax_no_warmup") and r.get("microlux")
        ]
        warm_pairs = [
            (r["jax_warmup"]["value_and_grad"]["steady_seconds"], r["microlux"]["value_and_grad"]["steady_seconds"])
            for r in records if r["profile"] == profile and r.get("jax_warmup") and r.get("microlux")
        ]
        no_win, no_ratio = aggregate(no_pairs)
        warm_win, warm_ratio = aggregate(warm_pairs)
        lines.append(
            f"| {profile} | {no_win} | {warm_win} | {fmt(no_ratio, 4)} | {fmt(warm_ratio, 4)} |"
        )

    lines.extend([
        "",
        "## Aggregate diagnostic T_VJP win rates",
        "",
        "This section is diagnostic only: T_VJP times the pullback without returning the scalar value. T_VG is the primary value-plus-gradient metric.",
        "",
        "| profile | no-warm faster micro | warm faster micro | median micro/no | median micro/warm |",
        "|---|---:|---:|---:|---:|",
    ])
    for profile in ("C0_uniform", "C1_linear_ld"):
        no_pairs = [
            (r["jax_no_warmup"]["vjp"]["steady_seconds"], r["microlux"]["vjp"]["steady_seconds"])
            for r in records if r["profile"] == profile and r.get("jax_no_warmup") and r.get("microlux")
        ]
        warm_pairs = [
            (r["jax_warmup"]["vjp"]["steady_seconds"], r["microlux"]["vjp"]["steady_seconds"])
            for r in records if r["profile"] == profile and r.get("jax_warmup") and r.get("microlux")
        ]
        no_win, no_ratio = aggregate(no_pairs)
        warm_win, warm_ratio = aggregate(warm_pairs)
        lines.append(
            f"| {profile} | {no_win} | {warm_win} | {fmt(no_ratio, 4)} | {fmt(warm_ratio, 4)} |"
        )

    lines.extend([
        "",
        "## Accuracy audit against VBM",
        "",
        f"Full-curve pass means every epoch is finite and has relative error at most {TOLERANCE:g}.",
        "",
        "| profile | JAX no-warm passes | JAX warm passes | microLUX passes | no invalid | warm invalid | micro invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for profile in ("C0_uniform", "C1_linear_ld"):
        subset = [record for record in records if record["profile"] == profile]
        no_lanes = [r["jax_no_warmup"] for r in subset if r.get("jax_no_warmup")]
        warm_lanes = [r["jax_warmup"] for r in subset if r.get("jax_warmup")]
        micro_lanes = [r["microlux"] for r in subset if r.get("microlux")]
        no_pass = sum(r["accuracy_vs_vbm"].get("within_tolerance", False) for r in no_lanes)
        warm_pass = sum(r["accuracy_vs_vbm"].get("within_tolerance", False) for r in warm_lanes)
        micro_pass = sum(r["accuracy_vs_vbm"].get("within_tolerance", False) for r in micro_lanes)
        no_invalid = sum(r["accuracy_vs_vbm"].get("invalid_epochs", 0) for r in no_lanes)
        warm_invalid = sum(r["accuracy_vs_vbm"].get("invalid_epochs", 0) for r in warm_lanes)
        micro_invalid = sum(r["accuracy_vs_vbm"].get("invalid_epochs", 0) for r in micro_lanes)
        lines.append(
            f"| {profile} | {no_pass}/{len(no_lanes)} | {warm_pass}/{len(warm_lanes)} | "
            f"{micro_pass}/{len(micro_lanes)} | {no_invalid} | {warm_invalid} | {micro_invalid} |"
        )

    lines.extend([
        "",
        "## Reproduction artifacts",
        "",
        f"- input baseline: {REFERENCE_RUN}",
        "- results.json contains both JAX lanes, native/microLUX/VBM curves, full Jacobians, model-only value-and-gradients and diagnostic VJPs, timing samples, and per-epoch VBM annulus counts.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(output_dir, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for profile in ("C0_uniform", "C1_linear_ld"):
        subset = [record for record in records if record["profile"] == profile]
        figure, axes = plt.subplots(2, 3, figsize=(15.0, 7.5), squeeze=False)
        for axis, record in zip(axes.ravel(), subset):
            times = np.asarray(record["times"], dtype=float)
            reference = np.asarray(record["vbm_values"], dtype=float)
            axis.plot(times, reference, label="VBM", color="tab:orange", lw=1.2)
            native = record["native"].get("warm_values")
            if native is not None:
                axis.plot(times, native, label="native warm", color="0.2", ls="--", lw=0.8)
            if record.get("jax_no_warmup"):
                axis.plot(
                    times,
                    record["jax_no_warmup"]["value"]["values"],
                    label="lcbinint JAX no warm-up",
                    color="tab:cyan",
                    lw=0.8,
                )
            if record.get("jax_warmup"):
                axis.plot(
                    times,
                    record["jax_warmup"]["value"]["values"],
                    label="lcbinint JAX warm-up",
                    color="tab:blue",
                    lw=0.9,
                )
            if record.get("microlux"):
                axis.plot(times, record["microlux"]["value"]["values"], label="microLUX", color="tab:green", lw=0.9)
            axis.set_title(record["case"])
            axis.set_xlabel("time")
            axis.set_ylabel("magnification")
            axis.grid(alpha=0.2)
        if subset:
            axes.ravel()[0].legend(fontsize=8)
        figure.suptitle(f"Six synthetic binary light curves: {profile}")
        figure.tight_layout()
        figure.savefig(output_dir / f"lightcurves_{profile}.pdf")
        figure.savefig(output_dir / f"lightcurves_{profile}.png", dpi=170)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(16.0, 5.5), squeeze=False)
    for axis, profile in zip(axes.ravel(), ("C0_uniform", "C1_linear_ld")):
        subset = [record for record in records if record["profile"] == profile]
        positions = np.arange(len(subset), dtype=float)
        width = 0.15
        for offset, label, color, getter in (
            (-2.0, "native warm", "0.25", lambda r: r["native"]["warmup"]),
            (-1.0, "JAX no warm", "tab:cyan", lambda r: (r.get("jax_no_warmup") or {}).get("value")),
            (0.0, "JAX warm", "tab:blue", lambda r: (r.get("jax_warmup") or {}).get("value")),
            (1.0, "microLUX", "tab:green", lambda r: (r.get("microlux") or {}).get("value")),
            (2.0, "VBM", "tab:orange", lambda r: r["vbm"]),
        ):
            values = []
            for record in subset:
                timing = getter(record)
                values.append(np.nan if timing is None else 1e3 * timing["steady_seconds"] / record["n_times"])
            axis.bar(positions + offset * width, values, width, label=label, color=color)
        axis.set_xticks(positions, [record["case"] for record in subset], rotation=35, ha="right")
        axis.set_yscale("log")
        axis.set_ylabel("steady ms / epoch")
        axis.set_title(profile)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Compiled steady-state light-curve speed comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "lightcurve_speed.pdf")
    figure.savefig(output_dir / "lightcurve_speed.png", dpi=170)
    plt.close(figure)


def _run_parent(args):
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = (output_dir / "jax_cache").resolve()
    _load_benchmark_backend()
    import lcbinint

    selected_cases = tuple(
        case for case in CASES
        if not args.cases or case["name"] in set(args.cases)
    )
    selected_profiles = (
        tuple(
            value
            for name, value in (("uniform", 0.0), ("linear", LIMB_DARKENING_C))
            if not args.profiles or name in set(args.profiles)
        )
    )
    jobs = [
        (case, profile_c)
        for case in selected_cases
        for profile_c in selected_profiles
    ]
    records = []
    for case, profile_c in jobs:
        profile = "C1_linear_ld" if profile_c else "C0_uniform"
        print(f"baseline {case['name']}/{profile}", flush=True)
        records.append(_build_baseline(lcbinint, case, profile_c))

    for record, (case, profile_c) in zip(records, jobs):
        print(
            f"JAX no-warmup {record['case']}/{record['profile']} "
            f"n={record['n_times']}",
            flush=True,
        )
        no_warmup_worker = _run_worker_subprocess(
            output_dir,
            case,
            profile_c,
            record["times"],
            record["vbm_nannuli_per_epoch"],
            cache,
            "no_warmup",
            False,
        )
        _augment_lane(record, no_warmup_worker, "jax_no_warmup")
        if no_warmup_worker.get("status") != "completed":
            print(
                f"  no-warmup worker failed: {no_warmup_worker.get('error')}",
                flush=True,
            )

        print(
            f"JAX warmup {record['case']}/{record['profile']} "
            f"n={record['n_times']} annuli="
            f"{record['vbm_event_n_annuli']}",
            flush=True,
        )
        warmup_worker = _run_worker_subprocess(
            output_dir,
            case,
            profile_c,
            record["times"],
            record["vbm_nannuli_per_epoch"],
            cache,
            "warmup",
            False,
        )
        _augment_lane(record, warmup_worker, "jax_warmup")
        if warmup_worker.get("status") != "completed":
            print(
                f"  warmup worker failed: {warmup_worker.get('error')}",
                flush=True,
            )
        print(
            f"microLUX event-specific groups {record['case']}/{record['profile']}",
            flush=True,
        )
        micro_lane, micro_groups = _run_micro_annulus_groups(
            output_dir, cache, record, case, profile_c
        )
        record.setdefault("workers", {})["microlux_groups"] = micro_groups
        record["microlux"] = micro_lane
        if micro_lane is None:
            failed = [
                group["worker"].get("error", "unknown worker failure")
                for group in micro_groups
                if group["worker"].get("status") != "completed"
            ]
            print(f"  microLUX group failure(s): {failed}", flush=True)
        # Keep the old key as a compatibility alias for small downstream
        # scripts; all new report tables use the explicit lane names.
        record["jax"] = record.get("jax_warmup")

    payload = {
        "benchmark": "synthetic_lightcurve_jax_gradient_microlux",
        "tolerance": TOLERANCE,
        "limb_darkening_coefficient": LIMB_DARKENING_C,
        "gradient_parameter_keys": list(GRADIENT_PARAMETER_KEYS),
        "gradient_definition": "G[i,k] = dA(t_i) / dtheta[k]",
        "sampler_parameter_keys": list(SAMPLER_PARAMETER_KEYS),
        "vjp_definition": "grad_theta dot(w, A(t; theta))",
        "cotangent_probe": "deterministic normalized sin/cos sequence from time-grid length only",
        "repeats_after_first": REPEATS,
        "cases": [case["name"] for case in CASES],
        "reference_run": str(REFERENCE_RUN),
        "annulus_policy": "event-specific VBM nannuli reused by grouped microLUX callables",
        "timing_policy": (
            "JAX no-warmup cold includes generic XLA compile; JAX warmup uses the "
            "JAX automatic route/bin proposal plus native reference certification, "
            "then fixed-plan value/Jacobian/VJP/value-and-grad compile; "
            "steady values are medians of repeated calls"
        ),
        "records": records,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_report_v2(output_dir / "REPORT.md", payload)
    _write_plots(output_dir, records)
    print(f"report: {output_dir / 'REPORT.md'}")
    print(f"json: {output_dir / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument(
        "--cases", nargs="+", choices=[case["name"] for case in CASES],
        help="limit a run to selected paper cases",
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=("uniform", "linear"),
        help="limit a run to selected limb-darkening profiles",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="rebuild the report and plots from an existing results.json",
    )
    args = parser.parse_args()
    if args.worker_input is not None:
        _run_worker(json.loads(args.worker_input.read_text(encoding="utf-8")))
    elif args.report_only:
        output_dir = args.output_dir.resolve()
        result_path = output_dir / "results.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["benchmark"] = "synthetic_lightcurve_jax_gradient_microlux"
        payload["gradient_parameter_keys"] = list(GRADIENT_PARAMETER_KEYS)
        payload["gradient_definition"] = "G[i,k] = dA(t_i) / dtheta[k]"
        payload["sampler_parameter_keys"] = list(SAMPLER_PARAMETER_KEYS)
        payload["vjp_definition"] = "grad_theta dot(w, A(t; theta))"
        payload["cotangent_probe"] = (
            "deterministic normalized sin/cos sequence from time-grid length only"
        )
        payload["timing_policy"] = (
            "JAX no-warmup cold includes generic XLA compile; JAX warmup uses the "
            "JAX automatic route/bin proposal plus native reference certification, "
            "then fixed-plan value/Jacobian/VJP/value-and-grad compile; "
            "steady values are medians of repeated calls"
        )
        for record in payload["records"]:
            reference = record["vbm_values"]
            for engine in ("jax_no_warmup", "jax_warmup", "jax", "microlux"):
                lane = record.get(engine)
                if lane is not None:
                    lane["accuracy_vs_vbm"] = _accuracy(
                        reference, lane["value"]["values"]
                    )
        result_path.write_text(
            json.dumps(payload, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        _write_report_v2(output_dir / "REPORT.md", payload)
        _write_plots(output_dir, payload["records"])
        print(f"report: {output_dir / 'REPORT.md'}")
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()

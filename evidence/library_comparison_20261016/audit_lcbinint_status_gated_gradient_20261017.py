#!/usr/bin/env python3
"""Matched uniform/LD timing audit for this branch's default LightCurve API."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.meta_path = [finder for finder in sys.meta_path if "_editable" not in type(finder).__module__]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "issue6_lcbinint_status_gated_gradient_20261017"
INPUT = HERE / "corrected_value_gradient_12cases" / "inputs.json"
CASE_IDS = ("large_source_uniform", "large_source_ld05_user")
TOL = 1.0e-3
REPETITIONS = 15

PYTHON_SOURCE = ROOT / "python" / "lcbinint"
BUILD_SOURCE = ROOT / "build-algebraic-boundary" / "python-api" / "lcbinint"
if not any(BUILD_SOURCE.glob("_lcbinint*.so")):
    raise RuntimeError(f"missing current algebraic-boundary build: {BUILD_SOURCE}")
sys.path.insert(0, str(ROOT / "python"))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "lcbinint",
    PYTHON_SOURCE / "__init__.py",
    submodule_search_locations=[str(PYTHON_SOURCE), str(BUILD_SOURCE)],
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load the in-tree lcbinint package")
lcbinint = importlib.util.module_from_spec(spec)
sys.modules["lcbinint"] = lcbinint
spec.loader.exec_module(lcbinint)

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def block_record(record, with_gradient: bool) -> None:
    leaves = [record.magnifications, record.status]
    if with_gradient:
        leaves.extend((record.gradients, record.gradient_valid))
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def call_timed(curve, times, params, with_gradient: bool):
    start = time.perf_counter()
    result = curve(times, params, with_gradient=with_gradient)
    block_record(result, with_gradient)
    return result, (time.perf_counter() - start) * 1.0e3


def paired_times(curve, times, params):
    value_result, value_first = call_timed(curve, times, params, False)
    grad_result, grad_first = call_timed(curve, times, params, True)
    samples = {"value_only": [], "value_plus_gradient": []}
    for i in range(REPETITIONS):
        order = ("value_only", "value_plus_gradient")
        if i % 2:
            order = tuple(reversed(order))
        for mode in order:
            _, elapsed = call_timed(
                curve, times, params, mode == "value_plus_gradient"
            )
            samples[mode].append(elapsed)
    medians = {key: float(np.median(values)) for key, values in samples.items()}
    timing = {
        "first_call_ms": {
            "value_only": value_first,
            "value_plus_gradient": grad_first,
        },
        "samples_ms": samples,
        "median_ms": medians,
        "p95_ms": {
            key: float(np.percentile(values, 95)) for key, values in samples.items()
        },
        "value_plus_gradient_over_value": (
            medians["value_plus_gradient"] / medians["value_only"]
        ),
    }
    return value_result, grad_result, timing


def count_values(values) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {str(key): int(count) for key, count in zip(unique, counts)}


def main() -> None:
    raw_inputs = json.loads(INPUT.read_text())
    cases = {case["id"]: case for case in raw_inputs["cases"]}
    uniform, dark = (cases[key] for key in CASE_IDS)
    t_uniform = np.asarray(uniform["times"], dtype=np.float64)
    t_dark = np.asarray(dark["times"], dtype=np.float64)
    if uniform["params"] != dark["params"] or not np.allclose(
        t_uniform, t_dark, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError("saved uniform and LD cases are not a matched pair")
    if OUT.exists():
        raise FileExistsError(f"preserving existing evidence directory: {OUT}")
    OUT.mkdir(parents=True)

    params = {
        "t0": uniform["params"]["t0"],
        "u0": uniform["params"]["u0"],
        "tE": uniform["params"]["tE"],
        "alpha": uniform["params"]["alpha"],
        "rho": uniform["params"]["rho"],
        "q": uniform["params"]["q"],
        "s": uniform["params"]["s"],
    }
    times = t_uniform
    results = {}
    raw = {"times": times}
    for profile, coefficient in (("uniform", None), ("linear_ld_0p5", 0.5)):
        kwargs = {
            "options": lcbinint.Options(
                coordinates="vbm",
                nbin="auto",
                finite_source_tol=TOL,
                finite_source_reltol=TOL,
            )
        }
        if coefficient is not None:
            kwargs["limb_darkening"] = lcbinint.LimbDarkening.linear(coefficient)
        curve = lcbinint.LightCurve(**kwargs)
        resolved_backend = curve.options.finite_source_backend
        if resolved_backend != "algebraic_boundary":
            raise RuntimeError(f"branch default resolved to {resolved_backend!r}")

        value, combined, timing = paired_times(curve, times, params)
        values = np.asarray(value.magnifications, dtype=np.float64).reshape(-1)
        combined_values = np.asarray(combined.magnifications, dtype=np.float64).reshape(-1)
        gradients = np.asarray(combined.gradients, dtype=np.float64)
        results[profile] = {
            "value_output_shape": list(values.shape),
            "full_jacobian_shape": list(gradients.shape),
            "gradient_parameters": list(combined.gradient_parameters),
            "gradient_valid_shape": list(np.asarray(combined.gradient_valid).shape),
            "gradient_valid_count": int(np.asarray(combined.gradient_valid).sum()),
            "gradient_valid_total": int(np.asarray(combined.gradient_valid).size),
            "gradient_eligible_count": int(
                np.count_nonzero(np.asarray(value.status) == lcbinint.LightCurveInfo.OK)
            ),
            "gradient_finite": bool(np.all(np.isfinite(gradients))),
            "value_only_finite": bool(np.all(np.isfinite(values))),
            "value_plus_gradient_finite": bool(np.all(np.isfinite(combined_values))),
            "value_consistency_max_abs": float(np.max(np.abs(values - combined_values))),
            "value_consistency_max_rel": float(np.max(
                np.abs(values - combined_values) / np.maximum(np.abs(values), 1.0e-300)
            )),
            "status_matches": bool(np.array_equal(value.status, combined.status)),
            "status_counts": count_values(value.status),
            "value_method_counts": count_values(
                value.diagnostics.get("finite_source_method_name", ["unknown"])
            ),
            "timing": timing,
            "resolved_backend": resolved_backend,
            "limb_darkening_coefficient": coefficient,
        }
        raw[f"value_only__{profile}"] = values
        raw[f"value_plus_gradient__{profile}"] = combined_values
        raw[f"jacobian__{profile}"] = gradients
        raw[f"status__{profile}"] = np.asarray(value.status)

    np.savez_compressed(OUT / "matched_lcbinint_ld_raw.npz", **raw)
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "repetitions": REPETITIONS,
        "tolerance": TOL,
        "input_case_ids": list(CASE_IDS),
        "same_parameters": True,
        "original_epoch_arrays_equal_bitwise": bool(np.array_equal(t_uniform, t_dark)),
        "original_epoch_arrays_max_abs_difference": float(np.max(np.abs(t_uniform - t_dark))),
        "benchmark_uses_one_shared_epoch_array": True,
        "n_epochs": int(times.size),
        "params": params,
        "route": "lcbinint.LightCurve branch default; verified finite_source_backend='algebraic_boundary'",
        "api": "same curve(times, params, with_gradient=False/True); algebraic-boundary gradients are evaluated only where the same call's value status is OK",
        "timing_policy": "15 interleaved pairs; alternate order; synchronize JAX leaves; first calls recorded separately",
        "results": results,
        "provenance": {
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
            "native_extension": str(Path(lcbinint._lcbinint.__file__).resolve()),
            "native_extension_sha256": sha256(Path(lcbinint._lcbinint.__file__).resolve()),
            "python_source_sha256": sha256(PYTHON_SOURCE / "__init__.py"),
            "jax_backend_source_sha256": sha256(PYTHON_SOURCE / "jax_backend.py"),
            "algebraic_boundary_source_sha256": sha256(PYTHON_SOURCE / "algebraic_boundary.py"),
            "input_file": str(INPUT),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (OUT / "inputs.json").write_text(json.dumps({
        "params": params,
        "times": times.tolist(),
        "limb_darkening_profiles": {"uniform": None, "linear_ld_0p5": 0.5},
        "source_cases": [uniform, dark],
    }, indent=2, allow_nan=False) + "\n")
    for profile, item in results.items():
        print("PROFILE", profile, item["timing"]["median_ms"],
              item["timing"]["value_plus_gradient_over_value"], flush=True)
    print(f"saved {OUT / 'metadata.json'}", flush=True)


if __name__ == "__main__":
    main()

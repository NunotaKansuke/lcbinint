#!/usr/bin/env python3
"""Audit the restored fused native gradient route through LightCurve."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import statistics
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

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = HERE / "corrected_value_gradient_12cases" / "inputs.json"
OUTPUT = HERE / "issue6_lcbinint_fused_api_20261017"
BUILD_PACKAGE = ROOT / "build-algebraic-boundary" / "python-api" / "lcbinint"
SOURCE_PACKAGE = ROOT / "python" / "lcbinint"
TOLERANCE = 1.0e-3
REPETITIONS = 15
CASE_IDS = ("large_source_uniform", "large_source_ld05_user")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def import_worktree_package():
    if not any(BUILD_PACKAGE.glob("_lcbinint*.so")):
        raise RuntimeError(f"missing build extension in {BUILD_PACKAGE}")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    # Prevent a PEP 660 finder from bypassing the explicit build package path.
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if "_editable" not in type(finder).__module__
    ]
    sys.path.insert(0, str(ROOT / "python"))
    spec = importlib.util.spec_from_file_location(
        "lcbinint",
        SOURCE_PACKAGE / "__init__.py",
        submodule_search_locations=[str(SOURCE_PACKAGE), str(BUILD_PACKAGE)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the in-tree lcbinint package")
    package = importlib.util.module_from_spec(spec)
    sys.modules["lcbinint"] = package
    spec.loader.exec_module(package)
    return package


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


def counts(values) -> dict[str, int]:
    unique, number = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {str(key): int(value) for key, value in zip(unique, number)}


def sha256_for_module(module) -> str:
    return digest(Path(module.__file__).resolve())


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"preserving existing evidence directory: {OUTPUT}")
    package = import_worktree_package()
    import jax

    jax.config.update("jax_enable_x64", True)
    fused_extension = importlib.import_module(
        "lcbinint._lcbinint_holonomic_fused"
    )
    cases = {
        case["id"]: case for case in json.loads(INPUT.read_text())["cases"]
    }
    uniform, dark = (cases[name] for name in CASE_IDS)
    if uniform["params"] != dark["params"]:
        raise ValueError("matched profiles do not share parameters")
    times = np.asarray(uniform["times"], dtype=np.float64)
    dark_times = np.asarray(dark["times"], dtype=np.float64)
    if not np.allclose(times, dark_times, rtol=0.0, atol=1.0e-15):
        raise ValueError("matched profiles do not share epochs")
    params = uniform["params"]

    results = {}
    raw = {"times": times}
    for case_id, profile, coefficient in (
        (CASE_IDS[0], "uniform", None),
        (CASE_IDS[1], "linear_ld_0p5", 0.5),
    ):
        options = package.Options(
            coordinates="vbm",
            nbin="auto",
            finite_source_tol=TOLERANCE,
            finite_source_reltol=TOLERANCE,
        )
        kwargs = {"options": options}
        if coefficient is not None:
            kwargs["limb_darkening"] = package.LimbDarkening.linear(coefficient)
        curve = package.LightCurve(**kwargs)
        value, value_first = call_timed(curve, times, params, False)
        combined, combined_first = call_timed(curve, times, params, True)

        samples = {"value_only": [], "value_plus_gradient": []}
        for repetition in range(REPETITIONS):
            order = (False, True) if repetition % 2 == 0 else (True, False)
            for with_gradient in order:
                _, elapsed = call_timed(curve, times, params, with_gradient)
                key = "value_plus_gradient" if with_gradient else "value_only"
                samples[key].append(elapsed)
        medians = {
            key: float(statistics.median(values))
            for key, values in samples.items()
        }

        value_magnification = np.asarray(value.magnifications, dtype=np.float64)
        combined_magnification = np.asarray(
            combined.magnifications, dtype=np.float64
        )
        gradients = np.asarray(combined.gradients, dtype=np.float64)
        gradient_valid = np.asarray(combined.gradient_valid, dtype=bool)
        value_delta = np.abs(value_magnification - combined_magnification)
        value_delta_rel = value_delta / np.maximum(
            np.abs(value_magnification), 1.0e-300
        )
        results[profile] = {
            "input_case_id": case_id,
            "n_epochs": int(times.size),
            "resolved_value_backend": curve.options.finite_source_backend,
            "gradient_route": "BinarySeededFusedGradient.evaluate",
            "gradient_parameters": list(combined.gradient_parameters),
            "first_call_ms": {
                "value_only": value_first,
                "value_plus_gradient": combined_first,
            },
            "samples_ms": samples,
            "median_ms": medians,
            "p95_ms": {
                key: float(np.percentile(values, 95))
                for key, values in samples.items()
            },
            "value_plus_gradient_over_value": (
                medians["value_plus_gradient"] / medians["value_only"]
            ),
            "value_only_finite": bool(np.all(np.isfinite(value_magnification))),
            "value_plus_gradient_finite": bool(
                np.all(np.isfinite(combined_magnification))
            ),
            "gradient_finite": bool(np.all(np.isfinite(gradients))),
            "gradient_valid_count": int(gradient_valid.sum()),
            "gradient_valid_total": int(gradient_valid.size),
            "value_converged_count": int(
                np.asarray(combined.value_converged, dtype=bool).sum()
            ),
            "support_certified_count": int(
                np.asarray(combined.support_certified, dtype=bool).sum()
            ),
            "value_status_counts": counts(value.status),
            "value_plus_gradient_status_counts": counts(combined.status),
            "status_matches": bool(
                np.array_equal(value.status, combined.status)
            ),
            "value_consistency_max_abs": float(value_delta.max()),
            "value_consistency_max_rel": float(value_delta_rel.max()),
        }
        raw[f"value_only__{profile}"] = value_magnification
        raw[f"value_plus_gradient__{profile}"] = combined_magnification
        raw[f"gradient__{profile}"] = gradients
        raw[f"gradient_valid__{profile}"] = gradient_valid
        raw[f"value_status__{profile}"] = np.asarray(value.status)
        raw[f"value_plus_gradient_status__{profile}"] = np.asarray(
            combined.status
        )
        raw[f"value_converged__{profile}"] = np.asarray(
            combined.value_converged
        )
        raw[f"support_certified__{profile}"] = np.asarray(
            combined.support_certified
        )

    OUTPUT.mkdir(parents=True)
    np.savez_compressed(OUTPUT / "matched_fused_route_raw.npz", **raw)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "repetitions": REPETITIONS,
        "tolerance": TOLERANCE,
        "parameters": params,
        "n_epochs": int(times.size),
        "same_shared_epoch_array_for_profiles": True,
        "value_only_route": "LightCurve algebraic-boundary value route",
        "value_plus_gradient_route": (
            "existing native seeded fused primal/Jacobian kernel through LightCurve"
        ),
        "results": results,
        "provenance": {
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--short")),
            "native_extension": str(Path(package._native.__file__).resolve()),
            "native_extension_sha256": sha256_for_module(package._native),
            "fused_extension": str(Path(fused_extension.__file__).resolve()),
            "fused_extension_sha256": sha256_for_module(fused_extension),
            "python_api_sha256": digest(SOURCE_PACKAGE / "__init__.py"),
            "fused_adapter_sha256": digest(SOURCE_PACKAGE / "_fused_gradient.py"),
            "fused_kernel_sha256": digest(
                ROOT / "src/lcbinint/magnification/holonomic/fused_gradient.cpp"
            ),
            "binding_sha256": digest(ROOT / "python/bind_holonomic_fused.cpp"),
            "cmake_sha256": digest(ROOT / "CMakeLists.txt"),
            "input_file": str(INPUT.resolve()),
            "script": str(Path(__file__).resolve()),
        },
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

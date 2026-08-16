#!/usr/bin/env python3
"""Benchmark synthetic light-curve API cases with and without warm-up.

The cases are deliberately synthetic and span close, resonant, and wide
binary geometries, several mass ratios, source radii, and impact parameters.
The benchmark compares the in-tree release build against VBMicrolensing and
also writes compact light-curve and timing figures.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "tests"
    / "diagnostics"
    / "results"
    / "recal2026"
    / "synthetic_lightcurve_benchmark_narrow_windows_20260816"
)
TOLERANCE = 1.0e-3
LD_COEFFICIENT = 0.5
DEFAULT_N_TIMES = 240
DEFAULT_REPEATS = 5


@dataclass(frozen=True)
class Case:
    name: str
    params: dict[str, float]
    t_min: float
    t_max: float
    n_times: int | None = None


BINARY_CASES = (
    Case(
        "resonant_high_mag",
        dict(s=0.95, q=1.0e-2, rho=5.0e-3, u0=-1.0e-3, alpha=0.5, t0=0.0, tE=1.0),
        -0.4,
        0.4,
    ),
    Case(
        "resonant_large_source",
        dict(s=0.95, q=1.0e-2, rho=2.0e-2, u0=-1.0e-2, alpha=0.5, t0=0.0, tE=1.0),
        -0.4,
        0.4,
    ),
    Case(
        "close_binary",
        dict(s=0.65, q=5.0e-3, rho=3.0e-3, u0=3.0e-2, alpha=1.1, t0=0.0, tE=1.0),
        -0.5,
        0.5,
    ),
    Case(
        "close_secondary_caustics",
        dict(s=0.65, q=2.0e-2, rho=4.0e-3, u0=0.214, alpha=3.75, t0=0.0, tE=1.0),
        -1.2,
        1.2,
        400,
    ),
    Case(
        "wide_planet",
        dict(
            s=2.5,
            q=1.0e-2,
            rho=2.0e-3,
            # A slightly oblique track crosses the planetary caustic while
            # the expanded time window also shows the host-side peak.  In
            # VBM coordinates t0 is the closest approach to the host/origin;
            # the planetary-caustic crossing therefore occurs near t=0.
            u0=0.294,
            alpha=3.0,
            t0=-2.07,
            tE=1.0,
        ),
        -2.8,
        0.8,
        600,
    ),
    Case(
        "small_q",
        dict(s=1.1, q=1.0e-4, rho=1.0e-3, u0=2.0e-2, alpha=0.3, t0=0.0, tE=1.0),
        -2.0,
        2.0,
    ),
    Case(
        "high_q",
        dict(s=1.0, q=1.0e-1, rho=1.0e-2, u0=5.0e-2, alpha=1.3, t0=0.0, tE=1.0),
        -1.0,
        1.0,
    ),
    Case(
        "cusp_small_source",
        dict(s=0.9, q=2.0e-2, rho=2.0e-3, u0=5.0e-3, alpha=1.2, t0=0.0, tE=1.0),
        -0.5,
        0.5,
    ),
)

TRIPLE_CASES = (
    Case(
        "triple_default",
        dict(
            s=1.2,
            q=1.0e-2,
            q2=1.0e-3,
            sep2=1.0,
            ang=0.5,
            u0=5.0e-2,
            alpha=0.0,
            rho=5.0e-3,
            t0=0.0,
            tE=1.0,
        ),
        -0.8,
        0.8,
    ),
    Case(
        "triple_close",
        dict(
            s=0.9,
            q=2.0e-2,
            q2=5.0e-3,
            sep2=0.7,
            ang=1.1,
            u0=2.0e-2,
            alpha=0.8,
            rho=3.0e-3,
            t0=0.0,
            tE=1.0,
        ),
        -1.0,
        1.0,
    ),
    Case(
        "triple_wide",
        dict(
            s=1.8,
            q=5.0e-3,
            q2=1.0e-3,
            sep2=1.3,
            ang=0.4,
            u0=8.0e-2,
            alpha=1.2,
            rho=4.0e-3,
            t0=0.0,
            tE=1.0,
        ),
        -1.5,
        1.5,
    ),
)


def load_lcbinint():
    """Load the current in-tree build even if an editable install is active."""

    build_path = ROOT / "build"
    package = build_path / "lcbinint"
    init_path = package / "__init__.py"
    extensions = sorted(package.glob("_lcbinint*.so"))
    if not init_path.is_file() or not extensions:
        return importlib.import_module("lcbinint")

    sys.modules.pop("lcbinint._lcbinint", None)
    sys.modules.pop("lcbinint", None)
    root_spec = importlib.util.spec_from_file_location(
        "lcbinint",
        init_path,
        submodule_search_locations=[str(package)],
    )
    if root_spec is None or root_spec.loader is None:
        raise RuntimeError(f"Could not load lcbinint from {init_path}")
    module = importlib.util.module_from_spec(root_spec)
    sys.modules["lcbinint"] = module
    extension_spec = importlib.util.spec_from_file_location(
        "lcbinint._lcbinint", extensions[0]
    )
    if extension_spec is None or extension_spec.loader is None:
        raise RuntimeError(f"Could not load lcbinint extension from {extensions[0]}")
    extension = importlib.util.module_from_spec(extension_spec)
    sys.modules["lcbinint._lcbinint"] = extension
    extension_spec.loader.exec_module(extension)
    root_spec.loader.exec_module(module)
    return module


def options(lcbinint: Any) -> Any:
    return lcbinint.Options(
        coordinates="vbm",
        nbin="auto",
        tol=TOLERANCE,
        reltol=TOLERANCE,
    )


def make_curve(lcbinint: Any, lens: str, ld: bool) -> Any:
    return lcbinint.LightCurve(
        lens=lens,
        options=options(lcbinint),
        limb_darkening=(
            lcbinint.LimbDarkening.linear(LD_COEFFICIENT)
            if ld
            else lcbinint.LimbDarkening.none()
        ),
    )


def timed_series(
    evaluate: Callable[[], np.ndarray], repeats: int
) -> tuple[np.ndarray, float, np.ndarray, list[float]]:
    """Return first result, first-call seconds, final result, steady samples."""

    started = time.perf_counter()
    first = np.asarray(evaluate(), dtype=float)
    cold_seconds = time.perf_counter() - started
    steady_values = None
    steady_seconds: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        steady_values = np.asarray(evaluate(), dtype=float)
        steady_seconds.append(time.perf_counter() - started)
    if steady_values is None:
        steady_values = first
    return first, cold_seconds, steady_values, steady_seconds


def timing_record(
    cold_seconds: float,
    steady_seconds: list[float],
    n_times: int,
    extra_seconds: float = 0.0,
) -> dict[str, Any]:
    median_steady = float(statistics.median(steady_seconds))
    total_seconds = extra_seconds + cold_seconds + sum(steady_seconds)
    return {
        "cold_ms": 1.0e3 * cold_seconds,
        "steady_median_ms": 1.0e3 * median_steady,
        "steady_samples_ms": [1.0e3 * value for value in steady_seconds],
        "steady_median_ms_per_epoch": 1.0e3 * median_steady / n_times,
        "total_ms_for_measured_calls": 1.0e3 * total_seconds,
        "extra_ms": 1.0e3 * extra_seconds,
    }


def accuracy(reference: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(reference) & np.isfinite(values)
    if not np.any(finite):
        return {
            "finite_epochs": 0,
            "max_abs": math.nan,
            "max_relative": math.nan,
            "p99_relative": math.nan,
            "median_relative": math.nan,
            "rms_relative": math.nan,
        }
    difference = np.abs(values[finite] - reference[finite])
    relative = difference / np.maximum(np.abs(reference[finite]), 1.0e-12)
    return {
        "finite_epochs": int(np.count_nonzero(finite)),
        "max_abs": float(np.max(difference)),
        "max_relative": float(np.max(relative)),
        "p99_relative": float(np.percentile(relative, 99.0)),
        "median_relative": float(np.median(relative)),
        "rms_relative": float(np.sqrt(np.mean(relative * relative))),
    }


def method_summary(curve: Any, times: np.ndarray, params: dict[str, float]) -> dict[str, Any]:
    info = curve.info(times, params)
    method_names = {
        0: "point_source",
        1: "hexadecapole",
        2: "inverse_ray_cartesian",
        3: "inverse_ray_polar",
        4: "inverse_ray_spine",
        5: "source_plane_quadrature",
    }
    methods = np.asarray(info.finite_source_methods, dtype=int)
    converged = np.asarray(info.finite_source_converged, dtype=bool)
    return {
        "methods": dict(
            Counter(method_names.get(int(value), f"method_{int(value)}") for value in methods)
        ),
        "converged_epochs": int(np.count_nonzero(converged)),
        "total_epochs": int(converged.size),
    }


def warmup_summary(report: Any) -> dict[str, Any]:
    return {
        "supported": True,
        "elapsed_ms": 1.0e3 * float(report.elapsed_seconds),
        "all_calibrated": bool(report.all_calibrated),
        "methods": dict(Counter(report.methods)),
        "statuses": dict(Counter(report.statuses)),
        "resolution_min": int(np.min(report.resolutions)),
        "resolution_max": int(np.max(report.resolutions)),
    }


def vbm_parameters(lens: str, params: dict[str, float]) -> list[float]:
    if lens == "binary":
        return [
            math.log(params["s"]),
            math.log(params["q"]),
            params["u0"],
            params["alpha"],
            math.log(params["rho"]),
            math.log(params["tE"]),
            params["t0"],
        ]
    return [
        math.log(params["s"]),
        math.log(params["q"]),
        params["u0"],
        params["alpha"],
        math.log(params["rho"]),
        math.log(params["tE"]),
        params["t0"],
        math.log(params["sep2"]),
        math.log(params["q2"]),
        params["ang"],
    ]


def measure_vbm(
    lens: str, times: np.ndarray, params: dict[str, float], ld: bool, repeats: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import VBMicrolensing

    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = TOLERANCE
    engine.RelTol = TOLERANCE
    engine.a1 = LD_COEFFICIENT if ld else 0.0
    engine.a2 = 0.0
    if lens == "triple" and hasattr(engine, "SetMethod") and hasattr(engine, "Multipoly"):
        engine.SetMethod(engine.Multipoly)
    parameter_array = vbm_parameters(lens, params)
    times_list = times.tolist()

    def evaluate() -> np.ndarray:
        if lens == "binary":
            result = engine.BinaryLightCurve(parameter_array, times_list)
        else:
            result = engine.TripleLightCurve(parameter_array, times_list)
        return np.asarray(result[0], dtype=float)

    first, cold, steady, samples = timed_series(evaluate, repeats)
    return timing_record(cold, samples, times.size), first, steady


def benchmark_binary(
    lcbinint: Any, case: Case, ld: bool, n_times: int, repeats: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_n_times = int(case.n_times or n_times)
    times = np.linspace(case.t_min, case.t_max, case_n_times)
    params = dict(case.params)

    no_curve = make_curve(lcbinint, "binary", ld)
    no_first, no_cold, no_steady, no_samples = timed_series(
        lambda: no_curve(times, params), repeats
    )
    no_timing = timing_record(no_cold, no_samples, case_n_times)
    no_methods = method_summary(no_curve, times, params)

    warm_curve = make_curve(lcbinint, "binary", ld)
    warm_started = time.perf_counter()
    warm_report = None
    warm_error = None
    try:
        warm_report = warm_curve.warmup(times, params, grid_timing_repeats=1)
    except Exception as error:  # retain the no-warm-up result for diagnosis
        warm_error = f"{type(error).__name__}: {error}"
    warmup_seconds = time.perf_counter() - warm_started

    warm_first = warm_steady = None
    warm_timing = None
    warm_accuracy = None
    warm_no_accuracy = None
    plot_warm = None
    if warm_report is not None:
        warm_first, warm_cold, warm_steady, warm_samples = timed_series(
            lambda: warm_curve(times, params), repeats
        )
        warm_timing = timing_record(
            warm_cold,
            warm_samples,
            case_n_times,
            extra_seconds=warmup_seconds,
        )
        warm_accuracy = accuracy(no_steady, warm_steady)
        warm_no_accuracy = warmup_summary(warm_report)

    vbm_timing, vbm_first, vbm_steady = measure_vbm(
        "binary", times, params, ld, repeats
    )
    record = {
        "lens": "binary",
        "case": case.name,
        "profile": "C1_linear_ld" if ld else "C0_uniform",
        "n_times": case_n_times,
        "repeats_after_first": int(repeats),
        "parameters": params,
        "time_range": [float(case.t_min), float(case.t_max)],
        "no_warmup": no_timing,
        "warmup": warm_timing
        if warm_timing is not None
        else {"supported": True, "error": warm_error},
        "vbm": vbm_timing,
        "routes": no_methods,
        "accuracy": {
            "no_warmup_vs_vbm": accuracy(vbm_steady, no_steady),
            "warmup_vs_vbm": (
                accuracy(vbm_steady, warm_steady)
                if warm_steady is not None
                else None
            ),
            "warmup_vs_no_warmup": warm_accuracy,
        },
        "warmup_plan": warm_no_accuracy,
    }
    plot = {
        "times": times,
        "no": no_steady,
        "warm": plot_warm if plot_warm is not None else warm_steady,
        "vbm": vbm_steady,
    }
    return record, plot


def benchmark_triple(
    lcbinint: Any, case: Case, ld: bool, n_times: int, repeats: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_n_times = int(case.n_times or n_times)
    times = np.linspace(case.t_min, case.t_max, case_n_times)
    params = dict(case.params)
    curve = make_curve(lcbinint, "triple", ld)
    first, cold, steady, samples = timed_series(
        lambda: curve(times, params), repeats
    )
    vbm_timing, vbm_first, vbm_steady = measure_vbm(
        "triple", times, params, ld, repeats
    )
    record = {
        "lens": "triple",
        "case": case.name,
        "profile": "C1_linear_ld" if ld else "C0_uniform",
        "n_times": case_n_times,
        "repeats_after_first": int(repeats),
        "parameters": params,
        "time_range": [float(case.t_min), float(case.t_max)],
        "no_warmup": timing_record(cold, samples, case_n_times),
        "warmup": {
            "supported": False,
            "reason": "LightCurve.warmup currently supports binary lenses only",
        },
        "vbm": vbm_timing,
        "routes": method_summary(curve, times, params),
        "accuracy": {"no_warmup_vs_vbm": accuracy(vbm_steady, steady)},
    }
    return record, {"times": times, "no": steady, "warm": None, "vbm": vbm_steady}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_report(
    path: Path,
    records: list[dict[str, Any]],
    n_times: int,
    repeats: int,
    triple_warmup_reason: str,
) -> None:
    epoch_overrides = sorted(
        {
            (record["case"], int(record["n_times"]))
            for record in records
            if int(record["n_times"]) != n_times
        }
    )
    epoch_note = f"{n_times} default epochs/case"
    if epoch_overrides:
        overrides = ", ".join(f"{case}={count}" for case, count in epoch_overrides)
        epoch_note += f"; overrides: {overrides}"
    lines = [
        "# Synthetic light-curve API benchmark",
        "",
        "合成パラメータのみで測定した `lcbinint` / VBMicrolensing 比較。",
        "",
        f"- in-tree Release build, OpenMP thread setting is taken from the process environment",
        f"- tolerance: `{TOLERANCE:g}` (both engines), {epoch_note}",
        f"- each row: first call + `{repeats}` steady calls; steady is the median of the latter",
        f"- binary warm-up setup time is included only in `warmup.extra_ms` and the measured-call total",
        f"- triple warm-up probe: `{triple_warmup_reason}`",
        "",
        "## Binary",
        "",
        "| case | profile | lcbinint no warm [ms/epoch] | lcbinint warm [ms/epoch] | warm setup [ms] | VBM [ms/epoch] | VBM/lc no-warm | VBM/lc warm | max rel. err no/warm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        if record["lens"] != "binary":
            continue
        no = record["no_warmup"]
        warm = record["warmup"]
        vbm = record["vbm"]
        err_no = record["accuracy"]["no_warmup_vs_vbm"]["max_relative"]
        err_warm = (
            None
            if record["accuracy"].get("warmup_vs_vbm") is None
            else record["accuracy"]["warmup_vs_vbm"]["max_relative"]
        )
        lines.append(
            "| `{case}` | {profile} | {no} | {warm} | {setup} | {vbm} | {ratio_no} | {ratio_warm} | {err_no} / {err_warm} |".format(
                case=record["case"],
                profile=record["profile"],
                no=fmt(no["steady_median_ms_per_epoch"]),
                warm=fmt(
                    warm.get("steady_median_ms_per_epoch")
                    if warm.get("supported", True)
                    else None
                ),
                setup=fmt(warm.get("extra_ms")),
                vbm=fmt(vbm["steady_median_ms_per_epoch"]),
                ratio_no=fmt(
                    vbm["steady_median_ms"] / no["steady_median_ms"]
                ),
                ratio_warm=fmt(
                    vbm["steady_median_ms"] / warm["steady_median_ms"]
                    if warm.get("steady_median_ms") is not None
                    else None
                ),
                err_no=fmt(err_no, 3),
                err_warm=fmt(err_warm, 3),
            )
        )

    lines.extend([
        "",
        "## Triple",
        "",
        "| case | profile | lcbinint [ms/epoch] | VBM [ms/epoch] | VBM/lc | max rel. err |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for record in records:
        if record["lens"] != "triple":
            continue
        no = record["no_warmup"]
        vbm = record["vbm"]
        err = record["accuracy"]["no_warmup_vs_vbm"]["max_relative"]
        lines.append(
            f"| `{record['case']}` | {record['profile']} | "
            f"{fmt(no['steady_median_ms_per_epoch'])} | "
            f"{fmt(vbm['steady_median_ms_per_epoch'])} | "
            f"{fmt(vbm['steady_median_ms'] / no['steady_median_ms'])} | "
            f"{fmt(err, 3)} |"
        )

    lines.extend(["", "## Route counts", ""])
    for record in records:
        route_text = ", ".join(
            f"{key}={value}" for key, value in record["routes"]["methods"].items()
        )
        lines.append(f"- `{record['lens']}/{record['case']}/{record['profile']}`: {route_text}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_curves(
    path: Path,
    plot_data: dict[tuple[str, str], dict[str, np.ndarray]],
    cases: tuple[Case, ...],
    lens: str,
) -> None:
    columns = (False, True)
    figure, axes = plt.subplots(
        len(cases), 2, figsize=(12.0, max(6.0, 2.35 * len(cases))), squeeze=False
    )
    for row_index, case in enumerate(cases):
        for column, ld in enumerate(columns):
            profile = "C1_linear_ld" if ld else "C0_uniform"
            axis = axes[row_index, column]
            data = plot_data[(case.name, profile)]
            axis.plot(data["times"], data["vbm"], color="tab:orange", lw=1.2, label="VBMicrolensing")
            axis.plot(data["times"], data["no"], color="0.25", lw=0.85, ls=":", label="lcbinint no warm")
            if data["warm"] is not None:
                axis.plot(data["times"], data["warm"], color="tab:blue", lw=1.0, ls="--", label="lcbinint warm")
            axis.set_title(
                f"{case.name} | {'C1 LD' if ld else 'C0 uniform'}\n"
                f"s={case.params['s']:.3g}, q={case.params['q']:.3g}, "
                f"rho={case.params['rho']:.3g}, u0={case.params['u0']:.3g}"
            )
            axis.set_ylabel("magnification")
            axis.grid(alpha=0.18)
            if row_index == 0 and column == 0:
                axis.legend(fontsize=8, loc="best")
            if row_index == len(cases) - 1:
                axis.set_xlabel("time")
    figure.suptitle(f"Synthetic {lens} light curves: lcbinint vs VBMicrolensing", y=0.999)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_speed(path: Path, records: list[dict[str, Any]], lens: str) -> None:
    selected = [record for record in records if record["lens"] == lens]
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 5.5), squeeze=False)
    axes_flat = axes[0]
    profiles = ("C0_uniform", "C1_linear_ld") if lens == "binary" else ("C0_uniform", "C1_linear_ld")
    for index, profile in enumerate(profiles):
        axis = axes_flat[index]
        rows = [record for record in selected if record["profile"] == profile]
        positions = np.arange(len(rows), dtype=float)
        width = 0.25
        no_values = [record["no_warmup"]["steady_median_ms_per_epoch"] for record in rows]
        vbm_values = [record["vbm"]["steady_median_ms_per_epoch"] for record in rows]
        if lens == "binary":
            warm_values = [record["warmup"].get("steady_median_ms_per_epoch", np.nan) for record in rows]
            axis.bar(positions - width, no_values, width, label="lcbinint no warm", color="0.35")
            axis.bar(positions, warm_values, width, label="lcbinint warm", color="tab:blue")
            axis.bar(positions + width, vbm_values, width, label="VBMicrolensing", color="tab:orange")
        else:
            axis.bar(positions - width / 2.0, no_values, width, label="lcbinint", color="0.35")
            axis.bar(positions + width / 2.0, vbm_values, width, label="VBMicrolensing", color="tab:orange")
        axis.set_xticks(positions, [record["case"] for record in rows], rotation=35, ha="right")
        axis.set_yscale("log")
        axis.set_ylabel("steady ms / epoch (log scale)")
        axis.set_title(profile.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(f"Synthetic {lens} speed comparison", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def probe_triple_warmup(lcbinint: Any) -> str:
    case = TRIPLE_CASES[0]
    curve = make_curve(lcbinint, "triple", False)
    times = np.linspace(case.t_min, case.t_max, 4)
    try:
        curve.warmup(times, case.params)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return "unexpectedly succeeded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-times", type=int, default=DEFAULT_N_TIMES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--skip-triple", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_times < 8 or args.repeats < 1:
        raise ValueError("--n-times must be >= 8 and --repeats must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lcbinint = load_lcbinint()
    records: list[dict[str, Any]] = []
    plot_data: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for case in BINARY_CASES:
        for ld in (False, True):
            profile = "C1_linear_ld" if ld else "C0_uniform"
            print(f"binary/{case.name}/{profile}", flush=True)
            record, data = benchmark_binary(lcbinint, case, ld, args.n_times, args.repeats)
            records.append(record)
            plot_data[(case.name, profile)] = data

    triple_reason = probe_triple_warmup(lcbinint)
    if not args.skip_triple:
        for case in TRIPLE_CASES:
            for ld in (False, True):
                profile = "C1_linear_ld" if ld else "C0_uniform"
                print(f"triple/{case.name}/{profile}", flush=True)
                record, data = benchmark_triple(lcbinint, case, ld, args.n_times, args.repeats)
                records.append(record)
                plot_data[(case.name, profile)] = data

    payload = {
        "benchmark": "synthetic_lightcurve_warmup",
        "tolerance": TOLERANCE,
        "limb_darkening_coefficient": LD_COEFFICIENT,
        "n_times": args.n_times,
        "repeats_after_first": args.repeats,
        "triple_warmup_probe": triple_reason,
        "records": records,
    }
    json_path = args.output_dir / "benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_report(args.output_dir / "REPORT.md", records, args.n_times, args.repeats, triple_reason)
    plot_curves(args.output_dir / "binary_lightcurves.png", plot_data, BINARY_CASES, "binary")
    plot_speed(args.output_dir / "binary_speed.png", records, "binary")
    if not args.skip_triple:
        plot_curves(args.output_dir / "triple_lightcurves.png", plot_data, TRIPLE_CASES, "triple")
        plot_speed(args.output_dir / "triple_speed.png", records, "triple")
    print(f"report: {args.output_dir / 'REPORT.md'}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()

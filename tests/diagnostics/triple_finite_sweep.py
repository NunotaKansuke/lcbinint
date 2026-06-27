from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    s: float
    q: float
    q2: float
    sep2: float
    ang: float
    rho: float
    x_min: float
    x_max: float
    y: float = 0.0
    points: int = 13


CASES = [
    Case("resonant_tiny_source", 1.0, 1.0e-3, 1.0e-4, 0.5, 1.2, 1.0e-3, -0.001, 0.001, 0.0, 25),
    Case("resonant_small_source", 1.0, 1.0e-3, 1.0e-4, 0.5, 1.2, 3.0e-3, -0.006, 0.006, 0.0, 25),
    Case("resonant_hex_band", 1.0, 1.0e-3, 1.0e-4, 0.5, 1.2, 5.0e-3, -0.24, -0.16, 0.0, 17),
    Case("moderate_inner_pair", 0.8, 0.03, 0.02, 0.35, -0.7, 2.0e-3, 0.30, 0.40, -0.22, 17),
    Case("wide_primary", 1.4, 0.2, 0.05, 0.7, 2.1, 3.0e-3, -0.52, -0.38, 0.18, 17),
]


def load_lcbinint(extension: Path | None):
    if extension is None:
        import lcbinint as module

        return module
    spec = importlib.util.spec_from_file_location("lcbinint", extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lcbinint extension from {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lcbinint"] = module
    spec.loader.exec_module(module)
    return module


def parse_bins(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def timed_best(func, repeat: int):
    best_value = None
    best_seconds = math.inf
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        value = func()
        seconds = time.perf_counter() - start
        if seconds < best_seconds:
            best_value = value
            best_seconds = seconds
    return best_value, best_seconds


def evaluate_case(lcbinint, case: Case, source_bins: int, caustic_bins: int, repeat: int):
    options = lcbinint.Options(source_bins=source_bins, caustic_bins=caustic_bins)
    curve = lcbinint.LightCurve(lens="triple_lens", options=options)
    times = np.linspace(case.x_min, case.x_max, case.points)
    params = {
        "t0": 0.0,
        "tE": 1.0,
        "u0": case.y,
        "alpha": 0.0,
        "s": case.s,
        "q": case.q,
        "q2": case.q2,
        "sep2": case.sep2,
        "ang": case.ang,
        "rho": case.rho,
    }
    info, seconds = timed_best(lambda: curve.info(times, params), repeat)
    values = np.asarray(info.magnifications, dtype=float)
    return {
        "values": values,
        "seconds": seconds,
        "ms_per_point": 1000.0 * seconds / max(1, len(times)),
        "methods": ",".join(sorted(set(info.finite_source_method_names))),
        "median_estimate": float(np.median(np.asarray(info.finite_source_error_estimates, dtype=float))),
        "max_estimate": float(np.max(np.asarray(info.finite_source_error_estimates, dtype=float))),
        "points": len(times),
    }


def summarize_by_case(rows: list[dict]) -> None:
    print("case summary")
    for case_name in sorted({row["case"] for row in rows}):
        subset = [row for row in rows if row["case"] == case_name]
        finite = [row for row in subset if row["source_bins"] != "reference"]
        if not finite:
            continue
        best = min(finite, key=lambda row: row["max_rel_error"])
        fastest = min(finite, key=lambda row: row["ms_per_point"])
        print(
            f"{case_name:24s} best_bins={best['source_bins']:>3} "
            f"best_max_rel={best['max_rel_error']:.3e} "
            f"fastest_bins={fastest['source_bins']:>3} "
            f"fastest_ms={fastest['ms_per_point']:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, default=None)
    parser.add_argument("--source-bins", default="8,12,16,24,32")
    parser.add_argument("--reference-bins", type=int, default=64)
    parser.add_argument("--caustic-bins", type=int, default=1400)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    lcbinint = load_lcbinint(args.extension)
    source_bins_values = parse_bins(args.source_bins)
    rows: list[dict] = []
    for case in CASES:
        reference = evaluate_case(
            lcbinint, case, args.reference_bins, args.caustic_bins, args.repeat)
        reference_values = reference["values"]
        rows.append({
            "case": case.name,
            "source_bins": "reference",
            "points": reference["points"],
            "rho": case.rho,
            "methods": reference["methods"],
            "ms_per_point": reference["ms_per_point"],
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "rms_rel_error": 0.0,
            "median_estimate": reference["median_estimate"],
            "max_estimate": reference["max_estimate"],
        })
        for source_bins in source_bins_values:
            actual = evaluate_case(lcbinint, case, source_bins, args.caustic_bins, args.repeat)
            values = actual["values"]
            abs_error = np.abs(values - reference_values)
            rel_error = abs_error / np.maximum(np.abs(reference_values), 1.0e-12)
            rows.append({
                "case": case.name,
                "source_bins": source_bins,
                "points": actual["points"],
                "rho": case.rho,
                "methods": actual["methods"],
                "ms_per_point": actual["ms_per_point"],
                "max_abs_error": float(np.max(abs_error)),
                "max_rel_error": float(np.max(rel_error)),
                "rms_rel_error": float(np.sqrt(np.mean(rel_error * rel_error))),
                "median_estimate": actual["median_estimate"],
                "max_estimate": actual["max_estimate"],
            })

    fieldnames = [
        "case",
        "source_bins",
        "points",
        "rho",
        "methods",
        "ms_per_point",
        "max_abs_error",
        "max_rel_error",
        "rms_rel_error",
        "median_estimate",
        "max_estimate",
    ]
    if args.csv is not None:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[name]) for name in fieldnames))
    print()
    summarize_by_case(rows)
    finite_rows = [row for row in rows if row["source_bins"] != "reference"]
    print()
    print(
        "overall "
        f"median_ms={statistics.median(row['ms_per_point'] for row in finite_rows):.3f} "
        f"median_max_rel={statistics.median(row['max_rel_error'] for row in finite_rows):.3e} "
        f"worst_max_rel={max(row['max_rel_error'] for row in finite_rows):.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

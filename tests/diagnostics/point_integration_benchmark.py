from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import math
import os
import statistics
import sys
import time

import numpy as np

import VBMicrolensing


def load_lcbinint_module():
    extension_path = os.environ.get("LCBININT_EXTENSION")
    if not extension_path:
        import lcbinint as module
        return module
    spec = importlib.util.spec_from_file_location("lcbinint", extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lcbinint extension from {extension_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lcbinint"] = module
    spec.loader.exec_module(module)
    return module


lcbinint = load_lcbinint_module()

from adaptive_source_bins_sweep import CASES, Case, random_cases


@dataclasses.dataclass(frozen=True)
class PointCase:
    case: Case
    time: float
    reference: float


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def timed_best(func, repeat: int) -> tuple[object, float]:
    best_value = None
    best_seconds = float("inf")
    for _ in range(max(repeat, 1)):
        start = time.perf_counter()
        value = func()
        seconds = time.perf_counter() - start
        if seconds < best_seconds:
            best_value = value
            best_seconds = seconds
    return best_value, best_seconds


def geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return float("nan")
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def vbm_binary_lightcurve(case: Case, times: np.ndarray, tol: float) -> np.ndarray:
    vbb = VBMicrolensing.VBMicrolensing()
    vbb.Tol = tol
    vbb.RelTol = 0.0
    if case.limb_darkening_c != 0.0:
        vbb.a1 = case.limb_darkening_c
    params = [
        math.log(case.separation),
        math.log(case.mass_ratio),
        case.u0,
        case.alpha,
        math.log(case.rho),
        0.0,
        0.0,
    ]
    return np.asarray(vbb.BinaryLightCurve(params, times.tolist())[0], dtype=float)


def vbm_point(case: Case, time_value: float, tol: float) -> float:
    return float(vbm_binary_lightcurve(case, np.asarray([time_value]), tol)[0])


def lcbinint_point(
    case: Case,
    time_value: float,
    source_bins: int,
    max_source_bins: int,
    reltol: float,
):
    options = lcbinint.Options(
        source_bins=source_bins,
        adaptive_source_bins=1,
        max_source_bins=max_source_bins,
        reltol=reltol,    )
    return lcbinint.light_curve_info(
        [time_value],
        u0=case.u0,
        alpha=case.alpha,
        s=case.separation,
        q=case.mass_ratio,
        rho=case.rho,
        limb_darkening=lcbinint.LimbDarkening(c=case.limb_darkening_c),
        options=options,
    )


def selected_points(case: Case, points_per_case: int, reference_tol: float) -> list[PointCase]:
    scan_count = max(121, points_per_case * 24)
    times = np.linspace(case.t_min, case.t_max, scan_count)
    reference = vbm_binary_lightcurve(case, times, reference_tol)
    selected_indices = {int(np.argmax(reference)), scan_count // 2}
    if points_per_case > 2:
        quantile_positions = np.linspace(0, scan_count - 1, points_per_case)
        selected_indices.update(int(round(position)) for position in quantile_positions)
    while len(selected_indices) < points_per_case:
        selected_indices.add(int((len(selected_indices) + 1) * scan_count / (points_per_case + 1)))

    rows = [
        PointCase(case, float(times[index]), float(reference[index]))
        for index in sorted(selected_indices)
    ]
    rows.sort(key=lambda point: point.reference, reverse=True)
    return rows[:points_per_case]


def summarize(rows: list[dict], key: str) -> list[dict]:
    groups: dict[object, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    summaries = []
    for value, subset in sorted(groups.items(), key=lambda item: item[0]):
        ratios = [row["lc_ms"] / row["vbb_ms"] for row in subset if row["vbb_ms"] > 0.0]
        summaries.append({
            key: value,
            "points": len(subset),
            "lc_ms_median": statistics.median(row["lc_ms"] for row in subset),
            "lc_ms_geo": geomean([row["lc_ms"] for row in subset]),
            "ratio_median": statistics.median(ratios) if ratios else float("nan"),
            "ratio_geo": geomean(ratios),
            "max_rel_median": statistics.median(row["rel_error"] for row in subset),
            "max_rel_p90": float(np.quantile([row["rel_error"] for row in subset], 0.90)),
            "refinement_median": statistics.median(row["refinement_level"] for row in subset),
            "max_refinement": max(row["refinement_level"] for row in subset),
            "unconverged": sum(0 if row["converged"] else 1 for row in subset),
            "accepted_bad": sum(row["accepted_bad"] for row in subset),
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bins", default="20,32,40,50,64,80")
    parser.add_argument("--reltols", default="3e-3,1e-3,3e-4")
    parser.add_argument("--max-bins", type=int, default=400)
    parser.add_argument("--points-per-case", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--random", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--reference-tol", type=float, default=1.0e-5)
    parser.add_argument("--vbm-tol", type=float, default=1.0e-3)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    source_bins_values = parse_csv_ints(args.source_bins)
    reltol_values = parse_csv_floats(args.reltols)
    cases = list(CASES) + random_cases(args.random, args.seed, args.points_per_case * 24)
    points: list[PointCase] = []
    for case in cases:
        points.extend(selected_points(case, args.points_per_case, args.reference_tol))

    rows: list[dict] = []
    for point_index, point in enumerate(points, start=1):
        case = point.case
        _, vbb_seconds = timed_best(
            lambda: vbm_point(case, point.time, args.vbm_tol),
            args.repeat,
        )
        vbb_ms = 1000.0 * vbb_seconds
        for reltol in reltol_values:
            for source_bins in source_bins_values:
                curve, lc_seconds = timed_best(
                    lambda: lcbinint_point(
                        case,
                        point.time,
                        source_bins,
                        args.max_bins,
                        reltol,
                    ),
                    args.repeat,
                )
                actual = float(curve.magnifications[0])
                estimate = float(curve.finite_source_error_estimates[0])
                level = int(curve.finite_source_refinement_levels[0])
                converged = bool(curve.finite_source_converged[0])
                abs_error = abs(actual - point.reference)
                target = reltol * max(abs(actual), 1.0)
                rel_error = abs(actual / point.reference - 1.0)
                rows.append({
                    "case": case.name,
                    "time": point.time,
                    "source_bins": source_bins,
                    "reltol": reltol,
                    "rho": case.rho,
                    "ld": case.limb_darkening_c != 0.0,
                    "reference": point.reference,
                    "actual": actual,
                    "max_mag": point.reference,
                    "vbb_ms": vbb_ms,
                    "lc_ms": 1000.0 * lc_seconds,
                    "rel_error": rel_error,
                    "abs_error": abs_error,
                    "estimate": estimate,
                    "target": target,
                    "refinement_level": level,
                    "converged": converged,
                    "accepted_bad": int(converged and abs_error > 1.05 * target),
                })
        if point_index % 16 == 0:
            print(f"processed {point_index}/{len(points)} points")

    print(f"lcbinint: {lcbinint.__file__}")
    print(
        "point integration benchmark "
        f"cases={len(cases)} points={len(points)} repeat={args.repeat} "
        f"max_bins={args.max_bins} vbm_tol={args.vbm_tol:g} reference_tol={args.reference_tol:g}"
    )

    print("\nby source_bins")
    print("bins points lc_ms_med lc_ms_geo lc/vbb_med lc/vbb_geo rel_med rel_p90 refined_med maxlev unconv bad")
    for row in summarize(rows, "source_bins"):
        print(
            f"{row['source_bins']:4d} {row['points']:6d} "
            f"{row['lc_ms_median']:9.4f} {row['lc_ms_geo']:9.4f} "
            f"{row['ratio_median']:10.3f} {row['ratio_geo']:10.3f} "
            f"{row['max_rel_median']:9.2e} {row['max_rel_p90']:9.2e} "
            f"{row['refinement_median']:11.1f} {row['max_refinement']:6d} "
            f"{row['unconverged']:6d} {row['accepted_bad']:4d}"
        )

    print("\nby reltol")
    print("reltol points lc_ms_med lc/vbb_med rel_med rel_p90 refined_med maxlev unconv bad")
    for row in summarize(rows, "reltol"):
        print(
            f"{row['reltol']:7.0e} {row['points']:6d} "
            f"{row['lc_ms_median']:9.4f} {row['ratio_median']:10.3f} "
            f"{row['max_rel_median']:9.2e} {row['max_rel_p90']:9.2e} "
            f"{row['refinement_median']:11.1f} {row['max_refinement']:6d} "
            f"{row['unconverged']:6d} {row['accepted_bad']:4d}"
        )

    print("\nby source_bins and reltol")
    print("reltol bins points lc_ms_med lc/vbb_med rel_med rel_p90 refined_med maxlev unconv bad")
    for reltol in reltol_values:
        for source_bins in source_bins_values:
            subset = [
                row for row in rows
                if row["reltol"] == reltol and row["source_bins"] == source_bins
            ]
            if not subset:
                continue
            ratios = [row["lc_ms"] / row["vbb_ms"] for row in subset if row["vbb_ms"] > 0.0]
            print(
                f"{reltol:7.0e} {source_bins:4d} {len(subset):6d} "
                f"{statistics.median(row['lc_ms'] for row in subset):9.4f} "
                f"{statistics.median(ratios):10.3f} "
                f"{statistics.median(row['rel_error'] for row in subset):9.2e} "
                f"{np.quantile([row['rel_error'] for row in subset], 0.90):9.2e} "
                f"{statistics.median(row['refinement_level'] for row in subset):11.1f} "
                f"{max(row['refinement_level'] for row in subset):6d} "
                f"{sum(0 if row['converged'] else 1 for row in subset):6d} "
                f"{sum(row['accepted_bad'] for row in subset):4d}"
            )

    print("\nwhere lcbinint is closest to vbm speed")
    print("ratio lc_ms vbb_ms relerr reltol bins case time rho ld mag level conv")
    for row in sorted(rows, key=lambda item: item["lc_ms"] / item["vbb_ms"])[:args.top]:
        print(
            f"{row['lc_ms'] / row['vbb_ms']:7.2f} {row['lc_ms']:7.4f} {row['vbb_ms']:7.4f} "
            f"{row['rel_error']:8.1e} {row['reltol']:7.0e} {row['source_bins']:4d} "
            f"{row['case']:28s} {row['time']: .5f} {row['rho']:.1e} {int(row['ld'])} "
            f"{row['max_mag']:8.2f} {row['refinement_level']:2d} {int(row['converged'])}"
        )

    print("\nwhere lcbinint is slowest vs vbm")
    print("ratio lc_ms vbb_ms relerr reltol bins case time rho ld mag level conv")
    for row in sorted(rows, key=lambda item: item["lc_ms"] / item["vbb_ms"], reverse=True)[:args.top]:
        print(
            f"{row['lc_ms'] / row['vbb_ms']:7.2f} {row['lc_ms']:7.4f} {row['vbb_ms']:7.4f} "
            f"{row['rel_error']:8.1e} {row['reltol']:7.0e} {row['source_bins']:4d} "
            f"{row['case']:28s} {row['time']: .5f} {row['rho']:.1e} {int(row['ld'])} "
            f"{row['max_mag']:8.2f} {row['refinement_level']:2d} {int(row['converged'])}"
        )

    print("\nworst accuracy")
    print("relerr abs_err est target reltol bins case time rho ld mag level conv")
    for row in sorted(rows, key=lambda item: item["rel_error"], reverse=True)[:args.top]:
        print(
            f"{row['rel_error']:8.1e} {row['abs_error']:8.2e} {row['estimate']:8.2e} "
            f"{row['target']:8.2e} {row['reltol']:7.0e} {row['source_bins']:4d} "
            f"{row['case']:28s} {row['time']: .5f} {row['rho']:.1e} {int(row['ld'])} "
            f"{row['max_mag']:8.2f} {row['refinement_level']:2d} {int(row['converged'])}"
        )

    accepted_bad = sum(row["accepted_bad"] for row in rows)
    if accepted_bad:
        print(f"\naccepted_bad_total={accepted_bad}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

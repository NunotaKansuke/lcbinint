from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import time

import numpy as np

import lcbinint
import VBMicrolensing


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    separation: float
    mass_ratio: float
    u0: float
    alpha: float
    rho: float
    t_min: float
    t_max: float
    n_times: int = 241
    limb_darkening_c: float = 0.0


CASES = [
    Case("planetary_small_source", 1.0, 1.0e-3, -0.01, 0.5, 1.0e-4, -0.8, 0.8),
    Case("planetary_large_source", 1.0, 1.0e-3, -0.01, 0.5, 1.0e-2, -0.8, 0.8),
    Case("wide_low_q", 1.5, 1.0e-3, -0.01, 0.5, 1.0e-4, -0.8, 0.8),
    Case("close_binary", 0.7, 0.3, -0.05, 1.2, 3.0e-3, -0.4, 0.4),
    Case("resonant_low_q", 1.0, 0.1, 0.1, 0.0, 3.0e-3, -0.25, 0.25),
    Case("wide_equal_mass", 1.5, 1.0, -0.2, 0.0, 3.0e-3, -0.4, 0.4),
    Case("planetary_small_source_ld", 1.0, 1.0e-3, -0.01, 0.5, 1.0e-4, -0.8, 0.8, limb_darkening_c=0.5),
    Case("planetary_large_source_ld", 1.0, 1.0e-3, -0.01, 0.5, 1.0e-2, -0.8, 0.8, limb_darkening_c=0.5),
]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def random_cases(count: int, seed: int, n_times: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases: list[Case] = []
    for index in range(count):
        family = index % 4
        if family == 0:
            separation = log_uniform(rng, 0.75, 1.35)
            mass_ratio = log_uniform(rng, 1.0e-5, 5.0e-3)
            u0 = rng.uniform(-0.03, 0.03)
            rho = log_uniform(rng, 5.0e-5, 2.0e-2)
            t_span = rng.uniform(0.25, 0.9)
        elif family == 1:
            separation = log_uniform(rng, 0.45, 2.2)
            mass_ratio = log_uniform(rng, 1.0e-2, 1.0)
            u0 = rng.uniform(-0.25, 0.25)
            rho = log_uniform(rng, 1.0e-4, 3.0e-2)
            t_span = rng.uniform(0.2, 0.8)
        elif family == 2:
            separation = log_uniform(rng, 0.9, 1.12)
            mass_ratio = log_uniform(rng, 1.0e-4, 3.0e-1)
            u0 = rng.uniform(-0.08, 0.08)
            rho = log_uniform(rng, 5.0e-5, 1.0e-2)
            t_span = rng.uniform(0.15, 0.55)
        else:
            separation = log_uniform(rng, 0.6, 1.8)
            mass_ratio = log_uniform(rng, 1.0e-4, 1.0)
            u0 = rng.uniform(-0.01, 0.01)
            rho = log_uniform(rng, 5.0e-5, 5.0e-3)
            t_span = rng.uniform(0.05, 0.35)
        alpha = rng.uniform(0.0, math.pi)
        limb_darkening_c = 0.5 if rng.random() < 0.35 else 0.0
        cases.append(
            Case(
                f"random_{index:03d}",
                separation,
                mass_ratio,
                u0,
                alpha,
                rho,
                -t_span,
                t_span,
                n_times=n_times,
                limb_darkening_c=limb_darkening_c,
            )
        )
    return cases


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return float("nan")
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def vbm_curve(case: Case, times: np.ndarray) -> np.ndarray:
    vbb = VBMicrolensing.VBMicrolensing()
    vbb.Tol = 1.0e-3
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


def lc_curve(case: Case, times: np.ndarray, options: lcbinint.Options):
    return lcbinint.light_curve_info(
        times.tolist(),
        u0=case.u0,
        alpha=case.alpha,
        s=case.separation,
        q=case.mass_ratio,
        rho=case.rho,
        limb_darkening=lcbinint.LimbDarkening(c=case.limb_darkening_c),
        options=options,
    )


def summarize_case(case: Case, source_bins: int, reltol: float, max_bins: int):
    times = np.linspace(case.t_min, case.t_max, case.n_times)
    reference, vbb_seconds = timed(lambda: vbm_curve(case, times))

    fixed_options = lcbinint.Options(
        source_bins=source_bins,    )
    fixed, fixed_seconds = timed(lambda: lc_curve(case, times, fixed_options))
    fixed_mag = np.asarray(fixed.magnifications, dtype=float)
    fixed_est = np.asarray(fixed.finite_source_error_estimates, dtype=float)
    fixed_rel = np.abs(fixed_mag / reference - 1.0)

    adaptive_options = lcbinint.Options(
        source_bins=source_bins,
        adaptive_source_bins=1,
        max_source_bins=max_bins,
        reltol=reltol,    )
    adaptive, adaptive_seconds = timed(lambda: lc_curve(case, times, adaptive_options))
    adaptive_mag = np.asarray(adaptive.magnifications, dtype=float)
    adaptive_est = np.asarray(adaptive.finite_source_error_estimates, dtype=float)
    adaptive_levels = np.asarray(adaptive.finite_source_refinement_levels, dtype=int)
    adaptive_converged = np.asarray(adaptive.finite_source_converged, dtype=bool)
    adaptive_rel = np.abs(adaptive_mag / reference - 1.0)

    target = reltol * np.maximum(np.abs(adaptive_mag), 1.0)
    abs_error = np.abs(adaptive_mag - reference)
    underestimated = abs_error > np.maximum(target, adaptive_est) * 1.05
    accepted_bad = (adaptive_est <= target) & (abs_error > target * 1.05)

    worst_index = int(np.argmax(adaptive_rel))
    return {
        "case": case.name,
        "separation": case.separation,
        "mass_ratio": case.mass_ratio,
        "rho": case.rho,
        "u0": case.u0,
        "limb_darkening_c": case.limb_darkening_c,
        "fixed_max_rel": float(np.max(fixed_rel)),
        "adaptive_max_rel": float(np.max(adaptive_rel)),
        "adaptive_p99_rel": float(np.quantile(adaptive_rel, 0.99)),
        "fixed_ms_per_point": 1000.0 * fixed_seconds / len(times),
        "adaptive_ms_per_point": 1000.0 * adaptive_seconds / len(times),
        "vbb_ms_per_point": 1000.0 * vbb_seconds / len(times),
        "max_refinement": int(np.max(adaptive_levels)),
        "refined_points": int(np.count_nonzero(adaptive_levels)),
        "nonzero_estimates": int(np.count_nonzero(adaptive_est > 0.0)),
        "underestimated_points": int(np.count_nonzero(underestimated)),
        "accepted_bad_points": int(np.count_nonzero(accepted_bad)),
        "unconverged_points": int(np.count_nonzero(~adaptive_converged)),
        "worst_time": float(times[worst_index]),
        "worst_rel": float(adaptive_rel[worst_index]),
        "worst_est": float(adaptive_est[worst_index]),
        "worst_level": int(adaptive_levels[worst_index]),
        "max_mag": float(np.max(reference)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bins", type=int, default=50)
    parser.add_argument("--max-bins", type=int, default=200)
    parser.add_argument("--reltol", type=float, default=1.0e-4)
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--random-times", type=int, default=121)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    cases = list(CASES)
    if args.random:
        cases.extend(random_cases(args.random, args.seed, args.random_times))

    rows = [summarize_case(case, args.source_bins, args.reltol, args.max_bins) for case in cases]
    print(f"lcbinint: {lcbinint.__file__}")
    print(f"source_bins={args.source_bins} max_bins={args.max_bins} reltol={args.reltol:g}")
    header = (
        "case fixed_max adaptive_max p99 refined maxlev unconverged bad accepted_bad "
        "vbb_ms fixed_ms adaptive_ms worst_t worst_est worst_level"
    )
    print(header)
    for row in rows:
        print(
            f"{row['case']} "
            f"{row['fixed_max_rel']:.3e} {row['adaptive_max_rel']:.3e} {row['adaptive_p99_rel']:.3e} "
            f"{row['refined_points']} {row['max_refinement']} "
            f"{row['unconverged_points']} "
            f"{row['underestimated_points']} {row['accepted_bad_points']} "
            f"{row['vbb_ms_per_point']:.4f} {row['fixed_ms_per_point']:.4f} {row['adaptive_ms_per_point']:.4f} "
            f"{row['worst_time']:.6g} {row['worst_est']:.3e} {row['worst_level']}"
        )

    accepted_bad = sum(row["accepted_bad_points"] for row in rows)
    worst_rows = sorted(rows, key=lambda row: row["adaptive_max_rel"], reverse=True)[:args.top]
    print("worst_cases")
    for row in worst_rows:
        print(
            f"{row['case']} "
            f"s={row['separation']:.6g} q={row['mass_ratio']:.3e} rho={row['rho']:.3e} "
            f"u0={row['u0']:.4g} ld={row['limb_darkening_c']:.2g} max_mag={row['max_mag']:.4g} "
            f"adaptive_max={row['adaptive_max_rel']:.3e} accepted_bad={row['accepted_bad_points']} "
            f"unconverged={row['unconverged_points']} "
            f"ms={row['adaptive_ms_per_point']:.4f}"
        )
    print("aggregate_speed")
    print("subset cases fixed/vbb_median fixed/vbb_geo adaptive/vbb_median adaptive/vbb_geo adaptive/fixed_median adaptive/fixed_geo")
    subsets = [
        ("all", rows),
        ("no_ld", [row for row in rows if row["limb_darkening_c"] == 0.0]),
        ("ld", [row for row in rows if row["limb_darkening_c"] != 0.0]),
    ]
    for name, subset in subsets:
        if not subset:
            continue
        fixed_vbb = [row["fixed_ms_per_point"] / row["vbb_ms_per_point"] for row in subset]
        adaptive_vbb = [row["adaptive_ms_per_point"] / row["vbb_ms_per_point"] for row in subset]
        adaptive_fixed = [row["adaptive_ms_per_point"] / row["fixed_ms_per_point"] for row in subset]
        print(
            f"{name} {len(subset)} "
            f"{statistics.median(fixed_vbb):.3f} {geomean(fixed_vbb):.3f} "
            f"{statistics.median(adaptive_vbb):.3f} {geomean(adaptive_vbb):.3f} "
            f"{statistics.median(adaptive_fixed):.3f} {geomean(adaptive_fixed):.3f}"
        )
    if accepted_bad:
        print(f"accepted_bad_total={accepted_bad}")
        return 2
    print(f"median_adaptive_max_rel={statistics.median(row['adaptive_max_rel'] for row in rows):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

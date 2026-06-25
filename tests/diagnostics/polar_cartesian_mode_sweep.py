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
    limb_darkening_c: float = 0.0


BASE_CASES = [
    Case("planet_resonant_tiny", 1.0, 1.0e-3, -1.0e-4, 0.5, 1.0e-4, -0.01, 0.01),
    Case("planet_resonant_small", 1.0, 1.0e-3, -1.0e-3, 0.5, 3.0e-3, -0.05, 0.05),
    Case("planet_resonant_mid", 0.95, 1.0e-2, -1.0e-3, 0.5, 5.0e-3, -0.05, 0.05),
    Case("planet_close_mid", 0.8, 1.0e-2, -1.0e-3, 0.3, 5.0e-3, -0.05, 0.05),
    Case("planet_large_source", 1.0, 1.0e-3, -1.0e-2, 0.5, 3.0e-2, -0.08, 0.08),
    Case("intermediate_q", 1.0, 0.1, 0.01, 0.0, 5.0e-3, -0.08, 0.08),
    Case("close_binary", 0.7, 0.3, -0.03, 1.2, 5.0e-3, -0.08, 0.08),
    Case("wide_equal_mass", 1.5, 1.0, -0.1, 0.0, 1.0e-2, -0.12, 0.12),
]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def random_cases(count: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases: list[Case] = []
    for index in range(count):
        family = index % 5
        if family == 0:
            separation = log_uniform(rng, 0.9, 1.12)
            mass_ratio = log_uniform(rng, 1.0e-5, 5.0e-3)
            rho = log_uniform(rng, 5.0e-5, 3.0e-3)
            u0 = rng.uniform(-2.0 * rho, 2.0 * rho)
            span = rng.uniform(0.01, 0.08)
        elif family == 1:
            separation = log_uniform(rng, 0.75, 1.35)
            mass_ratio = log_uniform(rng, 1.0e-3, 3.0e-2)
            rho = log_uniform(rng, 1.0e-3, 1.0e-2)
            u0 = rng.uniform(-0.02, 0.02)
            span = rng.uniform(0.04, 0.15)
        elif family == 2:
            separation = log_uniform(rng, 0.5, 2.0)
            mass_ratio = log_uniform(rng, 5.0e-2, 1.0)
            rho = log_uniform(rng, 1.0e-3, 3.0e-2)
            u0 = rng.uniform(-0.15, 0.15)
            span = rng.uniform(0.06, 0.25)
        elif family == 3:
            separation = log_uniform(rng, 0.7, 1.5)
            mass_ratio = log_uniform(rng, 1.0e-4, 1.0e-1)
            rho = log_uniform(rng, 1.0e-2, 1.0e-1)
            u0 = rng.uniform(-0.05, 0.05)
            span = rng.uniform(0.08, 0.3)
        else:
            separation = log_uniform(rng, 0.5, 2.5)
            mass_ratio = log_uniform(rng, 1.0e-5, 1.0)
            rho = log_uniform(rng, 5.0e-5, 1.0e-1)
            u0 = rng.uniform(-0.1, 0.1)
            span = rng.uniform(0.02, 0.3)
        cases.append(
            Case(
                f"random_{index:03d}",
                separation,
                mass_ratio,
                u0,
                rng.uniform(0.0, math.pi),
                rho,
                -span,
                span,
                0.5 if index % 2 else 0.0,
            )
        )
    return cases


def timed_best(func, repeat: int):
    best_value = None
    best_seconds = float("inf")
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        value = func()
        seconds = time.perf_counter() - start
        if seconds < best_seconds:
            best_value = value
            best_seconds = seconds
    return best_value, best_seconds


def vbm_curve(case: Case, times: np.ndarray, tol: float) -> np.ndarray:
    vbb = VBMicrolensing.VBMicrolensing()
    vbb.Tol = tol
    vbb.RelTol = 0.0
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


def lc_point(case: Case, time_value: float, inverse_ray_grid: str, source_bins: int):
    options = lcbinint.Options(
        coordinates="vbm",
        inverse_ray_grid=inverse_ray_grid,
        source_bins=source_bins,
        adaptive_source_bins=0,
        point_source_threshold=1.0e9,
        hexadecapole_threshold=1.0e9,
    )
    func = lcbinint.LightCurve(
        options=options,
        limb_darkening=lcbinint.LimbDarkening.linear(case.limb_darkening_c)
        if case.limb_darkening_c
        else lcbinint.LimbDarkening.none(),
    )
    return func.info(
        [time_value],
        t0=0.0,
        tE=1.0,
        u0=case.u0,
        alpha=case.alpha,
        s=case.separation,
        q=case.mass_ratio,
        rho=case.rho,
    )


def selected_times(case: Case, points: int, reference_tol: float) -> list[tuple[float, float]]:
    scan_count = max(101, points * 25)
    times = np.linspace(case.t_min, case.t_max, scan_count)
    reference = vbm_curve(case, times, reference_tol)
    indices = {0, scan_count - 1, int(np.argmax(reference))}
    if points > len(indices):
        order = np.argsort(reference)[::-1]
        for index in order:
            indices.add(int(index))
            if len(indices) >= points:
                break
    return [(float(times[index]), float(reference[index])) for index in sorted(indices)[:points]]


def bin_rho(rho: float) -> str:
    if rho < 3.0e-4:
        return "<3e-4"
    if rho < 1.0e-3:
        return "3e-4..1e-3"
    if rho < 3.0e-3:
        return "1e-3..3e-3"
    if rho < 1.0e-2:
        return "3e-3..1e-2"
    if rho < 3.0e-2:
        return "1e-2..3e-2"
    return ">=3e-2"


def bin_mag(mag: float) -> str:
    if mag < 10.0:
        return "<10"
    if mag < 30.0:
        return "10..30"
    if mag < 100.0:
        return "30..100"
    if mag < 300.0:
        return "100..300"
    if mag < 1000.0:
        return "300..1000"
    return ">=1000"


def bin_q(q: float) -> str:
    if q < 1.0e-4:
        return "<1e-4"
    if q < 1.0e-3:
        return "1e-4..1e-3"
    if q < 1.0e-2:
        return "1e-3..1e-2"
    if q < 1.0e-1:
        return "1e-2..1e-1"
    return ">=1e-1"


def bin_s(s: float) -> str:
    if s < 0.8:
        return "<0.8"
    if s < 0.95:
        return "0.8..0.95"
    if s < 1.05:
        return "0.95..1.05"
    if s < 1.3:
        return "1.05..1.3"
    return ">=1.3"


def summarize(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    output = []
    for value, subset in sorted(grouped.items()):
        ratios = [row["polar_ms"] / row["cart_ms"] for row in subset if row["cart_ms"] > 0.0]
        auto_ratios = [row["auto_ms"] / row["cart_ms"] for row in subset if row["cart_ms"] > 0.0]
        wins = sum(1 for ratio in ratios if ratio < 1.0)
        auto_wins = sum(1 for ratio in auto_ratios if ratio < 1.0)
        output.append(
            {
                key: value,
                "points": len(subset),
                "polar_win_rate": wins / len(ratios) if ratios else float("nan"),
                "auto_win_rate": auto_wins / len(auto_ratios) if auto_ratios else float("nan"),
                "ratio_median": statistics.median(ratios) if ratios else float("nan"),
                "auto_ratio_median": statistics.median(auto_ratios) if auto_ratios else float("nan"),
                "ratio_geo": math.exp(statistics.fmean(math.log(r) for r in ratios if r > 0.0))
                if ratios
                else float("nan"),
                "cart_ms_median": statistics.median(row["cart_ms"] for row in subset),
                "polar_ms_median": statistics.median(row["polar_ms"] for row in subset),
                "auto_ms_median": statistics.median(row["auto_ms"] for row in subset),
                "polar_rel_p90": float(np.quantile([row["polar_rel"] for row in subset], 0.90)),
                "auto_rel_p90": float(np.quantile([row["auto_rel"] for row in subset], 0.90)),
                "cart_rel_p90": float(np.quantile([row["cart_rel"] for row in subset], 0.90)),
                "polar_bad": sum(1 for row in subset if row["polar_rel"] > 3.0e-3),
                "auto_bad": sum(1 for row in subset if row["auto_rel"] > 3.0e-3),
            }
        )
    return output


def print_summary(rows: list[dict], key: str) -> None:
    print(f"\nby {key}")
    print("bin points polar_win auto_win polar/cart_med auto/cart_med cart_ms polar_ms auto_ms cart_p90 polar_p90 auto_p90 polar_bad auto_bad")
    for row in summarize(rows, key):
        print(
            f"{str(row[key]):12s} {row['points']:6d} "
            f"{row['polar_win_rate']:9.2%} {row['auto_win_rate']:8.2%} "
            f"{row['ratio_median']:14.3f} {row['auto_ratio_median']:13.3f} "
            f"{row['cart_ms_median']:7.3f} {row['polar_ms_median']:8.3f} "
            f"{row['auto_ms_median']:7.3f} {row['cart_rel_p90']:8.1e} "
            f"{row['polar_rel_p90']:9.1e} {row['auto_rel_p90']:8.1e} "
            f"{row['polar_bad']:9d} {row['auto_bad']:8d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bins", type=int, default=50)
    parser.add_argument("--points-per-case", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--random", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--reference-tol", type=float, default=1.0e-5)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    cases = list(BASE_CASES) + random_cases(args.random, args.seed)
    rows: list[dict] = []
    for case_index, case in enumerate(cases, start=1):
        for time_value, reference in selected_times(case, args.points_per_case, args.reference_tol):
            cart, cart_seconds = timed_best(
                lambda: lc_point(case, time_value, "cartesian", args.source_bins),
                args.repeat,
            )
            polar, polar_seconds = timed_best(
                lambda: lc_point(case, time_value, "polar", args.source_bins),
                args.repeat,
            )
            auto, auto_seconds = timed_best(
                lambda: lc_point(case, time_value, "auto", args.source_bins),
                args.repeat,
            )
            cart_mag = float(cart.magnifications[0])
            polar_mag = float(polar.magnifications[0])
            auto_mag = float(auto.magnifications[0])
            rows.append(
                {
                    "case": case.name,
                    "time": time_value,
                    "s": case.separation,
                    "q": case.mass_ratio,
                    "rho": case.rho,
                    "ld": case.limb_darkening_c != 0.0,
                    "mag": reference,
                    "rho_bin": bin_rho(case.rho),
                    "mag_bin": bin_mag(reference),
                    "q_bin": bin_q(case.mass_ratio),
                    "s_bin": bin_s(case.separation),
                    "ld_bin": "LD" if case.limb_darkening_c else "noLD",
                    "cart_ms": 1000.0 * cart_seconds,
                    "polar_ms": 1000.0 * polar_seconds,
                    "auto_ms": 1000.0 * auto_seconds,
                    "cart_mag": cart_mag,
                    "polar_mag": polar_mag,
                    "auto_mag": auto_mag,
                    "cart_rel": abs(cart_mag / reference - 1.0),
                    "polar_rel": abs(polar_mag / reference - 1.0),
                    "auto_rel": abs(auto_mag / reference - 1.0),
                    "cart_method": cart.finite_source_method_names[0],
                    "polar_method": polar.finite_source_method_names[0],
                    "auto_method": auto.finite_source_method_names[0],
                }
            )
        if case_index % 10 == 0:
            print(f"processed {case_index}/{len(cases)} cases")

    print(f"lcbinint: {lcbinint.__file__}")
    print(
        f"polar/cartesian mode sweep cases={len(cases)} points={len(rows)} "
        f"source_bins={args.source_bins} repeat={args.repeat}"
    )
    for key in ["rho_bin", "mag_bin", "q_bin", "s_bin", "ld_bin"]:
        print_summary(rows, key)

    print("\nfastest polar wins")
    print("ratio cart_ms polar_ms cart_rel polar_rel case t rho q s mag ld")
    for row in sorted(rows, key=lambda item: item["polar_ms"] / item["cart_ms"])[: args.top]:
        print(
            f"{row['polar_ms'] / row['cart_ms']:7.3f} {row['cart_ms']:7.3f} "
            f"{row['polar_ms']:8.3f} {row['cart_rel']:8.1e} {row['polar_rel']:9.1e} "
            f"{row['case']:24s} {row['time']: .5g} {row['rho']:.1e} "
            f"{row['q']:.1e} {row['s']:.3f} {row['mag']:.1f} {int(row['ld'])}"
        )

    print("\nworst polar losses")
    print("ratio cart_ms polar_ms cart_rel polar_rel case t rho q s mag ld")
    for row in sorted(rows, key=lambda item: item["polar_ms"] / item["cart_ms"], reverse=True)[: args.top]:
        print(
            f"{row['polar_ms'] / row['cart_ms']:7.3f} {row['cart_ms']:7.3f} "
            f"{row['polar_ms']:8.3f} {row['cart_rel']:8.1e} {row['polar_rel']:9.1e} "
            f"{row['case']:24s} {row['time']: .5g} {row['rho']:.1e} "
            f"{row['q']:.1e} {row['s']:.3f} {row['mag']:.1f} {int(row['ld'])}"
        )

    print("\nworst polar accuracy")
    print("polar_rel cart_rel ratio cart_ms polar_ms case t rho q s mag ld")
    for row in sorted(rows, key=lambda item: item["polar_rel"], reverse=True)[: args.top]:
        print(
            f"{row['polar_rel']:9.1e} {row['cart_rel']:8.1e} "
            f"{row['polar_ms'] / row['cart_ms']:7.3f} {row['cart_ms']:7.3f} "
            f"{row['polar_ms']:8.3f} {row['case']:24s} {row['time']: .5g} "
            f"{row['rho']:.1e} {row['q']:.1e} {row['s']:.3f} {row['mag']:.1f} {int(row['ld'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

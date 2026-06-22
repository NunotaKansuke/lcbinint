from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import time

import numpy as np

import lcbinint
import VBBinaryLensing


@dataclasses.dataclass(frozen=True)
class Geometry:
    name: str
    separation: float
    mass_ratio: float
    u0: float
    alpha: float
    t_min: float
    t_max: float


GEOMETRIES = [
    Geometry("planetary_resonant", 1.0, 1.0e-3, -0.01, 0.5, -0.8, 0.8),
    Geometry("planetary_wide", 1.35, 3.0e-3, -0.02, 0.7, -0.8, 0.8),
    Geometry("planetary_close", 0.8, 1.0e-2, -0.01, 0.3, -0.8, 0.8),
    Geometry("intermediate_q", 1.0, 0.1, 0.05, 0.0, -0.35, 0.35),
    Geometry("close_binary", 0.7, 0.3, -0.05, 1.2, -0.4, 0.4),
    Geometry("wide_equal_mass", 1.5, 1.0, -0.2, 0.0, -0.4, 0.4),
]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def random_geometries(count: int, seed: int) -> list[Geometry]:
    rng = np.random.default_rng(seed)
    geometries: list[Geometry] = []
    for index in range(count):
        family = index % 4
        if family == 0:
            separation = log_uniform(rng, 0.75, 1.35)
            mass_ratio = log_uniform(rng, 1.0e-5, 5.0e-3)
            u0 = rng.uniform(-0.03, 0.03)
            t_span = rng.uniform(0.25, 0.9)
        elif family == 1:
            separation = log_uniform(rng, 0.45, 2.2)
            mass_ratio = log_uniform(rng, 1.0e-2, 1.0)
            u0 = rng.uniform(-0.25, 0.25)
            t_span = rng.uniform(0.2, 0.8)
        elif family == 2:
            separation = log_uniform(rng, 0.9, 1.12)
            mass_ratio = log_uniform(rng, 1.0e-4, 3.0e-1)
            u0 = rng.uniform(-0.08, 0.08)
            t_span = rng.uniform(0.15, 0.55)
        else:
            separation = log_uniform(rng, 0.6, 1.8)
            mass_ratio = log_uniform(rng, 1.0e-4, 1.0)
            u0 = rng.uniform(-0.01, 0.01)
            t_span = rng.uniform(0.05, 0.35)
        geometries.append(
            Geometry(
                f"random_{index:03d}",
                separation,
                mass_ratio,
                u0,
                rng.uniform(0.0, math.pi),
                -t_span,
                t_span,
            )
        )
    return geometries


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return float("nan")
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def vbbl_curve(
    geometry: Geometry,
    rho: float,
    times: np.ndarray,
    limb_darkening_c: float,
    tol: float,
) -> np.ndarray:
    vbb = VBBinaryLensing.VBBinaryLensing()
    vbb.Tol = tol
    vbb.RelTol = 0.0
    vbb.a1 = limb_darkening_c
    params = [
        math.log(geometry.separation),
        math.log(geometry.mass_ratio),
        geometry.u0,
        geometry.alpha,
        math.log(rho),
        0.0,
        0.0,
    ]
    return np.asarray(vbb.BinaryLightCurve(params, times.tolist())[0], dtype=float)


def lcbinint_curve(
    geometry: Geometry,
    rho: float,
    times: np.ndarray,
    source_bins: int,
    limb_darkening_c: float,
) -> np.ndarray:
    params = lcbinint.LensParams(
        t0=0.0,
        tE=1.0,
        u0=geometry.u0,
        alpha=geometry.alpha,
        q=geometry.mass_ratio,
        sep=geometry.separation,
        rho=rho,
        limb_darkening_c=limb_darkening_c,
    )
    options = lcbinint.Options(
        source_bins=source_bins,
        adaptive_source_bins=0,
        vbbl_compatible=1,
    )
    curve = lcbinint.LensModel(params, options).light_curve(times.tolist())
    return np.asarray(curve.magnifications, dtype=float)


def summarize(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[float | int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    summary = []
    for value, subset in sorted(grouped.items()):
        ratios = [row["lc_ms"] / row["vbb_ms"] for row in subset]
        summary.append(
            {
                key: value,
                "cases": len(subset),
                "ratio_median": statistics.median(ratios),
                "ratio_geo": geomean(ratios),
                "lc_ms_median": statistics.median(row["lc_ms"] for row in subset),
                "vbb_ms_median": statistics.median(row["vbb_ms"] for row in subset),
                "max_rel_median": statistics.median(row["max_rel"] for row in subset),
                "max_rel_worst": max(row["max_rel"] for row in subset),
                "p99_rel_median": statistics.median(row["p99_rel"] for row in subset),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    parser.add_argument("--source-bins", default="25,35,50,70,100,140,200")
    parser.add_argument("--times", type=int, default=121)
    parser.add_argument("--limb-darkening-c", type=float, default=0.5)
    parser.add_argument("--vbbl-tol", type=float, default=1.0e-3)
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rhos = parse_csv_floats(args.rhos)
    source_bins_values = parse_csv_ints(args.source_bins)
    geometries = list(GEOMETRIES)
    if args.random:
        geometries.extend(random_geometries(args.random, args.seed))
    rows: list[dict] = []

    for geometry in geometries:
        times = np.linspace(geometry.t_min, geometry.t_max, args.times)
        for rho in rhos:
            reference, vbb_seconds = timed(
                lambda: vbbl_curve(
                    geometry, rho, times, args.limb_darkening_c, args.vbbl_tol
                )
            )
            vbb_ms = 1000.0 * vbb_seconds / len(times)
            for source_bins in source_bins_values:
                actual, lc_seconds = timed(
                    lambda: lcbinint_curve(
                        geometry, rho, times, source_bins, args.limb_darkening_c
                    )
                )
                rel = np.abs(actual / reference - 1.0)
                rows.append(
                    {
                        "geometry": geometry.name,
                        "rho": rho,
                        "source_bins": source_bins,
                        "vbb_ms": vbb_ms,
                        "lc_ms": 1000.0 * lc_seconds / len(times),
                        "ratio": (1000.0 * lc_seconds / len(times)) / vbb_ms,
                        "max_rel": float(np.max(rel)),
                        "p99_rel": float(np.quantile(rel, 0.99)),
                        "median_rel": float(np.median(rel)),
                        "max_mag": float(np.max(reference)),
                    }
                )

    print(f"lcbinint: {lcbinint.__file__}")
    print(
        "LD fixed-bin sweep "
        f"c={args.limb_darkening_c:g} vbbl_tol={args.vbbl_tol:g} times={args.times}"
    )
    print("\nby source_bins")
    print("bins cases lc/vbb_med lc/vbb_geo lc_ms_med vbb_ms_med maxrel_med maxrel_worst p99rel_med")
    for row in summarize(rows, "source_bins"):
        print(
            f"{row['source_bins']:4d} {row['cases']:5d} "
            f"{row['ratio_median']:10.3f} {row['ratio_geo']:10.3f} "
            f"{row['lc_ms_median']:9.4f} {row['vbb_ms_median']:10.4f} "
            f"{row['max_rel_median']:10.3e} {row['max_rel_worst']:12.3e} "
            f"{row['p99_rel_median']:10.3e}"
        )

    print("\nby rho")
    print("rho cases lc/vbb_med lc/vbb_geo lc_ms_med vbb_ms_med maxrel_med maxrel_worst p99rel_med")
    for row in summarize(rows, "rho"):
        print(
            f"{row['rho']:8.1e} {row['cases']:5d} "
            f"{row['ratio_median']:10.3f} {row['ratio_geo']:10.3f} "
            f"{row['lc_ms_median']:9.4f} {row['vbb_ms_median']:10.4f} "
            f"{row['max_rel_median']:10.3e} {row['max_rel_worst']:12.3e} "
            f"{row['p99_rel_median']:10.3e}"
        )

    print("\nworst speed ratios")
    print("geometry rho bins lc/vbb lc_ms vbb_ms maxrel p99rel maxmag")
    for row in sorted(rows, key=lambda item: item["ratio"], reverse=True)[: args.top]:
        print(
            f"{row['geometry']:18s} {row['rho']:8.1e} {row['source_bins']:4d} "
            f"{row['ratio']:7.2f} {row['lc_ms']:7.4f} {row['vbb_ms']:7.4f} "
            f"{row['max_rel']:9.2e} {row['p99_rel']:9.2e} {row['max_mag']:8.2f}"
        )

    print("\nworst accuracy")
    print("geometry rho bins lc/vbb lc_ms vbb_ms maxrel p99rel medrel maxmag")
    for row in sorted(rows, key=lambda item: item["max_rel"], reverse=True)[: args.top]:
        print(
            f"{row['geometry']:18s} {row['rho']:8.1e} {row['source_bins']:4d} "
            f"{row['ratio']:7.2f} {row['lc_ms']:7.4f} {row['vbb_ms']:7.4f} "
            f"{row['max_rel']:9.2e} {row['p99_rel']:9.2e} "
            f"{row['median_rel']:9.2e} {row['max_mag']:8.2f}"
        )

    print("\nbest speed ratios")
    print("geometry rho bins lc/vbb lc_ms vbb_ms maxrel p99rel maxmag")
    for row in sorted(rows, key=lambda item: item["ratio"])[: args.top]:
        print(
            f"{row['geometry']:18s} {row['rho']:8.1e} {row['source_bins']:4d} "
            f"{row['ratio']:7.2f} {row['lc_ms']:7.4f} {row['vbb_ms']:7.4f} "
            f"{row['max_rel']:9.2e} {row['p99_rel']:9.2e} {row['max_mag']:8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

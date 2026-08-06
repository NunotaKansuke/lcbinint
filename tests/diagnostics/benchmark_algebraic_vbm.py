"""Matched-case native algebraic-boundary versus VBMicrolensing benchmark.

Run after building ``benchmark_algebraic_boundary``.  Arguments are native
timing repeats, the minimum VBM annulus count for the linear profile, and the
case suite (defaults: 5, 8, and ``core``).  ``limb-sweep`` runs a broader
linear-limb-darkening-only geometry corpus.  The native executable
supplies its own high-resolution raster reference and timings; this driver
adds VBMicrolensing's raw contour integrator on the identical geometries.
Square-root limb darkening is intentionally omitted because VBM exposes only
the linear profile through this API.
"""

from __future__ import annotations

import csv
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


CASES = {
    "smooth_planetary": (1.1, 1.0e-3, 0.08, 0.03, 1.0e-2),
    "central_high_mag": (1.1, 1.0e-3, 0.002, 0.0007, 1.0e-3),
    "tiny_source": (1.1, 1.0e-3, 0.0015, -0.0004, 3.0e-4),
    "wide_binary": (2.0, 0.1, 0.10, 0.05, 1.0e-2),
    "equal_resonant": (1.0, 1.0, 0.01, 0.02, 2.0e-2),
    "smooth_low_q": (1.0, 0.1, 0.4, 0.3, 2.0e-2),
    "smooth_close": (0.7, 0.3, -0.6, 0.4, 3.0e-2),
    "smooth_wide_hard": (1.4, 0.4, 0.2, -0.15, 2.5e-2),
    "far_planetary": (1.0, 1.0e-3, 0.4, 0.3, 1.0e-2),
    "central_close": (0.6, 0.5, 0.05, 0.02, 1.0e-2),
    "central_planetary": (1.05, 1.0e-3, 0.05, 0.02, 5.0e-3),
    "central_equal": (1.0, 1.0, 0.01, 0.02, 2.0e-2),
    "caustic_crossing": (1.0, 0.2, 0.10, 0.05, 1.0e-2),
    "compact_close_caustic": (
        0.3, 1.0e-3, 0.008845738870878478, 0.001678002323850983, 3.0e-3),
    "wide_equal": (1.8, 1.0, 0.30, 0.10, 2.0e-2),
    "near_tangent_cap": (1.2, 0.1, 0.653, 0.020, 2.0e-2),
    "tiny_q_tangent": (
        1.3, 1.0e-4, 0.0014470002051158422, 0.0006812879524093549,
        0.0015515122255040925),
    "order_sensitive_tangent": (
        1.05, 1.0e-3, 0.010495776906606516, 0.004729643382853728,
        0.012087561335986492),
}
PROFILES = {"uniform": 0.0, "linear": 0.5}
ABSOLUTE_FLOOR = 1.0e-12


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    return values[max(0, math.ceil(probability * len(values)) - 1)]


def median(values) -> float:
    values = list(values)
    return statistics.median(values) if values else math.nan


def timed(call, repeats: int) -> tuple[float, float, float]:
    value = float(call())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = float(call())
        samples.append(time.perf_counter() - started)
    return value, statistics.median(samples), percentile(samples, 0.9)


def magnification_band(value: float) -> str:
    if value < 5.0:
        return "A<5"
    if value < 20.0:
        return "5<=A<20"
    if value < 100.0:
        return "20<=A<100"
    return "A>=100"


def main() -> int:
    import VBMicrolensing

    repeats = max(1, int(sys.argv[1])) if len(sys.argv) > 1 else 5
    linear_min_annuli = max(1, int(sys.argv[2])) if len(sys.argv) > 2 else 8
    suite = sys.argv[3] if len(sys.argv) > 3 else "core"
    if suite not in {"core", "limb-sweep"}:
        raise SystemExit("suite must be 'core' or 'limb-sweep'")
    root = Path(__file__).resolve().parents[2]
    executable = root / "build" / "benchmark_algebraic_boundary"
    completed = subprocess.run(
        [str(executable), str(repeats), suite],
        check=True, capture_output=True, text=True)
    native_rows = list(csv.DictReader(
        line for line in completed.stdout.splitlines() if not line.startswith("#")))

    output = []
    print(
        "case,profile,tolerance,reference,polar_alg_error,cart_alg_error,"
        "vbm_error,polar_alg_seconds,cart_alg_seconds,vbm_median_seconds,"
        "vbm_p90_seconds,vbm_over_polar_alg,vbm_over_cart_alg"
    )
    for row in native_rows:
        profile = row["profile"]
        if profile not in PROFILES:
            continue
        name = row["case"]
        tolerance = float(row["tolerance"])
        s, q, x, y, rho = CASES[name]
        c = PROFILES[profile]
        engine = VBMicrolensing.VBMicrolensing()
        engine.Tol = ABSOLUTE_FLOOR
        engine.RelTol = tolerance
        # These are direct FiniteSourceMagnifier coordinates, before the
        # public light-curve layer's x-frame conversion, so no x mirror is
        # applied here (the existing public-API VBM benchmarks do apply one).
        if c:
            # In the installed VBM 5.x binding the sixth argument is the
            # absolute accuracy.  The limb coefficient is object state.
            engine.a1 = c
            engine.minannuli = linear_min_annuli
            engine.SetLDprofile(engine.LDlinear)
            call = lambda: engine.BinaryMagDark(
                s, q, x, y, rho, ABSOLUTE_FLOOR)
        else:
            call = lambda: engine.BinaryMag(s, q, x, y, rho, ABSOLUTE_FLOOR)
        value, vbm_median, vbm_p90 = timed(call, repeats)
        reference = float(row["reference"])
        vbm_error = abs(value - reference) / max(abs(reference), 1.0)
        polar_seconds = float(row["polar_algebraic_seconds"])
        cart_seconds = float(row["cartesian_algebraic_seconds"])
        record = (
            profile,
            tolerance,
            bool(int(row["polar_alg_success"])),
            bool(int(row["cart_alg_success"])),
            vbm_error,
            float(row["polar_algebraic_error"]),
            float(row["cartesian_algebraic_error"]),
            vbm_median / polar_seconds,
            vbm_median / cart_seconds,
            float(row["polar_raster_seconds"]) / polar_seconds,
            float(row["polar_raster_seconds"]) / cart_seconds,
            name,
            row["family"],
            reference,
        )
        output.append(record)
        print(
            f"{name},{profile},{tolerance:.12g},{reference:.12g},"
            f"{float(row['polar_algebraic_error']):.12g},"
            f"{float(row['cartesian_algebraic_error']):.12g},{vbm_error:.12g},"
            f"{polar_seconds:.12g},{cart_seconds:.12g},{vbm_median:.12g},"
            f"{vbm_p90:.12g},{vbm_median / polar_seconds:.12g},"
            f"{vbm_median / cart_seconds:.12g}"
        )

    print(
        "# summary,profile,tolerance,total,polar_success,cart_success,vbm_p90_error,"
        "polar_p90_error,cart_p90_error,"
        "vbm_min_annuli,"
        "median_original_over_polar_alg,median_original_over_cart_alg,"
        "median_vbm_over_polar_alg,median_vbm_over_cart_alg"
    )
    for profile in sorted({item[0] for item in output}):
        for tolerance in sorted({item[1] for item in output}, reverse=True):
            subset = [item for item in output
                      if item[0] == profile and item[1] == tolerance]
            print(
                f"# summary,{profile},{tolerance:.12g},{len(subset)},"
                f"{sum(item[2] for item in subset)},{sum(item[3] for item in subset)},"
                f"{percentile([item[4] for item in subset], 0.9):.12g},"
                f"{percentile([item[5] for item in subset if item[2]], 0.9):.12g},"
                f"{percentile([item[6] for item in subset if item[3]], 0.9):.12g},"
                f"{linear_min_annuli if profile == 'linear' else 0},"
                f"{median(item[9] for item in subset if item[2]):.12g},"
                f"{median(item[10] for item in subset if item[3]):.12g},"
                f"{median(item[7] for item in subset if item[2]):.12g},"
                f"{median(item[8] for item in subset if item[3]):.12g}"
            )

    if suite == "limb-sweep":
        print(
            "# grouped,group,tolerance,total,polar_success,cart_success,"
            "vbm_p90_error,polar_p90_error,cart_p90_error,"
            "median_vbm_over_polar,median_vbm_over_cart,"
            "polar_faster_count,cart_faster_count"
        )
        for tolerance in sorted({item[1] for item in output}, reverse=True):
            rows = [(item, item[12], item[13]) for item in output
                    if item[1] == tolerance]
            groups: dict[str, list[tuple]] = {"all": rows}
            for item, family, reference in rows:
                groups.setdefault(f"family:{family}", []).append(
                    (item, family, reference))
                groups.setdefault(
                    f"magnification:{magnification_band(reference)}", []
                ).append((item, family, reference))
            for group, members in groups.items():
                items = [member[0] for member in members]
                polar_ratios = [item[7] for item in items if item[2]]
                cart_ratios = [item[8] for item in items if item[3]]
                print(
                    f"# grouped,{group},{tolerance:.12g},{len(items)},"
                    f"{sum(item[2] for item in items)},{sum(item[3] for item in items)},"
                    f"{percentile([item[4] for item in items], 0.9):.12g},"
                    f"{percentile([item[5] for item in items if item[2]], 0.9):.12g},"
                    f"{percentile([item[6] for item in items if item[3]], 0.9):.12g},"
                    f"{median(polar_ratios):.12g},"
                    f"{median(cart_ratios):.12g},"
                    f"{sum(ratio > 1.0 for ratio in polar_ratios)},"
                    f"{sum(ratio > 1.0 for ratio in cart_ratios)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

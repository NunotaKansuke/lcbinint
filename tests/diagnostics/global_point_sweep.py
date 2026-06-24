from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import VBMicrolensing


def load_lcbinint_module():
    extension_path = Path(sys.argv[sys.argv.index("--extension") + 1]) if "--extension" in sys.argv else None
    if extension_path is None:
        import lcbinint as module
        return module
    sys.argv.remove("--extension")
    sys.argv.remove(str(extension_path))
    spec = importlib.util.spec_from_file_location("lcbinint", extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lcbinint extension from {extension_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lcbinint"] = module
    spec.loader.exec_module(module)
    return module


lcbinint = load_lcbinint_module()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_source_bins_sweep import CASES, Case, random_cases  # noqa: E402
from point_integration_benchmark import geomean, selected_points  # noqa: E402


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


def vbm_point(case: Case, time_value: float, tol: float) -> float:
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
    return float(np.asarray(vbb.BinaryLightCurve(params, [time_value])[0], dtype=float)[0])


def lcbinint_point(case: Case, time_value: float, source_bins: int, max_bins: int, reltol: float):
    options = lcbinint.Options(
        source_bins=source_bins,
        adaptive_source_bins=1,
        max_source_bins=max_bins,
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


def bin_label(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def categories(row: dict) -> dict[str, str]:
    return {
        "ld_bin": "ld" if row["ld"] else "no_ld",
        "rho_bin": bin_label(row["rho"], [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], ["<1e-4", "1e-4-3e-4", "3e-4-1e-3", "1e-3-3e-3", "3e-3-1e-2", ">=1e-2"]),
        "mag_bin": bin_label(row["reference"], [5, 20, 100, 300], ["A<5", "5-20", "20-100", "100-300", ">=300"]),
        "q_bin": bin_label(row["mass_ratio"], [1e-4, 1e-3, 1e-2, 1e-1], ["q<1e-4", "1e-4-1e-3", "1e-3-1e-2", "1e-2-1e-1", "q>=1e-1"]),
        "s_bin": "close" if row["separation"] < 0.9 else "resonant" if row["separation"] <= 1.1 else "wide",
        "method": row["method"],
    }


def summarize(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    out = []
    for value, subset in sorted(grouped.items(), key=lambda item: item[0]):
        ratios = [r["ratio"] for r in subset if math.isfinite(r["ratio"])]
        wins = [r for r in subset if r["ratio"] < 1.0]
        out.append({
            "group": value,
            "n": len(subset),
            "win_rate": len(wins) / len(subset),
            "ratio_med": statistics.median(ratios) if ratios else float("nan"),
            "ratio_geo": geomean(ratios),
            "lc_ms_med": statistics.median(r["lc_ms"] for r in subset),
            "vbb_ms_med": statistics.median(r["vbb_ms"] for r in subset),
            "rel_p95": float(np.quantile([r["rel_error"] for r in subset], 0.95)),
            "accepted_bad": sum(r["accepted_bad"] for r in subset),
            "unconverged": sum(0 if r["converged"] else 1 for r in subset),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--points-per-case", type=int, default=5)
    parser.add_argument("--source-bins", type=int, default=50)
    parser.add_argument("--max-bins", type=int, default=400)
    parser.add_argument("--reltol", type=float, default=1e-3)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--reference-tol", type=float, default=1e-5)
    parser.add_argument("--vbm-tol", type=float, default=1e-3)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    cases = list(CASES) + random_cases(args.random, args.seed, args.points_per_case * 24)
    point_rows = []
    for case in cases:
        point_rows.extend(selected_points(case, args.points_per_case, args.reference_tol))

    rows = []
    for index, point in enumerate(point_rows, start=1):
        case = point.case
        reference, vbb_seconds = timed_best(lambda: vbm_point(case, point.time, args.vbm_tol), args.repeat)
        curve, lc_seconds = timed_best(
            lambda: lcbinint_point(case, point.time, args.source_bins, args.max_bins, args.reltol),
            args.repeat,
        )
        actual = float(curve.magnifications[0])
        abs_error = abs(actual - reference)
        target = args.reltol * max(abs(actual), 1.0)
        row = {
            "case": case.name,
            "separation": case.separation,
            "mass_ratio": case.mass_ratio,
            "u0": case.u0,
            "alpha": case.alpha,
            "rho": case.rho,
            "ld": case.limb_darkening_c != 0.0,
            "time": point.time,
            "reference": reference,
            "actual": actual,
            "vbb_ms": 1000.0 * vbb_seconds,
            "lc_ms": 1000.0 * lc_seconds,
            "ratio": (1000.0 * lc_seconds) / (1000.0 * vbb_seconds) if vbb_seconds > 0.0 else float("nan"),
            "abs_error": abs_error,
            "rel_error": abs(actual / reference - 1.0) if reference != 0.0 else float("nan"),
            "estimate": float(curve.finite_source_error_estimates[0]),
            "target": target,
            "refinement_level": int(curve.finite_source_refinement_levels[0]),
            "converged": bool(curve.finite_source_converged[0]),
            "method": (
                curve.finite_source_method_names[0]
                if hasattr(curve, "finite_source_method_names")
                else "unknown"
            ),
        }
        row.update(categories(row))
        row["accepted_bad"] = bool(row["converged"] and abs_error > 1.05 * target)
        rows.append(row)
        if index % 64 == 0:
            print(f"processed {index}/{len(point_rows)}")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"lcbinint={lcbinint.__file__}")
    print(f"cases={len(cases)} points={len(point_rows)} csv={csv_path}")
    print(f"source_bins={args.source_bins} max_bins={args.max_bins} reltol={args.reltol:g}")
    for key in ["ld_bin", "rho_bin", "mag_bin", "q_bin", "s_bin", "method"]:
        print(f"\nby {key}")
        print("group n win_rate ratio_med ratio_geo lc_ms_med vbb_ms_med rel_p95 accepted_bad unconverged")
        for row in summarize(rows, key):
            print(
                f"{row['group']:20s} {row['n']:5d} {row['win_rate']:8.3f} "
                f"{row['ratio_med']:9.3f} {row['ratio_geo']:9.3f} "
                f"{row['lc_ms_med']:9.4f} {row['vbb_ms_med']:10.4f} "
                f"{row['rel_p95']:9.2e} {row['accepted_bad']:12d} {row['unconverged']:11d}"
            )

    print("\nfastest lcbinint relative to vbm")
    for row in sorted(rows, key=lambda r: r["ratio"])[:args.top]:
        print(
            f"{row['ratio']:7.3f} lc={row['lc_ms']:.4f} vbb={row['vbb_ms']:.4f} "
            f"err={row['rel_error']:.2e} A={row['reference']:.2f} rho={row['rho']:.2e} "
            f"ld={int(row['ld'])} s={row['separation']:.3g} q={row['mass_ratio']:.2e} "
            f"{row['method']} {row['case']} t={row['time']:.5g} conv={int(row['converged'])}"
        )
    print("\naccepted_bad")
    bad = [row for row in rows if row["accepted_bad"]]
    for row in sorted(bad, key=lambda r: r["rel_error"], reverse=True)[:args.top]:
        print(
            f"err={row['rel_error']:.2e} abs={row['abs_error']:.2e} target={row['target']:.2e} "
            f"est={row['estimate']:.2e} A={row['reference']:.2f} rho={row['rho']:.2e} "
            f"ld={int(row['ld'])} s={row['separation']:.6g} q={row['mass_ratio']:.3e} "
            f"u0={row['u0']:.4g} alpha={row['alpha']:.4g} t={row['time']:.6g} "
            f"method={row['method']} level={row['refinement_level']} case={row['case']}"
        )
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

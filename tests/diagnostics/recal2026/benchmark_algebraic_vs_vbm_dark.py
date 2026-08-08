#!/usr/bin/env python3
"""Compare the experimental algebraic kernel with direct VBM integrators.

This is the algebraic counterpart of ``bench_grid_vs_vbm_dark.py``.  It uses
the same pure-grid-route corpus filter, four reference epochs per block, VBM
API, accuracy gate, and ratio definition.  The C++ batch executable measures
the experimental polar and Cartesian calls, including their internal seed,
support, band, and integration work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


TARGETS = (1.0e-2, 1.0e-3, 1.0e-4)
FACTORS = (0.0, 0.25, 0.5, 0.8, 0.95, 1.0, 1.1, 1.35, 1.7, 2.0)
REFERENCE_INDICES = (0, 7, 15, 23)
BLOCK_EPOCHS = 24
BLOCK_SPAN_IN_RADII = 0.4
VBM_ABSOLUTE_FLOOR = 1.0e-12


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_rows(directory):
    rows = []
    for path in sorted(Path(directory).glob("block-*.json")):
        rows.extend(json.loads(path.read_text()).get("rows", []))
    return rows


def pure_grid_route(row, target):
    entries = [
        entry for entry in row.get("engines", ())
        if entry.get("engine") == "lcbinint_auto"
        and abs(float(entry.get("knob", -1.0)) - target) < 1.0e-15
    ]
    if not entries:
        return False
    methods = set(entries[0].get("methods", ()))
    return (
        any(name.startswith("inverse_ray_") for name in methods)
        and not methods.intersection({
            "point_source", "hexadecapole", "source_plane_quadrature",
        })
    )


def usable(row, target):
    floor = row.get("reference_floor")
    if not finite(floor) or float(floor) > 0.1 * target:
        return False
    return all(
        str(index) in row.get("references", {})
        and row["references"][str(index)].get("status") == "ok"
        and finite(row["references"][str(index)].get("value"))
        for index in REFERENCE_INDICES
    )


def select_rows(rows, case_count, factors, seed):
    wanted = {round(float(value), 12) for value in factors}
    by_factor_case = {factor: {} for factor in wanted}
    for row in rows:
        factor = row.get("intended_distance_factor")
        if not finite(factor):
            continue
        factor = round(float(factor), 12)
        if factor not in wanted:
            continue
        by_factor_case[factor].setdefault(int(row["case_id"]), []).append(row)
    rng = np.random.default_rng(seed)
    selected = []
    for factor in sorted(by_factor_case):
        case_ids = sorted(by_factor_case[factor])
        if case_count < len(case_ids):
            indices = rng.choice(len(case_ids), case_count, replace=False)
            case_ids = [case_ids[int(index)] for index in indices]
        for case_id in sorted(case_ids):
            selected.extend(by_factor_case[factor][case_id])
    return selected


def references(row):
    return np.asarray([
        row["references"][str(index)]["value"]
        for index in REFERENCE_INDICES
    ], dtype=float)


def source_rows(row, job):
    rho = float(row["rho"])
    times = np.linspace(
        float(row["x"]) - 0.5 * BLOCK_SPAN_IN_RADII * rho,
        float(row["x"]) + 0.5 * BLOCK_SPAN_IN_RADII * rho,
        BLOCK_EPOCHS,
    )
    return [
        (
            job,
            epoch,
            float(row["s"]),
            float(row["q"]),
            -float(times[epoch]),
            float(row["y"]),
            rho,
        )
        for epoch in REFERENCE_INDICES
    ]


def run_algebraic(executable, rows, profile_c, target, repeats, timeout):
    lines = []
    for job, row in enumerate(rows):
        for values in source_rows(row, job):
            lines.append("\t".join(format(value, ".17g") for value in values))
    completed = subprocess.run(
        [str(executable), str(repeats), str(target), str(profile_c), "0.0"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=timeout if timeout > 0.0 else None,
        check=True,
    )
    records = {}
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 12:
            raise RuntimeError(f"malformed algebraic output: {line!r}")
        job, epoch, method = int(fields[0]), int(fields[1]), fields[2]
        records[(job, epoch, method)] = {
            "success": fields[3] == "1",
            "value": float(fields[4]),
            "seconds": float(fields[5]),
            "root_solves": int(fields[6]),
            "real_quartic": int(fields[7]),
            "real_fallbacks": int(fields[8]),
            "tangency_solves": int(fields[9]),
            "tangency_fallbacks": int(fields[10]),
            "unsafe": fields[11],
        }
    return records


def new_vbm(profile_c, reltol):
    import VBMicrolensing

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = VBM_ABSOLUTE_FLOOR
    vbm.RelTol = float(reltol)
    if profile_c:
        vbm.a1 = float(profile_c)
        vbm.a2 = 0.0
        vbm.SetLDprofile(vbm.LDlinear)
    return vbm


def vbm_one(vbm, row, x, profile_c):
    if profile_c:
        return float(vbm.BinaryMagDark(
            float(row["s"]), float(row["q"]), -float(x), float(row["y"]),
            float(row["rho"]), VBM_ABSOLUTE_FLOOR))
    return float(vbm.BinaryMag(
        float(row["s"]), float(row["q"]), -float(x), float(row["y"]),
        float(row["rho"]), VBM_ABSOLUTE_FLOOR))


def vbm_point(row, epoch, profile_c, target, repeats):
    rho = float(row["rho"])
    x = float(row["x"]) + (
        epoch / (BLOCK_EPOCHS - 1) - 0.5) * BLOCK_SPAN_IN_RADII * rho
    reference = float(row["references"][str(epoch)]["value"])
    pending = None
    reltol = float(target)
    # Match bench_grid_vs_vbm_dark.py exactly: the ladder stops at 1e-7.
    # Do not silently extend a target=1e-4 comparison to 1e-8 or 1e-9.
    candidates = []
    while reltol >= 1.0e-7 - 1.0e-15:
        candidates.append(reltol)
        reltol /= 10.0
    for reltol in candidates:
        vbm = new_vbm(profile_c, reltol)
        value = vbm_one(vbm, row, x, profile_c)
        error = abs(value - reference) / max(abs(reference), 1.0)
        if not finite(error) or error > target:
            continue
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            value = vbm_one(vbm, row, x, profile_c)
            samples.append(time.perf_counter() - started)
        return {
            "success": True,
            "value": value,
            "seconds": statistics.median(samples),
            "reltol": reltol,
            "error": abs(value - reference) / max(abs(reference), 1.0),
        }
    return {"success": False, "error": None}


def relative_error(value, reference):
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        os.environ.get("LCBININT_RECAL_CORPUS", "tests/diagnostics/results/recal2026/speed_discovery")))
    parser.add_argument("--executable", type=Path, default=Path(
        os.environ.get("LCBININT_ALGEBRAIC_BATCH", "build/benchmark_algebraic_batch")))
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--factors", type=float, nargs="+", default=list(FACTORS))
    parser.add_argument("--profiles", nargs="+", choices=("uniform", "linear"),
                        default=["uniform", "linear"])
    parser.add_argument("--q-min", type=float, default=1.0e-4)
    parser.add_argument("--d-max", type=float, default=2.01)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--targets", type=float, nargs="+", default=list(TARGETS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = select_rows(load_rows(args.input), args.case_count, args.factors, args.seed)
    rows = [row for row in rows
            if float(row["q"]) >= args.q_min
            and float(row["intended_distance_factor"]) < args.d_max]
    jobs = []
    for profile in args.profiles:
        profile_c = 0.5 if profile == "linear" else 0.0
        for target in args.targets:
            selected = [
                row for row in rows
                if row.get("profile") == profile
                and usable(row, target)
                and pure_grid_route(row, target)
            ]
            if not selected:
                continue
            records = run_algebraic(
                args.executable, selected, profile_c, target,
                args.repeats, args.timeout)
            for job, row in enumerate(selected):
                ref = references(row)
                for ref_pos, epoch in enumerate(REFERENCE_INDICES):
                    candidates = []
                    for method in ("polar", "cartesian"):
                        result = records.get((job, epoch, method), {})
                        if result.get("success") and relative_error(
                                result["value"], ref[ref_pos]) <= target:
                            candidates.append((result["seconds"], method, result))
                    algebraic = min(candidates, default=None)
                    vbm = vbm_point(row, epoch, profile_c, target, args.repeats)
                    ratio = None
                    if algebraic is not None and vbm.get("success"):
                        ratio = vbm["seconds"] / algebraic[0]
                    jobs.append({
                        "profile": profile,
                        "target": target,
                        "case_id": int(row["case_id"]),
                        "factor": float(row["intended_distance_factor"]),
                        "rho": float(row["rho"]),
                        "point_magnification": float(row["magnification"]),
                        "epoch": epoch,
                        "algebraic_method": None if algebraic is None else algebraic[1],
                        "algebraic_seconds": None if algebraic is None else algebraic[0],
                        "vbm_seconds": vbm.get("seconds"),
                        "vbm_reltol": vbm.get("reltol"),
                        "ratio_vbm_over_algebraic": ratio,
                        "algebraic_candidates": {
                            method: records.get((job, epoch, method))
                            for method in ("polar", "cartesian")
                        },
                    })

    summary = {}
    for profile in args.profiles:
        summary[profile] = {}
        for target in args.targets:
            subset = [row for row in jobs
                      if row["profile"] == profile and row["target"] == target]
            ratios = [row["ratio_vbm_over_algebraic"] for row in subset
                      if row["ratio_vbm_over_algebraic"] is not None]
            summary[profile][str(target)] = {
                "points": len(subset),
                "measured_points": len(ratios),
                "algebraic_wins": sum(value > 1.0 for value in ratios),
                "median_ratio_vbm_over_algebraic": (
                    None if not ratios else float(np.median(ratios))),
                "p10_ratio": None if not ratios else float(np.percentile(ratios, 10)),
                "p90_ratio": None if not ratios else float(np.percentile(ratios, 90)),
            }
    payload = {
        "input": str(args.input),
        "executable": str(args.executable),
        "case_count": args.case_count,
        "factors": list(args.factors),
        "profiles": args.profiles,
        "targets": list(args.targets),
        "repeats": args.repeats,
        "summary": summary,
        "points": jobs,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()

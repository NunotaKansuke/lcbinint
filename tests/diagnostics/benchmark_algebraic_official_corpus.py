#!/usr/bin/env python3
"""Compare the experimental algebraic kernels on the recal2026 speed corpus.

Unlike the small smoke benchmark, this uses the campaign's exact 24-epoch
blocks, reference epochs, frame convention, and delivered-accuracy rule.  The
default subset is restricted to the regime behind REPORT_master.md's speed
claim: linear limb darkening, d/rho < 0.95, A > 3, target accuracy 1e-4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.diagnostics.recal2026.speed_analysis import _usable, cheapest_meeting, load


CORPUS = Path(os.environ.get(
    "LCBININT_RECAL_CORPUS",
    ROOT / "tests/diagnostics/results/recal2026/speed_discovery"))
EXECUTABLE = ROOT / "build/benchmark_algebraic_boundary"
TARGET = 1.0e-4
VBM_RELATIVE_LADDER = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)
ABSOLUTE_FLOOR = 1.0e-12
EPOCHS = 24


def references(row):
    return {
        int(index): float(entry["value"])
        for index, entry in row["references"].items()
        if entry.get("status") == "ok" and math.isfinite(entry.get("value", math.nan))
    }


def worst_error(values, expected):
    return max(
        abs(float(values[index]) - value) / max(abs(value), 1.0)
        for index, value in expected.items()
    )


def select_rows(limit, profile="linear"):
    candidates = [
        row for row in load(CORPUS)
        if row.get("profile") == profile
        and row.get("intended_distance_factor", math.inf) < 0.95
        and row.get("magnification", 0.0) > 3.0
        and _usable(row, TARGET)
        and len(references(row)) > 0
    ]
    candidates.sort(key=lambda row: (
        math.log10(row["q"]), math.log10(row["rho"]),
        row["intended_distance_factor"], math.log10(row["magnification"])))
    if not limit or limit >= len(candidates):
        return candidates
    indices = np.linspace(0, len(candidates) - 1, limit).round().astype(int)
    return [candidates[index] for index in sorted(set(indices))]


def algebraic_ladder(row, repeats):
    command = [
        str(EXECUTABLE), str(repeats), "block",
        repr(row["s"]), repr(row["q"]), repr(-row["x"]), repr(row["y"]),
        repr(row["rho"]), repr(row["limb_darkening_c"]), "0.0",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    entries = []
    for item in csv.DictReader(completed.stdout.splitlines()):
        entries.append({
            "knob": float(item["tolerance"]),
            "polar_seconds": float(item["polar_seconds_per_epoch"]),
            "cart_seconds": float(item["cart_seconds_per_epoch"]),
            "polar_success": int(item["polar_success_epochs"]) == EPOCHS,
            "cart_success": int(item["cart_success_epochs"]) == EPOCHS,
            "polar_values": [float(value) for value in item["polar_values"].split(";")],
            "cart_values": [float(value) for value in item["cart_values"].split(";")],
        })
    return entries


def vbm_ladder(row, repeats):
    """VBMicrolensing scalar magnification calls at the same 24 positions.

    In VBMicrolensing 5.x ``a1`` and the profile live on the solver object;
    the sixth ``BinaryMagDark`` argument is the absolute accuracy, not the
    limb-darkening coefficient.  Its source frame is mirrored in both axes
    relative to the corpus trajectory parameters returned by lcbinint.
    """
    import VBMicrolensing

    span = 0.4 * row["rho"]
    source_x = -np.linspace(
        row["x"] - 0.5 * span, row["x"] + 0.5 * span, EPOCHS)
    source_y = -row["y"]
    entries = []
    for reltol in VBM_RELATIVE_LADDER:
        solver = VBMicrolensing.VBMicrolensing()
        solver.Tol = ABSOLUTE_FLOOR
        solver.RelTol = reltol
        profile_c = row["limb_darkening_c"]
        if profile_c:
            solver.a1 = profile_c
            solver.SetLDprofile(solver.LDlinear)

        def evaluate():
            if profile_c:
                return [solver.BinaryMagDark(
                    row["s"], row["q"], x, source_y, row["rho"],
                    ABSOLUTE_FLOOR) for x in source_x]
            return [solver.BinaryMag2(
                row["s"], row["q"], x, source_y, row["rho"])
                for x in source_x]

        values = evaluate()
        samples = []
        for _ in range(max(repeats, 1)):
            started = time.perf_counter()
            values = evaluate()
            samples.append((time.perf_counter() - started) / EPOCHS)
        entries.append({
            "knob": reltol,
            "seconds": float(np.median(samples)),
            "values": np.asarray(values, dtype=float),
        })
    return entries


def cheapest(entries, seconds_key, error_key, success_key=None):
    valid = [entry for entry in entries
             if (success_key is None or entry[success_key])
             and entry[error_key] <= TARGET]
    return min(valid, key=lambda entry: entry[seconds_key]) if valid else None


def percentile(values, probability):
    return float(np.percentile(values, 100.0 * probability)) if values else math.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=32,
                        help="evenly stratified rows; 0 means all qualifying rows")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--profile", choices=("uniform", "linear"), default="linear")
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.core})

    rows = select_rows(args.limit, args.profile)
    ratios = []
    baseline_ratios = []
    misses = {"vbm": 0, "polar": 0, "cart": 0}
    print("index,case_id,s,q,rho,d_over_rho,A,vbm_seconds,polar_seconds,"
          "cart_seconds,vbm_over_polar,vbm_over_cart,baseline_vbm_over_old_grid,"
          "vbm_knob,polar_knob,cart_knob")
    for index, row in enumerate(rows, 1):
        expected = references(row)
        algebraic = algebraic_ladder(row, args.repeat)
        for entry in algebraic:
            entry["polar_error"] = worst_error(entry["polar_values"], expected)
            entry["cart_error"] = worst_error(entry["cart_values"], expected)
        vbm = vbm_ladder(row, args.repeat)
        for entry in vbm:
            entry["error"] = worst_error(entry["values"], expected)
        best_vbm = cheapest(vbm, "seconds", "error")
        best_polar = cheapest(algebraic, "polar_seconds", "polar_error", "polar_success")
        best_cart = cheapest(algebraic, "cart_seconds", "cart_error", "cart_success")
        misses["vbm"] += best_vbm is None
        misses["polar"] += best_polar is None
        misses["cart"] += best_cart is None
        polar_ratio = (best_vbm["seconds"] / best_polar["polar_seconds"]
                       if best_vbm and best_polar else math.nan)
        cart_ratio = (best_vbm["seconds"] / best_cart["cart_seconds"]
                      if best_vbm and best_cart else math.nan)
        if math.isfinite(polar_ratio):
            ratios.append(polar_ratio)
        old_vbm = cheapest_meeting(row, "vbm", TARGET)
        old_cart = cheapest_meeting(row, "lcbinint_cartesian", TARGET)
        old_polar = cheapest_meeting(row, "lcbinint_polar", TARGET)
        old_grid = min(
            (entry for entry in (old_cart, old_polar) if entry is not None),
            default=None,
        )
        baseline_ratio = (old_vbm[0] / old_grid[0]
                          if old_vbm and old_grid else math.nan)
        if math.isfinite(baseline_ratio):
            baseline_ratios.append(baseline_ratio)
        print(
            f"{index},{row['case_id']},{row['s']:.12g},{row['q']:.12g},"
            f"{row['rho']:.12g},{row['intended_distance_factor']:.12g},"
            f"{row['magnification']:.12g},"
            f"{best_vbm['seconds'] if best_vbm else math.nan:.12g},"
            f"{best_polar['polar_seconds'] if best_polar else math.nan:.12g},"
            f"{best_cart['cart_seconds'] if best_cart else math.nan:.12g},"
            f"{polar_ratio:.12g},{cart_ratio:.12g},{baseline_ratio:.12g},"
            f"{best_vbm['knob'] if best_vbm else math.nan:.12g},"
            f"{best_polar['knob'] if best_polar else math.nan:.12g},"
            f"{best_cart['knob'] if best_cart else math.nan:.12g}",
            flush=True,
        )
    print(
        f"# summary,rows={len(rows)},vbm_missed={misses['vbm']},"
        f"polar_missed={misses['polar']},cart_missed={misses['cart']},"
        f"polar_wins={sum(ratio > 1.0 for ratio in ratios)},"
        f"median_vbm_over_polar={statistics.median(ratios) if ratios else math.nan:.12g},"
        f"p10={percentile(ratios, 0.1):.12g},p90={percentile(ratios, 0.9):.12g},"
        f"baseline_old_grid_wins={sum(ratio > 1.0 for ratio in baseline_ratios)},"
        f"baseline_median_vbm_over_old_grid="
        f"{statistics.median(baseline_ratios) if baseline_ratios else math.nan:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

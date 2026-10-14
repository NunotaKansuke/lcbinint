#!/usr/bin/env python3
"""Summarize paired tolerance-probe runs by profile and reference magnification."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw.tsv"
JOINED = ROOT / "joined_v2_vbm.tsv"
SUMMARY = HERE / "summary.tsv"


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t"))


def quantile(values, fraction):
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return float("nan")
    position = (len(values) - 1) * fraction
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return values[lo]
    weight = position - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def mag_bin(value):
    if value < 1.1:
        return "A_lt_1p1"
    if value < 1.5:
        return "A_1p1_to_1p5"
    if value < 5.0:
        return "A_1p5_to_5"
    return "A_ge_5"


def key(row):
    return (row["case_id"], row["profile"], row["d_bin_index"], row["epoch_index"])


def f(row, column):
    try:
        value = float(row[column])
    except (KeyError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def main():
    raw = read(RAW)
    joined = read(JOINED)
    vbm_by_key = {
        (row["case_id"], row["profile"], row["d_bin_index"], row["epoch_index"]): row
        for row in joined if abs(float(row["target"]) - 1e-3) < 1e-15
    }

    per_row = defaultdict(list)
    for row in raw:
        gkey = (*key(row), row["rtol"])
        per_row[gkey].append(row)

    cells = defaultdict(list)
    for (case, profile, d_bin, epoch, rtol), reps in per_row.items():
        first = reps[0]
        item = {
            "case_id": case,
            "profile": profile,
            "d_bin_index": d_bin,
            "epoch_index": epoch,
            "rtol": rtol,
            "A_ref": f(first, "reference_1e-6"),
            "seed_ms": quantile([f(row, "seed_ms") for row in reps], 0.5),
            "frontier_ms": quantile([f(row, "frontier_ms") for row in reps], 0.5),
            "full_ms": quantile([f(row, "full_ms") for row in reps], 0.5),
            "planning_ms": quantile([f(row, "planning_ms") for row in reps], 0.5),
            "integration_ms": quantile([f(row, "integration_ms") for row in reps], 0.5),
            "relerr": quantile([f(row, "relative_error_reference") for row in reps], 0.5),
            "esterr": quantile([f(row, "estimated_abs_error") for row in reps], 0.5),
            "nodes": quantile([f(row, "nodes") for row in reps], 0.5),
            "converged": all(row["value_converged"] == "1" for row in reps),
        }
        baseline = vbm_by_key[(case, profile, d_bin, epoch)]
        item["vbm_ms"] = f(baseline, "vbm_ms")
        for bucket in ("ALL", mag_bin(item["A_ref"])):
            cells[(profile, bucket, rtol)].append(item)

    fields = [
        "profile", "A_group", "rtol", "n", "converged_fraction",
        "vbm_ms_p50", "seed_ms_p50", "planning_ms_p50", "integration_ms_p50",
        "frontier_ms_p50", "full_ms_p50", "vbm_over_full_p50",
        "relative_error_p50", "relative_error_p95", "relative_error_max",
        "fraction_error_gt_1e-3", "fraction_error_gt_rtol",
        "estimated_abs_error_p50", "nodes_p50",
    ]
    with SUMMARY.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for (profile, bucket, rtol), group in sorted(cells.items()):
            errors = [item["relerr"] for item in group]
            row = {
                "profile": profile,
                "A_group": bucket,
                "rtol": rtol,
                "n": len(group),
                "converged_fraction": sum(item["converged"] for item in group) / len(group),
                "vbm_ms_p50": quantile([item["vbm_ms"] for item in group], 0.5),
                "seed_ms_p50": quantile([item["seed_ms"] for item in group], 0.5),
                "planning_ms_p50": quantile([item["planning_ms"] for item in group], 0.5),
                "integration_ms_p50": quantile([item["integration_ms"] for item in group], 0.5),
                "frontier_ms_p50": quantile([item["frontier_ms"] for item in group], 0.5),
                "full_ms_p50": quantile([item["full_ms"] for item in group], 0.5),
                "vbm_over_full_p50": quantile([
                    item["vbm_ms"] / item["full_ms"]
                    for item in group if item["full_ms"] > 0 and math.isfinite(item["vbm_ms"])
                ], 0.5),
                "relative_error_p50": quantile(errors, 0.5),
                "relative_error_p95": quantile(errors, 0.95),
                "relative_error_max": max((value for value in errors if math.isfinite(value)), default=float("nan")),
                "fraction_error_gt_1e-3": sum(value > 1e-3 for value in errors if math.isfinite(value)) / len(group),
                "fraction_error_gt_rtol": sum(value > float(rtol) for value in errors if math.isfinite(value)) / len(group),
                "estimated_abs_error_p50": quantile([item["esterr"] for item in group], 0.5),
                "nodes_p50": quantile([item["nodes"] for item in group], 0.5),
            }
            writer.writerow(row)
    print(f"wrote {SUMMARY} ({len(cells)} profile/group/tolerance cells)")


if __name__ == "__main__":
    main()

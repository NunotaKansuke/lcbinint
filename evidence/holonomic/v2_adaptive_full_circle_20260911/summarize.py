#!/usr/bin/env python3
"""Summarize the adaptive full-circle coverage rerun."""

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    x = (len(values) - 1) * p / 100.0
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return float(values[lo] + (values[hi] - values[lo]) * (x - lo))


def read_rows(path):
    with Path(path).open() as stream:
        next(stream)  # provenance comment
        header = next(stream).lstrip("# ").split()
        return [dict(zip(header, line.split())) for line in stream if line.strip()]


def condition_summary(rows):
    out = {}
    for profile in sorted({row["profile"] for row in rows}):
        for target in sorted({row["target"] for row in rows}):
            group = [row for row in rows
                     if row["profile"] == profile and row["target"] == target]
            errors = [abs(float(row["v2_mu"]) - float(row["reference"])) /
                      abs(float(row["reference"])) for row in group]
            times = [float(row["ms"]) for row in group]
            out[f"{profile}:target={target}"] = {
                "rows": len(group),
                "value_converged": sum(row["value_converged"] == "1"
                                        for row in group),
                "numerical_status": dict(Counter(row["numerical_status"]
                                                   for row in group)),
                "stop": dict(Counter(row["stop"] for row in group)),
                "relative_error_vs_input_reference": {
                    "p50": percentile(errors, 50),
                    "p95": percentile(errors, 95),
                    "p99": percentile(errors, 99),
                    "max": percentile(errors, 100),
                },
                "whole_ms": {
                    "p50": percentile(times, 50),
                    "p95": percentile(times, 95),
                    "p99": percentile(times, 99),
                    "max": percentile(times, 100),
                },
                "stage_ms_p50": {
                    key: statistics.median(float(row[key]) for row in group)
                    for key in ("setup_ms", "physics_ms", "estimator_ms",
                                "scheduler_ms", "setup_event_ms")
                },
            }
    return out


def diagnostic_summary(path):
    result = {"epochs": [], "sample_rejections": []}
    epoch_path = Path(path) / "epochs.csv"
    if epoch_path.exists():
        with epoch_path.open(newline="") as stream:
            result["epochs"] = list(csv.DictReader(stream))
    reason_path = Path(path) / "sample_reason_counts.csv"
    if reason_path.exists():
        with reason_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["count"]) or int(row["cold_count"]):
                    result["sample_rejections"].append(row)
    return result


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: summarize.py RESULTS CASE9_DIR CASE92_DIR")
    rows = read_rows(sys.argv[1])
    if len(rows) != 14432:
        raise RuntimeError(f"expected 14432 rows, got {len(rows)}")
    summary = {
        "rows": len(rows),
        "coverage": {
            "value_converged": sum(row["value_converged"] == "1"
                                    for row in rows),
            "numerical_status_ok": sum(row["numerical_status"] == "OK"
                                        for row in rows),
            "stop": dict(Counter(row["stop"] for row in rows)),
        },
        "conditions": condition_summary(rows),
        "hard_case_diagnostics": {
            "case9": diagnostic_summary(sys.argv[2]),
            "case92": diagnostic_summary(sys.argv[3]),
        },
    }
    destination = Path(sys.argv[1]).with_name("summary.json")
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()

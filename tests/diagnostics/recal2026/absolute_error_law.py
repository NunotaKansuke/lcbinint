#!/usr/bin/env python3
"""Calibrate an absolute-tolerance-only resolution law from stored ladders.

The resulting law is one branch of the common ``max``-budget policy.  It is
intentionally retained even though it is conservative: a caller that supplies
``reltol=0`` still needs a reproducible absolute-only initial resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from . import analysis, fit_rules
from .engines import BUCKETS
from .error_budget_law import _required, _score


GRIDS = ("cartesian", "polar")
ABSOLUTE_LEVELS = (
    1.0e-2, 5.0e-3, 3.0e-3, 2.0e-3,
    1.0e-3, 5.0e-4, 3.0e-4, 2.0e-4, 1.0e-4,
)
B0 = 1.0e-3
TARGET_COVERAGE = 0.99


def _records(rows, dataset_name):
    records = []
    for row_index, row in enumerate(rows):
        reference = row.get("reference") or {}
        try:
            reference_value = float(reference["value"])
        except (KeyError, TypeError, ValueError):
            continue
        scale = max(abs(reference_value), 1.0)
        try:
            point = abs(float(row.get("point_magnification")))
        except (TypeError, ValueError):
            point = 1.0
        for atol in ABSOLUTE_LEVELS:
            for grid in GRIDS:
                required = _required(row, grid, atol, 0.0)
                if required is None:
                    continue
                records.append({
                    "dataset": dataset_name,
                    "row_index": row_index,
                    "case_id": row.get("case_id"),
                    "grid": grid,
                    "absolute_tolerance": atol,
                    "budget": atol,
                    "reference": reference_value,
                    "reference_scale": scale,
                    "point_magnification": point,
                    "required_resolution": required,
                })
    return records


def _write_records(path, records):
    fields = list(records[0]) if records else []
    with path.open("w", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _predict(model, records):
    budget = np.asarray([row["budget"] for row in records], dtype=float)
    raw = np.exp2(
        model["intercept"]
        + model["slope"] * np.log2(B0 / budget)
        + model.get("safety_offset", 0.0)
    )
    return np.asarray([
        fit_rules._round_up(float(value)) for value in raw
    ], dtype=float)


def _fit_grid(discovery, holdout, grid):
    discovery = [row for row in discovery if row["grid"] == grid]
    holdout = [row for row in holdout if row["grid"] == grid]
    groups = []
    for atol in ABSOLUTE_LEVELS:
        values = np.asarray([
            row["required_resolution"] for row in discovery
            if row["absolute_tolerance"] == atol
        ], dtype=float)
        if values.size == 0:
            continue
        groups.append({
            "absolute_tolerance": atol,
            "rows": int(values.size),
            "required_p99": float(np.percentile(values, 99.0)),
        })
    x = np.asarray([
        math.log2(B0 / group["absolute_tolerance"])
        for group in groups
    ])
    y = np.log2(np.asarray([group["required_p99"] for group in groups]))
    slope, intercept = np.polyfit(x, y, 1)
    model = {
        "intercept": float(intercept),
        "slope": float(slope),
        "safety_offset": 0.0,
        "fit_method": "discovery_p99_by_absolute_tolerance_then_OLS",
        "baseline_absolute_tolerance": B0,
        "groups": groups,
    }

    # Require 99% coverage both overall and at every available atol level.
    for offset in np.linspace(0.0, 2.0, 401):
        model["safety_offset"] = float(offset)
        prediction = _predict(model, discovery)
        overall = _score(prediction, discovery)
        level_coverages = []
        for atol in ABSOLUTE_LEVELS:
            subset = [row for row in discovery
                      if row["absolute_tolerance"] == atol]
            if subset:
                level_coverages.append(_score(_predict(model, subset), subset)["coverage"])
        if (overall["coverage"] >= TARGET_COVERAGE
                and min(level_coverages, default=0.0) >= TARGET_COVERAGE):
            break

    discovery_score = _score(_predict(model, discovery), discovery)
    holdout_score = _score(_predict(model, holdout), holdout)
    per_level = {}
    for atol in ABSOLUTE_LEVELS:
        subset = [row for row in holdout
                  if row["absolute_tolerance"] == atol]
        if subset:
            per_level[str(atol)] = _score(_predict(model, subset), subset)
    return {
        "model": model,
        "effective_C": float(2.0 ** (intercept + model["safety_offset"])),
        "discovery": discovery_score,
        "holdout": holdout_score,
        "holdout_by_absolute_tolerance": per_level,
        "discovery_rows": len(discovery),
        "holdout_rows": len(holdout),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    discovery = _records(analysis.load(args.discovery), "discovery")
    holdout = _records(analysis.load(args.holdout), "holdout")
    _write_records(output / "absolute_error_records.csv", discovery + holdout)
    report = {
        "formula": {
            "target": "log2(N_required)",
            "budget": "absolute_tolerance",
            "baseline_absolute_tolerance": B0,
            "coverage_quantile": TARGET_COVERAGE,
            "absolute_levels": list(ABSOLUTE_LEVELS),
            "relative_tolerance": 0.0,
        },
        "inputs": {
            "discovery": str(args.discovery),
            "holdout": str(args.holdout),
            "usable_discovery_records": len(discovery),
            "usable_holdout_records": len(holdout),
        },
        "grids": {},
    }
    for grid in GRIDS:
        report["grids"][grid] = _fit_grid(discovery, holdout, grid)
    (output / "absolute_error_law.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(output),
        "files": sorted(path.name for path in output.iterdir()),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Calibrate an absolute-tolerance-only resolution law from stored ladders.

The resulting scalar law is the reference-certified branch of the common
``max``-budget policy.  Reference-limited rows are retained as lower-censored
observations, so the campaign does not establish a population-wide absolute
p99 at the tightest targets.  The Apoint-dependent candidate is recorded as a
diagnostic rather than silently promoted to a production selector.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np

from . import analysis, fit_rules
from .engines import BUCKETS
from .error_budget_law import (
    _fit_quantile_linear_light,
    _required_outcome,
    _score,
)


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
                outcome = _required_outcome(row, grid, atol, 0.0)
                if outcome["status"] == "invalid":
                    continue
                censored = outcome["status"] != "observed"
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
                    # For a censored row this is a lower bound, not an exact
                    # measurement.  Keep the flag and reason beside it.
                    "required_resolution": (
                        outcome["required"] if not censored
                        else outcome["lower_bound"]),
                    "required_lower_bound": outcome["lower_bound"],
                    "censored": censored,
                    "censor_reason": outcome["reason"],
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


def _apoint_features(records):
    return np.asarray([
        [
            math.log2(B0 / row["absolute_tolerance"]),
            math.log2(max(float(row["point_magnification"]), 1.0)),
        ]
        for row in records
    ], dtype=float)


def _predict_apoint(model, records):
    return fit_rules.predict_linear(model, _apoint_features(records))


def _fit_apoint_grid(discovery, holdout, grid):
    """Fit the diagnostic two-feature law on certified rows only.

    This is deliberately a diagnostic candidate, not the shipped selector:
    the point-magnification feature can explain conditional work, but it
    cannot turn a reference-limited lower bound into an exact observation.
    """
    discovery_all = [row for row in discovery if row["grid"] == grid]
    holdout_all = [row for row in holdout if row["grid"] == grid]
    discovery = [row for row in discovery_all if not row["censored"]]
    holdout = [row for row in holdout_all if not row["censored"]]
    target = np.log2(np.asarray(
        [row["required_resolution"] for row in discovery], dtype=float))
    base = _fit_quantile_linear_light(
        _apoint_features(discovery), target, TARGET_COVERAGE)
    selected = None
    for offset in np.linspace(0.0, 2.0, 401):
        candidate = copy.deepcopy(base)
        candidate["beta"][0] += float(offset)
        prediction = _predict_apoint(candidate, discovery)
        level_coverages = []
        for atol in ABSOLUTE_LEVELS:
            subset = [row for row in discovery
                      if row["absolute_tolerance"] == atol]
            if subset:
                level_coverages.append(_score(
                    _predict_apoint(candidate, subset), subset)["coverage"])
        if (_score(prediction, discovery)["coverage"] >= TARGET_COVERAGE
                and min(level_coverages, default=0.0) >= TARGET_COVERAGE):
            selected = candidate
            break
    if selected is None:
        selected = base

    mean = np.asarray(selected["mean"], dtype=float)
    std = np.asarray(selected["std"], dtype=float)
    beta = np.asarray(selected["beta"], dtype=float)
    coefficients = beta[1:] / std
    intercept = beta[0] - float(np.sum(coefficients * mean))
    return {
        "formula": (
            "log2(N99) = intercept + beta_atol*log2(1e-3/atol) "
            "+ beta_Apoint*log2(max(Apoint,1))"),
        "features": ["log2(1e-3/atol)", "log2(max(Apoint,1))"],
        "raw_intercept": float(intercept),
        "effective_C_at_Apoint_1": float(2.0 ** intercept),
        "raw_coefficients": {
            "log2(1e-3/atol)": float(coefficients[0]),
            "log2(max(Apoint,1))": float(coefficients[1]),
        },
        "model": selected,
        "discovery": _score(_predict_apoint(selected, discovery), discovery),
        "holdout": _score(_predict_apoint(selected, holdout), holdout),
        "discovery_lower_bound": _score(
            _predict_apoint(selected, discovery_all), discovery_all),
        "holdout_lower_bound": _score(
            _predict_apoint(selected, holdout_all), holdout_all),
        "discovery_rows": len(discovery),
        "holdout_rows": len(holdout),
        "discovery_censored_rows": len(discovery_all) - len(discovery),
        "holdout_censored_rows": len(holdout_all) - len(holdout),
        "scope": "reference-certified rows only",
    }


def _fit_grid(discovery, holdout, grid):
    discovery = [row for row in discovery if row["grid"] == grid]
    holdout = [row for row in holdout if row["grid"] == grid]
    discovery_exact = [row for row in discovery if not row["censored"]]
    holdout_exact = [row for row in holdout if not row["censored"]]
    groups = []
    lower_bound_groups = []
    for atol in ABSOLUTE_LEVELS:
        values = np.asarray([
            row["required_resolution"] for row in discovery
            if row["absolute_tolerance"] == atol
        ], dtype=float)
        exact_values = np.asarray([
            row["required_resolution"] for row in discovery_exact
            if row["absolute_tolerance"] == atol
        ], dtype=float)
        if values.size == 0 or exact_values.size == 0:
            continue
        groups.append({
            "absolute_tolerance": atol,
            "rows": int(exact_values.size),
            "required_p99": float(np.percentile(exact_values, 99.0)),
        })
        lower_bound_groups.append({
            "absolute_tolerance": atol,
            "rows": int(values.size),
            "observed_rows": int(exact_values.size),
            "censored_rows": int(values.size - exact_values.size),
            "censored_fraction": float(
                (values.size - exact_values.size) / values.size),
            "required_p50_lower_bound": float(np.percentile(values, 50.0)),
            "required_p99_lower_bound": float(np.percentile(values, 99.0)),
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

    # The power law is fitted only to exact, reference-certified observations.
    # Lower-censored observations cannot identify the true p99 beyond the
    # finest ladder bucket; they are retained below as explicit constraints.
    for offset in np.linspace(0.0, 2.0, 401):
        model["safety_offset"] = float(offset)
        prediction = _predict(model, discovery_exact)
        overall = _score(prediction, discovery_exact)
        level_coverages = []
        for atol in ABSOLUTE_LEVELS:
            subset = [row for row in discovery_exact
                      if row["absolute_tolerance"] == atol]
            if subset:
                level_coverages.append(_score(_predict(model, subset), subset)["coverage"])
        if (overall["coverage"] >= TARGET_COVERAGE
                and min(level_coverages, default=0.0) >= TARGET_COVERAGE):
            break

    discovery_score = _score(_predict(model, discovery_exact), discovery_exact)
    holdout_score = _score(_predict(model, holdout_exact), holdout_exact)
    discovery_lower_bound_score = _score(_predict(model, discovery), discovery)
    holdout_lower_bound_score = _score(_predict(model, holdout), holdout)
    per_level = {}
    per_level_lower_bound = {}
    for atol in ABSOLUTE_LEVELS:
        subset = [row for row in holdout_exact
                  if row["absolute_tolerance"] == atol]
        if subset:
            per_level[str(atol)] = _score(_predict(model, subset), subset)
        lower_subset = [row for row in holdout
                        if row["absolute_tolerance"] == atol]
        if lower_subset:
            per_level_lower_bound[str(atol)] = _score(
                _predict(model, lower_subset), lower_subset)
    lower_bound_identifiable = all(
        group["censored_fraction"] < 1.0 - TARGET_COVERAGE
        for group in lower_bound_groups
    )
    return {
        "model": model,
        "effective_C": float(2.0 ** (intercept + model["safety_offset"])),
        "discovery": discovery_score,
        "holdout": holdout_score,
        "discovery_lower_bound": discovery_lower_bound_score,
        "holdout_lower_bound": holdout_lower_bound_score,
        "holdout_by_absolute_tolerance": per_level,
        "holdout_lower_bound_by_absolute_tolerance": per_level_lower_bound,
        "discovery_rows": len(discovery),
        "holdout_rows": len(holdout),
        "discovery_exact_rows": len(discovery_exact),
        "holdout_exact_rows": len(holdout_exact),
        "discovery_censored_rows": len(discovery) - len(discovery_exact),
        "holdout_censored_rows": len(holdout) - len(holdout_exact),
        "lower_bound_groups": lower_bound_groups,
        "lower_bound_p99_identifiable_at_99pct": lower_bound_identifiable,
        "interpretation": (
            "conditional_exact_fit_only"
            if not lower_bound_identifiable else "population_fit"),
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
            "censoring": (
                "Rows whose reference uncertainty exceeds 10% of the budget "
                "are retained as lower-censored at the finest ladder bucket; "
                "the fitted power law uses exact rows only and reports whether "
                "the population p99 is identifiable."),
        },
        "inputs": {
            "discovery": str(args.discovery),
            "holdout": str(args.holdout),
            "records_discovery": len(discovery),
            "records_holdout": len(holdout),
        },
        "grids": {},
    }
    for grid in GRIDS:
        report["grids"][grid] = _fit_grid(discovery, holdout, grid)
        report["grids"][grid]["apoint_diagnostic"] = _fit_apoint_grid(
            discovery, holdout, grid)
    (output / "absolute_error_law.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(output),
        "files": sorted(path.name for path in output.iterdir()),
    }, indent=2))


if __name__ == "__main__":
    main()

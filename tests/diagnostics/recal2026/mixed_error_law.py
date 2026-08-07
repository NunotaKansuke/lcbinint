#!/usr/bin/env python3
"""Validate the max-budget combination of the absolute and relative laws.

The two one-dimensional laws are fitted on discovery data.  This script does
not refit them on the mixed cases.  It applies

    N_mix = min(N_abs(atol), N_rel(reltol))

to every pair of tolerances and scores the result against the same independent
holdout ladder.  The output is a compact, paper-friendly record containing the
coverage matrix and the checks that ``max`` gives the same required bucket as
the less-demanding pure branch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import analysis, fit_rules
from .empirical_law import ABSOLUTE_LEVELS, B0, RELATIVE_LEVELS
from .error_budget_law import GRIDS, _required


TARGET_COVERAGE = 0.99


def _model_prediction(model, epsilon):
    raw = 2.0 ** (
        model["intercept"]
        + model["slope"] * math.log2(B0 / epsilon)
        + model.get("safety_offset", 0.0)
    )
    return fit_rules._round_up(raw)


def _models(relative_report, absolute_report, grid):
    relative = relative_report["grids"][grid].get("selected")
    if relative is None:
        relative = relative_report["grids"][grid]["budget_power"]
    return relative["model"], absolute_report["grids"][grid]["model"]


def _score(predicted, required):
    predicted = np.asarray(predicted, dtype=float)
    required = np.asarray(required, dtype=float)
    if required.size == 0:
        return {"rows": 0, "coverage": None}
    return {
        "rows": int(required.size),
        "coverage": float(np.mean(predicted >= required)),
        "median_required": float(np.median(required)),
        "median_predicted": float(np.median(predicted)),
        "p90_predicted": float(np.percentile(predicted, 90.0)),
        "p99_predicted": float(np.percentile(predicted, 99.0)),
        "median_work_vs_required": float(
            np.median((predicted / required) ** 2)),
        "worst_prediction_to_required": float(
            np.min(predicted / required)),
    }


def _branch_prediction(row, grid, atol, reltol, branch_models):
    reference = row["reference"]["value"]
    scale = max(abs(float(reference)), 1.0)
    values = []
    if atol > 0.0:
        # The absolute law is calibrated in dimensional atol.  Only the
        # relative branch uses the magnification-normalized scale.
        values.append(_model_prediction(branch_models[1], atol))
    if reltol > 0.0:
        values.append(_model_prediction(branch_models[0], reltol))
    if not values:
        raise ValueError("at least one tolerance must be positive")
    return min(values)


def _evaluate(rows, dataset_name, relative_report, absolute_report):
    output = {}
    for grid in GRIDS:
        branch_models = _models(relative_report, absolute_report, grid)
        pairs = []
        identity_mismatches = 0
        identity_cases_checked = 0
        for atol in ABSOLUTE_LEVELS:
            for reltol in RELATIVE_LEVELS:
                required = []
                predicted = []
                for row in rows:
                    mixed = _required(row, grid, atol, reltol)
                    if mixed is None:
                        continue
                    absolute = _required(row, grid, atol, 0.0)
                    relative = _required(row, grid, 0.0, reltol)
                    if absolute is not None and relative is not None:
                        identity_cases_checked += 1
                        if mixed != min(absolute, relative):
                            identity_mismatches += 1
                    required.append(mixed)
                    predicted.append(_branch_prediction(
                        row, grid, atol, reltol, branch_models))
                score = _score(predicted, required)
                pairs.append({
                    "absolute_tolerance": atol,
                    "relative_tolerance": reltol,
                    **score,
                })
        matrix = []
        for atol in ABSOLUTE_LEVELS:
            matrix.append([
                next(item["coverage"] for item in pairs
                     if item["absolute_tolerance"] == atol
                     and item["relative_tolerance"] == reltol)
                for reltol in RELATIVE_LEVELS
            ])
        coverages = np.asarray([
            item["coverage"] for item in pairs if item["coverage"] is not None
        ])
        output[grid] = {
            "pairs": pairs,
            "coverage_matrix": matrix,
            "summary": {
                "pairs": len(pairs),
                "minimum_coverage": float(np.min(coverages)),
                "median_coverage": float(np.median(coverages)),
                "pairs_meeting_target": int(np.sum(coverages >= TARGET_COVERAGE)),
                "target_coverage": TARGET_COVERAGE,
                "identity_mismatches": identity_mismatches,
                "identity_cases_checked": identity_cases_checked,
            },
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--relative-report", required=True)
    parser.add_argument("--absolute-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    relative_report = json.loads(Path(args.relative_report).read_text())
    absolute_report = json.loads(Path(args.absolute_report).read_text())
    discovery_rows = analysis.load(args.discovery)
    holdout_rows = analysis.load(args.holdout)
    report = {
        "policy": {
            "dimensional_budget":
                "max(atol, reltol*max(abs(A_reference), 1))",
            "normalized_budget":
                "max(atol/max(abs(A_reference), 1), reltol)",
            "resolution_selection": "min(N_absolute, N_relative)",
            "target_coverage": TARGET_COVERAGE,
        },
        "inputs": {
            "discovery": str(args.discovery),
            "holdout": str(args.holdout),
            "discovery_rows": len(discovery_rows),
            "holdout_rows": len(holdout_rows),
            "absolute_levels": list(ABSOLUTE_LEVELS),
            "relative_levels": list(RELATIVE_LEVELS),
        },
        "discovery": _evaluate(
            discovery_rows, "discovery", relative_report, absolute_report),
        "holdout": _evaluate(
            holdout_rows, "holdout", relative_report, absolute_report),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "holdout": {
            grid: report["holdout"][grid]["summary"] for grid in GRIDS
        },
    }, indent=2))


if __name__ == "__main__":
    main()

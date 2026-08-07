#!/usr/bin/env python3
"""Fit a continuous empirical resolution law in error-budget units.

This is the paper-facing calibration layer.  It deliberately does not call
the runtime selector and does not modify C++ code.  It reuses the stored
``A(N)`` resolution ladders to ask, for many intermediate budgets, which
resolution was actually required.  The fitted law uses the dimensionless
budget ``epsilon = B / max(|A_ref|, 1)`` so it is comparable between
magnification levels.  With ``atol=0``, epsilon is exactly ``reltol``:

    log2(N_required) = alpha
        + beta log2(epsilon0 / epsilon).

where ``B = max(atol, reltol*max(|A_ref|, 1))``.  The two tolerances are
alternative allowances: passing either one is sufficient.  The proposed ``A_point`` term is
also fitted as a diagnostic candidate, but is retained only if it improves
holdout coverage at lower work.  All fits are judged on an independent
holdout.

The current campaign supplies relative-tolerance ladders.  The default fit
uses ``atol=0``; a second invocation with ``atol=1e-4`` checks the same
normalized-budget construction for mixed absolute/relative requests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import analysis, fit_rules
from .engines import BUCKETS


GRIDS = ("cartesian", "polar")
RELATIVE_LEVELS = (
    1.0e-2, 5.0e-3, 3.0e-3, 2.0e-3,
    1.0e-3, 5.0e-4, 3.0e-4, 2.0e-4, 1.0e-4,
)
TARGET_COVERAGE = 0.99
B0 = 1.0e-3

# Thresholds are selected on discovery only.  They are expressed as A_point
# rather than log(A_point) so the paper can state them intuitively.
APOINT_THRESHOLDS = (None, 2.0, 4.0, 8.0, 16.0, 32.0,
                     64.0, 128.0, 256.0, 512.0)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _ladder_entries(ladder):
    for bucket in BUCKETS:
        entry = ladder.get(bucket)
        if entry is None:
            entry = ladder.get(str(bucket))
        if isinstance(entry, dict) and _finite(entry.get("magnification")):
            yield bucket, entry


def _required(row, grid, atol, reltol):
    reference = row.get("reference") or {}
    reference_value = _float(reference.get("value"))
    uncertainty = _float(reference.get("uncertainty"))
    if not _finite(reference_value) or not _finite(uncertainty):
        return None
    scale = max(abs(reference_value), 1.0)
    # Match VBMicrolensing's stopping rule: continue only while both the
    # absolute and relative tests fail.  The effective allowance is therefore
    # the larger of the two, not their sum.
    budget = max(atol, reltol * scale)
    # Reference uncertainty is stored as a relative quantity.  Require a
    # decade of margin before letting a row constrain the empirical law.
    if uncertainty > 0.1 * budget / scale:
        return None

    measured = list(_ladder_entries(row.get(grid) or {}))
    inside = {}
    for bucket, entry in measured:
        value = _float(entry.get("magnification"))
        inside[bucket] = (
            bool(entry.get("support_proven"))
            and _finite(value)
            and abs(value - reference_value) <= budget
        )
    for index, bucket in enumerate(measured):
        if all(inside[later_bucket] for later_bucket, _ in measured[index:]):
            return bucket[0]
    return None


def _records(rows, dataset_name, atol=0.0):
    records = []
    for row_index, row in enumerate(rows):
        reference = row.get("reference") or {}
        reference_value = _float(reference.get("value"))
        if not _finite(reference_value):
            continue
        scale = max(abs(reference_value), 1.0)
        point = abs(_float(row.get("point_magnification")))
        if not _finite(point):
            point = 1.0
        for reltol in RELATIVE_LEVELS:
            budget = max(atol, reltol * scale)
            for grid in GRIDS:
                required = _required(row, grid, atol, reltol)
                if required is None:
                    continue
                records.append({
                    "dataset": dataset_name,
                    "row_index": row_index,
                    "case_id": row.get("case_id"),
                    "grid": grid,
                    "relative_tolerance": reltol,
                    "absolute_tolerance": atol,
                    "reference": reference_value,
                    "reference_scale": scale,
                    "budget": budget,
                    "relative_budget": budget / scale,
                    "point_magnification": point,
                    "required_resolution": required,
                    "log2_budget_ratio": math.log2(B0 / (budget / scale)),
                    "log2_point_magnification": math.log2(max(point, 1.0)),
                })
    return records


def _design(records, threshold):
    budget = np.asarray([row["log2_budget_ratio"] for row in records], dtype=float)
    point = np.asarray([row["log2_point_magnification"] for row in records], dtype=float)
    if threshold is None:
        features = budget[:, None]
        names = ("log2(B0/B)",)
    else:
        hinge = np.maximum(0.0, point - math.log2(threshold))
        features = np.column_stack([budget, hinge])
        names = ("log2(B0/B)", f"max(0, log2(Apoint/{threshold:g}))")
    target = np.log2(np.asarray(
        [row["required_resolution"] for row in records], dtype=float))
    return features, target, names


def _raw_coefficients(model, feature_names):
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    beta = np.asarray(model["beta"], dtype=float)
    coefficients = beta[1:] / std
    intercept = beta[0] - float(np.sum(coefficients * mean))
    return {
        "intercept": float(intercept),
        "coefficients": {
            name: float(value)
            for name, value in zip(feature_names, coefficients)
        },
    }


def _round_up(value):
    return fit_rules._round_up(float(value))


def _fit_quantile_linear_light(features, target, quantile):
    """Fit the same pinball-loss quantile model without a dense LP matrix.

    ``fit_rules.fit_quantile_linear`` is an exact LP implementation, but its
    residual-variable matrix is dense and scales as O(n^2).  The budget sweep
    intentionally creates many more labels than the original one-tolerance
    fit.  A smooth convex approximation to the same pinball loss is sufficient
    here; the intercept is then calibrated to the empirical target quantile on
    the discovery sample.  The final coverage is still measured explicitly.
    """
    n, k = features.shape
    if n == 0:
        return None
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1.0e-12] = 1.0
    scaled = (features - mean) / std
    design = np.column_stack([np.ones(n), scaled])
    target = np.asarray(target, dtype=float)
    epsilon = 0.03

    def objective(beta):
        residual = target - design @ beta
        return float(np.mean(
            quantile * residual
            + epsilon * np.logaddexp(0.0, -residual / epsilon)))

    def gradient(beta):
        residual = target - design @ beta
        weight = quantile - expit(-residual / epsilon)
        return -(design.T @ weight) / n

    initial = np.zeros(k + 1, dtype=float)
    initial[0] = float(np.quantile(target, quantile))
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    if not result.success:
        return None

    beta = np.asarray(result.x, dtype=float)
    # Put the finite-sample discovery quantile exactly at the requested
    # quantile; this removes the small smoothing bias before bucket rounding.
    residual = target - design @ beta
    beta[0] += float(np.quantile(residual, quantile))
    return {"mean": mean.tolist(), "std": std.tolist(),
            "beta": beta.tolist(),
            "fit_method": "smooth_pinball_with_quantile_intercept_calibration",
            "smooth_epsilon": epsilon,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message)}


def _score(prediction, records):
    required = np.asarray([row["required_resolution"] for row in records], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if required.size == 0:
        return {"rows": 0}
    ratios = prediction / required
    return {
        "rows": int(required.size),
        "coverage": float(np.mean(prediction >= required)),
        "median_required": float(np.median(required)),
        "median_predicted": float(np.median(prediction)),
        "p90_predicted": float(np.percentile(prediction, 90)),
        "p99_predicted": float(np.percentile(prediction, 99)),
        "median_overshoot_steps": float(np.median(
            np.log2(np.maximum(prediction, 1.0) /
                    np.maximum(required, 1.0)))),
        "median_work_vs_required": float(np.median(ratios ** 2)),
        "worst_prediction_to_required": float(np.min(ratios)),
    }


def _fit_candidate(records, threshold):
    features, target, feature_names = _design(records, threshold)
    model = _fit_quantile_linear_light(features, target, TARGET_COVERAGE)
    if model is None:
        return None
    prediction = fit_rules.predict_linear(model, features)
    return {
        "threshold": threshold,
        "features": list(feature_names),
        "model_standardized": model,
        "model_raw": _raw_coefficients(model, feature_names),
        "score": _score(prediction, records),
    }


def _budget_groups(records):
    """Return representative normalized-budget groups for a power-law fit."""
    budgets = np.asarray([row["relative_budget"] for row in records], dtype=float)
    log_budget = np.log10(budgets)
    unique = np.unique(budgets)
    if unique.size <= len(RELATIVE_LEVELS) + 1:
        masks = [budgets == value for value in unique]
    else:
        # Absolute-tolerance calibration makes epsilon vary continuously with
        # reference magnification.  Equal-count log bins keep each q99 point
        # statistically comparable without privileging the high-A tail.
        edges = np.unique(np.quantile(log_budget, np.linspace(0.02, 0.98, 17)))
        edges = np.concatenate(([-np.inf], edges, [np.inf]))
        masks = [
            (log_budget > edges[index]) & (log_budget <= edges[index + 1])
            for index in range(len(edges) - 1)
        ]
    groups = []
    for mask in masks:
        if int(mask.sum()) < 100:
            continue
        required = np.asarray(
            [row["required_resolution"] for row, keep in zip(records, mask)
             if keep], dtype=float)
        groups.append({
            "rows": int(mask.sum()),
            "median_relative_budget": float(np.median(budgets[mask])),
            "required_p99": float(np.percentile(required, 99.0)),
        })
    return groups


def _budget_power_prediction(model, records):
    budget = np.asarray([row["relative_budget"] for row in records], dtype=float)
    raw = np.exp2(
        model["intercept"]
        + model["slope"] * np.log2(B0 / budget)
        + model.get("safety_offset", 0.0)
    )
    return np.asarray([_round_up(value) for value in raw], dtype=float)


def _fit_budget_power(discovery, holdout, grid):
    discovery = [row for row in discovery if row["grid"] == grid]
    holdout = [row for row in holdout if row["grid"] == grid]
    groups = _budget_groups(discovery)
    x = np.asarray([
        math.log2(B0 / group["median_relative_budget"])
        for group in groups
    ], dtype=float)
    y = np.log2(np.asarray([group["required_p99"] for group in groups],
                           dtype=float))
    slope, intercept = np.polyfit(x, y, 1)

    # Choose the smallest upward calibration that covers 99% of discovery
    # rows after mapping the continuous law to the supported bucket ladder.
    model = {
        "intercept": float(intercept),
        "slope": float(slope),
        "safety_offset": 0.0,
    }
    for offset in np.linspace(0.0, 2.0, 401):
        model["safety_offset"] = float(offset)
        score = _score(_budget_power_prediction(model, discovery), discovery)
        level_coverages = []
        for level in RELATIVE_LEVELS:
            subset = [row for row in discovery
                      if row["relative_tolerance"] == level]
            if subset:
                level_coverages.append(_score(
                    _budget_power_prediction(model, subset), subset
                )["coverage"])
        if (score["coverage"] >= TARGET_COVERAGE
                and min(level_coverages, default=0.0) >= TARGET_COVERAGE):
            break
    model["normalization"] = "epsilon = (atol + reltol*scale) / scale"
    model["epsilon0"] = B0
    model["groups"] = groups
    model["fit_method"] = "discovery_p99_by_log_budget_bin_then_OLS"
    discovery_score = _score(_budget_power_prediction(model, discovery), discovery)
    holdout_score = _score(_budget_power_prediction(model, holdout), holdout)
    per_level = {}
    for level in RELATIVE_LEVELS:
        subset = [row for row in holdout
                  if row["relative_tolerance"] == level]
        if subset:
            per_level[str(level)] = _score(
                _budget_power_prediction(model, subset), subset)
    return {
        "model": model,
        "discovery": discovery_score,
        "holdout": holdout_score,
        "holdout_by_relative_tolerance": per_level,
        "discovery_rows": len(discovery),
        "holdout_rows": len(holdout),
    }


def _fit_apoint_candidates(discovery, holdout, grid):
    discovery = [row for row in discovery if row["grid"] == grid]
    holdout = [row for row in holdout if row["grid"] == grid]
    candidates = []
    for threshold in APOINT_THRESHOLDS:
        candidate = _fit_candidate(discovery, threshold)
        if candidate is None:
            continue
        features, _, _ = _design(holdout, threshold)
        prediction = fit_rules.predict_linear(
            candidate["model_standardized"], features)
        candidate["holdout"] = _score(prediction, holdout)
        candidates.append(candidate)

    # Select using discovery only.  Coverage is primary; cost is secondary.
    viable = [item for item in candidates
              if item["score"]["coverage"] >= TARGET_COVERAGE]
    selected = min(
        viable or candidates,
        key=lambda item: (
            item["score"]["median_predicted"],
            -item["score"]["coverage"],
            float("inf") if item["threshold"] is None else item["threshold"],
        ),
    )

    per_level = {}
    selected_features, _, _ = _design(holdout, selected["threshold"])
    selected_prediction = fit_rules.predict_linear(
        selected["model_standardized"], selected_features)
    for level in RELATIVE_LEVELS:
        subset = [row for row in holdout
                  if row["relative_tolerance"] == level]
        if not subset:
            continue
        features, _, _ = _design(subset, selected["threshold"])
        prediction = fit_rules.predict_linear(
            selected["model_standardized"], features)
        per_level[str(level)] = _score(prediction, subset)
    return {
        "selected": selected,
        "candidates": candidates,
        "holdout_by_relative_tolerance": per_level,
        "discovery_rows": len(discovery),
        "holdout_rows": len(holdout),
    }


def _write_records(path, records):
    if not records:
        path.write_text("")
        return
    fields = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot(report, records, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), squeeze=False)
    axes = axes[0]
    for axis, grid, colour in zip(axes, GRIDS, ("#1f4e79", "#c45a11")):
        selected = report["grids"][grid]["selected"]
        holdout = [row for row in records
                   if row["dataset"] == "holdout" and row["grid"] == grid]
        # Keep the paper figure legible and cheap; all rows are still scored
        # in JSON, while the plot only needs a deterministic visual sample.
        stride = max(1, len(holdout) // 5000)
        plot_rows = holdout[::stride]
        prediction = _budget_power_prediction(selected["model"], plot_rows)
        required = np.asarray(
            [row["required_resolution"] for row in plot_rows])
        axis.scatter(required, prediction, s=6, alpha=0.25, color=colour)
        lo = min(required.min(), prediction.min())
        hi = max(required.max(), prediction.max())
        axis.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("required resolution")
        axis.set_ylabel("predicted resolution")
        axis.set_title(f"{grid}, budget-power law")
        axis.grid(True, which="both", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    discovery_rows = analysis.load(arguments.discovery)
    holdout_rows = analysis.load(arguments.holdout)
    discovery = _records(
        discovery_rows, "discovery", arguments.absolute_tolerance)
    holdout = _records(
        holdout_rows, "holdout", arguments.absolute_tolerance)
    records = discovery + holdout
    _write_records(output / "error_budget_records.csv", records)

    report = {
        "formula": {
            "target": "log2(N_required)",
            "budget": "max(atol, reltol*max(abs(A_reference), 1))",
            "normalized_budget":
                "epsilon = budget/max(abs(A_reference), 1)",
            "baseline_normalized_budget": B0,
            "coverage_quantile": TARGET_COVERAGE,
            "relative_levels": list(RELATIVE_LEVELS),
            "absolute_tolerance": arguments.absolute_tolerance,
        },
        "inputs": {
            "discovery": str(arguments.discovery),
            "holdout": str(arguments.holdout),
            "discovery_rows": len(discovery_rows),
            "holdout_rows": len(holdout_rows),
            "usable_discovery_records": len(discovery),
            "usable_holdout_records": len(holdout),
        },
        "grids": {},
    }
    for grid in GRIDS:
        budget_power = _fit_budget_power(discovery, holdout, grid)
        apoint = _fit_apoint_candidates(discovery, holdout, grid)
        report["grids"][grid] = {
            "selected": budget_power,
            "budget_power": budget_power,
            "apoint_diagnostic": apoint,
        }
    (output / "error_budget_law.json").write_text(json.dumps(report, indent=2))
    if not arguments.no_plots:
        _plot(report, records, output / "error_budget_law.png")
    print(json.dumps({
        "output": str(output),
        "files": sorted(path.name for path in output.iterdir()),
    }, indent=2))


if __name__ == "__main__":
    main()

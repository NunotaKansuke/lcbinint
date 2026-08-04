#!/usr/bin/env python3
"""Choosing the replacement resolution rule, and reporting what it costs.

The shipping selector is a linear model in seven geometric features, fitted in
2026-07 when the grid density was what made the answer right.  The component
certificate changed that: correctness is now established per component and
verified before the answer is returned, so resolution buys accuracy rather than
validity.  This module asks what rule that leaves, and it is written to be able
to answer "a much simpler one", because that is what the measurement says.

Three candidates are compared on the same rows:

* **shipping** -- today's rule, reimplemented in ``sweep_resolution`` and
  evaluated against its own frozen ladder.
* **constant** -- one bin count per target tolerance and grid, and nothing else.
* **linear** -- constant plus a quantile regression on the same features the
  shipping rule uses, to test whether geometry still earns its place.

A selector is not a regression, so none of these is fitted to the mean.  Being
below the requirement means returning an answer outside the tolerance the caller
asked for, so what is fitted is a high quantile, and the achieved coverage is
measured rather than assumed.  Cost is reported as the work multiplier a caller
would actually see -- the square of the bin ratio, since the grid is
two-dimensional -- because a saving in ladder steps is not a saving anyone
feels.

Everything is fitted on the discovery sweep and reported on the holdout, which
was generated from an independent seed.  A rule that only holds on the data it
was fitted to is not a rule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .analysis import load, resolution_table
from .engines import BUCKETS
from .sweep_resolution import TARGET_TOLERANCES

FEATURE_NAMES = (
    "log_point", "log_rho", "log_q_small", "log_ratio",
    "proximity", "log_swallow", "limb_darkening_c",
)

# The fraction of rows a rule must be at least as coarse as.  Not 1.0: a rule
# forced to cover the single worst row of thousands pays for it on every row,
# and the fail-closed certificate means an uncovered row is refused rather than
# returned wrong -- the cost of missing is a retry at finer resolution, not a
# silently bad magnification.
TARGET_COVERAGE = 0.99

GRIDS = ("cartesian", "polar")


def _bucket_index():
    return {bucket: position for position, bucket in enumerate(BUCKETS)}


def _round_up(value):
    """The coarsest supported bucket that is at least ``value``."""
    for bucket in BUCKETS:
        if value <= bucket:
            return bucket
    return BUCKETS[-1]


def _nearest_bucket(value):
    """Map a measured bin count onto the ladder it came from.

    The library reports the resolution it actually used, which is the requested
    bucket after its own rounding -- 24 comes back as 23.  Comparing those
    directly against ladder positions would fail on the rounding rather than on
    the physics, so measured counts are snapped back to the nearest rung.
    """
    return min(BUCKETS, key=lambda bucket: abs(bucket - value))


def dataset(rows, grid, tolerance):
    """Features, requirement and the shipping rule's choice, for usable rows."""
    table = resolution_table(rows, tolerance)
    features, required, shipping = [], [], []
    censored = 0
    for record in table:
        need = record.get(grid)
        if need is None:
            censored += 1
            continue
        features.append([record[name] for name in FEATURE_NAMES])
        required.append(_nearest_bucket(need))
        current = record.get("current_rule")
        shipping.append(
            _nearest_bucket(current) if current is not None else np.nan)
    return (np.asarray(features, dtype=float).reshape(-1, len(FEATURE_NAMES)),
            np.asarray(required, dtype=float),
            np.asarray(shipping, dtype=float),
            censored)


def fit_quantile_linear(features, target, quantile):
    """Exact linear quantile regression, as a linear program.

    Solved rather than descended: the requirement distribution is close to
    degenerate at the loose tolerances -- almost every row lands on the same
    rung -- and on a surface that flat a subgradient method stalls wherever it
    started and reports a fit that is really its initialisation.  The LP has no
    such failure mode, and at this size it costs a second.
    """
    n, k = features.shape
    if n == 0:
        return None
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1.0e-12] = 1.0
    scaled = (features - mean) / std
    design = np.hstack([np.ones((n, 1)), scaled])

    # Variables: k+1 coefficients (free, split into +/-) then 2n residuals.
    # min q*sum(u) + (1-q)*sum(v)  s.t.  design@beta + u - v = target, u,v >= 0
    cost = np.concatenate([
        np.zeros(2 * (k + 1)),
        np.full(n, quantile),
        np.full(n, 1.0 - quantile),
    ])
    equality = np.hstack([design, -design, np.eye(n), -np.eye(n)])
    result = linprog(cost, A_eq=equality, b_eq=target,
                     bounds=[(0, None)] * (2 * (k + 1)) + [(0, None)] * (2 * n),
                     method="highs")
    if not result.success:
        return None
    beta = result.x[:k + 1] - result.x[k + 1:2 * (k + 1)]
    return {"mean": mean.tolist(), "std": std.tolist(), "beta": beta.tolist()}


def predict_linear(model, features):
    mean = np.asarray(model["mean"])
    std = np.asarray(model["std"])
    beta = np.asarray(model["beta"])
    scaled = (features - mean) / std
    raw = np.exp2(beta[0] + scaled @ beta[1:])
    return np.asarray([_round_up(value) for value in raw], dtype=float)


def score(predicted, required, shipping):
    """Coverage and cost of one rule on one set of rows."""
    index = _bucket_index()
    steps_predicted = np.asarray([index[int(b)] for b in predicted])
    steps_required = np.asarray([index[int(b)] for b in required])
    entry = {
        "rows": int(required.size),
        "coverage": float((predicted >= required).mean()),
        "median_bins": float(np.median(predicted)),
        "p99_bins": float(np.percentile(predicted, 99)),
        "median_overshoot_steps": float(
            np.median(steps_predicted - steps_required)),
        "median_work_vs_required": float(
            np.median((predicted / required) ** 2)),
    }
    finite = np.isfinite(shipping)
    if finite.any():
        entry["median_shipping_bins"] = float(np.median(shipping[finite]))
        entry["median_work_vs_shipping"] = float(
            np.median((predicted[finite] / shipping[finite]) ** 2))
    return entry


def compare(discovery, holdout, coverage):
    """Fit every candidate on discovery, score all of them on both."""
    report = {"coverage_target": coverage, "grids": {}}
    for grid in GRIDS:
        per_grid = {"tolerances": {}}
        for tolerance in TARGET_TOLERANCES:
            fit_x, fit_y, fit_ship, censored = dataset(
                discovery, grid, tolerance)
            out_x, out_y, out_ship, out_censored = dataset(
                holdout, grid, tolerance)
            if fit_y.size == 0 or out_y.size == 0:
                continue

            constant = _round_up(float(np.quantile(fit_y, coverage)))
            linear = fit_quantile_linear(fit_x, np.log2(fit_y), coverage)

            entry = {
                "unconverged_excluded": {"discovery": censored,
                                         "holdout": out_censored},
                "requirement": {
                    "median": float(np.median(fit_y)),
                    "p90": float(np.percentile(fit_y, 90)),
                    "p99": float(np.percentile(fit_y, 99)),
                    "max": float(fit_y.max()),
                    "distribution": {
                        str(int(b)): int((fit_y == b).sum())
                        for b in np.unique(fit_y)},
                },
                "constant_bins": constant,
                "rules": {},
            }
            entry["rules"]["shipping"] = {
                "discovery": score(np.where(np.isfinite(fit_ship), fit_ship,
                                            BUCKETS[-1]), fit_y, fit_ship),
                "holdout": score(np.where(np.isfinite(out_ship), out_ship,
                                          BUCKETS[-1]), out_y, out_ship),
            }
            entry["rules"]["constant"] = {
                "discovery": score(np.full(fit_y.size, constant, dtype=float),
                                   fit_y, fit_ship),
                "holdout": score(np.full(out_y.size, constant, dtype=float),
                                 out_y, out_ship),
            }
            if linear is not None:
                entry["rules"]["linear"] = {
                    "model": linear,
                    "discovery": score(predict_linear(linear, fit_x), fit_y,
                                       fit_ship),
                    "holdout": score(predict_linear(linear, out_x), out_y,
                                     out_ship),
                }
            per_grid["tolerances"][str(tolerance)] = entry
        report["grids"][grid] = per_grid
    return report


def grid_switch(rows, tolerance):
    """Which grid needed fewer bins, and whether that is predictable.

    The bin count is a proxy for cost, not cost itself -- a polar bin and a
    Cartesian bin are not the same work -- so this reports the resolution
    preference only.  The switch that ships has to be decided on the measured
    timings from the speed sweep; this is the part of the answer the resolution
    sweep can supply on its own.
    """
    index = _bucket_index()
    table = resolution_table(rows, tolerance)
    cartesian_wins = polar_wins = ties = 0
    margins = []
    for record in table:
        c, p = record.get("cartesian"), record.get("polar")
        if c is None or p is None:
            continue
        steps = index[_nearest_bucket(c)] - index[_nearest_bucket(p)]
        margins.append(steps)
        if steps < 0:
            cartesian_wins += 1
        elif steps > 0:
            polar_wins += 1
        else:
            ties += 1
    total = cartesian_wins + polar_wins + ties
    if not total:
        return {}
    return {
        "rows": total,
        "cartesian_cheaper": cartesian_wins / total,
        "polar_cheaper": polar_wins / total,
        "tied": ties / total,
        "median_steps_cartesian_minus_polar": float(np.median(margins)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--output")
    parser.add_argument("--coverage", type=float, default=TARGET_COVERAGE)
    arguments = parser.parse_args()

    discovery = load(arguments.discovery)
    holdout = load(arguments.holdout)
    result = compare(discovery, holdout, arguments.coverage)
    result["rows"] = {"discovery": len(discovery), "holdout": len(holdout)}
    result["ladder"] = list(BUCKETS)
    result["grid_switch"] = {
        str(tolerance): {
            "discovery": grid_switch(discovery, tolerance),
            "holdout": grid_switch(holdout, tolerance),
        }
        for tolerance in TARGET_TOLERANCES
    }
    rendered = json.dumps(result, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

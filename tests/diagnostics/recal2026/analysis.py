"""Turning the sweep tables into the numbers the rules are made of.

The sweeps decide nothing; they record.  This module is where the decisions
happen, and keeping it separate is what makes it possible to change a decision
without re-running a multi-hour sweep.  It is also why the loader is defensive:
a partially finished sweep must summarise correctly rather than crash, so that
the shape of the answer can be seen before the last case lands.

Three things are read out of the same table:

* the resolution actually required, per grid and per target tolerance, which is
  what replaces the 2026-07 quantile model;
* which grid required less, which is the Cartesian/polar switch;
* what the routes that skip the grid entirely -- point source, hexadecapole,
  grazing quadrature -- actually delivered, which is what the method boundaries
  are moved on.

Rows whose reference is not sharp enough for a given tolerance are excluded from
that tolerance and counted, never silently included with whatever number the
ladder produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .engines import BUCKETS
from .sweep_resolution import TARGET_TOLERANCES


def load(directory):
    """Every finished case in a sweep directory, as a list of rows.

    Cases still being written are skipped: the sweep writes to a ``.partial``
    file and renames it, so anything matching the final name is complete.
    """
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob("case-*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for row in payload.get("rows", []):
            if "error" in row:
                continue
            rows.append(row)
    return rows


def _ratio(row):
    rho = row.get("rho", float("nan"))
    distance = row.get("caustic_distance", float("nan"))
    if not (rho and math.isfinite(distance)):
        return float("inf")
    return distance / rho


def _q_small(row):
    q = abs(row.get("q", float("nan")))
    if not math.isfinite(q) or q <= 0.0:
        return float("nan")
    return q if q < 1.0 else 1.0 / q


def features(row):
    """The quantities a resolution rule is allowed to be a function of.

    Everything here is available at evaluation time inside the library, before
    the finite-source integral is run.  A feature the runtime cannot compute is
    useless no matter how well it fits.
    """
    ratio = _ratio(row)
    point = abs(row.get("point_magnification", float("nan")))
    rho = row.get("rho", float("nan"))
    q_small = _q_small(row)
    return {
        "log_point": math.log10(max(point, 1.0)),
        "log_rho": math.log10(max(rho, 1.0e-12)),
        "log_q_small": math.log10(max(q_small, 1.0e-12)),
        "log_ratio": math.log10(max(min(ratio, 1.0e6), 1.0e-3)),
        "proximity": max(0.0, 2.0 - min(ratio, 2.0)),
        "log_swallow": max(0.0, math.log10(
            max(4.0 * rho / max(q_small, 1.0e-12), 1.0))),
        "limb_darkening_c": row.get("limb_darkening_c", 0.0),
    }


def resolution_table(rows, tolerance):
    """One record per usable row: what each grid needed, and what today picks.

    ``required`` is ``None`` when no measured bucket was good enough, which is a
    different statement from a large bucket and is kept distinguishable: it
    means the ladder ran out before converging, so the row constrains the rule
    only from below.
    """
    key = str(tolerance)
    out = []
    for row in rows:
        entry = (row.get("required") or {}).get(key)
        if not entry or not entry.get("usable"):
            continue
        record = dict(features(row))
        record.update({
            "case_id": row.get("case_id"),
            "s": row.get("s"), "q": row.get("q"), "rho": row.get("rho"),
            "profile": row.get("profile"),
            "ratio": _ratio(row),
            "point_magnification": row.get("point_magnification"),
            "reference": (row.get("reference") or {}).get("value"),
            "reference_uncertainty": (row.get("reference") or {}).get(
                "uncertainty"),
            "cartesian": entry.get("cartesian"),
            "polar": entry.get("polar"),
            "current_rule": entry.get("current_rule"),
            "current_rule_prefers_polar": entry.get("current_rule_prefers_polar"),
            "auto_method": ((row.get("auto") or {}).get(key) or {}).get("method"),
            "auto_magnification": (
                (row.get("auto") or {}).get(key) or {}).get("magnification"),
            "cartesian_censored": (row.get("cartesian") or {}).get(
                "censored_before"),
            "polar_censored": (row.get("polar") or {}).get("censored_before"),
        })
        out.append(record)
    return out


def usability_report(rows):
    """How much of the sweep each tolerance can actually speak to."""
    report = {}
    for tolerance in TARGET_TOLERANCES:
        key = str(tolerance)
        usable = sum(
            1 for row in rows
            if ((row.get("required") or {}).get(key) or {}).get("usable"))
        report[key] = {"usable": usable, "total": len(rows)}
    return report


def bucket_shift(table):
    """New requirement versus the shipping rule, in bucket steps.

    Reported in steps rather than bins because the selector can only choose
    from the fixed ladder, so "two buckets coarser" is the actionable statement
    and "37 bins fewer" is not.
    """
    index = {bucket: position for position, bucket in enumerate(BUCKETS)}
    shifts = []
    for record in table:
        required, current = record.get("cartesian"), record.get("current_rule")
        if required is None or current is None:
            continue
        shifts.append(index[required] - index[current])
    if not shifts:
        return {}
    shifts = np.asarray(shifts)
    return {
        "count": int(shifts.size),
        "median_steps": float(np.median(shifts)),
        "mean_steps": float(shifts.mean()),
        "fraction_lower": float((shifts < 0).mean()),
        "fraction_equal": float((shifts == 0).mean()),
        "fraction_higher": float((shifts > 0).mean()),
        "p95_higher": float(np.percentile(shifts, 95)),
        "worst_higher": int(shifts.max()),
    }


def grid_preference(table):
    """Which grid needed fewer bins, and by how much.

    Ties are counted separately because they are the majority in the easy part
    of the space and would otherwise wash out the regions where the choice
    genuinely matters.
    """
    index = {bucket: position for position, bucket in enumerate(BUCKETS)}
    cartesian_wins = polar_wins = ties = 0
    both = 0
    for record in table:
        c, p = record.get("cartesian"), record.get("polar")
        if c is None or p is None:
            continue
        both += 1
        if index[c] < index[p]:
            cartesian_wins += 1
        elif index[p] < index[c]:
            polar_wins += 1
        else:
            ties += 1
    if not both:
        return {}
    return {
        "count": both,
        "cartesian_cheaper": cartesian_wins / both,
        "polar_cheaper": polar_wins / both,
        "tied": ties / both,
    }


def method_audit(rows, tolerance):
    """Did the routes that skip the grid stay inside the tolerance they claim?

    This is the evidence the point-source, hexadecapole and grazing-quadrature
    boundaries are moved on.  A route that is comfortably inside its budget
    everywhere it is used can be given more of the space; one that is outside
    it anywhere is where the boundary has to come in, regardless of how rare
    the violation is.
    """
    key = str(tolerance)
    audit = {}
    for row in rows:
        entry = (row.get("required") or {}).get(key)
        if not entry or not entry.get("usable"):
            continue
        result = (row.get("auto") or {}).get(key) or {}
        method = result.get("method")
        value = result.get("magnification")
        expected = (row.get("reference") or {}).get("value")
        if method is None or value is None or expected is None:
            continue
        if not (math.isfinite(value) and math.isfinite(expected)):
            continue
        error = abs(value - expected) / max(abs(expected), 1.0)
        record = audit.setdefault(method, {
            "count": 0, "violations": 0, "errors": [], "ratios": []})
        record["count"] += 1
        record["errors"].append(error)
        record["ratios"].append(_ratio(row))
        if error > tolerance:
            record["violations"] += 1
    for method, record in audit.items():
        errors = np.asarray(record.pop("errors"))
        ratios = np.asarray(record.pop("ratios"))
        finite = ratios[np.isfinite(ratios)]
        record["median_error"] = float(np.median(errors))
        record["p95_error"] = float(np.percentile(errors, 95))
        record["max_error"] = float(errors.max())
        record["violation_rate"] = record["violations"] / record["count"]
        if finite.size:
            record["ratio_range"] = [float(finite.min()), float(finite.max())]
            violating = ratios[errors > tolerance]
            violating = violating[np.isfinite(violating)]
            if violating.size:
                record["violating_ratio_range"] = [
                    float(violating.min()), float(violating.max())]
    return audit


def summarise(directory):
    """A first look at a sweep, safe to run while it is still going."""
    rows = load(directory)
    report = {
        "rows": len(rows),
        "usability": usability_report(rows),
        "tolerances": {},
    }
    for tolerance in TARGET_TOLERANCES:
        table = resolution_table(rows, tolerance)
        report["tolerances"][str(tolerance)] = {
            "rows": len(table),
            "unconverged": sum(1 for r in table if r["cartesian"] is None),
            "bucket_shift_vs_current": bucket_shift(table),
            "grid_preference": grid_preference(table),
            "method_audit": method_audit(rows, tolerance),
        }
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    arguments = parser.parse_args()
    print(json.dumps(summarise(arguments.directory), indent=2, default=str))


if __name__ == "__main__":
    main()

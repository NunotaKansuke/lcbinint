#!/usr/bin/env python3
"""Resolution-ladder sweep: the accuracy dataset the new rules are fitted to.

For every sampled source position this runs the complete supported resolution
ladder on both image-plane grids, builds a reference from the top of those
ladders plus an independent contour value, and records everything.  Nothing is
decided here.  Which bucket was required, where the method boundaries should
sit, and whether the certificate lets the resolution drop are all questions
answered later from this table, so that changing an answer does not mean
re-running the sweep.

Timings recorded here are per-call and include the flat per-entry setup cost;
they are cost proxies for ordering buckets, not the campaign's speed result.
Speed is measured separately, on light-curve blocks, by ``sweep_speed.py``.

The unit of checkpointing is one lens case, written atomically, so shards can
share an output directory and an interrupted run resumes by skipping finished
cases.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import reference
from .engines import BUCKETS, SHIPPING_BUCKET_FLOOR, PROFILES, LcbinintEngine
from .geometry import caustic_branches, make_lens_cases, sample_positions

# The tolerances the resolution rule is being made a function of.
TARGET_TOLERANCES = (1.0e-2, 1.0e-3, 1.0e-4)

# The ladder the shipping selector rounds against, which is not the ladder this
# campaign measures on: ``BUCKETS`` was extended below 16 to uncensor the coarse
# end.  The baseline has to keep its own floor, or extending the measurement
# would silently make the rule being compared against a different rule.
SHIPPING_BUCKETS = tuple(b for b in BUCKETS if b >= SHIPPING_BUCKET_FLOOR)


def current_rule_bucket(q, rho, caustic_distance, point_magnification,
                        limb_darkening_c, relative_tolerance, maximum_bins=400):
    """The bucket the current one-shot selector would choose.

    Reimplemented here from ``calibrated_binary_resolution`` so the old and new
    rules can be compared offline from the stored table, without a second
    sweep.  It is a pure function of quantities this sweep already records; the
    analysis step checks it against the native selector on a sample.
    """
    cap = max(maximum_bins, 1)
    a_point = abs(point_magnification)
    ratio = caustic_distance / rho if rho > 0.0 else float("inf")
    if not (math.isfinite(a_point) and math.isfinite(ratio)):
        return min(100, cap), True
    prefer_polar = a_point >= 200.0
    if 0.0 < relative_tolerance < 1.0e-3:
        bins = 200
    elif relative_tolerance >= 1.0e-2:
        bins = 50 if prefer_polar else 16
    else:
        bins = 100 if prefer_polar else 50
    return min(bins, cap), prefer_polar


def evaluate_row(case, position, profile_name, profile_c, ladder_budget):
    """Everything measured at one (position, profile)."""
    s, q, rho = case.separation, case.mass_ratio, case.source_radius
    x, y = position["x"], position["y"]

    started = time.perf_counter()
    row = {
        "case_id": case.case_id,
        "s": s, "q": q, "rho": rho, "x": x, "y": y,
        "profile": profile_name,
        "limb_darkening_c": profile_c,
        "intended_distance_factor": position["intended_distance_factor"],
    }

    # The shipping automatic path, at each tolerance the rule is to cover.
    # This is both the baseline the new rule has to beat and the source of the
    # routing diagnostics (measured caustic distance, point magnification) the
    # rules are functions of.
    row["auto"] = {}
    for tolerance in TARGET_TOLERANCES:
        engine = LcbinintEngine(
            grid=None, nbin="auto", profile_c=profile_c,
            reltol=tolerance, max_source_bins=400)
        try:
            result = engine(s, q, rho, x, y)
        except Exception as error:  # noqa: BLE001
            row["auto"][str(tolerance)] = {
                "error": f"{type(error).__name__}: {error}"}
            continue
        row["auto"][str(tolerance)] = {
            "magnification": result.magnification,
            "error_estimate": result.error_estimate,
            "converged": result.converged,
            "support_proven": result.support_proven,
            "method": result.method,
            "seconds": result.seconds,
        }
        row.setdefault("caustic_distance", result.extra["caustic_distance"])
        row.setdefault("point_magnification", result.extra["point_magnification"])
        row.setdefault("image_count", result.extra["image_count"])

    row["cartesian"] = reference.evaluate_ladder(
        s, q, rho, x, y, profile_c, grid="cartesian", time_budget=ladder_budget)
    row["polar"] = reference.evaluate_ladder(
        s, q, rho, x, y, profile_c, grid="polar", time_budget=ladder_budget)

    # The contour witness gets the same budget as one grid ladder.  It is not
    # a free third opinion: on high-magnification rows a tight contour call is
    # by far the most expensive thing in the row, so it is escalated only while
    # it stays affordable and reports how far it got.
    contour = reference.contour_reference(
        s, q, rho, x, y, profile_c, time_budget=ladder_budget)
    row["contour"] = contour

    row["reference"] = reference.build(
        s, q, rho, x, y, profile_c,
        cartesian=row["cartesian"], polar=row["polar"],
        contour=contour["value"], contour_self_gap=contour["self_gap"])

    # The bucket each grid actually needed, per target tolerance, and what the
    # shipping rule would have picked for comparison.
    row["required"] = {}
    for tolerance in TARGET_TOLERANCES:
        entry = {"usable": reference.usable_for(row["reference"], tolerance)}
        if entry["usable"]:
            value = row["reference"]["value"]
            entry["cartesian"] = reference.required_bucket(
                row["cartesian"], value, tolerance)
            entry["polar"] = reference.required_bucket(
                row["polar"], value, tolerance)
        distance = row.get("caustic_distance", float("inf"))
        point = row.get("point_magnification", float("nan"))
        if math.isfinite(point):
            selected, prefer_polar = current_rule_bucket(
                q, rho, distance, point, profile_c, tolerance)
            entry["current_rule"] = selected
            entry["current_rule_prefers_polar"] = prefer_polar
        row["required"][str(tolerance)] = entry

    row["row_seconds"] = time.perf_counter() - started
    return row


def run_case(case, seed, per_case, ladder_budget, profiles):
    rng = np.random.default_rng(seed + 7919 * case.case_id)
    try:
        branches = caustic_branches(case.separation, case.mass_ratio)
    except Exception as error:  # noqa: BLE001
        return {"case": case.as_dict(),
                "error": f"caustics: {type(error).__name__}: {error}",
                "rows": []}
    positions = sample_positions(case, branches, rng, per_case)
    rows = []
    for position in positions:
        for profile_name in profiles:
            profile_c = PROFILES[profile_name]
            try:
                rows.append(evaluate_row(
                    case, position, profile_name, profile_c, ladder_budget))
            except Exception as error:  # noqa: BLE001
                rows.append({
                    "case_id": case.case_id,
                    "x": position["x"], "y": position["y"],
                    "profile": profile_name,
                    "error": f"{type(error).__name__}: {error}",
                })
    return {"case": case.as_dict(), "rows": rows}


def _worker(payload):
    case, seed, per_case, ladder_budget, profiles, output = payload
    target = Path(output) / f"case-{case.case_id:05d}.json"
    if target.exists():
        return case.case_id, "skipped", 0.0
    started = time.perf_counter()
    result = run_case(case, seed, per_case, ladder_budget, profiles)
    elapsed = time.perf_counter() - started
    result["seconds"] = elapsed
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.replace(target)
    return case.case_id, "done", elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--per-case", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--ladder-budget", type=float, default=25.0,
                        help="seconds per grid ladder before censoring")
    parser.add_argument("--profiles", default="uniform,linear")
    arguments = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    profiles = [p.strip() for p in arguments.profiles.split(",") if p.strip()]

    cases = make_lens_cases(arguments.cases, arguments.seed)
    (output / "manifest.json").write_text(json.dumps({
        "seed": arguments.seed,
        "cases": arguments.cases,
        "per_case": arguments.per_case,
        "profiles": profiles,
        "buckets": list(BUCKETS),
        "target_tolerances": list(TARGET_TOLERANCES),
        "ladder_budget_seconds": arguments.ladder_budget,
        "lens_cases": [c.as_dict() for c in cases],
    }, indent=2))

    payloads = [
        (case, arguments.seed, arguments.per_case, arguments.ladder_budget,
         profiles, str(output))
        for case in cases
    ]

    import multiprocessing

    started = time.perf_counter()
    done = 0
    with multiprocessing.Pool(arguments.workers) as pool:
        for case_id, status, elapsed in pool.imap_unordered(_worker, payloads):
            done += 1
            rate = done / max(time.perf_counter() - started, 1e-9)
            remaining = (len(payloads) - done) / max(rate, 1e-9)
            print(f"[{done}/{len(payloads)}] case {case_id} {status} "
                  f"{elapsed:.1f}s  eta {remaining / 60:.1f} min", flush=True)
    print(f"finished in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reading the seeding ablation: what each stage costs and what it protects.

Two rules govern what counts here.

Only rows the integrator actually seeded are compared.  A position that exits
through the point-source, hexadecapole or grazing-quadrature route never builds
a seed set, so including it would dilute every ratio with rows the policy could
not have affected -- and the grazing-quadrature rows carry a known accuracy
problem of their own, which would land on the seeding stage's account.

An ablation is judged against the independent witness, not against the full
policy.  Reproducing the full policy exactly is the strongest evidence
available, but "differs from full" is not the same as "wrong": a different seed
set can traverse the same components in a different order.  So the difference
from the full policy is reported as the detector -- it is what catches a
dropped component, which is a discrete jump, not a rounding difference -- and
the error against VBMicrolensing decides whether a difference was a loss.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# Below this the two policies traversed the same components and differ only in
# the order the flood fill claimed cells; above it something structural moved.
STRUCTURAL_DIFFERENCE = 1.0e-9

GRID_METHODS = ("inverse_ray_cartesian", "inverse_ray_polar")


def load(path):
    payload = json.loads(Path(path).read_text())
    return payload


def _grid_row(entry):
    return (isinstance(entry, dict) and "error" not in entry
            and entry.get("method") in GRID_METHODS)


def _relative(value, reference):
    if not (math.isfinite(value) and math.isfinite(reference)) or reference == 0.0:
        return float("nan")
    return abs(value - reference) / abs(reference)


def cost_table(full):
    """Probe counts and seconds, per resolution, over the seeded rows."""
    out = {}
    for bucket in [str(b) for b in full["buckets"]]:
        ring, heuristic, certified, offered, extrema = [], [], [], [], []
        ring_s, heuristic_s, cert_s, certify_s, call_s = [], [], [], [], []
        for row in full["rows"]:
            entry = row["measured"].get(bucket)
            if not _grid_row(entry):
                continue
            counters = entry["counters"]
            ring.append(counters["ring_solves"])
            heuristic.append(counters.get("heuristic_solves", 0))
            certified.append(counters["certified_solves"])
            offered.append(counters["certified_offered"])
            extrema.append(counters["certified_extrema"])
            ring_s.append(counters["ring_seconds"])
            heuristic_s.append(counters.get("heuristic_seconds", 0.0))
            cert_s.append(counters["certified_seconds"])
            certify_s.append(counters["certify_seconds"])
            call_s.append(entry["call_seconds"])
        if not ring:
            continue
        ring, heuristic, certified = (
            np.asarray(ring), np.asarray(heuristic), np.asarray(certified))
        seeding = (np.asarray(ring_s) + np.asarray(heuristic_s) +
                   np.asarray(cert_s) + np.asarray(certify_s))
        call = np.asarray(call_s)
        out[bucket] = {
            "rows": int(ring.size),
            "ring_solves": {"median": float(np.median(ring)),
                            "p90": float(np.percentile(ring, 90)),
                            "total": int(ring.sum())},
            "heuristic_solves": {"median": float(np.median(heuristic)),
                                  "p90": float(np.percentile(heuristic, 90)),
                                  "total": int(heuristic.sum())},
            "certified_solves": {"median": float(np.median(certified)),
                                 "p90": float(np.percentile(certified, 90)),
                                 "total": int(certified.sum())},
            "certified_offered_total": int(np.sum(offered)),
            "certified_extrema_median": float(np.median(extrema)),
            "solves_ring_over_certified": (
                float(ring.sum() / certified.sum()) if certified.sum() else float("inf")),
            "seconds": {
                "ring": float(np.sum(ring_s)),
                "branch_heuristic": float(np.sum(heuristic_s)),
                "certified_probes": float(np.sum(cert_s)),
                "certify_support": float(np.sum(certify_s)),
                "seeding_total": float(seeding.sum()),
                "call_total": float(call.sum()),
                # The per-call time carries the harness's fixed setup, so this
                # share is a lower bound on what the seeding is of the work
                # that a light curve actually repeats.  probe_timing.py
                # measures the same share on blocks, where the setup is
                # amortised away.
                "seeding_share_of_call": float(seeding.sum() / call.sum()),
            },
        }
    return out


def compare(full, other, name):
    """One ablation against the full policy, on the rows both seeded."""
    by_id = {row["row_id"]: row for row in other["rows"]}
    buckets = [str(b) for b in full["buckets"]]
    report = {"policy": other["policy"], "buckets": {}}
    families = defaultdict(lambda: {"rows": 0, "structural": 0, "worst": 0.0})

    for bucket in buckets:
        compared = 0
        structural = []
        refused = 0
        gained = 0
        full_vs_witness, other_vs_witness = [], []
        worst_rows = []
        for row in full["rows"]:
            mine = row["measured"].get(bucket)
            theirs = (by_id.get(row["row_id"], {}).get("measured") or {}).get(bucket)
            if not _grid_row(mine) or not isinstance(theirs, dict) or "error" in theirs:
                continue
            compared += 1
            difference = _relative(theirs["magnification"], mine["magnification"])
            if math.isfinite(difference):
                structural.append(difference)
            if mine["support_proven"] and not theirs["support_proven"]:
                refused += 1
            if theirs["support_proven"] and not mine["support_proven"]:
                gained += 1
            witness = (row.get("vbm") or {}).get("magnification")
            if witness:
                full_error = _relative(mine["magnification"], witness)
                other_error = _relative(theirs["magnification"], witness)
                full_vs_witness.append(full_error)
                other_vs_witness.append(other_error)
            family = families[row["family"]]
            family["rows"] += 1
            if difference > STRUCTURAL_DIFFERENCE:
                family["structural"] += 1
                family["worst"] = max(family["worst"], difference)
                worst_rows.append({
                    "row_id": row["row_id"], "family": row["family"],
                    "s": row["s"], "q": row["q"], "rho": row["rho"],
                    "d_over_rho": mine["caustic_distance"] / row["rho"],
                    "intended_cap_depth": row.get("intended_cap_depth"),
                    "full": mine["magnification"],
                    "ablated": theirs["magnification"],
                    "witness": witness,
                    "relative_difference": difference,
                })

        if not compared:
            continue
        structural = np.asarray(structural)
        entry = {
            "rows": compared,
            "identical": int((structural == 0.0).sum()),
            "structural_differences": int((structural > STRUCTURAL_DIFFERENCE).sum()),
            "worst_difference": float(structural.max()) if structural.size else 0.0,
            "newly_refused": refused,
            "newly_proven": gained,
            "worst_rows": sorted(
                worst_rows, key=lambda r: -r["relative_difference"])[:10],
        }
        if full_vs_witness:
            full_error = np.asarray(full_vs_witness)
            other_error = np.asarray(other_vs_witness)
            entry["vs_witness"] = {
                "rows": int(full_error.size),
                "full_median": float(np.median(full_error)),
                "full_p99": float(np.percentile(full_error, 99)),
                "ablated_median": float(np.median(other_error)),
                "ablated_p99": float(np.percentile(other_error, 99)),
                # The decisive number: rows the ablation made materially worse
                # against an implementation that shares no code with either.
                "degraded_rows": int(np.sum(
                    other_error > np.maximum(10.0 * full_error, 1.0e-6))),
            }
        report["buckets"][bucket] = entry

    report["families"] = {name: dict(value) for name, value in families.items()}
    return report


def savings(full, other):
    """How much probe work the ablation actually removed."""
    by_id = {row["row_id"]: row for row in other["rows"]}
    out = {}
    for bucket in [str(b) for b in full["buckets"]]:
        base_solves = base_seconds = 0.0
        cut_solves = cut_seconds = 0.0
        for row in full["rows"]:
            mine = row["measured"].get(bucket)
            theirs = (by_id.get(row["row_id"], {}).get("measured") or {}).get(bucket)
            if not _grid_row(mine) or not isinstance(theirs, dict) or "error" in theirs:
                continue
            for entry, solves, seconds in (
                    (mine, "base", "base"), (theirs, "cut", "cut")):
                counters = entry["counters"]
                total = (counters["ring_solves"] +
                         counters.get("heuristic_solves", 0) +
                         counters["certified_solves"])
                clock = (counters["ring_seconds"] +
                         counters.get("heuristic_seconds", 0.0) +
                         counters["certified_seconds"]
                         + counters["certify_seconds"])
                if solves == "base":
                    base_solves += total
                    base_seconds += clock
                else:
                    cut_solves += total
                    cut_seconds += clock
        if base_solves:
            out[bucket] = {
                "solves_full": int(base_solves),
                "solves_ablated": int(cut_solves),
                "solves_removed_fraction": 1.0 - cut_solves / base_solves,
                "seeding_seconds_full": base_seconds,
                "seeding_seconds_ablated": cut_seconds,
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    directory = Path(arguments.directory)
    full = load(directory / "full.json")
    result = {
        "full_policy": full["policy"],
        "cost": cost_table(full),
        "ablations": {},
    }
    for path in sorted(directory.glob("*.json")):
        if path.stem == "full":
            continue
        other = load(path)
        result["ablations"][path.stem] = {
            "comparison": compare(full, other, path.stem),
            "savings": savings(full, other),
        }
    rendered = json.dumps(result, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

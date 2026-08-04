#!/usr/bin/env python3
"""The speed sweep's tables: accuracy against cost, and where each grid wins.

The sweep records; this decides.  Three questions are answered from one table:

* **Magnification against speed.**  For each engine, the cheapest setting that
  actually reached a given accuracy on a block -- not the setting that was
  asked for.  A knob is a request; the error column is what happened.  Pairing
  the two per block is the only way a Pareto front means anything, because a
  setting that misses its tolerance on half the geometries is not a point on
  anyone's front.

* **Which grid to use.**  Decided on measured time at equal delivered accuracy,
  not on bin counts.  Stage 1 found Cartesian needs fewer bins, but bins are
  not seconds: a polar bin and a Cartesian bin are different amounts of work,
  and the answer changes with rho and with mass ratio.

* **rho against speed.**  The secondary axis, reported as the cost of a fixed
  delivered accuracy across the sampled radii.

Blocks whose reference floor is coarser than the accuracy being asked about are
excluded from that comparison and counted.  Including them would credit an
engine for meeting a tolerance the reference cannot resolve.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# The accuracies the front is read at.  These are delivered accuracies, not
# requested tolerances, so they are stated once here and used for every engine.
ACCURACY_TARGETS = (1.0e-2, 1.0e-3, 1.0e-4)

GRIDS = ("lcbinint_cartesian", "lcbinint_polar")


def load(directory):
    """Every finished block in a sweep directory."""
    rows = []
    for path in sorted(Path(directory).glob("block-*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for row in payload.get("rows", []):
            if "error" in row:
                continue
            rows.append(row)
    return rows


def _error(entry):
    """The worst per-epoch relative error the entry actually delivered."""
    error = entry.get("error")
    if not isinstance(error, dict):
        return None
    worst = error.get("worst")
    if worst is None or not math.isfinite(worst):
        return None
    if not error.get("epochs"):
        return None
    return float(worst)


def _usable(row, target):
    """Can this block adjudicate an accuracy of ``target`` at all?

    The reference floor is the uncertainty of the converged ladder the sweep
    built for the block.  Asking whether an engine reached 1e-4 on a block whose
    reference is only good to 1e-3 is a question about the reference.
    """
    floor = row.get("reference_floor")
    return floor is not None and math.isfinite(floor) and floor <= 0.1 * target


def cheapest_meeting(row, engine_prefix, target):
    """Cheapest setting of one engine that met ``target`` on this block.

    Returns ``(seconds_per_epoch, knob)`` or ``None`` if no setting did.  The
    scan is over settings, not over the requested knob order, because the
    engines are not all monotone in their knob: microlux caps its adaptive
    budget, and the grids can overshoot a bucket.
    """
    best = None
    for entry in row.get("engines", []):
        name = entry.get("engine")
        if name != engine_prefix:
            continue
        seconds = entry.get("seconds_per_epoch")
        if seconds is None or not math.isfinite(seconds):
            continue
        error = _error(entry)
        if error is None or error > target:
            continue
        if best is None or seconds < best[0]:
            best = (float(seconds), entry.get("knob"))
    return best


def pareto_table(rows, engines):
    """Per-accuracy cost of each engine, over the blocks that can judge it."""
    table = {}
    for target in ACCURACY_TARGETS:
        judged = [row for row in rows if _usable(row, target)]
        per_engine = {}
        for engine in engines:
            costs, misses = [], 0
            for row in judged:
                best = cheapest_meeting(row, engine, target)
                if best is None:
                    misses += 1
                else:
                    costs.append(best[0])
            per_engine[engine] = {
                "blocks_judged": len(judged),
                "blocks_met": len(costs),
                "blocks_missed": misses,
                "median_seconds_per_epoch":
                    float(np.median(costs)) if costs else None,
                "p90_seconds_per_epoch":
                    float(np.percentile(costs, 90)) if costs else None,
            }
        table[target] = {
            "blocks_judged": len(judged),
            "blocks_excluded": len(rows) - len(judged),
            "engines": per_engine,
        }
    return table


def grid_switch(rows, target):
    """Cartesian against polar, per block, on measured time at equal accuracy.

    Only blocks where *both* grids reached the accuracy are compared.  A block
    where one grid failed outright is a different fact -- reported separately --
    and folding it into a speed ratio would report a failure as infinite speed.
    """
    comparisons = []
    only_cartesian = only_polar = neither = 0
    for row in rows:
        if not _usable(row, target):
            continue
        cartesian = cheapest_meeting(row, "lcbinint_cartesian", target)
        polar = cheapest_meeting(row, "lcbinint_polar", target)
        if cartesian and polar:
            comparisons.append({
                "s": row["s"], "q": row["q"], "rho": row["rho"],
                "distance_factor": row.get("intended_distance_factor"),
                "magnification": row.get("magnification"),
                "cartesian_seconds": cartesian[0], "cartesian_bins": cartesian[1],
                "polar_seconds": polar[0], "polar_bins": polar[1],
                # Above one, Cartesian is the cheaper grid on this block.
                "polar_over_cartesian": polar[0] / cartesian[0],
            })
        elif cartesian:
            only_cartesian += 1
        elif polar:
            only_polar += 1
        else:
            neither += 1
    return {
        "compared": comparisons,
        "cartesian_only": only_cartesian,
        "polar_only": only_polar,
        "neither": neither,
    }


def _bucket_edges(values, count=4):
    """Quantile edges, so each bucket holds a comparable number of blocks."""
    if not values:
        return []
    quantiles = np.linspace(0.0, 100.0, count + 1)
    return sorted(set(float(v) for v in np.percentile(values, quantiles)))


def switch_rule(switch, key, buckets=4):
    """Where the grid preference flips, as a function of one variable.

    Quantiles, not just a median: the median ratio sits near one over the whole
    sweep, and reporting only that would say the two grids are interchangeable.
    They are not -- the ratio runs from about 0.75 to about 20 -- so what a
    switching rule has to predict is which tail a geometry lands in, and the
    median is exactly the statistic that hides it.
    """
    comparisons = switch["compared"]
    if not comparisons:
        return []
    edges = _bucket_edges([c[key] for c in comparisons], buckets)
    report = []
    for low, high in zip(edges, edges[1:]):
        inside = [c for c in comparisons if low <= c[key] <= high]
        if not inside:
            continue
        ratios = np.array([c["polar_over_cartesian"] for c in inside])
        report.append({
            "low": low, "high": high, "blocks": len(inside),
            "median_polar_over_cartesian": float(np.median(ratios)),
            "p10": float(np.percentile(ratios, 10)),
            "p90": float(np.percentile(ratios, 90)),
            "cartesian_wins": int((ratios > 1.0).sum()),
            # The decisive cases: where one grid is at least twice the other.
            "cartesian_wins_big": int((ratios > 2.0).sum()),
            "polar_wins_big": int((ratios < 0.5).sum()),
        })
    return report


def rho_curve(rows, target, engines):
    """Cost against source radius, at one delivered accuracy."""
    per_engine = defaultdict(lambda: defaultdict(list))
    radii = sorted({row["rho"] for row in rows})
    for row in rows:
        if not _usable(row, target):
            continue
        for engine in engines:
            best = cheapest_meeting(row, engine, target)
            if best:
                per_engine[engine][row["rho"]].append(best[0])
    return {
        "radii": radii,
        "engines": {
            engine: {
                str(rho): {
                    "blocks": len(costs),
                    "median_seconds_per_epoch": float(np.median(costs)),
                }
                for rho, costs in sorted(by_rho.items())
            }
            for engine, by_rho in per_engine.items()
        },
    }


def summarise(directory, profile=None):
    """One sweep directory, optionally restricted to a single source profile.

    Pooling the profiles is the wrong default for anything but a headline count.
    A limb-darkened source is a different integrand -- the grid engines carry a
    radial profile through the same cells, and VBM switches to its own
    limb-darkening machinery -- so a pooled median is an average over two
    regimes that need not agree, and a switching rule read off it would be a
    rule for neither.
    """
    rows = load(directory)
    if profile is not None:
        rows = [row for row in rows if row.get("profile") == profile]
    engines = sorted({
        entry.get("engine")
        for row in rows for entry in row.get("engines", [])
        if entry.get("engine")
    })
    switches = {
        target: grid_switch(rows, target) for target in ACCURACY_TARGETS
    }
    return {
        "directory": str(directory),
        "profile": profile or "all",
        "blocks": len(rows),
        "engines": engines,
        "pareto": pareto_table(rows, engines),
        "grid_switch": {
            str(target): {
                "compared": len(switch["compared"]),
                "cartesian_only": switch["cartesian_only"],
                "polar_only": switch["polar_only"],
                "neither": switch["neither"],
                "median_polar_over_cartesian": (
                    float(np.median([c["polar_over_cartesian"]
                                     for c in switch["compared"]]))
                    if switch["compared"] else None),
                "cartesian_wins": int(sum(
                    c["polar_over_cartesian"] > 1.0
                    for c in switch["compared"])),
                "by_rho": switch_rule(switch, "rho"),
                "by_mass_ratio": switch_rule(switch, "q"),
                "by_magnification": switch_rule(switch, "magnification"),
            }
            for target, switch in switches.items()
        },
        "rho_curves": {
            str(target): rho_curve(rows, target, engines)
            for target in ACCURACY_TARGETS
        },
    }


def _print(summary):
    print(f"=== profile: {summary['profile']}   blocks: {summary['blocks']}   "
          f"engines: {', '.join(summary['engines'])}")
    for target, entry in summary["pareto"].items():
        print(f"\n--- delivered accuracy {target:g}   "
              f"blocks judged {entry['blocks_judged']} "
              f"(excluded for a coarse reference: {entry['blocks_excluded']})")
        for engine, stat in sorted(
                entry["engines"].items(),
                key=lambda item: (item[1]["median_seconds_per_epoch"] is None,
                                  item[1]["median_seconds_per_epoch"] or 0.0)):
            median = stat["median_seconds_per_epoch"]
            p90 = stat["p90_seconds_per_epoch"]
            shown = "  --  " if median is None else f"{median * 1e3:7.3f}"
            shown90 = "  --  " if p90 is None else f"{p90 * 1e3:7.3f}"
            print(f"    {engine:22s} met {stat['blocks_met']:4d}  "
                  f"missed {stat['blocks_missed']:4d}   "
                  f"median {shown} ms/ep   p90 {shown90} ms/ep")
    for target, entry in summary["grid_switch"].items():
        median = entry["median_polar_over_cartesian"]
        if median is None:
            continue
        print(f"\n--- grid switch at {float(target):g}   compared {entry['compared']} "
              f"blocks   cartesian-only {entry['cartesian_only']}  "
              f"polar-only {entry['polar_only']}  neither {entry['neither']}")
        print(f"    median polar/cartesian time {median:.3f}   "
              f"cartesian cheaper on {entry['cartesian_wins']}/{entry['compared']}")
        for label, key in (("rho", "by_rho"), ("q", "by_mass_ratio"),
                           ("A", "by_magnification")):
            for band in entry[key]:
                print(f"      {label} {band['low']:.3e}..{band['high']:.3e}  "
                      f"n={band['blocks']:4d}  ratio p10/med/p90 "
                      f"{band['p10']:6.3f}/{band['median_polar_over_cartesian']:6.3f}"
                      f"/{band['p90']:7.3f}   "
                      f"cart>2x {band['cartesian_wins_big']:4d}  "
                      f"polar>2x {band['polar_wins_big']:4d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", default="")
    parser.add_argument("--profile", default="",
                        help="uniform, linear, or empty for every profile "
                             "reported separately")
    arguments = parser.parse_args()
    if arguments.profile:
        profiles = [arguments.profile]
    else:
        profiles = sorted({
            row.get("profile") for row in load(arguments.directory)
            if row.get("profile")
        })
    summaries = {}
    for profile in profiles:
        summary = summarise(arguments.directory, profile)
        summaries[profile] = summary
        _print(summary)
        print()
    if arguments.output:
        Path(arguments.output).write_text(json.dumps(summaries, indent=2))
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()

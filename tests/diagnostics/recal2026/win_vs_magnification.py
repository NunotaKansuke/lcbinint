#!/usr/bin/env python3
"""Where lcbinint beats VBM, as a function of magnification.

The headline number from the speed sweep -- one win rate over the whole corpus
-- describes the corpus, not the method.  The sweep samples geometry uniformly
in its own parameters, and roughly half its blocks land below A=3, where every
engine agrees and the contour is unbeatable.  Averaging over that says more
about how the positions were drawn than about when to use which integrator.

So the comparison is reported split by magnification.  Each block contributes
one number: the cheapest lcbinint setting that actually delivered the accuracy,
against the cheapest VBM setting that did, both taken from the same block and
the same reference ladder.  A block where neither engine reached the accuracy
is not a speed fact and is counted separately rather than folded in; a block
where only one did is reported as a robustness count, because dividing by a
failure would report it as infinite speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .speed_analysis import ACCURACY_TARGETS, GRIDS, _usable, cheapest_meeting, load

# Open at the top: the last bin is unbounded because the interesting claim is
# about the tail, and closing it would silently drop the blocks that motivated
# the whole split.
MAGNIFICATION_EDGES = (0.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, float("inf"))


def _ours(row, target):
    """Cheapest lcbinint setting meeting the target, over both grids.

    Both grids are offered because a user picks one per problem, and the honest
    comparison is against the better choice rather than against an arbitrary
    default.  The grid that won is carried along so the switching rule can be
    checked against the same blocks.
    """
    best = None
    for grid in GRIDS:
        found = cheapest_meeting(row, grid, target)
        if found is None:
            continue
        if best is None or found[0] < best[0]:
            best = (found[0], found[1], grid)
    return best


def compare(rows, target):
    """Per-block speed ratios at one delivered accuracy."""
    paired, ours_only, theirs_only, neither, unjudged = [], 0, 0, 0, 0
    for row in rows:
        if not _usable(row, target):
            unjudged += 1
            continue
        ours = _ours(row, target)
        theirs = cheapest_meeting(row, "vbm", target)
        if ours and theirs:
            paired.append({
                "magnification": row.get("magnification"),
                "rho": row["rho"], "s": row["s"], "q": row["q"],
                "ours_seconds": ours[0], "ours_knob": ours[1],
                "ours_grid": ours[2], "vbm_seconds": theirs[0],
                # Above one, lcbinint is the faster engine on this block.
                "speedup": theirs[0] / ours[0],
            })
        elif ours:
            ours_only += 1
        elif theirs:
            theirs_only += 1
        else:
            neither += 1
    return {"paired": paired, "lcbinint_only": ours_only,
            "vbm_only": theirs_only, "neither": neither,
            "unjudgeable": unjudged}


def _summarise(entries):
    speedups = np.array([e["speedup"] for e in entries], dtype=float)
    return {
        "blocks": int(speedups.size),
        "win_rate": float(np.mean(speedups > 1.0)),
        "median_speedup": float(np.median(speedups)),
        "p90_speedup": float(np.percentile(speedups, 90)),
        "median_vbm_ms": float(np.median(
            [e["vbm_seconds"] for e in entries]) * 1e3),
        "median_ours_ms": float(np.median(
            [e["ours_seconds"] for e in entries]) * 1e3),
        "polar_share": float(np.mean(
            [e["ours_grid"] == "lcbinint_polar" for e in entries])),
    }


def binned(comparison, key="magnification", edges=MAGNIFICATION_EDGES):
    """The comparison split into bins, plus the pooled row."""
    paired = [e for e in comparison["paired"] if e.get(key) is not None]
    report = []
    for low, high in zip(edges, edges[1:]):
        inside = [e for e in paired if low <= e[key] < high]
        if not inside:
            continue
        report.append({"low": low, "high": high, **_summarise(inside)})
    return {"bins": report,
            "all": _summarise(paired) if paired else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+",
                        help="sweep output directories to pool")
    parser.add_argument("--profile", default="linear")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    rows = []
    for directory in arguments.directories:
        found = [row for row in load(directory)
                 if row.get("profile") == arguments.profile]
        print(f"{len(found):5d} {arguments.profile} blocks  {directory}")
        rows.extend(found)
    print(f"{len(rows):5d} pooled\n")

    report = {"profile": arguments.profile,
              "directories": list(arguments.directories),
              "targets": {}}
    for target in ACCURACY_TARGETS:
        comparison = compare(rows, target)
        split = binned(comparison)
        report["targets"][f"{target:g}"] = {
            "counts": {k: v for k, v in comparison.items() if k != "paired"},
            **split,
        }
        print(f"target {target:g}   paired={len(comparison['paired'])} "
              f"lcbinint_only={comparison['lcbinint_only']} "
              f"vbm_only={comparison['vbm_only']} "
              f"neither={comparison['neither']}")
        print(f"  {'A range':>16} {'n':>5} {'win':>7} {'median':>8} "
              f"{'p90':>8} {'vbm ms':>8} {'ours ms':>8} {'polar':>6}")
        for entry in split["bins"]:
            label = (f"{entry['low']:g}-{entry['high']:g}"
                     if np.isfinite(entry["high"]) else f"{entry['low']:g}+")
            print(f"  {label:>16} {entry['blocks']:5d} "
                  f"{entry['win_rate']:6.1%} {entry['median_speedup']:8.2f} "
                  f"{entry['p90_speedup']:8.2f} {entry['median_vbm_ms']:8.3f} "
                  f"{entry['median_ours_ms']:8.3f} {entry['polar_share']:5.0%}")
        if split["all"]:
            entry = split["all"]
            print(f"  {'pooled':>16} {entry['blocks']:5d} "
                  f"{entry['win_rate']:6.1%} {entry['median_speedup']:8.2f} "
                  f"{entry['p90_speedup']:8.2f} {entry['median_vbm_ms']:8.3f} "
                  f"{entry['median_ours_ms']:8.3f} {entry['polar_share']:5.0%}")
        print()

    if arguments.output:
        Path(arguments.output).write_text(json.dumps(report, indent=2))
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()

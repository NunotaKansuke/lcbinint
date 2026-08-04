#!/usr/bin/env python3
"""microlux and the JAX backend, put on the speed sweep's own Pareto plane.

``sweep_ext`` runs one engine setting per process pass and writes each pass to
its own directory, so the first job here is to put a row back together: the
passes are joined on ``(case_id, profile, x)``, which is what identifies a row
once ``case_id`` alone does not -- a case contributes nine of them, one per
distance factor, and the two profiles double that.  The joined rows are then
merged with the speed sweep's blocks, so every engine in the comparison is
scored against the same stored reference rather than against one rebuilt here.

Two things make the join more than bookkeeping.

The first is the control.  ``sweep_ext`` cannot run at the speed sweep's
concurrency -- JAX holds compiled executables per process -- so its timings are
taken under lighter load than the numbers they are compared against.  Rather
than assume that difference away, both runs timed the same two native Cartesian
buckets, and their ratio is the scale factor between them, measured per row.
It is reported, not silently applied: a factor near one means the comparison
needs no correction, and a factor far from one means the cross-run comparison
should be stated with it.

The second is compilation.  The JAX backend takes its grid resolution as a
static argument derived from the geometry, so every new (s, q, rho) is a fresh
compilation -- 546 address-space mappings and seconds of wall time apiece.  The
per-epoch timings exclude it, which is right for a fit that evaluates thousands
of epochs per geometry and wrong for anything that does not, so the first-call
cost is carried through to the summary as its own column instead of being
folded in or dropped.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .speed_analysis import ACCURACY_TARGETS, _error, _usable, load

FAMILIES = ("microlux", "lcbinint_jax")
CONTROL = "lcbinint_cartesian_control"


def _key(row):
    """What identifies a row across passes and across the two runs."""
    return (row.get("case_id"), row.get("profile"), row.get("x"))


def load_ext(directory):
    """Every pass under ``directory``, joined back into one row per geometry.

    Passes are separate directories because they are separate processes; a row
    that a later pass censored still appears, carrying the cost that stopped it,
    so a censored point is never mistaken for a missing one.
    """
    root = Path(directory)
    passes = sorted(p for p in root.glob("*/") if p.is_dir())
    joined = {}
    seen = defaultdict(int)
    for pass_directory in passes:
        for path in sorted(pass_directory.glob("ext-*.json")):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            for row in payload.get("rows", []):
                if "error" in row:
                    continue
                key = _key(row)
                target = joined.setdefault(
                    key, {k: v for k, v in row.items() if k != "engines"})
                target.setdefault("engines", []).extend(row.get("engines", []))
                seen[pass_directory.name] += 1
    return list(joined.values()), dict(seen)


def cheapest_meeting(row, engine, target):
    """The cheapest setting of one engine that actually delivered ``target``.

    A setting that exhausted microlux's sampling budget is not a point on its
    accuracy curve -- it is the ceiling being hit, and its error is whatever the
    truncated contour happened to give -- so it is not eligible.
    """
    best = None
    for entry in row.get("engines", []):
        if entry.get("engine") != engine:
            continue
        if entry.get("budget_exhausted"):
            continue
        seconds = entry.get("seconds_per_epoch")
        if seconds is None:
            continue
        error = _error(entry)
        if error is None or error > target:
            continue
        if best is None or seconds < best[0]:
            best = (seconds, entry)
    return best


def control_scale(ext_rows, speed_rows):
    """How much faster this run was than the speed sweep, per row and bucket.

    The comparison is only ever between the same bucket on the same geometry,
    so what comes out is the machine-load difference between the two runs and
    nothing else.
    """
    stored = {}
    for row in speed_rows:
        for entry in row.get("engines", []):
            if entry.get("engine") != "lcbinint_cartesian":
                continue
            seconds = entry.get("seconds_per_epoch")
            if seconds is not None:
                stored[(_key(row), entry.get("knob"))] = seconds
    ratios = defaultdict(list)
    for row in ext_rows:
        for entry in row.get("engines", []):
            if entry.get("engine") != CONTROL:
                continue
            reference = stored.get((_key(row), entry.get("knob")))
            seconds = entry.get("seconds_per_epoch")
            if reference and seconds:
                ratios[entry.get("knob")].append(reference / seconds)
    return {
        str(bucket): {
            "n": len(values),
            "median_speed_sweep_over_ext": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }
        for bucket, values in sorted(ratios.items())
        if values
    }


def row_scale(ext_rows, speed_rows):
    """The cross-run load factor for each row, from its own control timings.

    ``control_scale`` answers whether a correction is needed; this answers what
    it is on the row being corrected.  Per row rather than one scalar because a
    scalar would assume the two runs differed by a constant, and that is exactly
    the thing worth checking rather than assuming.
    """
    stored = {}
    for row in speed_rows:
        for entry in row.get("engines", []):
            if entry.get("engine") != "lcbinint_cartesian":
                continue
            seconds = entry.get("seconds_per_epoch")
            if seconds is not None:
                stored[(_key(row), entry.get("knob"))] = seconds
    gathered = defaultdict(list)
    for row in ext_rows:
        for entry in row.get("engines", []):
            if entry.get("engine") != CONTROL:
                continue
            reference = stored.get((_key(row), entry.get("knob")))
            seconds = entry.get("seconds_per_epoch")
            if reference and seconds:
                gathered[_key(row)].append(reference / seconds)
    return {key: float(np.median(values)) for key, values in gathered.items()}


def head_to_head(ext_rows, speed_rows, profile, target, scales=None):
    """Every extra engine against lcbinint's own best grid, at equal accuracy.

    Two ratios come out, and the difference between them is the cross-run load
    factor.  The raw one divides by the native seconds the speed sweep stored.
    The calibrated one divides by what those same seconds are worth in this
    run, using the row's own control measurement -- which is the like-for-like
    number, and the reason the control pass was run at all.
    """
    scales = scales or {}
    speed = {_key(row): row for row in speed_rows}
    out = {"profile": profile, "target": target, "engines": {}}
    for family in FAMILIES:
        ratios, wins, compared, chosen_only = [], 0, 0, 0
        calibrated, calibrated_wins, calibrated_n = [], 0, 0
        first_call = []
        for row in ext_rows:
            if row.get("profile") != profile:
                continue
            partner = speed.get(_key(row))
            if partner is None or not _usable(partner, target):
                continue
            ours = cheapest_meeting(row, family, target)
            if ours is None:
                continue
            chosen_only += 1
            if ours[1].get("first_call_seconds") is not None:
                first_call.append(ours[1]["first_call_seconds"])
            native = None
            for grid in ("lcbinint_cartesian", "lcbinint_polar"):
                candidate = cheapest_meeting(partner, grid, target)
                if candidate and (native is None or candidate[0] < native[0]):
                    native = candidate
            if native is None:
                continue
            compared += 1
            ratios.append(ours[0] / native[0])
            wins += ours[0] < native[0]
            scale = scales.get(_key(row))
            if scale:
                # The stored native seconds are ``scale`` times what the same
                # work costs in this run, so the honest denominator is smaller
                # by that factor and the ratio larger by it.
                value = (ours[0] / native[0]) * scale
                calibrated.append(value)
                calibrated_wins += value < 1.0
                calibrated_n += 1
        entry = {
            "reached_target": chosen_only,
            "compared_against_native": compared,
            "win_rate": float(wins / compared) if compared else None,
            "median_ext_over_native":
                float(np.median(ratios)) if ratios else None,
            "p10": float(np.percentile(ratios, 10)) if ratios else None,
            "p90": float(np.percentile(ratios, 90)) if ratios else None,
            "calibrated_n": calibrated_n,
            "calibrated_win_rate":
                float(calibrated_wins / calibrated_n) if calibrated_n else None,
            "calibrated_median_ext_over_native":
                float(np.median(calibrated)) if calibrated else None,
            "calibrated_p10":
                float(np.percentile(calibrated, 10)) if calibrated else None,
            "calibrated_p90":
                float(np.percentile(calibrated, 90)) if calibrated else None,
        }
        if first_call:
            entry["median_first_call_seconds"] = float(np.median(first_call))
        out["engines"][family] = entry
    return out


def summarise(ext_directory, speed_directory):
    ext_rows, counts = load_ext(ext_directory)
    speed_rows = load(speed_directory)
    out = {
        "ext_directory": str(ext_directory),
        "speed_directory": str(speed_directory),
        "ext_rows": len(ext_rows),
        "speed_rows": len(speed_rows),
        "rows_per_pass": counts,
        "control_scale": control_scale(ext_rows, speed_rows),
        "head_to_head": [],
    }
    scales = row_scale(ext_rows, speed_rows)
    out["calibrated_rows"] = len(scales)
    for profile in ("uniform", "linear"):
        for target in ACCURACY_TARGETS:
            out["head_to_head"].append(
                head_to_head(ext_rows, speed_rows, profile, target, scales))
    return out


def _print(result):
    print(f"ext rows {result['ext_rows']}  speed rows {result['speed_rows']}")
    print("rows per pass: " + ", ".join(
        f"{name}={count}" for name, count in sorted(
            result["rows_per_pass"].items())))
    if result["control_scale"]:
        print("\ncontrol (speed sweep seconds / this run's seconds, same "
              "bucket and row):")
        for bucket, entry in result["control_scale"].items():
            print(f"  nbin {bucket:>4s}  n={entry['n']:4d}  median "
                  f"{entry['median_speed_sweep_over_ext']:.3f}  "
                  f"[{entry['p10']:.3f}, {entry['p90']:.3f}]")
    else:
        print("\ncontrol: no paired rows -- cross-run comparison uncalibrated")
    for block in result["head_to_head"]:
        print(f"\n===== {block['profile']} @ {block['target']:g} =====")
        for family, entry in block["engines"].items():
            if not entry["compared_against_native"]:
                print(f"  {family:14s} reached target on "
                      f"{entry['reached_target']} rows, none comparable")
                continue
            print(f"  {family:14s} n={entry['compared_against_native']:4d}  "
                  f"win {entry['win_rate']:6.1%}  "
                  f"median x{entry['median_ext_over_native']:.3f} vs native  "
                  f"[{entry['p10']:.3f}, {entry['p90']:.3f}]  (raw)")
            if entry.get("calibrated_n"):
                print(f"  {'':14s} n={entry['calibrated_n']:4d}  "
                      f"win {entry['calibrated_win_rate']:6.1%}  "
                      f"median x"
                      f"{entry['calibrated_median_ext_over_native']:.3f} "
                      f"vs native  [{entry['calibrated_p10']:.3f}, "
                      f"{entry['calibrated_p90']:.3f}]  (load-calibrated)")
            if "median_first_call_seconds" in entry:
                print(f"  {'':14s} first call (compile) median "
                      f"{entry['median_first_call_seconds']:.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ext_directory")
    parser.add_argument("speed_directory")
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()
    result = summarise(arguments.ext_directory, arguments.speed_directory)
    _print(result)
    if arguments.output:
        with open(arguments.output, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()

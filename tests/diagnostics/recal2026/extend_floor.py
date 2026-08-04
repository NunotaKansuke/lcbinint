#!/usr/bin/env python3
"""Uncensoring the coarse end of a finished resolution sweep.

The first pass measured a ladder whose coarsest rung was 16 bins, because that
is the shipping selector's floor, and found that 99.4% of geometries already
met a 1e-2 target there.  That is not the measurement it looks like.  A ladder
that starts at 16 cannot tell a row that genuinely needed 16 from one that
would have been fine with 4; it can only report the floor back.  Since the
component certificate decoupled correctness from grid density, the whole
question of how far the resolution can drop lives underneath that floor.

Re-running the sweep to answer it would be wasteful.  The expensive part of a
row is its reference -- a convergence ladder at 256 and 400 bins, an
independently discretised polar grid, and an escalating contour integration --
and all of that is already on disk.  What is missing is only a handful of very
cheap evaluations per row.  So this reopens each stored row, measures the new
coarse buckets against the reference that row already carries, merges them into
the stored ladder and recomputes the requirement over the full extended ladder.

The recomputation is exact rather than approximate: ``required_bucket`` asks for
the coarsest bucket that is inside the budget *and stays inside it at every
finer measured bucket*, so it needs the whole ladder present, and after the
merge it is.  The answer is identical to what a full re-run would have produced,
because the reference and the finer rungs are the same numbers either way.

Rows whose reference was never sharp enough for a tolerance stay unusable, and
rows whose ladder was censored for time keep their censoring: neither is
something a coarser bucket can fix.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from . import reference as reference_module
from .engines import BUCKETS, SHIPPING_BUCKET_FLOOR, lcbinint_fixed
from .sweep_resolution import TARGET_TOLERANCES

NEW_BUCKETS = tuple(b for b in BUCKETS if b < SHIPPING_BUCKET_FLOOR)


def extend_row(row):
    """Measure the new coarse buckets and recompute what the row required.

    Returns the number of evaluations actually run, so the caller can report
    the cost of the extension rather than assert it was small.
    """
    s, q, rho = row.get("s"), row.get("q"), row.get("rho")
    x, y = row.get("x"), row.get("y")
    profile_c = row.get("limb_darkening_c", 0.0)
    if None in (s, q, rho, x, y):
        return 0

    evaluations = 0
    for grid in ("cartesian", "polar"):
        ladder = row.get(grid)
        if not isinstance(ladder, dict):
            continue
        for bucket in NEW_BUCKETS:
            if str(bucket) in ladder or bucket in ladder:
                continue
            engine = lcbinint_fixed(grid, bucket, profile_c)
            started = time.perf_counter()
            try:
                result = engine(s, q, rho, x, y)
            except Exception as error:  # noqa: BLE001
                ladder[str(bucket)] = {
                    "error": f"{type(error).__name__}: {error}"}
                evaluations += 1
                continue
            ladder[str(bucket)] = {
                "magnification": result.magnification,
                "error_estimate": result.error_estimate,
                "converged": result.converged,
                "support_proven": result.support_proven,
                "method": result.method,
                "seconds": result.seconds,
                "measured_by": "extend_floor",
                "wall_seconds": time.perf_counter() - started,
            }
            evaluations += 1

    # The stored ladders came back from JSON with string keys; required_bucket
    # indexes by the integer bucket, so a view with integer keys is built rather
    # than rewriting the stored form, which stays as it was written.
    def as_integer_keyed(ladder):
        out = {}
        for key, value in (ladder or {}).items():
            try:
                out[int(key)] = value
            except (TypeError, ValueError):
                out[key] = value
        return out

    built = row.get("reference") or {}
    for tolerance in TARGET_TOLERANCES:
        key = str(tolerance)
        entry = (row.get("required") or {}).get(key)
        if not entry:
            continue
        if not reference_module.usable_for(built, tolerance):
            continue
        for grid in ("cartesian", "polar"):
            entry[grid] = reference_module.required_bucket(
                as_integer_keyed(row.get(grid)), built["value"], tolerance)
    return evaluations


def extend_file(path):
    payload = json.loads(Path(path).read_text())
    evaluations = 0
    for row in payload.get("rows", []):
        if "error" in row:
            continue
        evaluations += extend_row(row)
    payload["floor_extension"] = {
        "buckets": list(NEW_BUCKETS),
        "evaluations": evaluations,
    }
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)
    return evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    paths = []
    for directory in arguments.directories:
        paths.extend(sorted(Path(directory).glob("case-*.json")))

    import multiprocessing

    started = time.perf_counter()
    total = 0
    with multiprocessing.Pool(arguments.workers) as pool:
        for done, count in enumerate(
                pool.imap_unordered(extend_file, paths), start=1):
            total += count
            print(f"[{done}/{len(paths)}] {count} evaluations", flush=True)
    print(f"{total} evaluations in "
          f"{(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

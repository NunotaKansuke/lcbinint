#!/usr/bin/env python3
"""What the seeding stage costs, and what removing parts of it changes.

Two questions, deliberately measured apart.

**Cost.**  The probes are a fixed per-evaluation charge: their count depends on
the caustic geometry, not on the integration grid.  So their share of an
evaluation is not a single number, it is a function of the resolution, and it
grows as the resolution falls.  That matters now precisely because the
component certificate let the resolution fall -- the campaign's new rule picks
4 to 16 bins at a 1e-2 target where the shipping rule picked 100 -- so a charge
that was noise against a 200-bin grid need not be noise against a 6-bin one.
Counts are reported alongside seconds because counts are immune to the harness:
a per-call timing carries ~1.4 ms of setup, and a probe count does not.

**Safety.**  Cost only licenses removing a stage if the answer survives it, and
the answer is not tested by re-running the geometries the rings already handle.
``probe_corpus`` supplies caps two orders of magnitude thinner than the finest
ring step, which is where a heuristic can only be right by luck.  Each ablation
is compared against the full policy at the same resolution -- a missing image
component is a discrete drop in magnification, not a small error -- and against
VBMicrolensing as an independent witness, since agreeing with the full policy
is only reassuring if the full policy is itself right.

One policy per process: the policy is read from the environment once, which is
what keeps it out of the hot path.  ``probe_sweep.sh`` drives the set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import probe_build

lcbinint = probe_build.activate()

from .probe_corpus import rows as corpus_rows  # noqa: E402

# Resolutions to measure at.  The first three are the constants this campaign's
# resolution study settled on for the Cartesian grid at 1e-2, 1e-3 and 1e-4;
# the last is far above any of them and is there as the internal reference,
# where a seeding failure cannot hide behind a coarse grid.
STUDY_BUCKETS = (16, 50, 128)
REFERENCE_BUCKET = 400

COUNTER_KEYS = (
    "ring_solves", "certified_solves", "certified_offered", "certified_extrema",
    "certifications", "unproven", "ring_seconds", "certified_seconds",
    "certify_seconds",
)


def _scalar(value):
    return float(np.asarray(value).ravel()[0])


def _curve(nbin, profile_c=0.0):
    return lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        coordinates="vbm", nbin=nbin, caustic_bins=1400,
        inverse_ray_grid="cartesian", max_source_bins=nbin,
        point_source_threshold=0.0, hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0))


def _evaluate(curve, row, profile_c=0.0):
    """One position, with the probe tallies it spent."""
    native = lcbinint._lcbinint
    native.reset_probe_counters()
    started = time.perf_counter()
    info = curve.info(
        row["x"], t0=0.0, tE=1.0, u0=row["y"], alpha=0.0,
        s=row["s"], q=row["q"], rho=row["rho"], limb_darkening_c=profile_c)
    elapsed = time.perf_counter() - started
    counters = native.probe_counters()
    error = _scalar(info.finite_source_error_estimates)
    return {
        "magnification": _scalar(info.finite_source_magnifications),
        "error_estimate": error,
        "converged": bool(_scalar(info.finite_source_converged)),
        "support_proven": math.isfinite(error),
        "method": str(np.asarray(info.finite_source_method_names).ravel()[0]),
        "caustic_distance": _scalar(info.caustic_distances),
        "point_magnification": _scalar(info.point_source_magnifications),
        "image_count": int(_scalar(info.image_counts)),
        "call_seconds": elapsed,
        "counters": {key: counters[key] for key in COUNTER_KEYS},
    }


def _vbm_witness(row, tol=1.0e-5):
    """An independent value, from an implementation that shares no code.

    Agreement with the full policy would otherwise only show that an ablation
    reproduces whatever the full policy does, including its mistakes.  The
    x-mirror is VBM's frame convention relative to ``coordinates='vbm'``;
    ``frames.py`` measures it rather than assuming it.
    """
    try:
        import VBMicrolensing
    except ImportError:
        return None
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = 1.0e-12
    vbm.RelTol = tol
    try:
        value = float(vbm.BinaryMag2(
            row["s"], row["q"], -row["x"], row["y"], row["rho"]))
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}
    return {"magnification": value, "reltol": tol}


def run(output, *, witness=True, buckets=STUDY_BUCKETS, reference=True,
        limit=0, stride=1):
    native = lcbinint._lcbinint
    policy = native.probe_counters()["policy"]
    if not native.probe_counters()["enabled"]:
        raise RuntimeError("set LCBININT_PROBE_STATS=1")

    rows = corpus_rows()[::max(stride, 1)]
    if limit:
        rows = rows[:limit]
    # One curve per resolution, reused across the corpus: the caustic cache is
    # keyed on the lens inside the curve, so rebuilding per row would measure
    # cache construction instead of seeding.
    curves = {nbin: _curve(nbin) for nbin in buckets}
    if reference:
        curves[REFERENCE_BUCKET] = _curve(REFERENCE_BUCKET)

    started = time.perf_counter()
    for index, row in enumerate(rows):
        row["measured"] = {}
        for nbin, curve in curves.items():
            try:
                row["measured"][str(nbin)] = _evaluate(curve, row)
            except Exception as error:  # noqa: BLE001
                row["measured"][str(nbin)] = {
                    "error": f"{type(error).__name__}: {error}"}
        if witness:
            row["vbm"] = _vbm_witness(row)
        if (index + 1) % 50 == 0:
            rate = (index + 1) / (time.perf_counter() - started)
            print(f"[{index + 1}/{len(rows)}] {rate:.2f} rows/s", flush=True)

    payload = {
        "policy": policy,
        "buckets": list(buckets),
        "reference_bucket": REFERENCE_BUCKET if reference else None,
        "rows": rows,
        "seconds": time.perf_counter() - started,
    }
    Path(output).write_text(json.dumps(payload))
    print(f"wrote {output} in {payload['seconds'] / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-witness", action="store_true")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    arguments = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run(arguments.output, witness=not arguments.no_witness,
        reference=not arguments.no_reference,
        limit=arguments.limit, stride=arguments.stride)


if __name__ == "__main__":
    main()

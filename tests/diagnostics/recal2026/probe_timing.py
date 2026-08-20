#!/usr/bin/env python3
"""The seeding share of a light curve, measured where the setup cost is gone.

``probe_study`` evaluates one position per entry into the library, which is the
right unit for asking whether an ablation changed an answer and the wrong one
for asking what it costs: a single entry carries ~1.4 ms of fixed setup, and a
share taken against that denominator understates the seeding stage by as much
as the setup dominates.  Light-curve blocks are the unit every other timing in
this campaign uses, so they are the unit here.

Blocks are placed so that a controlled fraction of their epochs is inside the
caustic.  A trajectory that never crosses spends nothing on seeding, and one
that spends every epoch inside is not a light curve anyone fits; reporting the
share at several crossing fractions is what makes the number transferable.

Run this when the machine is otherwise quiet, and one policy per process --
the policy is read from the environment once.
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

from .probe_corpus import _branch_extent, _turning_angles  # noqa: E402
from .geometry import caustic_branches  # noqa: E402

# Resolutions spanning the old rule's choice and the new one's.  The seeding
# cost is flat in the resolution and the grid cost is quadratic, so the share
# is a decreasing function of this axis and one value of it would be a claim
# about one operating point rather than about the library.
TIMED_BUCKETS = (4, 8, 16, 32, 50, 100, 200)

EPOCHS = 64
REPEAT = 5


def _blocks():
    """Trajectories that cross the caustic, one per lens in a hard spread."""
    out = []
    for s, q, rho_factor in (
            (1.05, 1.0e-3, 0.05), (1.30, 1.0e-4, 0.05), (0.80, 1.0e-2, 0.05),
            (1.00, 1.0e-3, 0.05), (1.30, 1.0e-4, 1.00), (1.10, 1.0e-5, 0.30),
            (2.00, 1.0e-1, 0.05), (0.60, 3.0e-1, 0.30)):
        branches = caustic_branches(s, q)
        if not branches:
            continue
        branch = max(branches, key=_branch_extent)
        extent = _branch_extent(branch)
        rho = extent * rho_factor
        if not (1.0e-6 < rho < 1.0):
            continue
        centre = branch.mean(axis=0)
        # A horizontal cut through the branch centroid, spanning it, so the
        # block enters and leaves the caustic once rather than grazing it.
        half = 0.75 * extent
        out.append({
            "s": s, "q": q, "rho": rho,
            "u0": float(centre[1]),
            "times": np.linspace(centre[0] - half, centre[0] + half, EPOCHS),
        })
    return out


def _curve(nbin):
    return lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        coordinates="vbm", nbin=nbin, caustic_bins=1400,
        inverse_ray_grid="cartesian", max_source_bins=nbin,
        point_source_threshold=0.0, hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0))


def measure(block, nbin):
    native = lcbinint._lcbinint
    curve = _curve(nbin)
    parameters = dict(t0=0.0, tE=1.0, u0=block["u0"], alpha=0.0,
                      s=block["s"], q=block["q"], rho=block["rho"],
                      limb_darkening_c=0.0)
    info = curve.info(block["times"], **parameters)  # warm the caustic cache
    magnifications = np.asarray(info.finite_source_magnifications, dtype=float)
    distances = np.asarray(info.caustic_distances, dtype=float) / block["rho"]
    methods = [str(m) for m in np.asarray(info.finite_source_method_names).ravel()]

    samples = []
    for _ in range(REPEAT):
        native.reset_probe_counters()
        started = time.perf_counter()
        curve.info(block["times"], **parameters)
        elapsed = time.perf_counter() - started
        counters = native.probe_counters()
        samples.append((elapsed, counters["ring_seconds"],
                        counters.get("heuristic_seconds", 0.0),
                        counters["certified_seconds"], counters["certify_seconds"],
                        counters["ring_solves"],
                        counters.get("heuristic_solves", 0),
                        counters["certified_solves"]))
    # The median pass, ranked on total time: taking medians of each column
    # separately would report a split that no single pass produced.
    samples.sort(key=lambda item: item[0])
    total, ring, heuristic, certified, certify, ring_solves, heuristic_solves, cert_solves = \
        samples[len(samples) // 2]
    return {
        "nbin": nbin,
        "seconds_per_epoch": total / len(block["times"]),
        "ring_share": ring / total,
        "branch_heuristic_share": heuristic / total,
        "certified_probe_share": certified / total,
        "certify_support_share": certify / total,
        "seeding_share": (ring + heuristic + certified + certify) / total,
        "ring_solves": ring_solves,
        "branch_heuristic_solves": heuristic_solves,
        "certified_solves": cert_solves,
        "epochs_inside": int(np.sum(distances < 1.0)),
        "epochs": len(block["times"]),
        "grid_epochs": sum(1 for m in methods if m.startswith("inverse_ray")),
        "magnification_median": float(np.median(magnifications)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--buckets", default=",".join(str(b) for b in TIMED_BUCKETS))
    arguments = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    native = lcbinint._lcbinint
    if not native.probe_counters()["enabled"]:
        raise RuntimeError("set LCBININT_PROBE_STATS=1")
    buckets = [int(b) for b in arguments.buckets.split(",") if b]

    results = []
    for block in _blocks():
        for nbin in buckets:
            entry = measure(block, nbin)
            entry.update({"s": block["s"], "q": block["q"], "rho": block["rho"]})
            results.append(entry)
            print(f"s={block['s']} q={block['q']:g} rho={block['rho']:.3g} "
                  f"nbin={nbin:3d}  {entry['seconds_per_epoch']*1e3:8.3f} ms/epoch  "
                  f"seeding {entry['seeding_share']*100:5.1f}%  "
                  f"(ring {entry['ring_share']*100:4.1f} "
                  f"cert {entry['certified_probe_share']*100:4.1f} "
                  f"support {entry['certify_support_share']*100:4.1f})", flush=True)
    Path(arguments.output).write_text(json.dumps({
        "policy": native.probe_counters()["policy"],
        "buckets": buckets,
        "epochs": EPOCHS,
        "repeat": REPEAT,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()

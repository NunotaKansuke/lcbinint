#!/usr/bin/env python3
"""Where does lcbinint's inverse ray beat VBM's contour, and by how much?

Integration against integration, and nothing else.  A library's shipping entry
point answers an easy position with a hexadecapole or a point source and never
builds a grid or a contour, so timing that and calling it "inverse-ray speed"
times the absence of an inverse ray.  Worse, the shortcuts do not fire in the
same places for the two libraries, so a mixture is not even a consistent
mixture: it reads as a speed difference when it is a routing difference.

That is not hypothetical here.  ``engines.VbmEngine`` -- correct for what it
was built for, a comparison against lcbinint's shipping automatic path -- calls
``BinaryMag2`` for uniform sources, and ``BinaryMag2`` applies a quadrupole
test before deciding whether to integrate.  Timed against lcbinint's
``force_grid=True``, which cannot take any shortcut, that is a contour with an
escape hatch versus a grid without one.  This module closes the hatch:
``BinaryMag`` is the same contour with no test in front of it.

Uniform sources only.  The limb-darkened comparison is already fair in the main
sweep, because ``BinaryMagDark`` has no quadrupole test to skip.

Positions sit on the caustic's own normal at controlled multiples of ``rho``,
because that is where an integrator is actually exercised and because it makes
the distance to the caustic an axis rather than an accident.  Bin counts come
from the campaign's own ``nbin_rule.json`` rather than being invented here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .engines import LcbinintEngine, lcbinint_fixed

LENS_S = 1.1
LENS_Q = 1.0e-3

SOURCE_RADII = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)

# Distance from the caustic, in source radii.  Zero is excluded deliberately: a
# tangency is a measure-zero configuration whose cost is set by how close the
# sampler happened to land, not by the geometry.
DISTANCE_FACTORS = (0.1, 0.3, 1.0, 3.0)

TOLERANCES = (1.0e-2, 1.0e-3, 1.0e-4)

GRIDS = ("cartesian", "polar")

# VBM's contour takes an absolute accuracy.  Pinning it below anything
# reachable leaves RelTol as the binding criterion, which is the convention the
# rest of the campaign uses and the one that does not penalise VBM at high
# magnification, where an absolute goal would demand far more than the relative
# one the grids are being asked for.
VBM_ABSOLUTE_FLOOR = 1.0e-12

# The reference.  VBM rather than lcbinint, so that "lcbinint met its
# tolerance" is not graded against lcbinint's own idea of the answer.
REFERENCE_RELTOL = 1.0e-9

# VBM and lcbinint disagree on the sign of x; frames.verify() measures this for
# the main sweep and refuses to run if it stops being true.
X_SIGN = -1.0

# Positions are timed in blocks rather than one at a time, and the block is a
# short segment of a light curve at fixed impact parameter -- the same 0.4-rho
# span the main sweep uses.
#
# Not a stylistic choice.  A single evaluation carries about 1 ms of per-call
# setup, measured as the offset in a block-size scan (rho=0.1, nbin=16: 1.93 ms
# at N=1 falling to 0.87 ms/position by N=64; the cost is linear in N with an
# offset, not superlinear).  VBM's contour has no such setup, so timing both
# one position at a time would compare lcbinint's setup against VBM's
# integration -- at 0.03 ms per contour the offset alone would decide every row.
#
# Repeating one position instead of using distinct ones does not work either:
# an identical position is up to 63x cheaper than a neighbour 1e-6 rho away
# whose magnification agrees to 3e-7, so a repeated block measures a cache.
BLOCK = 16
BLOCK_SPAN_IN_RADII = 0.4

# Bins to try on each grid.  A ladder rather than the single constant from
# nbin_rule.json, because that constant is the 99%-coverage value -- what you
# would ship when you cannot inspect the position -- and spending it at every
# position would hand VBM the comparison by handicap: VBM adapts its own
# subdivision per point, so lcbinint has to be allowed to as well.  What is
# compared is the cheapest setting of each engine that actually hit the target.
NBIN_LADDER = (4, 8, 16, 32, 64, 128, 256, 400)

NBIN_RULE_PATH = "tests/diagnostics/results/recal2026/nbin_rule.json"

# The 99%-coverage constants from nbin_rule.json as it stood when this was
# written, kept as a fallback so the benchmark still runs without the results
# directory.  The file wins when it is present.
NBIN_FALLBACK = {
    ("cartesian", 1.0e-2): 16, ("cartesian", 1.0e-3): 50,
    ("cartesian", 1.0e-4): 128,
    ("polar", 1.0e-2): 24, ("polar", 1.0e-3): 100, ("polar", 1.0e-4): 320,
}


def nbin_rule(path=NBIN_RULE_PATH):
    """Bins per grid and tolerance, from the campaign's own fitted rule."""
    try:
        with open(path) as stream:
            payload = json.load(stream)
    except OSError:
        return dict(NBIN_FALLBACK)
    out = {}
    for grid, entry in payload.get("grids", {}).items():
        for tolerance, values in entry.get("tolerances", {}).items():
            if "constant_bins" in values:
                out[(grid, float(tolerance))] = int(values["constant_bins"])
    return out or dict(NBIN_FALLBACK)


def caustic_positions(rho, factors=DISTANCE_FACTORS, per_factor=10,
                      seed=20260806):
    """Points at ``factor * rho`` along the caustic's outward normal.

    Sampled around the whole caustic rather than at a chosen fold or cusp: the
    cost of an integration varies along the curve -- a cusp has more images
    arriving and leaving than a fold does -- and picking one feature by hand
    would hide that variation while looking like precision.
    """
    from .geometry import caustic_branches

    branches = caustic_branches(LENS_S, LENS_Q)
    rng = np.random.default_rng(seed)
    out = []
    for factor in factors:
        for _ in range(per_factor):
            branch = branches[rng.integers(len(branches))]
            index = int(rng.integers(len(branch)))
            point = branch[index]
            following = branch[(index + 1) % len(branch)]
            tangent = following - point
            norm = math.hypot(tangent[0], tangent[1])
            if norm <= 0.0:
                continue
            normal = np.array([-tangent[1] / norm, tangent[0] / norm])
            # Both signs are offered because which one leaves the caustic
            # depends on where on the branch the sample landed, and guessing
            # would bias the set toward disk interiors.
            sign = 1.0 if rng.random() < 0.5 else -1.0
            position = point + sign * factor * rho * normal
            out.append({"x": float(position[0]), "y": float(position[1]),
                        "rho": float(rho), "distance_factor": float(factor)})
    return out


def _timed(call, repeat=3):
    """Median of ``repeat`` warm calls, and the value.

    One untimed call first: the engines here build caustic caches on demand,
    and the first call would otherwise report that instead of the integration.
    """
    value = call()
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        value = call()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples)), value


class VbmContour:
    """VBM's contour with no quadrupole test in front of it.

    ``BinaryMag`` is the raw contour; ``BinaryMag2`` is the one that tries a
    multipole first and skips the contour when it succeeds.  Only the first is
    comparable against a grid that has been forbidden to take shortcuts.
    """

    def __init__(self, reltol):
        import VBMicrolensing

        self._vbm = VBMicrolensing.VBMicrolensing()
        self._vbm.Tol = VBM_ABSOLUTE_FLOOR
        self._vbm.RelTol = reltol

    def block(self, rho, xs, y, repeat=3):
        """The same positions the grid is given, timed the same way."""
        def call():
            return [self._vbm.BinaryMag(LENS_S, LENS_Q, X_SIGN * x, y, rho,
                                        VBM_ABSOLUTE_FLOOR) for x in xs]

        seconds, values = _timed(call, repeat)
        return seconds / len(xs), np.asarray(values, dtype=float)


def run(per_factor=10, repeat=3, block=BLOCK):
    """Every engine setting on every block.  Returns one row per block."""
    reference_engine = VbmContour(REFERENCE_RELTOL)
    vbm = {tol: VbmContour(tol) for tol in TOLERANCES}
    grids = {(grid, nbin): lcbinint_fixed(grid, nbin, 0.0)
             for grid in GRIDS for nbin in NBIN_LADDER}
    offsets = np.linspace(-0.5, 0.5, block) * BLOCK_SPAN_IN_RADII

    rows = []
    for rho in SOURCE_RADII:
        for probe in caustic_positions(rho, per_factor=per_factor):
            y = probe["y"]
            xs = probe["x"] + offsets * rho
            _, truth = reference_engine.block(rho, xs, y, repeat=1)
            if not np.all(np.isfinite(truth)) or np.any(truth <= 0.0):
                continue
            row = dict(probe, reference=float(np.median(truth)))

            for tol in TOLERANCES:
                seconds, values = vbm[tol].block(rho, xs, y, repeat)
                row[f"vbm/{tol:g}"] = {
                    "seconds": seconds,
                    "error": float(np.max(np.abs(values - truth) / truth))}

            # The grid is given the block as a light curve, which is how it is
            # actually used and what lets its setup amortise the way VBM has
            # nothing to amortise.
            for grid in GRIDS:
                for nbin in NBIN_LADDER:
                    engine = grids[(grid, nbin)]
                    curve = engine._ensure()
                    keywords = dict(t0=0.0, tE=1.0, u0=y, alpha=0.0,
                                    s=LENS_S, q=LENS_Q, rho=rho,
                                    limb_darkening_c=0.0)
                    seconds, values = _timed(
                        lambda: np.asarray(curve.magnification(xs, **keywords)),
                        repeat)
                    if not np.all(np.isfinite(values)):
                        continue
                    error = float(np.max(np.abs(values - truth) / truth))
                    row[f"{grid}/{nbin}"] = {
                        "seconds": seconds / len(xs), "error": error,
                        "nbin": nbin}
                    # Finer grids cost more and are only wanted for their
                    # accuracy, so once the tightest target is met the rest of
                    # the ladder cannot win any comparison and is not run.
                    if error <= min(TOLERANCES):
                        break

            centre = engine(LENS_S, LENS_Q, rho, probe["x"], y, time_it=False)
            row["point_magnification"] = centre.extra["point_magnification"]
            row["caustic_distance"] = centre.extra["caustic_distance"]
            rows.append(row)
    return rows


def cheapest_meeting(row, prefixes, target):
    """Fastest setting of an engine that actually hit ``target``.

    Comparing engines at equal *requested* tolerance would compare requests,
    not results, and the two do not track each other: a grid that overshoots
    its tolerance is paying for accuracy the comparison never credits it with,
    and one that undershoots is being timed on a job it did not finish.
    """
    best = None
    for key, entry in row.items():
        if not isinstance(entry, dict) or "/" not in str(key):
            continue
        if key.split("/")[0] not in prefixes:
            continue
        if not math.isfinite(entry["error"]) or entry["error"] > target:
            continue
        if best is None or entry["seconds"] < best["seconds"]:
            best = dict(entry, setting=key)
    return best


def report(rows, targets=(1.0e-2, 1.0e-3, 1.0e-4)):
    """Where the grid wins, split by the axes that were varied."""
    lines = []
    for target in targets:
        lines.append(f"\n=== error target {target:g} "
                     f"(uniform, s={LENS_S}, q={LENS_Q:g}) ===")
        lines.append(f"{'rho':>8}{'d/rho':>7}{'n':>4}{'A_fs':>10}{'A_pt':>10}"
                     f"{'vbm ms':>9}{'best ms':>9}{'grid':>10}{'speedup':>9}"
                     f"{'win':>6}")
        buckets = {}
        for row in rows:
            buckets.setdefault((row["rho"], row["distance_factor"]),
                               []).append(row)
        for key in sorted(buckets):
            group = buckets[key]
            ratios, vbm_ms, our_ms, winners, mags, points = [], [], [], [], [], []
            for row in group:
                best_vbm = cheapest_meeting(row, ("vbm",), target)
                entry = cheapest_meeting(row, GRIDS, target)
                if best_vbm is None or entry is None:
                    continue
                ratios.append(best_vbm["seconds"] / entry["seconds"])
                vbm_ms.append(best_vbm["seconds"] * 1e3)
                our_ms.append(entry["seconds"] * 1e3)
                winners.append(entry["setting"])
                mags.append(row["reference"])
                points.append(row.get("point_magnification") or float("nan"))
            if not ratios:
                continue
            speedup = float(np.median(ratios))
            grid = max(set(winners), key=winners.count)
            lines.append(
                f"{key[0]:8.3g}{key[1]:7.2g}{len(ratios):4d}"
                f"{np.median(mags):10.2f}{np.nanmedian(points):10.2f}"
                f"{np.median(vbm_ms):9.3f}{np.median(our_ms):9.3f}"
                f"{grid:>10}{speedup:9.2f}"
                f"{np.mean(np.array(ratios) > 1.0):6.0%}")

        # Which magnification predicts the speed ratio better?  Asked rather
        # than assumed: the finite-source value is what the integrator actually
        # has to resolve, but the point-source value is what a scheduler would
        # have available before deciding.
        pairs = []
        for row in rows:
            best_vbm = cheapest_meeting(row, ("vbm",), target)
            entry = cheapest_meeting(row, GRIDS, target)
            point = row.get("point_magnification")
            if best_vbm is None or entry is None or point is None:
                continue
            pairs.append((row["reference"], point,
                          best_vbm["seconds"] / entry["seconds"]))
        if len(pairs) > 8:
            array = np.array(pairs)
            keep = np.all(np.isfinite(array), axis=1) & (array[:, 1] > 0)
            array = array[keep]
            finite = np.corrcoef(np.log(array[:, 0]), np.log(array[:, 2]))[0, 1]
            pointwise = np.corrcoef(np.log(array[:, 1]), np.log(array[:, 2]))[0, 1]
            lines.append(f"  log-log correlation with speedup:  "
                         f"finite-source A {finite:+.3f}   "
                         f"point-source A {pointwise:+.3f}   (n={len(array)})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", default="40")
    parser.add_argument("--per-factor", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output")
    arguments = parser.parse_args()

    cores = {int(item) for item in arguments.cores.split(",")}
    os.sched_setaffinity(0, cores)
    print(f"pinned to cores {sorted(cores)}, load before start: "
          f"{os.getloadavg()[0]:.1f}", flush=True)

    started = time.perf_counter()
    rows = run(per_factor=arguments.per_factor, repeat=arguments.repeat)
    print(f"{len(rows)} positions in {time.perf_counter() - started:.0f} s",
          flush=True)
    print(report(rows))

    if arguments.output:
        with open(arguments.output, "w") as stream:
            json.dump({"lens": {"s": LENS_S, "q": LENS_Q},
                       "profile": "uniform", "rows": rows}, stream, indent=1)
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()

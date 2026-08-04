"""Lens cases and source positions for the recalibration sweeps.

The quantity every rule in this campaign turns on is the distance from the
source centre to the nearest caustic, in units of the source radius.  Sampling
positions uniformly in the plane would spend almost all of its trials far from
any caustic, where every method already agrees, so positions are instead placed
at intended multiples of ``rho`` from a sampled caustic point.

Intended is not achieved: stepping along a polyline normal lands near the
target but not on it, and near a cusp it can land beside a different branch
entirely.  Nothing downstream uses the intended distance.  The sweeps record
what lcbinint's own refined segment distance measures at the position that was
actually reached, and bin on that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import lcbinint


# Separations chosen so the close/resonant/wide topology boundaries are
# straddled rather than sampled around: the caustic count changes across them,
# and with it which method can be correct.
ANCHOR_SEPARATIONS = (0.3, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0, 3.0)
ANCHOR_MASS_RATIOS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0)

# Source radii.  The upper end is deliberately far above the 0.1 ceiling of the
# 2026-07 sweep: disks that swallow a planetary caustic whole are exactly the
# regime the retired resolvability guard made unreachable, and the certificate
# is what makes them answerable now.
RADIUS_RANGE = (3.0e-5, 1.0)

# Intended caustic distances in source radii.  Zero means "centre on the
# caustic"; the band from 0.8 to 1.35 is where the limb touches, which is where
# the source-plane quadrature routes live.
DISTANCE_FACTORS = (
    0.0, 0.25, 0.5, 0.8, 0.95, 1.0, 1.1, 1.35, 1.7, 2.0, 3.0, 5.0, 10.0, 30.0,
)


@dataclass(frozen=True)
class LensCase:
    case_id: int
    separation: float
    mass_ratio: float
    source_radius: float

    def as_dict(self):
        return {
            "case_id": self.case_id,
            "s": self.separation,
            "q": self.mass_ratio,
            "rho": self.source_radius,
        }


def _log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def make_lens_cases(count, seed):
    """Topology anchors first, then a log-space fill.

    The anchors guarantee that a fixed seed still covers every topology even
    when ``count`` is small, which matters because the holdout runs at a
    different seed and must remain comparable.
    """
    rng = np.random.default_rng(seed)
    cases = []
    anchors = [
        (s, q)
        for s in ANCHOR_SEPARATIONS
        for q in ANCHOR_MASS_RATIOS
    ]
    rng.shuffle(anchors)
    radii = np.exp(np.linspace(
        math.log(RADIUS_RANGE[0]), math.log(RADIUS_RANGE[1]), 9))
    for index in range(min(count, len(anchors))):
        s, q = anchors[index]
        cases.append(LensCase(
            index, float(s), float(q), float(radii[index % len(radii)])))
    while len(cases) < count:
        index = len(cases)
        cases.append(LensCase(
            index,
            _log_uniform(rng, 0.2, 4.0),
            _log_uniform(rng, 1.0e-6, 1.0),
            _log_uniform(rng, *RADIUS_RANGE),
        ))
    return cases


def caustic_branches(separation, mass_ratio):
    """Phase-ordered caustic polylines in lcbinint's own frame."""
    curve = lcbinint.LightCurve(
        lens="binary", options=lcbinint.Options(coordinates="vbm"))
    branches = curve.caustics(s=separation, q=mass_ratio)
    out = []
    for xs, ys in zip(branches.x, branches.y):
        points = np.column_stack((
            np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) >= 3:
            out.append(points)
    return out


def sample_positions(case, branches, rng, per_case):
    """Positions at intended multiples of ``rho`` from sampled caustic points.

    Each position steps along the local outward normal of a randomly chosen
    caustic segment.  The sign of "outward" is not determined here: both signs
    are offered and one is chosen at random, because which one leaves the
    caustic depends on where on the branch the sample landed, and guessing
    wrong would quietly bias the sample toward disk interiors.
    """
    if not branches:
        return []
    positions = []
    rho = case.source_radius
    # Cycle the factors from a rotating start rather than iterating them in
    # order.  Taking them in order means a case that only affords six
    # positions never leaves the first six factors, which silently removes the
    # 3--30 rho band -- the band where the hexadecapole route lives -- from the
    # entire sample.
    offset = int(rng.integers(len(DISTANCE_FACTORS)))
    factors = [
        DISTANCE_FACTORS[(offset + i) % len(DISTANCE_FACTORS)]
        for i in range(len(DISTANCE_FACTORS))
    ]
    while len(positions) < per_case:
        for factor in factors:
            if len(positions) >= per_case:
                break
            branch = branches[rng.integers(len(branches))]
            index = int(rng.integers(len(branch)))
            point = branch[index]
            following = branch[(index + 1) % len(branch)]
            tangent = following - point
            norm = math.hypot(tangent[0], tangent[1])
            if norm <= 0.0:
                continue
            normal = np.array([-tangent[1] / norm, tangent[0] / norm])
            sign = 1.0 if rng.random() < 0.5 else -1.0
            # A little jitter along the branch keeps repeated factors from
            # stacking on the same handful of vertices.
            jitter = (rng.random() - 0.5) * 0.5 * rho
            offset = point + sign * factor * rho * normal + jitter * (tangent / norm)
            positions.append({
                "x": float(offset[0]),
                "y": float(offset[1]),
                "intended_distance_factor": float(factor),
            })
    # A few genuine field points, so the sweep is not entirely caustic-relative
    # and the far-field point-source exits are exercised at their real spacing.
    span = 2.0 + case.separation
    for _ in range(max(1, per_case // 8)):
        positions.append({
            "x": float(rng.uniform(-span, span)),
            "y": float(rng.uniform(-span, span)),
            "intended_distance_factor": float("nan"),
        })
    return positions

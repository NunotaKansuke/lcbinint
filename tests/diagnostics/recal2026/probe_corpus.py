"""Geometries chosen to make the seeding stage fail if it can be made to fail.

The campaign's main sweeps sample positions at multiples of ``rho`` from a
random caustic point, which is the right sampling for a resolution rule and the
wrong one for this question.  What breaks a seeding heuristic is not a typical
crossing, it is a component too thin for the heuristic's fixed step: the ring
probes move in steps of 0.02--0.35 ``rho`` around a caustic point, so a cap of
depth 1e-4 ``rho`` is invisible to them however many of them there are.  A
corpus that never produces such a cap would report that the rings are safe to
remove because it never asked them anything hard.

So the positions here are placed *at* the geometry that hurts:

``fold_tangency``
    Centre offset from a fold so the disk crosses it by a controlled depth,
    swept from a comfortable 0.1 ``rho`` cap down to 1e-5 ``rho``, and past
    tangency to the outside.  This is the case the certificate was written for.
``cusp_tangency``
    The same sweep anchored on a cusp instead of a smooth arc.  A cusp's wedge
    opens along the tangent, so a probe ladder that only steps along the normal
    misses it at every offset -- the reason the certificate probes four
    directions rather than two, and the case that should punish the
    ``tangents=0`` ablation.
``swallowed``
    A disk large enough to contain a planetary caustic whole, where the
    heuristic rings do cover the disk but the components are many and small.
``double_contact``
    Wide topology with the disk placed to touch two caustic branches at once,
    so the "first crossing then the rest" staging of the rings is exercised.
``resonant``
    ``s`` within 1e-3 of the topology change, where the caustic is long and
    thin and the polyline that both stages consume is at its least trustworthy.

Every position records what it was constructed to be, but nothing downstream
trusts that: the study reports the caustic distance lcbinint itself measures.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import caustic_branches

# Cap depths, as a fraction of the source radius, for the tangency families.
# The top of the range is a crossing any method handles; the bottom is two
# orders of magnitude below the finest ring step, which is the regime where the
# rings can only be right by luck.
CAP_DEPTHS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 1.0e-4, 1.0e-5)

# Signed offsets past tangency, in units of the source radius: just inside,
# exactly on, and just outside the limb.
TANGENCY_OFFSETS = (-1.0e-3, -1.0e-5, 0.0, 1.0e-5, 1.0e-3)


def _unit(vector):
    norm = math.hypot(vector[0], vector[1])
    return (vector[0] / norm, vector[1] / norm) if norm > 0.0 else (0.0, 0.0)


def _turning_angles(branch):
    """Exterior angle at each vertex, as the cusp indicator."""
    previous = np.roll(branch, 1, axis=0)
    following = np.roll(branch, -1, axis=0)
    incoming = branch - previous
    outgoing = following - branch
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = incoming[:, 0] * outgoing[:, 0] + incoming[:, 1] * outgoing[:, 1]
    return np.abs(np.arctan2(cross, dot))


def _outward_normal(branch, index):
    """Unit normal at a vertex, oriented away from the branch's centroid.

    Which side of the caustic is "outside" is not a property of the polyline
    orientation, and guessing it wrong places every position on the far side of
    the arc from where it was meant to go.  The centroid is a crude reference
    but an unambiguous one, and the study reports the achieved distance anyway.
    """
    count = len(branch)
    tangent = _unit(branch[(index + 1) % count] - branch[(index - 1) % count])
    normal = (-tangent[1], tangent[0])
    outward = branch[index] - branch.mean(axis=0)
    if normal[0] * outward[0] + normal[1] * outward[1] < 0.0:
        normal = (-normal[0], -normal[1])
    return normal


def _cusp_indices(branch, count=3):
    """The sharpest vertices of a branch, well separated from each other."""
    angles = _turning_angles(branch)
    order = np.argsort(angles)[::-1]
    chosen = []
    minimum_gap = max(1, len(branch) // 16)
    for index in order:
        if all(min(abs(index - other), len(branch) - abs(index - other))
               >= minimum_gap for other in chosen):
            chosen.append(int(index))
        if len(chosen) >= count:
            break
    return chosen


def _tangency_positions(branch, index, rho, family):
    """Centres whose disk cuts a cap of prescribed depth out of the caustic.

    The offset is measured along the outward normal from the anchor vertex, so
    a centre at ``rho - depth`` leaves a cap of depth ``depth`` on the far side
    of the arc -- exactly to the extent that the arc is straight there, which
    near a cusp it is not.  That is the point: the constructed depth is the
    intent, and what the disk really cuts is what the study measures.
    """
    anchor = branch[index]
    normal = _outward_normal(branch, index)
    positions = []
    for depth in CAP_DEPTHS:
        distance = rho * (1.0 - depth)
        positions.append({
            "x": float(anchor[0] + normal[0] * distance),
            "y": float(anchor[1] + normal[1] * distance),
            "family": family,
            "intended_cap_depth": float(depth),
            "intended_distance_factor": float(1.0 - depth),
        })
    for offset in TANGENCY_OFFSETS:
        distance = rho * (1.0 + offset)
        positions.append({
            "x": float(anchor[0] + normal[0] * distance),
            "y": float(anchor[1] + normal[1] * distance),
            "family": family,
            "intended_cap_depth": float(-offset),
            "intended_distance_factor": float(1.0 + offset),
        })
    return positions


def _branch_extent(branch):
    return float(max(np.ptp(branch[:, 0]), np.ptp(branch[:, 1])))


def _lens_family(name, s, q, rho):
    return {"family": name, "s": float(s), "q": float(q), "rho": float(rho)}


def build(seed=20260803):
    """The corpus, as a list of ``(lens, positions)`` pairs."""
    rng = np.random.default_rng(seed)
    corpus = []

    # Fold and cusp tangency, over lenses whose caustics differ in scale by
    # four orders of magnitude so the depths are not always the same absolute
    # size.  rho is set from the caustic extent rather than fixed, or the
    # planetary cases would degenerate into "disk swallows everything".
    tangency_lenses = [
        (1.05, 1.0e-3), (1.30, 1.0e-4), (0.80, 1.0e-2), (1.00, 1.0e-3),
        (2.00, 1.0e-1), (0.60, 3.0e-1), (1.10, 1.0e-5), (1.50, 1.0),
    ]
    for s, q in tangency_lenses:
        branches = caustic_branches(s, q)
        if not branches:
            continue
        branch = max(branches, key=_branch_extent)
        extent = _branch_extent(branch)
        for rho_factor in (0.05, 0.3):
            rho = extent * rho_factor
            if not (1.0e-6 < rho < 1.0):
                continue
            lens = _lens_family("tangency", s, q, rho)
            positions = []
            # A smooth arc: the vertex whose turning angle is smallest, which
            # is as far from a cusp as the polyline offers.
            angles = _turning_angles(branch)
            smooth = int(np.argsort(angles)[len(angles) // 2])
            positions += _tangency_positions(branch, smooth, rho, "fold_tangency")
            for cusp in _cusp_indices(branch):
                positions += _tangency_positions(branch, cusp, rho, "cusp_tangency")
            corpus.append((lens, positions))

    # Disks that swallow a planetary caustic whole.  The retired resolvability
    # guard used to refuse these; they are the regime with the most components
    # per disk and therefore the most for a seeding stage to miss.
    for s, q in ((1.30, 1.0e-4), (1.30, 1.0e-5), (0.70, 1.0e-4), (2.00, 1.0e-3)):
        branches = caustic_branches(s, q)
        if len(branches) < 2:
            continue
        small = min(branches, key=_branch_extent)
        centre = small.mean(axis=0)
        extent = _branch_extent(small)
        for multiple in (1.0, 2.0, 5.0):
            rho = extent * multiple
            lens = _lens_family("swallowed", s, q, rho)
            positions = []
            for shift in (0.0, 0.3, 0.7, 1.0):
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                positions.append({
                    "x": float(centre[0] + shift * rho * math.cos(angle)),
                    "y": float(centre[1] + shift * rho * math.sin(angle)),
                    "family": "swallowed",
                    "intended_cap_depth": float("nan"),
                    "intended_distance_factor": float(shift),
                })
            corpus.append((lens, positions))

    # Two branches within one disk: the staging of the ring probes assumes a
    # first crossing and then the rest, and this is where that ordering has to
    # cope with more than one arc.
    for s, q in ((1.30, 1.0e-3), (1.60, 1.0e-2)):
        branches = caustic_branches(s, q)
        if len(branches) < 2:
            continue
        first, second = branches[0], branches[1]
        gap = np.linalg.norm(first.mean(axis=0) - second.mean(axis=0))
        rho = float(gap * 0.6)
        midpoint = 0.5 * (first.mean(axis=0) + second.mean(axis=0))
        lens = _lens_family("double_contact", s, q, rho)
        positions = [{
            "x": float(midpoint[0] + dx * rho),
            "y": float(midpoint[1] + dy * rho),
            "family": "double_contact",
            "intended_cap_depth": float("nan"),
            "intended_distance_factor": float("nan"),
        } for dx, dy in ((0.0, 0.0), (0.2, 0.0), (0.0, 0.2), (-0.2, 0.1))]
        corpus.append((lens, positions))

    # The topology boundary, where the caustic is longest and thinnest and the
    # polyline both stages read is at its least trustworthy.
    for s in (1.0 - 1.0e-3, 1.0, 1.0 + 1.0e-3):
        for q in (1.0e-3, 1.0e-1):
            branches = caustic_branches(s, q)
            if not branches:
                continue
            branch = max(branches, key=_branch_extent)
            rho = _branch_extent(branch) * 0.02
            lens = _lens_family("resonant", s, q, rho)
            positions = []
            angles = _turning_angles(branch)
            smooth = int(np.argsort(angles)[len(angles) // 2])
            positions += _tangency_positions(branch, smooth, rho, "resonant")
            for cusp in _cusp_indices(branch, count=2):
                positions += _tangency_positions(branch, cusp, rho, "resonant_cusp")
            corpus.append((lens, positions))

    return corpus


def rows(seed=20260803):
    """The corpus flattened to one dict per evaluation."""
    out = []
    for lens, positions in build(seed):
        for position in positions:
            row = dict(lens)
            row.update(position)
            out.append(row)
    for index, row in enumerate(out):
        row["row_id"] = index
    return out

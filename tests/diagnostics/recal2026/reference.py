"""Reference values, and an honest statement of how far they can be trusted.

Nothing in this campaign can be more accurate than the reference it is measured
against, so the reference is not a single fine-grid number.  It is a
convergence ladder on one grid, corroborated by an independently discretised
grid and by an independent contour integrator, and it carries an uncertainty
derived from how much those three disagree.

That uncertainty is what makes a row usable or not, and it is per-row and
per-tolerance: a reference good to 3e-5 can calibrate a 1e-3 rule and cannot
calibrate a 1e-4 one.  Rows are therefore never globally "trusted" or
"untrusted"; :func:`usable_for` answers the question for one target tolerance
at a time.  Rows where the witnesses disagree are reported, not quietly given
whichever value looked reasonable.
"""

from __future__ import annotations

import math

import numpy as np

from .engines import BUCKETS, lcbinint_fixed

# The top of the ladder.  Below the finest bucket there is nothing to compare
# against, so the finest two buckets on each grid define the self-convergence
# gap and everything coarser is the material the resolution rule is fitted to.
LADDER_TOP = (256, 400)

# A reference must be at least this much better than the tolerance it is used
# to calibrate.  A tenth means a measured error of one tolerance unit is known
# to a 10% relative accuracy, which is enough to place a bucket boundary.
UNCERTAINTY_MARGIN = 0.1

# Contour witness accuracies, coarse to fine.  These are VBMicrolensing's
# *relative* precision (``RelTol``), not its ``Tol``, which is an absolute
# accuracy on the magnification: asking for Tol=1e-7 at A=1.4e4 is asking for
# 7e-12 relative, and one such row in the trial spent 171 s inside a single VBM
# call -- 84% of the row -- for an answer that RelTol=1e-7 reproduced to 2.5e-6
# in 9 s.  ``Tol`` is therefore pinned below anything reachable so that the
# relative criterion is the one that stops the integration.
CONTOUR_RELATIVE_LEVELS = (1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10)
CONTOUR_ABSOLUTE_FLOOR = 1.0e-12

# How much more expensive the next contour level is assumed to be.  The levels
# are two decades apart and the measured growth across them is roughly twenty-
# fold, so this projects the next call before paying for it.
CONTOUR_LEVEL_COST_FACTOR = 20.0


def evaluate_ladder(s, q, rho, x, y, profile_c, *, grid, buckets=BUCKETS,
                    time_budget=None):
    """Every bucket on one grid, coarse to fine, with per-row censoring.

    Large disks at 400 bins can cost seconds, and a handful of them would
    otherwise monopolise the sweep.  When the budget runs out the remaining
    buckets are left absent rather than being recorded as failures: an absent
    bucket means "not measured", which is a different claim from "did not
    converge", and conflating them would bias the resolution rule upward.
    """
    import time as _time

    rows = {}
    spent = 0.0
    last_bucket = None
    last_seconds = None
    for bucket in buckets:
        # Stop before an expensive bucket rather than after it.  Cost grows
        # roughly with the square of the bin count, so a ladder that checks its
        # budget only afterwards can overshoot by an order of magnitude on the
        # step that matters -- one small-source, high-magnification row in the
        # trial spent 224 s that way.  Projecting from the last measured bucket
        # keeps the tail bounded without capping the cheap rows.
        if (time_budget is not None and last_seconds is not None
                and last_bucket):
            projected = last_seconds * (bucket / last_bucket) ** 2
            if spent + projected > time_budget:
                rows["censored_before"] = bucket
                rows["censored_projection_seconds"] = projected
                break
        engine = lcbinint_fixed(grid, bucket, profile_c)
        started = _time.perf_counter()
        try:
            result = engine(s, q, rho, x, y)
        except Exception as error:  # noqa: BLE001
            rows[bucket] = {"error": f"{type(error).__name__}: {error}"}
            spent += _time.perf_counter() - started
            continue
        elapsed = _time.perf_counter() - started
        spent += elapsed
        last_bucket, last_seconds = bucket, elapsed
        rows[bucket] = {
            "magnification": result.magnification,
            "error_estimate": result.error_estimate,
            "converged": result.converged,
            "support_proven": result.support_proven,
            "method": result.method,
            "seconds": result.seconds,
        }
        if time_budget is not None and spent > time_budget:
            rows["censored_after"] = bucket
            break
    return rows


def contour_reference(s, q, rho, x, y, profile_c, *,
                      levels=CONTOUR_RELATIVE_LEVELS, time_budget=None):
    """The contour witness, escalated until it stops moving or costs too much.

    A single contour call at one tolerance is a number with no error bar, and
    treating it as exact would let the reference claim an agreement finer than
    the witness's own noise.  Running consecutive relative tolerances gives the
    witness a self-convergence gap of its own, which :func:`build` then uses as
    a floor on what the contour is allowed to certify.

    The escalation is projected before it is paid for, in the same way as the
    grid ladder: these levels are two decades apart and cost roughly twenty
    times more each, which is enough to swallow a whole sweep on the few
    high-magnification rows where contour integration is slowest.
    """
    import time as _time

    from .engines import VbmEngine

    measured = []
    spent = 0.0
    for reltol in levels:
        if (time_budget is not None and measured
                and spent + measured[-1][2] * CONTOUR_LEVEL_COST_FACTOR
                > time_budget):
            break
        engine = VbmEngine(tol=CONTOUR_ABSOLUTE_FLOOR, profile_c=profile_c,
                           reltol=reltol)
        started = _time.perf_counter()
        try:
            value = engine(s, q, rho, x, y).magnification
        except Exception as error:  # noqa: BLE001
            return {"value": float("nan"), "self_gap": float("inf"),
                    "error": f"{type(error).__name__}: {error}",
                    "levels": [level for level, _, _ in measured],
                    "seconds": spent}
        elapsed = _time.perf_counter() - started
        spent += elapsed
        measured.append((reltol, value, elapsed))
        if not math.isfinite(value):
            break

    if not measured:
        return {"value": float("nan"), "self_gap": float("inf"),
                "levels": [], "seconds": spent}

    reltol, value, _ = measured[-1]
    self_gap = float("inf")
    if len(measured) >= 2:
        previous = measured[-2][1]
        if math.isfinite(previous) and math.isfinite(value):
            self_gap = abs(value - previous) / max(abs(value), 1.0)
    return {
        "value": value,
        "self_gap": self_gap,
        "reltol": reltol,
        "levels": [level for level, _, _ in measured],
        "values": [item for _, item, _ in measured],
        "seconds": spent,
    }


def build(s, q, rho, x, y, profile_c, *, cartesian, polar, contour=None,
          contour_self_gap=0.0, contour_budget=None):
    """Combine the two ladders and a contour value into a reference.

    The value is taken from the finest certified Cartesian bucket.  Cartesian
    rather than polar because it is the grid the resolution rule is mainly for,
    and certified because an uncertified value is not a magnification of the
    whole source -- it is a magnification of the components that happened to be
    found, which no amount of agreement with another engine would reveal.
    """
    def finest_certified(ladder):
        for bucket in reversed([b for b in BUCKETS if b in ladder]):
            row = ladder.get(bucket) or {}
            if row.get("support_proven") and math.isfinite(
                    row.get("magnification", float("nan"))):
                return bucket, row
        return None, None

    cart_bucket, cart_row = finest_certified(cartesian)
    polar_bucket, polar_row = finest_certified(polar)
    if cart_row is None:
        return {
            "status": "no certified cartesian value",
            "value": float("nan"),
            "uncertainty": float("inf"),
        }

    value = cart_row["magnification"]
    scale = max(abs(value), 1.0)

    # Self-convergence: how much the Cartesian answer still moved between the
    # last two certified buckets.  This is the part of the uncertainty that
    # more resolution could remove.
    ladder_gap = float("inf")
    previous = [
        b for b in BUCKETS
        if b < cart_bucket and (cartesian.get(b) or {}).get("support_proven")
    ]
    if previous:
        prior = cartesian[previous[-1]]["magnification"]
        if math.isfinite(prior):
            ladder_gap = abs(value - prior) / scale

    # Independent witnesses.  The polar grid discretises the same integral
    # differently; the contour integrator does not discretise it at all.  Only
    # the closer of the two is used: a reference is as good as its best
    # independent confirmation, and a disagreeing third engine is recorded
    # below rather than being allowed to inflate every uncertainty.
    witnesses = {}
    if polar_row is not None and math.isfinite(polar_row["magnification"]):
        witnesses["polar"] = abs(value - polar_row["magnification"]) / scale
    if contour is None:
        built = contour_reference(s, q, rho, x, y, profile_c,
                                  time_budget=contour_budget)
        contour, contour_self_gap = built["value"], built["self_gap"]
    if math.isfinite(contour):
        # A witness cannot confirm a value to better than its own convergence.
        # Without this floor a contour call that happened to land close would
        # certify the reference to a precision it never demonstrated.
        witnesses["contour"] = max(abs(value - contour) / scale,
                                   contour_self_gap)

    if not witnesses:
        return {
            "status": "no independent witness",
            "value": value,
            "uncertainty": float("inf"),
            "cartesian_bucket": cart_bucket,
            "ladder_gap": ladder_gap,
        }

    best_witness = min(witnesses.values())
    uncertainty = max(ladder_gap, best_witness)
    return {
        "status": "ok",
        "value": value,
        "uncertainty": uncertainty,
        "cartesian_bucket": cart_bucket,
        "polar_bucket": polar_bucket,
        "ladder_gap": ladder_gap,
        "witness_gaps": witnesses,
        "contour": contour,
        "contour_self_gap": contour_self_gap,
        "polar": polar_row["magnification"] if polar_row else float("nan"),
    }


def usable_for(reference, relative_tolerance):
    """Whether this reference is sharp enough to judge one target tolerance."""
    return (
        reference.get("status") == "ok"
        and math.isfinite(reference.get("uncertainty", float("inf")))
        and reference["uncertainty"] <= UNCERTAINTY_MARGIN * relative_tolerance
    )


def required_bucket(ladder, reference_value, relative_tolerance,
                    absolute_tolerance=0.0):
    """The coarsest bucket that is right and stays right.

    "Stays right" is the whole point: a coarse grid can cross the reference by
    luck on its way to converging, and accepting that crossing would fit a rule
    to a coincidence.  A bucket qualifies only when it and every finer measured
    bucket are inside the budget, and only certified buckets can qualify at
    all.
    """
    budget = absolute_tolerance + relative_tolerance * max(
        abs(reference_value), 1.0)
    measured = [b for b in BUCKETS if b in ladder and "magnification" in ladder[b]]
    if not measured:
        return None
    inside = {}
    for bucket in measured:
        row = ladder[bucket]
        value = row["magnification"]
        inside[bucket] = (
            bool(row.get("support_proven"))
            and math.isfinite(value)
            and abs(value - reference_value) <= budget
        )
    for index, bucket in enumerate(measured):
        if all(inside[b] for b in measured[index:]):
            return bucket
    return None


def summarise_uncertainties(references):
    """Distribution of reference uncertainty, for the paper's methods section."""
    values = np.asarray([
        r["uncertainty"] for r in references
        if r.get("status") == "ok" and math.isfinite(r.get("uncertainty", np.inf))
    ])
    if values.size == 0:
        return {}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }

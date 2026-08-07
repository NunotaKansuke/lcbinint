"""Paper-facing empirical resolution laws for the August 2026 recalibration.

This module is an offline calibration record.  It is not imported by the C++
runtime selector.  The common policy is deliberately the same for every
integration route:

    B = max(a_tol, r_tol * max(abs(A), 1))

The two tolerances are alternative allowances, as in VBMicrolensing: either
criterion passing is enough.  The Cartesian and polar routes retain their own
measured convergence law, but share this definition of the requested budget.
"""

from __future__ import annotations

import math

from .engines import BUCKETS


B0 = 1.0e-3
TARGET_COVERAGE = 0.99

RELATIVE_LEVELS = (
    1.0e-2, 5.0e-3, 3.0e-3, 2.0e-3, 1.0e-3,
    5.0e-4, 3.0e-4, 2.0e-4, 1.0e-4,
)
ABSOLUTE_LEVELS = RELATIVE_LEVELS

# Discovery fits before supported-bucket rounding.  The coverage is measured
# on the independent holdout, so this record cannot be mistaken for an
# in-sample fit only.
RELATIVE_LAW = {
    "cartesian": {
        "C": 45.31962548354008,
        "beta": 0.47670215379590497,
        "holdout_coverage": 0.9967076856649395,
    },
    "polar": {
        "C": 94.5708573771618,
        "beta": 0.5952070961585817,
        "holdout_coverage": 0.9980563654033042,
    },
}

# Absolute-only fits.  They are much more conservative because an absolute
# target does not remove the magnification scale, but they are required for a
# well-defined ``reltol=0`` branch.
ABSOLUTE_LAW = {
    "cartesian": {
        "C": 140.46968254869913,
        "beta": 0.1102830028286264,
        "holdout_coverage": 0.996678402976151,
    },
    "polar": {
        "C": 201.04855445710095,
        "beta": 0.22862348842444857,
        "holdout_coverage": 0.9975390754905221,
    },
}

# Coverage of the final supported relative bucket, evaluated separately at
# each tolerance on the independent holdout.
HOLDOUT_COVERAGE = {
    "cartesian": {
        1.0e-2: 0.9957904583723106,
        5.0e-3: 0.9957865168539326,
        3.0e-3: 0.9967228464419475,
        2.0e-3: 0.9985948477751756,
        1.0e-3: 0.9981220657276996,
        5.0e-4: 0.9957183634633682,
        3.0e-4: 0.9970588235294118,
        2.0e-4: 0.9943991853360489,
        1.0e-4: 0.998282770463652,
    },
    "polar": {
        1.0e-2: 1.0,
        5.0e-3: 0.9985955056179775,
        3.0e-3: 1.0,
        2.0e-3: 0.9985948477751756,
        1.0e-3: 0.9976525821596244,
        5.0e-4: 0.9976213130352045,
        3.0e-4: 0.9946078431372549,
        2.0e-4: 0.9954151808456444,
        1.0e-4: 1.0,
    },
}

ABSOLUTE_HOLDOUT_COVERAGE = {
    "cartesian": {
        1.0e-2: 0.9985141158989599,
        5.0e-3: 0.994452849218356,
        3.0e-3: 0.993801652892562,
        2.0e-3: 1.0,
        1.0e-3: 0.9983324068927182,
        5.0e-4: 0.9957882069795427,
        3.0e-4: 0.9986403806934059,
        2.0e-4: 0.9946112394149346,
        1.0e-4: 0.9949647532729103,
    },
    "polar": {
        1.0e-2: 0.9960376423972264,
        5.0e-3: 0.9974785678265254,
        3.0e-3: 0.9958677685950413,
        2.0e-3: 0.9989423585404548,
        1.0e-3: 0.9988882712618121,
        5.0e-4: 0.9975932611311673,
        3.0e-4: 0.998638529611981,
        2.0e-4: 0.9953596287703016,
        1.0e-4: 1.0,
    },
}


def normalized_budget(atol: float, reltol: float, magnification: float) -> float:
    """Return the dimensionless effective budget under the common policy."""

    scale = max(abs(float(magnification)), 1.0)
    absolute = max(float(atol), 0.0) / scale
    relative = max(float(reltol), 0.0)
    return max(absolute, relative)


def effective_budget(atol: float, reltol: float, magnification: float) -> float:
    """Return the dimensional budget used by the common acceptance rule."""

    scale = max(abs(float(magnification)), 1.0)
    return normalized_budget(atol, reltol, magnification) * scale


def continuous_resolution(epsilon: float, grid: str,
                          branch: str = "relative") -> float:
    """Evaluate the unrounded p99 law at normalized budget ``epsilon``."""

    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    laws = {"relative": RELATIVE_LAW, "absolute": ABSOLUTE_LAW}
    try:
        law = laws[branch][grid]
    except KeyError as error:
        raise ValueError(f"unknown branch/grid: {branch}/{grid}") from error
    return law["C"] * (epsilon / B0) ** (-law["beta"])


def bucket_resolution(epsilon: float, grid: str,
                      branch: str = "relative") -> int:
    """Round a continuous p99 prediction upward to a measured bucket."""

    raw = continuous_resolution(epsilon, grid, branch)
    return next((bucket for bucket in BUCKETS if raw <= bucket), BUCKETS[-1])


def branch_resolution(atol: float, reltol: float, magnification: float,
                      grid: str, branch: str):
    """Return the branch prediction, or ``None`` for an inactive branch."""

    scale = max(abs(float(magnification)), 1.0)
    if branch == "absolute":
        # The absolute branch was fitted against the dimensional absolute
        # tolerance itself.  Do not divide it by the magnification scale: that
        # would silently turn an absolute law into a relative one.
        epsilon = max(float(atol), 0.0)
    elif branch == "relative":
        epsilon = max(float(reltol), 0.0)
    else:
        raise ValueError(f"unknown branch: {branch}")
    return None if epsilon <= 0.0 else bucket_resolution(epsilon, grid, branch)


def mixed_bucket_resolution(atol: float, reltol: float, magnification: float,
                           grid: str) -> int:
    """Apply the mixed policy by taking the less demanding branch.

    The dimensional acceptance budget is the larger allowance, so the
    required resolution is the smaller of the two pure-branch requirements.
    The branch laws are intentionally kept separate: the relative branch is
    fitted in normalized tolerance, while the absolute branch is fitted in
    dimensional ``atol``.
    """

    predictions = [branch_resolution(atol, reltol, magnification, grid, branch)
                   for branch in ("absolute", "relative")]
    active = [value for value in predictions if value is not None]
    if not active:
        raise ValueError("at least one tolerance must be positive")
    return min(active)


def supported_table(grid: str):
    """Return ``(epsilon, fitted_bucket, holdout_coverage)`` rows."""

    return tuple(
        (epsilon, bucket_resolution(epsilon, grid),
         HOLDOUT_COVERAGE[grid][epsilon])
        for epsilon in RELATIVE_LEVELS
    )


def absolute_supported_table(grid: str):
    """Return ``(atol, fitted_bucket, holdout_coverage)`` rows."""

    return tuple(
        (atol, bucket_resolution(atol, grid, "absolute"),
         ABSOLUTE_HOLDOUT_COVERAGE[grid][atol])
        for atol in ABSOLUTE_LEVELS
    )

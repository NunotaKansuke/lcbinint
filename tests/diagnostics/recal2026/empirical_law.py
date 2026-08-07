"""Paper-facing empirical resolution law for the August 2026 recalibration.

This module is deliberately an offline calibration record.  It is not imported
by the C++ runtime selector.  Keeping the fitted coefficients and the supported
bucket mapping here gives the report and its regression test one small,
machine-readable definition of the result.
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

# These are the discovery fits before supported-bucket rounding.  The
# independent holdout coverage is recorded alongside them so that this file
# cannot be mistaken for an in-sample fit only.
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

# Coverage of the final supported bucket, evaluated separately at each
# relative tolerance on the independent holdout.
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


def normalized_budget(atol: float, reltol: float, magnification: float) -> float:
    """Return the dimensionless common error budget used by the fit."""

    scale = max(abs(float(magnification)), 1.0)
    return (max(float(atol), 0.0) + max(float(reltol), 0.0) * scale) / scale


def continuous_resolution(epsilon: float, grid: str) -> float:
    """Evaluate the unrounded p99 law at normalized budget ``epsilon``."""

    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    try:
        law = RELATIVE_LAW[grid]
    except KeyError as error:
        raise ValueError(f"unknown grid: {grid}") from error
    return law["C"] * (epsilon / B0) ** (-law["beta"])


def bucket_resolution(epsilon: float, grid: str) -> int:
    """Round the continuous p99 prediction upward to a measured bucket."""

    raw = continuous_resolution(epsilon, grid)
    return next((bucket for bucket in BUCKETS if raw <= bucket), BUCKETS[-1])


def supported_table(grid: str) -> tuple[tuple[float, int, float], ...]:
    """Return ``(epsilon, fitted_bucket, holdout_coverage)`` rows."""

    return tuple(
        (epsilon, bucket_resolution(epsilon, grid),
         HOLDOUT_COVERAGE[grid][epsilon])
        for epsilon in RELATIVE_LEVELS
    )

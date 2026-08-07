"""Paper-facing empirical resolution laws for the August 2026 recalibration.

This module is an offline calibration record.  It is not imported by the C++
runtime selector.  The common policy is deliberately the same for every
integration route:

    B = max(a_tol, r_tol * max(abs(A), 1))

The two tolerances are alternative allowances, as in VBMicrolensing: either
criterion passing is enough.  The Cartesian and polar routes retain their own
measured convergence law, but share this definition of the requested budget.
The scalar coefficients below are conditional on reference-certified rows:
the lower-censored audit found that the stored campaign cannot identify a
population-wide p99 at the tightest targets.  The final binary selector uses
the Apoint-dependent absolute branch recorded below; its larger safety factor
is intentional because the raw conditional fit under-covered the integer-ceil
holdout at several absolute levels.
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
ABSOLUTE_LEVELS = (
    1.0e-2, 5.0e-3, 3.0e-3, 2.0e-3, 1.0e-3,
    5.0e-4, 3.0e-4, 2.0e-4,
)
ABSOLUTE_DIAGNOSTIC_LEVELS = ABSOLUTE_LEVELS + (1.0e-4,)

# Discovery fits before supported-bucket rounding.  The coverage is measured
# on the independent holdout, so this record cannot be mistaken for an
# in-sample fit only.
RELATIVE_LAW = {
    "cartesian": {
        "C": 49.5929807101336,
        "beta": 0.47670215379590497,
        "holdout_coverage": 0.9921568627450981,
    },
    "polar": {
        "C": 105.29723705815378,
        "beta": 0.5952070961585817,
        "holdout_coverage": 0.9946078431372549,
    },
}

# Absolute Apoint laws on reference-certified rows.  The safety envelope was
# selected from discovery before the holdout was inspected.  The formal
# production domain ends at 2e-4; 1e-4 remains diagnostic only.
ABSOLUTE_LAW = {
    "cartesian": {
        "C": 138.06382198454384,
        "beta": 0.4265493297299796,
        "gamma": 0.34119845152344075,
        "holdout_coverage": 0.9986403806934059,
    },
    "polar": {
        "C": 396.47500160748996,
        "beta": 0.5337641762207631,
        "gamma": 0.2458039343900396,
        "holdout_coverage": 1.0,
    },
}

# Coverage of the final supported relative bucket, evaluated separately at
# each tolerance on the independent holdout.
HOLDOUT_COVERAGE = {
    "cartesian": {
        1.0e-2: 0.9957904583723106,
        5.0e-3: 0.9957865168539326,
        3.0e-3: 0.9943820224719101,
        2.0e-3: 0.994847775175644,
        1.0e-3: 0.9981220657276996,
        5.0e-4: 0.9957183634633682,
        3.0e-4: 0.9921568627450981,
        2.0e-4: 0.9943991853360489,
        1.0e-4: 0.9925586720091586,
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
        1.0e-2: 1.0,
        5.0e-3: 1.0,
        3.0e-3: 1.0,
        2.0e-3: 1.0,
        1.0e-3: 1.0,
        5.0e-4: 0.9993983152827918,
        3.0e-4: 0.9986403806934059,
        2.0e-4: 1.0,
    },
    "polar": {
        1.0e-2: 1.0,
        5.0e-3: 1.0,
        3.0e-3: 1.0,
        2.0e-3: 1.0,
        1.0e-3: 1.0,
        5.0e-4: 1.0,
        3.0e-4: 1.0,
        2.0e-4: 1.0,
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
                          branch: str = "relative",
                          point_magnification: float = 1.0) -> float:
    """Evaluate the unrounded law at a branch tolerance."""

    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    laws = {"relative": RELATIVE_LAW, "absolute": ABSOLUTE_LAW}
    try:
        law = laws[branch][grid]
    except KeyError as error:
        raise ValueError(f"unknown branch/grid: {branch}/{grid}") from error
    result = law["C"] * (epsilon / B0) ** (-law["beta"])
    if branch == "absolute":
        result *= max(abs(float(point_magnification)), 1.0) ** law["gamma"]
    return result


def bucket_resolution(epsilon: float, grid: str,
                      branch: str = "relative",
                      point_magnification: float = 1.0) -> int:
    """Mirror the runtime's upward integer ceil and hard cap."""

    raw = continuous_resolution(
        epsilon, grid, branch, point_magnification)
    return min(400, max(1, math.ceil(raw)))


def branch_resolution(atol: float, reltol: float, magnification: float,
                      grid: str, branch: str):
    """Return the branch prediction, or ``None`` for an inactive branch."""

    if branch == "absolute":
        # The absolute branch is dimensional and carries Apoint explicitly;
        # it is not a normalized relative law.
        epsilon = max(float(atol), 0.0)
    elif branch == "relative":
        epsilon = max(float(reltol), 0.0)
    else:
        raise ValueError(f"unknown branch: {branch}")
    return None if epsilon <= 0.0 else bucket_resolution(
        epsilon, grid, branch, magnification)


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

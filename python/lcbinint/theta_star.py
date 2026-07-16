"""Color--surface-brightness relations for angular source radii."""
from __future__ import annotations

import math


_LN10 = math.log(10.0)


def _evaluate(
    magnitude,
    color,
    *,
    magnitude_error,
    color_error,
    intercept,
    slope,
    intercept_error,
    slope_error,
    fractional_scatter,
):
    values = {
        "magnitude": magnitude,
        "color": color,
        "magnitude_error": magnitude_error,
        "color_error": color_error,
    }
    values = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("theta-star relation inputs must be finite")
    if values["magnitude_error"] < 0.0 or values["color_error"] < 0.0:
        raise ValueError("magnitude and color errors must be >= 0")

    log10_diameter = intercept + slope * values["color"] - 0.2 * values["magnitude"]
    variance_log10 = (
        intercept_error**2
        + values["color"] ** 2 * slope_error**2
        + slope**2 * values["color_error"] ** 2
        + 0.2**2 * values["magnitude_error"] ** 2
        + (fractional_scatter / _LN10) ** 2
    )

    # Published relations give log10(2 thetaS); lcbinint uses ln(thetaS/mas).
    log_median = _LN10 * log10_diameter - math.log(2.0)
    log_sigma = _LN10 * math.sqrt(variance_log10)
    return log_median, log_sigma


def VI(*, I, V_I, I_error=0.0, V_I_error=0.0):
    """Return ``(ln median, sigma_ln)`` from dereddened I and V-I.

    ``thetaS`` is in mas. The relation coefficients are from Fukui et al.
    (2015), with the coefficient uncertainties and intrinsic scatter used by
    the Nataf analysis.
    """
    return _evaluate(
        I,
        V_I,
        magnitude_error=I_error,
        color_error=V_I_error,
        intercept=0.5014135,
        slope=0.41968496,
        intercept_error=0.5014135 * 0.05,
        slope_error=0.41968496 * 0.05,
        fractional_scatter=0.051,
    )


def IH(*, I, I_H, I_error=0.0, I_H_error=0.0):
    """Return ``(ln median, sigma_ln)`` from dereddened I and I-H.

    ``thetaS`` is in mas. The relation is from Boyajian et al. (2013).
    """
    return _evaluate(
        I,
        I_H,
        magnitude_error=I_error,
        color_error=I_H_error,
        intercept=0.53026,
        slope=0.36595,
        intercept_error=0.00077,
        slope_error=0.00079,
        fractional_scatter=0.074,
    )


def JH(*, H, J_H, H_error=0.0, J_H_error=0.0):
    """Return ``(ln median, sigma_ln)`` from dereddened H and J-H.

    ``thetaS`` is in mas. The relation is from Kervella et al. (2004).
    """
    return _evaluate(
        H,
        J_H,
        magnitude_error=H_error,
        color_error=J_H_error,
        intercept=0.5013,
        slope=0.4312,
        intercept_error=0.044,
        slope_error=0.12,
        fractional_scatter=0.1031,
    )


def VH(*, H, V_H, H_error=0.0, V_H_error=0.0):
    """Return ``(ln median, sigma_ln)`` from dereddened H and V-H.

    ``thetaS`` is in mas. The V-H sign convention is converted internally
    from the H-V form of the published relation.
    """
    return _evaluate(
        H,
        V_H,
        magnitude_error=H_error,
        color_error=V_H_error,
        intercept=0.5145,
        slope=0.0892,
        intercept_error=0.0019,
        slope_error=0.0009,
        fractional_scatter=0.0112,
    )


__all__ = ["VI", "IH", "JH", "VH"]

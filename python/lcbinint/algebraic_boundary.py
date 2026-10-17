"""Value and convergence diagnostics for the algebraic-boundary backend.

The existing point-source/hexadecapole selector remains shared with the native
LightCurve router. Deep finite-source rows use the algebraic image-boundary integral.
"""

import math

import numpy as np


def _validated_value_request(curve, times, params, mu_atol, mu_rtol):
    values = curve._merge_params(params)
    names = ("t0", "u0", "tE", "alpha", "rho", "q", "s")
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError("missing light-curve parameters: " + ", ".join(missing))
    t0, u0, tE, alpha, rho, q, separation = (float(values[name]) for name in names)
    if not all(math.isfinite(x) for x in (t0, u0, tE, alpha, rho, q, separation)):
        raise ValueError("light-curve parameters must be finite")
    if tE == 0 or rho <= 0 or q <= 0 or separation <= 0:
        raise ValueError("tE, rho, q and s must be positive (tE nonzero)")
    model = curve._native.model
    if curve.parallax or model.xallarap != "none":
        raise ValueError("algebraic-boundary light curves require no parallax or xallarap")
    if curve._native.orbital_motion != "static":
        raise ValueError("algebraic-boundary light curves require a static binary")
    if model.lens != "binary" or model.source != "single":
        raise ValueError("algebraic-boundary light curves require a binary lens")

    options = curve.options
    option_atol = float(options.finite_source_tol)
    option_rtol = float(options.finite_source_reltol)
    if option_atol == 0.0 and option_rtol == 0.0:
        option_atol, option_rtol = 1.0e-4, 1.0e-3
    requested_atol = option_atol if mu_atol is None else float(mu_atol)
    requested_rtol = option_rtol if mu_rtol is None else float(mu_rtol)
    if not all(math.isfinite(x) and x >= 0.0 for x in (requested_atol, requested_rtol)):
        raise ValueError("mu_atol and mu_rtol must be finite and nonnegative")
    if requested_atol != option_atol or requested_rtol != option_rtol:
        # A different integrator must receive the same accuracy contract as
        # the selector. Set finite_source_tol/reltol in Options to compare at
        # another tolerance.
        raise ValueError(
            "algebraic-boundary integration tolerances must match the LightCurve Options; "
            "set finite_source_tol and finite_source_reltol there"
        )

    t = np.asarray(times, dtype=float)
    scalar = t.ndim == 0
    t = t.reshape(1) if scalar else t.reshape(-1)
    if not t.size or not np.all(np.isfinite(t)):
        raise ValueError("times must be a nonempty finite sequence")
    return t, values, requested_atol, requested_rtol, scalar


def lightcurve_value(
    curve, times, params=None, *, mu_atol=None, mu_rtol=None, with_gradient=False
):
    """Routed algebraic-boundary values with error and method diagnostics.

    The point/hex selector is shared with ordinary LightCurve. The native C++
    router owns the caustic and root caches across all epochs in this call.
    """
    t, values, requested_atol, requested_rtol, scalar = _validated_value_request(
        curve, times, params, mu_atol, mu_rtol
    )
    output = curve._native._evaluate_algebraic_boundary(
        t, values, requested_atol, requested_rtol, bool(with_gradient)
    )
    for key in (
        "mu", "point_source_magnification", "error_estimate",
        "value_converged", "support_certified",
        "seed_support_proven", "center_image_count", "augmented_seed_count",
        "node_evaluations", "finite_source_method", "fully_converged",
        "frontier_discovery_completed",
        "gradients", "gradient_valid",
    ):
        if key in output:
            output[key] = np.asarray(output[key])
    output["value_converged"] = output["value_converged"].astype(bool)
    output["fully_converged"] = output["fully_converged"].astype(bool)
    output["frontier_discovery_completed"] = output[
        "frontier_discovery_completed"
    ].astype(bool)
    output["support_certified"] = output["support_certified"].astype(bool)
    output["seed_support_proven"] = output["seed_support_proven"].astype(bool)
    if "gradient_valid" in output:
        output["gradient_valid"] = output["gradient_valid"].astype(bool)
    output["value_fallback_used"] = np.zeros_like(
        output["support_certified"], dtype=bool
    )
    output["value_fallback_kind"] = np.full(
        output["support_certified"].shape, "none", dtype=object
    )
    output["integrator_used"] = (
        output["finite_source_method"] == 7
    )
    if scalar:
        for key in output:
            if key not in ("stop", "finite_source_method_name"):
                output[key] = output[key][0]
        output["stop"] = output["stop"][0]
        output["finite_source_method_name"] = output["finite_source_method_name"][0]
    else:
        output["stop"] = list(output["stop"])
        output["finite_source_method_name"] = list(
            output["finite_source_method_name"]
        )
    return output


def lightcurve_magnification(curve, times, params=None, *, mu_atol=None, mu_rtol=None):
    """Return scalar-only routed values, requiring value convergence.

    Uses the same selector and algebraic-boundary integral as
    ``lightcurve_value``. Diagnostics remain available there.
    """
    t, values, requested_atol, requested_rtol, scalar = _validated_value_request(
        curve, times, params, mu_atol, mu_rtol
    )
    output = curve._native._magnification_algebraic_boundary(
        t, values, requested_atol, requested_rtol
    )
    return float(output[0]) if scalar else output

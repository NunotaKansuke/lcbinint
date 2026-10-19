"""Value and convergence diagnostics for the algebraic-boundary backend.

The existing point-source/hexadecapole selector remains shared with the native
LightCurve router. Deep finite-source rows use caustic-aware seeds and nested-Fejer radial frontiers.
"""

import math

import numpy as np


def _validated_value_request(curve, times, params, mu_atol, mu_rtol):
    # Match lcbi_default_params rather than requiring optional trajectory keys.
    values = dict(t0=0., u0=0., tE=1., alpha=0., rho=0., q=1., s=1.)
    aliases = {
        "t_0": "t0",
        "t_E": "tE",
        "umin": "u0",
        "theta": "alpha",
        "sep": "s",
        "rho1": "rho",
    }
    for name, value in curve._merge_params(params).items():
        values[aliases.get(name, name)] = value
    names = ("t0", "u0", "tE", "alpha", "rho", "q", "s")
    t0, u0, tE, alpha, rho, q, separation = (float(values[name]) for name in names)
    if not all(math.isfinite(x) for x in (t0, u0, tE, alpha, rho, q, separation)):
        raise ValueError("light-curve parameters must be finite")
    if tE == 0 or rho < 0 or q <= 0 or separation <= 0:
        raise ValueError("rho must be nonnegative; q and s positive; tE nonzero")
    model = curve._native.model
    if curve.parallax or model.xallarap != "none":
        raise ValueError("algebraic-boundary light curves require no parallax or xallarap")
    if curve._native.orbital_motion != "static":
        raise ValueError("algebraic-boundary light curves require a static binary")
    if model.lens != "binary" or model.source != "single":
        raise ValueError("algebraic-boundary light curves require a binary lens")

    limb_d = float(curve._native.ld_d)
    if curve._native.ld_c == 0.0 and limb_d == 0.0:
        limb_d = float(values.get("limb_darkening_d", 0.0))
    if limb_d != 0.0:
        raise ValueError("algebraic_boundary supports uniform and linear limb darkening; d must be zero")

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
    curve, times, params=None, *, mu_atol=None, mu_rtol=None, with_gradient=False,
    with_diagnostics=True,
):
    """Routed algebraic-boundary values with error and method diagnostics.

    The point/hex selector is shared with ordinary LightCurve. The native C++
    router owns the caustic and root caches across all epochs in this call.
    """
    t, values, requested_atol, requested_rtol, scalar = _validated_value_request(
        curve, times, params, mu_atol, mu_rtol
    )
    output = curve._native._evaluate_algebraic_boundary(
        t, values, requested_atol, requested_rtol, bool(with_gradient),
        bool(with_diagnostics),
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
    for key in ("mu", "gradients"):
        if key in output and not np.all(np.isfinite(output[key])):
            raise RuntimeError("algebraic_boundary: no finite " + key + " candidate")
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
        for key, value in tuple(output.items()):
            if isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim):
                output[key] = value[0]
    else:
        if "stop" in output:
            output["stop"] = list(output["stop"])
        if "finite_source_method_name" in output:
            output["finite_source_method_name"] = list(output["finite_source_method_name"])
    return output


def lightcurve_magnification(curve, times, params=None, *, mu_atol=None, mu_rtol=None):
    """Return routed value candidates; use lightcurve_value for status flags.

    Uses the same selector and seeded radial-frontier integral as
    ``lightcurve_value``. Diagnostics remain available there.
    """
    t, values, requested_atol, requested_rtol, scalar = _validated_value_request(
        curve, times, params, mu_atol, mu_rtol
    )
    output = curve._native._magnification_algebraic_boundary(
        t, values, requested_atol, requested_rtol
    )
    return float(output[0]) if scalar else output

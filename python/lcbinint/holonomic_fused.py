"""Opt-in seeded radial values and forward gradients for static binary curves.

The value API preserves the main point-source/hexadecapole selector and uses
caustic-aware radial-frontier integration only on its integral branch. The
separate gradient API computes the seeded radial path directly. Gradients are
with respect to (t0, u0, tE, alpha, rho, q, s); their precision is not certified.
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
        raise ValueError("seeded algebraic-boundary light curves require no parallax or xallarap")
    if curve._native.orbital_motion != "static":
        raise ValueError("seeded algebraic-boundary light curves require a static binary")
    if model.lens != "binary" or model.source != "single":
        raise ValueError("seeded algebraic-boundary light curves require a binary lens")

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


def lightcurve_value(curve, times, params=None, *, mu_atol=None, mu_rtol=None):
    """Routed radial-frontier values with error, method and support diagnostics.

    The point/hex selector is shared with ordinary LightCurve. Geometry/seed
    caches are cleared once per call and shared across its epochs. Diagnostics
    retain uncertified last iterates; use ``lightcurve_magnification`` for a
    scalar-only call with the ordinary convergence contract.
    """
    t, values, requested_atol, requested_rtol, scalar = _validated_value_request(
        curve, times, params, mu_atol, mu_rtol
    )
    output = curve._native._evaluate_holonomic(
        t, values, requested_atol, requested_rtol
    )
    for key in (
        "mu", "error_estimate", "value_converged", "support_certified",
        "seed_support_proven", "center_image_count", "augmented_seed_count",
        "node_evaluations", "finite_source_method", "fully_converged",
        "frontier_discovery_completed",
    ):
        output[key] = np.asarray(output[key])
    output["value_converged"] = output["value_converged"].astype(bool)
    output["fully_converged"] = output["fully_converged"].astype(bool)
    output["frontier_discovery_completed"] = output[
        "frontier_discovery_completed"
    ].astype(bool)
    output["support_certified"] = output["support_certified"].astype(bool)
    output["seed_support_proven"] = output["seed_support_proven"].astype(bool)
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
    """Return scalar-only routed values, requiring convergence and support.

    Uses the same selector, seeded integrator and cold geometry boundary as
    ``lightcurve_value``. Failure follows the ordinary scalar API; diagnostic
    last iterates and their support flags remain available from that function.
    """
    t, values, requested_atol, requested_rtol, scalar = _validated_value_request(
        curve, times, params, mu_atol, mu_rtol
    )
    output = curve._native._magnification_holonomic(
        t, values, requested_atol, requested_rtol
    )
    return float(output[0]) if scalar else output


def lightcurve_gradient(curve, times, params=None, *, mu_atol=1e-8, mu_rtol=1e-4):
    """Evaluate the optional fused magnification gradient along a straight trajectory.

    Supports a static binary and no parallax or xallarap. The curve's
    ``param_type`` controls the VBM and center-of-mass coordinate convention.
    ``params`` uses the same ``t0, u0, tE, alpha, rho, q, s`` keys as LightCurve.
    """
    try:
        from _lcbinint_holonomic_fused import BinarySeededFusedGradient
    except ImportError as error:
        raise ImportError(
            "build the optional fused-gradient module and add its build/python "
            "directory to PYTHONPATH"
        ) from error

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
        raise ValueError("fused light-curve gradients currently require no parallax or xallarap")
    if curve._native.orbital_motion != "static":
        raise ValueError("fused light-curve gradients currently require a static binary")
    if model.lens != "binary" or model.source != "single":
        raise ValueError("fused light-curve gradients require a binary lens")

    convention = curve.options.param_type
    vbm = convention.startswith("vbm")
    center_of_mass = convention.endswith("center_of_mass")
    t = np.asarray(times, dtype=float)
    scalar = t.ndim == 0
    t = t.reshape(1) if scalar else t.reshape(-1)
    if not t.size or not np.all(np.isfinite(t)):
        raise ValueError("times must be a nonempty finite sequence")
    geometry = curve.finite_source_geometry(t, values)
    xs = np.asarray(geometry.source_x, dtype=float)
    ys = np.asarray(geometry.source_y, dtype=float)
    separations = np.asarray(geometry.separation, dtype=float)
    solver_qs = np.asarray(geometry.mass_ratio, dtype=float)
    radii = np.asarray(geometry.source_radius, dtype=float)
    limb_coefficients = np.asarray(geometry.limb_darkening_c, dtype=float)
    if scalar:
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        separations = np.atleast_1d(separations)
        solver_qs, radii = np.atleast_1d(solver_qs), np.atleast_1d(radii)
        limb_coefficients = np.atleast_1d(limb_coefficients)
    seeder = BinarySeededFusedGradient(
        float(separations[0]), float(solver_qs[0]), float(radii[0])
    )
    tau = (t - t0) / tE
    c, s = math.cos(alpha), math.sin(alpha)
    if vbm:
        dx_dt0, dy_dt0 = np.full_like(t, -c/tE), np.full_like(t, -s/tE)
        dx_du0, dy_du0 = np.full_like(t, -s), np.full_like(t, c)
        dx_dtE, dy_dtE = -tau*c/tE, -tau*s/tE
        dx_dalpha, dy_dalpha = -tau*s-u0*c, tau*c-u0*s
    else:
        dx_dt0, dy_dt0 = np.full_like(t, -c/tE), np.full_like(t, s/tE)
        dx_du0, dy_du0 = np.full_like(t, s), np.full_like(t, c)
        dx_dtE, dy_dtE = -tau*c/tE, tau*s/tE
        dx_dalpha, dy_dalpha = -tau*s+u0*c, -u0*s-tau*c

    offset_q = 0.0
    offset_s = 0.0
    if not vbm and not center_of_mass and separation > 1.0:
        mass2 = q/(1+q)
        offset_q = (separation-1/separation)/(1+q)**2
        offset_s = mass2*(1+1/separation**2)

    result = []
    for xi, yi, source_r, limb in zip(xs, ys, radii, limb_coefficients):
        source_x, source_y = float(xi), float(yi)
        result.append(seeder.evaluate(
            source_x, source_y, float(limb), mu_atol, mu_rtol,
            vbm or center_of_mass))
    raw = np.asarray([r["grad_mu"] for r in result], dtype=float)
    raw_valid = np.asarray([r["gradient_valid"] for r in result], dtype=bool)
    gx, gy, grho, gq, gs = raw.T
    grad = np.column_stack((
        gx*dx_dt0+gy*dy_dt0,
        gx*dx_du0+gy*dy_du0,
        gx*dx_dtE+gy*dy_dtE,
        gx*dx_dalpha+gy*dy_dalpha,
        grho,
        gq*(-1/q**2 if vbm else 1.0)-gx*offset_q,
        gs-gx*offset_s,
    ))
    position_valid = raw_valid[:, 0] & raw_valid[:, 1]
    q_valid = raw_valid[:, 3] & (raw_valid[:, 0] if offset_q else True)
    s_valid = raw_valid[:, 4] & (raw_valid[:, 0] if offset_s else True)
    gradient_valid = np.column_stack((
        position_valid, position_valid, position_valid, position_valid,
        raw_valid[:, 2], q_valid, s_valid,
    ))
    output = {
        "mu": np.asarray([r["mu"] for r in result]),
        "grad_mu": grad,
        "param_order": names,
        "value_converged": np.asarray([r["value_converged"] for r in result]),
        "gradient_valid": gradient_valid,
        "gradient_precision_checked": False,
        "support_certified": np.asarray([r["support_certified"] for r in result]),
        "seed_support_proven": np.asarray([r["seed_support_proven"] for r in result]),
        "center_image_count": np.asarray([r["center_image_count"] for r in result]),
        "augmented_seed_count": np.asarray([r["augmented_seed_count"] for r in result]),
        "stop": [r["stop"] for r in result],
    }
    if scalar:
        for key in (
            "mu", "grad_mu", "value_converged", "gradient_valid",
            "support_certified", "seed_support_proven", "center_image_count",
            "augmented_seed_count",
        ):
            output[key] = output[key][0]
        output["stop"] = output["stop"][0]
    return output

"""JAX execution backend for the public :mod:`lcbinint` light-curve API."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache


_DEFAULT_PARAMETERS = {
    "t0": 0.0,
    "tE": 1.0,
    "u0": 0.0,
    "alpha": 0.0,
    "s": 1.0,
    "q": 1.0,
    "rho": 0.0,
    "q2": 0.0,
    "sep2": 0.0,
    "ang": 0.0,
    "limb_darkening_c": 0.0,
    "limb_darkening_d": 0.0,
}

_PARAMETER_ALIASES = {
    "t_0": "t0",
    "t_E": "tE",
    "umin": "u0",
    "theta": "alpha",
    "sep": "s",
    "rho1": "rho",
    "omega_xa": "w1",
    "inc_xa": "w2",
    "phi_xa": "w3",
}

# ``params_from_dict`` is the native public mapping contract.  Keep this
# explicit rather than silently ignoring a misspelled inference parameter.
_VALID_PARAMETER_NAMES = frozenset({
    "t0", "t_0", "tE", "t_E", "u0", "umin", "alpha", "theta",
    "s", "sep", "q", "rho", "rho1", "piEN", "piEE", "q2", "sep2",
    "ang", "ra", "dec", "tfix", "obs_lat", "obs_lon",
    "limb_darkening_c", "limb_darkening_d", "g1", "g2", "g3",
    "lom_szs", "lom_ar", "v_sep", "xi_1", "xi_2", "period_xa",
    "ecc_xa", "peri_xa", "inc_xa", "w1", "omega_xa", "w2", "w3",
    "phi_xa", "xa_szs", "xa_ar", "orbital_motion_mode", "t0_2",
    "u0_2", "rho2", "flux_ratio", "source_mass_ratio",
})


def _normalize_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    """Match the native mapping's canonical parameter aliases.

    ``params_from_dict`` applies aliases in mapping iteration order.  Preserve
    that precedence here while retaining the original binary-source keys such
    as ``rho1`` for downstream binary-source handling.
    """

    unknown = next(
        (name for name in parameters if name not in _VALID_PARAMETER_NAMES),
        None,
    )
    if unknown is not None:
        raise KeyError(f"lcbinint: unknown parameter '{unknown}'")

    normalized = dict(parameters)
    canonical_values: dict[str, object] = {}
    for name, value in parameters.items():
        canonical_values[_PARAMETER_ALIASES.get(name, name)] = value
    normalized.update(canonical_values)
    # Native ``inc_xa`` and ``w2`` share the same storage.  The JAX backend
    # uses the former for element modes and the latter for velocity modes.
    # Synchronize both names with the native mapping's insertion-order rule.
    if "w2" in canonical_values:
        normalized["inc_xa"] = canonical_values["w2"]
    return normalized


def _concrete_float(value: object) -> float | None:
    """Return a host scalar when available, without forcing JAX tracers."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_curve_parameters(native_curve, parameters: Mapping[str, object]):
    """Mirror native validation and return a mask for dynamic ``q2`` values."""

    model = native_curve.model
    requires_reference_time = (
        model.parallax
        or native_curve.orbital_motion != "static"
        or (native_curve.source == "binary" and model.xallarap != "none")
    )
    if requires_reference_time and native_curve.t_ref is None:
        raise RuntimeError(
            "LightCurve: t_ref must be set when using parallax, orbital motion, "
            "or binary-source xallarap"
        )

    q2 = parameters.get("q2", 0.0)
    concrete_q2 = _concrete_float(q2)
    if native_curve.lens == "triple":
        if "q2" not in parameters or (
            concrete_q2 is not None and concrete_q2 <= 0.0
        ):
            raise RuntimeError(
                "LightCurve: lens='triple' requires a positive q2 parameter"
            )
    elif concrete_q2 is not None and concrete_q2 > 0.0:
        raise RuntimeError(
            "LightCurve: lens='binary' cannot be used with a positive q2 parameter"
        )

    if concrete_q2 is None:
        # q2 may be an AD tracer (notably in HMC), for which a Python ``if``
        # is invalid.  Return a device-side mask instead of a host callback:
        # invalid compiled inputs become NaN rather than a finite result for
        # the wrong lens model, while valid q2 values remain differentiable.
        if native_curve.lens == "triple":
            return q2 > 0.0
        return q2 <= 0.0
    return None


def _value(parameters: Mapping[str, object], name: str):
    return parameters.get(name, _DEFAULT_PARAMETERS[name])


def _source_trajectory(time, parameters, *, t0_name="t0", u0_name="u0"):
    t0 = parameters.get(t0_name, _DEFAULT_PARAMETERS["t0"])
    u0 = parameters.get(u0_name, _DEFAULT_PARAMETERS["u0"])
    timescale = _value(parameters, "tE")
    angle = _value(parameters, "alpha")
    tau = (time - t0) / timescale
    return tau, u0, angle


def _lens_frame_source(time, parameters, options, *, t0_name="t0", u0_name="u0"):
    import jax.numpy as jnp

    tau, impact_parameter, angle = _source_trajectory(
        time,
        parameters,
        t0_name=t0_name,
        u0_name=u0_name,
    )
    parameter_type = options.param_type
    vbm_compatible = parameter_type in ("vbm", "vbm_center_of_mass")
    if vbm_compatible:
        source_x = (
            tau * jnp.cos(angle) - impact_parameter * jnp.sin(angle)
        )
        source_y = (
            tau * jnp.sin(angle) + impact_parameter * jnp.cos(angle)
        )
    else:
        source_x = (
            impact_parameter * jnp.sin(angle) + tau * jnp.cos(angle)
        )
        source_y = (
            impact_parameter * jnp.cos(angle) - tau * jnp.sin(angle)
        )
        if parameter_type != "center_of_mass":
            separation = jnp.abs(_value(parameters, "s"))
            mass_ratio = jnp.abs(_value(parameters, "q"))
            secondary_mass = mass_ratio / (1.0 + mass_ratio)
            wide_offset = jnp.where(
                separation > 1.0,
                secondary_mass * (separation - 1.0 / separation),
                0.0,
            )
            source_x = source_x - wide_offset
    return source_x, source_y


@lru_cache(maxsize=1)
def _earth_ephemeris():
    from lcbinint_jax import load_earth_ephemeris

    return load_earth_ephemeris()


def _higher_order_offsets(
    native_curve,
    time,
    parameters,
    *,
    include_xallarap=True,
):
    import jax.numpy as jnp

    from lcbinint_jax import (
        annual_parallax_offsets,
        terrestrial_parallax_offsets,
        xallarap_offsets,
    )

    tau_offset = jnp.zeros_like(time)
    beta_offset = jnp.zeros_like(time)
    model = native_curve.model
    reference_time = native_curve.t_ref
    if reference_time is None:
        reference_time = _value(parameters, "t0")

    if model.parallax:
        if native_curve.sky is None:
            raise ValueError("a JAX parallax model requires sky coordinates")
        sky = native_curve.sky
        pi_en = parameters.get("piEN", 0.0)
        pi_ee = parameters.get("piEE", 0.0)
        annual_tau, annual_beta = annual_parallax_offsets(
            time,
            pi_en,
            pi_ee,
            sky.ra_deg,
            sky.dec_deg,
            reference_time,
            _earth_ephemeris(),
        )
        tau_offset = tau_offset + annual_tau
        beta_offset = beta_offset + annual_beta
        if native_curve.site is not None:
            site = native_curve.site
            if site.kind == "space":
                if site.has_position:
                    from lcbinint_jax import space_parallax_offsets

                    site_tau, site_beta = space_parallax_offsets(
                        time,
                        pi_en,
                        pi_ee,
                        sky.ra_deg,
                        sky.dec_deg,
                        site.ephemeris_time,
                        site.ephemeris_position,
                    )
                    tau_offset = tau_offset + site_tau
                    beta_offset = beta_offset + site_beta
            elif model.terrestrial and site.has_position:
                site_tau, site_beta = terrestrial_parallax_offsets(
                    time,
                    pi_en,
                    pi_ee,
                    sky.ra_deg,
                    sky.dec_deg,
                    site.lat_deg,
                    site.lon_deg,
                )
                tau_offset = tau_offset + site_tau
                beta_offset = beta_offset + site_beta

    if include_xallarap and model.xallarap != "none":
        mode = model.xallarap
        xallarap_parameters = {
            "mode": mode,
            "xi_1": parameters.get("xi_1", 0.0),
            "xi_2": parameters.get("xi_2", 0.0),
            "reference_time": reference_time,
        }
        if mode in ("circular_elements", "orbital_elements"):
            xallarap_parameters.update(
                {
                    "period": parameters.get("period_xa", 1.0),
                    "inclination": parameters.get("inc_xa", 0.0),
                    "eccentricity": parameters.get("ecc_xa", 0.0),
                    "periapsis": parameters.get("peri_xa", 0.0),
                }
            )
        else:
            xallarap_parameters.update(
                {
                    "w1": parameters.get("w1", 0.0),
                    "w2": parameters.get("w2", 0.0),
                    "w3": parameters.get("w3", 0.0),
                    "line_of_sight_ratio": parameters.get("xa_szs", 0.0),
                    "semimajor_axis_ratio": parameters.get("xa_ar", 1.0),
                }
            )
        xallarap_tau, xallarap_beta = xallarap_offsets(
            time, **xallarap_parameters
        )
        tau_offset = tau_offset + xallarap_tau
        beta_offset = beta_offset + xallarap_beta
    return tau_offset, beta_offset, reference_time


def _lens_frame_geometry(
    native_curve,
    time,
    parameters,
    options,
    *,
    t0_name="t0",
    u0_name="u0",
):
    dynamic = (
        native_curve.model.parallax
        or native_curve.model.xallarap != "none"
        or native_curve.orbital_motion != "static"
    )
    if not dynamic:
        source_x, source_y = _lens_frame_source(
            time,
            parameters,
            options,
            t0_name=t0_name,
            u0_name=u0_name,
        )
        return source_x, source_y, _value(parameters, "s")
    if options.param_type not in ("vbm", "vbm_center_of_mass"):
        raise NotImplementedError(
            "higher-order JAX trajectories currently require VBM coordinates"
        )
    if native_curve.lens == "triple" and native_curve.orbital_motion != "static":
        raise NotImplementedError(
            "the native triple-lens model also requires a static lens"
        )

    from lcbinint_jax import binary_lens_trajectory

    tau_offset, beta_offset, reference_time = _higher_order_offsets(
        native_curve, time, parameters
    )
    geometry = binary_lens_trajectory(
        time,
        t0=parameters.get(t0_name, _DEFAULT_PARAMETERS["t0"]),
        timescale=_value(parameters, "tE"),
        impact_parameter=parameters.get(
            u0_name, _DEFAULT_PARAMETERS["u0"]
        ),
        separation=_value(parameters, "s"),
        angle=_value(parameters, "alpha"),
        reference_time=reference_time,
        lens_orbit=native_curve.orbital_motion,
        g1=parameters.get("g1", 0.0),
        g2=parameters.get("g2", 0.0),
        g3=parameters.get("g3", 0.0),
        line_of_sight_ratio=parameters.get("lom_szs", 0.0),
        semimajor_axis_ratio=parameters.get("lom_ar", 1.0),
        tau_offset=tau_offset,
        beta_offset=beta_offset,
    )
    return geometry.source_x, geometry.source_y, geometry.separation


def _integration_tolerances(options, lens):
    absolute = options.finite_source_tol
    relative = options.finite_source_reltol
    if not (absolute > 0.0):
        absolute = 0.0 if lens == "triple" else 1.0e-4
    if not (relative > 0.0):
        relative = 1.0e-4
    return absolute, relative


def _limb_darkening(native_curve, parameters):
    limb_c = native_curve.ld_c
    limb_d = native_curve.ld_d
    if limb_c == 0.0 and limb_d == 0.0:
        limb_c = _value(parameters, "limb_darkening_c")
        limb_d = _value(parameters, "limb_darkening_d")
    return limb_c, limb_d


def _magnification_from_geometry(
    native_curve,
    options,
    parameters,
    source_x,
    source_y,
    separation,
    source_radius,
):
    import jax.numpy as jnp

    from lcbinint_jax import (
        binary_magnification_trajectory,
        triple_magnification_batch,
    )

    source_radius = (
        jnp.abs(source_radius) if native_curve.model.finite_source else 0.0
    )
    limb_c, limb_d = _limb_darkening(native_curve, parameters)
    absolute_tolerance, relative_tolerance = _integration_tolerances(
        options, native_curve.lens
    )

    if native_curve.lens == "triple":
        convention = (
            "vbm"
            if options.param_type in ("vbm", "vbm_center_of_mass")
            else "center_of_mass"
        )
        return triple_magnification_batch(
            source_x,
            source_y,
            _value(parameters, "s"),
            _value(parameters, "q"),
            _value(parameters, "q2"),
            _value(parameters, "sep2"),
            _value(parameters, "ang"),
            source_radius,
            limb_c,
            limb_d,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            convention=convention,
        ).magnification

    mass_ratio = _value(parameters, "q")
    if options.param_type in ("vbm", "vbm_center_of_mass"):
        mass_ratio = 1.0 / mass_ratio
    return binary_magnification_trajectory(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        expanded_cartesian_fallback=True,
    ).magnification


def _source_magnification(
    native_curve,
    options,
    time,
    parameters,
    source_radius,
    *,
    t0_name="t0",
    u0_name="u0",
):
    source_x, source_y, separation = _lens_frame_geometry(
        native_curve,
        time,
        parameters,
        options,
        t0_name=t0_name,
        u0_name=u0_name,
    )
    return _magnification_from_geometry(
        native_curve,
        options,
        parameters,
        source_x,
        source_y,
        separation,
        source_radius,
    )


def _binary_source_magnification(
    native_curve,
    options,
    time,
    parameters,
):
    from lcbinint_jax import binary_source_trajectories

    model = native_curve.model
    xallarap_mode = model.xallarap
    source_coordinates = model.source_orbit_coordinates
    required = ["rho1", "rho2", "flux_ratio"]
    if xallarap_mode == "none" or source_coordinates == "trajectory_offset":
        required.extend(("t0_2", "u0_2"))
    if xallarap_mode != "none":
        required.append("source_mass_ratio")
    missing = tuple(name for name in required if name not in parameters)
    if missing:
        raise ValueError(
            "binary source requires " + ", ".join(missing)
        )

    dynamic = (
        model.parallax
        or xallarap_mode != "none"
        or native_curve.orbital_motion != "static"
    )
    if dynamic and options.param_type not in ("vbm", "vbm_center_of_mass"):
        raise NotImplementedError(
            "higher-order JAX trajectories currently require VBM coordinates"
        )
    if native_curve.lens == "triple" and native_curve.orbital_motion != "static":
        raise NotImplementedError(
            "the native triple-lens model also requires a static lens"
        )

    tau_offset, beta_offset, reference_time = _higher_order_offsets(
        native_curve,
        time,
        parameters,
        include_xallarap=False,
    )
    xallarap_parameters = {}
    if xallarap_mode in ("circular_elements", "orbital_elements"):
        xallarap_parameters = {
            "period": parameters.get("period_xa", 1.0),
            "inclination": parameters.get("inc_xa", 0.0),
            "eccentricity": parameters.get("ecc_xa", 0.0),
            "periapsis": parameters.get("peri_xa", 0.0),
        }
    elif xallarap_mode in ("circular_velocity", "kepler_velocity"):
        xallarap_parameters = {
            "w1": parameters.get("w1", 0.0),
            "w2": parameters.get("w2", 0.0),
            "w3": parameters.get("w3", 0.0),
            "line_of_sight_ratio": parameters.get("xa_szs", 0.0),
            "semimajor_axis_ratio": parameters.get("xa_ar", 1.0),
        }

    geometry = binary_source_trajectories(
        time,
        t0=_value(parameters, "t0"),
        timescale=_value(parameters, "tE"),
        impact_parameter=_value(parameters, "u0"),
        separation=_value(parameters, "s"),
        angle=_value(parameters, "alpha"),
        t0_2=parameters.get("t0_2"),
        impact_parameter_2=parameters.get("u0_2"),
        source_mass_ratio=parameters.get("source_mass_ratio"),
        source_orbit_coordinates=source_coordinates,
        xallarap_mode=xallarap_mode,
        xi_1=parameters.get("xi_1", 0.0),
        xi_2=parameters.get("xi_2", 0.0),
        xallarap_parameters=xallarap_parameters,
        reference_time=reference_time,
        tau_offset=tau_offset,
        beta_offset=beta_offset,
        lens_orbit=native_curve.orbital_motion,
        g1=parameters.get("g1", 0.0),
        g2=parameters.get("g2", 0.0),
        g3=parameters.get("g3", 0.0),
        line_of_sight_ratio=parameters.get("lom_szs", 0.0),
        semimajor_axis_ratio=parameters.get("lom_ar", 1.0),
    )
    first = _magnification_from_geometry(
        native_curve,
        options,
        parameters,
        geometry.source1.source_x,
        geometry.source1.source_y,
        geometry.source1.separation,
        parameters["rho1"],
    )
    second = _magnification_from_geometry(
        native_curve,
        options,
        parameters,
        geometry.source2.source_x,
        geometry.source2.source_y,
        geometry.source2.separation,
        parameters["rho2"],
    )
    flux_ratio = parameters["flux_ratio"]
    return (first + flux_ratio * second) / (1.0 + flux_ratio)


def magnification(native_curve, options, time, parameters):
    """Evaluate the public LightCurve contract with JAX arrays and tracers."""

    import jax.numpy as jnp

    if not isinstance(parameters, Mapping):
        raise TypeError("JAX light curves require a parameter mapping")
    parameters = _normalize_parameters(parameters)
    q2_valid = _validate_curve_parameters(native_curve, parameters)
    time = jnp.asarray(time)
    if time.ndim == 0:
        time = jnp.reshape(time, (1,))
    elif time.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    if native_curve.source == "single":
        result = _source_magnification(
            native_curve,
            options,
            time,
            parameters,
            _value(parameters, "rho"),
        )
    elif native_curve.model.xallarap == "none":
        required = ("rho1", "rho2", "flux_ratio", "t0_2", "u0_2")
        missing = tuple(name for name in required if name not in parameters)
        if missing:
            raise ValueError(
                "binary source requires " + ", ".join(missing)
            )
        first = _source_magnification(
            native_curve,
            options,
            time,
            parameters,
            parameters["rho1"],
        )
        second = _source_magnification(
            native_curve,
            options,
            time,
            parameters,
            parameters["rho2"],
            t0_name="t0_2",
            u0_name="u0_2",
        )
        flux_ratio = parameters["flux_ratio"]
        result = (first + flux_ratio * second) / (1.0 + flux_ratio)
    else:
        result = _binary_source_magnification(
            native_curve,
            options,
            time,
            parameters,
        )
    if q2_valid is not None:
        result = jnp.where(q2_valid, result, jnp.nan)
    return result

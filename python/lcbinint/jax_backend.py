"""JAX execution backend for the public :mod:`lcbinint` light-curve API."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import NamedTuple

_NATIVE_METHOD_NAMES = (
    "point_source",
    "hexadecapole",
    "inverse_ray_cartesian",
    "inverse_ray_polar",
    "inverse_ray_spine",
    "source_plane_quadrature",
)


class JaxLightCurveInfo(NamedTuple):
    """Array-valued counterpart of the native ``LightCurveInfo`` diagnostics."""

    times: object
    source_x: object
    source_y: object
    separations: object
    mass_ratios: object
    magnifications: object
    finite_source_magnifications: object
    finite_source_methods: object
    finite_source_error_estimates: object
    finite_source_converged: object
    finite_source_refinement_levels: object
    caustic_distances: object
    point_source_magnifications: object
    point_source_quadrupole_indicators: object
    point_source_cusp_indicators: object
    point_source_ghost_indicators: object
    point_source_planetary_distances2: object
    point_source_ghost_counts: object
    point_source_safety_flags: object
    point_source_safety_tolerances: object
    image_counts: object
    root_candidate_counts: object
    root_duplicate_counts: object
    root_max_residuals: object
    root_needs_high_precision: object
    root_polish_failure_counts: object
    root_used_cold_retry: object
    root_used_high_precision: object
    root_used_warm_start: object

    @property
    def finite_source_method_names(self):
        import numpy as np

        return [
            _NATIVE_METHOD_NAMES[int(method)]
            for method in np.asarray(self.finite_source_methods)
        ]

    @property
    def all_converged(self):
        import numpy as np

        return bool(np.all(np.asarray(self.finite_source_converged)))

    @property
    def unconverged_indices(self):
        import numpy as np

        return np.flatnonzero(
            ~np.asarray(self.finite_source_converged, dtype=bool)
        ).tolist()


class JaxSourceTrajectory(NamedTuple):
    """JAX-array counterpart of the native ``SourceTrajectory``."""

    times: object
    x: object
    y: object


class JaxFiniteSourceGeometry(NamedTuple):
    """JAX-array counterpart of the native finite-source geometry record."""

    source_x: object
    source_y: object
    separation: object
    mass_ratio: object
    source_radius: object
    limb_darkening_c: object
    limb_darkening_d: object
    absolute_tolerance: object
    relative_tolerance: object


class JaxBinarySourceComponent(NamedTuple):
    magnification: object
    trajectory: JaxSourceTrajectory


class JaxBinarySourceComponents(NamedTuple):
    total: object
    source1: JaxBinarySourceComponent
    source2: JaxBinarySourceComponent


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
_VALID_PARAMETER_NAMES = frozenset(
    {
        "t0",
        "t_0",
        "tE",
        "t_E",
        "u0",
        "umin",
        "alpha",
        "theta",
        "s",
        "sep",
        "q",
        "rho",
        "rho1",
        "piEN",
        "piEE",
        "q2",
        "sep2",
        "ang",
        "ra",
        "dec",
        "tfix",
        "obs_lat",
        "obs_lon",
        "limb_darkening_c",
        "limb_darkening_d",
        "g1",
        "g2",
        "g3",
        "lom_szs",
        "lom_ar",
        "v_sep",
        "xi_1",
        "xi_2",
        "period_xa",
        "ecc_xa",
        "peri_xa",
        "inc_xa",
        "w1",
        "omega_xa",
        "w2",
        "w3",
        "phi_xa",
        "xa_szs",
        "xa_ar",
        "orbital_motion_mode",
        "t0_2",
        "u0_2",
        "rho2",
        "flux_ratio",
        "source_mass_ratio",
    }
)


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


def _time_limit_mask(native_curve, options, time):
    """Device-side validity mask for traced parallax evaluation times."""

    import jax.numpy as jnp

    if not native_curve.model.parallax or options.t_lim is None:
        return jnp.ones(jnp.shape(time), dtype=jnp.bool_)
    lower, upper = options.t_lim
    return (time >= lower) & (time <= upper)


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
        if "q2" not in parameters or (concrete_q2 is not None and concrete_q2 <= 0.0):
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
        source_x = tau * jnp.cos(angle) - impact_parameter * jnp.sin(angle)
        source_y = tau * jnp.sin(angle) + impact_parameter * jnp.cos(angle)
    else:
        source_x = impact_parameter * jnp.sin(angle) + tau * jnp.cos(angle)
        source_y = impact_parameter * jnp.cos(angle) - tau * jnp.sin(angle)
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


def _absolute_ephemeris_time(value):
    return value + 2450000.0 if value < 2450000.0 else value


def _ephemeris_window_indices(times, windows, *, padding=2):
    """Return interpolation-safe host indices for one or more time windows."""

    import numpy as np

    times = np.asarray(times)
    selected = []
    for lower, upper in windows:
        if lower < times[0] or upper > times[-1]:
            raise ValueError("Options.t_lim lies outside the available ephemeris table")
        start = max(0, int(np.searchsorted(times, lower, side="right")) - padding)
        stop = min(
            times.size,
            int(np.searchsorted(times, upper, side="left")) + padding + 1,
        )
        selected.extend(range(start, stop))
    return np.unique(np.asarray(selected, dtype=np.int64))


def _trim_ephemeris_arrays(times, *values, t_lim, reference_time=None):
    """Slice constants while retaining bracketing and light-time rows."""

    import numpy as np

    host_times = np.asarray(times)
    lower, upper = (_absolute_ephemeris_time(value) for value in t_lim)
    windows = [(lower, upper)]
    if reference_time is not None:
        reference = _absolute_ephemeris_time(reference_time)
        windows.append((reference, reference))
    indices = _ephemeris_window_indices(host_times, windows)
    return (host_times[indices],) + tuple(
        np.asarray(value)[indices] for value in values
    )


@lru_cache(maxsize=32)
def _earth_ephemeris(t_lim=None, reference_time=None):
    from lcbinint_jax import load_earth_ephemeris

    return load_earth_ephemeris(
        t_lim=t_lim,
        reference_time=reference_time,
    )


def _higher_order_offsets(
    native_curve,
    options,
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
        parallax_time = time
        if options.t_lim is not None:
            lower, upper = options.t_lim
            parallax_time = jnp.clip(time, lower, upper)
        if native_curve.sky is None:
            raise ValueError("a JAX parallax model requires sky coordinates")
        sky = native_curve.sky
        pi_en = parameters.get("piEN", 0.0)
        pi_ee = parameters.get("piEE", 0.0)
        annual_tau, annual_beta = annual_parallax_offsets(
            parallax_time,
            pi_en,
            pi_ee,
            sky.ra_deg,
            sky.dec_deg,
            reference_time,
            _earth_ephemeris(options.t_lim, reference_time),
        )
        tau_offset = tau_offset + annual_tau
        beta_offset = beta_offset + annual_beta
        if native_curve.site is not None:
            site = native_curve.site
            if site.kind == "space":
                if site.has_position:
                    from lcbinint_jax import space_parallax_offsets

                    ephemeris_time = site.ephemeris_time
                    ephemeris_position = site.ephemeris_position
                    if options.t_lim is not None:
                        ephemeris_time, ephemeris_position = _trim_ephemeris_arrays(
                            ephemeris_time,
                            ephemeris_position,
                            t_lim=options.t_lim,
                        )
                    site_tau, site_beta = space_parallax_offsets(
                        parallax_time,
                        pi_en,
                        pi_ee,
                        sky.ra_deg,
                        sky.dec_deg,
                        ephemeris_time,
                        ephemeris_position,
                    )
                    tau_offset = tau_offset + site_tau
                    beta_offset = beta_offset + site_beta
            elif model.terrestrial and site.has_position:
                site_tau, site_beta = terrestrial_parallax_offsets(
                    parallax_time,
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
        xallarap_tau, xallarap_beta = xallarap_offsets(time, **xallarap_parameters)
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
        native_curve, options, time, parameters
    )
    geometry = binary_lens_trajectory(
        time,
        t0=parameters.get(t0_name, _DEFAULT_PARAMETERS["t0"]),
        timescale=_value(parameters, "tE"),
        impact_parameter=parameters.get(u0_name, _DEFAULT_PARAMETERS["u0"]),
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
    explicit = absolute > 0.0 or relative > 0.0
    if not explicit:
        absolute = 0.0 if lens == "triple" else 1.0e-4
        relative = 1.0e-4
    else:
        # Match native semantics: once either component is explicit, zero in
        # the other component means zero rather than "insert its default".
        absolute = max(absolute, 0.0)
        relative = max(relative, 0.0)
    return absolute, relative


def _limb_darkening(native_curve, parameters):
    limb_c = native_curve.ld_c
    limb_d = native_curve.ld_d
    if limb_c == 0.0 and limb_d == 0.0:
        limb_c = _value(parameters, "limb_darkening_c")
        limb_d = _value(parameters, "limb_darkening_d")
    return limb_c, limb_d


def _moment_mode(limb_c, limb_d):
    """Choose the smallest static moment kernel for concrete LD settings."""

    if float(limb_d) != 0.0:
        return "two_coefficient"
    if float(limb_c) != 0.0:
        return "linear"
    return "uniform"


def _jax_resolution_bucket(limit):
    """Clamp a native maximum to the largest compiled calibrated bucket."""

    buckets = (16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400)
    eligible = tuple(bucket for bucket in buckets if bucket <= int(limit))
    return eligible[-1] if eligible else buckets[0]


def _fixed_cartesian_capacity(resolution):
    target = max(256, int(resolution) * int(resolution))
    return 1 << (target - 1).bit_length()


def binary_ray_shooting(
    x,
    y,
    *,
    s,
    q,
    rho,
    limb_darkening,
    options,
):
    """JAX implementation of the public direct finite-source helper."""

    import jax.numpy as jnp

    from lcbinint_jax import (
        binary_inverse_ray,
        binary_inverse_ray_polar,
        binary_magnification_calibrated,
    )

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    separation = jnp.asarray(s)
    mass_ratio = jnp.asarray(q)
    source_radius = jnp.asarray(rho)
    concrete = tuple(
        _concrete_float(value)
        for value in (x, y, separation, mass_ratio, source_radius)
    )
    if all(value is not None for value in concrete):
        import math

        if (
            not all(math.isfinite(value) for value in concrete)
            or concrete[3] <= 0.0
            or concrete[4] <= 0.0
        ):
            raise ValueError(
                "binary_ray_shooting requires finite x, y, s, q and positive rho"
            )
    limb_c = float(limb_darkening.c)
    limb_d = float(limb_darkening.d)
    mode = _moment_mode(limb_c, limb_d)
    absolute_tolerance, relative_tolerance = _integration_tolerances(options, "binary")

    if options.nbin == "auto" and options.inverse_ray_grid == "auto":
        result = binary_magnification_calibrated(
            x,
            y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_source_bins=_jax_resolution_bucket(options.max_source_bins),
            moment_mode=mode,
        )
        value = jnp.where(
            result.support_valid & result.value_converged,
            result.magnification,
            jnp.nan,
        )
        valid_input = (
            jnp.isfinite(x)
            & jnp.isfinite(y)
            & jnp.isfinite(separation)
            & jnp.isfinite(mass_ratio)
            & jnp.isfinite(source_radius)
            & (mass_ratio > 0.0)
            & (source_radius > 0.0)
        )
        return jnp.where(valid_input, value, jnp.nan)

    resolution = int(options.source_bins)
    if options.inverse_ray_grid == "polar":
        angular_target = max(2048, 64 * resolution)
        angular_bins = 1 << (angular_target - 1).bit_length()
        radial_target = max(256, 8 * resolution)
        radial_capacity = 1 << (radial_target - 1).bit_length()
        result = binary_inverse_ray_polar(
            x,
            y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            resolution=resolution,
            angular_bins=angular_bins,
            radial_capacity=radial_capacity,
            limb_samples=64,
            angular_chunk_size=256,
            boundary_subdivision=4,
            moment_mode=mode,
        )
    else:
        result = binary_inverse_ray(
            x,
            y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            resolution=resolution,
            tile_capacity=_fixed_cartesian_capacity(resolution),
            limb_samples=32,
        )
    value = jnp.where(result.support_valid, result.magnification, jnp.nan)
    valid_input = (
        jnp.isfinite(x)
        & jnp.isfinite(y)
        & jnp.isfinite(separation)
        & jnp.isfinite(mass_ratio)
        & jnp.isfinite(source_radius)
        & (mass_ratio > 0.0)
        & (source_radius > 0.0)
    )
    return jnp.where(valid_input, value, jnp.nan)


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
        binary_magnification_native_pipeline_trajectory,
        binary_magnification_trajectory,
        triple_magnification_batch,
    )

    source_radius = jnp.abs(source_radius) if native_curve.model.finite_source else 0.0
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
    grid = options.inverse_ray_grid
    polar_options = {}
    if grid == "cartesian":
        polar_options = {
            "polar_magnification_threshold": jnp.inf,
            "polar_fallback_on_overflow": False,
        }
    elif grid == "polar":
        polar_options = {
            "polar_magnification_threshold": -jnp.inf,
            "polar_max_source_radius": jnp.inf,
            "polar_min_mass_ratio": 0.0,
            "polar_fallback_on_overflow": True,
        }
    if options.nbin != "auto":
        resolution = int(options.source_bins)
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
            resolution=resolution,
            tile_capacity=_fixed_cartesian_capacity(resolution),
            polar_resolution=(
                int(options.polar_source_bins)
                if options.polar_source_bins is not None
                else resolution
            ),
            expanded_cartesian_fallback=False,
            **polar_options,
        ).magnification
    result = binary_magnification_native_pipeline_trajectory(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        maximum_source_bins=_jax_resolution_bucket(options.max_source_bins),
        expanded_cartesian_fallback=True,
        **polar_options,
    )
    return jnp.where(
        result.support_valid & result.value_converged,
        result.magnification,
        jnp.nan,
    )


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
        raise ValueError("binary source requires " + ", ".join(missing))

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
        options,
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
            raise ValueError("binary source requires " + ", ".join(missing))
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
    result = jnp.where(_time_limit_mask(native_curve, options, time), result, jnp.nan)
    if q2_valid is not None:
        result = jnp.where(q2_valid, result, jnp.nan)
    return result


def magnification_batch(native_curve, options, time, parameter_rows):
    """Vectorize the public JAX curve over independent parameter mappings."""

    import jax
    import jax.numpy as jnp

    rows = tuple(_normalize_parameters(row) for row in parameter_rows)
    times = jnp.asarray(time)
    output_size = 1 if times.ndim == 0 else times.shape[0]
    if not rows:
        return jnp.empty((0, output_size), dtype=times.dtype)
    key_set = frozenset(rows[0])
    if any(frozenset(row) != key_set for row in rows[1:]):
        # Native accepts independently sparse mappings. Preserve that contract
        # when row PyTrees differ; the common inference case below is vmapped.
        return jnp.stack(
            tuple(magnification(native_curve, options, times, row) for row in rows)
        )
    stacked = {
        key: jnp.stack(tuple(jnp.asarray(row[key]) for row in rows)) for key in key_set
    }
    return jax.vmap(lambda active: magnification(native_curve, options, times, active))(
        stacked
    )


def light_curve_log_likelihood_batch(
    native_curve,
    options,
    times,
    flux,
    error,
    parameter_rows,
    distribution,
    flux_mode,
    nu,
    sampled_source,
    sampled_blend,
):
    """JAX likelihood counterpart of the native batch inference helper."""

    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    if distribution not in ("gaussian", "student_t"):
        raise ValueError("distribution must be 'gaussian' or 'student_t'")
    if flux_mode not in ("fit", "sample", "marginalize"):
        raise ValueError("flux_mode must be 'fit', 'sample', or 'marginalize'")
    if flux_mode == "marginalize" and distribution != "gaussian":
        raise ValueError("marginalize currently requires a gaussian likelihood")
    if distribution == "student_t" and not nu > 0.0:
        raise ValueError("nu must be positive")
    times = jnp.asarray(times)
    flux = jnp.asarray(flux)
    error = jnp.asarray(error)
    if times.ndim != 1 or flux.ndim != 1 or error.ndim != 1:
        raise ValueError("times, flux, and error must be one-dimensional")
    if times.shape != flux.shape or times.shape != error.shape:
        raise ValueError("times, flux, and error must have the same shape")
    concrete_flux = None
    concrete_error = None
    try:
        import numpy as np

        concrete_flux = np.asarray(flux)
        concrete_error = np.asarray(error)
    except Exception:
        pass
    if concrete_error is not None and (
        not np.all(np.isfinite(concrete_flux))
        or not np.all(np.isfinite(concrete_error))
        or np.any(concrete_error <= 0.0)
    ):
        raise ValueError("flux must be finite and errors must be finite and positive")

    lens_rows = tuple(
        {key: value for key, value in row.items() if not key.startswith(("Fs_", "Fb_"))}
        for row in parameter_rows
    )
    magnifications = magnification_batch(native_curve, options, times, lens_rows)
    row_count = magnifications.shape[0]
    weights = 1.0 / jnp.square(error)
    if flux_mode in ("fit", "marginalize"):
        aa = jnp.sum(weights * jnp.square(magnifications), axis=1)
        ab = jnp.sum(weights * magnifications, axis=1)
        bb = jnp.sum(weights) * jnp.ones(row_count, dtype=flux.dtype)
        ay = jnp.sum(weights * magnifications * flux, axis=1)
        by = jnp.sum(weights * flux) * jnp.ones(row_count, dtype=flux.dtype)
        determinant = aa * bb - ab * ab
        source = (ay * bb - by * ab) / determinant
        blend = (by * aa - ay * ab) / determinant
    else:
        if sampled_source is None or sampled_blend is None:
            raise ValueError(
                "sample flux mode requires source_flux and blend_flux arrays"
            )
        source = jnp.asarray(sampled_source)
        blend = jnp.asarray(sampled_blend)
        if source.shape != (row_count,) or blend.shape != (row_count,):
            raise ValueError("sampled flux arrays must match the parameter-row count")
        determinant = jnp.ones(row_count, dtype=flux.dtype)

    residual = (
        flux[None, :] - source[:, None] * magnifications - blend[:, None]
    ) / error[None, :]
    chi2 = jnp.sum(jnp.square(residual), axis=1)
    if distribution == "gaussian":
        if flux_mode == "marginalize":
            degrees = times.shape[0] - 2
            log_likelihood = -0.5 * degrees * jnp.log(chi2) - 0.5 * jnp.log(determinant)
        else:
            log_likelihood = -0.5 * chi2
    else:
        constant = (
            gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * jnp.log(nu * jnp.pi)
        )
        log_likelihood = jnp.sum(
            constant
            - jnp.log(error)[None, :]
            - 0.5 * (nu + 1.0) * jnp.log1p(jnp.square(residual) / nu),
            axis=1,
        )
    if flux_mode == "marginalize":
        conditional_df = jnp.full((row_count,), times.shape[0] - 2, dtype=flux.dtype)
        conditional_scale = jnp.sqrt(
            chi2 / conditional_df * jnp.sum(weights) / determinant
        )
    else:
        conditional_df = jnp.full((row_count,), jnp.nan, dtype=flux.dtype)
        conditional_scale = jnp.full((row_count,), jnp.nan, dtype=flux.dtype)
    return {
        "log_likelihood": log_likelihood,
        "source_flux": source,
        "blend_flux": blend,
        "conditional_scale": conditional_scale,
        "conditional_df": conditional_df,
    }


def source_trajectory(native_curve, options, time, parameters):
    """Return the selected JAX trajectory without solving lens images."""

    import jax.numpy as jnp

    if native_curve.source != "single":
        raise NotImplementedError(
            "use binary_source_components for a binary-source trajectory"
        )
    parameters = _normalize_parameters(parameters)
    _validate_curve_parameters(native_curve, parameters)
    times = jnp.asarray(time)
    if times.ndim == 0:
        times = jnp.reshape(times, (1,))
    elif times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    source_x, source_y, _ = _lens_frame_geometry(
        native_curve, times, parameters, options
    )
    valid_time = _time_limit_mask(native_curve, options, times)
    return JaxSourceTrajectory(
        times=times,
        x=jnp.where(valid_time, source_x, jnp.nan),
        y=jnp.where(valid_time, source_y, jnp.nan),
    )


def binary_source_components(native_curve, options, time, parameters):
    """Return differentiable static binary-source component curves."""

    import jax.numpy as jnp

    if native_curve.source != "binary":
        raise RuntimeError(
            "binary_source_components requires LightCurve(source='binary')"
        )
    if native_curve.model.xallarap != "none":
        raise NotImplementedError(
            "JAX binary_source_components for xallarap is not yet exposed; "
            "the combined LightCurve remains differentiable"
        )
    parameters = _normalize_parameters(parameters)
    required = ("rho1", "rho2", "flux_ratio", "t0_2", "u0_2")
    missing = tuple(name for name in required if name not in parameters)
    if missing:
        raise ValueError("binary source requires " + ", ".join(missing))
    times = jnp.asarray(time)
    if times.ndim == 0:
        times = jnp.reshape(times, (1,))
    elif times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")

    first_x, first_y, first_separation = _lens_frame_geometry(
        native_curve, times, parameters, options
    )
    second_x, second_y, second_separation = _lens_frame_geometry(
        native_curve,
        times,
        parameters,
        options,
        t0_name="t0_2",
        u0_name="u0_2",
    )
    first_magnification = _magnification_from_geometry(
        native_curve,
        options,
        parameters,
        first_x,
        first_y,
        first_separation,
        parameters["rho1"],
    )
    second_magnification = _magnification_from_geometry(
        native_curve,
        options,
        parameters,
        second_x,
        second_y,
        second_separation,
        parameters["rho2"],
    )
    flux_ratio = parameters["flux_ratio"]
    total = (first_magnification + flux_ratio * second_magnification) / (
        1.0 + flux_ratio
    )
    valid_time = _time_limit_mask(native_curve, options, times)
    first_x = jnp.where(valid_time, first_x, jnp.nan)
    first_y = jnp.where(valid_time, first_y, jnp.nan)
    second_x = jnp.where(valid_time, second_x, jnp.nan)
    second_y = jnp.where(valid_time, second_y, jnp.nan)
    first_magnification = jnp.where(valid_time, first_magnification, jnp.nan)
    second_magnification = jnp.where(valid_time, second_magnification, jnp.nan)
    total = jnp.where(valid_time, total, jnp.nan)
    return JaxBinarySourceComponents(
        total=total,
        source1=JaxBinarySourceComponent(
            magnification=first_magnification,
            trajectory=JaxSourceTrajectory(times=times, x=first_x, y=first_y),
        ),
        source2=JaxBinarySourceComponent(
            magnification=second_magnification,
            trajectory=JaxSourceTrajectory(times=times, x=second_x, y=second_y),
        ),
    )


def finite_source_geometry(native_curve, options, time, parameters):
    """Return differentiable geometry using the same public model pipeline."""

    import jax.numpy as jnp

    if native_curve.lens != "binary" or native_curve.source != "single":
        raise NotImplementedError(
            "JAX finite_source_geometry currently supports a "
            "single-source binary LightCurve"
        )
    parameters = _normalize_parameters(parameters)
    _validate_curve_parameters(native_curve, parameters)
    times = jnp.asarray(time)
    if times.ndim == 0:
        times = jnp.reshape(times, (1,))
    elif times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    source_x, source_y, separation = _lens_frame_geometry(
        native_curve, times, parameters, options
    )
    separation = jnp.asarray(separation)
    if separation.ndim == 0:
        separation = jnp.full_like(source_x, separation)
    mass_ratio = _value(parameters, "q")
    if options.param_type in ("vbm", "vbm_center_of_mass"):
        mass_ratio = 1.0 / mass_ratio
    source_radius = jnp.abs(_value(parameters, "rho"))
    limb_c, limb_d = _limb_darkening(native_curve, parameters)
    absolute_tolerance, relative_tolerance = _integration_tolerances(options, "binary")
    valid_time = _time_limit_mask(native_curve, options, times)
    source_x = jnp.where(valid_time, source_x, jnp.nan)
    source_y = jnp.where(valid_time, source_y, jnp.nan)
    separation = jnp.where(valid_time, separation, jnp.nan)
    return JaxFiniteSourceGeometry(
        source_x=source_x,
        source_y=source_y,
        separation=separation,
        mass_ratio=jnp.full_like(source_x, mass_ratio),
        source_radius=jnp.full_like(source_x, source_radius),
        limb_darkening_c=jnp.full_like(source_x, limb_c),
        limb_darkening_d=jnp.full_like(source_x, limb_d),
        absolute_tolerance=jnp.full_like(source_x, absolute_tolerance),
        relative_tolerance=jnp.full_like(source_x, relative_tolerance),
    )


def separation(native_curve, options, time, parameters):
    """Return scalar or epoch-dependent JAX lens separation."""

    import jax.numpy as jnp

    parameters = _normalize_parameters(parameters)
    _validate_curve_parameters(native_curve, parameters)
    if time is None:
        if native_curve.orbital_motion == "static":
            return jnp.asarray(_value(parameters, "s"))
        time = parameters.get("t0", _DEFAULT_PARAMETERS["t0"])
    times = jnp.asarray(time)
    scalar = times.ndim == 0
    if scalar:
        times = jnp.reshape(times, (1,))
    elif times.ndim != 1:
        raise ValueError("times must be a scalar or one-dimensional array")
    _, _, values = _lens_frame_geometry(native_curve, times, parameters, options)
    values = jnp.asarray(values)
    if values.ndim == 0:
        values = jnp.full_like(times, values)
    values = jnp.where(_time_limit_mask(native_curve, options, times), values, jnp.nan)
    return values[0] if scalar else values


def info(native_curve, options, time, parameters):
    """Return JAX-backend diagnostics for a single-source binary curve."""

    import jax
    import jax.numpy as jnp

    from lcbinint_jax import (
        binary_hexadecapole,
        binary_hexadecapole_batch_ffi,
        binary_magnification_native_pipeline_trajectory,
        binary_magnification_trajectory,
        binary_routing_diagnostics_batch_ffi,
        cpp_binary_routing_diagnostics_batch_ffi_available,
    )

    if native_curve.lens != "binary" or native_curve.source != "single":
        raise NotImplementedError(
            "JAX info currently supports a single-source binary LightCurve"
        )
    if not isinstance(parameters, Mapping):
        raise TypeError("JAX light curves require a parameter mapping")
    parameters = _normalize_parameters(parameters)
    _validate_curve_parameters(native_curve, parameters)
    times = jnp.asarray(time)
    if times.ndim == 0:
        times = jnp.reshape(times, (1,))
    elif times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    source_x, source_y, separation = _lens_frame_geometry(
        native_curve, times, parameters, options
    )
    source_radius = jnp.abs(_value(parameters, "rho"))
    if not native_curve.model.finite_source:
        source_radius = jnp.asarray(0.0, dtype=source_x.dtype)
    limb_c, limb_d = _limb_darkening(native_curve, parameters)
    absolute_tolerance, relative_tolerance = _integration_tolerances(options, "binary")
    mass_ratio = _value(parameters, "q")
    if options.param_type in ("vbm", "vbm_center_of_mass"):
        mass_ratio = 1.0 / mass_ratio
    grid = options.inverse_ray_grid
    polar_options = {}
    if grid == "cartesian":
        polar_options = {
            "polar_magnification_threshold": jnp.inf,
            "polar_fallback_on_overflow": False,
        }
    elif grid == "polar":
        polar_options = {
            "polar_magnification_threshold": -jnp.inf,
            "polar_max_source_radius": jnp.inf,
            "polar_min_mass_ratio": 0.0,
            "polar_fallback_on_overflow": True,
        }
    if options.nbin == "auto":
        result = binary_magnification_native_pipeline_trajectory(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_source_bins=_jax_resolution_bucket(options.max_source_bins),
            expanded_cartesian_fallback=True,
            **polar_options,
        )
    else:
        resolution = int(options.source_bins)
        result = binary_magnification_trajectory(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            resolution=resolution,
            tile_capacity=_fixed_cartesian_capacity(resolution),
            polar_resolution=(
                int(options.polar_source_bins)
                if options.polar_source_bins is not None
                else resolution
            ),
            expanded_cartesian_fallback=False,
            **polar_options,
        )

    separation_array = jnp.asarray(separation)
    if (
        separation_array.ndim == 0
        and cpp_binary_routing_diagnostics_batch_ffi_available()
    ):
        expansion = binary_hexadecapole_batch_ffi(
            source_x,
            source_y,
            separation_array,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        )
        routing = binary_routing_diagnostics_batch_ffi(
            source_x,
            source_y,
            jax.lax.stop_gradient(expansion.point_magnification),
            separation_array,
            mass_ratio,
            source_radius,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        caustic_distance = routing.caustic_distance
        quadrupole = routing.quadrupole_indicator
        cusp = routing.cusp_indicator
        ghost = routing.ghost_indicator
        planetary_distance2 = routing.planetary_distance2
        ghost_count = routing.ghost_count
        safety_flags = routing.safety_flags
        safety_tolerance = routing.point_absolute_tolerance
        image_count = routing.image_count
    else:
        separation_vector = (
            jnp.full_like(source_x, separation_array)
            if separation_array.ndim == 0
            else separation_array
        )

        def expansion_epoch(position):
            return binary_hexadecapole(
                position[0],
                position[1],
                position[2],
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
            )

        expansion = jax.lax.map(
            expansion_epoch, (source_x, source_y, separation_vector)
        )
        shape = source_x.shape
        caustic_distance = jnp.full(shape, jnp.inf, dtype=source_x.dtype)
        quadrupole = jnp.full(shape, jnp.nan, dtype=source_x.dtype)
        cusp = jnp.full(shape, jnp.nan, dtype=source_x.dtype)
        ghost = jnp.full(shape, jnp.nan, dtype=source_x.dtype)
        planetary_distance2 = jnp.full(shape, jnp.nan, dtype=source_x.dtype)
        ghost_count = jnp.zeros(shape, dtype=jnp.int32)
        safety_flags = jnp.zeros(shape, dtype=jnp.int32)
        safety_tolerance = jnp.full(shape, jnp.nan, dtype=source_x.dtype)
        image_count = jnp.zeros(shape, dtype=jnp.int32)

    native_method_codes = jnp.asarray((1, 2, 3, 5, 0), dtype=jnp.int32)[
        result.method
    ]
    valid_time = _time_limit_mask(native_curve, options, times)
    value_valid = result.support_valid & result.value_converged
    magnification = jnp.where(valid_time & value_valid, result.magnification, jnp.nan)
    source_x = jnp.where(valid_time, source_x, jnp.nan)
    source_y = jnp.where(valid_time, source_y, jnp.nan)
    caustic_distance = jnp.where(valid_time, caustic_distance, jnp.nan)
    point_magnification = jnp.where(valid_time, expansion.point_magnification, jnp.nan)
    converged = valid_time & value_valid & jnp.isfinite(magnification)
    separation_vector = (
        jnp.full_like(source_x, separation_array)
        if separation_array.ndim == 0
        else separation_array
    )
    integer_zeros = jnp.zeros(source_x.shape, dtype=jnp.int32)
    floating_zeros = jnp.zeros(source_x.shape, dtype=source_x.dtype)
    boolean_zeros = jnp.zeros(source_x.shape, dtype=jnp.bool_)
    return JaxLightCurveInfo(
        times=times,
        source_x=source_x,
        source_y=source_y,
        separations=separation_vector,
        mass_ratios=jnp.full_like(source_x, mass_ratio),
        magnifications=magnification,
        # Keep the unaccepted last iterate available only in the explicitly
        # diagnostic field; the ordinary magnification fails closed.
        finite_source_magnifications=jnp.where(
            valid_time & result.support_valid, result.magnification, jnp.nan
        ),
        finite_source_methods=native_method_codes,
        finite_source_error_estimates=result.estimated_error,
        finite_source_converged=converged,
        finite_source_refinement_levels=integer_zeros,
        caustic_distances=caustic_distance,
        point_source_magnifications=point_magnification,
        point_source_quadrupole_indicators=quadrupole,
        point_source_cusp_indicators=cusp,
        point_source_ghost_indicators=ghost,
        point_source_planetary_distances2=planetary_distance2,
        point_source_ghost_counts=ghost_count,
        point_source_safety_flags=safety_flags,
        point_source_safety_tolerances=safety_tolerance,
        image_counts=image_count,
        root_candidate_counts=integer_zeros,
        root_duplicate_counts=integer_zeros,
        root_max_residuals=floating_zeros,
        root_needs_high_precision=boolean_zeros,
        root_polish_failure_counts=integer_zeros,
        root_used_cold_retry=boolean_zeros,
        root_used_high_precision=boolean_zeros,
        root_used_warm_start=boolean_zeros,
    )

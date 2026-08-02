"""Differentiable microlensing trajectories with higher-order effects.

The functions in this module intentionally stop at geometry.  JAX constructs
``source_x(t)``, ``source_y(t)``, and ``separation(t)``; the fused C++ inverse-
ray trajectory kernel then differentiates magnification with respect to those
three quantities.  JAX applies the remaining chain rule to physical event,
parallax, lens-orbit, and xallarap parameters.
"""

from importlib import resources
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_PI = np.pi
_DEGREE_TO_RADIAN = _PI / 180.0
_RJD_ORIGIN = 2450000.0
_AU_LIGHT_TRAVEL_DAYS = 0.005775518331436995
_BRANCH_EPS = 1.0e-8


class OrbitalState(NamedTuple):
    """Projected two-body state in Einstein-radius units."""

    separation: jax.Array
    angle: jax.Array
    line_of_sight_separation: jax.Array
    valid: jax.Array


class EarthEphemeris(NamedTuple):
    """Earth barycentric state table in AU and AU/day."""

    time: jax.Array
    position: jax.Array
    velocity: jax.Array


class TrajectoryGeometry(NamedTuple):
    """Per-epoch binary-lens geometry accepted by the fused trajectory FFI."""

    source_x: jax.Array
    source_y: jax.Array
    separation: jax.Array
    lens_angle: jax.Array
    line_of_sight_separation: jax.Array
    valid: jax.Array


class BinarySourceTrajectory(NamedTuple):
    """Lens-frame trajectories of both members of a binary source."""

    source1: TrajectoryGeometry
    source2: TrajectoryGeometry


class HigherOrderMagnificationResult(NamedTuple):
    """A fused magnification result together with its physical geometry."""

    magnification: object
    geometry: TrajectoryGeometry


class BinarySourceMagnificationResult(NamedTuple):
    """Flux-weighted binary-source result and both component evaluations."""

    total: jax.Array
    source1: HigherOrderMagnificationResult
    source2: HigherOrderMagnificationResult
    flux_ratio: jax.Array


def _safe_sqrt(value):
    return jnp.sqrt(jnp.maximum(value, 0.0))


def _normalize(value):
    norm = _safe_sqrt(jnp.sum(value * value, axis=-1, keepdims=True))
    return value / jnp.where(norm > 0.0, norm, jnp.nan)


def _solve_kepler(mean_anomaly, eccentricity):
    eccentric_anomaly = mean_anomaly + eccentricity * jnp.sin(mean_anomaly)

    def iteration(_, current):
        residual = current - eccentricity * jnp.sin(current) - mean_anomaly
        derivative = 1.0 - eccentricity * jnp.cos(current)
        return current - residual / derivative

    return jax.lax.fori_loop(0, 10, iteration, eccentric_anomaly)


def circular_orbital_motion(
    time,
    separation,
    angle,
    w1,
    w2,
    w3,
    reference_time,
):
    """Match lcbinint's circular 3-D lens-orbit parameterization."""

    time = jnp.asarray(time)
    w13 = _safe_sqrt(w1 * w1 + w3 * w3)
    w123 = _safe_sqrt(w13 * w13 + w2 * w2)
    inclined = w13 > _BRANCH_EPS
    safe_w13 = jnp.where(inclined, w13, 1.0)
    w3_effective = jnp.maximum(w3, _BRANCH_EPS)
    inclined_frequency = w3_effective * w123 / safe_w13
    cos_inclination = jnp.clip(
        w2 * w3_effective / (safe_w13 * jnp.maximum(w123, _BRANCH_EPS)),
        -1.0,
        1.0,
    )
    inclination = jnp.where(inclined, jnp.arccos(cos_inclination), 0.0)
    phase0 = jnp.where(
        inclined,
        jnp.arctan2(-w1 * w123, w3_effective * safe_w13),
        0.0,
    )
    frequency = jnp.where(inclined, inclined_frequency, w2)

    cos_phase0 = jnp.cos(phase0)
    sin_phase0 = jnp.sin(phase0)
    cos_inclination = jnp.cos(inclination)
    sin_inclination = jnp.sin(inclination)
    denominator0 = _safe_sqrt(
        cos_phase0 * cos_phase0
        + cos_inclination * cos_inclination * sin_phase0 * sin_phase0
    )
    true_separation = separation / denominator0

    cos_angle0 = jnp.cos(angle)
    sin_angle0 = jnp.sin(angle)
    cos_node = (
        cos_phase0 * cos_angle0
        + cos_inclination * sin_angle0 * sin_phase0
    ) / denominator0
    sin_node = (
        cos_phase0 * sin_angle0
        - cos_inclination * cos_angle0 * sin_phase0
    ) / denominator0

    phase = frequency * (time - reference_time) + phase0
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)
    denominator = _safe_sqrt(
        cos_phase * cos_phase
        + cos_inclination * cos_inclination * sin_phase * sin_phase
    )
    projected_separation = true_separation * denominator
    sin_angle = (
        cos_phase * sin_node + cos_inclination * sin_phase * cos_node
    ) / denominator
    cos_angle = (
        cos_phase * cos_node - cos_inclination * sin_phase * sin_node
    ) / denominator
    projected_angle = jnp.arctan2(sin_angle, cos_angle)
    line_of_sight = true_separation * sin_inclination * sin_phase
    valid = (
        jnp.isfinite(projected_separation)
        & jnp.isfinite(projected_angle)
        & (projected_separation > 0.0)
    )
    return OrbitalState(
        projected_separation,
        projected_angle,
        line_of_sight,
        valid,
    )


def kepler_orbital_motion(
    time,
    separation,
    angle,
    w1,
    w2,
    w3,
    line_of_sight_ratio,
    semimajor_axis_ratio,
    reference_time,
):
    """Match lcbinint's bound Kepler orbit defined by its reference state."""

    time = jnp.asarray(time)
    ar = semimajor_axis_ratio + _BRANCH_EPS
    szs = line_of_sight_ratio
    smix = 1.0 + szs * szs
    sqsmix = _safe_sqrt(smix)
    w11 = w1 * w1
    w22 = w2 * w2
    w33 = w3 * w3
    w12 = w11 + w22
    wt2 = w12 + w33
    arm1 = ar - 1.0
    arm2 = 2.0 * ar - 1.0
    mean_motion = _safe_sqrt(wt2 / arm2 / smix) / ar

    z_axis = _normalize(jnp.asarray((-szs * w2, szs * w1 - w3, w2)))
    x_axis_unscaled = jnp.asarray(
        (
            -ar * w11 + arm1 * w22 - arm2 * szs * w1 * w3 + arm1 * w33,
            -arm2 * w2 * (w1 + szs * w3),
            arm1 * szs * w12 - arm2 * w1 * w3 - ar * szs * w33,
        )
    )
    x_norm = _safe_sqrt(jnp.sum(x_axis_unscaled * x_axis_unscaled))
    x_axis = x_axis_unscaled / x_norm
    eccentricity = x_norm / (ar * sqsmix * wt2)
    y_axis = jnp.cross(z_axis, x_axis)

    conu = (x_axis[0] + x_axis[2] * szs) / sqsmix
    cos_e0 = jnp.clip(
        (conu + eccentricity) / (1.0 + eccentricity * conu),
        -1.0,
        1.0,
    )
    sign = jnp.where(y_axis[0] + y_axis[2] * szs > 0.0, 1.0, -1.0)
    e0 = jnp.arccos(cos_e0) * sign
    sin_e0 = _safe_sqrt(1.0 - cos_e0 * cos_e0) * sign
    time_periapsis = reference_time - (
        e0 - eccentricity * sin_e0
    ) / mean_motion
    semimajor_axis = ar * separation * sqsmix

    mean_anomaly = mean_motion * (time - time_periapsis)
    eccentric_anomaly = _solve_kepler(mean_anomaly, eccentricity)
    r0 = semimajor_axis * (jnp.cos(eccentric_anomaly) - eccentricity)
    r1 = (
        semimajor_axis
        * _safe_sqrt(1.0 - eccentricity * eccentricity)
        * jnp.sin(eccentric_anomaly)
    )
    position = r0[..., None] * x_axis + r1[..., None] * y_axis
    projected_separation = _safe_sqrt(
        position[..., 0] * position[..., 0]
        + position[..., 1] * position[..., 1]
    )
    projected_angle = angle + jnp.arctan2(position[..., 1], position[..., 0])
    valid_geometry = (
        jnp.isfinite(mean_motion)
        & (mean_motion > 0.0)
        & jnp.all(jnp.isfinite(z_axis))
        & jnp.isfinite(x_norm)
        & (x_norm > 0.0)
        & jnp.isfinite(eccentricity)
        & (eccentricity >= 0.0)
        & (eccentricity < 1.0)
        & (semimajor_axis_ratio > 0.5)
    )
    valid = (
        valid_geometry
        & jnp.isfinite(projected_separation)
        & jnp.isfinite(projected_angle)
        & (projected_separation > 0.0)
    )
    return OrbitalState(
        jnp.where(valid, projected_separation, jnp.nan),
        jnp.where(valid, projected_angle, jnp.nan),
        jnp.where(valid, position[..., 2], jnp.nan),
        valid,
    )


def _sky_basis(ra_degrees, dec_degrees):
    ra = ra_degrees * _DEGREE_TO_RADIAN
    dec = dec_degrees * _DEGREE_TO_RADIAN
    event = jnp.asarray(
        (jnp.cos(ra) * jnp.cos(dec), jnp.sin(ra) * jnp.cos(dec), jnp.sin(dec))
    )
    east = _normalize(jnp.cross(jnp.asarray((0.0, 0.0, 1.0)), event))
    north = jnp.cross(event, east)
    return event, north, east


def _interpolate_ephemeris(ephemeris, time):
    position = jnp.stack(
        tuple(
            jnp.interp(time, ephemeris.time, ephemeris.position[:, axis])
            for axis in range(3)
        ),
        axis=-1,
    )
    velocity = jnp.stack(
        tuple(
            jnp.interp(time, ephemeris.time, ephemeris.velocity[:, axis])
            for axis in range(3)
        ),
        axis=-1,
    )
    return position, velocity


def annual_parallax_offsets(
    time,
    pi_en,
    pi_ee,
    ra_degrees,
    dec_degrees,
    reference_time,
    ephemeris,
):
    """Return native-convention annual parallax offsets in ``(tau, beta)``."""

    time = jnp.asarray(time)
    time_offset = jnp.where(reference_time < _RJD_ORIGIN, _RJD_ORIGIN, 0.0)
    ephemeris_time = time + time_offset
    reference_ephemeris_time = reference_time + time_offset
    event, north, east = _sky_basis(ra_degrees, dec_degrees)

    def light_time_correct(observation_time):
        emit_time = observation_time
        for _ in range(5):
            position, _ = _interpolate_ephemeris(ephemeris, emit_time)
            emit_time = observation_time - (
                jnp.sum(position * event, axis=-1) * _AU_LIGHT_TRAVEL_DAYS
            )
        return emit_time

    emit_time = light_time_correct(ephemeris_time)
    reference_emit_time = light_time_correct(reference_ephemeris_time)
    position, _ = _interpolate_ephemeris(ephemeris, emit_time)
    reference_position, reference_velocity = _interpolate_ephemeris(
        ephemeris, reference_emit_time
    )
    projected_position = jnp.stack(
        (-jnp.sum(position * north, axis=-1), -jnp.sum(position * east, axis=-1)),
        axis=-1,
    )
    projected_reference = jnp.asarray(
        (
            -jnp.sum(reference_position * north),
            -jnp.sum(reference_position * east),
        )
    )
    projected_velocity = jnp.asarray(
        (
            -jnp.sum(reference_velocity * north),
            -jnp.sum(reference_velocity * east),
        )
    )
    displacement = (
        projected_position
        - projected_reference
        - (ephemeris_time - reference_ephemeris_time)[..., None]
        * projected_velocity
    )
    tau = pi_en * displacement[..., 0] + pi_ee * displacement[..., 1]
    beta = -pi_ee * displacement[..., 0] + pi_en * displacement[..., 1]
    valid = (
        (time >= ephemeris.time[0] - time_offset)
        & (time <= ephemeris.time[-1] - time_offset)
    )
    return jnp.where(valid, tau, jnp.nan), jnp.where(valid, beta, jnp.nan)


def terrestrial_parallax_offsets(
    time,
    pi_en,
    pi_ee,
    ra_degrees,
    dec_degrees,
    latitude_degrees,
    longitude_degrees,
):
    """Return the native geocentric observatory correction."""

    earth_radius_au = 4.2635212e-5
    sidereal_degrees_per_day = 360.98564736629
    time = jnp.asarray(time)
    latitude = latitude_degrees * _DEGREE_TO_RADIAN
    longitude = longitude_degrees * _DEGREE_TO_RADIAN
    julian_date = time + jnp.where(time < _RJD_ORIGIN, _RJD_ORIGIN, 0.0)
    gmst = (
        280.46061837
        + sidereal_degrees_per_day * (julian_date - 2451545.0)
    ) * _DEGREE_TO_RADIAN
    hour_angle = gmst + longitude
    telescope = earth_radius_au * jnp.stack(
        (
            jnp.cos(latitude) * jnp.cos(hour_angle),
            jnp.cos(latitude) * jnp.sin(hour_angle),
            jnp.broadcast_to(jnp.sin(latitude), time.shape),
        ),
        axis=-1,
    )
    _, north, east = _sky_basis(ra_degrees, dec_degrees)
    projected_north = -jnp.sum(telescope * north, axis=-1)
    projected_east = -jnp.sum(telescope * east, axis=-1)
    return (
        pi_en * projected_north + pi_ee * projected_east,
        -pi_ee * projected_north + pi_en * projected_east,
    )


def space_parallax_offsets(
    time,
    pi_en,
    pi_ee,
    ra_degrees,
    dec_degrees,
    ephemeris_time,
    position=None,
):
    """Return native/VBM-compatible satellite parallax offsets.

    Pass either ``(ephemeris_time, position)`` with Cartesian AU coordinates,
    or pass a native/VBM-style ``(N, 4)`` table as ``ephemeris_time`` with
    columns ``JD, RA_deg, Dec_deg, distance_AU``.
    """

    time = jnp.asarray(time)
    query_time = time + jnp.where(time < _RJD_ORIGIN, _RJD_ORIGIN, 0.0)
    ephemeris_time = jnp.asarray(ephemeris_time)
    if position is None:
        table = ephemeris_time
        if table.ndim != 2 or table.shape[1] != 4:
            raise ValueError("space ephemeris table must have shape (N, 4)")
        ephemeris_time = table[:, 0]
        satellite_ra = table[:, 1] * _DEGREE_TO_RADIAN
        satellite_dec = table[:, 2] * _DEGREE_TO_RADIAN
        distance = table[:, 3]
        along_equator = distance * jnp.cos(satellite_dec) * jnp.cos(satellite_ra)
        along_quad = distance * jnp.cos(satellite_dec) * jnp.sin(satellite_ra)
        along_north = distance * jnp.sin(satellite_dec)
        cos_obliquity = 0.9174820003578725
        sin_obliquity = 0.3977772982704228
        position = jnp.stack(
            (
                along_equator,
                along_quad * cos_obliquity + along_north * sin_obliquity,
                -along_quad * sin_obliquity + along_north * cos_obliquity,
            ),
            axis=-1,
        )
    else:
        position = jnp.asarray(position)
    interpolated = jnp.stack(
        tuple(
            jnp.interp(query_time, ephemeris_time, position[:, axis])
            for axis in range(3)
        ),
        axis=-1,
    )
    cos_obliquity = 0.9174820003578725
    sin_obliquity = 0.3977772982704228
    ra = ra_degrees * _DEGREE_TO_RADIAN
    dec = dec_degrees * _DEGREE_TO_RADIAN
    object_direction = jnp.asarray(
        (
            jnp.cos(ra) * jnp.cos(dec),
            jnp.sin(ra) * jnp.cos(dec) * cos_obliquity
            + jnp.sin(dec) * sin_obliquity,
            -jnp.sin(ra) * jnp.cos(dec) * sin_obliquity
            + jnp.sin(dec) * cos_obliquity,
        )
    )
    north_2000 = jnp.asarray((0.0, sin_obliquity, cos_obliquity))
    radial = _normalize(
        -north_2000
        + object_direction * jnp.sum(north_2000 * object_direction)
    )
    tangential = jnp.cross(radial, object_direction)
    radial_projection = jnp.sum(interpolated * radial, axis=-1)
    tangential_projection = jnp.sum(interpolated * tangential, axis=-1)
    tau = pi_en * radial_projection + pi_ee * tangential_projection
    beta = pi_en * tangential_projection - pi_ee * radial_projection
    valid = (query_time >= ephemeris_time[0]) & (
        query_time <= ephemeris_time[-1]
    )
    return jnp.where(valid, tau, jnp.nan), jnp.where(valid, beta, jnp.nan)


def _keplerian_position(time, reference_time, period, eccentricity, periapsis):
    mean_anomaly = 2.0 * _PI / period * (time - reference_time) + periapsis
    eccentric_anomaly = _solve_kepler(mean_anomaly, eccentricity)
    return jnp.stack(
        (
            jnp.cos(eccentric_anomaly) - eccentricity,
            _safe_sqrt(1.0 - eccentricity * eccentricity)
            * jnp.sin(eccentric_anomaly),
        ),
        axis=-1,
    )


def xallarap_offsets(
    time,
    *,
    mode,
    xi_1,
    xi_2,
    reference_time,
    period=1.0,
    eccentricity=0.0,
    periapsis=0.0,
    w1=0.0,
    w2=0.0,
    w3=0.0,
    line_of_sight_ratio=0.0,
    semimajor_axis_ratio=1.0,
    inclination=0.0,
):
    """Return native-convention xallarap offsets in ``(tau, beta)``."""

    time = jnp.asarray(time)
    if mode in ("circular_elements", "orbital_elements"):
        active_eccentricity = (
            0.0 if mode == "circular_elements" else eccentricity
        )
        position = _keplerian_position(
            time,
            reference_time,
            period,
            active_eccentricity,
            periapsis,
        )
        position0 = _keplerian_position(
            reference_time,
            reference_time,
            period,
            active_eccentricity,
            periapsis,
        )
        small_step = period * 1.0e-7
        position_step = _keplerian_position(
            reference_time + small_step,
            reference_time,
            period,
            active_eccentricity,
            periapsis,
        )
        velocity0 = (position_step - position0) / small_step
        deviation = (
            position - position0 - (time - reference_time)[..., None] * velocity0
        )
        displacement0 = jnp.sin(inclination) * deviation[..., 0]
        displacement1 = deviation[..., 1]
        return (
            xi_1 * displacement0 + xi_2 * displacement1,
            xi_2 * displacement0 - xi_1 * displacement1,
        )
    xi_separation = _safe_sqrt(xi_1 * xi_1 + xi_2 * xi_2)
    xi_angle = jnp.arctan2(xi_2, xi_1)
    if mode == "circular_velocity":
        state = circular_orbital_motion(
            time, xi_separation, xi_angle, w1, w2, w3, reference_time
        )
    elif mode == "kepler_velocity":
        state = kepler_orbital_motion(
            time,
            xi_separation,
            xi_angle,
            w1,
            w2,
            w3,
            line_of_sight_ratio,
            semimajor_axis_ratio,
            reference_time,
        )
    elif mode in ("none", None):
        return jnp.zeros_like(time), jnp.zeros_like(time)
    else:
        raise ValueError(f"unsupported xallarap mode: {mode!r}")
    return (
        state.separation * jnp.cos(state.angle),
        state.separation * jnp.sin(state.angle),
    )


def binary_lens_trajectory(
    time,
    *,
    t0,
    timescale,
    impact_parameter,
    separation,
    angle,
    reference_time=None,
    lens_orbit="static",
    g1=0.0,
    g2=0.0,
    g3=0.0,
    line_of_sight_ratio=0.0,
    semimajor_axis_ratio=1.0,
    tau_offset=0.0,
    beta_offset=0.0,
):
    """Compose source motion and an optional circular/Kepler lens orbit.

    ``tau_offset`` and ``beta_offset`` are additive arrays.  Annual,
    terrestrial, satellite-parallax, and xallarap terms can therefore be
    summed without coupling the observational geometry to the inverse-ray
    kernel.
    """

    time = jnp.asarray(time)
    reference_time = t0 if reference_time is None else reference_time
    tau = (time - t0) / timescale + tau_offset
    beta = impact_parameter + beta_offset
    if lens_orbit == "static":
        orbit = OrbitalState(
            jnp.full_like(time, separation),
            jnp.full_like(time, angle),
            jnp.zeros_like(time),
            jnp.ones_like(time, dtype=jnp.bool_),
        )
    elif lens_orbit == "circular":
        orbit = circular_orbital_motion(
            time, separation, angle, g1, g2, g3, reference_time
        )
    elif lens_orbit == "kepler":
        orbit = kepler_orbital_motion(
            time,
            separation,
            angle,
            g1,
            g2,
            g3,
            line_of_sight_ratio,
            semimajor_axis_ratio,
            reference_time,
        )
    else:
        raise ValueError(f"unsupported lens orbit: {lens_orbit!r}")
    source_x = tau * jnp.cos(orbit.angle) - beta * jnp.sin(orbit.angle)
    source_y = tau * jnp.sin(orbit.angle) + beta * jnp.cos(orbit.angle)
    valid = (
        orbit.valid
        & jnp.isfinite(source_x)
        & jnp.isfinite(source_y)
        & jnp.isfinite(orbit.separation)
    )
    return TrajectoryGeometry(
        source_x,
        source_y,
        orbit.separation,
        orbit.angle,
        orbit.line_of_sight_separation,
        valid,
    )


def binary_source_trajectories(
    time,
    *,
    t0,
    timescale,
    impact_parameter,
    separation,
    angle,
    t0_2=None,
    impact_parameter_2=None,
    source_mass_ratio=None,
    source_orbit_coordinates="none",
    xallarap_mode="none",
    xi_1=0.0,
    xi_2=0.0,
    xallarap_parameters=None,
    reference_time=None,
    tau_offset=0.0,
    beta_offset=0.0,
    **lens_orbit_parameters,
):
    """Construct native-compatible binary-source trajectories.

    ``source_orbit_coordinates="xallarap"`` treats ``xi_1, xi_2`` as the
    primary source's position about the source center of mass.
    ``"trajectory_offset"`` instead derives that state from the two tangent
    trajectories ``(t0, u0)`` and ``(t0_2, u0_2)`` at the reference epoch.
    """

    reference_time = t0 if reference_time is None else reference_time
    xallarap_parameters = (
        {} if xallarap_parameters is None else dict(xallarap_parameters)
    )
    if xallarap_mode in ("none", None):
        if t0_2 is None or impact_parameter_2 is None:
            raise ValueError("a static binary source requires t0_2 and u0_2")
        first_t0, first_u0 = t0, impact_parameter
        second_t0, second_u0 = t0_2, impact_parameter_2
        first_xi = (0.0, 0.0)
        second_xi = (0.0, 0.0)
    else:
        if source_mass_ratio is None:
            raise ValueError("xallarap requires source_mass_ratio")
        if source_orbit_coordinates == "trajectory_offset":
            if t0_2 is None or impact_parameter_2 is None:
                raise ValueError(
                    "trajectory_offset requires t0_2 and impact_parameter_2"
                )
            inverse_total_mass = 1.0 / (1.0 + source_mass_ratio)
            relative_tau = (t0 - t0_2) / timescale
            relative_beta = impact_parameter_2 - impact_parameter
            center_t0 = (
                t0 + source_mass_ratio * t0_2
            ) * inverse_total_mass
            center_u0 = (
                impact_parameter
                + source_mass_ratio * impact_parameter_2
            ) * inverse_total_mass
            first_t0 = second_t0 = center_t0
            first_u0 = second_u0 = center_u0
            first_xi = (
                -source_mass_ratio * inverse_total_mass * relative_tau,
                -source_mass_ratio * inverse_total_mass * relative_beta,
            )
        elif source_orbit_coordinates in ("xallarap", "none"):
            first_t0 = second_t0 = t0
            first_u0 = second_u0 = impact_parameter
            first_xi = (xi_1, xi_2)
        else:
            raise ValueError(
                "source_orbit_coordinates must be 'xallarap' or "
                "'trajectory_offset'"
            )
        second_xi = (
            -first_xi[0] / source_mass_ratio,
            -first_xi[1] / source_mass_ratio,
        )

    def make_component(component_t0, component_u0, component_xi):
        xa_tau, xa_beta = xallarap_offsets(
            time,
            mode=xallarap_mode,
            xi_1=component_xi[0],
            xi_2=component_xi[1],
            reference_time=reference_time,
            **xallarap_parameters,
        )
        return binary_lens_trajectory(
            time,
            t0=component_t0,
            timescale=timescale,
            impact_parameter=component_u0,
            separation=separation,
            angle=angle,
            reference_time=reference_time,
            tau_offset=tau_offset + xa_tau,
            beta_offset=beta_offset + xa_beta,
            **lens_orbit_parameters,
        )

    return BinarySourceTrajectory(
        make_component(first_t0, first_u0, first_xi),
        make_component(second_t0, second_u0, second_xi),
    )


def binary_magnification_light_curve(
    time,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    trajectory_parameters,
    integration_parameters=None,
):
    """Evaluate one physical binary-lens light curve through the fused FFI."""

    from .trajectory import binary_magnification_trajectory

    geometry = binary_lens_trajectory(time, **trajectory_parameters)
    integration_parameters = (
        {} if integration_parameters is None else dict(integration_parameters)
    )
    magnification = binary_magnification_trajectory(
        geometry.source_x,
        geometry.source_y,
        geometry.separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        **integration_parameters,
    )
    return HigherOrderMagnificationResult(magnification, geometry)


def binary_source_magnification_light_curve(
    time,
    mass_ratio,
    source_radius_1,
    source_radius_2,
    flux_ratio,
    limb_c_1=0.0,
    limb_d_1=0.0,
    limb_c_2=0.0,
    limb_d_2=0.0,
    *,
    trajectory_parameters,
    integration_parameters=None,
):
    """Evaluate and flux-weight two native-compatible source trajectories."""

    from .trajectory import binary_magnification_trajectory

    geometry = binary_source_trajectories(time, **trajectory_parameters)
    integration_parameters = (
        {} if integration_parameters is None else dict(integration_parameters)
    )

    def evaluate(component, radius, limb_c, limb_d):
        result = binary_magnification_trajectory(
            component.source_x,
            component.source_y,
            component.separation,
            mass_ratio,
            radius,
            limb_c,
            limb_d,
            **integration_parameters,
        )
        return HigherOrderMagnificationResult(result, component)

    source1 = evaluate(geometry.source1, source_radius_1, limb_c_1, limb_d_1)
    source2 = evaluate(geometry.source2, source_radius_2, limb_c_2, limb_d_2)
    total = (
        source1.magnification.magnification
        + flux_ratio * source2.magnification.magnification
    ) / (1.0 + flux_ratio)
    return BinarySourceMagnificationResult(total, source1, source2, flux_ratio)


def _limited_ephemeris_indices(times, t_lim, reference_time):
    """Select contiguous interpolation windows before creating JAX arrays."""

    def absolute_time(value):
        return value + _RJD_ORIGIN if value < _RJD_ORIGIN else value

    lower, upper = (absolute_time(float(value)) for value in t_lim)
    windows = [(lower, upper)]
    if reference_time is not None:
        reference = absolute_time(float(reference_time))
        windows.append((reference, reference))
    selected = []
    for window_lower, window_upper in windows:
        if window_lower < times[0] or window_upper > times[-1]:
            raise ValueError(
                "Options.t_lim lies outside the available ephemeris table"
            )
        start = max(
            0,
            int(np.searchsorted(times, window_lower, side="right")) - 2,
        )
        stop = min(
            times.size,
            int(np.searchsorted(times, window_upper, side="left")) + 3,
        )
        selected.extend(range(start, stop))
    return np.unique(np.asarray(selected, dtype=np.int64))


def load_earth_ephemeris(path=None, *, t_lim=None, reference_time=None):
    """Load an optional interpolation-safe window of the Earth state table."""

    if path is None:
        # Traversable.joinpath accepted only one child argument on Python 3.9.
        resource = (
            resources.files("lcbinint_jax")
            .joinpath("data")
            .joinpath("earth_orbital_parallax_table.txt")
        )
        try:
            stream = resource.open(encoding="utf-8")
            description = "the bundled Earth ephemeris"
        except FileNotFoundError:
            # The source tree deliberately keeps data outside the Python
            # package. Retain that layout for editable/developer imports;
            # installed wheels use the resource above.
            source_path = Path(__file__).resolve().parents[2] / "data" / (
                "earth_orbital_parallax_table.txt"
            )
            stream = source_path.open(encoding="utf-8")
            description = str(source_path)
    else:
        source_path = Path(path)
        stream = source_path.open(encoding="utf-8")
        description = str(source_path)
    rows = []
    in_block = False
    with stream:
        for line in stream:
            stripped = line.strip()
            if not in_block:
                in_block = stripped.startswith("$$SOE")
                continue
            if stripped.startswith("$$EOE"):
                break
            fields = []
            for column, token in enumerate(stripped.split(",")):
                token = token.strip()
                if not token or column == 1:
                    continue
                try:
                    fields.append(float(token))
                except ValueError:
                    pass
            if len(fields) >= 7:
                rows.append(fields[:7])
    table = np.asarray(rows, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] < 2:
        raise ValueError(f"no usable Earth ephemeris rows in {description}")
    if t_lim is not None:
        indices = _limited_ephemeris_indices(
            table[:, 0], t_lim, reference_time
        )
        table = table[indices]
    return EarthEphemeris(
        jnp.asarray(table[:, 0]),
        jnp.asarray(table[:, 1:4]),
        jnp.asarray(table[:, 4:7]),
    )

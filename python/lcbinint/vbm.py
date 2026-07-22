"""Compatibility helpers for complete VBMicrolensing example models."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def binary_source_binary_lens(
    times,
    params: Mapping[str, float],
    *,
    sky,
    options=None,
    t_ref: float | None = None,
    mass_luminosity_exponent: float = 4.0,
    mass_radius_exponent: float = 0.9,
):
    """Evaluate the VBMicrolensing ``BinSourceBinLensLightCurve`` convention.

    This is the coupled binary-source, binary-lens model used by the published
    VBMicrolensing Python example.  ``g1/g2/g3`` describe the lens orbit while
    ``w1/w2/w3`` are the relative velocity components of the two sources.
    ``q_source`` is their flux ratio; the source mass ratio is inferred with
    the supplied mass--luminosity exponent, as in VBMicrolensing.
    """
    # Import after package initialization to avoid a circular module import.
    from . import LightCurve, Model, Options, binary_ray_shooting

    values = {key: float(value) for key, value in params.items()}
    required = (
        "s", "q", "u0", "alpha", "rho", "tE", "t0", "piEN", "piEE",
        "g1", "g2", "g3", "u0_2", "t0_2", "q_source", "w1", "w2", "w3",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f"missing binary-source parameters: {', '.join(missing)}")
    if values["tE"] <= 0.0 or values["rho"] <= 0.0:
        raise ValueError("tE and rho must be positive")
    if values["q_source"] <= 0.0:
        raise ValueError("q_source must be positive")
    if mass_luminosity_exponent <= 0.0:
        raise ValueError("mass_luminosity_exponent must be positive")

    epochs = np.asarray(times, dtype=float)
    if epochs.ndim != 1 or not np.all(np.isfinite(epochs)):
        raise ValueError("times must be a finite one-dimensional array")
    if options is None:
        options = Options(coordinates="vbm")
    reference_time = values["t0"] if t_ref is None else float(t_ref)

    # Recover the two annual-parallax displacements from lcbinint's own Earth
    # ephemeris.  The public trajectory uses the internally rotated frame, so
    # negate both axes before inverting the standard alpha rotation.
    annual = LightCurve(
        model=Model(parallax=True, sky=sky, t_ref=reference_time),
        options=Options(coordinates="vbm"),
    ).source_trajectory(epochs, values)
    y1 = -np.asarray(annual.x)
    y2 = -np.asarray(annual.y)
    cos_alpha = math.cos(values["alpha"])
    sin_alpha = math.sin(values["alpha"])
    tau = -y1 * cos_alpha - y2 * sin_alpha
    beta = y1 * sin_alpha - y2 * cos_alpha
    inverse_tE = 1.0 / values["tE"]
    parallax_tau = tau - (epochs - values["t0"]) * inverse_tE
    parallax_beta = beta - values["u0"]

    # Circular lens orbit, algebraically identical to BinaryLightCurveOrbital.
    lens_w1, lens_w2, lens_w3 = values["g1"], values["g2"], values["g3"]
    lens_w13 = math.hypot(lens_w1, lens_w3)
    lens_w123 = math.sqrt(lens_w13 * lens_w13 + lens_w2 * lens_w2)
    if lens_w13 > 1.0e-8:
        lens_w3_effective = lens_w3 if lens_w3 > 1.0e-8 else 1.0e-8
        lens_omega = lens_w3_effective * lens_w123 / lens_w13
        cos_inclination = np.clip(
            lens_w2 * lens_w3_effective / (lens_w13 * lens_w123), -1.0, 1.0
        )
        lens_inclination = math.acos(float(cos_inclination))
        lens_phase0 = math.atan2(
            -lens_w1 * lens_w123, lens_w3_effective * lens_w13
        )
    else:
        lens_omega = lens_w2
        lens_inclination = 0.0
        lens_phase0 = 0.0
    cos_phase0 = math.cos(lens_phase0)
    sin_phase0 = math.sin(lens_phase0)
    cos_inclination = math.cos(lens_inclination)
    lens_den0 = math.hypot(cos_phase0, cos_inclination * sin_phase0)
    true_lens_separation = values["s"] / lens_den0
    cos_omega = (
        cos_phase0 * cos_alpha
        + cos_inclination * sin_alpha * sin_phase0
    ) / lens_den0
    sin_omega = (
        cos_phase0 * sin_alpha
        - cos_inclination * cos_alpha * sin_phase0
    ) / lens_den0

    # Reconstruct the circular binary-source orbit from its projected position
    # and relative velocity at t0, including VBMicrolensing's tiny zero guard.
    source_velocity = np.asarray(
        [values["w1"], values["w2"], values["w3"]], dtype=float
    ) + 1.01e-15
    source_t01 = values["t0"]
    source_t02 = values["t0_2"] + source_velocity[0] * (
        values["t0_2"] - values["t0"]
    ) / inverse_tE
    source_u1 = values["u0"]
    source_u2 = values["u0_2"] + source_velocity[1] * (
        values["t0"] - values["t0_2"]
    )
    relative_position = np.asarray(
        [
            (source_t01 - source_t02) * inverse_tE,
            source_u2 - source_u1,
            0.0,
        ]
    )
    if abs(source_velocity[2]) <= 1.0e-15:
        raise ValueError("w3 must be non-zero for the VBM circular source orbit")
    relative_position[2] = -float(
        np.dot(relative_position[:2], source_velocity[:2])
    ) / source_velocity[2]
    source_separation_3d = float(np.linalg.norm(relative_position))
    if source_separation_3d <= 0.0:
        raise ValueError("the two source trajectories must be distinct")
    source_omega = float(np.linalg.norm(source_velocity)) / source_separation_3d

    angular_momentum = np.cross(relative_position, source_velocity)
    node_norm = math.hypot(angular_momentum[0], angular_momentum[1])
    momentum_norm = float(np.linalg.norm(angular_momentum))
    if node_norm > 0.0 and momentum_norm > 0.0:
        node = np.asarray(
            [-angular_momentum[1] / node_norm,
             angular_momentum[0] / node_norm, 0.0]
        )
        angular_momentum /= momentum_norm
    else:
        node = np.asarray([1.0, 0.0, 0.0])
        angular_momentum = np.asarray([0.0, -1.0, 0.0])
    orthogonal = np.asarray(
        [
            -angular_momentum[2] * node[1],
            angular_momentum[2] * node[0],
            angular_momentum[0] * node[1] - angular_momentum[1] * node[0],
        ]
    )
    phase_argument = float(
        np.dot(relative_position[:2], node[:2]) / source_separation_3d
    )
    source_phase0 = math.acos(float(np.clip(
        phase_argument, -0.99999999999999, 0.99999999999999
    )))
    if relative_position[2] < 0.0:
        source_phase0 = -source_phase0

    flux_ratio = values["q_source"]
    source_mass_ratio = flux_ratio ** (1.0 / mass_luminosity_exponent)
    barycenter_t0 = (
        source_t01 + source_t02 * source_mass_ratio
    ) / (1.0 + source_mass_ratio)
    barycenter_u0 = (
        source_u1 + source_u2 * source_mass_ratio
    ) / (1.0 + source_mass_ratio)
    barycenter_t0 = (barycenter_t0 - values["t0"]) * inverse_tE
    barycenter_velocity_t = (
        source_velocity[0] * source_mass_ratio / (1.0 + source_mass_ratio)
        + inverse_tE
    )
    barycenter_velocity_u = (
        source_velocity[1] * source_mass_ratio / (1.0 + source_mass_ratio)
    )
    source_angle = math.atan2(barycenter_velocity_u, barycenter_velocity_t)
    source2_radius = source_separation_3d / (1.0 + source_mass_ratio)
    source1_radius = source2_radius * source_mass_ratio
    secondary_rho = values["rho"] * values["q"] ** (
        mass_radius_exponent / mass_luminosity_exponent
    )

    magnifications = np.empty(len(epochs), dtype=float)
    for index, time in enumerate(epochs):
        elapsed = time - values["t0"]
        lens_phase = elapsed * lens_omega + lens_phase0
        cos_phase = math.cos(lens_phase)
        sin_phase = math.sin(lens_phase)
        lens_den = math.hypot(cos_phase, cos_inclination * sin_phase)
        separation = true_lens_separation * lens_den

        barycenter_tau = (
            elapsed * barycenter_velocity_t - barycenter_t0
            + parallax_tau[index] * math.cos(source_angle)
            - parallax_beta[index] * math.sin(source_angle)
        )
        barycenter_beta = (
            barycenter_u0 + barycenter_velocity_u * elapsed
            + parallax_tau[index] * math.sin(source_angle)
            + parallax_beta[index] * math.cos(source_angle)
        )
        source_phase = elapsed * source_omega + source_phase0
        relative_tau = (
            node[0] * math.cos(source_phase)
            + orthogonal[0] * math.sin(source_phase)
        )
        relative_beta = (
            node[1] * math.cos(source_phase)
            + orthogonal[1] * math.sin(source_phase)
        )

        source_magnifications = []
        for source_tau, source_beta, source_rho in (
            (
                barycenter_tau - relative_tau * source1_radius,
                barycenter_beta - relative_beta * source1_radius,
                values["rho"],
            ),
            (
                barycenter_tau + relative_tau * source2_radius,
                barycenter_beta + relative_beta * source2_radius,
                secondary_rho,
            ),
        ):
            source_y1 = (
                cos_phase * (source_beta * sin_omega - source_tau * cos_omega)
                + cos_inclination * sin_phase
                * (source_beta * cos_omega + source_tau * sin_omega)
            ) / lens_den
            source_y2 = (
                -cos_phase * (source_beta * cos_omega + source_tau * sin_omega)
                - cos_inclination * sin_phase
                * (source_tau * cos_omega - source_beta * sin_omega)
            ) / lens_den
            source_magnifications.append(binary_ray_shooting(
                source_y1,
                source_y2,
                s=separation,
                q=values["q"],
                rho=source_rho,
                options=options,
            ))
        magnifications[index] = (
            source_magnifications[0] + flux_ratio * source_magnifications[1]
        ) / (1.0 + flux_ratio)
    return magnifications

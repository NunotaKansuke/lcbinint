"""Native-calibrated binary inverse-ray grid and resolution selection."""

import jax
import jax.numpy as jnp

from ._config import as_float64, require_x64
from .types import BinaryResolutionSelection


_FEATURE_MEAN = (
    1.1820756488388118,
    -2.9036106609012546,
    -2.6986179919546345,
    0.03688102633633496,
    0.8972087621766296,
    1.1341442606439753,
    0.24869438061416335,
)
_FEATURE_STD = (
    1.0111131697060847,
    1.1601305065348657,
    1.8500204276846226,
    0.7895410239966703,
    0.7222204031861401,
    1.42955624048966,
    0.2499965906927929,
)
_COEFFICIENTS = (
    5.139848840914074,
    -0.026354983398495537,
    -0.008665567347890256,
    0.028914523534964386,
    0.09884746535594117,
    0.0757379504124179,
    0.03068462762322574,
    -0.15822137689559143,
)
_BUCKETS = (16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400)


def _upper_bucket(predicted):
    buckets = jnp.asarray(_BUCKETS, dtype=jnp.int32)
    eligible = buckets >= predicted
    index = jnp.argmax(eligible)
    return jnp.where(jnp.any(eligible), buckets[index], buckets[-1])


@jax.jit
def select_binary_resolution(
    mass_ratio,
    source_radius,
    caustic_distance,
    point_source_magnification,
    limb_darkening_c=0.0,
    requested_relative_tolerance=1.0e-3,
    maximum_bins=400,
) -> BinaryResolutionSelection:
    """Reproduce the native binary grid/``nbin`` calibration.

    The inputs describe one source position. The returned integer bucket and
    method mask are stopped-gradient routing data; the selected magnification
    kernel remains responsible for the physical derivatives.
    """

    require_x64()
    (
        mass_ratio,
        source_radius,
        caustic_distance,
        point_source_magnification,
        limb_darkening_c,
        requested_relative_tolerance,
    ) = (
        as_float64(value)
        for value in (
            mass_ratio,
            source_radius,
            caustic_distance,
            point_source_magnification,
            limb_darkening_c,
            requested_relative_tolerance,
        )
    )
    maximum_bins = jnp.maximum(jnp.asarray(maximum_bins, dtype=jnp.int32), 1)
    feature_mean = jnp.asarray(_FEATURE_MEAN, dtype=jnp.float64)
    feature_std = jnp.asarray(_FEATURE_STD, dtype=jnp.float64)
    coefficients = jnp.asarray(_COEFFICIENTS, dtype=jnp.float64)
    point_magnification = jnp.abs(point_source_magnification)
    distance_ratio = jnp.where(
        source_radius > 0.0,
        caustic_distance / source_radius,
        jnp.inf,
    )
    finite_geometry = jnp.isfinite(point_magnification) & jnp.isfinite(
        distance_ratio
    )
    prefer_polar = (point_magnification >= 300.0) | (
        (point_magnification >= 100.0) & (distance_ratio < 0.3)
    )

    polar_tolerance_scale = jnp.where(
        (requested_relative_tolerance > 0.0)
        & (requested_relative_tolerance < 1.0e-3),
        jnp.sqrt(1.0e-3 / requested_relative_tolerance),
        1.0,
    )
    polar_bins = _upper_bucket(64.0 * polar_tolerance_scale)

    absolute_mass_ratio = jnp.abs(mass_ratio)
    q_small = jnp.where(
        absolute_mass_ratio < 1.0,
        absolute_mass_ratio,
        jnp.where(absolute_mass_ratio > 0.0, 1.0 / absolute_mass_ratio, 1.0e-12),
    )
    q_small = jnp.maximum(q_small, 1.0e-12)
    companion_risk_ratio = 4.0 * source_radius / q_small
    features = jnp.stack(
        (
            jnp.log10(jnp.maximum(point_magnification, 1.0)),
            jnp.log10(jnp.maximum(source_radius, 1.0e-12)),
            jnp.log10(q_small),
            jnp.log10(jnp.maximum(distance_ratio, 1.0e-3)),
            jnp.maximum(0.0, 2.0 - jnp.minimum(distance_ratio, 2.0)),
            jnp.maximum(0.0, jnp.log10(jnp.maximum(companion_risk_ratio, 1.0))),
            limb_darkening_c,
        )
    )
    log2_bins = coefficients[0] + jnp.dot(
        coefficients[1:],
        (features - feature_mean) / feature_std,
    )
    predicted_cartesian = 1.10 * jnp.exp2(log2_bins)
    tighter = (requested_relative_tolerance > 0.0) & (
        requested_relative_tolerance < 1.0e-3
    )
    tolerance_ratio = 1.0e-3 / jnp.maximum(requested_relative_tolerance, 1.0e-300)
    first_order_risk = (distance_ratio < 2.0) | (companion_risk_ratio > 50.0)
    tolerance_scale = jnp.where(
        first_order_risk,
        tolerance_ratio,
        jnp.sqrt(tolerance_ratio),
    )
    predicted_cartesian = jnp.where(
        tighter,
        predicted_cartesian * tolerance_scale,
        predicted_cartesian,
    )
    cartesian_bins = _upper_bucket(predicted_cartesian)
    cartesian_bins = jnp.where(
        (distance_ratio > 0.9) & (distance_ratio < 1.1),
        jnp.maximum(cartesian_bins, 100),
        cartesian_bins,
    )
    cartesian_bins = jnp.where(
        companion_risk_ratio > 50.0,
        jnp.maximum(cartesian_bins, 80),
        cartesian_bins,
    )

    selected_bins = jnp.where(prefer_polar, polar_bins, cartesian_bins)
    selected_bins = jnp.where(finite_geometry, selected_bins, 100)
    prefer_polar = jnp.where(finite_geometry, prefer_polar, True)
    return BinaryResolutionSelection(
        source_bins=jax.lax.stop_gradient(jnp.minimum(selected_bins, maximum_bins)),
        prefer_polar=jax.lax.stop_gradient(prefer_polar),
    )

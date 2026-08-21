"""Native-calibrated binary inverse-ray grid and resolution selection.

The selector deliberately mirrors the native C++ calibration.  The native
rule is continuous in the requested tolerances; JAX only rounds its result up
to a fixed execution bucket because a jitted dispatcher needs static branch
shapes.  Geometry/profile features do not belong in this first resolution
decision: the native calibration already accounts for the two relevant
regimes (Cartesian and polar) and the subsequent certificate decides whether
the selected grid is actually sufficient.
"""

import jax
import jax.numpy as jnp

from ._config import as_float64, require_x64
from .types import BinaryResolutionSelection


# Keep these values synchronized with the native selector in
# finite_source_magnifier.cpp.  The relative and absolute branches are
# alternatives: the smaller required grid is sufficient for the combined
# error budget.  JAX does not duplicate the C++ bucket ladder here; it rounds
# the continuous result up to the same ladder below.
_BASELINE_TOLERANCE = 1.0e-3
_MINIMUM_RELATIVE_TOLERANCE = 1.0e-4
_MINIMUM_ABSOLUTE_TOLERANCE = 1.0e-4
_MAXIMUM_TOLERANCE = 1.0e-2
_DEFAULT_ABSOLUTE_TOLERANCE = 1.0e-4
_DEFAULT_RELATIVE_TOLERANCE = 1.0e-3
# The C++ coefficients were calibrated for the native inverse-ray kernel.
# JAX uses the same continuous law but a different FFI kernel and a static
# bucket ladder.  A single backend-wide margin covers that kernel-to-kernel
# difference without reintroducing geometry/profile-specific feature floors.
_JAX_EXECUTION_SAFETY_FACTOR = 1.10

_CARTESIAN_RELATIVE_C = 49.5929807101336
_CARTESIAN_RELATIVE_BETA = 0.47670215379590497
_POLAR_RELATIVE_C = 105.29723705815378
_POLAR_RELATIVE_BETA = 0.5952070961585817
_CARTESIAN_ABSOLUTE_C = 138.06382198454384
_CARTESIAN_ABSOLUTE_BETA = 0.4265493297299796
_CARTESIAN_ABSOLUTE_GAMMA = 0.34119845152344075
_POLAR_ABSOLUTE_C = 396.47500160748996
_POLAR_ABSOLUTE_BETA = 0.5337641762207631
_POLAR_ABSOLUTE_GAMMA = 0.2458039343900396

_BUCKETS = (16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400)


def _upper_bucket(predicted):
    buckets = jnp.asarray(_BUCKETS, dtype=jnp.int32)
    # Keep the helper scalar-compatible for the selector and batch-compatible
    # for diagnostics/benchmarks.  The final axis is the fixed bucket ladder;
    # taking its minimum avoids a data-dependent Python loop and preserves the
    # shape of ``predicted``.
    eligible = buckets >= jnp.expand_dims(predicted, axis=-1)
    return jnp.min(jnp.where(eligible, buckets, buckets[-1]), axis=-1)


@jax.jit
def select_binary_resolution(
    mass_ratio,
    source_radius,
    caustic_distance,
    point_source_magnification,
    limb_darkening_c=0.0,
    requested_relative_tolerance=1.0e-3,
    maximum_bins=400,
    requested_absolute_tolerance=0.0,
) -> BinaryResolutionSelection:
    """Reproduce the native binary grid/``nbin`` calibration.

    The inputs describe one source position.  Native C++ returns the ceiling
    of a continuous power law.  The JAX path rounds that ceiling upward to a
    fixed bucket, then applies ``maximum_bins`` as a hard cap.  The returned
    integer bucket and method mask are stopped-gradient routing data; the
    selected magnification kernel remains responsible for the physical
    derivatives.

    ``requested_absolute_tolerance`` is intentionally separate from the
    relative tolerance.  Combining the two into a synthetic relative
    tolerance changes the native calibration, especially at high
    magnification, and was the main source of the old feature/bucket drift.
    """

    require_x64()
    (
        mass_ratio,
        source_radius,
        caustic_distance,
        point_source_magnification,
        limb_darkening_c,
        requested_relative_tolerance,
        requested_absolute_tolerance,
    ) = (
        as_float64(value)
        for value in (
            mass_ratio,
            source_radius,
            caustic_distance,
            point_source_magnification,
            limb_darkening_c,
            requested_relative_tolerance,
            requested_absolute_tolerance,
        )
    )
    maximum_bins = jnp.maximum(jnp.asarray(maximum_bins, dtype=jnp.int32), 1)
    point_magnification = jnp.abs(point_source_magnification)
    finite_magnification = jnp.isfinite(point_magnification)
    distance_ratio = jnp.where(
        source_radius > 0.0,
        caustic_distance / source_radius,
        jnp.inf,
    )
    # Resolution follows the native power law below.  Route selection remains
    # a JAX-kernel concern: unlike native C++, the JAX polar FFI has a known
    # grazing angular bias outside the caustic band.  Keep the stable polar
    # high-magnification branch and the explicitly near-caustic branch, while
    # leaving the rest to Cartesian certification.
    prefer_polar = (point_magnification >= 300.0) | (
        (point_magnification >= 100.0) & (distance_ratio < 0.3)
    )

    absolute_tolerance = jnp.where(
        jnp.isfinite(requested_absolute_tolerance),
        jnp.maximum(requested_absolute_tolerance, 0.0),
        0.0,
    )
    relative_tolerance = jnp.where(
        jnp.isfinite(requested_relative_tolerance),
        jnp.maximum(requested_relative_tolerance, 0.0),
        0.0,
    )
    use_defaults = (absolute_tolerance <= 0.0) & (relative_tolerance <= 0.0)
    absolute_tolerance = jnp.where(
        use_defaults, _DEFAULT_ABSOLUTE_TOLERANCE, absolute_tolerance
    )
    relative_tolerance = jnp.where(
        use_defaults, _DEFAULT_RELATIVE_TOLERANCE, relative_tolerance
    )

    relative_supported = (
        (relative_tolerance >= _MINIMUM_RELATIVE_TOLERANCE)
        & (relative_tolerance <= _MAXIMUM_TOLERANCE)
    )
    absolute_supported = (
        (absolute_tolerance > _MINIMUM_ABSOLUTE_TOLERANCE)
        & (absolute_tolerance <= _MAXIMUM_TOLERANCE)
    )
    relative_c = jnp.where(prefer_polar, _POLAR_RELATIVE_C, _CARTESIAN_RELATIVE_C)
    relative_beta = jnp.where(
        prefer_polar, _POLAR_RELATIVE_BETA, _CARTESIAN_RELATIVE_BETA
    )
    absolute_c = jnp.where(prefer_polar, _POLAR_ABSOLUTE_C, _CARTESIAN_ABSOLUTE_C)
    absolute_beta = jnp.where(
        prefer_polar, _POLAR_ABSOLUTE_BETA, _CARTESIAN_ABSOLUTE_BETA
    )
    absolute_gamma = jnp.where(
        prefer_polar, _POLAR_ABSOLUTE_GAMMA, _CARTESIAN_ABSOLUTE_GAMMA
    )
    relative_bins = jnp.where(
        relative_supported,
        relative_c
        * jnp.power(relative_tolerance / _BASELINE_TOLERANCE, -relative_beta),
        jnp.inf,
    )
    absolute_bins = jnp.where(
        absolute_supported,
        absolute_c
        * jnp.power(absolute_tolerance / _BASELINE_TOLERANCE, -absolute_beta)
        * jnp.power(jnp.maximum(point_magnification, 1.0), absolute_gamma),
        jnp.inf,
    )
    required_bins = jnp.minimum(relative_bins, absolute_bins)
    continuous_bins = jnp.where(
        (~jnp.isfinite(required_bins)) | (required_bins >= maximum_bins),
        maximum_bins,
        jnp.maximum(
            1.0,
            jnp.ceil(required_bins * _JAX_EXECUTION_SAFETY_FACTOR),
        ),
    )
    selected_bins = _upper_bucket(continuous_bins)
    selected_bins = jnp.minimum(selected_bins, maximum_bins)
    selected_bins = jnp.where(finite_magnification, selected_bins, 100)
    prefer_polar = jnp.where(finite_magnification, prefer_polar, True)
    return BinaryResolutionSelection(
        source_bins=jax.lax.stop_gradient(jnp.minimum(selected_bins, maximum_bins)),
        prefer_polar=jax.lax.stop_gradient(prefer_polar),
    )

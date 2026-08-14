"""Opt-in native-resolution-calibrated binary magnification routing."""

import jax
import jax.numpy as jnp

from ._config import as_float64, require_x64
from .api import binary_magnification_auto
from .cpp_backend import (
    binary_routing_diagnostics_batch_ffi,
    cpp_binary_routing_diagnostics_batch_ffi_available,
    cpp_cartesian_epoch_ffi_available,
    cpp_polar_epoch_ffi_available,
)
from .multipole import binary_point_source_magnification
from .resolution import select_binary_resolution
from .trajectory import _CALIBRATED_EXECUTION_BUCKETS
from .types import CalibratedMagnificationResult

# Capacity follows the observed quadratic tile-count scaling.  It is a
# conservative initial floor, not a proof that every geometry fits; overflow
# remains fail-closed in the underlying dispatcher.
#
# The bounded frontier admits every tile whose image can reach the source disk
# rather than only those a sample lands in, which costs 1.4--1.8x more tiles.
# The 100--160 buckets are the ones that tipped over: on the resonant holdout
# (s=0.9, q=1e-3, rho=3e-3) they need 32768, 32768 and 65536 where the sampled
# frontier fitted in half that, and without the bump the dispatcher falls back
# to the polar route it is meant to avoid there.
# The Cartesian capacities come from the trajectory ladder rather than being
# restated here; see the note on _CALIBRATED_EXECUTION_BUCKETS for why a
# capacity of resolution^2 silently capped the reachable magnification near 47.
_EXECUTION_BUCKETS = _CALIBRATED_EXECUTION_BUCKETS


def _polar_angular_bins(resolution):
    # The native selector's 200--400 radial buckets still need at least 65536
    # angles for small-q, grazing high-magnification cusps.  With 32768 angles
    # the radial coarse/fine pair can agree while sharing a 5e-2 angular bias.
    target = max(65536, 64 * resolution)
    return 1 << (target - 1).bit_length()


def binary_magnification_calibrated(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    absolute_tolerance=1.0e-4,
    relative_tolerance=1.0e-4,
    caustic_bins=1400,
    maximum_source_bins=400,
    moment_mode="two_coefficient",
    root_backend="auto",
):
    """Use native-calibrated ``nbin`` and polar preselection on JAX CPU.

    This is deliberately opt-in while the full method selector is calibrated.
    It reproduces the native resolution regression and floors, rounds to the
    same 14 buckets, and pairs every bucket with a quadratic Cartesian support
    capacity.  Binary caustic distance and all routing masks are
    stopped-gradient; the selected physical kernel retains its analytic AD.

    Native point-safety and caustic-branch scans route safe point sources and
    grazing sources before the image-plane kernels.  Polar routing uses the
    calibrated radial/angular capacity floors established by the stress
    holdout.  Unsupported or non-converged support fails closed or falls back
    to an independently checked grid path rather than returning a partial
    value. Method codes are 0 hexadecapole, 1 Cartesian, 2 polar, 3
    source-direct, and 4 point-source.
    """

    require_x64()
    if maximum_source_bins not in {item[0] for item in _EXECUTION_BUCKETS}:
        raise ValueError(
            "maximum_source_bins must be one of the calibrated resolution buckets"
        )
    if jax.default_backend() != "cpu":
        raise RuntimeError("calibrated binary routing currently requires JAX CPU")
    if not (
        cpp_binary_routing_diagnostics_batch_ffi_available()
        and cpp_cartesian_epoch_ffi_available()
        and cpp_polar_epoch_ffi_available()
    ):
        raise RuntimeError("calibrated binary routing requires the CPU FFI backends")

    source_x, source_y, separation, mass_ratio, source_radius, limb_c, limb_d = (
        as_float64(value)
        for value in (
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        )
    )
    point_solution = binary_point_source_magnification(
        source_x,
        source_y,
        separation,
        mass_ratio,
        root_backend=root_backend,
    )
    point_magnification = jax.lax.stop_gradient(point_solution.magnification)
    routing = binary_routing_diagnostics_batch_ffi(
        jnp.reshape(source_x, (1,)),
        jnp.reshape(source_y, (1,)),
        jnp.reshape(point_magnification, (1,)),
        separation,
        mass_ratio,
        source_radius,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        caustic_bins=caustic_bins,
    )
    caustic_distance = routing.caustic_distance[0]
    point_safe = routing.point_safe[0]
    point_route = point_safe & ~point_solution.root_failure
    explicit_tolerance = (absolute_tolerance > 0.0) | (relative_tolerance > 0.0)
    requested_relative_tolerance = jnp.where(
        explicit_tolerance,
        relative_tolerance
        + absolute_tolerance / jnp.maximum(jnp.abs(point_magnification), 1.0),
        1.0e-3,
    )
    selection = select_binary_resolution(
        mass_ratio,
        source_radius,
        caustic_distance,
        point_magnification,
        limb_c,
        requested_relative_tolerance,
        maximum_source_bins,
    )
    selected_bins = jnp.where(source_radius > 0.0, selection.source_bins, 16)
    bucket_resolutions = jnp.asarray(
        tuple(item[0] for item in _EXECUTION_BUCKETS), dtype=jnp.int32
    )
    bucket_index = jnp.argmax(bucket_resolutions >= selected_bins)
    prefer_polar = jax.lax.stop_gradient((source_radius > 0.0) & selection.prefer_polar)
    # Native skips local multipoles inside the configured caustic-distance
    # threshold.  Retain the JAX self-consistency guard outside that region.
    multipole_safety = jnp.where(
        (source_radius <= 0.0) | (caustic_distance >= 3.0 * source_radius),
        4.0,
        jnp.inf,
    )


    branches = []
    for resolution, tile_capacity in _EXECUTION_BUCKETS:
        angular_bins = _polar_angular_bins(resolution)
        # Near a planetary/resonant caustic, the image fingers can extend far
        # beyond the source-sized radial window.  Four source diameters was the
        # smallest static floor that covered the high-magnification holdout;
        # keep the capacity in the same power-of-two bucket family.
        radial_capacity = 1 << (max(256, 8 * resolution) - 1).bit_length()

        def evaluate(
            _,
            resolution=resolution,
            tile_capacity=tile_capacity,
            angular_bins=angular_bins,
            radial_capacity=radial_capacity,
        ):
            return binary_magnification_auto(
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                multipole_safety_factor=multipole_safety,
                resolution=resolution,
                tile_size=16,
                tile_capacity=tile_capacity,
                limb_samples=32,
                polar_resolution=resolution,
                polar_angular_bins=angular_bins,
                polar_radial_capacity=radial_capacity,
                polar_limb_samples=64,
                polar_angular_chunk_size=256,
                polar_boundary_subdivision=4,
                polar_magnification_threshold=jnp.inf,
                polar_max_source_radius=jnp.inf,
                polar_min_mass_ratio=0.0,
                polar_require_topology_stable=False,
                polar_force=prefer_polar,
                polar_fallback_on_overflow=True,
                moment_mode=moment_mode,
                cartesian_backend="ffi",
                root_backend=root_backend,
            )

        branches.append(evaluate)

    capacities = jnp.asarray(
        tuple(item[1] for item in _EXECUTION_BUCKETS), dtype=jnp.int32
    )

    def result(
        *,
        magnification,
        method,
        estimated_error,
        support_valid,
        used_multipole,
        used_polar,
        used_source_plane,
        used_expanded_cartesian,
        comparison_resolution,
        executed_resolution,
        tile_capacity,
        value_error,
        value_budget,
        value_converged,
    ):
        return CalibratedMagnificationResult(
            magnification=magnification,
            method=method,
            estimated_error=estimated_error,
            support_valid=support_valid,
            used_multipole=used_multipole,
            used_polar=used_polar,
            used_source_plane=used_source_plane,
            used_expanded_cartesian=used_expanded_cartesian,
            selected_source_bins=selection.source_bins,
            comparison_resolution=comparison_resolution,
            executed_resolution=executed_resolution,
            tile_capacity=tile_capacity,
            caustic_distance=caustic_distance,
            prefer_polar=prefer_polar,
            point_safe=point_safe,
            chord_band=routing.chord_band[0],
            tangent_band=routing.tangent_band[0],
            grazing_ring_band=routing.grazing_ring_band[0],
            value_error=value_error,
            value_budget=value_budget,
            value_converged=jax.lax.stop_gradient(value_converged),
        )

    def grid_path(_):
        hybrid = jax.lax.switch(bucket_index, tuple(branches), operand=None)
        coarse_index = jnp.maximum(bucket_index - 1, 0)
        image_plane_method = (hybrid.method == 1) | (hybrid.method == 2)
        coarse = jax.lax.cond(
            image_plane_method,
            lambda _: jax.lax.switch(coarse_index, tuple(branches), operand=None),
            lambda _: hybrid,
            None,
        )
        upper_index = jnp.minimum(bucket_index + 1, len(_EXECUTION_BUCKETS) - 1)
        # At the bottom rung `coarse_index` clamps to `bucket_index`, so the
        # "coarser" evaluation is the selected one, the two never disagree, and
        # `comparison_index == bucket_index` forced `image_plane_converged`
        # False however accurate the value was -- a fail-closed NaN with a
        # measured error of exactly zero.  Reach upward instead, the way the
        # trajectory ladder's `use_upper = (bucket_index == 0) | ~lower_support`
        # already does: differencing against the finer neighbour estimates the
        # discretisation error just as well as differencing against a coarser
        # one would.
        use_upper_comparison = (
            image_plane_method
            & (bucket_index < len(_EXECUTION_BUCKETS) - 1)
            & (
                (bucket_index == 0)
                | ~coarse.support_valid
                | (coarse.method != hybrid.method)
            )
        )
        comparison = jax.lax.cond(
            use_upper_comparison,
            lambda _: jax.lax.switch(upper_index, tuple(branches), operand=None),
            lambda _: coarse,
            None,
        )
        comparison_index = jnp.where(use_upper_comparison, upper_index, coarse_index)
        value_error = jnp.where(
            image_plane_method,
            jnp.abs(hybrid.magnification - comparison.magnification),
            hybrid.estimated_error,
        )
        value_budget = absolute_tolerance + relative_tolerance * jnp.maximum(
            jnp.abs(hybrid.magnification), 1.0
        )
        image_plane_converged = (
            (comparison_index != bucket_index)
            & comparison.support_valid
            & hybrid.support_valid
            & jnp.isfinite(value_error)
            & (value_error <= value_budget)
        )
        intrinsic_converged = (
            hybrid.support_valid
            & jnp.isfinite(value_error)
            & (value_error <= value_budget)
        )
        return result(
            magnification=hybrid.magnification,
            method=hybrid.method,
            estimated_error=hybrid.estimated_error,
            support_valid=hybrid.support_valid,
            used_multipole=hybrid.used_multipole,
            used_polar=hybrid.used_polar,
            used_source_plane=hybrid.used_source_plane,
            used_expanded_cartesian=hybrid.used_expanded_cartesian,
            comparison_resolution=bucket_resolutions[comparison_index],
            executed_resolution=bucket_resolutions[bucket_index],
            tile_capacity=capacities[bucket_index],
            value_error=value_error,
            value_budget=value_budget,
            value_converged=jnp.where(
                image_plane_method,
                image_plane_converged,
                intrinsic_converged,
            ),
        )

    def nonpoint_path(_):
        # The calibrated dispatcher is image-plane only.  Grazing/chord bands
        # proceed directly to the Cartesian/polar ladder; there is no hidden
        # source-plane candidate.
        return grid_path(None)


    def point_path(_):
        value_budget = absolute_tolerance + relative_tolerance * jnp.maximum(
            jnp.abs(point_solution.magnification), 1.0
        )
        return result(
            magnification=point_solution.magnification,
            method=jnp.asarray(4, dtype=jnp.int32),
            estimated_error=routing.point_error_estimate[0],
            support_valid=jnp.asarray(True),
            used_multipole=jnp.asarray(False),
            used_polar=jnp.asarray(False),
            used_source_plane=jnp.asarray(False),
            used_expanded_cartesian=jnp.asarray(False),
            comparison_resolution=jnp.asarray(0, dtype=jnp.int32),
            executed_resolution=jnp.asarray(0, dtype=jnp.int32),
            tile_capacity=jnp.asarray(0, dtype=jnp.int32),
            value_error=routing.point_error_estimate[0],
            value_budget=value_budget,
            value_converged=point_safe,
        )

    return jax.lax.cond(
        point_route,
        point_path,
        nonpoint_path,
        None,
    )

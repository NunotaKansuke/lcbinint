"""Differentiable trajectory dispatcher with scalar conditional epochs."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import require_x64
from .api import _multipole_dispatch_masks, binary_magnification_auto
from .cpp_backend import (
    binary_hexadecapole_batch_ffi,
    binary_inverse_ray_polar_ffi,
    binary_magnification_trajectory_ffi,
    binary_routing_diagnostics_batch_ffi,
    binary_inverse_ray_cartesian_batch_ffi,
    cpp_binary_routing_diagnostics_batch_ffi_available,
    cpp_cartesian_batch_ffi_available,
    cpp_trajectory_ffi_available,
)
from .multipole import binary_hexadecapole
from .resolution import select_binary_resolution
from .source_plane import binary_source_plane_quadrature
from .types import HybridMagnificationResult, TrajectoryMagnificationResult

_CALIBRATED_EXECUTION_BUCKETS = (
    (16, 256),
    (24, 1024),
    (32, 1024),
    (40, 2048),
    (50, 4096),
    (64, 4096),
    (80, 8192),
    (100, 16384),
    (128, 16384),
    (160, 32768),
    (200, 65536),
    (256, 65536),
    (320, 131072),
    (400, 262144),
)


@partial(
    jax.jit,
    static_argnames=(
        "resolution",
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "kernel",
        "polar_resolution",
        "polar_angular_bins",
        "polar_radial_capacity",
        "polar_band_capacity",
        "polar_limb_samples",
        "polar_angular_chunk_size",
        "polar_fallback_on_overflow",
        "moment_mode",
        "source_plane_fallback",
        "source_plane_rule",
        "source_plane_coarse_order",
        "source_plane_fine_order",
        "source_plane_angular_multiplier",
        "expanded_cartesian_fallback",
        "expanded_coarse_resolution",
        "expanded_fine_resolution",
        "expanded_coarse_tile_capacity",
        "expanded_fine_tile_capacity",
        "expanded_limb_samples",
        "cartesian_backend",
        "root_backend",
        "trajectory_backend",
    ),
)
def _binary_magnification_trajectory(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
    *,
    absolute_tolerance,
    relative_tolerance,
    multipole_safety_factor,
    resolution,
    tile_size,
    tile_capacity,
    limb_samples,
    kernel,
    polar_resolution,
    polar_angular_bins,
    polar_radial_capacity,
    polar_band_capacity,
    polar_limb_samples,
    polar_angular_chunk_size,
    polar_magnification_threshold,
    polar_max_source_radius,
    polar_min_mass_ratio,
    polar_fallback_on_overflow,
    moment_mode,
    source_plane_fallback,
    source_plane_rule,
    source_plane_coarse_order,
    source_plane_fine_order,
    source_plane_angular_multiplier,
    expanded_cartesian_fallback,
    expanded_coarse_resolution,
    expanded_fine_resolution,
    expanded_coarse_tile_capacity,
    expanded_fine_tile_capacity,
    expanded_limb_samples,
    cartesian_backend,
    root_backend,
    trajectory_backend,
):
    def evaluate_epoch(position):
        return binary_magnification_auto(
            position[0],
            position[1],
            position[2],
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            multipole_safety_factor=multipole_safety_factor,
            resolution=resolution,
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            kernel=kernel,
            polar_resolution=polar_resolution,
            polar_angular_bins=polar_angular_bins,
            polar_radial_capacity=polar_radial_capacity,
            polar_band_capacity=polar_band_capacity,
            polar_limb_samples=polar_limb_samples,
            polar_angular_chunk_size=polar_angular_chunk_size,
            polar_magnification_threshold=polar_magnification_threshold,
            polar_max_source_radius=polar_max_source_radius,
            polar_min_mass_ratio=polar_min_mass_ratio,
            polar_fallback_on_overflow=polar_fallback_on_overflow,
            source_plane_fallback=source_plane_fallback,
            source_plane_rule=source_plane_rule,
            source_plane_coarse_order=source_plane_coarse_order,
            source_plane_fine_order=source_plane_fine_order,
            source_plane_angular_multiplier=source_plane_angular_multiplier,
            expanded_cartesian_fallback=expanded_cartesian_fallback,
            expanded_coarse_resolution=expanded_coarse_resolution,
            expanded_fine_resolution=expanded_fine_resolution,
            expanded_coarse_tile_capacity=expanded_coarse_tile_capacity,
            expanded_fine_tile_capacity=expanded_fine_tile_capacity,
            expanded_limb_samples=expanded_limb_samples,
            moment_mode=moment_mode,
            cartesian_backend=cartesian_backend,
            root_backend=root_backend,
        )

    use_batch = trajectory_backend == "batch" or (
        trajectory_backend == "auto"
        and jax.default_backend() == "cpu"
        and kernel == "real"
        and cartesian_backend != "jax"
        and root_backend != "jax"
        and cpp_cartesian_batch_ffi_available()
    )
    use_integrated_ffi = use_batch and cpp_trajectory_ffi_available()
    if use_integrated_ffi:
        fused = binary_magnification_trajectory_ffi(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            multipole_safety_factor=multipole_safety_factor,
            polar_magnification_threshold=polar_magnification_threshold,
            polar_max_source_radius=polar_max_source_radius,
            polar_min_mass_ratio=polar_min_mass_ratio,
            resolution=resolution,
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            polar_resolution=polar_resolution,
            polar_angular_bins=polar_angular_bins,
            polar_radial_capacity=polar_radial_capacity,
            polar_band_capacity=polar_band_capacity,
            polar_limb_samples=polar_limb_samples,
            polar_angular_chunk_size=polar_angular_chunk_size,
            polar_fallback_on_overflow=polar_fallback_on_overflow,
            moment_mode=moment_mode,
        )
        provisional = HybridMagnificationResult(
            magnification=fused.magnification,
            method=fused.method,
            estimated_error=fused.estimated_error,
            support_valid=fused.support_valid,
            used_multipole=fused.used_multipole,
            used_polar=fused.used_polar,
            used_source_plane=jnp.zeros_like(fused.support_valid),
            used_expanded_cartesian=jnp.zeros_like(fused.support_valid),
        )

        def evaluate_fused_exception(operand):
            position, use_scalar, fast_result = operand
            return jax.lax.cond(
                use_scalar,
                evaluate_epoch,
                lambda _: fast_result,
                position,
            )

        result = jax.lax.map(
            evaluate_fused_exception,
            ((source_x, source_y, separation), fused.needs_fallback, provisional),
        )
    elif use_batch:

        def evaluate_hexadecapole(position):
            return binary_hexadecapole(
                position[0],
                position[1],
                position[2],
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                root_backend=root_backend,
            )

        hexadecapole = jax.lax.map(
            evaluate_hexadecapole,
            (source_x, source_y, separation),
        )
        accept_multipole, polar_allowed = _multipole_dispatch_masks(
            hexadecapole,
            mass_ratio,
            source_radius,
            absolute_tolerance,
            relative_tolerance,
            multipole_safety_factor,
            polar_min_mass_ratio,
        )
        preselect_polar = jax.lax.stop_gradient(
            (hexadecapole.point_magnification >= polar_magnification_threshold)
            & (source_radius <= polar_max_source_radius)
            & (source_radius > 0.0)
            & polar_allowed
        )
        active_cartesian = ~accept_multipole & ~preselect_polar
        cartesian = binary_inverse_ray_cartesian_batch_ffi(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            active=active_cartesian,
            cell_size=jax.lax.stop_gradient(source_radius / resolution),
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            moment_mode=moment_mode,
            boundary_subdivision=4 if moment_mode == "two_coefficient" else 3,
        )
        provisional = HybridMagnificationResult(
            magnification=jnp.where(
                accept_multipole,
                hexadecapole.magnification,
                cartesian.magnification,
            ),
            method=jnp.where(accept_multipole, 0, 1).astype(jnp.int32),
            estimated_error=jnp.where(
                accept_multipole,
                hexadecapole.estimated_error,
                jnp.nan,
            ),
            support_valid=accept_multipole | cartesian.support_valid,
            used_multipole=accept_multipole,
            used_polar=jnp.zeros_like(accept_multipole),
            used_source_plane=jnp.zeros_like(accept_multipole),
            used_expanded_cartesian=jnp.zeros_like(accept_multipole),
        )
        use_scalar_dispatcher = ~accept_multipole & (
            preselect_polar | ~cartesian.support_valid
        )

        def evaluate_exception(operand):
            position, use_scalar, fast_result = operand
            return jax.lax.cond(
                use_scalar,
                evaluate_epoch,
                lambda _: fast_result,
                position,
            )

        result = jax.lax.map(
            evaluate_exception,
            (
                (source_x, source_y, separation),
                use_scalar_dispatcher,
                provisional,
            ),
        )
    else:
        result = jax.lax.map(evaluate_epoch, (source_x, source_y, separation))
    attempted_counts = jnp.stack(
        (
            jnp.sum(result.used_multipole, dtype=jnp.int32),
            jnp.sum(~result.used_multipole, dtype=jnp.int32),
            jnp.sum(
                result.used_source_plane | result.used_expanded_cartesian,
                dtype=jnp.int32,
            ),
            jnp.sum(result.used_expanded_cartesian, dtype=jnp.int32),
        )
    )
    return TrajectoryMagnificationResult(
        magnification=result.magnification,
        method=result.method,
        estimated_error=result.estimated_error,
        support_valid=result.support_valid,
        value_converged=result.support_valid,
        used_multipole=result.used_multipole,
        used_polar=result.used_polar,
        used_source_plane=result.used_source_plane,
        used_expanded_cartesian=result.used_expanded_cartesian,
        attempted_counts=attempted_counts,
    )


def binary_magnification_trajectory(
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
    multipole_safety_factor=4.0,
    resolution=96,
    tile_size=16,
    tile_capacity=4096,
    limb_samples=24,
    kernel="real",
    polar_resolution=128,
    polar_angular_bins=4096,
    polar_radial_capacity=256,
    polar_band_capacity=4,
    polar_limb_samples=32,
    polar_angular_chunk_size=1024,
    polar_magnification_threshold=80.0,
    polar_max_source_radius=0.01,
    polar_min_mass_ratio=5.0e-3,
    polar_fallback_on_overflow=False,
    moment_mode="two_coefficient",
    source_plane_fallback=True,
    source_plane_rule="chord",
    source_plane_coarse_order=16,
    source_plane_fine_order=32,
    source_plane_angular_multiplier=4,
    expanded_cartesian_fallback=False,
    expanded_coarse_resolution=64,
    expanded_fine_resolution=128,
    expanded_coarse_tile_capacity=4096,
    expanded_fine_tile_capacity=16384,
    expanded_limb_samples=32,
    cartesian_backend="auto",
    root_backend="auto",
    trajectory_backend="auto",
):
    """Evaluate a one-dimensional trajectory with per-epoch conditionals.

    On CPU, ``trajectory_backend="auto"`` uses one masked C++ FFI call for
    independent Cartesian epochs while retaining scalar conditionals for polar
    and fallback epochs. ``trajectory_backend="scalar"`` preserves the
    one-epoch-at-a-time reference path. The trajectory defaults use the
    calibrated 96/4096 Cartesian bucket; the scalar API keeps its lower-latency
    64/1024 default. Method selection is stopped-gradient while every selected
    magnification remains differentiable.
    """

    require_x64()
    source_x = jnp.asarray(source_x)
    source_y = jnp.asarray(source_y)
    if source_x.ndim != 1 or source_y.shape != source_x.shape:
        raise ValueError("source_x and source_y must have the same 1-D shape")
    separation = jnp.asarray(separation)
    if separation.ndim == 0:
        separation = jnp.full_like(source_x, separation)
    elif separation.shape != source_x.shape:
        raise ValueError("separation must be scalar or match source_x")
    if trajectory_backend not in ("auto", "scalar", "batch"):
        raise ValueError("trajectory_backend must be 'auto', 'scalar', or 'batch'")
    if trajectory_backend == "batch":
        if (
            jax.default_backend() != "cpu"
            or kernel != "real"
            or cartesian_backend == "jax"
            or root_backend == "jax"
            or not cpp_cartesian_batch_ffi_available()
        ):
            raise RuntimeError(
                "the batched trajectory backend requires the CPU real-kernel "
                "Cartesian/root FFI"
            )
    return _binary_magnification_trajectory(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        multipole_safety_factor=multipole_safety_factor,
        resolution=resolution,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        kernel=kernel,
        polar_resolution=polar_resolution,
        polar_angular_bins=polar_angular_bins,
        polar_radial_capacity=polar_radial_capacity,
        polar_band_capacity=polar_band_capacity,
        polar_limb_samples=polar_limb_samples,
        polar_angular_chunk_size=polar_angular_chunk_size,
        polar_magnification_threshold=polar_magnification_threshold,
        polar_max_source_radius=polar_max_source_radius,
        polar_min_mass_ratio=polar_min_mass_ratio,
        polar_fallback_on_overflow=polar_fallback_on_overflow,
        moment_mode=moment_mode,
        source_plane_fallback=source_plane_fallback,
        source_plane_rule=source_plane_rule,
        source_plane_coarse_order=source_plane_coarse_order,
        source_plane_fine_order=source_plane_fine_order,
        source_plane_angular_multiplier=source_plane_angular_multiplier,
        expanded_cartesian_fallback=expanded_cartesian_fallback,
        expanded_coarse_resolution=expanded_coarse_resolution,
        expanded_fine_resolution=expanded_fine_resolution,
        expanded_coarse_tile_capacity=expanded_coarse_tile_capacity,
        expanded_fine_tile_capacity=expanded_fine_tile_capacity,
        expanded_limb_samples=expanded_limb_samples,
        cartesian_backend=cartesian_backend,
        root_backend=root_backend,
        trajectory_backend=trajectory_backend,
    )


def binary_magnification_native_pipeline_trajectory(
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
    **trajectory_options,
):
    """Batched trajectory with native point and grazing-source routing.

    The fused trajectory remains the image-plane engine. For a static binary
    lens, one native diagnostic batch adds the exact point-source safety and
    caustic-band decisions used by ``lcbinint``. Point and converged chord
    48/96 source-plane epochs replace the fused result. Dynamic-separation
    trajectories retain the fused route because this diagnostic FFI shares
    one static caustic cache.
    """

    require_x64()
    source_x = jnp.asarray(source_x)
    source_y = jnp.asarray(source_y)
    separation_array = jnp.asarray(separation)
    if (
        separation_array.ndim != 0
        or not cpp_binary_routing_diagnostics_batch_ffi_available()
    ):
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
            **trajectory_options,
        )

    base = binary_magnification_trajectory(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        **trajectory_options,
    )

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
        caustic_bins=caustic_bins,
    )
    point_route = jax.lax.stop_gradient(routing.point_safe & ~expansion.root_failure)
    source_route = jax.lax.stop_gradient(
        ~point_route & (routing.chord_band | routing.grazing_ring_band)
    )

    def source_epoch(operand):
        position, active = operand

        def evaluate(_):
            candidate = binary_source_plane_quadrature(
                position[0],
                position[1],
                separation_array,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                rule="chord",
                coarse_order=48,
                fine_order=96,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            return (
                candidate.magnification,
                candidate.estimated_error,
                candidate.converged,
            )

        return jax.lax.cond(
            active,
            evaluate,
            lambda _: (
                jnp.asarray(jnp.nan, dtype=source_x.dtype),
                jnp.asarray(jnp.inf, dtype=source_x.dtype),
                jnp.asarray(False),
            ),
            None,
        )

    source_value, source_error, source_converged = jax.lax.map(
        source_epoch,
        ((source_x, source_y), source_route),
    )
    accept_source = jax.lax.stop_gradient(source_route & source_converged)

    if maximum_source_bins not in {
        resolution for resolution, _ in _CALIBRATED_EXECUTION_BUCKETS
    }:
        raise ValueError("maximum_source_bins must be a calibrated resolution bucket")
    explicit_tolerance = (absolute_tolerance > 0.0) | (relative_tolerance > 0.0)
    requested_relative_tolerance = jnp.where(
        explicit_tolerance,
        relative_tolerance
        + absolute_tolerance / jnp.maximum(jnp.abs(expansion.point_magnification), 1.0),
        1.0e-3,
    )
    selections = jax.vmap(
        lambda distance, point, tolerance: select_binary_resolution(
            mass_ratio,
            source_radius,
            distance,
            point,
            limb_c,
            tolerance,
            maximum_source_bins,
        )
    )(
        routing.caustic_distance,
        jax.lax.stop_gradient(expansion.point_magnification),
        requested_relative_tolerance,
    )
    bucket_resolutions = jnp.asarray(
        tuple(item[0] for item in _CALIBRATED_EXECUTION_BUCKETS),
        dtype=jnp.int32,
    )
    bucket_index = jnp.argmax(
        bucket_resolutions[:, None] >= selections.source_bins[None, :],
        axis=0,
    )
    multipole_safety = jnp.where(
        (source_radius <= 0.0) | (routing.caustic_distance >= 3.0 * source_radius),
        4.0,
        jnp.inf,
    )
    accept_multipole, _ = _multipole_dispatch_masks(
        expansion,
        mass_ratio,
        source_radius,
        absolute_tolerance,
        relative_tolerance,
        multipole_safety,
        0.0,
        False,
    )
    accept_multipole = jax.lax.stop_gradient(
        ~point_route & ~accept_source & accept_multipole
    )
    polar_candidate = jax.lax.stop_gradient(
        ~point_route & ~accept_source & ~accept_multipole & selections.prefer_polar
    )
    cartesian_candidate = jax.lax.stop_gradient(
        ~point_route & ~accept_source & ~accept_multipole & ~selections.prefer_polar
    )
    moment_mode = trajectory_options.get("moment_mode", "two_coefficient")
    boundary_subdivision = 4 if moment_mode == "two_coefficient" else 3

    polar_branches = []
    for resolution, _ in _CALIBRATED_EXECUTION_BUCKETS:
        radial_target = max(256, 8 * resolution)
        radial_capacity = 1 << (radial_target - 1).bit_length()

        def polar_branch(
            position,
            resolution=resolution,
            radial_capacity=radial_capacity,
        ):
            candidate = binary_inverse_ray_polar_ffi(
                position[0],
                position[1],
                separation_array,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                resolution=resolution,
                angular_bins=65536,
                radial_capacity=radial_capacity,
                limb_samples=64,
                angular_chunk_size=256,
                boundary_subdivision=4,
                moment_mode=moment_mode,
            )
            return candidate.magnification, candidate.support_valid

        polar_branches.append(polar_branch)

    def polar_epoch(operand):
        position, selected_index, active = operand

        def evaluate(_):
            selected_value, selected_valid = jax.lax.switch(
                selected_index, tuple(polar_branches), position
            )
            lower = jnp.maximum(selected_index - 1, 0)
            upper = jnp.minimum(
                selected_index + 1,
                len(_CALIBRATED_EXECUTION_BUCKETS) - 1,
            )
            comparison_index = jnp.where(selected_index == 0, upper, lower)
            comparison_value, comparison_valid = jax.lax.switch(
                comparison_index, tuple(polar_branches), position
            )
            initial_error = jnp.abs(selected_value - comparison_value)
            initial_budget = absolute_tolerance + relative_tolerance * jnp.maximum(
                jnp.abs(selected_value), 1.0
            )
            initial_converged = (
                (comparison_index != selected_index)
                & selected_valid
                & comparison_valid
                & jnp.isfinite(initial_error)
                & (initial_error <= initial_budget)
            )
            upper_value, upper_valid = jax.lax.switch(
                upper, tuple(polar_branches), position
            )
            retry_with_upper = ~initial_converged & (upper != selected_index)
            value = jnp.where(retry_with_upper, upper_value, selected_value)
            valid = jnp.where(retry_with_upper, upper_valid, selected_valid)
            retry_error = jnp.abs(upper_value - selected_value)
            error = jnp.where(retry_with_upper, retry_error, initial_error)
            budget = absolute_tolerance + relative_tolerance * jnp.maximum(
                jnp.abs(value), 1.0
            )
            retry_converged = (
                retry_with_upper
                & selected_valid
                & upper_valid
                & jnp.isfinite(retry_error)
                & (retry_error <= budget)
            )
            converged = initial_converged | retry_converged
            return value, error, valid, converged

        return jax.lax.cond(
            active,
            evaluate,
            lambda _: (
                jnp.asarray(jnp.nan, dtype=source_x.dtype),
                jnp.asarray(jnp.inf, dtype=source_x.dtype),
                jnp.asarray(False),
                jnp.asarray(False),
            ),
            None,
        )

    polar_value, polar_error, polar_support, polar_converged = jax.lax.map(
        polar_epoch,
        ((source_x, source_y), bucket_index, polar_candidate),
    )
    accept_polar = jax.lax.stop_gradient(polar_candidate & polar_converged)
    cartesian_results = []
    for index, (resolution, tile_capacity) in enumerate(_CALIBRATED_EXECUTION_BUCKETS):
        needed = cartesian_candidate & (
            (bucket_index == index)
            | (bucket_index == index - 1)
            | (bucket_index == index + 1)
        )
        cartesian_results.append(
            binary_inverse_ray_cartesian_batch_ffi(
                source_x,
                source_y,
                separation_array,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                active=needed,
                cell_size=jax.lax.stop_gradient(source_radius / resolution),
                tile_size=16,
                tile_capacity=tile_capacity,
                limb_samples=32,
                moment_mode=moment_mode,
                boundary_subdivision=boundary_subdivision,
            )
        )
    cartesian_values = jnp.stack(
        tuple(item.magnification for item in cartesian_results)
    )
    cartesian_support = jnp.stack(
        tuple(item.support_valid for item in cartesian_results)
    )

    def gather_epoch(values, indices):
        return jnp.take_along_axis(values, indices[None, :], axis=0)[0]

    selected_cartesian = gather_epoch(cartesian_values, bucket_index)
    selected_support = gather_epoch(cartesian_support, bucket_index)
    lower_index = jnp.maximum(bucket_index - 1, 0)
    lower_support = gather_epoch(cartesian_support, lower_index)
    upper_index = jnp.minimum(bucket_index + 1, len(_CALIBRATED_EXECUTION_BUCKETS) - 1)
    upper_value = gather_epoch(cartesian_values, upper_index)
    upper_support = gather_epoch(cartesian_support, upper_index)
    use_upper = (bucket_index == 0) | ~lower_support
    comparison_index = jnp.where(use_upper, upper_index, lower_index)
    comparison_value = gather_epoch(cartesian_values, comparison_index)
    comparison_support = gather_epoch(cartesian_support, comparison_index)
    initial_cartesian_error = jnp.abs(selected_cartesian - comparison_value)
    initial_cartesian_budget = absolute_tolerance + relative_tolerance * jnp.maximum(
        jnp.abs(selected_cartesian), 1.0
    )
    initial_cartesian_converged = (
        cartesian_candidate
        & (comparison_index != bucket_index)
        & selected_support
        & comparison_support
        & jnp.isfinite(initial_cartesian_error)
        & (initial_cartesian_error <= initial_cartesian_budget)
    )
    retry_cartesian = jax.lax.stop_gradient(
        cartesian_candidate
        & ~initial_cartesian_converged
        & (upper_index != bucket_index)
    )
    retry_cartesian_error = jnp.abs(upper_value - selected_cartesian)
    selected_cartesian = jnp.where(retry_cartesian, upper_value, selected_cartesian)
    selected_support = jnp.where(retry_cartesian, upper_support, selected_support)
    cartesian_error = jnp.where(
        retry_cartesian,
        retry_cartesian_error,
        initial_cartesian_error,
    )
    cartesian_budget = absolute_tolerance + relative_tolerance * jnp.maximum(
        jnp.abs(selected_cartesian), 1.0
    )
    accept_cartesian = jax.lax.stop_gradient(
        initial_cartesian_converged
        | (
            retry_cartesian
            & selected_support
            & gather_epoch(cartesian_support, bucket_index)
            & jnp.isfinite(retry_cartesian_error)
            & (retry_cartesian_error <= cartesian_budget)
        )
    )
    # Preserve the selected calibrated value even if the adjacent-resolution
    # check misses the requested budget. Falling back to ``base`` here used a
    # lower-resolution value and could make a stricter tolerance less accurate.
    grid_magnification = jnp.where(
        cartesian_candidate, selected_cartesian, base.magnification
    )
    grid_error = jnp.where(cartesian_candidate, cartesian_error, base.estimated_error)
    grid_support = jnp.where(cartesian_candidate, selected_support, base.support_valid)
    selected_magnification = jnp.where(
        accept_multipole,
        expansion.magnification,
        jnp.where(polar_candidate, polar_value, grid_magnification),
    )
    selected_method = jnp.where(
        accept_multipole,
        0,
        jnp.where(
            polar_candidate,
            2,
            jnp.where(cartesian_candidate, 1, base.method),
        ),
    ).astype(jnp.int32)
    selected_error = jnp.where(
        accept_multipole,
        expansion.estimated_error,
        jnp.where(polar_candidate, polar_error, grid_error),
    )
    selected_support = (
        accept_multipole
        | (polar_candidate & polar_support)
        | (~polar_candidate & grid_support)
    )
    magnification = jnp.where(
        point_route,
        expansion.point_magnification,
        jnp.where(accept_source, source_value, selected_magnification),
    )
    method = jnp.where(
        point_route,
        4,
        jnp.where(accept_source, 3, selected_method),
    ).astype(jnp.int32)
    estimated_error = jnp.where(
        point_route,
        routing.point_error_estimate,
        jnp.where(accept_source, source_error, selected_error),
    )
    support_valid = point_route | accept_source | selected_support
    value_converged = jax.lax.stop_gradient(
        point_route
        | accept_source
        | accept_multipole
        | (polar_candidate & polar_converged)
        | (cartesian_candidate & accept_cartesian)
    )
    used_multipole = accept_multipole
    used_polar = polar_candidate
    fallback_selected = (
        ~point_route
        & ~accept_source
        & ~accept_multipole
        & ~polar_candidate
        & ~cartesian_candidate
    )
    used_source_plane = accept_source | (fallback_selected & base.used_source_plane)
    used_expanded_cartesian = fallback_selected & base.used_expanded_cartesian
    attempted_counts = jnp.stack(
        (
            jnp.sum(used_multipole, dtype=jnp.int32),
            jnp.sum(~used_multipole, dtype=jnp.int32),
            jnp.sum(
                used_source_plane | used_expanded_cartesian,
                dtype=jnp.int32,
            ),
            jnp.sum(used_expanded_cartesian, dtype=jnp.int32),
        )
    )
    return TrajectoryMagnificationResult(
        magnification=magnification,
        method=method,
        estimated_error=estimated_error,
        support_valid=support_valid,
        value_converged=value_converged,
        used_multipole=used_multipole,
        used_polar=used_polar,
        used_source_plane=used_source_plane,
        used_expanded_cartesian=used_expanded_cartesian,
        attempted_counts=attempted_counts,
    )

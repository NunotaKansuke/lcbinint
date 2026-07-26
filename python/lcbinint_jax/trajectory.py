"""Differentiable trajectory dispatcher with scalar conditional epochs."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import require_x64
from .api import _multipole_dispatch_masks, binary_magnification_auto
from .cpp_backend import (
    binary_magnification_trajectory_ffi,
    binary_inverse_ray_cartesian_batch_ffi,
    cpp_cartesian_batch_ffi_available,
    cpp_trajectory_ffi_available,
)
from .multipole import binary_hexadecapole
from .types import HybridMagnificationResult, TrajectoryMagnificationResult


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
            ((source_x, source_y), fused.needs_fallback, provisional),
        )
    elif use_batch:

        def evaluate_hexadecapole(position):
            return binary_hexadecapole(
                position[0],
                position[1],
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                root_backend=root_backend,
            )

        hexadecapole = jax.lax.map(
            evaluate_hexadecapole,
            (source_x, source_y),
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
            ((source_x, source_y), use_scalar_dispatcher, provisional),
        )
    else:
        result = jax.lax.map(evaluate_epoch, (source_x, source_y))
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

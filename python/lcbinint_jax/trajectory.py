"""Differentiable trajectory dispatcher with scalar conditional epochs."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import require_x64
from .api import binary_magnification_auto
from .types import TrajectoryMagnificationResult


@partial(
    jax.jit,
    static_argnames=(
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
        )

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
):
    """Evaluate a one-dimensional trajectory with per-epoch conditionals.

    ``lax.map`` deliberately preserves scalar ``lax.cond`` branches. On CPU
    this is substantially faster than batching the branch predicate with
    ``vmap``, which can evaluate expensive unselected methods. The trajectory
    defaults use the calibrated 96/4096 Cartesian bucket; the scalar API keeps
    its lower-latency 64/1024 default. ``cartesian_backend="auto"`` selects the
    typed C++ FFI for the real CPU kernel when available and otherwise retains
    the pure-JAX implementation. Method selection is stopped-gradient while
    every selected magnification remains differentiable.
    """

    require_x64()
    source_x = jnp.asarray(source_x)
    source_y = jnp.asarray(source_y)
    if source_x.ndim != 1 or source_y.shape != source_x.shape:
        raise ValueError("source_x and source_y must have the same 1-D shape")
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
    )

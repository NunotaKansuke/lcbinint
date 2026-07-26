"""Experimental end-to-end JAX inverse-ray entry points."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import require_x64
from .discovery import discover_binary_macro_tiles
from .images import binary_images
from .integrate import binary_inverse_ray_fixed_support
from .lens import binary_lens_map_and_derivatives_real
from .multipole import binary_hexadecapole
from .polar import binary_inverse_ray_polar
from .source_plane import binary_source_plane_quadrature
from .types import (
    AutoInverseRayResult,
    HybridMagnificationResult,
    InverseRayResult,
)


def _binary_inverse_ray(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
    *,
    resolution,
    tile_size,
    tile_capacity,
    limb_samples,
    kernel,
    moment_mode,
):
    cell_size = jax.lax.stop_gradient(source_radius / resolution)
    discovery = discover_binary_macro_tiles(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        cell_size,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
    )
    integrated = binary_inverse_ray_fixed_support(
        discovery.tile_origins,
        discovery.tile_mask,
        cell_size,
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        tile_size=tile_size,
        kernel=kernel,
        moment_mode=moment_mode,
    )
    return _cartesian_result(discovery, integrated)


def _cartesian_result(discovery, integrated):
    support_valid = ~(discovery.overflow | discovery.root_failure)
    return InverseRayResult(
        magnification=integrated.magnification,
        moments=integrated.moments,
        boundary_cells=integrated.boundary_cells,
        active_cells=integrated.active_cells,
        tile_count=discovery.visited_count,
        discovery_overflow=discovery.overflow,
        root_failure=discovery.root_failure,
        support_valid=support_valid,
    )


def _auto_result(result, used_polar):
    return AutoInverseRayResult(
        magnification=result.magnification,
        moments=result.moments,
        boundary_cells=result.boundary_cells,
        active_cells=result.active_cells,
        support_count=result.tile_count,
        discovery_overflow=result.discovery_overflow,
        root_failure=result.root_failure,
        support_valid=result.support_valid,
        used_polar=jnp.asarray(used_polar),
    )


def _point_source_magnification(
    source_x,
    source_y,
    separation,
    mass_ratio,
):
    images = binary_images(source_x + 1j * source_y, separation, mass_ratio)
    _, _, du_dx, du_dy, dv_dx, dv_dy = binary_lens_map_and_derivatives_real(
        jnp.real(images.roots),
        jnp.imag(images.roots),
        separation,
        mass_ratio,
    )
    determinant = du_dx * dv_dy - du_dy * dv_dx
    safe = images.physical & (jnp.abs(determinant) > 1.0e-14)
    return jax.lax.stop_gradient(
        jnp.sum(jnp.where(safe, 1.0 / jnp.abs(determinant), 0.0))
    )


@partial(
    jax.jit,
    static_argnames=("tile_size", "tile_capacity", "limb_samples", "kernel"),
)
def binary_inverse_ray(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
):
    """Discover image support and integrate a finite binary-lens source."""

    require_x64()
    return _binary_inverse_ray(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
        resolution=resolution,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        kernel=kernel,
        moment_mode="two_coefficient",
    )


@partial(
    jax.jit,
    static_argnames=("tile_size", "tile_capacity", "limb_samples", "kernel"),
)
def binary_inverse_ray_uniform(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    *,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
):
    """Specialized inverse-ray path for a uniform source."""

    require_x64()
    return _binary_inverse_ray(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        0.0,
        0.0,
        resolution=resolution,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        kernel=kernel,
        moment_mode="uniform",
    )


@partial(
    jax.jit,
    static_argnames=("tile_size", "tile_capacity", "limb_samples", "kernel"),
)
def binary_inverse_ray_linear(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    *,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
):
    """Specialized inverse-ray path for linear limb darkening."""

    require_x64()
    return _binary_inverse_ray(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        0.0,
        resolution=resolution,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        kernel=kernel,
        moment_mode="linear",
    )


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
    ),
)
def binary_inverse_ray_auto(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    kernel="real",
    polar_resolution=128,
    polar_angular_bins=4096,
    polar_radial_capacity=256,
    polar_band_capacity=4,
    polar_limb_samples=32,
    polar_angular_chunk_size=1024,
    polar_magnification_threshold=80.0,
    polar_max_source_radius=0.01,
    polar_allowed=True,
    polar_fallback_on_overflow=True,
    moment_mode="two_coefficient",
):
    """Automatically dispatch between Cartesian and polar inverse rays.

    Tiny high-magnification sources take the polar path before Cartesian tile
    discovery. Other sources use Cartesian integration, with polar as a
    fixed-capacity overflow fallback.
    """

    require_x64()
    if moment_mode not in ("uniform", "linear", "two_coefficient"):
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )

    def polar_path(_):
        result = binary_inverse_ray_polar(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            resolution=polar_resolution,
            angular_bins=polar_angular_bins,
            radial_capacity=polar_radial_capacity,
            band_capacity=polar_band_capacity,
            limb_samples=polar_limb_samples,
            angular_chunk_size=polar_angular_chunk_size,
            kernel=kernel,
            moment_mode=moment_mode,
        )
        return _auto_result(result, True)

    def cartesian_or_fallback(_):
        cell_size = jax.lax.stop_gradient(source_radius / resolution)
        discovery = discover_binary_macro_tiles(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            cell_size,
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
        )

        def fallback(_):
            return polar_path(None)

        def integrate_cartesian(_):
            integrated = binary_inverse_ray_fixed_support(
                discovery.tile_origins,
                discovery.tile_mask,
                cell_size,
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                tile_size=tile_size,
                kernel=kernel,
                moment_mode=moment_mode,
                boundary_subdivision=(4 if moment_mode == "two_coefficient" else 3),
            )
            return _auto_result(_cartesian_result(discovery, integrated), False)

        use_fallback = (
            discovery.overflow
            & jnp.asarray(polar_allowed)
            & jnp.asarray(polar_fallback_on_overflow)
        )
        return jax.lax.cond(use_fallback, fallback, integrate_cartesian, None)

    point_magnification = _point_source_magnification(
        source_x,
        source_y,
        separation,
        mass_ratio,
    )
    preselect_polar = (
        (point_magnification >= polar_magnification_threshold)
        & (source_radius <= polar_max_source_radius)
        & (source_radius > 0.0)
        & jnp.asarray(polar_allowed)
    )
    return jax.lax.cond(
        preselect_polar,
        polar_path,
        cartesian_or_fallback,
        None,
    )


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
    ),
)
def binary_magnification_auto(
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
    resolution=64,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
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
    moment_mode="two_coefficient",
):
    """Dispatch safely-far epochs to a differentiable hexadecapole expansion.

    Method codes are 0 for hexadecapole, 1 for Cartesian inverse rays, 2 for
    polar inverse rays, and 3 for source-plane quadrature. Source-plane
    quadrature is attempted only when fixed-capacity image support is invalid,
    and is accepted only when its independent coarse/fine check converges.
    ``expanded_cartesian_fallback=True`` adds a costly coarse/fine Cartesian
    retry after source-plane rejection; it is disabled by default so trajectory
    callers can bucket only failed epochs into that executable.
    The multipole error check remains an empirical guard rather than a proof
    that no unresolved caustic lies inside the source.
    """

    require_x64()
    hexadecapole = binary_hexadecapole(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
    )
    budget = absolute_tolerance + relative_tolerance * jnp.maximum(
        jnp.abs(hexadecapole.magnification), 1.0
    )
    correction_scale = jnp.maximum(jnp.abs(hexadecapole.quadrupole_correction), budget)
    expansion_well_ordered = (
        hexadecapole.estimated_error <= 0.25 * correction_scale
    ) & (
        jnp.abs(hexadecapole.quadrupole_correction)
        <= 0.1 * jnp.maximum(jnp.abs(hexadecapole.magnification), 1.0)
    )
    accept_multipole = jax.lax.stop_gradient(
        (source_radius >= 0.0)
        & hexadecapole.topology_stable
        & ~hexadecapole.root_failure
        & jnp.isfinite(hexadecapole.magnification)
        & expansion_well_ordered
        & (multipole_safety_factor * hexadecapole.estimated_error <= budget)
    )
    absolute_mass_ratio = jnp.abs(mass_ratio)
    symmetric_mass_ratio = jnp.minimum(
        absolute_mass_ratio,
        jnp.where(
            absolute_mass_ratio > 0.0,
            1.0 / absolute_mass_ratio,
            0.0,
        ),
    )
    polar_allowed = jax.lax.stop_gradient(
        hexadecapole.topology_stable & (symmetric_mass_ratio >= polar_min_mass_ratio)
    )

    def multipole_path(_):
        return HybridMagnificationResult(
            magnification=hexadecapole.magnification,
            method=jnp.asarray(0, dtype=jnp.int32),
            estimated_error=hexadecapole.estimated_error,
            support_valid=jnp.asarray(True),
            used_multipole=jnp.asarray(True),
            used_polar=jnp.asarray(False),
            used_source_plane=jnp.asarray(False),
            used_expanded_cartesian=jnp.asarray(False),
        )

    def inverse_ray_path(_):
        result = binary_inverse_ray_auto(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
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
            polar_allowed=polar_allowed,
            polar_fallback_on_overflow=polar_fallback_on_overflow,
            moment_mode=moment_mode,
        )

        def image_plane_result(_):
            return HybridMagnificationResult(
                magnification=result.magnification,
                method=jnp.where(result.used_polar, 2, 1).astype(jnp.int32),
                estimated_error=jnp.asarray(jnp.nan),
                support_valid=jnp.asarray(True),
                used_multipole=jnp.asarray(False),
                used_polar=result.used_polar,
                used_source_plane=jnp.asarray(False),
                used_expanded_cartesian=jnp.asarray(False),
            )

        def source_plane_result(_):
            source_plane = binary_source_plane_quadrature(
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                rule=source_plane_rule,
                coarse_order=source_plane_coarse_order,
                fine_order=source_plane_fine_order,
                angular_multiplier=source_plane_angular_multiplier,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            valid = source_plane.converged

            def accepted_source_plane(_):
                return HybridMagnificationResult(
                    magnification=source_plane.magnification,
                    method=jnp.asarray(3, dtype=jnp.int32),
                    estimated_error=source_plane.estimated_error,
                    support_valid=jnp.asarray(True),
                    used_multipole=jnp.asarray(False),
                    used_polar=jnp.asarray(False),
                    used_source_plane=jnp.asarray(True),
                    used_expanded_cartesian=jnp.asarray(False),
                )

            def rejected_source_plane(_):
                return HybridMagnificationResult(
                    magnification=jnp.asarray(jnp.nan),
                    method=jnp.asarray(3, dtype=jnp.int32),
                    estimated_error=source_plane.estimated_error,
                    support_valid=jnp.asarray(False),
                    used_multipole=jnp.asarray(False),
                    used_polar=jnp.asarray(False),
                    used_source_plane=jnp.asarray(True),
                    used_expanded_cartesian=jnp.asarray(False),
                )

            def expanded_cartesian_result(_):
                coarse = _binary_inverse_ray(
                    source_x,
                    source_y,
                    separation,
                    mass_ratio,
                    source_radius,
                    limb_c,
                    limb_d,
                    resolution=expanded_coarse_resolution,
                    tile_size=tile_size,
                    tile_capacity=expanded_coarse_tile_capacity,
                    limb_samples=expanded_limb_samples,
                    kernel=kernel,
                    moment_mode=moment_mode,
                )
                fine = _binary_inverse_ray(
                    source_x,
                    source_y,
                    separation,
                    mass_ratio,
                    source_radius,
                    limb_c,
                    limb_d,
                    resolution=expanded_fine_resolution,
                    tile_size=tile_size,
                    tile_capacity=expanded_fine_tile_capacity,
                    limb_samples=expanded_limb_samples,
                    kernel=kernel,
                    moment_mode=moment_mode,
                )
                error = jnp.abs(fine.magnification - coarse.magnification)
                retry_budget = absolute_tolerance + relative_tolerance * jnp.maximum(
                    jnp.abs(fine.magnification), 1.0
                )
                retry_valid = (
                    coarse.support_valid
                    & fine.support_valid
                    & jnp.isfinite(error)
                    & (error <= retry_budget)
                )
                return HybridMagnificationResult(
                    magnification=jnp.where(retry_valid, fine.magnification, jnp.nan),
                    method=jnp.asarray(1, dtype=jnp.int32),
                    estimated_error=error,
                    support_valid=retry_valid,
                    used_multipole=jnp.asarray(False),
                    used_polar=jnp.asarray(False),
                    used_source_plane=jnp.asarray(False),
                    used_expanded_cartesian=jnp.asarray(True),
                )

            if expanded_cartesian_fallback:
                return jax.lax.cond(
                    valid,
                    accepted_source_plane,
                    expanded_cartesian_result,
                    None,
                )
            return jax.lax.cond(
                valid,
                accepted_source_plane,
                rejected_source_plane,
                None,
            )

        def invalid_image_plane_result(_):
            return HybridMagnificationResult(
                magnification=jnp.asarray(jnp.nan),
                method=jnp.where(result.used_polar, 2, 1).astype(jnp.int32),
                estimated_error=jnp.asarray(jnp.nan),
                support_valid=jnp.asarray(False),
                used_multipole=jnp.asarray(False),
                used_polar=result.used_polar,
                used_source_plane=jnp.asarray(False),
                used_expanded_cartesian=jnp.asarray(False),
            )

        if source_plane_fallback:
            return jax.lax.cond(
                result.support_valid,
                image_plane_result,
                source_plane_result,
                None,
            )
        return jax.lax.cond(
            result.support_valid,
            image_plane_result,
            invalid_image_plane_result,
            None,
        )

    return jax.lax.cond(accept_multipole, multipole_path, inverse_ray_path, None)

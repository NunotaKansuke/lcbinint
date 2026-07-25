"""Experimental end-to-end JAX inverse-ray entry points."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import require_x64
from .discovery import discover_binary_macro_tiles
from .images import binary_images
from .integrate import binary_inverse_ray_fixed_support
from .lens import binary_lens_map_and_derivatives_real
from .polar import binary_inverse_ray_polar
from .types import AutoInverseRayResult, InverseRayResult


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
    limb_samples=32,
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
    limb_samples=32,
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
    limb_samples=32,
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
    limb_samples=32,
    kernel="real",
    polar_resolution=128,
    polar_angular_bins=8192,
    polar_radial_capacity=512,
    polar_band_capacity=4,
    polar_angular_chunk_size=32,
    polar_magnification_threshold=80.0,
    polar_max_source_radius=0.01,
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
            limb_samples=limb_samples,
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
            )
            return _auto_result(_cartesian_result(discovery, integrated), False)

        return jax.lax.cond(discovery.overflow, fallback, integrate_cartesian, None)

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
    )
    return jax.lax.cond(
        preselect_polar,
        polar_path,
        cartesian_or_fallback,
        None,
    )

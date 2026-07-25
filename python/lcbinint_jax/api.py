"""Experimental end-to-end JAX inverse-ray entry points."""

from functools import partial

import jax

from ._config import require_x64
from .discovery import discover_binary_macro_tiles
from .integrate import binary_inverse_ray_fixed_support
from .types import InverseRayResult


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
    )
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

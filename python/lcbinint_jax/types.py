"""Fixed-shape result types for the experimental JAX inverse-ray kernel."""

from typing import NamedTuple

import jax


class FixedSupportResult(NamedTuple):
    """Result of integrating a caller-supplied set of image-plane tiles."""

    magnification: jax.Array
    moments: jax.Array
    boundary_cells: jax.Array
    active_cells: jax.Array


class DiscoveryResult(NamedTuple):
    """Fixed-capacity macro-tile support discovered from image seeds."""

    tile_indices: jax.Array
    tile_origins: jax.Array
    tile_mask: jax.Array
    active_mask: jax.Array
    overflow: jax.Array
    visited_count: jax.Array
    active_count: jax.Array
    seed_count: jax.Array
    root_failure: jax.Array


class InverseRayResult(NamedTuple):
    """Magnification and discovery diagnostics from the automatic MVP path."""

    magnification: jax.Array
    moments: jax.Array
    boundary_cells: jax.Array
    active_cells: jax.Array
    tile_count: jax.Array
    discovery_overflow: jax.Array
    root_failure: jax.Array
    support_valid: jax.Array

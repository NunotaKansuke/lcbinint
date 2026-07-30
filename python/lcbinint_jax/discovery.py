"""Fixed-capacity macro-tile discovery for binary inverse-ray integration."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._config import as_float64
from .cpp_backend import (
    binary_images_ffi,
    cpp_binary_image_roots_ffi_available,
)
from .images import binary_images
from .lens import binary_lens_map_real
from .types import DiscoveryResult


class ImageSeedPoints(NamedTuple):
    roots: jax.Array
    physical: jax.Array
    residuals: jax.Array
    root_failure: jax.Array


@partial(jax.jit, static_argnames=("limb_samples", "root_backend"))
def binary_image_seed_points(
    source_x: jax.Array,
    source_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    source_radius: jax.Array,
    *,
    limb_samples: int = 16,
    root_backend: str = "auto",
) -> ImageSeedPoints:
    """Return physical centre and source-limb images as stopped-gradient seeds."""

    source_x, source_y, separation, mass_ratio, source_radius = (
        as_float64(value)
        for value in (source_x, source_y, separation, mass_ratio, source_radius)
    )
    if root_backend not in ("auto", "jax", "ffi"):
        raise ValueError("root_backend must be 'auto', 'jax', or 'ffi'")
    use_ffi = root_backend == "ffi" or (
        root_backend == "auto" and cpp_binary_image_roots_ffi_available()
    )
    image_function = binary_images_ffi if use_ffi else binary_images
    angles = (
        2.0
        * jnp.pi
        * jnp.arange(limb_samples, dtype=jnp.result_type(source_x, source_y))
        / limb_samples
    )
    centre = source_x + 1j * source_y
    limb = centre + source_radius * jnp.exp(1j * angles)
    sources = jnp.concatenate((jnp.reshape(centre, (1,)), limb))
    images = jax.vmap(lambda source: image_function(source, separation, mass_ratio))(
        sources
    )
    physical_counts = jnp.sum(images.physical, axis=1)
    # Exactly on a caustic the merging image pair is a repeated root, so the
    # deduplicated physical set legitimately contains four images.  The
    # finite-source limb samples still seed both adjacent image regions; do
    # not invalidate the whole support because an unused polynomial slot did
    # not converge at that measure-zero sample.
    valid_physical_count = (physical_counts >= 3) & (physical_counts <= 5)
    root_failure = jnp.any(~valid_physical_count)
    return ImageSeedPoints(
        roots=jax.lax.stop_gradient(images.roots.reshape(-1)),
        physical=jax.lax.stop_gradient(images.physical.reshape(-1)),
        residuals=jax.lax.stop_gradient(images.residuals.reshape(-1)),
        root_failure=jax.lax.stop_gradient(root_failure),
    )


def _tile_has_inside_probe(
    tile_index: jax.Array,
    tile_width: jax.Array,
    source_x: jax.Array,
    source_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    source_radius: jax.Array,
) -> jax.Array:
    fractions = jnp.asarray((0.0, 0.5, 1.0), dtype=tile_width.dtype)
    offset_x, offset_y = jnp.meshgrid(fractions, fractions, indexing="xy")
    origin = tile_index.astype(tile_width.dtype) * tile_width
    image_x = origin[0] + tile_width * offset_x.ravel()
    image_y = origin[1] + tile_width * offset_y.ravel()
    mapped_x, mapped_y = binary_lens_map_real(image_x, image_y, separation, mass_ratio)
    distance_squared = (mapped_x - source_x) * (mapped_x - source_x) + (
        mapped_y - source_y
    ) * (mapped_y - source_y)
    inside = jnp.isfinite(distance_squared) & (
        distance_squared <= source_radius * source_radius
    )
    return jnp.any(inside)


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "root_backend",
    ),
)
def discover_binary_macro_tiles(
    source_x: jax.Array,
    source_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    source_radius: jax.Array,
    cell_size: jax.Array,
    *,
    tile_size: int = 16,
    tile_capacity: int = 1024,
    limb_samples: int = 16,
    root_backend: str = "auto",
) -> DiscoveryResult:
    """Discover a one-tile halo around connected finite-source images."""

    source_x, source_y, separation, mass_ratio, source_radius, cell_size = (
        as_float64(value)
        for value in (
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            cell_size,
        )
    )
    frozen_cell_size = jax.lax.stop_gradient(cell_size)
    tile_width = jax.lax.stop_gradient(frozen_cell_size * tile_size)
    seeds = binary_image_seed_points(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_samples=limb_samples,
        root_backend=root_backend,
    )
    safe_seed_real = jnp.where(seeds.physical, jnp.real(seeds.roots), 0.0)
    safe_seed_imag = jnp.where(seeds.physical, jnp.imag(seeds.roots), 0.0)
    seed_indices = jnp.stack(
        (
            jnp.floor(safe_seed_real / tile_width).astype(jnp.int32),
            jnp.floor(safe_seed_imag / tile_width).astype(jnp.int32),
        ),
        axis=1,
    )

    empty_indices = jnp.zeros((tile_capacity, 2), dtype=jnp.int32)
    empty_mask = jnp.zeros(tile_capacity, dtype=bool)
    initial_state = (
        empty_indices,
        empty_mask,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )

    def insert_tile(state, candidate_and_valid):
        tile_indices, tile_mask, count, overflow = state
        candidate, valid = candidate_and_valid
        exists = jnp.any(
            tile_mask & jnp.all(tile_indices == candidate[None, :], axis=1)
        )
        novel = valid & ~exists
        has_capacity = count < tile_capacity
        should_insert = novel & has_capacity
        target = jnp.minimum(count, tile_capacity - 1)
        previous = tile_indices[target]
        tile_indices = tile_indices.at[target].set(
            jnp.where(should_insert, candidate, previous)
        )
        tile_mask = tile_mask.at[target].set(tile_mask[target] | should_insert)
        return (
            tile_indices,
            tile_mask,
            count + should_insert.astype(jnp.int32),
            overflow | (novel & ~has_capacity),
        )

    seed_state = jax.lax.fori_loop(
        0,
        seed_indices.shape[0],
        lambda index, state: insert_tile(
            state, (seed_indices[index], seeds.physical[index])
        ),
        initial_state,
    )
    seed_indices_unique, seed_mask, seed_count, seed_overflow = seed_state
    active_mask_0 = jnp.zeros(tile_capacity, dtype=bool)
    neighbours = jnp.asarray(((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=jnp.int32)

    bfs_initial = (
        seed_indices_unique,
        seed_mask,
        active_mask_0,
        seed_count,
        jnp.asarray(0, dtype=jnp.int32),
        seed_overflow,
    )

    def bfs_condition(state):
        _, _, _, count, head, _ = state
        return head < count

    def bfs_step(state):
        tile_indices, tile_mask, active_mask, count, head, overflow = state
        tile_index = tile_indices[head]
        is_seed = jnp.any(
            seed_mask & jnp.all(seed_indices_unique == tile_index[None, :], axis=1)
        )
        has_inside_probe = _tile_has_inside_probe(
            tile_index,
            tile_width,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
        )
        is_active = is_seed | has_inside_probe
        active_mask = active_mask.at[head].set(is_active)

        insertion_state = (tile_indices, tile_mask, count, overflow)
        insertion_state = jax.lax.fori_loop(
            0,
            neighbours.shape[0],
            lambda neighbour_index, insertion: insert_tile(
                insertion,
                (
                    tile_index + neighbours[neighbour_index],
                    is_active,
                ),
            ),
            insertion_state,
        )
        tile_indices, tile_mask, count, overflow = insertion_state
        return (
            tile_indices,
            tile_mask,
            active_mask,
            count,
            head + jnp.asarray(1, dtype=jnp.int32),
            overflow,
        )

    (
        tile_indices,
        tile_mask,
        active_mask,
        visited_count,
        _,
        overflow,
    ) = jax.lax.while_loop(bfs_condition, bfs_step, bfs_initial)
    tile_origins = tile_indices.astype(tile_width.dtype) * tile_width
    return jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        DiscoveryResult(
            tile_indices=tile_indices,
            tile_origins=tile_origins,
            tile_mask=tile_mask,
            active_mask=active_mask,
            overflow=overflow,
            visited_count=visited_count,
            active_count=jnp.sum(active_mask, dtype=jnp.int32),
            seed_count=seed_count,
            root_failure=seeds.root_failure,
        ),
    )

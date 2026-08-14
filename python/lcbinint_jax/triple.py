"""Differentiable dense and sparse CPU inverse rays for triple lenses."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._config import as_float64, require_x64
from .cell_moments import (
    resolved_cell_moments,
    resolved_cell_moments_linear,
    resolved_cell_moments_uniform,
)
from .cpp_backend import (
    cpp_triple_cartesian_epoch_ffi_available,
    triple_caustic_distance_batch_ffi,
    triple_hexadecapole_batch_ffi,
    triple_images_ffi,
    triple_inverse_ray_cartesian_batch_ffi,
    triple_inverse_ray_cartesian_ffi,
    triple_inverse_ray_polar_batch_ffi,
    triple_point_source_batch_ffi,
)
from .limb_darkening import combine_limb_darkening_moments
from .source_plane import triple_source_plane_quadrature


class TripleLensGeometry(NamedTuple):
    """Three lens positions and normalized masses."""

    positions: jax.Array
    masses: jax.Array


class TripleInverseRayResult(NamedTuple):
    """Dense-support triple-lens inverse-ray result."""

    magnification: jax.Array
    moments: jax.Array
    boundary_cells: jax.Array
    contributing_cells: jax.Array
    support_valid: jax.Array
    cell_size: jax.Array


class TripleDiscoveryResult(NamedTuple):
    tile_indices: jax.Array
    tile_origins: jax.Array
    tile_mask: jax.Array
    active_mask: jax.Array
    overflow: jax.Array
    root_failure: jax.Array
    visited_count: jax.Array
    active_count: jax.Array
    seed_count: jax.Array


class TripleAdaptiveInverseRayResult(NamedTuple):
    magnification: jax.Array
    moments: jax.Array
    boundary_cells: jax.Array
    contributing_cells: jax.Array
    support_valid: jax.Array
    cell_size: jax.Array
    visited_tiles: jax.Array
    active_tiles: jax.Array
    seed_tiles: jax.Array
    overflow: jax.Array
    root_failure: jax.Array


class TripleMagnificationResult(NamedTuple):
    magnification: jax.Array
    method: jax.Array
    estimated_error: jax.Array
    support_valid: jax.Array
    used_multipole: jax.Array
    root_failure: jax.Array
    discovery_overflow: jax.Array
    gradient_resolution: jax.Array
    gradient_extrapolated: jax.Array


def triple_lens_geometry(
    separation,
    mass_ratio,
    tertiary_mass_ratio,
    tertiary_separation,
    tertiary_angle,
    *,
    convention="center_of_mass",
):
    """Construct native-compatible triple-lens positions and masses.

    ``center_of_mass`` matches ``make_triple_lens_geometry``.  Here the first
    separation locates lens 1 relative to the center of mass of lenses 2+3,
    and ``tertiary_separation`` is the lens-2/lens-3 separation.

    ``vbm`` matches ``make_triple_lens_geometry_vbm`` and accepts VBM's
    primary-secondary and primary-tertiary separations.
    """

    q = as_float64(mass_ratio)
    q2 = as_float64(tertiary_mass_ratio)
    separation = as_float64(separation)
    tertiary_separation = as_float64(tertiary_separation)
    tertiary_angle = as_float64(tertiary_angle)
    total = 1.0 + q + q2
    masses = jnp.asarray((1.0 / total, q / total, q2 / total))
    if convention == "center_of_mass":
        epsilon1, epsilon2, epsilon3 = masses
        epsilon4 = epsilon2 + epsilon3
        lens1 = jnp.asarray((-epsilon4 * separation, 0.0))
        group = jnp.asarray((epsilon1 * separation, 0.0))
        delta = tertiary_separation * jnp.asarray(
            (jnp.cos(tertiary_angle), jnp.sin(tertiary_angle))
        )
        lens2 = group + epsilon3 / epsilon4 * delta
        lens3 = group - epsilon2 / epsilon4 * delta
        positions = jnp.stack((lens1, lens2, lens3))
    elif convention == "vbm":
        lens1_x = q * separation / (1.0 + q)
        lens2_x = -separation / (1.0 + q)
        lens1 = jnp.asarray((lens1_x, 0.0))
        lens2 = jnp.asarray((lens2_x, 0.0))
        lens3 = lens1 - tertiary_separation * jnp.asarray(
            (jnp.cos(tertiary_angle), jnp.sin(tertiary_angle))
        )
        positions = jnp.stack((lens1, lens2, lens3))
    else:
        raise ValueError("convention must be 'center_of_mass' or 'vbm'")
    return TripleLensGeometry(positions, masses)


def triple_lens_map_and_derivatives_real(image_x, image_y, geometry):
    """Map image-plane coordinates and return the two-by-two Jacobian."""

    displacement_x = image_x[..., None] - geometry.positions[:, 0]
    displacement_y = image_y[..., None] - geometry.positions[:, 1]
    radius_squared = (
        displacement_x * displacement_x + displacement_y * displacement_y
    )
    inverse_radius_squared = 1.0 / radius_squared
    weighted_inverse = geometry.masses * inverse_radius_squared
    source_x = image_x - jnp.sum(weighted_inverse * displacement_x, axis=-1)
    source_y = image_y - jnp.sum(weighted_inverse * displacement_y, axis=-1)
    inverse_radius_fourth = inverse_radius_squared * inverse_radius_squared
    shear_real = jnp.sum(
        geometry.masses
        * (displacement_x * displacement_x - displacement_y * displacement_y)
        * inverse_radius_fourth,
        axis=-1,
    )
    shear_cross = jnp.sum(
        2.0
        * geometry.masses
        * displacement_x
        * displacement_y
        * inverse_radius_fourth,
        axis=-1,
    )
    return (
        source_x,
        source_y,
        1.0 + shear_real,
        shear_cross,
        shear_cross,
        1.0 - shear_real,
    )


def triple_lens_map_real(image_x, image_y, geometry):
    """Map real image-plane coordinates to the source plane."""

    return triple_lens_map_and_derivatives_real(image_x, image_y, geometry)[:2]


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "convention",
    ),
)
def discover_triple_macro_tiles(
    source_x,
    source_y,
    separation,
    mass_ratio,
    tertiary_mass_ratio,
    tertiary_separation,
    tertiary_angle,
    source_radius,
    cell_size,
    *,
    tile_size=16,
    tile_capacity=2048,
    limb_samples=24,
    convention="center_of_mass",
):
    """Discover connected triple-image macro tiles from degree-10 roots."""

    (
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        source_radius,
        cell_size,
    ) = (
        as_float64(value)
        for value in (
            source_x,
            source_y,
            separation,
            mass_ratio,
            tertiary_mass_ratio,
            tertiary_separation,
            tertiary_angle,
            source_radius,
            cell_size,
        )
    )
    if tile_capacity <= 0:
        raise ValueError("tile_capacity must be positive")
    if limb_samples < 8:
        raise ValueError("limb_samples must be at least 8")
    geometry = triple_lens_geometry(
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        convention=convention,
    )
    dtype = jnp.result_type(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        source_radius,
    )
    angles = 2.0 * jnp.pi * jnp.arange(limb_samples, dtype=dtype) / limb_samples
    sample_x = jnp.concatenate(
        (
            jnp.reshape(jnp.asarray(source_x, dtype=dtype), (1,)),
            source_x + source_radius * jnp.cos(angles),
        )
    )
    sample_y = jnp.concatenate(
        (
            jnp.reshape(jnp.asarray(source_y, dtype=dtype), (1,)),
            source_y + source_radius * jnp.sin(angles),
        )
    )
    roots = jax.vmap(
        lambda x, y: triple_images_ffi(
            x,
            y,
            separation,
            mass_ratio,
            tertiary_mass_ratio,
            tertiary_separation,
            tertiary_angle,
            convention=convention,
        )
    )(sample_x, sample_y)
    physical_count = jnp.sum(roots.physical, axis=1)
    valid_count = (
        (physical_count == 4)
        | (physical_count == 6)
        | (physical_count == 8)
        | (physical_count == 10)
    )
    root_failure = jnp.any(
        ~valid_count | ~jnp.all(roots.converged | ~roots.physical, axis=1)
    )
    tile_width = jax.lax.stop_gradient(cell_size * tile_size)
    coordinates = roots.coordinates.reshape((-1, 2))
    physical = roots.physical.reshape((-1,))
    safe_coordinates = jnp.where(physical[:, None], coordinates, 0.0)
    seed_indices = jnp.floor(safe_coordinates / tile_width).astype(jnp.int32)
    empty_indices = jnp.zeros((tile_capacity, 2), dtype=jnp.int32)
    empty_mask = jnp.zeros(tile_capacity, dtype=jnp.bool_)

    def insert_tile(state, candidate_and_valid):
        indices, mask, count, overflow = state
        candidate, valid = candidate_and_valid
        exists = jnp.any(mask & jnp.all(indices == candidate[None, :], axis=1))
        novel = valid & ~exists
        has_capacity = count < tile_capacity
        insert = novel & has_capacity
        target = jnp.minimum(count, tile_capacity - 1)
        indices = indices.at[target].set(
            jnp.where(insert, candidate, indices[target])
        )
        mask = mask.at[target].set(mask[target] | insert)
        return (
            indices,
            mask,
            count + insert.astype(jnp.int32),
            overflow | (novel & ~has_capacity),
        )

    seed_state = jax.lax.fori_loop(
        0,
        seed_indices.shape[0],
        lambda index, state: insert_tile(
            state, (seed_indices[index], physical[index])
        ),
        (
            empty_indices,
            empty_mask,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        ),
    )
    seed_indices_unique, seed_mask, seed_count, seed_overflow = seed_state
    neighbours = jnp.asarray(((1, 0), (-1, 0), (0, 1), (0, -1)), jnp.int32)

    def tile_has_inside_probe(tile_index):
        """Bound ``min |f(z) - zeta|`` over the tile rather than sampling it.

        See ``lcbinint_jax.discovery._tile_has_inside_probe`` for the argument;
        the three lenses are not collinear here, so the tile-to-lens distance
        clamps in both coordinates.
        """

        half_width = 0.5 * tile_width
        centre = tile_index.astype(dtype) * tile_width + half_width
        offset = jnp.maximum(
            0.0, jnp.abs(centre[None, :] - geometry.positions) - half_width
        )
        distances_squared = jnp.sum(offset * offset, axis=1)
        contains_lens = jnp.any(distances_squared <= 0.0)

        mapped_x, mapped_y = triple_lens_map_real(centre[0], centre[1], geometry)
        distance = jnp.hypot(mapped_x - source_x, mapped_y - source_y)
        lipschitz = 1.0 + jnp.sum(
            geometry.masses / jnp.where(contains_lens, 1.0, distances_squared)
        )
        half_diagonal = half_width * jnp.sqrt(jnp.asarray(2.0, dtype=dtype))
        admissible = distance - lipschitz * half_diagonal <= source_radius
        return contains_lens | ~jnp.isfinite(distance) | admissible

    def condition(state):
        _, _, _, count, head, _ = state
        return head < count

    def step(state):
        indices, mask, active_mask, count, head, overflow = state
        tile_index = indices[head]
        is_seed = jnp.any(
            seed_mask
            & jnp.all(seed_indices_unique == tile_index[None, :], axis=1)
        )
        active = is_seed | tile_has_inside_probe(tile_index)
        active_mask = active_mask.at[head].set(active)
        insertion = (indices, mask, count, overflow)
        insertion = jax.lax.fori_loop(
            0,
            4,
            lambda neighbour, current: insert_tile(
                current,
                (tile_index + neighbours[neighbour], active),
            ),
            insertion,
        )
        indices, mask, count, overflow = insertion
        return indices, mask, active_mask, count, head + 1, overflow

    indices, mask, active_mask, visited, _, overflow = jax.lax.while_loop(
        condition,
        step,
        (
            seed_indices_unique,
            seed_mask,
            jnp.zeros(tile_capacity, dtype=jnp.bool_),
            seed_count,
            jnp.asarray(0, dtype=jnp.int32),
            seed_overflow,
        ),
    )
    result = TripleDiscoveryResult(
        indices,
        indices.astype(dtype) * tile_width,
        mask,
        active_mask,
        overflow,
        root_failure,
        visited,
        jnp.sum(active_mask, dtype=jnp.int32),
        seed_count,
    )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, result)


def _triple_phi_gradient_laplacian(
    image_x,
    image_y,
    source_x,
    source_y,
    source_radius,
    geometry,
):
    mapped_x, mapped_y, du_dx, du_dy, dv_dx, dv_dy = (
        triple_lens_map_and_derivatives_real(image_x, image_y, geometry)
    )
    residual_x = mapped_x - source_x
    residual_y = mapped_y - source_y
    inverse_radius_squared = 1.0 / (source_radius * source_radius)
    phi = 1.0 - (
        residual_x * residual_x + residual_y * residual_y
    ) * inverse_radius_squared
    gradient_x = -2.0 * inverse_radius_squared * (
        residual_x * du_dx + residual_y * dv_dx
    )
    gradient_y = -2.0 * inverse_radius_squared * (
        residual_x * du_dy + residual_y * dv_dy
    )
    laplacian = -2.0 * inverse_radius_squared * (
        du_dx * du_dx + du_dy * du_dy + dv_dx * dv_dx + dv_dy * dv_dy
    )
    return phi, gradient_x, gradient_y, laplacian


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "convention",
        "moment_mode",
        "boundary_subdivision",
    ),
)
def triple_inverse_ray_fixed_support(
    tile_origins,
    tile_mask,
    cell_size,
    source_x,
    source_y,
    separation,
    mass_ratio,
    tertiary_mass_ratio,
    tertiary_separation,
    tertiary_angle,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    tile_size=16,
    convention="center_of_mass",
    moment_mode="two_coefficient",
    boundary_subdivision=4,
):
    """Integrate a stopped-gradient, caller-supplied triple-image support."""

    require_x64()
    (
        tile_origins,
        cell_size,
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        source_radius,
        limb_c,
        limb_d,
    ) = (
        as_float64(value)
        for value in (
            tile_origins,
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            tertiary_mass_ratio,
            tertiary_separation,
            tertiary_angle,
            source_radius,
            limb_c,
            limb_d,
        )
    )
    if convention not in ("center_of_mass", "vbm"):
        raise ValueError("convention must be 'center_of_mass' or 'vbm'")
    if moment_mode == "uniform":
        resolved_moments = resolved_cell_moments_uniform
        moment_count = 1
    elif moment_mode == "linear":
        resolved_moments = resolved_cell_moments_linear
        moment_count = 2
    elif moment_mode == "two_coefficient":
        resolved_moments = resolved_cell_moments
        moment_count = 3
    else:
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )
    if boundary_subdivision not in (1, 2, 3, 4):
        raise ValueError("boundary_subdivision must be 1, 2, 3, or 4")

    geometry = triple_lens_geometry(
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        convention=convention,
    )
    frozen_origins = jax.lax.stop_gradient(tile_origins)
    frozen_mask = jax.lax.stop_gradient(tile_mask)
    frozen_cell_size = jax.lax.stop_gradient(cell_size)
    one_dimensional = (
        jnp.arange(tile_size, dtype=frozen_cell_size.dtype) + 0.5
    ) * frozen_cell_size
    offset_x, offset_y = jnp.meshgrid(
        one_dimensional, one_dimensional, indexing="xy"
    )
    offset_x = offset_x.ravel()
    offset_y = offset_y.ravel()
    cells_per_tile = tile_size * tile_size

    def phi_kernel(x, y):
        return _triple_phi_gradient_laplacian(
            x,
            y,
            source_x,
            source_y,
            source_radius,
            geometry,
        )

    def second_order_interior_moments(phi, gradient_x, gradient_y, laplacian):
        safe_phi = jnp.where(phi > 0.0, phi, 1.0)
        area = frozen_cell_size * frozen_cell_size
        moment_0 = jnp.full_like(phi, area)
        if moment_mode == "uniform":
            return moment_0[:, None]
        sqrt_phi = jnp.sqrt(safe_phi)
        gradient_term = frozen_cell_size * frozen_cell_size * (
            gradient_x * gradient_x + gradient_y * gradient_y
        )
        laplacian_term = frozen_cell_size * frozen_cell_size * laplacian
        moment_half = area * (
            sqrt_phi
            + laplacian_term / (48.0 * sqrt_phi)
            - gradient_term / (96.0 * safe_phi * sqrt_phi)
        )
        if moment_mode == "linear":
            return jnp.stack((moment_0, moment_half), axis=1)
        fourth_root = jnp.sqrt(sqrt_phi)
        moment_quarter = area * (
            fourth_root
            + laplacian_term / (96.0 * sqrt_phi * fourth_root)
            - gradient_term
            / (128.0 * safe_phi * sqrt_phi * fourth_root)
        )
        return jnp.stack((moment_0, moment_half, moment_quarter), axis=1)

    def inactive_tile(_):
        return (
            jnp.zeros(moment_count, dtype=frozen_cell_size.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def integrate_tile(carry, inputs):
        origin, active = inputs

        def active_tile(active_origin):
            image_x = active_origin[0] + offset_x
            image_y = active_origin[1] + offset_y
            phi, gradient_x, gradient_y, laplacian = jax.vmap(phi_kernel)(
                image_x, image_y
            )
            extent = 0.5 * frozen_cell_size * (
                jnp.abs(gradient_x) + jnp.abs(gradient_y)
            )
            fully_inside = phi - extent > 0.0
            fully_outside = phi + extent <= 0.0
            detailed = ~(fully_inside | fully_outside)
            if moment_mode == "two_coefficient":
                relative_variation = (
                    extent
                    + 0.125
                    * jnp.abs(laplacian)
                    * frozen_cell_size
                    * frozen_cell_size
                ) / jnp.maximum(phi, 1.0e-30)
                detailed = detailed | (
                    jax.lax.stop_gradient(limb_d != 0.0)
                    & fully_inside
                    & (relative_variation > 0.2)
                )
            bulk_inside = fully_inside & ~detailed
            interior = second_order_interior_moments(
                phi, gradient_x, gradient_y, laplacian
            )
            interior = jnp.where(
                bulk_inside[:, None], interior, jnp.zeros_like(interior)
            )
            detailed_count = jnp.sum(detailed, dtype=jnp.int32)

            def integrate_detailed(_):
                indices = jnp.nonzero(
                    detailed, size=cells_per_tile, fill_value=0
                )[0]
                slot_mask = jnp.arange(cells_per_tile) < detailed_count
                subdivision = boundary_subdivision
                subcell_size = frozen_cell_size / subdivision
                suboffset = (
                    (jnp.arange(subdivision, dtype=frozen_cell_size.dtype) + 0.5)
                    / subdivision
                    - 0.5
                ) * frozen_cell_size
                suboffset_x, suboffset_y = jnp.meshgrid(
                    suboffset, suboffset, indexing="xy"
                )
                suboffset_x = suboffset_x.ravel()
                suboffset_y = suboffset_y.ravel()

                def integrate_cell(index):
                    sub_x = image_x[index] + suboffset_x
                    sub_y = image_y[index] + suboffset_y
                    sub_phi, sub_dx, sub_dy, _ = jax.vmap(phi_kernel)(
                        sub_x, sub_y
                    )
                    sub_moments, _, _ = jax.vmap(
                        lambda value, dx, dy: resolved_moments(
                            value, dx, dy, subcell_size
                        )
                    )(sub_phi, sub_dx, sub_dy)
                    return jnp.sum(sub_moments, axis=0)

                packed = jax.vmap(integrate_cell)(indices)
                packed = jnp.where(
                    slot_mask[:, None], packed, jnp.zeros_like(packed)
                )
                return jnp.sum(packed, axis=0)

            detailed_moments = jax.lax.cond(
                detailed_count > 0,
                integrate_detailed,
                lambda _: jnp.zeros(moment_count, dtype=phi.dtype),
                operand=None,
            )
            return (
                jnp.sum(interior, axis=0) + detailed_moments,
                detailed_count,
                jnp.sum(bulk_inside, dtype=jnp.int32) + detailed_count,
            )

        tile_moments, tile_boundary, tile_contributing = jax.lax.cond(
            active,
            jax.checkpoint(active_tile),
            inactive_tile,
            origin,
        )
        moments, boundary_cells, contributing_cells = carry
        return (
            moments + tile_moments,
            boundary_cells + tile_boundary,
            contributing_cells + tile_contributing,
        ), None

    initial = (
        jnp.zeros(
            moment_count,
            dtype=jnp.result_type(
                tile_origins,
                source_radius,
                separation,
                tertiary_separation,
            ),
        ),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    (moments, boundary_cells, contributing_cells), _ = jax.lax.scan(
        integrate_tile, initial, (frozen_origins, frozen_mask)
    )
    if moment_mode == "uniform":
        full_moments = jnp.stack(
            (moments[0], moments[0] * 2.0 / 3.0, moments[0] * 4.0 / 5.0)
        )
    elif moment_mode == "linear":
        full_moments = jnp.stack(
            (moments[0], moments[1], moments[0] * 4.0 / 5.0)
        )
    else:
        full_moments = moments
    magnification = combine_limb_darkening_moments(
        full_moments, source_radius, limb_c, limb_d
    )
    return TripleInverseRayResult(
        magnification,
        full_moments,
        boundary_cells,
        contributing_cells,
        jnp.asarray(True),
        cell_size,
    )


@partial(
    jax.jit,
    static_argnames=(
        "resolution",
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "convention",
        "moment_mode",
        "boundary_subdivision",
        "use_ffi",
    ),
)
def triple_inverse_ray_adaptive(
    source_x,
    source_y,
    separation,
    mass_ratio,
    tertiary_mass_ratio,
    tertiary_separation,
    tertiary_angle,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    resolution=64,
    tile_size=8,
    tile_capacity=131072,
    limb_samples=12,
    convention="center_of_mass",
    moment_mode="two_coefficient",
    boundary_subdivision=4,
    use_ffi=True,
):
    """Discover and integrate sparse triple-image support."""

    if resolution < 8:
        raise ValueError("resolution must be at least 8")
    cell_size = jax.lax.stop_gradient(
        jnp.asarray(source_radius) / jnp.asarray(resolution)
    )
    if use_ffi and cpp_triple_cartesian_epoch_ffi_available():
        integrated = triple_inverse_ray_cartesian_ffi(
            source_x,
            source_y,
            separation,
            mass_ratio,
            tertiary_mass_ratio,
            tertiary_separation,
            tertiary_angle,
            source_radius,
            limb_c,
            limb_d,
            cell_size=cell_size,
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            convention=convention,
            moment_mode=moment_mode,
            boundary_subdivision=boundary_subdivision,
        )
        if moment_mode == "uniform":
            full_moments = jnp.stack(
                (
                    integrated.moments[0],
                    integrated.moments[0] * 2.0 / 3.0,
                    integrated.moments[0] * 4.0 / 5.0,
                )
            )
        elif moment_mode == "linear":
            full_moments = jnp.stack(
                (
                    integrated.moments[0],
                    integrated.moments[1],
                    integrated.moments[0] * 4.0 / 5.0,
                )
            )
        else:
            full_moments = integrated.moments
        return TripleAdaptiveInverseRayResult(
            integrated.magnification,
            full_moments,
            integrated.boundary_cells,
            integrated.active_cells,
            integrated.support_valid,
            cell_size,
            integrated.tile_count,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            integrated.discovery_overflow,
            integrated.root_failure,
        )
    discovery = discover_triple_macro_tiles(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        source_radius,
        cell_size,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        convention=convention,
    )
    integrated = triple_inverse_ray_fixed_support(
        discovery.tile_origins,
        discovery.active_mask,
        cell_size,
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        source_radius,
        limb_c,
        limb_d,
        tile_size=tile_size,
        convention=convention,
        moment_mode=moment_mode,
        boundary_subdivision=boundary_subdivision,
    )
    support_valid = ~(discovery.overflow | discovery.root_failure)
    return TripleAdaptiveInverseRayResult(
        integrated.magnification,
        integrated.moments,
        integrated.boundary_cells,
        integrated.contributing_cells,
        support_valid,
        cell_size,
        discovery.visited_count,
        discovery.active_count,
        discovery.seed_count,
        discovery.overflow,
        discovery.root_failure,
    )


@partial(
    jax.jit,
    static_argnames=(
        "resolution",
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "convention",
        "moment_mode",
        "boundary_subdivision",
    ),
)
def triple_inverse_ray_batch(
    source_x,
    source_y,
    separation,
    mass_ratio,
    tertiary_mass_ratio,
    tertiary_separation,
    tertiary_angle,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    active=None,
    resolution=64,
    tile_size=8,
    tile_capacity=131072,
    limb_samples=12,
    convention="center_of_mass",
    moment_mode="two_coefficient",
    boundary_subdivision=4,
):
    """Evaluate a one-dimensional triple trajectory in one parallel CPU FFI."""

    if resolution < 8:
        raise ValueError("resolution must be at least 8")
    return triple_inverse_ray_cartesian_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        source_radius,
        limb_c,
        limb_d,
        active=active,
        cell_size=jax.lax.stop_gradient(source_radius / resolution),
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        convention=convention,
        moment_mode=moment_mode,
        boundary_subdivision=boundary_subdivision,
    )


@partial(
    jax.jit,
    static_argnames=(
        "resolution",
        "polar_resolution",
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "polar_limb_samples",
        "convention",
        "moment_mode",
        "boundary_subdivision",
    ),
)
def triple_magnification_batch(
    source_x,
    source_y,
    separation,
    mass_ratio,
    tertiary_mass_ratio,
    tertiary_separation,
    tertiary_angle,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    absolute_tolerance=0.0,
    relative_tolerance=1.0e-4,
    point_safety_factor=4.0,
    multipole_safety_factor=1.0,
    multipole_max_magnification=100.0,
    resolution=96,
    polar_resolution=64,
    tile_size=8,
    tile_capacity=131072,
    limb_samples=12,
    polar_limb_samples=64,
    convention="center_of_mass",
    moment_mode="two_coefficient",
    boundary_subdivision=4,
):
    """Dispatch a triple trajectory across native-calibrated finite-source paths.

    Multipoles are deliberately disabled above
    ``multipole_max_magnification``.  Near a caustic the local series can look
    well ordered while still missing a nearby singularity.  Caustic-clear
    high-magnification rows use polar inverse rays; all other finite rows use
    Cartesian inverse rays. Failed image-plane support is reported as
    unsupported rather than falling back to source-plane quadrature.
    """

    require_x64()
    source_x = jnp.asarray(source_x)
    source_y = jnp.asarray(source_y)
    if source_x.ndim != 1 or source_y.shape != source_x.shape:
        raise ValueError("source_x and source_y must have the same 1-D shape")
    point_source = triple_point_source_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        convention=convention,
    )
    point_magnitude = jnp.maximum(
        jnp.abs(point_source.magnification), 1.0
    )
    point_budget = (
        absolute_tolerance + relative_tolerance * point_magnitude
    )
    point_error = (
        point_source.derivative_indicator * source_radius * source_radius
    )
    point_source_valid = (
        ~point_source.root_failure
        & jnp.isfinite(point_source.magnification)
    )
    zero_radius = source_radius == 0.0
    inverse_ray_radius = jnp.where(source_radius > 0.0, source_radius, 1.0)
    caustic_distance = triple_caustic_distance_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        inverse_ray_radius,
        convention=convention,
    )
    point_caustic_safe = (
        jnp.isfinite(caustic_distance)
        & (caustic_distance >= 20.0 * inverse_ray_radius)
    )
    multipole_magnification_safe = (
        point_magnitude < multipole_max_magnification
    )
    accept_point = jax.lax.stop_gradient(
        point_source_valid
        & (
            zero_radius
            | (
                (source_radius > 0.0)
                & point_caustic_safe
                & multipole_magnification_safe
                & jnp.isfinite(point_error)
                & (point_safety_factor * point_error <= point_budget)
            )
        )
    )
    hexadecapole = triple_hexadecapole_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        source_radius,
        limb_c,
        limb_d,
        active=~accept_point,
        convention=convention,
    )
    hex_magnitude = jnp.maximum(
        jnp.abs(hexadecapole.magnification), 1.0
    )
    budget = absolute_tolerance + relative_tolerance * hex_magnitude
    correction_scale = jnp.maximum(
        jnp.abs(hexadecapole.quadrupole_correction), budget
    )
    expansion_well_ordered = (
        hexadecapole.estimated_error <= 0.25 * correction_scale
    ) & (
        jnp.abs(hexadecapole.quadrupole_correction)
        <= 0.1 * hex_magnitude
    )
    accept_hexadecapole = jax.lax.stop_gradient(
        ~accept_point
        & multipole_magnification_safe
        & (source_radius > 0.0)
        & jnp.isfinite(caustic_distance)
        & (caustic_distance >= 5.0 * inverse_ray_radius)
        & hexadecapole.topology_stable
        & ~hexadecapole.root_failure
        & jnp.isfinite(hexadecapole.magnification)
        & expansion_well_ordered
        & (
            multipole_safety_factor * hexadecapole.estimated_error
            <= budget
        )
    )
    needs_finite_integration = ~(accept_point | accept_hexadecapole)
    accept_polar_far = jax.lax.stop_gradient(
        needs_finite_integration
        & (point_magnitude >= 100.0)
        & jnp.isfinite(caustic_distance)
        & (caustic_distance >= 3.0 * inverse_ray_radius)
    )
    # In the inner three-rho band, high-magnification Cartesian macro-tile
    # support grows along extremely long, sub-cell-thin fold arcs and can hit
    # its capacity.  Seed-complete polar is calibrated for the deeply
    # crossing (d < 0.8 rho) or extreme-magnification tail.  The narrow
    # limb-contact band stays Cartesian: polar can over-connect a tiny fold
    # component there.
    accept_polar_near = jax.lax.stop_gradient(
        needs_finite_integration
        & (point_magnitude < 1.0e8)
        & jnp.isfinite(caustic_distance)
        & (caustic_distance < 3.0 * inverse_ray_radius)
        & (
            (point_magnitude >= 1.0e4)
            | (
                (point_magnitude >= 100.0)
                & (caustic_distance < 0.8 * inverse_ray_radius)
            )
        )
    )
    accept_polar_ultra = jax.lax.stop_gradient(
        needs_finite_integration
        & (point_magnitude >= 1.0e8)
        & jnp.isfinite(caustic_distance)
        & (caustic_distance < 3.0 * inverse_ray_radius)
    )
    accept_polar = accept_polar_far | accept_polar_near | accept_polar_ultra
    polar_far = triple_inverse_ray_polar_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        inverse_ray_radius,
        limb_c,
        limb_d,
        active=accept_polar_far,
        resolution=polar_resolution,
        angular_bins=0,
        limb_samples=polar_limb_samples,
        convention=convention,
        moment_mode=moment_mode,
    )
    polar_near = triple_inverse_ray_polar_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        inverse_ray_radius,
        limb_c,
        limb_d,
        active=accept_polar_near,
        resolution=polar_resolution,
        angular_bins=0,
        limb_samples=polar_limb_samples,
        convention=convention,
        moment_mode=moment_mode,
        gradient_backend="image_plane",
    )
    polar_ultra = triple_inverse_ray_polar_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        inverse_ray_radius,
        limb_c,
        limb_d,
        active=accept_polar_ultra,
        resolution=32,
        angular_bins=0,
        limb_samples=polar_limb_samples,
        convention=convention,
        moment_mode=moment_mode,
        gradient_backend="image_plane",
    )
    polar_magnification = jnp.where(
        accept_polar_far,
        polar_far.magnification,
        jnp.where(
            accept_polar_near,
            polar_near.magnification,
            polar_ultra.magnification,
        ),
    )
    polar_support_valid = jnp.where(
        accept_polar_far,
        polar_far.support_valid,
        jnp.where(
            accept_polar_near,
            polar_near.support_valid,
            polar_ultra.support_valid,
        ),
    )
    polar_root_failure = jnp.where(
        accept_polar_far,
        polar_far.root_failure,
        jnp.where(
            accept_polar_near,
            polar_near.root_failure,
            polar_ultra.root_failure,
        ),
    )
    polar_overflow = jnp.where(
        accept_polar_far,
        polar_far.discovery_overflow,
        jnp.where(
            accept_polar_near,
            polar_near.discovery_overflow,
            polar_ultra.discovery_overflow,
        ),
    )
    polar_active_cells = jnp.where(
        accept_polar_far,
        polar_far.active_cells,
        jnp.where(
            accept_polar_near,
            polar_near.active_cells,
            polar_ultra.active_cells,
        ),
    )
    # Source-plane quadrature is intentionally excluded from the automatic
    # triple dispatcher.  Grazing rows continue to the image-plane Cartesian
    # route below (or remain unsupported if its certificate fails).
    grazing_candidate = jnp.zeros_like(needs_finite_integration)

    def source_plane_rule(
        source_x_value,
        source_y_value,
        rule,
        coarse_order,
        fine_order,
    ):
        return triple_source_plane_quadrature(
            source_x_value,
            source_y_value,
            separation,
            mass_ratio,
            tertiary_mass_ratio,
            tertiary_separation,
            tertiary_angle,
            inverse_ray_radius,
            limb_c,
            limb_d,
            rule=rule,
            coarse_order=coarse_order,
            fine_order=fine_order,
            angular_multiplier=4,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            convention=convention,
        )

    def evaluate_source_plane(_):
        def evaluate_one(values, rule):
            x, y, is_active = values

            def run(_):
                result = source_plane_rule(x, y, rule, 32, 64)
                return result.magnification, result.root_failure

            return jax.lax.cond(
                is_active,
                run,
                lambda _: (
                    jnp.asarray(0.0, dtype=source_x.dtype),
                    jnp.asarray(False),
                ),
                operand=None,
            )

        ring_magnification_value, ring_failure_value = jax.lax.map(
            lambda values: evaluate_one(values, "ring"),
            (source_x, source_y, grazing_candidate),
        )
        chord_magnification_value, chord_failure_value = jax.lax.map(
            lambda values: evaluate_one(values, "chord"),
            (source_x, source_y, grazing_candidate),
        )
        return (
            ring_magnification_value,
            ring_failure_value,
            chord_magnification_value,
            chord_failure_value,
        )

    def skip_source_plane(_):
        return (
            jnp.zeros_like(source_x),
            jnp.zeros_like(source_x, dtype=jnp.bool_),
            jnp.zeros_like(source_x),
            jnp.zeros_like(source_x, dtype=jnp.bool_),
        )

    (
        ring_magnification,
        ring_root_failure,
        chord_magnification,
        chord_root_failure,
    ) = jax.lax.cond(
        jnp.any(grazing_candidate),
        evaluate_source_plane,
        skip_source_plane,
        operand=None,
    )
    source_plane_difference = jnp.abs(
        ring_magnification - chord_magnification
    )
    source_plane_budget = (
        absolute_tolerance
        + relative_tolerance
        * jnp.maximum(jnp.abs(chord_magnification), 1.0)
    )
    accept_source_plane_low = jax.lax.stop_gradient(
        grazing_candidate
        & ~ring_root_failure
        & ~chord_root_failure
        & jnp.isfinite(ring_magnification)
        & jnp.isfinite(chord_magnification)
        & (40.0 * source_plane_difference <= source_plane_budget)
    )
    needs_source_plane_escalation = jax.lax.stop_gradient(
        grazing_candidate
        & ~accept_source_plane_low
        & ~ring_root_failure
        & ~chord_root_failure
        & jnp.isfinite(ring_magnification)
        & jnp.isfinite(chord_magnification)
    )

    def evaluate_chord_160_256(_):
        def evaluate_one(values):
            x, y, is_active = values

            def run(_):
                result = source_plane_rule(x, y, "chord", 160, 256)
                return (
                    result.magnification,
                    result.estimated_error,
                    result.root_failure,
                    result.converged,
                )

            return jax.lax.cond(
                is_active,
                run,
                lambda _: (
                    jnp.asarray(0.0, dtype=source_x.dtype),
                    jnp.asarray(jnp.inf, dtype=source_x.dtype),
                    jnp.asarray(False),
                    jnp.asarray(False),
                ),
                operand=None,
            )

        return jax.lax.map(
            evaluate_one,
            (source_x, source_y, needs_source_plane_escalation),
        )

    def skip_escalated_source_plane(_):
        return (
            jnp.zeros_like(source_x),
            jnp.full_like(source_x, jnp.inf),
            jnp.zeros_like(source_x, dtype=jnp.bool_),
            jnp.zeros_like(source_x, dtype=jnp.bool_),
        )

    (
        medium_magnification,
        medium_error,
        medium_root_failure,
        medium_converged,
    ) = jax.lax.cond(
        jnp.any(needs_source_plane_escalation),
        evaluate_chord_160_256,
        skip_escalated_source_plane,
        operand=None,
    )
    accept_source_plane_medium = jax.lax.stop_gradient(
        needs_source_plane_escalation
        & ~medium_root_failure
        & jnp.isfinite(medium_magnification)
        & medium_converged
    )
    needs_source_plane_tail = jax.lax.stop_gradient(
        needs_source_plane_escalation
        & ~medium_root_failure
        & jnp.isfinite(medium_magnification)
        & ~medium_converged
    )

    def evaluate_chord_400_512(_):
        def evaluate_one(values):
            x, y, is_active = values

            def run(_):
                result = source_plane_rule(x, y, "chord", 400, 512)
                return (
                    result.magnification,
                    result.estimated_error,
                    result.root_failure,
                    result.converged,
                )

            return jax.lax.cond(
                is_active,
                run,
                lambda _: (
                    jnp.asarray(0.0, dtype=source_x.dtype),
                    jnp.asarray(jnp.inf, dtype=source_x.dtype),
                    jnp.asarray(False),
                    jnp.asarray(False),
                ),
                operand=None,
            )

        return jax.lax.map(
            evaluate_one,
            (source_x, source_y, needs_source_plane_tail),
        )

    (
        tail_magnification,
        tail_error,
        tail_root_failure,
        tail_converged,
    ) = jax.lax.cond(
        jnp.any(needs_source_plane_tail),
        evaluate_chord_400_512,
        skip_escalated_source_plane,
        operand=None,
    )
    select_source_plane_tail = jax.lax.stop_gradient(
        needs_source_plane_tail
        & ~tail_root_failure
        & jnp.isfinite(tail_magnification)
    )
    select_source_plane = (
        accept_source_plane_low
        | accept_source_plane_medium
        | select_source_plane_tail
    )
    source_plane_magnification = jnp.where(
        accept_source_plane_low,
        chord_magnification,
        jnp.where(
            accept_source_plane_medium,
            medium_magnification,
            tail_magnification,
        ),
    )
    selected_source_plane_error = jnp.where(
        accept_source_plane_low,
        source_plane_difference,
        jnp.where(
            accept_source_plane_medium,
            medium_error,
            tail_error,
        ),
    )
    source_plane_converged = (
        accept_source_plane_low
        | accept_source_plane_medium
        | (select_source_plane_tail & tail_converged)
    )
    accept_noncartesian = (
        accept_point
        | accept_hexadecapole
        | accept_polar
        | select_source_plane
    )
    inverse_ray = triple_inverse_ray_cartesian_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        inverse_ray_radius,
        limb_c,
        limb_d,
        active=~accept_noncartesian,
        cell_size=jax.lax.stop_gradient(inverse_ray_radius / resolution),
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        convention=convention,
        moment_mode=moment_mode,
        boundary_subdivision=boundary_subdivision,
    )
    polar_recovery_candidate = jax.lax.stop_gradient(
        ~accept_noncartesian & ~inverse_ray.support_valid
    )
    polar_recovery_coarse = triple_inverse_ray_polar_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        inverse_ray_radius,
        limb_c,
        limb_d,
        active=polar_recovery_candidate,
        resolution=48,
        angular_bins=0,
        limb_samples=polar_limb_samples,
        convention=convention,
        moment_mode=moment_mode,
        gradient_backend="image_plane",
    )
    polar_recovery_fine = triple_inverse_ray_polar_batch_ffi(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        inverse_ray_radius,
        limb_c,
        limb_d,
        active=polar_recovery_candidate,
        resolution=64,
        angular_bins=0,
        limb_samples=polar_limb_samples,
        convention=convention,
        moment_mode=moment_mode,
        gradient_backend="image_plane",
    )
    polar_recovery_error = jnp.abs(
        polar_recovery_fine.magnification
        - polar_recovery_coarse.magnification
    )
    polar_recovery_budget = (
        absolute_tolerance
        + relative_tolerance
        * jnp.maximum(
            jnp.abs(polar_recovery_fine.magnification), 1.0
        )
    )
    accept_polar_recovery = jax.lax.stop_gradient(
        polar_recovery_candidate
        & polar_recovery_coarse.support_valid
        & polar_recovery_fine.support_valid
        & jnp.isfinite(polar_recovery_error)
        & (polar_recovery_error <= polar_recovery_budget)
    )
    image_plane_polar_gradient = (
        accept_polar_near | accept_polar_ultra | accept_polar_recovery
    )
    gradient_active_cells = jnp.where(
        accept_polar_recovery,
        polar_recovery_fine.active_cells,
        polar_active_cells,
    )
    gradient_primal_resolution = jnp.where(
        accept_polar_ultra,
        jnp.asarray(32, dtype=jnp.int32),
        jnp.where(
            accept_polar_recovery,
            jnp.asarray(64, dtype=jnp.int32),
            jnp.asarray(polar_resolution, dtype=jnp.int32),
        ),
    )
    gradient_high_fits = (
        gradient_active_cells
        * (256.0 / gradient_primal_resolution) ** 2
        <= 30_000_000
    )
    gradient_medium_fits = (
        gradient_active_cells
        * (128.0 / gradient_primal_resolution) ** 2
        <= 30_000_000
    )
    gradient_resolution = jnp.where(
        image_plane_polar_gradient,
        jnp.where(
            gradient_high_fits,
            jnp.asarray(256, dtype=jnp.int32),
            jnp.where(
                gradient_medium_fits,
                jnp.asarray(128, dtype=jnp.int32),
                gradient_primal_resolution,
            ),
        ),
        jnp.asarray(0, dtype=jnp.int32),
    )
    gradient_extrapolated = (
        image_plane_polar_gradient
        & ~gradient_high_fits
        & ~gradient_medium_fits
    )
    magnification = jnp.where(
        accept_point,
        point_source.magnification,
        jnp.where(
            accept_hexadecapole,
            hexadecapole.magnification,
            jnp.where(
                accept_polar,
                polar_magnification,
                jnp.where(
                    select_source_plane,
                    source_plane_magnification,
                    jnp.where(
                        accept_polar_recovery,
                        polar_recovery_fine.magnification,
                        inverse_ray.magnification,
                    ),
                ),
            ),
        ),
    )
    accept_multipole = accept_point | accept_hexadecapole
    support_valid = (
        accept_multipole
        | (accept_polar & polar_support_valid)
        | source_plane_converged
        | accept_polar_recovery
        | inverse_ray.support_valid
    )
    root_failure = jnp.where(
        accept_point,
        point_source.root_failure,
        jnp.where(
            accept_hexadecapole,
            hexadecapole.root_failure,
            jnp.where(
                accept_polar,
                polar_root_failure,
                jnp.where(
                    select_source_plane,
                    (
                        (accept_source_plane_low
                         & (chord_root_failure | ring_root_failure))
                        | (accept_source_plane_medium & medium_root_failure)
                        | (select_source_plane_tail & tail_root_failure)
                    ),
                    jnp.where(
                        accept_polar_recovery,
                        polar_recovery_fine.root_failure,
                        inverse_ray.root_failure,
                    ),
                ),
            ),
        ),
    )
    return TripleMagnificationResult(
        magnification=magnification,
        method=jnp.where(
            accept_point,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                accept_hexadecapole,
                jnp.asarray(1, dtype=jnp.int32),
                jnp.where(
                    accept_polar,
                    jnp.asarray(3, dtype=jnp.int32),
                    jnp.where(
                        select_source_plane,
                        jnp.asarray(4, dtype=jnp.int32),
                        jnp.where(
                            accept_polar_recovery,
                            jnp.asarray(3, dtype=jnp.int32),
                            jnp.asarray(2, dtype=jnp.int32),
                        ),
                    ),
                ),
            ),
        ),
        estimated_error=jnp.where(
            accept_point,
            point_error,
            jnp.where(
                accept_hexadecapole,
                hexadecapole.estimated_error,
                jnp.where(
                    select_source_plane,
                    selected_source_plane_error,
                    jnp.where(
                        accept_polar_recovery,
                        polar_recovery_error,
                        jnp.asarray(jnp.nan, dtype=magnification.dtype),
                    ),
                ),
            ),
        ),
        support_valid=support_valid,
        used_multipole=accept_multipole,
        root_failure=root_failure,
        discovery_overflow=jnp.where(
            accept_polar,
            polar_overflow,
            jnp.where(
                accept_polar_recovery,
                polar_recovery_fine.discovery_overflow,
                inverse_ray.discovery_overflow,
            ),
        ),
        gradient_resolution=gradient_resolution,
        gradient_extrapolated=gradient_extrapolated,
    )


@partial(
    jax.jit,
    static_argnames=(
        "resolution",
        "polar_resolution",
        "tile_size",
        "tile_capacity",
        "limb_samples",
        "polar_limb_samples",
        "convention",
        "moment_mode",
        "boundary_subdivision",
    ),
)
def triple_magnification_auto(
    source_x,
    source_y,
    separation,
    mass_ratio,
    tertiary_mass_ratio,
    tertiary_separation,
    tertiary_angle,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    absolute_tolerance=0.0,
    relative_tolerance=1.0e-4,
    point_safety_factor=4.0,
    multipole_safety_factor=1.0,
    multipole_max_magnification=100.0,
    resolution=96,
    polar_resolution=64,
    tile_size=8,
    tile_capacity=131072,
    limb_samples=12,
    polar_limb_samples=64,
    convention="center_of_mass",
    moment_mode="two_coefficient",
    boundary_subdivision=4,
):
    """Scalar form of :func:`triple_magnification_batch`."""

    result = triple_magnification_batch(
        jnp.reshape(jnp.asarray(source_x), (1,)),
        jnp.reshape(jnp.asarray(source_y), (1,)),
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        source_radius,
        limb_c,
        limb_d,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        point_safety_factor=point_safety_factor,
        multipole_safety_factor=multipole_safety_factor,
        multipole_max_magnification=multipole_max_magnification,
        resolution=resolution,
        polar_resolution=polar_resolution,
        tile_size=tile_size,
        tile_capacity=tile_capacity,
        limb_samples=limb_samples,
        polar_limb_samples=polar_limb_samples,
        convention=convention,
        moment_mode=moment_mode,
        boundary_subdivision=boundary_subdivision,
    )
    return jax.tree_util.tree_map(lambda value: value[0], result)


@partial(
    jax.jit,
    static_argnames=("resolution", "image_extent", "convention", "moment_mode"),
)
def triple_inverse_ray_dense(
    source_x,
    source_y,
    separation,
    mass_ratio,
    tertiary_mass_ratio,
    tertiary_separation,
    tertiary_angle,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    resolution=1024,
    image_extent=3.0,
    convention="center_of_mass",
    moment_mode="two_coefficient",
):
    """Integrate a triple lens on one explicit square image-plane support.

    This is the first correctness-oriented triple-lens inverse-ray kernel.  It
    deliberately uses dense support so no degree-10 root or topology decision
    is hidden in the differentiable path.  ``support_valid`` is false when a
    contributing source preimage touches the square boundary; callers must
    then enlarge ``image_extent``.
    """

    require_x64()
    (
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        source_radius,
        limb_c,
        limb_d,
    ) = (
        as_float64(value)
        for value in (
            source_x,
            source_y,
            separation,
            mass_ratio,
            tertiary_mass_ratio,
            tertiary_separation,
            tertiary_angle,
            source_radius,
            limb_c,
            limb_d,
        )
    )
    if resolution < 16:
        raise ValueError("resolution must be at least 16")
    if image_extent <= 0.0:
        raise ValueError("image_extent must be positive")
    if convention not in ("center_of_mass", "vbm"):
        raise ValueError("convention must be 'center_of_mass' or 'vbm'")
    if moment_mode == "uniform":
        resolved_moments = resolved_cell_moments_uniform
        moment_count = 1
    elif moment_mode == "linear":
        resolved_moments = resolved_cell_moments_linear
        moment_count = 2
    elif moment_mode == "two_coefficient":
        resolved_moments = resolved_cell_moments
        moment_count = 3
    else:
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )

    geometry = triple_lens_geometry(
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        tertiary_separation,
        tertiary_angle,
        convention=convention,
    )
    dtype = jnp.result_type(
        source_x,
        source_y,
        separation,
        mass_ratio,
        tertiary_mass_ratio,
        source_radius,
    )
    cell_size = jnp.asarray(2.0 * image_extent / resolution, dtype=dtype)
    frozen_cell_size = jax.lax.stop_gradient(cell_size)
    coordinate = (
        jnp.arange(resolution, dtype=dtype) + 0.5
    ) * frozen_cell_size - image_extent
    inverse_source_radius_squared = 1.0 / (source_radius * source_radius)
    x_edge = (jnp.arange(resolution) == 0) | (
        jnp.arange(resolution) == resolution - 1
    )

    def integrate_row(carry, row_index):
        image_y = coordinate[row_index]
        (
            mapped_x,
            mapped_y,
            du_dx,
            du_dy,
            dv_dx,
            dv_dy,
        ) = triple_lens_map_and_derivatives_real(coordinate, image_y, geometry)
        residual_x = mapped_x - source_x
        residual_y = mapped_y - source_y
        phi = 1.0 - (
            residual_x * residual_x + residual_y * residual_y
        ) * inverse_source_radius_squared
        gradient_x = -2.0 * inverse_source_radius_squared * (
            residual_x * du_dx + residual_y * dv_dx
        )
        gradient_y = -2.0 * inverse_source_radius_squared * (
            residual_x * du_dy + residual_y * dv_dy
        )
        affine_moments, _, _ = jax.vmap(
            lambda value, dx, dy: resolved_moments(
                value, dx, dy, frozen_cell_size
            )
        )(phi, gradient_x, gradient_y)
        half_delta_x = 0.5 * gradient_x * frozen_cell_size
        half_delta_y = 0.5 * gradient_y * frozen_cell_size
        extent = jnp.abs(half_delta_x) + jnp.abs(half_delta_y)
        fully_inside = phi - extent > 0.0
        fully_outside = phi + extent <= 0.0
        boundary = ~(fully_inside | fully_outside)
        safe_phi = jnp.where(phi > 0.0, phi, 1.0)
        area = frozen_cell_size * frozen_cell_size
        moment_0 = jnp.full_like(phi, area)
        if moment_mode == "uniform":
            interior_moments = moment_0[:, None]
        else:
            gradient_squared_term = (
                frozen_cell_size
                * frozen_cell_size
                * (gradient_x * gradient_x + gradient_y * gradient_y)
            )
            laplacian = (
                -2.0
                * inverse_source_radius_squared
                * (
                    du_dx * du_dx
                    + du_dy * du_dy
                    + dv_dx * dv_dx
                    + dv_dy * dv_dy
                )
            )
            laplacian_term = frozen_cell_size * frozen_cell_size * laplacian
            sqrt_phi = jnp.sqrt(safe_phi)
            moment_half = area * (
                sqrt_phi
                + laplacian_term / (48.0 * sqrt_phi)
                - gradient_squared_term / (96.0 * safe_phi * sqrt_phi)
            )
            if moment_mode == "linear":
                interior_moments = jnp.stack((moment_0, moment_half), axis=1)
            else:
                fourth_root = jnp.sqrt(sqrt_phi)
                moment_quarter = area * (
                    fourth_root
                    + laplacian_term / (96.0 * sqrt_phi * fourth_root)
                    - gradient_squared_term
                    / (128.0 * safe_phi * sqrt_phi * fourth_root)
                )
                interior_moments = jnp.stack(
                    (moment_0, moment_half, moment_quarter), axis=1
                )
        moments = jnp.where(
            fully_inside[:, None],
            interior_moments,
            jnp.where(
                boundary[:, None],
                affine_moments,
                jnp.zeros_like(affine_moments),
            ),
        )
        contributing = ~fully_outside
        row_is_edge = (row_index == 0) | (row_index == resolution - 1)
        edge_touched = jnp.any(contributing & (x_edge | row_is_edge))
        accumulated, boundary_count, contributing_count, support_touched = carry
        return (
            accumulated + jnp.sum(moments, axis=0),
            boundary_count + jnp.sum(boundary, dtype=jnp.int32),
            contributing_count + jnp.sum(contributing, dtype=jnp.int32),
            support_touched | edge_touched,
        ), None

    initial = (
        jnp.zeros(moment_count, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )
    (moments, boundary_cells, contributing_cells, support_touched), _ = jax.lax.scan(
        integrate_row,
        initial,
        jnp.arange(resolution, dtype=jnp.int32),
    )
    lens_bound = jnp.max(
        jnp.sqrt(jnp.sum(geometry.positions * geometry.positions, axis=1))
    )
    source_bound = jnp.sqrt(source_x * source_x + source_y * source_y) + source_radius
    # Outside the circle containing every lens,
    # |zeta| >= |z| - 1 / (|z| - lens_bound).  Solving the corresponding
    # quadratic gives a conservative radius containing the preimage of the
    # complete source disk.
    required_radius = 0.5 * (
        source_bound
        + lens_bound
        + jnp.sqrt((source_bound - lens_bound) ** 2 + 4.0)
    )
    support_bound_clear = image_extent >= (
        required_radius + jnp.sqrt(0.5) * frozen_cell_size
    )
    if moment_mode == "uniform":
        full_moments = jnp.stack(
            (moments[0], moments[0] * 2.0 / 3.0, moments[0] * 4.0 / 5.0)
        )
    elif moment_mode == "linear":
        full_moments = jnp.stack(
            (moments[0], moments[1], moments[0] * 4.0 / 5.0)
        )
    else:
        full_moments = moments
    magnification = combine_limb_darkening_moments(
        full_moments,
        source_radius,
        limb_c,
        limb_d,
    )
    return TripleInverseRayResult(
        magnification,
        full_moments,
        boundary_cells,
        contributing_cells,
        (~support_touched) & support_bound_clear,
        cell_size,
    )

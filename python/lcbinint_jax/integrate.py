"""Fixed-support Cartesian macro-tile inverse-ray integration."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import require_x64
from .cell_moments import (
    midpoint_cell_moments,
    midpoint_cell_moments_linear,
    midpoint_cell_moments_uniform,
    resolved_cell_moments,
    resolved_cell_moments_linear,
    resolved_cell_moments_uniform,
)
from .lens import (
    binary_lens_map_and_derivatives_real,
    binary_lens_map_complex,
)
from .limb_darkening import combine_limb_darkening_moments
from .types import FixedSupportResult


def _tile_offsets(tile_size: int, cell_size: jax.Array) -> tuple[jax.Array, jax.Array]:
    frozen_cell_size = jax.lax.stop_gradient(cell_size)
    one_dimensional = (
        jnp.arange(tile_size, dtype=frozen_cell_size.dtype) + 0.5
    ) * frozen_cell_size
    offset_x, offset_y = jnp.meshgrid(one_dimensional, one_dimensional, indexing="xy")
    return offset_x.ravel(), offset_y.ravel()


def _phi_and_gradient_real(
    image_x: jax.Array,
    image_y: jax.Array,
    source_x: jax.Array,
    source_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    source_radius: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mapped_x, mapped_y, du_dx, du_dy, dv_dx, dv_dy = (
        binary_lens_map_and_derivatives_real(image_x, image_y, separation, mass_ratio)
    )
    residual_x = mapped_x - source_x
    residual_y = mapped_y - source_y
    inverse_radius_squared = 1.0 / (source_radius * source_radius)
    phi = (
        1.0
        - (residual_x * residual_x + residual_y * residual_y) * inverse_radius_squared
    )
    gradient_x = (
        -2.0 * inverse_radius_squared * (residual_x * du_dx + residual_y * dv_dx)
    )
    gradient_y = (
        -2.0 * inverse_radius_squared * (residual_x * du_dy + residual_y * dv_dy)
    )
    return phi, gradient_x, gradient_y


def _phi_and_gradient_complex(
    image_x: jax.Array,
    image_y: jax.Array,
    source_x: jax.Array,
    source_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    source_radius: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    image = image_x + 1j * image_y

    def scalar_phi(real_xy):
        mapped = binary_lens_map_complex(
            real_xy[0] + 1j * real_xy[1], separation, mass_ratio
        )
        residual_x = jnp.real(mapped) - source_x
        residual_y = jnp.imag(mapped) - source_y
        return 1.0 - (residual_x * residual_x + residual_y * residual_y) / (
            source_radius * source_radius
        )

    phi = scalar_phi(jnp.stack((jnp.real(image), jnp.imag(image))))
    gradient = jax.grad(scalar_phi)(jnp.stack((jnp.real(image), jnp.imag(image))))
    return phi, gradient[0], gradient[1]


@partial(
    jax.jit,
    static_argnames=("tile_size", "kernel", "moment_mode", "boundary_capacity"),
)
def binary_inverse_ray_fixed_support(
    tile_origins: jax.Array,
    tile_mask: jax.Array,
    cell_size: jax.Array,
    source_x: jax.Array,
    source_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    source_radius: jax.Array,
    limb_c: jax.Array = 0.0,
    limb_d: jax.Array = 0.0,
    *,
    tile_size: int = 8,
    kernel: str = "real",
    moment_mode: str = "two_coefficient",
    boundary_capacity: int = 32,
) -> FixedSupportResult:
    """Integrate caller-supplied, non-overlapping image-plane macro-tiles.

    ``tile_origins`` contains the lower-left physical coordinate of each tile.
    Numerical support and grid geometry are stopped-gradient by design.
    """

    require_x64()
    if kernel not in ("real", "complex"):
        raise ValueError("kernel must be 'real' or 'complex'")
    if moment_mode not in ("uniform", "linear", "two_coefficient"):
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )

    frozen_origins = jax.lax.stop_gradient(tile_origins)
    frozen_mask = jax.lax.stop_gradient(tile_mask)
    offset_x, offset_y = _tile_offsets(tile_size, cell_size)
    if moment_mode == "uniform":
        midpoint_moments = midpoint_cell_moments_uniform
        resolved_moments = resolved_cell_moments_uniform
        moment_count = 1
    elif moment_mode == "linear":
        midpoint_moments = midpoint_cell_moments_linear
        resolved_moments = resolved_cell_moments_linear
        moment_count = 2
    else:
        midpoint_moments = midpoint_cell_moments
        resolved_moments = resolved_cell_moments
        moment_count = 3
    phi_kernel = (
        _phi_and_gradient_real if kernel == "real" else _phi_and_gradient_complex
    )

    def integrate_tile(carry, inputs):
        origin, active = inputs

        def active_tile(active_origin):
            image_x = active_origin[0] + offset_x
            image_y = active_origin[1] + offset_y
            phi, gradient_x, gradient_y = jax.vmap(
                lambda x, y: phi_kernel(
                    x,
                    y,
                    source_x,
                    source_y,
                    separation,
                    mass_ratio,
                    source_radius,
                )
            )(image_x, image_y)
            half_delta_x = 0.5 * gradient_x * cell_size
            half_delta_y = 0.5 * gradient_y * cell_size
            extent = jnp.abs(half_delta_x) + jnp.abs(half_delta_y)
            fully_inside = phi - extent > 0.0
            fully_outside = phi + extent <= 0.0
            boundary = ~(fully_inside | fully_outside)

            def detailed_tile(_):
                boundary_count = jnp.sum(boundary, dtype=jnp.int32)

                def packed_boundary(_):
                    safe_phi = jnp.where(fully_inside, phi, 1.0)
                    area = cell_size * cell_size
                    delta_squared = (
                        cell_size
                        * cell_size
                        * (gradient_x * gradient_x + gradient_y * gradient_y)
                    )
                    moment_0 = jnp.full_like(phi, area)
                    if moment_mode == "uniform":
                        interior_moments = moment_0[:, None]
                    elif moment_mode == "linear":
                        sqrt_phi = jnp.sqrt(safe_phi)
                        moment_half = area * (
                            sqrt_phi - delta_squared / (96.0 * safe_phi * sqrt_phi)
                        )
                        interior_moments = jnp.stack((moment_0, moment_half), axis=1)
                    else:
                        sqrt_phi = jnp.sqrt(safe_phi)
                        moment_half = area * (
                            sqrt_phi - delta_squared / (96.0 * safe_phi * sqrt_phi)
                        )
                        fourth_root = jnp.sqrt(sqrt_phi)
                        moment_quarter = area * (
                            fourth_root
                            - delta_squared
                            / (128.0 * safe_phi * sqrt_phi * fourth_root)
                        )
                        interior_moments = jnp.stack(
                            (moment_0, moment_half, moment_quarter), axis=1
                        )
                    interior_moments = jnp.where(
                        fully_inside[:, None],
                        interior_moments,
                        jnp.zeros_like(interior_moments),
                    )
                    boundary_indices = jnp.nonzero(
                        boundary,
                        size=boundary_capacity,
                        fill_value=0,
                    )[0]
                    boundary_slot_mask = (
                        jnp.arange(boundary_capacity, dtype=jnp.int32) < boundary_count
                    )
                    packed_moments, _, _ = jax.vmap(
                        lambda value, dx, dy: resolved_moments(value, dx, dy, cell_size)
                    )(
                        phi[boundary_indices],
                        gradient_x[boundary_indices],
                        gradient_y[boundary_indices],
                    )
                    packed_moments = jnp.where(
                        boundary_slot_mask[:, None],
                        packed_moments,
                        jnp.zeros_like(packed_moments),
                    )
                    return (
                        jnp.sum(interior_moments, axis=0)
                        + jnp.sum(packed_moments, axis=0),
                        boundary_count,
                        jnp.sum(fully_inside, dtype=jnp.int32) + boundary_count,
                    )

                def dense_boundary(_):
                    cell_moments, detailed_boundary, contributing = jax.vmap(
                        lambda value, dx, dy: resolved_moments(value, dx, dy, cell_size)
                    )(phi, gradient_x, gradient_y)
                    return (
                        jnp.sum(cell_moments, axis=0),
                        jnp.sum(detailed_boundary, dtype=jnp.int32),
                        jnp.sum(contributing, dtype=jnp.int32),
                    )

                return jax.lax.cond(
                    boundary_count <= boundary_capacity,
                    packed_boundary,
                    dense_boundary,
                    operand=None,
                )

            def boundary_free_tile(_):
                def interior_tile(_):
                    cell_moments = jax.vmap(
                        lambda value: midpoint_moments(value, cell_size)
                    )(phi)
                    cell_moments = jnp.where(
                        fully_inside[:, None],
                        cell_moments,
                        jnp.zeros_like(cell_moments),
                    )
                    return (
                        jnp.sum(cell_moments, axis=0),
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.sum(
                            fully_inside.astype(jnp.int32),
                            dtype=jnp.int32,
                        ),
                    )

                return jax.lax.cond(
                    jnp.any(fully_inside),
                    interior_tile,
                    inactive_tile,
                    active_origin,
                )

            return jax.lax.cond(
                jnp.any(boundary),
                detailed_tile,
                boundary_free_tile,
                active_origin,
            )

        def inactive_tile(_):
            return (
                jnp.zeros(moment_count, dtype=carry[0].dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )

        tile_moments, tile_boundary, tile_contributing = jax.lax.cond(
            active,
            jax.checkpoint(active_tile),
            inactive_tile,
            origin,
        )
        moments, boundary_count, active_count = carry
        return (
            moments + tile_moments,
            boundary_count + tile_boundary,
            active_count + tile_contributing,
        ), None

    initial = (
        jnp.zeros(
            moment_count,
            dtype=jnp.result_type(tile_origins, source_radius),
        ),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    (moments, boundary_cells, active_cells), _ = jax.lax.scan(
        integrate_tile, initial, (frozen_origins, frozen_mask)
    )
    if moment_mode == "uniform":
        magnification = moments[0] / (jnp.pi * source_radius * source_radius)
    elif moment_mode == "linear":
        magnification = ((1.0 - limb_c) * moments[0] + limb_c * moments[1]) / (
            jnp.pi * source_radius * source_radius * (1.0 - limb_c / 3.0)
        )
    else:
        magnification = combine_limb_darkening_moments(
            moments, source_radius, limb_c, limb_d
        )
    return FixedSupportResult(
        magnification=magnification,
        moments=moments,
        boundary_cells=boundary_cells,
        active_cells=active_cells,
    )

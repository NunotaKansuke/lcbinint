"""Fixed-support Cartesian macro-tile inverse-ray integration."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import as_float64, require_x64
from .cell_moments import (
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


def _phi_gradient_laplacian_real(
    image_x: jax.Array,
    image_y: jax.Array,
    source_x: jax.Array,
    source_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    source_radius: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
    laplacian = (
        -2.0
        * inverse_radius_squared
        * (du_dx * du_dx + du_dy * du_dy + dv_dx * dv_dx + dv_dy * dv_dy)
    )
    return phi, gradient_x, gradient_y, laplacian


def _phi_gradient_laplacian_complex(
    image_x: jax.Array,
    image_y: jax.Array,
    source_x: jax.Array,
    source_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    source_radius: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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

    real_xy = jnp.stack((jnp.real(image), jnp.imag(image)))
    phi, gradient = jax.value_and_grad(scalar_phi)(real_xy)
    hessian = jax.hessian(scalar_phi)(real_xy)
    return phi, gradient[0], gradient[1], jnp.trace(hessian)


def _phi_and_gradient_real(*args):
    """Compatibility kernel used by the polar integrator."""

    return _phi_gradient_laplacian_real(*args)[:3]


def _phi_and_gradient_complex(*args):
    """Compatibility kernel used by the polar integrator."""

    return _phi_gradient_laplacian_complex(*args)[:3]


@partial(
    jax.jit,
    static_argnames=(
        "tile_size",
        "kernel",
        "moment_mode",
        "boundary_capacity",
        "boundary_subdivision",
        "boundary_adaptive_threshold",
    ),
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
    boundary_capacity: int = 256,
    boundary_subdivision: int = 4,
    boundary_adaptive_threshold: float = 0.08,
) -> FixedSupportResult:
    """Integrate caller-supplied, non-overlapping image-plane macro-tiles.

    ``tile_origins`` contains the lower-left physical coordinate of each tile.
    Numerical support and grid geometry are stopped-gradient by design.
    ``boundary_subdivision=0`` uses 2x2 on low-curvature detailed cells and
    4x4 where the stopped-gradient curvature indicator exceeds
    ``boundary_adaptive_threshold``.
    """

    require_x64()
    if kernel not in ("real", "complex"):
        raise ValueError("kernel must be 'real' or 'complex'")
    if moment_mode not in ("uniform", "linear", "two_coefficient"):
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )
    if boundary_subdivision not in (0, 1, 2, 3, 4):
        raise ValueError("boundary_subdivision must be 0, 1, 2, 3, or 4")
    if boundary_adaptive_threshold <= 0.0:
        raise ValueError("boundary_adaptive_threshold must be positive")
    if boundary_capacity < tile_size * tile_size:
        raise ValueError(
            "boundary_capacity must hold every cell in one tile "
            "(boundary_capacity >= tile_size**2)"
        )

    (
        tile_origins,
        cell_size,
        source_x,
        source_y,
        separation,
        mass_ratio,
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
            source_radius,
            limb_c,
            limb_d,
        )
    )

    frozen_origins = jax.lax.stop_gradient(tile_origins)
    frozen_mask = jax.lax.stop_gradient(tile_mask)
    offset_x, offset_y = _tile_offsets(tile_size, cell_size)
    if moment_mode == "uniform":
        resolved_moments = resolved_cell_moments_uniform
        moment_count = 1
    elif moment_mode == "linear":
        resolved_moments = resolved_cell_moments_linear
        moment_count = 2
    else:
        resolved_moments = resolved_cell_moments
        moment_count = 3
    phi_kernel = (
        _phi_gradient_laplacian_real
        if kernel == "real"
        else _phi_gradient_laplacian_complex
    )

    def second_order_interior_moments(
        phi,
        gradient_x,
        gradient_y,
        laplacian,
    ):
        safe_phi = jnp.where(phi > 0.0, phi, 1.0)
        area = cell_size * cell_size
        moment_0 = jnp.full_like(phi, area)
        if moment_mode == "uniform":
            return moment_0[:, None]
        sqrt_phi = jnp.sqrt(safe_phi)
        delta_squared = (
            cell_size * cell_size * (gradient_x * gradient_x + gradient_y * gradient_y)
        )
        laplacian_term = cell_size * cell_size * laplacian
        moment_half = area * (
            sqrt_phi
            + laplacian_term / (48.0 * sqrt_phi)
            - delta_squared / (96.0 * safe_phi * sqrt_phi)
        )
        if moment_mode == "linear":
            return jnp.stack((moment_0, moment_half), axis=1)
        fourth_root = jnp.sqrt(sqrt_phi)
        moment_quarter = area * (
            fourth_root
            + laplacian_term / (96.0 * sqrt_phi * fourth_root)
            - delta_squared / (128.0 * safe_phi * sqrt_phi * fourth_root)
        )
        return jnp.stack(
            (moment_0, moment_half, moment_quarter),
            axis=1,
        )

    def integrate_tile(carry, inputs):
        origin, active = inputs

        def active_tile(active_origin):
            image_x = active_origin[0] + offset_x
            image_y = active_origin[1] + offset_y
            phi, gradient_x, gradient_y, laplacian = jax.vmap(
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
            geometric_boundary = ~(fully_inside | fully_outside)
            if moment_mode != "two_coefficient":
                detailed = geometric_boundary
            else:
                quarter_power_active = jax.lax.stop_gradient(limb_d != 0.0)
                relative_variation = (
                    extent + 0.125 * jnp.abs(laplacian) * cell_size * cell_size
                ) / jnp.maximum(phi, 1.0e-30)
                detailed = geometric_boundary | (
                    quarter_power_active & fully_inside & (relative_variation > 0.2)
                )
            bulk_inside = fully_inside & ~detailed

            def detailed_tile(_):
                boundary_count = jnp.sum(detailed, dtype=jnp.int32)

                def packed_boundary(_):
                    interior_moments = second_order_interior_moments(
                        phi,
                        gradient_x,
                        gradient_y,
                        laplacian,
                    )
                    interior_moments = jnp.where(
                        bulk_inside[:, None],
                        interior_moments,
                        jnp.zeros_like(interior_moments),
                    )

                    def boundary_moments(index, subdivision):
                        if subdivision == 1:
                            moments, _, _ = resolved_moments(
                                phi[index],
                                gradient_x[index],
                                gradient_y[index],
                                cell_size,
                            )
                            return moments

                        subcell_size = cell_size / subdivision
                        one_dimensional_subcell_offsets = (
                            (
                                jnp.arange(
                                    subdivision,
                                    dtype=cell_size.dtype,
                                )
                                + 0.5
                            )
                            / subdivision
                            - 0.5
                        ) * cell_size
                        subcell_offset_x, subcell_offset_y = jnp.meshgrid(
                            one_dimensional_subcell_offsets,
                            one_dimensional_subcell_offsets,
                            indexing="xy",
                        )
                        subcell_x = image_x[index] + subcell_offset_x.ravel()
                        subcell_y = image_y[index] + subcell_offset_y.ravel()
                        (
                            sub_phi,
                            sub_gradient_x,
                            sub_gradient_y,
                            _,
                        ) = jax.vmap(
                            lambda x, y: phi_kernel(
                                x,
                                y,
                                source_x,
                                source_y,
                                separation,
                                mass_ratio,
                                source_radius,
                            )
                        )(subcell_x, subcell_y)
                        subcell_moments, _, _ = jax.vmap(
                            lambda value, dx, dy: resolved_moments(
                                value,
                                dx,
                                dy,
                                subcell_size,
                            )
                        )(
                            sub_phi,
                            sub_gradient_x,
                            sub_gradient_y,
                        )
                        return jnp.sum(subcell_moments, axis=0)

                    def integrate_boundary_slots(
                        slot_capacity,
                        selection,
                        selection_count,
                        subdivision,
                    ):
                        boundary_indices = jnp.nonzero(
                            selection,
                            size=slot_capacity,
                            fill_value=0,
                        )[0]
                        boundary_slot_mask = (
                            jnp.arange(slot_capacity, dtype=jnp.int32) < selection_count
                        )
                        packed_moments = jax.vmap(
                            lambda index: boundary_moments(index, subdivision)
                        )(boundary_indices)
                        packed_moments = jnp.where(
                            boundary_slot_mask[:, None],
                            packed_moments,
                            jnp.zeros_like(packed_moments),
                        )
                        return jnp.sum(packed_moments, axis=0)

                    slot_tiers = tuple(
                        tier
                        for tier in (8, 16, 32, 64, 128)
                        if tier < boundary_capacity
                    ) + (boundary_capacity,)

                    def select_slot_tier(
                        tiers,
                        selection,
                        selection_count,
                        subdivision,
                    ):
                        tier = tiers[0]
                        if len(tiers) == 1:
                            return integrate_boundary_slots(
                                tier,
                                selection,
                                selection_count,
                                subdivision,
                            )
                        return jax.lax.cond(
                            selection_count <= tier,
                            lambda _: integrate_boundary_slots(
                                tier,
                                selection,
                                selection_count,
                                subdivision,
                            ),
                            lambda _: select_slot_tier(
                                tiers[1:],
                                selection,
                                selection_count,
                                subdivision,
                            ),
                            operand=None,
                        )

                    if boundary_subdivision == 0:
                        curvature_ratio = (
                            jnp.abs(laplacian) * cell_size * cell_size
                        ) / jnp.maximum(extent, 1.0e-30)
                        high_order = jax.lax.stop_gradient(
                            detailed & (curvature_ratio > boundary_adaptive_threshold)
                        )
                        low_order = detailed & ~high_order
                        low_order_count = jnp.sum(low_order, dtype=jnp.int32)
                        high_order_count = jnp.sum(high_order, dtype=jnp.int32)
                        packed_sum = jax.lax.cond(
                            low_order_count > 0,
                            lambda _: select_slot_tier(
                                slot_tiers,
                                low_order,
                                low_order_count,
                                2,
                            ),
                            lambda _: jnp.zeros(moment_count, dtype=phi.dtype),
                            operand=None,
                        )
                        packed_sum = packed_sum + jax.lax.cond(
                            high_order_count > 0,
                            lambda _: select_slot_tier(
                                slot_tiers,
                                high_order,
                                high_order_count,
                                4,
                            ),
                            lambda _: jnp.zeros_like(packed_sum),
                            operand=None,
                        )
                    else:
                        packed_sum = select_slot_tier(
                            slot_tiers,
                            detailed,
                            boundary_count,
                            boundary_subdivision,
                        )
                    return (
                        jnp.sum(interior_moments, axis=0) + packed_sum,
                        boundary_count,
                        jnp.sum(bulk_inside, dtype=jnp.int32) + boundary_count,
                    )

                return packed_boundary(None)

            def boundary_free_tile(_):
                def interior_tile(_):
                    cell_moments = second_order_interior_moments(
                        phi,
                        gradient_x,
                        gradient_y,
                        laplacian,
                    )
                    cell_moments = jnp.where(
                        bulk_inside[:, None],
                        cell_moments,
                        jnp.zeros_like(cell_moments),
                    )
                    return (
                        jnp.sum(cell_moments, axis=0),
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.sum(
                            bulk_inside.astype(jnp.int32),
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
                jnp.any(detailed),
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

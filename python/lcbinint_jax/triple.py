"""Initial differentiable Cartesian inverse-ray integrator for triple lenses."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._config import require_x64
from .cell_moments import (
    resolved_cell_moments,
    resolved_cell_moments_linear,
    resolved_cell_moments_uniform,
)
from .limb_darkening import combine_limb_darkening_moments


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

    q = jnp.asarray(mass_ratio)
    q2 = jnp.asarray(tertiary_mass_ratio, dtype=q.dtype)
    separation = jnp.asarray(separation, dtype=q.dtype)
    tertiary_separation = jnp.asarray(tertiary_separation, dtype=q.dtype)
    tertiary_angle = jnp.asarray(tertiary_angle, dtype=q.dtype)
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

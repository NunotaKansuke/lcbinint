"""Fixed-shape polar inverse-ray integration over image-informed radial bands."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import require_x64
from .cell_moments import _affine_unit_square_moment
from .discovery import binary_image_seed_points
from .integrate import _phi_and_gradient_complex, _phi_and_gradient_real
from .limb_darkening import combine_limb_darkening_moments
from .types import InverseRayResult, PolarSupportResult


@partial(
    jax.jit,
    static_argnames=("limb_samples", "band_capacity", "root_backend"),
)
def discover_binary_polar_bands(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    *,
    limb_samples=16,
    band_capacity=16,
    padding_factor=0.25,
    root_backend="auto",
):
    """Merge physical centre/limb image radii into disjoint radial bands."""

    seeds = binary_image_seed_points(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_samples=limb_samples,
        root_backend=root_backend,
    )
    radii = jnp.abs(seeds.roots)
    padding = jax.lax.stop_gradient(padding_factor * source_radius)
    lower = jnp.where(seeds.physical, jnp.maximum(0.0, radii - padding), jnp.inf)
    upper = jnp.where(seeds.physical, radii + padding, -jnp.inf)
    order = jnp.argsort(lower)
    lower = lower[order]
    upper = upper[order]

    initial = (
        jnp.zeros(band_capacity, dtype=lower.dtype),
        jnp.zeros(band_capacity, dtype=upper.dtype),
        jnp.zeros(band_capacity, dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )

    def insert_interval(state, interval):
        band_lower, band_upper, band_mask, count, overflow = state
        candidate_lower, candidate_upper = interval
        valid = jnp.isfinite(candidate_lower)
        previous_index = jnp.maximum(count - 1, 0)
        overlaps = valid & (count > 0) & (candidate_lower <= band_upper[previous_index])
        band_upper = band_upper.at[previous_index].set(
            jnp.where(
                overlaps,
                jnp.maximum(band_upper[previous_index], candidate_upper),
                band_upper[previous_index],
            )
        )
        novel = valid & ~overlaps
        has_capacity = count < band_capacity
        target = jnp.minimum(count, band_capacity - 1)
        write = novel & has_capacity
        band_lower = band_lower.at[target].set(
            jnp.where(write, candidate_lower, band_lower[target])
        )
        band_upper = band_upper.at[target].set(
            jnp.where(write, candidate_upper, band_upper[target])
        )
        band_mask = band_mask.at[target].set(band_mask[target] | write)
        return (
            band_lower,
            band_upper,
            band_mask,
            count + write.astype(jnp.int32),
            overflow | (novel & ~has_capacity),
        ), None

    (band_lower, band_upper, band_mask, count, overflow), _ = jax.lax.scan(
        insert_interval, initial, (lower, upper)
    )
    return jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        PolarSupportResult(
            lower=band_lower,
            upper=band_upper,
            mask=band_mask,
            band_count=count,
            overflow=overflow,
            root_failure=seeds.root_failure,
        ),
    )


@partial(
    jax.jit,
    static_argnames=(
        "resolution",
        "angular_bins",
        "radial_capacity",
        "band_capacity",
        "limb_samples",
        "kernel",
        "moment_mode",
        "angular_padding_factor",
        "angular_chunk_size",
        "boundary_capacity",
        "boundary_subdivision",
        "root_backend",
    ),
)
def binary_inverse_ray_polar(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    resolution=64,
    angular_bins=2048,
    radial_capacity=512,
    band_capacity=4,
    limb_samples=16,
    padding_factor=0.25,
    angular_padding_factor=4.0,
    angular_chunk_size=256,
    boundary_capacity=2048,
    boundary_subdivision=2,
    kernel="real",
    moment_mode="two_coefficient",
    root_backend="auto",
):
    """Integrate image-informed radial bands on a polar image-plane grid."""

    require_x64()
    if kernel not in ("real", "complex"):
        raise ValueError("kernel must be 'real' or 'complex'")
    if moment_mode not in ("uniform", "linear", "two_coefficient"):
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )
    angular_chunk_size = min(angular_chunk_size, angular_bins)
    if angular_bins % angular_chunk_size != 0:
        raise ValueError("angular_bins must be divisible by angular_chunk_size")
    if boundary_subdivision < 1:
        raise ValueError("boundary_subdivision must be positive")
    support = discover_binary_polar_bands(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_samples=limb_samples,
        band_capacity=band_capacity,
        padding_factor=padding_factor,
        root_backend=root_backend,
    )
    seeds = binary_image_seed_points(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_samples=limb_samples,
        root_backend=root_backend,
    )
    seed_radii = jax.lax.stop_gradient(jnp.abs(seeds.roots))
    seed_angles = jax.lax.stop_gradient(jnp.angle(seeds.roots))
    seed_physical = jax.lax.stop_gradient(seeds.physical)
    dr = jax.lax.stop_gradient(source_radius / resolution)
    padding = jax.lax.stop_gradient(padding_factor * source_radius)
    dtheta = jnp.asarray(2.0 * jnp.pi / angular_bins, dtype=dr.dtype)
    radial_index = jnp.arange(radial_capacity, dtype=dr.dtype)
    angular_padding = jax.lax.stop_gradient(
        angular_padding_factor * 2.0 * jnp.pi / limb_samples
    )
    phi_kernel = (
        _phi_and_gradient_real if kernel == "real" else _phi_and_gradient_complex
    )
    if moment_mode == "uniform":
        powers = jnp.asarray((0.0,), dtype=dr.dtype)
    elif moment_mode == "linear":
        powers = jnp.asarray((0.0, 0.5), dtype=dr.dtype)
    else:
        powers = jnp.asarray((0.0, 0.5, 0.25), dtype=dr.dtype)

    seed_in_band = (
        seed_physical[None, :]
        & (seed_radii[None, :] >= support.lower[:, None])
        & (seed_radii[None, :] <= support.upper[:, None])
    )

    def integrate_chunk(carry, chunk_index):
        angular_index = chunk_index * angular_chunk_size + jnp.arange(
            angular_chunk_size, dtype=jnp.int32
        )
        theta = (angular_index.astype(dr.dtype) + 0.5) * dtheta
        cosine = jnp.cos(theta)[:, None, None]
        sine = jnp.sin(theta)[:, None, None]
        angle_distance = jnp.abs(
            jnp.atan2(
                jnp.sin(seed_angles[None, :] - theta[:, None]),
                jnp.cos(seed_angles[None, :] - theta[:, None]),
            )
        )
        nearby = seed_in_band[None, :, :] & (
            angle_distance[:, None, :] <= angular_padding
        )
        nearby_count = jnp.sum(nearby, axis=2, dtype=jnp.int32)
        local_lower = jnp.maximum(
            0.0,
            jnp.min(
                jnp.where(nearby, seed_radii[None, None, :], jnp.inf),
                axis=2,
            )
            - padding,
        )
        local_upper = (
            jnp.max(
                jnp.where(nearby, seed_radii[None, None, :], -jnp.inf),
                axis=2,
            )
            + padding
        )
        band_enabled = support.mask[None, :] & (nearby_count > 0)
        local_overflow = band_enabled & (
            local_upper - local_lower > radial_capacity * dr
        )
        radii = local_lower[:, :, None] + (radial_index + 0.5) * dr
        radial_mask = band_enabled[:, :, None] & (radii < local_upper[:, :, None])
        safe_radii = jnp.where(radial_mask, radii, 1.0)
        image_x = safe_radii * cosine
        image_y = safe_radii * sine
        if kernel == "real":
            phi, gradient_x, gradient_y = phi_kernel(
                image_x,
                image_y,
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
            )
        else:
            shape = image_x.shape
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
            )(image_x.ravel(), image_y.ravel())
            phi = phi.reshape(shape)
            gradient_x = gradient_x.reshape(shape)
            gradient_y = gradient_y.reshape(shape)
        gradient_r = gradient_x * cosine + gradient_y * sine
        gradient_theta = safe_radii * (-gradient_x * sine + gradient_y * cosine)
        delta_r = gradient_r * dr
        delta_theta = gradient_theta * dtheta
        area = safe_radii * dr * dtheta
        extent = 0.5 * (jnp.abs(delta_r) + jnp.abs(delta_theta))
        fully_inside = radial_mask & (phi - extent > 0.0)
        boundary = radial_mask & (phi - extent <= 0.0) & (phi + extent > 0.0)
        chunk_boundary_count = jnp.sum(boundary, dtype=jnp.int32)

        flat_phi = phi.ravel()
        flat_delta_r = delta_r.ravel()
        flat_delta_theta = delta_theta.ravel()
        flat_area = area.ravel()
        flat_radius = safe_radii.ravel()
        flat_theta = jnp.broadcast_to(theta[:, None, None], safe_radii.shape).ravel()
        boundary_indices = jnp.nonzero(
            boundary.ravel(),
            size=boundary_capacity,
            fill_value=0,
        )[0]
        boundary_slot_mask = (
            jnp.arange(boundary_capacity, dtype=jnp.int32) < chunk_boundary_count
        )
        safe_interior_phi = jnp.where(fully_inside, phi, 1.0)
        safe_interior_area = jnp.where(fully_inside, area, 0.0)
        delta_squared = delta_r * delta_r + delta_theta * delta_theta
        interior_0 = jnp.sum(safe_interior_area)
        if moment_mode == "uniform":
            interior_moments = interior_0[None]
        elif moment_mode == "linear":
            interior_moments = jnp.stack(
                (
                    interior_0,
                    jnp.sum(
                        safe_interior_area
                        * (
                            jnp.sqrt(safe_interior_phi)
                            - delta_squared
                            / (96.0 * safe_interior_phi * jnp.sqrt(safe_interior_phi))
                        )
                    ),
                )
            )
        else:
            sqrt_phi = jnp.sqrt(safe_interior_phi)
            interior_moments = jnp.stack(
                (
                    interior_0,
                    jnp.sum(
                        safe_interior_area
                        * (
                            sqrt_phi
                            - delta_squared / (96.0 * safe_interior_phi * sqrt_phi)
                        )
                    ),
                    jnp.sum(
                        safe_interior_area
                        * (
                            jnp.sqrt(sqrt_phi)
                            - delta_squared
                            / (
                                128.0
                                * safe_interior_phi
                                * sqrt_phi
                                * jnp.sqrt(sqrt_phi)
                            )
                        )
                    ),
                )
            )

        boundary_phi = flat_phi[boundary_indices]
        boundary_delta_r = flat_delta_r[boundary_indices]
        boundary_delta_theta = flat_delta_theta[boundary_indices]
        boundary_area = flat_area[boundary_indices]
        if boundary_subdivision == 1:
            boundary_lower_left = (
                boundary_phi - 0.5 * boundary_delta_r - 0.5 * boundary_delta_theta
            )
            boundary_unit_moments = jax.vmap(
                lambda power: boundary_area
                * _affine_unit_square_moment(
                    boundary_lower_left,
                    boundary_delta_r,
                    boundary_delta_theta,
                    power,
                )
            )(powers)
        else:
            subdivision = jnp.asarray(boundary_subdivision, dtype=dr.dtype)
            offsets = (
                jnp.arange(boundary_subdivision, dtype=dr.dtype) + 0.5
            ) / subdivision - 0.5
            radial_offset, angular_offset = jnp.meshgrid(
                offsets, offsets, indexing="ij"
            )
            boundary_radius = flat_radius[boundary_indices, None]
            boundary_theta = flat_theta[boundary_indices, None]
            sub_radius = boundary_radius + radial_offset.ravel() * dr
            sub_theta = boundary_theta + angular_offset.ravel() * dtheta
            sub_cosine = jnp.cos(sub_theta)
            sub_sine = jnp.sin(sub_theta)
            sub_x = sub_radius * sub_cosine
            sub_y = sub_radius * sub_sine
            if kernel == "real":
                sub_phi, sub_gradient_x, sub_gradient_y = phi_kernel(
                    sub_x,
                    sub_y,
                    source_x,
                    source_y,
                    separation,
                    mass_ratio,
                    source_radius,
                )
            else:
                sub_shape = sub_x.shape
                sub_phi, sub_gradient_x, sub_gradient_y = jax.vmap(
                    lambda x, y: phi_kernel(
                        x,
                        y,
                        source_x,
                        source_y,
                        separation,
                        mass_ratio,
                        source_radius,
                    )
                )(sub_x.ravel(), sub_y.ravel())
                sub_phi = sub_phi.reshape(sub_shape)
                sub_gradient_x = sub_gradient_x.reshape(sub_shape)
                sub_gradient_y = sub_gradient_y.reshape(sub_shape)
            sub_gradient_r = sub_gradient_x * sub_cosine + sub_gradient_y * sub_sine
            sub_gradient_theta = sub_radius * (
                -sub_gradient_x * sub_sine + sub_gradient_y * sub_cosine
            )
            sub_delta_r = sub_gradient_r * dr / subdivision
            sub_delta_theta = sub_gradient_theta * dtheta / subdivision
            sub_area = sub_radius * (dr / subdivision) * (dtheta / subdivision)
            sub_lower_left = sub_phi - 0.5 * sub_delta_r - 0.5 * sub_delta_theta
            boundary_unit_moments = jax.vmap(
                lambda power: jnp.sum(
                    sub_area
                    * _affine_unit_square_moment(
                        sub_lower_left,
                        sub_delta_r,
                        sub_delta_theta,
                        power,
                    ),
                    axis=1,
                )
            )(powers)
        boundary_moments = jax.vmap(
            lambda values: jnp.sum(
                jnp.where(
                    boundary_slot_mask,
                    values,
                    0.0,
                )
            )
        )(boundary_unit_moments)
        chunk_moments = interior_moments + boundary_moments
        packing_overflow = chunk_boundary_count > boundary_capacity
        moments, boundary_count, active_count, overflow = carry
        return (
            moments + chunk_moments,
            boundary_count + jnp.sum(boundary, dtype=jnp.int32),
            active_count
            + jnp.sum(fully_inside, dtype=jnp.int32)
            + jnp.sum(boundary, dtype=jnp.int32),
            overflow | jnp.any(local_overflow) | packing_overflow,
        ), None

    initial = (
        jnp.zeros(powers.shape[0], dtype=dr.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )
    (moments, boundary_cells, active_cells, radial_overflow), _ = jax.lax.scan(
        jax.checkpoint(integrate_chunk),
        initial,
        jnp.arange(angular_bins // angular_chunk_size, dtype=jnp.int32),
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
    overflow = support.overflow | radial_overflow
    return InverseRayResult(
        magnification=magnification,
        moments=moments,
        boundary_cells=boundary_cells,
        active_cells=active_cells,
        tile_count=support.band_count,
        discovery_overflow=overflow,
        root_failure=support.root_failure,
        support_valid=~(overflow | support.root_failure),
    )

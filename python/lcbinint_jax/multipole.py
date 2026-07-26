"""Differentiable point-source and hexadecapole binary-lens magnification."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._config import require_x64
from .cpp_backend import (
    binary_images_ffi,
    cpp_binary_image_roots_ffi_available,
)
from .images import binary_images
from .lens import (
    binary_lens_map_and_derivatives_real,
    binary_lens_map_complex,
)


class PointSourceResult(NamedTuple):
    magnification: jax.Array
    image_count: jax.Array
    root_failure: jax.Array


class HexadecapoleResult(NamedTuple):
    magnification: jax.Array
    point_magnification: jax.Array
    quadrupole_correction: jax.Array
    hexadecapole_correction: jax.Array
    estimated_error: jax.Array
    topology_stable: jax.Array
    root_failure: jax.Array


@jax.custom_jvp
def _differentiable_binary_image_roots(source, separation, mass_ratio):
    """Solve for images, with root tangents from the implicit lens equation."""

    return binary_images(source, separation, mass_ratio).roots


@_differentiable_binary_image_roots.defjvp
def _differentiable_binary_image_roots_jvp(primals, tangents):
    source, separation, mass_ratio = primals
    source_dot, separation_dot, mass_ratio_dot = tangents
    solved = binary_images(source, separation, mass_ratio)
    roots = solved.roots

    _, _, du_dx, du_dy, dv_dx, dv_dy = binary_lens_map_and_derivatives_real(
        jnp.real(roots),
        jnp.imag(roots),
        separation,
        mass_ratio,
    )

    def map_at_fixed_roots(active_separation, active_mass_ratio):
        mapped = binary_lens_map_complex(roots, active_separation, active_mass_ratio)
        return jnp.stack((jnp.real(mapped), jnp.imag(mapped)), axis=-1)

    _, parameter_map_dot = jax.jvp(
        map_at_fixed_roots,
        (separation, mass_ratio),
        (separation_dot, mass_ratio_dot),
    )
    rhs_x = jnp.real(source_dot) - parameter_map_dot[:, 0]
    rhs_y = jnp.imag(source_dot) - parameter_map_dot[:, 1]
    determinant = du_dx * dv_dy - du_dy * dv_dx
    safe_determinant = jnp.where(jnp.abs(determinant) > 1.0e-12, determinant, 1.0)
    root_x_dot = (dv_dy * rhs_x - du_dy * rhs_y) / safe_determinant
    root_y_dot = (-dv_dx * rhs_x + du_dx * rhs_y) / safe_determinant
    usable = solved.physical & (jnp.abs(determinant) > 1.0e-12)
    roots_dot = jnp.where(usable, root_x_dot + 1j * root_y_dot, 0.0 + 0.0j)
    return roots, roots_dot


def binary_point_source_magnification(
    source_x,
    source_y,
    separation,
    mass_ratio,
    *,
    root_backend="auto",
) -> PointSourceResult:
    """Return point-source magnification with implicit root derivatives."""

    require_x64()
    if root_backend not in ("auto", "jax", "ffi"):
        raise ValueError("root_backend must be 'auto', 'jax', or 'ffi'")
    source = source_x + 1j * source_y
    use_ffi = root_backend == "ffi" or (
        root_backend == "auto" and cpp_binary_image_roots_ffi_available()
    )
    ffi_images = binary_images_ffi(source, separation, mass_ratio) if use_ffi else None
    roots = (
        ffi_images.roots
        if use_ffi
        else _differentiable_binary_image_roots(source, separation, mass_ratio)
    )
    mapped = binary_lens_map_complex(roots, separation, mass_ratio)
    residuals = jnp.abs(mapped - source)
    physical = (
        ffi_images.physical
        if use_ffi
        else jax.lax.stop_gradient(
            jnp.isfinite(residuals) & (residuals <= 1.0e-9 * (1.0 + jnp.abs(source)))
        )
    )
    _, _, du_dx, du_dy, dv_dx, dv_dy = binary_lens_map_and_derivatives_real(
        jnp.real(roots),
        jnp.imag(roots),
        separation,
        mass_ratio,
    )
    determinant = du_dx * dv_dy - du_dy * dv_dx
    safe = physical & (jnp.abs(determinant) > 1.0e-14)
    magnification = jnp.sum(jnp.where(safe, 1.0 / jnp.abs(determinant), 0.0))
    image_count = jnp.sum(physical, dtype=jnp.int32)
    return PointSourceResult(
        magnification=magnification,
        image_count=image_count,
        root_failure=~jnp.all(jnp.isfinite(roots))
        | ~((image_count == 3) | (image_count == 5)),
    )


@partial(jax.jit, static_argnames=("root_backend",))
def binary_hexadecapole(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    root_backend="auto",
) -> HexadecapoleResult:
    """Thirteen-point finite-source expansion used by native ``lcbinint``."""

    require_x64()
    dtype = jnp.result_type(source_x, source_y, separation, mass_ratio, source_radius)
    sqrt_half = jnp.asarray(jnp.sqrt(0.5), dtype=dtype)
    cardinal_x = jnp.asarray((1.0, 0.0, -1.0, 0.0), dtype=dtype)
    cardinal_y = jnp.asarray((0.0, 1.0, 0.0, -1.0), dtype=dtype)
    diagonal_x = jnp.asarray(
        (sqrt_half, -sqrt_half, -sqrt_half, sqrt_half), dtype=dtype
    )
    diagonal_y = jnp.asarray(
        (sqrt_half, sqrt_half, -sqrt_half, -sqrt_half), dtype=dtype
    )
    offsets_x = jnp.concatenate(
        (
            jnp.zeros(1, dtype=dtype),
            source_radius * cardinal_x,
            0.5 * source_radius * cardinal_x,
            source_radius * diagonal_x,
        )
    )
    offsets_y = jnp.concatenate(
        (
            jnp.zeros(1, dtype=dtype),
            source_radius * cardinal_y,
            0.5 * source_radius * cardinal_y,
            source_radius * diagonal_y,
        )
    )
    samples = jax.vmap(
        lambda dx, dy: binary_point_source_magnification(
            source_x + dx,
            source_y + dy,
            separation,
            mass_ratio,
            root_backend=root_backend,
        )
    )(offsets_x, offsets_y)
    a0 = samples.magnification[0]
    a1_plus = jnp.mean(samples.magnification[1:5]) - a0
    a2_plus = jnp.mean(samples.magnification[5:9]) - a0
    a1_cross = jnp.mean(samples.magnification[9:13]) - a0
    a2rho2 = (16.0 * a2_plus - a1_plus) / 3.0
    a4rho4 = 0.5 * (a1_plus + a1_cross) - a2rho2

    denominator = 15.0 - 5.0 * limb_c - 3.0 * limb_d
    gamma = jnp.where(denominator != 0.0, 10.0 * limb_c / denominator, 0.0)
    lambda_coefficient = jnp.where(denominator != 0.0, 12.0 * limb_d / denominator, 0.0)
    quadrupole_correction = (
        0.5 * a2rho2 * (1.0 - 0.2 * gamma - lambda_coefficient / 9.0)
    )
    hexadecapole_correction = (
        a4rho4 / 3.0 * (1.0 - 11.0 * gamma / 35.0 - 7.0 * lambda_coefficient / 39.0)
    )
    magnification = a0 + quadrupole_correction + hexadecapole_correction
    image_counts = samples.image_count
    topology_stable = jnp.all(image_counts == image_counts[0])
    root_failure = jnp.any(samples.root_failure)
    return HexadecapoleResult(
        magnification=magnification,
        point_magnification=a0,
        quadrupole_correction=quadrupole_correction,
        hexadecapole_correction=hexadecapole_correction,
        estimated_error=jnp.abs(hexadecapole_correction),
        topology_stable=topology_stable,
        root_failure=root_failure,
    )

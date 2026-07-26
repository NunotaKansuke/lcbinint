"""Differentiable source-plane quadrature for binary finite sources."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._config import require_x64
from .multipole import binary_point_source_magnification


class SourcePlaneQuadratureResult(NamedTuple):
    """Fine value and coarse/fine diagnostics for a fixed quadrature rule."""

    magnification: jax.Array
    coarse_magnification: jax.Array
    estimated_error: jax.Array
    sample_count: jax.Array
    root_failure: jax.Array
    converged: jax.Array
    used_chord: jax.Array


def _surface_brightness(radius_squared, limb_c, limb_d):
    mu = jnp.sqrt(jnp.maximum(1.0 - radius_squared, 0.0))
    return (1.0 - limb_c - limb_d) + limb_c * mu + limb_d * jnp.sqrt(mu)


def _point_samples(
    source_x,
    source_y,
    separation,
    mass_ratio,
    root_backend,
):
    flat_x = jnp.ravel(source_x)
    flat_y = jnp.ravel(source_y)
    return jax.vmap(
        lambda x, y: binary_point_source_magnification(
            x,
            y,
            separation,
            mass_ratio,
            root_backend=root_backend,
        )
    )(flat_x, flat_y)


def _ring_quadrature(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
    *,
    radial_bins,
    angular_bins,
    root_backend,
):
    dtype = jnp.result_type(source_x, source_y, separation, mass_ratio, source_radius)
    radius_squared = (
        jnp.arange(radial_bins, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) / radial_bins
    radius = jnp.sqrt(radius_squared)
    angle = 2.0 * jnp.pi * (jnp.arange(angular_bins, dtype=dtype) + 0.5) / angular_bins
    sample_x = source_x + source_radius * radius[:, None] * jnp.cos(angle)[None, :]
    sample_y = source_y + source_radius * radius[:, None] * jnp.sin(angle)[None, :]
    points = _point_samples(
        sample_x,
        sample_y,
        separation,
        mass_ratio,
        root_backend,
    )
    magnifications = points.magnification.reshape((radial_bins, angular_bins))
    brightness = _surface_brightness(radius_squared, limb_c, limb_d)
    ring_mean = jnp.mean(magnifications, axis=1)
    normalizer = jnp.sum(brightness)
    value = jnp.sum(brightness * ring_mean) / normalizer
    invalid = jnp.any(points.root_failure) | ~jnp.isfinite(value) | ~(normalizer > 0.0)
    return value, invalid


def _chord_quadrature(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
    *,
    order,
    root_backend,
):
    nodes_np, weights_np = np.polynomial.legendre.leggauss(order)
    dtype = jnp.result_type(source_x, source_y, separation, mass_ratio, source_radius)
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    weights = jnp.asarray(weights_np, dtype=dtype)
    eta = nodes[:, None]
    half_chord = jnp.sqrt(jnp.maximum(1.0 - eta * eta, 0.0))
    xi = half_chord * nodes[None, :]
    sample_x = source_x + source_radius * xi
    sample_y = source_y + source_radius * jnp.broadcast_to(eta, xi.shape)
    points = _point_samples(
        sample_x,
        sample_y,
        separation,
        mass_ratio,
        root_backend,
    )
    magnifications = points.magnification.reshape((order, order))
    radius_squared = xi * xi + eta * eta
    brightness = _surface_brightness(radius_squared, limb_c, limb_d)
    area_weights = weights[:, None] * weights[None, :] * half_chord
    normalizer = jnp.sum(area_weights * brightness)
    value = jnp.sum(area_weights * brightness * magnifications) / normalizer
    invalid = jnp.any(points.root_failure) | ~jnp.isfinite(value) | ~(normalizer > 0.0)
    return value, invalid


@partial(
    jax.jit,
    static_argnames=(
        "rule",
        "coarse_order",
        "fine_order",
        "angular_multiplier",
        "root_backend",
    ),
)
def binary_source_plane_quadrature(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    rule="ring",
    coarse_order=8,
    fine_order=16,
    angular_multiplier=4,
    absolute_tolerance=1.0e-4,
    relative_tolerance=1.0e-4,
    root_backend="auto",
) -> SourcePlaneQuadratureResult:
    """Integrate point-source magnification over the source disk.

    ``ring`` uses midpoint nodes uniform in squared radius and a fixed angular
    grid. ``chord`` uses the native lcbinint tensor Gauss--Legendre mapping.
    Both have static shapes and differentiate through the implicit image-root
    rule used by :func:`binary_point_source_magnification`.
    """

    require_x64()
    if coarse_order < 1 or fine_order <= coarse_order:
        raise ValueError("require 1 <= coarse_order < fine_order")
    if angular_multiplier < 1:
        raise ValueError("angular_multiplier must be positive")

    if rule == "ring":
        coarse, coarse_failure = _ring_quadrature(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            radial_bins=coarse_order,
            angular_bins=angular_multiplier * coarse_order,
            root_backend=root_backend,
        )
        fine, fine_failure = _ring_quadrature(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            radial_bins=fine_order,
            angular_bins=angular_multiplier * fine_order,
            root_backend=root_backend,
        )
        sample_count = angular_multiplier * (
            coarse_order * coarse_order + fine_order * fine_order
        )
        used_chord = False
    elif rule == "chord":
        coarse, coarse_failure = _chord_quadrature(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            order=coarse_order,
            root_backend=root_backend,
        )
        fine, fine_failure = _chord_quadrature(
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
            order=fine_order,
            root_backend=root_backend,
        )
        sample_count = coarse_order * coarse_order + fine_order * fine_order
        used_chord = True
    else:
        raise ValueError("rule must be 'ring' or 'chord'")

    root_failure = coarse_failure | fine_failure
    estimated_error = jnp.abs(fine - coarse)
    budget = absolute_tolerance + relative_tolerance * jnp.maximum(jnp.abs(fine), 1.0)
    converged = (
        ~root_failure & jnp.isfinite(estimated_error) & (estimated_error <= budget)
    )
    return SourcePlaneQuadratureResult(
        magnification=jnp.where(root_failure, jnp.nan, fine),
        coarse_magnification=coarse,
        estimated_error=estimated_error,
        sample_count=jnp.asarray(sample_count, dtype=jnp.int32),
        root_failure=root_failure,
        converged=converged,
        used_chord=jnp.asarray(used_chord),
    )

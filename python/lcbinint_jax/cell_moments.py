"""Differentiable positive-part moments for one inverse-ray cell."""

import jax
import jax.numpy as jnp

_POWERS = (0.0, 0.5, 0.25)


def _positive_power(value: jax.Array, power: jax.Array) -> jax.Array:
    positive = value > 0.0
    safe_value = jnp.where(positive, value, 1.0)
    return jnp.where(positive, jnp.power(safe_value, power), 0.0)


def _affine_unit_square_moment(
    lower_left: jax.Array,
    delta_x: jax.Array,
    delta_y: jax.Array,
    power: jax.Array,
) -> jax.Array:
    """Integrate ``[a + bx + cy]_+**power`` over the unit square."""

    scale = jnp.maximum(
        1.0e-14,
        jnp.maximum(
            jnp.abs(lower_left),
            jnp.maximum(jnp.abs(delta_x), jnp.abs(delta_y)),
        ),
    )
    slope_threshold = 1.0e-6 * scale
    x_small = jnp.abs(delta_x) <= slope_threshold
    y_small = jnp.abs(delta_y) <= slope_threshold

    exponent_2d = power + 2.0
    safe_delta_x = jnp.where(x_small, 1.0, delta_x)
    safe_delta_y = jnp.where(y_small, 1.0, delta_y)
    numerator_2d = (
        _positive_power(lower_left + delta_x + delta_y, exponent_2d)
        - _positive_power(lower_left + delta_x, exponent_2d)
        - _positive_power(lower_left + delta_y, exponent_2d)
        + _positive_power(lower_left, exponent_2d)
    )
    moment_2d = numerator_2d / (
        safe_delta_x * safe_delta_y * (power + 1.0) * (power + 2.0)
    )

    exponent_1d = power + 1.0
    midpoint_y_intercept = lower_left + 0.5 * delta_y
    midpoint_x_intercept = lower_left + 0.5 * delta_x
    moment_x = (
        _positive_power(midpoint_y_intercept + delta_x, exponent_1d)
        - _positive_power(midpoint_y_intercept, exponent_1d)
    ) / (safe_delta_x * (power + 1.0))
    moment_y = (
        _positive_power(midpoint_x_intercept + delta_y, exponent_1d)
        - _positive_power(midpoint_x_intercept, exponent_1d)
    ) / (safe_delta_y * (power + 1.0))

    centre = lower_left + 0.5 * (delta_x + delta_y)
    constant = jnp.where(
        power == 0.0,
        jnp.where(centre > 0.0, 1.0, 0.0),
        _positive_power(centre, power),
    )
    return jnp.where(
        x_small & y_small,
        constant,
        jnp.where(x_small, moment_y, jnp.where(y_small, moment_x, moment_2d)),
    )


def affine_cell_moments(
    phi_centre: jax.Array,
    phi_gradient_x: jax.Array,
    phi_gradient_y: jax.Array,
    cell_size: jax.Array,
) -> jax.Array:
    """Return ``(M0, M1/2, M1/4)`` for a locally affine source boundary."""

    phi_centre = jnp.asarray(phi_centre)
    phi_gradient_x = jnp.asarray(phi_gradient_x, dtype=phi_centre.dtype)
    phi_gradient_y = jnp.asarray(phi_gradient_y, dtype=phi_centre.dtype)
    cell_size = jnp.asarray(cell_size, dtype=phi_centre.dtype)
    frozen_cell_size = jax.lax.stop_gradient(cell_size)
    delta_x = phi_gradient_x * frozen_cell_size
    delta_y = phi_gradient_y * frozen_cell_size
    lower_left = phi_centre - 0.5 * (delta_x + delta_y)
    powers = jnp.asarray(_POWERS, dtype=phi_centre.dtype)
    unit_moments = jax.vmap(
        lambda power: _affine_unit_square_moment(lower_left, delta_x, delta_y, power)
    )(powers)
    return frozen_cell_size * frozen_cell_size * unit_moments


def midpoint_cell_moments(
    phi_centre: jax.Array,
    cell_size: jax.Array,
) -> jax.Array:
    """Midpoint moments for a cell known to lie fully inside the source."""

    phi_centre = jnp.asarray(phi_centre)
    cell_size = jnp.asarray(cell_size, dtype=phi_centre.dtype)
    frozen_cell_size = jax.lax.stop_gradient(cell_size)
    area = frozen_cell_size * frozen_cell_size
    positive = phi_centre > 0.0
    safe_phi = jnp.where(positive, phi_centre, 1.0)
    return area * jnp.stack(
        (
            positive.astype(phi_centre.dtype),
            jnp.where(positive, jnp.sqrt(safe_phi), 0.0),
            jnp.where(positive, jnp.power(safe_phi, 0.25), 0.0),
        )
    )


def resolved_cell_moments(
    phi_centre: jax.Array,
    phi_gradient_x: jax.Array,
    phi_gradient_y: jax.Array,
    cell_size: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select outside, midpoint-interior, or affine-boundary cell moments."""

    phi_centre = jnp.asarray(phi_centre)
    phi_gradient_x = jnp.asarray(phi_gradient_x, dtype=phi_centre.dtype)
    phi_gradient_y = jnp.asarray(phi_gradient_y, dtype=phi_centre.dtype)
    cell_size = jnp.asarray(cell_size, dtype=phi_centre.dtype)
    frozen_cell_size = jax.lax.stop_gradient(cell_size)
    half_delta_x = 0.5 * phi_gradient_x * frozen_cell_size
    half_delta_y = 0.5 * phi_gradient_y * frozen_cell_size
    corner_offsets = jnp.stack(
        (
            -half_delta_x - half_delta_y,
            -half_delta_x + half_delta_y,
            half_delta_x - half_delta_y,
            half_delta_x + half_delta_y,
        )
    )
    corner_phi = phi_centre + corner_offsets
    fully_inside = jnp.min(corner_phi) > 0.0
    fully_outside = jnp.max(corner_phi) <= 0.0
    boundary = ~(fully_inside | fully_outside)

    inside_moments = midpoint_cell_moments(phi_centre, frozen_cell_size)
    boundary_moments = affine_cell_moments(
        phi_centre, phi_gradient_x, phi_gradient_y, frozen_cell_size
    )
    moments = jnp.where(
        fully_outside,
        jnp.zeros_like(inside_moments),
        jnp.where(fully_inside, inside_moments, boundary_moments),
    )
    return moments, boundary, ~fully_outside

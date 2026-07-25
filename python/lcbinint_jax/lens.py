"""Binary-lens maps used by the JAX inverse-ray hot path."""

import jax
import jax.numpy as jnp


def binary_lens_positions_and_masses(
    separation: jax.Array,
    mass_ratio: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return center-of-mass lens positions and normalized masses."""

    total = 1.0 + mass_ratio
    mass_1 = 1.0 / total
    mass_2 = mass_ratio / total
    lens_1_x = -mass_2 * separation
    lens_2_x = mass_1 * separation
    return lens_1_x, lens_2_x, mass_1, mass_2


def binary_lens_map_complex(
    image: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
) -> jax.Array:
    """Map complex image-plane coordinates to the source plane."""

    lens_1_x, lens_2_x, mass_1, mass_2 = binary_lens_positions_and_masses(
        separation, mass_ratio
    )
    image_conjugate = jnp.conjugate(image)
    return (
        image
        - mass_1 / (image_conjugate - lens_1_x)
        - mass_2 / (image_conjugate - lens_2_x)
    )


def binary_lens_map_real(
    image_x: jax.Array,
    image_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Real-arithmetic binary lens map with structure-of-arrays output."""

    lens_1_x, lens_2_x, mass_1, mass_2 = binary_lens_positions_and_masses(
        separation, mass_ratio
    )
    dx_1 = image_x - lens_1_x
    dx_2 = image_x - lens_2_x
    radius_1_squared = dx_1 * dx_1 + image_y * image_y
    radius_2_squared = dx_2 * dx_2 + image_y * image_y
    source_x = (
        image_x - mass_1 * dx_1 / radius_1_squared - mass_2 * dx_2 / radius_2_squared
    )
    source_y = (
        image_y
        - mass_1 * image_y / radius_1_squared
        - mass_2 * image_y / radius_2_squared
    )
    return source_x, source_y


def binary_lens_map_and_derivatives_real(
    image_x: jax.Array,
    image_y: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the real lens map and its two-by-two image-plane Jacobian."""

    lens_1_x, lens_2_x, mass_1, mass_2 = binary_lens_positions_and_masses(
        separation, mass_ratio
    )
    dx_1 = image_x - lens_1_x
    dx_2 = image_x - lens_2_x
    y_squared = image_y * image_y
    radius_1_squared = dx_1 * dx_1 + y_squared
    radius_2_squared = dx_2 * dx_2 + y_squared
    inverse_radius_1_squared = 1.0 / radius_1_squared
    inverse_radius_2_squared = 1.0 / radius_2_squared

    source_x = (
        image_x
        - mass_1 * dx_1 * inverse_radius_1_squared
        - mass_2 * dx_2 * inverse_radius_2_squared
    )
    source_y = (
        image_y
        - mass_1 * image_y * inverse_radius_1_squared
        - mass_2 * image_y * inverse_radius_2_squared
    )

    shear_real = (
        mass_1
        * (dx_1 * dx_1 - y_squared)
        * inverse_radius_1_squared
        * inverse_radius_1_squared
        + mass_2
        * (dx_2 * dx_2 - y_squared)
        * inverse_radius_2_squared
        * inverse_radius_2_squared
    )
    shear_cross = (
        2.0
        * image_y
        * (
            mass_1 * dx_1 * inverse_radius_1_squared * inverse_radius_1_squared
            + mass_2 * dx_2 * inverse_radius_2_squared * inverse_radius_2_squared
        )
    )

    du_dx = 1.0 + shear_real
    du_dy = shear_cross
    dv_dx = shear_cross
    dv_dy = 1.0 - shear_real
    return source_x, source_y, du_dx, du_dy, dv_dx, dv_dy

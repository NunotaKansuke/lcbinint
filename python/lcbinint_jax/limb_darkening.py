"""Limb-darkening algebra based on image-plane brightness moments."""

import jax
import jax.numpy as jnp


def combine_limb_darkening_moments(
    moments: jax.Array,
    source_radius: jax.Array,
    limb_c: jax.Array = 0.0,
    limb_d: jax.Array = 0.0,
) -> jax.Array:
    """Combine ``(M0, M1/2, M1/4)`` into finite-source magnification."""

    uniform, mu, sqrt_mu = moments
    image_flux = (1.0 - limb_c - limb_d) * uniform + limb_c * mu + limb_d * sqrt_mu
    source_flux = (
        jnp.pi * source_radius * source_radius * (1.0 - limb_c / 3.0 - limb_d / 5.0)
    )
    valid = (source_radius > 0.0) & (source_flux > 0.0)
    return jnp.where(valid, image_flux / source_flux, jnp.nan)

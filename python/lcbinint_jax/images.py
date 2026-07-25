"""Stopped-gradient physical-image seeds for inverse-ray support discovery."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .lens import (
    binary_lens_map_and_derivatives_real,
    binary_lens_map_complex,
)
from .polynomial import (
    binary_lens_polynomial_coefficients,
    binary_lens_polynomial_roots,
)


class BinaryImages(NamedTuple):
    roots: jax.Array
    physical: jax.Array
    residuals: jax.Array
    root_converged: jax.Array
    iterations: jax.Array


def _newton_polish_binary_images(
    roots: jax.Array,
    root_converged: jax.Array,
    source: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    steps: int,
) -> jax.Array:
    def polish_step(_, active_roots):
        mapped_x, mapped_y, du_dx, du_dy, dv_dx, dv_dy = jax.vmap(
            lambda root: binary_lens_map_and_derivatives_real(
                jnp.real(root), jnp.imag(root), separation, mass_ratio
            )
        )(active_roots)
        residual_x = mapped_x - jnp.real(source)
        residual_y = mapped_y - jnp.imag(source)
        determinant = du_dx * dv_dy - du_dy * dv_dx
        safe_determinant = jnp.where(jnp.abs(determinant) > 1.0e-14, determinant, 1.0)
        delta_x = (dv_dy * residual_x - du_dy * residual_y) / safe_determinant
        delta_y = (-dv_dx * residual_x + du_dx * residual_y) / safe_determinant
        usable = (
            root_converged
            & (jnp.abs(determinant) > 1.0e-14)
            & jnp.isfinite(delta_x)
            & jnp.isfinite(delta_y)
        )
        step_scale = jnp.maximum(1.0, jnp.hypot(delta_x, delta_y) / 0.5)
        update = (delta_x + 1j * delta_y) / step_scale
        return jnp.where(usable, active_roots - update, active_roots)

    return jax.lax.fori_loop(0, steps, polish_step, roots)


@partial(
    jax.jit,
    static_argnames=("max_iterations", "polish_steps"),
)
def binary_images(
    source: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
    *,
    max_iterations: int = 80,
    tolerance: float = 1.0e-12,
    residual_tolerance: float = 1.0e-9,
    initial_phase: float = 0.0,
    polish_steps: int = 2,
) -> BinaryImages:
    """Return stopped-gradient physical images for one binary-lens source."""

    coefficients = binary_lens_polynomial_coefficients(source, separation, mass_ratio)
    solved = binary_lens_polynomial_roots(
        coefficients,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_phase=initial_phase,
    )
    polished = _newton_polish_binary_images(
        solved.roots,
        solved.converged,
        source,
        separation,
        mass_ratio,
        polish_steps,
    )
    mapped = binary_lens_map_complex(polished, separation, mass_ratio)
    residuals = jnp.abs(mapped - source)
    residual_limit = residual_tolerance * (1.0 + jnp.abs(source))
    physical = (
        solved.converged & jnp.isfinite(residuals) & (residuals <= residual_limit)
    )
    return jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        BinaryImages(
            roots=polished,
            physical=physical,
            residuals=residuals,
            root_converged=solved.converged,
            iterations=solved.iterations,
        ),
    )

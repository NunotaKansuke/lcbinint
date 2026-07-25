"""Static-degree polynomial construction and root solving for binary lenses."""

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from ._config import require_x64
from .lens import binary_lens_positions_and_masses


class PolynomialRoots(NamedTuple):
    roots: jax.Array
    converged: jax.Array
    iterations: jax.Array
    maximum_update: jax.Array


def binary_lens_polynomial_coefficients(
    source: jax.Array,
    separation: jax.Array,
    mass_ratio: jax.Array,
) -> jax.Array:
    """Return descending coefficients of the binary-lens quintic.

    The lens positions use the center-of-mass convention from
    :func:`lcbinint_jax.lens.binary_lens_positions_and_masses`.
    """

    require_x64()
    lens_1_x, lens_2_x, mass_1, mass_2 = binary_lens_positions_and_masses(
        separation, mass_ratio
    )
    dtype = jnp.result_type(source, jnp.complex128)
    lens_1_x = jnp.asarray(lens_1_x, dtype=dtype)
    lens_2_x = jnp.asarray(lens_2_x, dtype=dtype)
    mass_1 = jnp.asarray(mass_1, dtype=dtype)
    mass_2 = jnp.asarray(mass_2, dtype=dtype)

    factor_1 = jnp.stack((-lens_1_x, jnp.asarray(1.0, dtype=dtype)))
    factor_2 = jnp.stack((-lens_2_x, jnp.asarray(1.0, dtype=dtype)))
    denominator = jnp.convolve(factor_1, factor_2)
    numerator = (
        jnp.conjugate(source) * denominator
        + mass_1
        * jnp.stack(
            (-lens_2_x, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype))
        )
        + mass_2
        * jnp.stack(
            (-lens_1_x, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype))
        )
    )
    conjugate_offset_1 = numerator - lens_1_x * denominator
    conjugate_offset_2 = numerator - lens_2_x * denominator
    source_factor = jnp.stack((-source, jnp.asarray(1.0, dtype=dtype)))
    polynomial_ascending = jnp.convolve(
        jnp.convolve(source_factor, conjugate_offset_1),
        conjugate_offset_2,
    )
    deflection_terms = mass_1 * jnp.convolve(
        denominator, conjugate_offset_2
    ) + mass_2 * jnp.convolve(denominator, conjugate_offset_1)
    polynomial_ascending = polynomial_ascending.at[:5].add(-deflection_terms)
    return polynomial_ascending[::-1]


def _polynomial_derivative_coefficients(coefficients: jax.Array) -> jax.Array:
    degree = coefficients.shape[0] - 1
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.real.dtype)
    return coefficients[:-1] * powers


@partial(jax.jit, static_argnames=("max_iterations",))
def polynomial_roots_ehrlich_aberth(
    coefficients: jax.Array,
    *,
    max_iterations: int = 80,
    tolerance: float = 1.0e-12,
    initial_phase: float = 0.0,
    initial_roots: Optional[jax.Array] = None,
) -> PolynomialRoots:
    """Find all roots with a bounded, static-shape Ehrlich-Aberth iteration."""

    coefficients = jnp.asarray(coefficients)
    degree = coefficients.shape[0] - 1
    leading = coefficients[0]
    safe_leading = jnp.where(jnp.abs(leading) > 0.0, leading, 1.0 + 0.0j)
    normalized = coefficients / safe_leading
    derivative = _polynomial_derivative_coefficients(normalized)

    if initial_roots is None:
        radius = 1.0 + jnp.max(jnp.abs(normalized[1:]))
        angles = (
            initial_phase
            + 2.0 * jnp.pi * jnp.arange(degree, dtype=normalized.real.dtype) / degree
        )
        roots_0 = radius * jnp.exp(1j * angles)
    else:
        roots_0 = jnp.asarray(initial_roots, dtype=normalized.dtype)

    real_dtype = normalized.real.dtype
    epsilon = 64.0 * jnp.finfo(real_dtype).eps
    tolerance_array = jnp.asarray(tolerance, dtype=real_dtype)

    def iteration(_, state):
        roots, converged, iterations, maximum_update = state
        values = jnp.polyval(normalized, roots)
        derivatives = jnp.polyval(derivative, roots)
        differences = roots[:, None] - roots[None, :]
        identity = jnp.eye(degree, dtype=bool)
        safe_differences = jnp.where(identity, 1.0 + 0.0j, differences)
        reciprocal_sum = jnp.sum(
            jnp.where(identity, 0.0 + 0.0j, 1.0 / safe_differences),
            axis=1,
        )
        denominator = derivatives - values * reciprocal_sum
        safe_denominator = jnp.where(
            jnp.abs(denominator) > epsilon,
            denominator,
            jnp.where(jnp.abs(derivatives) > epsilon, derivatives, 1.0 + 0.0j),
        )
        update = values / safe_denominator
        proposed_roots = roots - update
        step_valid = jnp.isfinite(jnp.abs(update)) & jnp.isfinite(
            jnp.abs(proposed_roots)
        )
        active_update = jnp.where(converged | ~step_valid, 0.0 + 0.0j, update)
        roots_next = roots - active_update
        update_size = jnp.abs(active_update)
        newly_converged = step_valid & (
            update_size <= tolerance_array * (1.0 + jnp.abs(roots_next))
        )
        converged_next = converged | newly_converged
        iterations_next = jnp.where(
            converged, iterations, iterations + jnp.asarray(1, dtype=jnp.int32)
        )
        maximum_update_next = jnp.max(update_size)
        return (
            roots_next,
            converged_next,
            iterations_next,
            maximum_update_next,
        )

    initial_state = (
        roots_0,
        jnp.zeros(degree, dtype=bool),
        jnp.zeros(degree, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=real_dtype),
    )
    roots, converged, iterations, maximum_update = jax.lax.fori_loop(
        0, max_iterations, iteration, initial_state
    )
    valid_polynomial = jnp.isfinite(jnp.abs(leading)) & (jnp.abs(leading) > epsilon)
    converged = converged & valid_polynomial & jnp.isfinite(jnp.abs(roots))
    return PolynomialRoots(roots, converged, iterations, maximum_update)


@partial(jax.jit, static_argnames=("max_iterations",))
def binary_lens_polynomial_roots(
    coefficients: jax.Array,
    *,
    max_iterations: int = 80,
    tolerance: float = 1.0e-12,
    initial_phase: float = 0.0,
    degree_drop_tolerance: float = 1.0e-10,
) -> PolynomialRoots:
    """Solve a binary quintic, dropping its vanishing leading term if needed.

    For sources on or extremely close to the binary axis, the formal quintic
    reduces to a quartic and its fifth algebraic root moves to infinity.
    Returning a padded, invalid fifth slot keeps discovery arrays static.
    """

    coefficient_scale = jnp.max(jnp.abs(coefficients))
    use_quartic = jnp.abs(coefficients[0]) <= (
        degree_drop_tolerance * jnp.maximum(coefficient_scale, 1.0e-30)
    )

    def solve_quintic(active_coefficients):
        return polynomial_roots_ehrlich_aberth(
            active_coefficients,
            max_iterations=max_iterations,
            tolerance=tolerance,
            initial_phase=initial_phase,
        )

    def solve_quartic(active_coefficients):
        solved = polynomial_roots_ehrlich_aberth(
            active_coefficients[1:],
            max_iterations=max_iterations,
            tolerance=tolerance,
            initial_phase=initial_phase,
        )
        return PolynomialRoots(
            roots=jnp.concatenate(
                (solved.roots, jnp.zeros(1, dtype=solved.roots.dtype))
            ),
            converged=jnp.concatenate((solved.converged, jnp.zeros(1, dtype=bool))),
            iterations=jnp.concatenate(
                (solved.iterations, jnp.zeros(1, dtype=jnp.int32))
            ),
            maximum_update=solved.maximum_update,
        )

    return jax.lax.cond(use_quartic, solve_quartic, solve_quintic, coefficients)

"""Coarse/fine convergence diagnostics for the inverse-ray MVP."""

from functools import partial

import jax
import jax.numpy as jnp

from ._config import require_x64
from .api import binary_inverse_ray
from .types import ConvergenceResult


def _normalized_observables(result, source_radius):
    """Return magnification and three source-area-normalized image moments."""

    source_area = jnp.pi * source_radius * source_radius
    return jnp.concatenate(
        (
            jnp.reshape(result.magnification, (1,)),
            result.moments / source_area,
        )
    )


@partial(
    jax.jit,
    static_argnames=(
        "coarse_resolution",
        "fine_resolution",
        "tile_size",
        "coarse_tile_capacity",
        "fine_tile_capacity",
        "limb_samples",
        "kernel",
        "check_gradient",
    ),
)
def _binary_inverse_ray_convergence(
    parameters,
    direction,
    *,
    coarse_resolution,
    fine_resolution,
    tile_size,
    coarse_tile_capacity,
    fine_tile_capacity,
    limb_samples,
    kernel,
    atol,
    rtol,
    gradient_atol,
    gradient_rtol,
    check_gradient,
):
    def evaluate(active_parameters, resolution, tile_capacity):
        result = binary_inverse_ray(
            *active_parameters,
            resolution=resolution,
            tile_size=tile_size,
            tile_capacity=tile_capacity,
            limb_samples=limb_samples,
            kernel=kernel,
        )
        observables = _normalized_observables(result, active_parameters[4])
        auxiliary = (result.tile_count, result.support_valid)
        return observables, auxiliary

    def evaluate_coarse(active_parameters):
        return evaluate(active_parameters, coarse_resolution, coarse_tile_capacity)

    def evaluate_fine(active_parameters):
        return evaluate(active_parameters, fine_resolution, fine_tile_capacity)

    if check_gradient:
        coarse, coarse_derivative, coarse_auxiliary = jax.jvp(
            evaluate_coarse,
            (parameters,),
            (direction,),
            has_aux=True,
        )
        fine, fine_derivative, fine_auxiliary = jax.jvp(
            evaluate_fine,
            (parameters,),
            (direction,),
            has_aux=True,
        )
    else:
        coarse, coarse_auxiliary = evaluate_coarse(parameters)
        fine, fine_auxiliary = evaluate_fine(parameters)
        coarse_derivative = jnp.full_like(coarse, jnp.nan)
        fine_derivative = jnp.full_like(fine, jnp.nan)

    coarse_tile_count, coarse_support_valid = coarse_auxiliary
    fine_tile_count, fine_support_valid = fine_auxiliary
    support_valid = coarse_support_valid & fine_support_valid

    observable_errors = jnp.abs(fine - coarse)
    observable_budgets = atol + rtol * jnp.maximum(jnp.abs(fine), 1.0)
    observable_converged = support_valid & (observable_errors <= observable_budgets)

    derivative_errors = jnp.abs(fine_derivative - coarse_derivative)
    derivative_budgets = gradient_atol + gradient_rtol * jnp.maximum(
        jnp.abs(fine_derivative), 1.0
    )
    derivative_converged = support_valid & (derivative_errors <= derivative_budgets)
    gradient_checked = jnp.asarray(check_gradient)

    return ConvergenceResult(
        coarse_observables=coarse,
        fine_observables=fine,
        observable_errors=observable_errors,
        observable_budgets=observable_budgets,
        coarse_directional_derivatives=coarse_derivative,
        fine_directional_derivatives=fine_derivative,
        derivative_errors=derivative_errors,
        derivative_budgets=derivative_budgets,
        coarse_tile_count=coarse_tile_count,
        fine_tile_count=fine_tile_count,
        coarse_support_valid=coarse_support_valid,
        fine_support_valid=fine_support_valid,
        value_converged=observable_converged[0],
        moments_converged=jnp.all(observable_converged[1:]),
        gradient_checked=gradient_checked,
        gradient_converged=gradient_checked & jnp.all(derivative_converged),
    )


def binary_inverse_ray_convergence(
    parameters,
    *,
    direction=None,
    coarse_resolution=32,
    fine_resolution=64,
    tile_size=16,
    coarse_tile_capacity=1024,
    fine_tile_capacity=2048,
    limb_samples=16,
    kernel="real",
    atol=1.0e-3,
    rtol=1.0e-3,
    gradient_atol=1.0e-3,
    gradient_rtol=1.0e-3,
):
    """Compare two resolution buckets, optionally including a directional JVP.

    The four observables are magnification followed by
    ``(M0, M1/2, M1/4) / (pi * rho**2)``. A direction, when supplied, must
    follow the parameter order ``(x, y, s, q, rho, c, d)``.
    """

    require_x64()
    parameters = jnp.asarray(parameters)
    if parameters.shape != (7,):
        raise ValueError("parameters must have shape (7,)")
    if fine_resolution <= coarse_resolution:
        raise ValueError("fine_resolution must be greater than coarse_resolution")

    check_gradient = direction is not None
    if direction is None:
        direction_array = jnp.zeros_like(parameters)
    else:
        direction_array = jnp.asarray(direction, dtype=parameters.dtype)
        if direction_array.shape != parameters.shape:
            raise ValueError("direction must have shape (7,)")

    return _binary_inverse_ray_convergence(
        parameters,
        direction_array,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
        tile_size=tile_size,
        coarse_tile_capacity=coarse_tile_capacity,
        fine_tile_capacity=fine_tile_capacity,
        limb_samples=limb_samples,
        kernel=kernel,
        atol=atol,
        rtol=rtol,
        gradient_atol=gradient_atol,
        gradient_rtol=gradient_rtol,
        check_gradient=check_gradient,
    )

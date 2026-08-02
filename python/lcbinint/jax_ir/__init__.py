"""Compatibility namespace for the internal :mod:`lcbinint_jax` kernels."""

from lcbinint_jax import (
    ConvergenceResult,
    DiscoveryResult,
    FixedSupportResult,
    InverseRayResult,
    binary_inverse_ray,
    binary_inverse_ray_convergence,
    binary_inverse_ray_fixed_support,
    binary_inverse_ray_linear,
    binary_inverse_ray_uniform,
    combine_limb_darkening_moments,
    discover_binary_macro_tiles,
)

__all__ = [
    "ConvergenceResult",
    "DiscoveryResult",
    "FixedSupportResult",
    "InverseRayResult",
    "binary_inverse_ray",
    "binary_inverse_ray_convergence",
    "binary_inverse_ray_fixed_support",
    "binary_inverse_ray_linear",
    "binary_inverse_ray_uniform",
    "combine_limb_darkening_moments",
    "discover_binary_macro_tiles",
]

"""Compatibility namespace for the standalone :mod:`lcbinint_jax` package."""

from lcbinint_jax import (
    DiscoveryResult,
    FixedSupportResult,
    InverseRayResult,
    binary_inverse_ray,
    binary_inverse_ray_fixed_support,
    combine_limb_darkening_moments,
    discover_binary_macro_tiles,
)

__all__ = [
    "DiscoveryResult",
    "FixedSupportResult",
    "InverseRayResult",
    "binary_inverse_ray",
    "binary_inverse_ray_fixed_support",
    "combine_limb_darkening_moments",
    "discover_binary_macro_tiles",
]

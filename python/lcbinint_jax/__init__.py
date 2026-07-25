"""CPU-oriented differentiable inverse-ray integration in JAX."""

from .api import binary_inverse_ray, binary_inverse_ray_linear
from .convergence import binary_inverse_ray_convergence
from .discovery import discover_binary_macro_tiles
from .integrate import binary_inverse_ray_fixed_support
from .limb_darkening import combine_limb_darkening_moments
from .types import (
    ConvergenceResult,
    DiscoveryResult,
    FixedSupportResult,
    InverseRayResult,
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
    "combine_limb_darkening_moments",
    "discover_binary_macro_tiles",
]

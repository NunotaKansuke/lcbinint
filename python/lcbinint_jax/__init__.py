"""CPU-oriented differentiable inverse-ray integration in JAX."""

from .api import (
    binary_magnification_auto,
    binary_inverse_ray,
    binary_inverse_ray_auto,
    binary_inverse_ray_linear,
    binary_inverse_ray_uniform,
)
from .multipole import (
    HexadecapoleResult,
    PointSourceResult,
    binary_hexadecapole,
    binary_point_source_magnification,
)
from .convergence import binary_inverse_ray_convergence
from .discovery import discover_binary_macro_tiles
from .integrate import binary_inverse_ray_fixed_support
from .limb_darkening import combine_limb_darkening_moments
from .polar import binary_inverse_ray_polar, discover_binary_polar_bands
from .types import (
    ConvergenceResult,
    AutoInverseRayResult,
    DiscoveryResult,
    FixedSupportResult,
    InverseRayResult,
    HybridMagnificationResult,
    PolarSupportResult,
)

__all__ = [
    "AutoInverseRayResult",
    "ConvergenceResult",
    "DiscoveryResult",
    "FixedSupportResult",
    "HexadecapoleResult",
    "HybridMagnificationResult",
    "InverseRayResult",
    "PointSourceResult",
    "PolarSupportResult",
    "binary_hexadecapole",
    "binary_magnification_auto",
    "binary_point_source_magnification",
    "binary_inverse_ray",
    "binary_inverse_ray_auto",
    "binary_inverse_ray_convergence",
    "binary_inverse_ray_fixed_support",
    "binary_inverse_ray_linear",
    "binary_inverse_ray_polar",
    "binary_inverse_ray_uniform",
    "combine_limb_darkening_moments",
    "discover_binary_macro_tiles",
    "discover_binary_polar_bands",
]

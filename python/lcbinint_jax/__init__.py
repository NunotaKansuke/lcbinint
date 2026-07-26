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
from .cpp_backend import (
    CppFixedSupportResult,
    binary_images_ffi,
    binary_inverse_ray_cartesian_batch_ffi,
    binary_inverse_ray_cartesian_ffi,
    binary_inverse_ray_fixed_support_cpp,
    binary_inverse_ray_fixed_support_ffi,
    cpp_binary_image_roots_ffi_available,
    cpp_cartesian_batch_ffi_available,
    cpp_cartesian_epoch_ffi_available,
    cpp_fixed_support_ffi_available,
    cpp_macro_tile_discovery_ffi_available,
    discover_binary_macro_tiles_ffi,
)
from .discovery import discover_binary_macro_tiles
from .integrate import binary_inverse_ray_fixed_support
from .limb_darkening import combine_limb_darkening_moments
from .polar import binary_inverse_ray_polar, discover_binary_polar_bands
from .source_plane import (
    SourcePlaneQuadratureResult,
    binary_source_plane_quadrature,
)
from .trajectory import binary_magnification_trajectory
from .types import (
    ConvergenceResult,
    AutoInverseRayResult,
    DiscoveryResult,
    FixedSupportResult,
    InverseRayResult,
    HybridMagnificationResult,
    PolarSupportResult,
    TrajectoryMagnificationResult,
)

__all__ = [
    "AutoInverseRayResult",
    "ConvergenceResult",
    "CppFixedSupportResult",
    "DiscoveryResult",
    "FixedSupportResult",
    "HexadecapoleResult",
    "HybridMagnificationResult",
    "InverseRayResult",
    "PointSourceResult",
    "PolarSupportResult",
    "SourcePlaneQuadratureResult",
    "TrajectoryMagnificationResult",
    "binary_hexadecapole",
    "binary_images_ffi",
    "binary_inverse_ray_cartesian_batch_ffi",
    "binary_inverse_ray_cartesian_ffi",
    "binary_magnification_auto",
    "binary_magnification_trajectory",
    "binary_point_source_magnification",
    "binary_source_plane_quadrature",
    "binary_inverse_ray",
    "binary_inverse_ray_auto",
    "binary_inverse_ray_convergence",
    "binary_inverse_ray_fixed_support",
    "binary_inverse_ray_fixed_support_cpp",
    "binary_inverse_ray_fixed_support_ffi",
    "binary_inverse_ray_linear",
    "binary_inverse_ray_polar",
    "binary_inverse_ray_uniform",
    "combine_limb_darkening_moments",
    "cpp_binary_image_roots_ffi_available",
    "cpp_cartesian_batch_ffi_available",
    "cpp_cartesian_epoch_ffi_available",
    "cpp_fixed_support_ffi_available",
    "cpp_macro_tile_discovery_ffi_available",
    "discover_binary_macro_tiles",
    "discover_binary_macro_tiles_ffi",
    "discover_binary_polar_bands",
]

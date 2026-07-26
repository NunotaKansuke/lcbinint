"""Experimental native execution backend for fixed-support inverse rays.

This interface is intentionally NumPy-callable and not yet JAX-transformable.
It is the raw-kernel performance and semantic gate before a JAX CPU FFI call
and analytic differentiation rules are added.
"""

from typing import NamedTuple

import numpy as np


class CppFixedSupportResult(NamedTuple):
    """Host result returned by the experimental C++ forward kernel."""

    magnification: float
    moments: np.ndarray
    boundary_cells: int
    active_cells: int


def binary_inverse_ray_fixed_support_cpp(
    tile_origins,
    tile_mask,
    cell_size,
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    tile_size=8,
    moment_mode="two_coefficient",
    boundary_subdivision=4,
):
    """Run the fixed-support JAX cell algorithm in the C++ prototype.

    Arrays and scalar parameters are copied/coerced to host NumPy values.
    Consequently this prototype cannot be used under ``jax.jit`` or
    differentiated.  Those capabilities belong to the subsequent FFI/JVP
    implementation gate.
    """

    try:
        from lcbinint import _native
    except ImportError as error:
        raise RuntimeError(
            "the lcbinint native extension is required for the C++ backend"
        ) from error

    values = _native._jax_ir.fixed_support_forward(
        np.asarray(tile_origins, dtype=np.float64, order="C"),
        np.asarray(tile_mask, dtype=np.bool_, order="C"),
        float(cell_size),
        float(source_x),
        float(source_y),
        float(separation),
        float(mass_ratio),
        float(source_radius),
        float(limb_c),
        float(limb_d),
        tile_size=tile_size,
        moment_mode=moment_mode,
        boundary_subdivision=boundary_subdivision,
    )
    return CppFixedSupportResult(*values)

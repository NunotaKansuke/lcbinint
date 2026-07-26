"""Experimental native execution backends for fixed-support inverse rays.

The host interface is NumPy-callable.  The typed CPU FFI interface supports
``jax.jit`` and sequential ``jax.vmap``; analytic differentiation rules are
the next implementation gate.
"""

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._config import require_x64
from .types import FixedSupportResult

_FFI_TARGET = "lcbinint_jax_fixed_support_forward_f64_v1"
_MOMENT_COUNTS = {
    "uniform": 1,
    "linear": 2,
    "two_coefficient": 3,
}


class CppFixedSupportResult(NamedTuple):
    """Host result returned by the experimental C++ forward kernel."""

    magnification: float
    moments: np.ndarray
    boundary_cells: int
    active_cells: int


def _native_module():
    try:
        from lcbinint import _native
    except ImportError as error:
        raise RuntimeError(
            "the lcbinint native extension is required for the C++ backend"
        ) from error
    return _native


@lru_cache(maxsize=1)
def _register_fixed_support_ffi():
    native = _native_module()
    try:
        capsule = native._jax_ir.fixed_support_forward_ffi()
    except AttributeError as error:
        raise RuntimeError(
            "lcbinint was built without JAX FFI headers; rebuild with "
            "LCBININT_ENABLE_JAX_FFI=ON"
        ) from error
    jax.ffi.register_ffi_target(_FFI_TARGET, capsule, platform="cpu")


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

    native = _native_module()

    values = native._jax_ir.fixed_support_forward(
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


def binary_inverse_ray_fixed_support_ffi(
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
    """Run the C++ fixed-support forward kernel inside JAX CPU programs.

    The typed FFI call supports ``jax.jit`` and sequential ``jax.vmap``.
    Analytic JVP/VJP rules are a separate implementation gate, so attempting
    to differentiate this function currently raises JAX's explicit FFI
    differentiation error.
    """

    require_x64()
    if jax.default_backend() != "cpu":
        raise RuntimeError("the experimental fixed-support FFI is CPU-only")
    if moment_mode not in _MOMENT_COUNTS:
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if boundary_subdivision not in (1, 2, 3, 4):
        raise ValueError("boundary_subdivision must be 1, 2, 3, or 4")

    origins = jax.lax.stop_gradient(jnp.asarray(tile_origins, dtype=jnp.float64))
    mask = jax.lax.stop_gradient(jnp.asarray(tile_mask, dtype=jnp.bool_))
    if origins.ndim != 2 or origins.shape[1] != 2:
        raise ValueError("tile_origins must have shape (N, 2)")
    if mask.ndim != 1 or mask.shape[0] != origins.shape[0]:
        raise ValueError("tile_mask must have shape (N,)")
    scalars = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        )
    )
    if any(value.ndim != 0 for value in scalars):
        raise ValueError("physical parameters must be scalars")

    _register_fixed_support_ffi()
    moment_count = _MOMENT_COUNTS[moment_mode]
    output_specifications = (
        jax.ShapeDtypeStruct((), jnp.float64),
        jax.ShapeDtypeStruct((moment_count,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    magnification, moments, boundary_cells, active_cells = jax.ffi.ffi_call(
        _FFI_TARGET,
        output_specifications,
        vmap_method="sequential",
    )(
        origins,
        mask,
        *scalars,
        tile_size=np.int64(tile_size),
        moment_mode=np.int64(moment_count),
        boundary_subdivision=np.int64(boundary_subdivision),
    )
    return FixedSupportResult(
        magnification=magnification,
        moments=moments,
        boundary_cells=boundary_cells,
        active_cells=active_cells,
    )

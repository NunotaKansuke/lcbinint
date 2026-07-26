"""Experimental native execution backends for fixed-support inverse rays.

The host interface is NumPy-callable.  The typed CPU FFI interface supports
``jax.jit``, sequential ``jax.vmap``, and analytic forward/reverse automatic
differentiation through a custom JVP.
"""

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._config import require_x64
from .types import FixedSupportResult

_FFI_FORWARD_TARGET = "lcbinint_jax_fixed_support_forward_f64_v1"
_FFI_VALUE_JACOBIAN_TARGET = "lcbinint_jax_fixed_support_value_jacobian_f64_v1"
_FFI_DISCOVERY_TARGET = "lcbinint_jax_macro_tile_discovery_f64_v1"
_FFI_BINARY_ROOT_TARGET = "lcbinint_jax_binary_image_roots_f64_v1"
_FFI_BINARY_ROOT_JACOBIAN_TARGET = "lcbinint_jax_binary_image_roots_jacobian_f64_v1"
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


class _FfiBinaryRootResult(NamedTuple):
    coordinates: jax.Array
    physical: jax.Array
    residuals: jax.Array
    converged: jax.Array


def _native_module():
    try:
        from lcbinint import _native
    except ImportError as error:
        raise RuntimeError(
            "the lcbinint native extension is required for the C++ backend"
        ) from error
    return _native


def cpp_fixed_support_ffi_available():
    """Return whether the typed fixed-support FFI can run on this JAX backend."""

    if jax.default_backend() != "cpu":
        return False
    try:
        native = _native_module()
        jax_ir = native._jax_ir
        return hasattr(jax_ir, "fixed_support_forward_ffi") and hasattr(
            jax_ir, "fixed_support_value_jacobian_ffi"
        )
    except (AttributeError, RuntimeError):
        return False


def cpp_macro_tile_discovery_ffi_available():
    """Return whether the typed macro-tile discovery FFI is available."""

    if jax.default_backend() != "cpu":
        return False
    try:
        native = _native_module()
        return hasattr(native._jax_ir, "macro_tile_discovery_ffi")
    except (AttributeError, RuntimeError):
        return False


def cpp_binary_image_roots_ffi_available():
    """Return whether binary-image roots and their Jacobian can run via FFI."""

    if jax.default_backend() != "cpu":
        return False
    try:
        native = _native_module()
        jax_ir = native._jax_ir
        return hasattr(jax_ir, "binary_image_roots_ffi") and hasattr(
            jax_ir, "binary_image_roots_jacobian_ffi"
        )
    except (AttributeError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def _register_fixed_support_ffi():
    native = _native_module()
    try:
        forward_capsule = native._jax_ir.fixed_support_forward_ffi()
        value_jacobian_capsule = native._jax_ir.fixed_support_value_jacobian_ffi()
    except AttributeError as error:
        raise RuntimeError(
            "lcbinint was built without JAX FFI headers; rebuild with "
            "LCBININT_ENABLE_JAX_FFI=ON"
        ) from error
    jax.ffi.register_ffi_target(_FFI_FORWARD_TARGET, forward_capsule, platform="cpu")
    jax.ffi.register_ffi_target(
        _FFI_VALUE_JACOBIAN_TARGET,
        value_jacobian_capsule,
        platform="cpu",
    )


@lru_cache(maxsize=1)
def _register_macro_tile_discovery_ffi():
    native = _native_module()
    try:
        discovery_capsule = native._jax_ir.macro_tile_discovery_ffi()
    except AttributeError as error:
        raise RuntimeError(
            "lcbinint was built without the macro-tile discovery FFI; rebuild "
            "with LCBININT_ENABLE_JAX_FFI=ON"
        ) from error
    jax.ffi.register_ffi_target(
        _FFI_DISCOVERY_TARGET,
        discovery_capsule,
        platform="cpu",
    )


@lru_cache(maxsize=1)
def _register_binary_image_roots_ffi():
    native = _native_module()
    try:
        forward_capsule = native._jax_ir.binary_image_roots_ffi()
        jacobian_capsule = native._jax_ir.binary_image_roots_jacobian_ffi()
    except AttributeError as error:
        raise RuntimeError(
            "lcbinint was built without the binary-image root FFI; rebuild "
            "with LCBININT_ENABLE_JAX_FFI=ON"
        ) from error
    jax.ffi.register_ffi_target(
        _FFI_BINARY_ROOT_TARGET,
        forward_capsule,
        platform="cpu",
    )
    jax.ffi.register_ffi_target(
        _FFI_BINARY_ROOT_JACOBIAN_TARGET,
        jacobian_capsule,
        platform="cpu",
    )


def _binary_root_output_specifications(include_jacobian):
    outputs = (
        jax.ShapeDtypeStruct((5, 2), jnp.float64),
        jax.ShapeDtypeStruct((5,), jnp.bool_),
        jax.ShapeDtypeStruct((5,), jnp.float64),
        jax.ShapeDtypeStruct((5,), jnp.bool_),
    )
    if include_jacobian:
        return outputs + (jax.ShapeDtypeStruct((5, 2, 4), jnp.float64),)
    return outputs


def _binary_root_ffi_call(target, scalars, *, include_jacobian):
    return jax.ffi.ffi_call(
        target,
        _binary_root_output_specifications(include_jacobian),
        vmap_method="sequential",
    )(*scalars)


@jax.custom_jvp
def _binary_image_roots_ffi_transformable(
    source_x,
    source_y,
    separation,
    mass_ratio,
):
    outputs = _binary_root_ffi_call(
        _FFI_BINARY_ROOT_TARGET,
        (source_x, source_y, separation, mass_ratio),
        include_jacobian=False,
    )
    return _FfiBinaryRootResult(*outputs)


@_binary_image_roots_ffi_transformable.defjvp
def _binary_image_roots_ffi_jvp(primals, tangents):
    outputs = _binary_root_ffi_call(
        _FFI_BINARY_ROOT_JACOBIAN_TARGET,
        primals,
        include_jacobian=True,
    )
    primal = _FfiBinaryRootResult(*outputs[:4])
    parameter_tangent = jnp.stack(tangents)
    coordinate_tangent = jnp.tensordot(
        outputs[4],
        parameter_tangent,
        axes=1,
    )
    tangent = _FfiBinaryRootResult(
        coordinates=coordinate_tangent,
        physical=jnp.zeros_like(primal.physical, dtype=jax.dtypes.float0),
        residuals=jnp.zeros_like(primal.residuals),
        converged=jnp.zeros_like(primal.converged, dtype=jax.dtypes.float0),
    )
    return primal, tangent


def binary_images_ffi(source, separation, mass_ratio):
    """Solve binary-lens images with implicit root derivatives in C++."""

    from .images import BinaryImages

    require_x64()
    if jax.default_backend() != "cpu":
        raise RuntimeError("the binary-image root FFI is CPU-only")
    source = jnp.asarray(source, dtype=jnp.complex128)
    scalars = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            jnp.real(source),
            jnp.imag(source),
            separation,
            mass_ratio,
        )
    )
    if source.ndim != 0 or any(value.ndim != 0 for value in scalars):
        raise ValueError("source and lens parameters must be scalars")
    _register_binary_image_roots_ffi()
    result = _binary_image_roots_ffi_transformable(*scalars)
    return BinaryImages(
        roots=result.coordinates[:, 0] + 1j * result.coordinates[:, 1],
        physical=jax.lax.stop_gradient(result.physical),
        residuals=jax.lax.stop_gradient(result.residuals),
        root_converged=jax.lax.stop_gradient(result.converged),
        iterations=jnp.zeros((5,), dtype=jnp.int32),
    )


def discover_binary_macro_tiles_ffi(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    cell_size,
    *,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    root_backend="auto",
):
    """Discover stopped-gradient Cartesian support with a typed CPU FFI BFS."""

    from .discovery import binary_image_seed_points
    from .types import DiscoveryResult

    require_x64()
    if jax.default_backend() != "cpu":
        raise RuntimeError("the macro-tile discovery FFI is CPU-only")
    if tile_size <= 0 or tile_capacity <= 0:
        raise ValueError("tile_size and tile_capacity must be positive")

    frozen_cell_size = jax.lax.stop_gradient(jnp.asarray(cell_size, dtype=jnp.float64))
    tile_width = frozen_cell_size * tile_size
    seeds = binary_image_seed_points(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_samples=limb_samples,
        root_backend=root_backend,
    )
    seed_coordinates = jnp.stack(
        (jnp.real(seeds.roots), jnp.imag(seeds.roots)),
        axis=1,
    )
    stopped_scalars = tuple(
        jax.lax.stop_gradient(jnp.asarray(value, dtype=jnp.float64))
        for value in (
            tile_width,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
        )
    )
    _register_macro_tile_discovery_ffi()
    outputs = jax.ffi.ffi_call(
        _FFI_DISCOVERY_TARGET,
        (
            jax.ShapeDtypeStruct((tile_capacity, 2), jnp.int32),
            jax.ShapeDtypeStruct((tile_capacity, 2), jnp.float64),
            jax.ShapeDtypeStruct((tile_capacity,), jnp.bool_),
            jax.ShapeDtypeStruct((tile_capacity,), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
        ),
        vmap_method="sequential",
    )(
        seed_coordinates,
        seeds.physical,
        *stopped_scalars,
    )
    return jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        DiscoveryResult(
            tile_indices=outputs[0],
            tile_origins=outputs[1],
            tile_mask=outputs[2],
            active_mask=outputs[3],
            overflow=outputs[4],
            visited_count=outputs[5],
            active_count=outputs[6],
            seed_count=outputs[7],
            root_failure=seeds.root_failure,
        ),
    )


def _output_specifications(moment_count, include_jacobian):
    outputs = (
        jax.ShapeDtypeStruct((), jnp.float64),
        jax.ShapeDtypeStruct((moment_count,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    if include_jacobian:
        return outputs + (
            jax.ShapeDtypeStruct((7,), jnp.float64),
            jax.ShapeDtypeStruct((moment_count, 7), jnp.float64),
        )
    return outputs


def _ffi_call(
    target,
    tile_size,
    moment_count,
    boundary_subdivision,
    origins,
    mask,
    scalars,
    *,
    include_jacobian,
):
    return jax.ffi.ffi_call(
        target,
        _output_specifications(moment_count, include_jacobian),
        vmap_method="sequential",
    )(
        origins,
        mask,
        *scalars,
        tile_size=np.int64(tile_size),
        moment_mode=np.int64(moment_count),
        boundary_subdivision=np.int64(boundary_subdivision),
    )


def _as_fixed_support_result(outputs):
    return FixedSupportResult(
        magnification=outputs[0],
        moments=outputs[1],
        boundary_cells=outputs[2],
        active_cells=outputs[3],
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _fixed_support_ffi_transformable(
    tile_size,
    moment_count,
    boundary_subdivision,
    origins,
    mask,
    cell_size,
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
):
    outputs = _ffi_call(
        _FFI_FORWARD_TARGET,
        tile_size,
        moment_count,
        boundary_subdivision,
        origins,
        mask,
        (
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        ),
        include_jacobian=False,
    )
    return _as_fixed_support_result(outputs)


@_fixed_support_ffi_transformable.defjvp
def _fixed_support_ffi_jvp(
    tile_size,
    moment_count,
    boundary_subdivision,
    primals,
    tangents,
):
    (
        origins,
        mask,
        cell_size,
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
    ) = primals
    (
        _,
        _,
        _,
        source_x_tangent,
        source_y_tangent,
        separation_tangent,
        mass_ratio_tangent,
        source_radius_tangent,
        limb_c_tangent,
        limb_d_tangent,
    ) = tangents
    outputs = _ffi_call(
        _FFI_VALUE_JACOBIAN_TARGET,
        tile_size,
        moment_count,
        boundary_subdivision,
        origins,
        mask,
        (
            cell_size,
            source_x,
            source_y,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        ),
        include_jacobian=True,
    )
    primal_result = _as_fixed_support_result(outputs)
    magnification_jacobian, moments_jacobian = outputs[4:]
    parameter_tangent = jnp.stack(
        (
            source_x_tangent,
            source_y_tangent,
            separation_tangent,
            mass_ratio_tangent,
            source_radius_tangent,
            limb_c_tangent,
            limb_d_tangent,
        )
    )
    float0_boundary = jnp.zeros_like(
        primal_result.boundary_cells, dtype=jax.dtypes.float0
    )
    float0_active = jnp.zeros_like(primal_result.active_cells, dtype=jax.dtypes.float0)
    tangent_result = FixedSupportResult(
        magnification=jnp.vdot(magnification_jacobian, parameter_tangent),
        moments=moments_jacobian @ parameter_tangent,
        boundary_cells=float0_boundary,
        active_cells=float0_active,
    )
    return primal_result, tangent_result


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
    Its custom JVP obtains a seven-parameter analytic Jacobian from C++; JAX
    transposes the resulting linear map for reverse-mode differentiation.
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
    scalars = (
        jax.lax.stop_gradient(jnp.asarray(cell_size, dtype=jnp.float64)),
    ) + tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
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
    return _fixed_support_ffi_transformable(
        tile_size,
        moment_count,
        boundary_subdivision,
        origins,
        mask,
        *scalars,
    )

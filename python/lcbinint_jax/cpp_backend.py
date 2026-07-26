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
from .types import FixedSupportResult, InverseRayResult

_FFI_FORWARD_TARGET = "lcbinint_jax_fixed_support_forward_f64_v1"
_FFI_VALUE_JACOBIAN_TARGET = "lcbinint_jax_fixed_support_value_jacobian_f64_v1"
_FFI_DISCOVERY_TARGET = "lcbinint_jax_macro_tile_discovery_f64_v1"
_FFI_BINARY_ROOT_TARGET = "lcbinint_jax_binary_image_roots_f64_v1"
_FFI_BINARY_ROOT_JACOBIAN_TARGET = "lcbinint_jax_binary_image_roots_jacobian_f64_v1"
_FFI_CARTESIAN_EPOCH_TARGET = "lcbinint_jax_cartesian_epoch_f64_v1"
_FFI_CARTESIAN_EPOCH_JACOBIAN_TARGET = "lcbinint_jax_cartesian_epoch_jacobian_f64_v1"
_FFI_CARTESIAN_BATCH_TARGET = "lcbinint_jax_cartesian_batch_f64_v1"
_FFI_CARTESIAN_BATCH_JACOBIAN_TARGET = "lcbinint_jax_cartesian_batch_jacobian_f64_v1"
_FFI_HEX_BATCH_TARGET = "lcbinint_jax_hexadecapole_batch_f64_v1"
_FFI_HEX_BATCH_JACOBIAN_TARGET = "lcbinint_jax_hexadecapole_batch_jacobian_f64_v1"
_FFI_POLAR_EPOCH_TARGET = "lcbinint_jax_polar_epoch_f64_v1"
_FFI_POLAR_EPOCH_JACOBIAN_TARGET = "lcbinint_jax_polar_epoch_jacobian_f64_v1"
_FFI_TRAJECTORY_TARGET = "lcbinint_jax_trajectory_f64_v1"
_FFI_TRAJECTORY_JACOBIAN_TARGET = "lcbinint_jax_trajectory_jacobian_f64_v1"
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


class _FfiCartesianEpochResult(NamedTuple):
    magnification: jax.Array
    moments: jax.Array
    boundary_cells: jax.Array
    active_cells: jax.Array
    tile_count: jax.Array
    overflow: jax.Array
    root_failure: jax.Array


class _FfiHexadecapoleResult(NamedTuple):
    magnification: jax.Array
    point_magnification: jax.Array
    quadrupole_correction: jax.Array
    hexadecapole_correction: jax.Array
    topology_stable: jax.Array
    root_failure: jax.Array


class FfiTrajectoryResult(NamedTuple):
    magnification: jax.Array
    method: jax.Array
    estimated_error: jax.Array
    support_valid: jax.Array
    used_multipole: jax.Array
    used_polar: jax.Array
    needs_fallback: jax.Array


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


def cpp_cartesian_epoch_ffi_available():
    """Return whether the fused Cartesian epoch and Jacobian FFI are available."""

    if jax.default_backend() != "cpu":
        return False
    try:
        native = _native_module()
        jax_ir = native._jax_ir
        return hasattr(jax_ir, "cartesian_epoch_forward_ffi") and hasattr(
            jax_ir, "cartesian_epoch_value_jacobian_ffi"
        )
    except (AttributeError, RuntimeError):
        return False


def cpp_cartesian_batch_ffi_available():
    """Return whether masked Cartesian batch FFI handlers are available."""

    if jax.default_backend() != "cpu":
        return False
    try:
        native = _native_module()
        jax_ir = native._jax_ir
        return hasattr(jax_ir, "cartesian_batch_forward_ffi") and hasattr(
            jax_ir, "cartesian_batch_value_jacobian_ffi"
        )
    except (AttributeError, RuntimeError):
        return False


def cpp_hexadecapole_batch_ffi_available():
    """Return whether the batched hexadecapole FFI is available."""

    if jax.default_backend() != "cpu":
        return False
    try:
        native = _native_module()
        return hasattr(native._jax_ir, "hexadecapole_batch_ffi") and hasattr(
            native._jax_ir, "hexadecapole_batch_jacobian_ffi"
        )
    except (AttributeError, RuntimeError):
        return False


def cpp_polar_epoch_ffi_available():
    """Return whether the fused polar epoch FFI is available."""

    if jax.default_backend() != "cpu":
        return False
    try:
        native = _native_module()
        return hasattr(native._jax_ir, "polar_epoch_forward_ffi") and hasattr(
            native._jax_ir, "polar_epoch_jacobian_ffi"
        )
    except (AttributeError, RuntimeError):
        return False


def cpp_trajectory_ffi_available():
    """Return whether the integrated trajectory FFI is available."""

    if jax.default_backend() != "cpu":
        return False
    try:
        native = _native_module()
        return hasattr(native._jax_ir, "trajectory_forward_ffi") and hasattr(
            native._jax_ir, "trajectory_jacobian_ffi"
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


@lru_cache(maxsize=1)
def _register_cartesian_epoch_ffi():
    native = _native_module()
    try:
        forward_capsule = native._jax_ir.cartesian_epoch_forward_ffi()
        jacobian_capsule = native._jax_ir.cartesian_epoch_value_jacobian_ffi()
    except AttributeError as error:
        raise RuntimeError(
            "lcbinint was built without the fused Cartesian epoch FFI; rebuild "
            "with LCBININT_ENABLE_JAX_FFI=ON"
        ) from error
    jax.ffi.register_ffi_target(
        _FFI_CARTESIAN_EPOCH_TARGET,
        forward_capsule,
        platform="cpu",
    )
    jax.ffi.register_ffi_target(
        _FFI_CARTESIAN_EPOCH_JACOBIAN_TARGET,
        jacobian_capsule,
        platform="cpu",
    )


@lru_cache(maxsize=1)
def _register_cartesian_batch_ffi():
    native = _native_module()
    try:
        forward_capsule = native._jax_ir.cartesian_batch_forward_ffi()
        jacobian_capsule = native._jax_ir.cartesian_batch_value_jacobian_ffi()
    except AttributeError as error:
        raise RuntimeError(
            "lcbinint was built without the Cartesian batch FFI; rebuild with "
            "LCBININT_ENABLE_JAX_FFI=ON"
        ) from error
    jax.ffi.register_ffi_target(
        _FFI_CARTESIAN_BATCH_TARGET,
        forward_capsule,
        platform="cpu",
    )
    jax.ffi.register_ffi_target(
        _FFI_CARTESIAN_BATCH_JACOBIAN_TARGET,
        jacobian_capsule,
        platform="cpu",
    )


@lru_cache(maxsize=1)
def _register_hexadecapole_batch_ffi():
    native = _native_module()
    try:
        forward = native._jax_ir.hexadecapole_batch_ffi()
        jacobian = native._jax_ir.hexadecapole_batch_jacobian_ffi()
    except AttributeError as error:
        raise RuntimeError(
            "lcbinint was built without the batched hexadecapole FFI"
        ) from error
    jax.ffi.register_ffi_target(
        _FFI_HEX_BATCH_TARGET,
        forward,
        platform="cpu",
    )
    jax.ffi.register_ffi_target(
        _FFI_HEX_BATCH_JACOBIAN_TARGET,
        jacobian,
        platform="cpu",
    )


@lru_cache(maxsize=1)
def _register_polar_epoch_ffi():
    native = _native_module()
    try:
        forward = native._jax_ir.polar_epoch_forward_ffi()
        jacobian = native._jax_ir.polar_epoch_jacobian_ffi()
    except AttributeError as error:
        raise RuntimeError("lcbinint was built without the polar epoch FFI") from error
    jax.ffi.register_ffi_target(
        _FFI_POLAR_EPOCH_TARGET,
        forward,
        platform="cpu",
    )
    jax.ffi.register_ffi_target(
        _FFI_POLAR_EPOCH_JACOBIAN_TARGET,
        jacobian,
        platform="cpu",
    )


@lru_cache(maxsize=1)
def _register_trajectory_ffi():
    native = _native_module()
    try:
        forward = native._jax_ir.trajectory_forward_ffi()
        jacobian = native._jax_ir.trajectory_jacobian_ffi()
    except AttributeError as error:
        raise RuntimeError(
            "lcbinint was built without the integrated trajectory FFI"
        ) from error
    jax.ffi.register_ffi_target(_FFI_TRAJECTORY_TARGET, forward, platform="cpu")
    jax.ffi.register_ffi_target(
        _FFI_TRAJECTORY_JACOBIAN_TARGET,
        jacobian,
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


def _cartesian_epoch_output_specifications(moment_count, include_jacobian):
    outputs = (
        jax.ShapeDtypeStruct((), jnp.float64),
        jax.ShapeDtypeStruct((moment_count,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.bool_),
        jax.ShapeDtypeStruct((), jnp.bool_),
    )
    if include_jacobian:
        return outputs + (
            jax.ShapeDtypeStruct((7,), jnp.float64),
            jax.ShapeDtypeStruct((moment_count, 7), jnp.float64),
        )
    return outputs


def _cartesian_epoch_ffi_call(
    target,
    tile_size,
    tile_capacity,
    limb_samples,
    moment_count,
    boundary_subdivision,
    scalars,
    *,
    include_jacobian,
):
    return jax.ffi.ffi_call(
        target,
        _cartesian_epoch_output_specifications(moment_count, include_jacobian),
        vmap_method="sequential",
    )(
        *scalars,
        tile_size=np.int64(tile_size),
        tile_capacity=np.int64(tile_capacity),
        limb_samples=np.int64(limb_samples),
        moment_mode=np.int64(moment_count),
        boundary_subdivision=np.int64(boundary_subdivision),
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4))
def _cartesian_epoch_ffi_transformable(
    tile_size,
    tile_capacity,
    limb_samples,
    moment_count,
    boundary_subdivision,
    cell_size,
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
):
    outputs = _cartesian_epoch_ffi_call(
        _FFI_CARTESIAN_EPOCH_TARGET,
        tile_size,
        tile_capacity,
        limb_samples,
        moment_count,
        boundary_subdivision,
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
    return _FfiCartesianEpochResult(*outputs)


@_cartesian_epoch_ffi_transformable.defjvp
def _cartesian_epoch_ffi_jvp(
    tile_size,
    tile_capacity,
    limb_samples,
    moment_count,
    boundary_subdivision,
    primals,
    tangents,
):
    (
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
        source_x_tangent,
        source_y_tangent,
        separation_tangent,
        mass_ratio_tangent,
        source_radius_tangent,
        limb_c_tangent,
        limb_d_tangent,
    ) = tangents
    outputs = _cartesian_epoch_ffi_call(
        _FFI_CARTESIAN_EPOCH_JACOBIAN_TARGET,
        tile_size,
        tile_capacity,
        limb_samples,
        moment_count,
        boundary_subdivision,
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
    primal_result = _FfiCartesianEpochResult(*outputs[:7])
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
    tangent_result = _FfiCartesianEpochResult(
        magnification=jnp.vdot(outputs[7], parameter_tangent),
        moments=outputs[8] @ parameter_tangent,
        boundary_cells=jnp.zeros_like(
            primal_result.boundary_cells, dtype=jax.dtypes.float0
        ),
        active_cells=jnp.zeros_like(
            primal_result.active_cells, dtype=jax.dtypes.float0
        ),
        tile_count=jnp.zeros_like(primal_result.tile_count, dtype=jax.dtypes.float0),
        overflow=jnp.zeros_like(primal_result.overflow, dtype=jax.dtypes.float0),
        root_failure=jnp.zeros_like(
            primal_result.root_failure, dtype=jax.dtypes.float0
        ),
    )
    return primal_result, tangent_result


def binary_inverse_ray_cartesian_ffi(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    cell_size,
    tile_size=16,
    tile_capacity=1024,
    limb_samples=16,
    moment_mode="two_coefficient",
    boundary_subdivision=4,
):
    """Fuse binary roots, Cartesian discovery, and integration in one CPU FFI."""

    require_x64()
    if jax.default_backend() != "cpu":
        raise RuntimeError("the fused Cartesian epoch FFI is CPU-only")
    if moment_mode not in _MOMENT_COUNTS:
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )
    if tile_size <= 0 or tile_capacity <= 0 or limb_samples <= 0:
        raise ValueError("tile_size, tile_capacity, and limb_samples must be positive")
    if boundary_subdivision not in (1, 2, 3, 4):
        raise ValueError("boundary_subdivision must be 1, 2, 3, or 4")
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
    _register_cartesian_epoch_ffi()
    result = _cartesian_epoch_ffi_transformable(
        tile_size,
        tile_capacity,
        limb_samples,
        _MOMENT_COUNTS[moment_mode],
        boundary_subdivision,
        *scalars,
    )
    support_valid = ~(result.overflow | result.root_failure)
    return InverseRayResult(
        magnification=result.magnification,
        moments=result.moments,
        boundary_cells=result.boundary_cells,
        active_cells=result.active_cells,
        tile_count=result.tile_count,
        discovery_overflow=result.overflow,
        root_failure=result.root_failure,
        support_valid=support_valid,
    )


def _polar_epoch_call(
    target,
    configuration,
    scalars,
    moment_count,
    include_jacobian,
):
    outputs = (
        jax.ShapeDtypeStruct((), jnp.float64),
        jax.ShapeDtypeStruct((moment_count,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.bool_),
        jax.ShapeDtypeStruct((), jnp.bool_),
    )
    if include_jacobian:
        outputs += (
            jax.ShapeDtypeStruct((7,), jnp.float64),
            jax.ShapeDtypeStruct((moment_count, 7), jnp.float64),
        )
    (
        resolution,
        angular_bins,
        radial_capacity,
        band_capacity,
        limb_samples,
        padding_factor,
        angular_padding_factor,
        angular_chunk_size,
        boundary_capacity,
        boundary_subdivision,
    ) = configuration
    return jax.ffi.ffi_call(target, outputs, vmap_method="sequential")(
        *scalars,
        resolution=np.int64(resolution),
        angular_bins=np.int64(angular_bins),
        radial_capacity=np.int64(radial_capacity),
        band_capacity=np.int64(band_capacity),
        limb_samples=np.int64(limb_samples),
        padding_factor=np.float64(padding_factor),
        angular_padding_factor=np.float64(angular_padding_factor),
        angular_chunk_size=np.int64(angular_chunk_size),
        boundary_capacity=np.int64(boundary_capacity),
        moment_mode=np.int64(moment_count),
        boundary_subdivision=np.int64(boundary_subdivision),
    )


@partial(jax.custom_jvp, nondiff_argnums=tuple(range(11)))
def _polar_epoch_transformable(
    resolution,
    angular_bins,
    radial_capacity,
    band_capacity,
    limb_samples,
    padding_factor,
    angular_padding_factor,
    angular_chunk_size,
    boundary_capacity,
    boundary_subdivision,
    moment_count,
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
):
    outputs = _polar_epoch_call(
        _FFI_POLAR_EPOCH_TARGET,
        (
            resolution,
            angular_bins,
            radial_capacity,
            band_capacity,
            limb_samples,
            padding_factor,
            angular_padding_factor,
            angular_chunk_size,
            boundary_capacity,
            boundary_subdivision,
        ),
        (source_x, source_y, separation, mass_ratio, source_radius, limb_c, limb_d),
        moment_count,
        False,
    )
    return _FfiCartesianEpochResult(*outputs)


@_polar_epoch_transformable.defjvp
def _polar_epoch_jvp(*arguments):
    *configuration, primals, tangents = arguments
    *kernel_configuration, moment_count = configuration
    outputs = _polar_epoch_call(
        _FFI_POLAR_EPOCH_JACOBIAN_TARGET,
        tuple(kernel_configuration),
        tuple(primals),
        moment_count,
        True,
    )
    primal = _FfiCartesianEpochResult(*outputs[:7])
    parameter_tangent = jnp.stack(tangents)
    tangent = _FfiCartesianEpochResult(
        magnification=jnp.vdot(outputs[7], parameter_tangent),
        moments=outputs[8] @ parameter_tangent,
        boundary_cells=jnp.zeros_like(primal.boundary_cells, dtype=jax.dtypes.float0),
        active_cells=jnp.zeros_like(primal.active_cells, dtype=jax.dtypes.float0),
        tile_count=jnp.zeros_like(primal.tile_count, dtype=jax.dtypes.float0),
        overflow=jnp.zeros_like(primal.overflow, dtype=jax.dtypes.float0),
        root_failure=jnp.zeros_like(primal.root_failure, dtype=jax.dtypes.float0),
    )
    return primal, tangent


def binary_inverse_ray_polar_ffi(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    resolution=64,
    angular_bins=2048,
    radial_capacity=512,
    band_capacity=4,
    limb_samples=16,
    padding_factor=0.25,
    angular_padding_factor=4.0,
    angular_chunk_size=256,
    boundary_capacity=2048,
    boundary_subdivision=2,
    moment_mode="two_coefficient",
):
    """Evaluate one differentiable polar inverse-ray epoch in C++."""

    require_x64()
    if moment_mode not in _MOMENT_COUNTS:
        raise ValueError("invalid moment_mode")
    # The FFI currently uses a fixed three-moment ABI. Zero coefficients make
    # the extra moments inert for uniform/linear callers.
    scalars = tuple(
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
        raise ValueError("polar epoch parameters must be scalars")
    _register_polar_epoch_ffi()
    result = _polar_epoch_transformable(
        resolution,
        angular_bins,
        radial_capacity,
        band_capacity,
        limb_samples,
        padding_factor,
        angular_padding_factor,
        angular_chunk_size,
        boundary_capacity,
        boundary_subdivision,
        _MOMENT_COUNTS[moment_mode],
        *scalars,
    )
    return InverseRayResult(
        magnification=result.magnification,
        moments=result.moments,
        boundary_cells=result.boundary_cells,
        active_cells=result.active_cells,
        tile_count=result.tile_count,
        discovery_overflow=result.overflow,
        root_failure=result.root_failure,
        support_valid=~(result.overflow | result.root_failure),
    )


def _hexadecapole_batch_specs(batch_size, include_jacobian):
    outputs = (
        jax.ShapeDtypeStruct((batch_size,), jnp.float64),
        jax.ShapeDtypeStruct((batch_size,), jnp.float64),
        jax.ShapeDtypeStruct((batch_size,), jnp.float64),
        jax.ShapeDtypeStruct((batch_size,), jnp.float64),
        jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
        jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
    )
    if include_jacobian:
        return outputs + (jax.ShapeDtypeStruct((batch_size, 4, 7), jnp.float64),)
    return outputs


def _hexadecapole_batch_call(target, source_x, source_y, scalars, include_jacobian):
    return jax.ffi.ffi_call(
        target,
        _hexadecapole_batch_specs(source_x.shape[0], include_jacobian),
        vmap_method="sequential",
    )(source_x, source_y, *scalars)


@jax.custom_jvp
def _hexadecapole_batch_transformable(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
):
    outputs = _hexadecapole_batch_call(
        _FFI_HEX_BATCH_TARGET,
        source_x,
        source_y,
        (separation, mass_ratio, source_radius, limb_c, limb_d),
        False,
    )
    return _FfiHexadecapoleResult(*outputs)


@_hexadecapole_batch_transformable.defjvp
def _hexadecapole_batch_jvp(primals, tangents):
    source_x, source_y, separation, mass_ratio, source_radius, limb_c, limb_d = primals
    (
        source_x_tangent,
        source_y_tangent,
        separation_tangent,
        mass_ratio_tangent,
        source_radius_tangent,
        limb_c_tangent,
        limb_d_tangent,
    ) = tangents
    outputs = _hexadecapole_batch_call(
        _FFI_HEX_BATCH_JACOBIAN_TARGET,
        source_x,
        source_y,
        (separation, mass_ratio, source_radius, limb_c, limb_d),
        True,
    )
    primal = _FfiHexadecapoleResult(*outputs[:6])
    parameter_tangent = jnp.stack(
        (
            source_x_tangent,
            source_y_tangent,
            jnp.full_like(source_x, separation_tangent),
            jnp.full_like(source_x, mass_ratio_tangent),
            jnp.full_like(source_x, source_radius_tangent),
            jnp.full_like(source_x, limb_c_tangent),
            jnp.full_like(source_x, limb_d_tangent),
        ),
        axis=1,
    )
    numeric_tangent = jnp.einsum("noq,nq->no", outputs[6], parameter_tangent)
    tangent = _FfiHexadecapoleResult(
        magnification=numeric_tangent[:, 0],
        point_magnification=numeric_tangent[:, 1],
        quadrupole_correction=numeric_tangent[:, 2],
        hexadecapole_correction=numeric_tangent[:, 3],
        topology_stable=jnp.zeros_like(primal.topology_stable, dtype=jax.dtypes.float0),
        root_failure=jnp.zeros_like(primal.root_failure, dtype=jax.dtypes.float0),
    )
    return primal, tangent


def binary_hexadecapole_batch_ffi(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
):
    """Evaluate differentiable 13-point expansions in one CPU FFI call."""

    require_x64()
    if jax.default_backend() != "cpu":
        raise RuntimeError("the batched hexadecapole FFI is CPU-only")
    source_x = jnp.asarray(source_x, dtype=jnp.float64)
    source_y = jnp.asarray(source_y, dtype=jnp.float64)
    if source_x.ndim != 1 or source_y.shape != source_x.shape:
        raise ValueError("source_x and source_y must have the same 1-D shape")
    scalars = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (separation, mass_ratio, source_radius, limb_c, limb_d)
    )
    if any(value.ndim != 0 for value in scalars):
        raise ValueError("lens and source parameters must be scalars")
    _register_hexadecapole_batch_ffi()
    result = _hexadecapole_batch_transformable(source_x, source_y, *scalars)
    from .multipole import HexadecapoleResult

    return HexadecapoleResult(
        magnification=result.magnification,
        point_magnification=result.point_magnification,
        quadrupole_correction=result.quadrupole_correction,
        hexadecapole_correction=result.hexadecapole_correction,
        estimated_error=jnp.abs(result.hexadecapole_correction),
        topology_stable=result.topology_stable,
        root_failure=result.root_failure,
    )


def _trajectory_specs(batch_size, include_jacobian):
    outputs = (
        jax.ShapeDtypeStruct((batch_size,), jnp.float64),
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        jax.ShapeDtypeStruct((batch_size,), jnp.float64),
        jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
        jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
        jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
        jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
    )
    if include_jacobian:
        outputs += (jax.ShapeDtypeStruct((batch_size, 7), jnp.float64),)
    return outputs


def _trajectory_call(target, configuration, arguments, include_jacobian):
    (
        cartesian_resolution,
        tile_size,
        tile_capacity,
        cartesian_limb_samples,
        polar_resolution,
        polar_angular_bins,
        polar_radial_capacity,
        polar_band_capacity,
        polar_limb_samples,
        polar_padding_factor,
        polar_angular_padding_factor,
        polar_angular_chunk_size,
        polar_boundary_capacity,
        polar_boundary_subdivision,
        polar_fallback_on_overflow,
        moment_count,
    ) = configuration
    return jax.ffi.ffi_call(
        target,
        _trajectory_specs(arguments[0].shape[0], include_jacobian),
        vmap_method="sequential",
    )(
        *arguments,
        cartesian_resolution=np.int64(cartesian_resolution),
        tile_size=np.int64(tile_size),
        tile_capacity=np.int64(tile_capacity),
        cartesian_limb_samples=np.int64(cartesian_limb_samples),
        polar_resolution=np.int64(polar_resolution),
        polar_angular_bins=np.int64(polar_angular_bins),
        polar_radial_capacity=np.int64(polar_radial_capacity),
        polar_band_capacity=np.int64(polar_band_capacity),
        polar_limb_samples=np.int64(polar_limb_samples),
        polar_padding_factor=np.float64(polar_padding_factor),
        polar_angular_padding_factor=np.float64(polar_angular_padding_factor),
        polar_angular_chunk_size=np.int64(polar_angular_chunk_size),
        polar_boundary_capacity=np.int64(polar_boundary_capacity),
        polar_boundary_subdivision=np.int64(polar_boundary_subdivision),
        polar_fallback_on_overflow=np.int64(polar_fallback_on_overflow),
        moment_mode=np.int64(moment_count),
    )


@partial(jax.custom_jvp, nondiff_argnums=tuple(range(16)))
def _trajectory_transformable(
    cartesian_resolution,
    tile_size,
    tile_capacity,
    cartesian_limb_samples,
    polar_resolution,
    polar_angular_bins,
    polar_radial_capacity,
    polar_band_capacity,
    polar_limb_samples,
    polar_padding_factor,
    polar_angular_padding_factor,
    polar_angular_chunk_size,
    polar_boundary_capacity,
    polar_boundary_subdivision,
    polar_fallback_on_overflow,
    moment_count,
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
    absolute_tolerance,
    relative_tolerance,
    multipole_safety_factor,
    polar_magnification_threshold,
    polar_max_source_radius,
    polar_min_mass_ratio,
):
    return FfiTrajectoryResult(
        *_trajectory_call(
            _FFI_TRAJECTORY_TARGET,
            (
                cartesian_resolution,
                tile_size,
                tile_capacity,
                cartesian_limb_samples,
                polar_resolution,
                polar_angular_bins,
                polar_radial_capacity,
                polar_band_capacity,
                polar_limb_samples,
                polar_padding_factor,
                polar_angular_padding_factor,
                polar_angular_chunk_size,
                polar_boundary_capacity,
                polar_boundary_subdivision,
                polar_fallback_on_overflow,
                moment_count,
            ),
            (
                source_x,
                source_y,
                separation,
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                absolute_tolerance,
                relative_tolerance,
                multipole_safety_factor,
                polar_magnification_threshold,
                polar_max_source_radius,
                polar_min_mass_ratio,
            ),
            False,
        )
    )


@_trajectory_transformable.defjvp
def _trajectory_jvp(*arguments):
    *configuration, primals, tangents = arguments
    outputs = _trajectory_call(
        _FFI_TRAJECTORY_JACOBIAN_TARGET,
        tuple(configuration),
        tuple(primals),
        True,
    )
    primal = FfiTrajectoryResult(*outputs[:7])
    (
        source_x_tangent,
        source_y_tangent,
        separation_tangent,
        *scalar_tangents,
    ) = tangents
    parameter_tangent = jnp.stack(
        (
            source_x_tangent,
            source_y_tangent,
            separation_tangent,
            *(jnp.full_like(primals[0], tangent) for tangent in scalar_tangents[:4]),
        ),
        axis=1,
    )
    tangent = FfiTrajectoryResult(
        magnification=jnp.sum(outputs[7] * parameter_tangent, axis=1),
        method=jnp.zeros_like(primal.method, dtype=jax.dtypes.float0),
        estimated_error=jnp.zeros_like(primal.estimated_error),
        support_valid=jnp.zeros_like(primal.support_valid, dtype=jax.dtypes.float0),
        used_multipole=jnp.zeros_like(primal.used_multipole, dtype=jax.dtypes.float0),
        used_polar=jnp.zeros_like(primal.used_polar, dtype=jax.dtypes.float0),
        needs_fallback=jnp.zeros_like(primal.needs_fallback, dtype=jax.dtypes.float0),
    )
    return primal, tangent


def binary_magnification_trajectory_ffi(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
    *,
    absolute_tolerance,
    relative_tolerance,
    multipole_safety_factor,
    polar_magnification_threshold,
    polar_max_source_radius,
    polar_min_mass_ratio,
    resolution,
    tile_size,
    tile_capacity,
    limb_samples,
    polar_resolution,
    polar_angular_bins,
    polar_radial_capacity,
    polar_band_capacity,
    polar_limb_samples,
    polar_angular_chunk_size,
    polar_fallback_on_overflow,
    moment_mode,
):
    """Run hex/Cartesian/polar dispatch for a trajectory in one CPU FFI."""

    source_x = jnp.asarray(source_x, dtype=jnp.float64)
    source_y = jnp.asarray(source_y, dtype=jnp.float64)
    separation = jnp.asarray(separation, dtype=jnp.float64)
    if separation.ndim == 0:
        separation = jnp.full_like(source_x, separation)
    if separation.shape != source_x.shape:
        raise ValueError("separation must be scalar or match source_x")
    dynamic = (
        source_x,
        source_y,
        separation,
        *(
            jnp.asarray(value, dtype=jnp.float64)
            for value in (
                mass_ratio,
                source_radius,
                limb_c,
                limb_d,
                absolute_tolerance,
                relative_tolerance,
                multipole_safety_factor,
                polar_magnification_threshold,
                polar_max_source_radius,
                polar_min_mass_ratio,
            )
        ),
    )
    _register_trajectory_ffi()
    return _trajectory_transformable(
        resolution,
        tile_size,
        tile_capacity,
        limb_samples,
        polar_resolution,
        polar_angular_bins,
        polar_radial_capacity,
        polar_band_capacity,
        polar_limb_samples,
        0.25,
        4.0,
        polar_angular_chunk_size,
        2048,
        2,
        polar_fallback_on_overflow,
        _MOMENT_COUNTS[moment_mode],
        *dynamic,
    )


def _cartesian_batch_output_specifications(
    batch_size,
    moment_count,
    include_jacobian,
):
    outputs = (
        jax.ShapeDtypeStruct((batch_size,), jnp.float64),
        jax.ShapeDtypeStruct((batch_size, moment_count), jnp.float64),
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
        jax.ShapeDtypeStruct((batch_size,), jnp.bool_),
    )
    if include_jacobian:
        return outputs + (
            jax.ShapeDtypeStruct((batch_size, 7), jnp.float64),
            jax.ShapeDtypeStruct((batch_size, moment_count, 7), jnp.float64),
        )
    return outputs


def _cartesian_batch_ffi_call(
    target,
    tile_size,
    tile_capacity,
    limb_samples,
    moment_count,
    boundary_subdivision,
    source_x,
    source_y,
    active,
    scalars,
    *,
    include_jacobian,
):
    return jax.ffi.ffi_call(
        target,
        _cartesian_batch_output_specifications(
            source_x.shape[0],
            moment_count,
            include_jacobian,
        ),
        vmap_method="sequential",
    )(
        source_x,
        source_y,
        active,
        *scalars,
        tile_size=np.int64(tile_size),
        tile_capacity=np.int64(tile_capacity),
        limb_samples=np.int64(limb_samples),
        moment_mode=np.int64(moment_count),
        boundary_subdivision=np.int64(boundary_subdivision),
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4))
def _cartesian_batch_ffi_transformable(
    tile_size,
    tile_capacity,
    limb_samples,
    moment_count,
    boundary_subdivision,
    source_x,
    source_y,
    active,
    cell_size,
    separation,
    mass_ratio,
    source_radius,
    limb_c,
    limb_d,
):
    outputs = _cartesian_batch_ffi_call(
        _FFI_CARTESIAN_BATCH_TARGET,
        tile_size,
        tile_capacity,
        limb_samples,
        moment_count,
        boundary_subdivision,
        source_x,
        source_y,
        active,
        (
            cell_size,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        ),
        include_jacobian=False,
    )
    return _FfiCartesianEpochResult(*outputs)


@_cartesian_batch_ffi_transformable.defjvp
def _cartesian_batch_ffi_jvp(
    tile_size,
    tile_capacity,
    limb_samples,
    moment_count,
    boundary_subdivision,
    primals,
    tangents,
):
    (
        source_x,
        source_y,
        active,
        cell_size,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
    ) = primals
    (
        source_x_tangent,
        source_y_tangent,
        _,
        _,
        separation_tangent,
        mass_ratio_tangent,
        source_radius_tangent,
        limb_c_tangent,
        limb_d_tangent,
    ) = tangents
    outputs = _cartesian_batch_ffi_call(
        _FFI_CARTESIAN_BATCH_JACOBIAN_TARGET,
        tile_size,
        tile_capacity,
        limb_samples,
        moment_count,
        boundary_subdivision,
        source_x,
        source_y,
        active,
        (
            cell_size,
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        ),
        include_jacobian=True,
    )
    primal_result = _FfiCartesianEpochResult(*outputs[:7])
    parameter_tangent = jnp.stack(
        (
            source_x_tangent,
            source_y_tangent,
            jnp.full_like(source_x, separation_tangent),
            jnp.full_like(source_x, mass_ratio_tangent),
            jnp.full_like(source_x, source_radius_tangent),
            jnp.full_like(source_x, limb_c_tangent),
            jnp.full_like(source_x, limb_d_tangent),
        ),
        axis=1,
    )
    tangent_result = _FfiCartesianEpochResult(
        magnification=jnp.sum(outputs[7] * parameter_tangent, axis=1),
        moments=jnp.einsum("nmq,nq->nm", outputs[8], parameter_tangent),
        boundary_cells=jnp.zeros_like(
            primal_result.boundary_cells, dtype=jax.dtypes.float0
        ),
        active_cells=jnp.zeros_like(
            primal_result.active_cells, dtype=jax.dtypes.float0
        ),
        tile_count=jnp.zeros_like(primal_result.tile_count, dtype=jax.dtypes.float0),
        overflow=jnp.zeros_like(primal_result.overflow, dtype=jax.dtypes.float0),
        root_failure=jnp.zeros_like(
            primal_result.root_failure, dtype=jax.dtypes.float0
        ),
    )
    return primal_result, tangent_result


def binary_inverse_ray_cartesian_batch_ffi(
    source_x,
    source_y,
    separation,
    mass_ratio,
    source_radius,
    limb_c=0.0,
    limb_d=0.0,
    *,
    active=None,
    cell_size,
    tile_size=16,
    tile_capacity=4096,
    limb_samples=24,
    moment_mode="two_coefficient",
    boundary_subdivision=4,
):
    """Evaluate masked independent Cartesian epochs in one differentiable FFI."""

    require_x64()
    if jax.default_backend() != "cpu":
        raise RuntimeError("the Cartesian batch FFI is CPU-only")
    if moment_mode not in _MOMENT_COUNTS:
        raise ValueError(
            "moment_mode must be 'uniform', 'linear', or 'two_coefficient'"
        )
    source_x = jnp.asarray(source_x, dtype=jnp.float64)
    source_y = jnp.asarray(source_y, dtype=jnp.float64)
    if source_x.ndim != 1 or source_y.shape != source_x.shape:
        raise ValueError("source_x and source_y must have the same 1-D shape")
    if active is None:
        active = jnp.ones(source_x.shape, dtype=jnp.bool_)
    active = jax.lax.stop_gradient(jnp.asarray(active, dtype=jnp.bool_))
    if active.shape != source_x.shape:
        raise ValueError("active must have the same shape as source_x")
    scalars = (
        jax.lax.stop_gradient(jnp.asarray(cell_size, dtype=jnp.float64)),
    ) + tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            separation,
            mass_ratio,
            source_radius,
            limb_c,
            limb_d,
        )
    )
    if any(value.ndim != 0 for value in scalars):
        raise ValueError("lens and source parameters must be scalars")
    _register_cartesian_batch_ffi()
    result = _cartesian_batch_ffi_transformable(
        tile_size,
        tile_capacity,
        limb_samples,
        _MOMENT_COUNTS[moment_mode],
        boundary_subdivision,
        source_x,
        source_y,
        active,
        *scalars,
    )
    support_valid = active & ~(result.overflow | result.root_failure)
    return InverseRayResult(
        magnification=result.magnification,
        moments=result.moments,
        boundary_cells=result.boundary_cells,
        active_cells=result.active_cells,
        tile_count=result.tile_count,
        discovery_overflow=result.overflow,
        root_failure=result.root_failure,
        support_valid=support_valid,
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

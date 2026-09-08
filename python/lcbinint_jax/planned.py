"""Fixed-route execution plans for geometry-owned JAX light curves.

The route and resolution decisions in this module are Python constants.  The
source geometry and binary parameters remain JAX arguments, so a caller can
reuse one compiled grouped executable while differentiating through the
geometry supplied by MOASARC.  This is deliberately a small lower-level API;
the public :class:`lcbinint.LightCurve` owns model-level convenience methods.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cpp_backend import (
    binary_hexadecapole_batch_ffi,
    binary_inverse_ray_cartesian_batch_ffi,
    binary_inverse_ray_polar_ffi,
)
from .multipole import binary_point_source_magnification
from .trajectory import _tile_capacity


POINT_SOURCE = 0
HEXADECAPOLE = 1
CARTESIAN = 2
POLAR = 3

METHOD_NAMES = (
    "point_source",
    "hexadecapole",
    "inverse_ray_cartesian",
    "inverse_ray_polar",
)

_METHOD_CODES = {name: code for code, name in enumerate(METHOD_NAMES)}


class PlannedResult(NamedTuple):
    """Magnification and a stopped-gradient support certificate."""

    magnification: jax.Array
    support_valid: jax.Array


def _normalise_plan(methods, resolutions):
    methods = tuple(
        _METHOD_CODES.get(str(method), method)
        if not isinstance(method, int)
        else int(method)
        for method in methods
    )
    resolutions = tuple(int(value) for value in resolutions)
    if not methods or len(methods) != len(resolutions):
        raise ValueError(
            "planned routes need equal, non-empty method/resolution tuples"
        )
    unsupported = tuple(
        method for method in methods
        if method not in (POINT_SOURCE, HEXADECAPOLE, CARTESIAN, POLAR)
    )
    if unsupported:
        raise ValueError(f"unsupported planned route code(s): {unsupported!r}")
    for method, resolution in zip(methods, resolutions):
        if method in (CARTESIAN, POLAR) and resolution < 2:
            raise ValueError("inverse-ray planned resolutions must be at least two")
    return methods, resolutions


def _polar_boundary_capacity(resolution: int) -> int:
    return max(4096, 32 * int(resolution) * int(resolution))


def _set_group(array, indices, values):
    return array.at[jnp.asarray(indices, dtype=jnp.int32)].set(values)


def _point_batch(source_x, source_y, separation, mass_ratio):
    result = jax.vmap(
        lambda x, y: binary_point_source_magnification(
            x, y, separation, mass_ratio
        )
    )(source_x, source_y)
    return result.magnification, ~result.root_failure


def _make_result(methods, resolutions, moment_mode):
    methods, resolutions = _normalise_plan(methods, resolutions)
    if moment_mode not in ("uniform", "linear", "two_coefficient"):
        raise ValueError(f"invalid planned moment mode: {moment_mode!r}")

    def result(
        source_x,
        source_y,
        separation,
        mass_ratio,
        source_radius,
        limb_c,
        limb_d,
    ):
        source_x = jnp.asarray(source_x, dtype=jnp.float64)
        source_y = jnp.broadcast_to(
            jnp.asarray(source_y, dtype=jnp.float64), source_x.shape
        )
        magnification = jnp.full_like(source_x, jnp.nan)
        support_valid = jnp.zeros(source_x.shape, dtype=jnp.bool_)

        # Every group is static.  The Python loops are unrolled while tracing,
        # leaving grouped native calls and indexed scatters in the executable.
        for method in (POINT_SOURCE, HEXADECAPOLE, POLAR, CARTESIAN):
            for resolution in sorted(
                {
                    resolutions[index]
                    for index, planned_method in enumerate(methods)
                    if planned_method == method
                }
            ):
                indices = tuple(
                    index
                    for index, planned_method in enumerate(methods)
                    if planned_method == method
                    and resolutions[index] == resolution
                )
                if not indices:
                    continue
                index_array = jnp.asarray(indices, dtype=jnp.int32)
                group_x = source_x[index_array]
                group_y = source_y[index_array]

                if method == POINT_SOURCE:
                    group_value, group_support = _point_batch(
                        group_x, group_y, separation, mass_ratio
                    )
                elif method == HEXADECAPOLE:
                    expansion = binary_hexadecapole_batch_ffi(
                        group_x,
                        group_y,
                        separation,
                        mass_ratio,
                        source_radius,
                        limb_c,
                        limb_d,
                    )
                    group_value = expansion.magnification
                    group_support = (
                        expansion.topology_stable & ~expansion.root_failure
                    )
                elif method == CARTESIAN:
                    grid = binary_inverse_ray_cartesian_batch_ffi(
                        group_x,
                        group_y,
                        separation,
                        mass_ratio,
                        source_radius,
                        limb_c,
                        limb_d,
                        active=jnp.ones(group_x.shape, dtype=jnp.bool_),
                        cell_size=source_radius / int(resolution),
                        tile_size=16,
                        tile_capacity=_tile_capacity(int(resolution)),
                        limb_samples=32,
                        moment_mode=moment_mode,
                        boundary_subdivision=4,
                    )
                    group_value = grid.magnification
                    group_support = grid.support_valid
                else:
                    radial_target = max(256, 8 * int(resolution))
                    radial_capacity = 1 << (radial_target - 1).bit_length()

                    def evaluate(active_x, active_y):
                        return binary_inverse_ray_polar_ffi(
                            active_x,
                            active_y,
                            separation,
                            mass_ratio,
                            source_radius,
                            limb_c,
                            limb_d,
                            resolution=int(resolution),
                            angular_bins=65536,
                            radial_capacity=radial_capacity,
                            limb_samples=64,
                            angular_chunk_size=256,
                            boundary_capacity=_polar_boundary_capacity(resolution),
                            boundary_subdivision=4,
                            moment_mode=moment_mode,
                        )

                    grid = jax.vmap(evaluate)(group_x, group_y)
                    group_value = grid.magnification
                    group_support = grid.support_valid

                magnification = _set_group(magnification, indices, group_value)
                support_valid = _set_group(
                    support_valid, indices, group_support
                )

        return PlannedResult(magnification, support_valid)

    return result


def make_result_function(methods, resolutions, moment_mode="two_coefficient"):
    """Compile and return one fixed grouped execution plan."""

    return jax.jit(_make_result(methods, resolutions, moment_mode))


__all__ = [
    "CARTESIAN",
    "HEXADECAPOLE",
    "METHOD_NAMES",
    "POINT_SOURCE",
    "POLAR",
    "PlannedResult",
    "make_result_function",
]

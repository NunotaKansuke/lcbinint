"""Fixed-route, fixed-resolution execution plans for JAX light curves.

The ordinary JAX light-curve path deliberately keeps routing and resolution
selection inside the compiled trajectory dispatcher.  That is the right
choice for a one-off evaluation, but it is not the same execution contract as
the public warm-up path: warm-up captures the automatic JAX route and ``nbin``
for each epoch, certifies that fixed choice against the native numerical
reference, and then reuses the decisions.

This module is the small, static counterpart of that dispatcher.  The route
and resolution tuples are Python constants captured when the callable is
constructed, while all physical parameters remain JAX arguments.  Equal
route/resolution rows are grouped into one FFI call, so a warmed light curve
does not pay one scalar kernel launch per epoch and does not carry an
epoch-by-epoch Python loop in the steady-state path.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cpp_backend import (
    binary_hexadecapole_batch_ffi,
    binary_inverse_ray_cartesian_batch_ffi,
    binary_inverse_ray_polar_ffi,
    binary_point_source_batch_ffi,
)
from .trajectory import _tile_capacity


POINT_SOURCE = 0
HEXADECAPOLE = 1
CARTESIAN = 2
POLAR = 3

_METHOD_CODES = {
    "point_source": POINT_SOURCE,
    "hexadecapole": HEXADECAPOLE,
    "inverse_ray_cartesian": CARTESIAN,
    "inverse_ray_polar": POLAR,
}


class PlannedResult(NamedTuple):
    """Value and support mask produced by one fixed execution plan."""

    magnification: jax.Array
    support_valid: jax.Array


def _polar_boundary_capacity(resolution: int) -> int:
    """Provide enough boundary slots for the fixed polar support certificate."""

    # The capacity is only an overflow ceiling.  The quadratic floor keeps the
    # FFI fail-closed for caustic-adjacent rows without allocating that many
    # cells up front.
    return max(4096, 32 * int(resolution) * int(resolution))


def _normalise_plan(methods, resolutions):
    methods = tuple(
        _METHOD_CODES.get(str(method), method)
        if not isinstance(method, int)
        else int(method)
        for method in methods
    )
    resolutions = tuple(int(value) for value in resolutions)
    if len(methods) != len(resolutions) or not methods:
        raise ValueError("JAX warm-up plans need equal, non-empty method/resolution tuples")
    unsupported = tuple(
        method for method in methods
        if method not in (POINT_SOURCE, HEXADECAPOLE, CARTESIAN, POLAR)
    )
    if unsupported:
        raise NotImplementedError(
            "JAX fixed warm-up plans do not support native methods "
            f"{unsupported!r}"
        )
    for method, resolution in zip(methods, resolutions):
        if method in (CARTESIAN, POLAR) and resolution < 2:
            raise ValueError("inverse-ray warm-up resolutions must be at least two")
    return methods, resolutions


def _set_group(array, indices, values):
    return array.at[jnp.asarray(indices, dtype=jnp.int32)].set(values)


def _make_result(methods, resolutions, moment_mode):
    """Create the un-jitted fixed-plan result function."""

    methods, resolutions = _normalise_plan(methods, resolutions)
    if moment_mode not in ("uniform", "linear", "two_coefficient"):
        raise ValueError(f"invalid JAX moment mode: {moment_mode!r}")

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
        active = jnp.ones(source_x.shape, dtype=jnp.bool_)

        # Grouping is static.  The Python loops disappear during tracing and
        # leave only the grouped FFI calls plus indexed scatters in the XLA
        # executable.
        for method in (POINT_SOURCE, HEXADECAPOLE, POLAR, CARTESIAN):
            for resolution in sorted({
                resolutions[index]
                for index, planned_method in enumerate(methods)
                if planned_method == method
            }):
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
                    point = binary_point_source_batch_ffi(
                        group_x,
                        group_y,
                        separation,
                        mass_ratio,
                    )
                    group_value = point.magnification
                    group_support = ~point.root_failure
                elif method == HEXADECAPOLE:
                    expansion = binary_hexadecapole_batch_ffi(
                        group_x,
                        group_y,
                        separation,
                        mass_ratio,
                        source_radius,
                        limb_c,
                        limb_d,
                        active=active[index_array],
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
                        active=active[index_array],
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

                magnification = _set_group(
                    magnification, indices, group_value
                )
                support_valid = _set_group(
                    support_valid, indices, group_support
                )

        return PlannedResult(magnification, support_valid)

    return result


def make_result_function(methods, resolutions, moment_mode):
    """Return a compiled result callable for one static route/bin plan."""

    return jax.jit(_make_result(methods, resolutions, moment_mode))

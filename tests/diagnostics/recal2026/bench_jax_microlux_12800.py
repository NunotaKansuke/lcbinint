#!/usr/bin/env python3
"""Compare batched compiled ``lcbinint_jax`` and microLUX on 12,800 epochs.

The input is the controlled pure-kernel corpus produced by
``bench_grid_vs_vbm_pure_kernel.py``.  Each result row contains four reference
epochs, so the 3,200 rows in ``report_v2/results.json`` represent 12,800
source positions.

This benchmark follows the native/VBM comparison's warm-up contract:

* the saved native ``chosen_grid``/``chosen_nbin`` plan is used as the JAX
  per-epoch route/bin warm-up plan; rows without a native plan use the JAX
  calibrated route selector;
* JAX evaluates all four reference epochs through grouped direct polar or
  Cartesian FFI calls, and the warmed result is checked against the saved
  reference;
* only the selected, already-compiled callable is timed, with forward and
  source-trajectory ``dA/dt`` measured as separate batched calls;
* microLUX keeps its public default ``n_annuli=10`` behaviour.  The same
  default callable is used for the timing and for the reported comparison;
  there is no per-row annulus ladder.

The output stores both the full four-epoch block wall time and its per-epoch
equivalent.  The latter is only a normalization for comparison; the block
time is the primary measurement because compiled JAX work is not generally
linear in the number of epochs.

Example smoke run::

    OMP_NUM_THREADS=1 python tests/diagnostics/recal2026/\\
        bench_jax_microlux_12800.py \\
        --profiles uniform --max-jobs 1 --repeats 2 \\
        --output /tmp/jax-microlux-smoke.json

Full corpus run::

    OMP_NUM_THREADS=1 python tests/diagnostics/recal2026/\\
        bench_jax_microlux_12800.py \\
        --output tests/diagnostics/results/recal2026/\\
        jax_microlux_12800/results.json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load_repo_build_backend():
    """Load the extension built in this checkout before importing JAX code.

    The development environment currently has an older ``lcbinint`` shared
    object in site-packages.  Importing the source package normally combines
    with that installed extension, which makes the Python wrappers and C++ FFI
    disagree about their ABI.  A benchmark must fail closed here instead of
    silently timing that mixed installation.
    """

    configured = os.environ.get("LCBININT_BENCH_BUILD_ROOT")
    build_root = Path(configured).expanduser() if configured else ROOT / "build"
    build_root = build_root.resolve()
    package_dir = build_root / "lcbinint"
    extensions = sorted(package_dir.glob("_lcbinint*.so"))
    if not extensions:
        raise RuntimeError(
            "current lcbinint build extension was not found under "
            f"{package_dir}; build the checkout or set "
            "LCBININT_BENCH_BUILD_ROOT"
        )
    extension_path = extensions[-1].resolve()

    existing = sys.modules.get("lcbinint")
    if existing is not None:
        loaded = getattr(getattr(existing, "_native", None), "__file__", None)
        if loaded and Path(loaded).resolve() == extension_path:
            return build_root, extension_path
        raise RuntimeError(
            "lcbinint was imported before the benchmark backend could be "
            f"validated (loaded={loaded!r}, expected={str(extension_path)!r})"
        )
    if "lcbinint_jax" in sys.modules:
        raise RuntimeError(
            "lcbinint_jax was imported before the benchmark backend could be "
            "validated"
        )

    package_init = package_dir / "__init__.py"
    package_spec = importlib.util.spec_from_file_location(
        "lcbinint",
        package_init,
        submodule_search_locations=[str(package_dir)],
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError(f"could not load {package_init}")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["lcbinint"] = package

    extension_spec = importlib.util.spec_from_file_location(
        "lcbinint._lcbinint", extension_path
    )
    if extension_spec is None or extension_spec.loader is None:
        raise RuntimeError(f"could not load {extension_path}")
    extension = importlib.util.module_from_spec(extension_spec)
    sys.modules["lcbinint._lcbinint"] = extension
    extension_spec.loader.exec_module(extension)
    package._lcbinint = extension
    package_spec.loader.exec_module(package)

    jax_dir = build_root / "lcbinint_jax"
    jax_init = jax_dir / "__init__.py"
    jax_spec = importlib.util.spec_from_file_location(
        "lcbinint_jax",
        jax_init,
        submodule_search_locations=[str(jax_dir)],
    )
    if jax_spec is None or jax_spec.loader is None:
        raise RuntimeError(f"could not load {jax_init}")
    jax_package = importlib.util.module_from_spec(jax_spec)
    sys.modules["lcbinint_jax"] = jax_package
    jax_spec.loader.exec_module(jax_package)

    loaded = Path(package._native.__file__).resolve()
    if loaded != extension_path:
        raise RuntimeError(
            "loaded lcbinint extension does not match the checkout build: "
            f"loaded={loaded}, expected={extension_path}"
        )
    return build_root, extension_path


RUNTIME_BUILD_ROOT, RUNTIME_EXTENSION = _load_repo_build_backend()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from lcbinint_jax.cpp_backend import (  # noqa: E402
    binary_inverse_ray_cartesian_batch_ffi,
    binary_inverse_ray_polar_directional_ffi,
    binary_inverse_ray_polar_ffi,
    binary_point_source_batch_ffi,
    binary_routing_diagnostics_batch_ffi,
    cpp_binary_routing_diagnostics_batch_ffi_available,
    cpp_cartesian_batch_ffi_available,
    cpp_hexadecapole_batch_ffi_available,
    cpp_point_source_batch_ffi_available,
    cpp_polar_epoch_ffi_available,
    cpp_trajectory_ffi_available,
)
from lcbinint_jax.trajectory import (  # noqa: E402
    _tile_capacity,
)
from lcbinint_jax.resolution import select_binary_resolution  # noqa: E402
from lcbinint_jax.types import InverseRayResult  # noqa: E402


JAX_FFI_CAPABILITIES = {
    "point_batch": bool(cpp_point_source_batch_ffi_available()),
    "hexadecapole_batch": bool(cpp_hexadecapole_batch_ffi_available()),
    "routing_diagnostics_batch": bool(
        cpp_binary_routing_diagnostics_batch_ffi_available()
    ),
    "polar_epoch": bool(cpp_polar_epoch_ffi_available()),
    "cartesian_batch": bool(cpp_cartesian_batch_ffi_available()),
    "trajectory": bool(cpp_trajectory_ffi_available()),
}
if not all(JAX_FFI_CAPABILITIES.values()):
    raise RuntimeError(
        "the current build is missing a required native-routed JAX FFI: "
        f"extension={RUNTIME_EXTENSION}, capabilities={JAX_FFI_CAPABILITIES}"
    )


DEFAULT_INPUT = (
    ROOT
    / "tests/diagnostics/results/recal2026/"
    / "pure_kernel_balanced_loguniform_20260812/report_v2/results.json"
)
REFERENCE_INDICES = (0, 7, 15, 23)
BLOCK_EPOCHS = 24
BLOCK_SPAN_IN_RADII = 0.4
LIMB_C = 0.5
DEFAULT_STRATEGY = (30, 30, 60, 120, 240)
TIGHT_STRATEGY = (60, 60, 120, 240, 480)
MICROLUX_DEFAULT_N_ANNULI = 10
# Kept as a one-element tuple so old output/CLI plumbing remains compatible.
MICROLUX_ANNULI_LADDER = (MICROLUX_DEFAULT_N_ANNULI,)
REFERENCE_UNCERTAINTY_FRACTION = 0.1
# The polar kernel is refined by a global angular ladder.  Starting from a
# fixed power-of-two grid makes the warm-up policy independent of q, source
# size, and case ID; the adjacent-grid certificate decides whether refinement
# is actually needed.  A global ceiling keeps a non-convergent angular error
# from turning into an unbounded warm-up.
DEFAULT_POLAR_ANGULAR_BINS = 65_536
MAX_POLAR_ANGULAR_BINS = 2_097_152
# Boundary cells are the dominant deterministic quadrature error in the
# Cartesian FFI.  Use the inexpensive 4x4 rule as the normal path.  A
# native-certified row whose 4x4 result still misses the target is rechecked
# with the same compiled kernel at 8x8; this is a quadrature-order convergence
# step, not a lens- or case-specific fallback.
CARTESIAN_BOUNDARY_SUBDIVISION = 4
CARTESIAN_REFINED_BOUNDARY_SUBDIVISION = 8


def _finite(value):
    return value is not None and np.isfinite(float(value))


def _times(row):
    """Reconstruct the four reference epochs used by the corpus."""

    rho = float(row["rho"])
    full = np.linspace(
        float(row["x"]) - 0.5 * BLOCK_SPAN_IN_RADII * rho,
        float(row["x"]) + 0.5 * BLOCK_SPAN_IN_RADII * rho,
        BLOCK_EPOCHS,
    )
    return full[list(REFERENCE_INDICES)]


def _relative_error(value, reference):
    if not _finite(value) or not _finite(reference):
        return float("nan")
    return float(
        abs(float(value) - float(reference))
        / max(abs(float(reference)), 1.0)
    )


def _polar_boundary_capacity(resolution):
    """Keep the JAX polar support certificate from truncating caustic tails."""

    # The FFI default (2048) is a reasonable small smoke-test value, but the
    # recalibration corpus can have tens of thousands of boundary cells even
    # at modest source-bin counts.  Overflow makes the result uncertified and
    # must never be silently treated as a valid timing sample.
    return max(4096, 32 * int(resolution) * int(resolution))


def _block(result):
    """Synchronize a JAX array or pytree before stopping the timer."""

    return jax.block_until_ready(result)


class _DeadlineExceeded(TimeoutError):
    pass


@contextmanager
def _deadline(seconds):
    """Interrupt one synchronous CPU JAX call after a bounded wait.

    The benchmark runs in a single main thread per lane, so SIGALRM gives us
    a recoverable row-level boundary without putting the timed executable in a
    different process (which would include compilation in every sample).  A
    non-positive value disables the guard.
    """

    seconds = None if seconds is None else float(seconds)
    if seconds is None or seconds <= 0.0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def handler(_signum, _frame):
        raise _DeadlineExceeded(
            f"synchronous call exceeded {seconds:g} seconds"
        )

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(
                signal.ITIMER_REAL, previous_timer[0], previous_timer[1]
            )


def _checkout(module):
    repository = Path(module.__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _arguments(row, times):
    """Return dynamic JAX arguments for a four-epoch source block.

    The corpus stores VBM coordinates.  The low-level lcbinint JAX kernel uses
    the reciprocal mass-ratio convention, while the source trajectory itself
    is unchanged for this ``t0=0, tE=1, alpha=0`` corpus.
    """

    return (
        jnp.asarray(times, dtype=jnp.float64),
        jnp.asarray(float(row["y"]), dtype=jnp.float64),
        jnp.asarray(float(row["s"]), dtype=jnp.float64),
        jnp.asarray(float(row["q"]), dtype=jnp.float64),
        jnp.asarray(float(row["rho"]), dtype=jnp.float64),
    )


def _native_plan(row, fallback=None):
    """Return the maximum Nbin from the saved native warm-up plan.

    ``fallback`` is retained for old callers, but the benchmark no longer
    uses it to hide a missing plan.  A missing plan is a real diagnostic state
    and is returned as ``None`` when no fallback was explicitly requested.
    """

    values = row.get("chosen_nbin")
    if isinstance(values, (list, tuple)):
        finite = [int(value) for value in values if _finite(value)]
        if finite:
            return max(2, max(finite)), "saved_native_chosen_nbin"
    if fallback is None:
        return None, "missing_saved_native_plan"
    return max(2, int(fallback)), "fallback_without_saved_native_plan"


def _native_plan_certified(row):
    """Whether the saved native warm-up actually met this row's target."""

    values = row.get("chosen_vbm_errors")
    if not isinstance(values, (list, tuple)) or not values:
        return False
    finite = [float(value) for value in values if _finite(value)]
    return bool(finite) and all(
        value <= float(row["target"]) for value in finite
    )


def _microlux_strategy(target):
    """Give tighter retol values enough contour budget to be meaningful."""

    return TIGHT_STRATEGY if float(target) < 1.0e-3 else DEFAULT_STRATEGY


class JaxBatchExecutor:
    """Native-plan JAX callables for one four-epoch block.

    The saved native route/bin plan is the initial candidate.  Accuracy
    selection then uses one generic radial/azimuthal convergence ladder; the
    final callable contains only the selected static route and grid.
    """

    def __init__(
        self, max_source_bins=400,
        polar_angular_bins=DEFAULT_POLAR_ANGULAR_BINS,
        polar_angular_ratio=64.0,
        boundary_subdivision=CARTESIAN_BOUNDARY_SUBDIVISION,
    ):
        self.max_source_bins = int(max_source_bins)
        if self.max_source_bins < 2:
            raise ValueError("max_source_bins must be at least 2")
        self.polar_angular_bins = int(polar_angular_bins)
        if self.polar_angular_bins < 0:
            raise ValueError("polar_angular_bins must be non-negative")
        self._polar_angular_base_bins = self.polar_angular_bins
        self.polar_angular_ratio = float(polar_angular_ratio)
        if not np.isfinite(self.polar_angular_ratio) or self.polar_angular_ratio <= 0.0:
            raise ValueError("polar_angular_ratio must be finite and positive")
        if boundary_subdivision not in (
            CARTESIAN_BOUNDARY_SUBDIVISION,
            CARTESIAN_REFINED_BOUNDARY_SUBDIVISION,
        ):
            raise ValueError(
                "boundary_subdivision must be the normal 4x4 rule or "
                "the refined 8x8 rule"
            )
        self.boundary_subdivision = int(boundary_subdivision)
        self._results = {}
        self._values = {}
        self._derivatives = {}
        self.compile_records = {}

    @staticmethod
    def _key(profile, profile_c, target, route_plan, resolution_plan,
             polar_angular_bins, polar_angular_ratio, boundary_subdivision):
        return (
            str(profile),
            float(profile_c),
            float(target),
            tuple(str(value) for value in route_plan),
            tuple(int(value) for value in resolution_plan),
            int(polar_angular_bins),
            float(polar_angular_ratio),
            int(boundary_subdivision),
        )

    @staticmethod
    def _raw_result(profile, profile_c, target, route_plan, resolution_plan,
                    polar_angular_bins, polar_angular_ratio,
                    boundary_subdivision):
        moment_mode = "linear" if profile == "linear" else "uniform"
        limb_c = float(profile_c) if profile == "linear" else 0.0
        moment_count = 2 if profile == "linear" else 1
        route_plan = tuple(str(value) for value in route_plan)
        resolution_plan = tuple(int(value) for value in resolution_plan)

        if len(route_plan) != len(resolution_plan):
            raise ValueError("route and resolution plans must have equal length")
        if any(value not in {"polar", "cartesian"} for value in route_plan):
            raise ValueError(f"unsupported route plan: {route_plan}")

        def scatter(array, indices, values):
            return array.at[jnp.asarray(indices, dtype=jnp.int32)].set(values)

        def result(times, u0, separation, mass_ratio, source_radius):
            source_y = jnp.full_like(times, u0)
            magnification = jnp.full(times.shape, jnp.nan, dtype=times.dtype)
            moments = jnp.full(
                times.shape + (moment_count,), jnp.nan, dtype=times.dtype
            )
            boundary_cells = jnp.zeros(times.shape, dtype=jnp.int32)
            active_cells = jnp.zeros(times.shape, dtype=jnp.int32)
            tile_count = jnp.zeros(times.shape, dtype=jnp.int32)
            discovery_overflow = jnp.zeros(times.shape, dtype=jnp.bool_)
            root_failure = jnp.zeros(times.shape, dtype=jnp.bool_)
            support_valid = jnp.zeros(times.shape, dtype=jnp.bool_)

            # The native warm-up plan is per epoch.  Group equal route/bin
            # pairs so each group still uses one batched FFI call; mixed
            # polar/Cartesian blocks never enter the pathological all-branch
            # trajectory dispatcher.
            for route in ("polar", "cartesian"):
                for resolution in sorted(
                    {
                        resolution_plan[index]
                        for index, item in enumerate(route_plan)
                        if item == route
                    }
                ):
                    indices = tuple(
                        index
                        for index, item in enumerate(route_plan)
                        if item == route and resolution_plan[index] == resolution
                    )
                    if not indices:
                        continue
                    index_array = jnp.asarray(indices, dtype=jnp.int32)
                    group_x = times[index_array]
                    group_y = source_y[index_array]
                    if route == "polar":
                        radial_target = max(256, 8 * int(resolution))
                        radial_capacity = 1 << (radial_target - 1).bit_length()

                        def evaluate(source_x, source_y):
                            return binary_inverse_ray_polar_ffi(
                                source_x,
                                source_y,
                                separation,
                                1.0 / mass_ratio,
                                source_radius,
                                limb_c,
                                0.0,
                                resolution=int(resolution),
                                angular_bins=int(polar_angular_bins),
                                angular_padding_factor=float(
                                    polar_angular_ratio
                                ),
                                radial_capacity=radial_capacity,
                                limb_samples=64,
                                angular_chunk_size=256,
                                boundary_capacity=_polar_boundary_capacity(
                                    resolution
                                ),
                                boundary_subdivision=4,
                                moment_mode=moment_mode,
                            )

                        group = jax.vmap(evaluate)(group_x, group_y)
                    else:
                        group = binary_inverse_ray_cartesian_batch_ffi(
                            group_x,
                            group_y,
                            separation,
                            1.0 / mass_ratio,
                            source_radius,
                            limb_c,
                            0.0,
                            cell_size=source_radius / int(resolution),
                            tile_size=16,
                            tile_capacity=int(_tile_capacity(resolution)),
                            limb_samples=32,
                            moment_mode=moment_mode,
                            boundary_subdivision=boundary_subdivision,
                        )
                    magnification = scatter(
                        magnification, indices, group.magnification
                    )
                    moments = scatter(moments, indices, group.moments)
                    boundary_cells = scatter(
                        boundary_cells, indices, group.boundary_cells
                    )
                    active_cells = scatter(
                        active_cells, indices, group.active_cells
                    )
                    tile_count = scatter(tile_count, indices, group.tile_count)
                    discovery_overflow = scatter(
                        discovery_overflow,
                        indices,
                        group.discovery_overflow,
                    )
                    root_failure = scatter(
                        root_failure, indices, group.root_failure
                    )
                    support_valid = scatter(
                        support_valid, indices, group.support_valid
                    )
            return InverseRayResult(
                magnification=magnification,
                moments=moments,
                boundary_cells=boundary_cells,
                active_cells=active_cells,
                tile_count=tile_count,
                discovery_overflow=discovery_overflow,
                root_failure=root_failure,
                support_valid=support_valid,
            )

        return result

    @classmethod
    def _raw_value(cls, profile, profile_c, target, route_plan,
                   resolution_plan, polar_angular_bins, polar_angular_ratio,
                   boundary_subdivision):
        result = cls._raw_result(
            profile, profile_c, target, route_plan, resolution_plan,
            polar_angular_bins, polar_angular_ratio, boundary_subdivision,
        )

        def value(times, u0, separation, mass_ratio, source_radius):
            return result(
                times, u0, separation, mass_ratio, source_radius
            ).magnification

        return value

    @classmethod
    def _raw_derivative(cls, profile, profile_c, target, route_plan,
                        resolution_plan, polar_angular_bins, polar_angular_ratio,
                        boundary_subdivision):
        moment_mode = "linear" if profile == "linear" else "uniform"
        limb_c = float(profile_c) if profile == "linear" else 0.0
        route_plan = tuple(str(value) for value in route_plan)
        resolution_plan = tuple(int(value) for value in resolution_plan)

        def derivative(times, u0, separation, mass_ratio, source_radius):
            source_y = jnp.full_like(times, u0)
            tangent_values = jnp.zeros_like(times)
            for route in ("polar", "cartesian"):
                for resolution in sorted(
                    {
                        resolution_plan[index]
                        for index, item in enumerate(route_plan)
                        if item == route
                    }
                ):
                    indices = tuple(
                        index
                        for index, item in enumerate(route_plan)
                        if item == route
                        and resolution_plan[index] == resolution
                    )
                    if not indices:
                        continue
                    index_array = jnp.asarray(indices, dtype=jnp.int32)
                    group_x = times[index_array]
                    group_y = source_y[index_array]
                    if route == "polar":
                        radial_target = max(256, 8 * int(resolution))
                        radial_capacity = 1 << (radial_target - 1).bit_length()

                        def evaluate(active_x, active_y):
                            _, tangent = binary_inverse_ray_polar_directional_ffi(
                                active_x,
                                active_y,
                                separation,
                                1.0 / mass_ratio,
                                source_radius,
                                limb_c,
                                0.0,
                                source_x_tangent=1.0,
                                resolution=int(resolution),
                                angular_bins=int(polar_angular_bins),
                                angular_padding_factor=float(
                                    polar_angular_ratio
                                ),
                                radial_capacity=radial_capacity,
                                limb_samples=64,
                                angular_chunk_size=256,
                                boundary_capacity=_polar_boundary_capacity(
                                    resolution
                                ),
                                boundary_subdivision=4,
                                moment_mode=moment_mode,
                            )
                            return tangent.magnification

                        group = jax.vmap(evaluate)(group_x, group_y)
                    else:
                        def evaluate(active_times):
                            result = binary_inverse_ray_cartesian_batch_ffi(
                                active_times,
                                group_y,
                                separation,
                                1.0 / mass_ratio,
                                source_radius,
                                limb_c,
                                0.0,
                                cell_size=source_radius / int(resolution),
                                tile_size=16,
                                tile_capacity=int(_tile_capacity(resolution)),
                                limb_samples=32,
                                moment_mode=moment_mode,
                                boundary_subdivision=boundary_subdivision,
                            )
                            return result.magnification

                        _, group = jax.jvp(
                            evaluate,
                            (group_x,),
                            (jnp.ones_like(group_x),),
                        )
                    tangent_values = tangent_values.at[index_array].set(group)
            return tangent_values

        return derivative

    def _compile(self, key, kind, function, arguments):
        record_key = (
            f"{kind}:{key[0]}:{key[1]:g}:target={key[2]:g}:"
            f"routes={','.join(key[3])}:bins={','.join(map(str, key[4]))}:"
            f"angular={key[5]}:ratio={key[6]:g}:"
            f"boundary={key[7]}"
        )
        if record_key not in self.compile_records:
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _block(function(*arguments))
            self.compile_records[record_key] = {
                "kind": kind,
                "profile": key[0],
                "limb_darkening_c": key[1],
                "target": key[2],
                "route_plan": list(key[3]),
                "resolution_plan": list(key[4]),
                "polar_angular_bins": key[5],
                "polar_angular_ratio": key[6],
                "warnings": [str(item.message) for item in caught],
                "compile_and_first_seconds": time.perf_counter() - started,
            }
        return function

    def _function(self, collection, kind, profile, profile_c, target,
                  route_plan, resolution_plan, arguments, factory):
        key = self._key(
            profile, profile_c, target, route_plan, resolution_plan,
            self.polar_angular_bins, self.polar_angular_ratio,
            self.boundary_subdivision,
        )
        function = collection.get(key)
        if function is None:
            function = jax.jit(
                factory(
                    profile,
                    profile_c,
                    target,
                    route_plan,
                    resolution_plan,
                    self.polar_angular_bins,
                    self.polar_angular_ratio,
                    self.boundary_subdivision,
                )
            )
            collection[key] = function
        return self._compile(key, kind, function, arguments)

    def _result_function(self, profile, profile_c, target, route_plan,
                         resolution_plan, arguments):
        return self._function(
            self._results,
            "jax_convergence",
            profile,
            profile_c,
            target,
            route_plan,
            resolution_plan,
            arguments,
            self._raw_result,
        )

    def _value_function(self, profile, profile_c, target, route_plan,
                        resolution_plan, arguments):
        return self._function(
            self._values,
            "jax_forward",
            profile,
            profile_c,
            target,
            route_plan,
            resolution_plan,
            arguments,
            self._raw_value,
        )

    def _derivative_function(self, profile, profile_c, target, route_plan,
                             resolution_plan, arguments):
        return self._function(
            self._derivatives,
            "jax_dA_dt",
            profile,
            profile_c,
            target,
            route_plan,
            resolution_plan,
            arguments,
            self._raw_derivative,
        )

    @staticmethod
    def _saved_route_plan(row, epoch_count):
        grids = row.get("chosen_grid")
        bins = row.get("chosen_nbin")
        if not isinstance(grids, (list, tuple)) or not isinstance(
            bins, (list, tuple)
        ):
            return None
        if len(grids) != epoch_count or len(bins) != epoch_count:
            return None
        route_plan = tuple(str(value).lower() for value in grids)
        if any(value not in {"polar", "cartesian"} for value in route_plan):
            return None
        if not all(_finite(value) for value in bins):
            return None
        resolution_plan = tuple(max(2, int(value)) for value in bins)
        return route_plan, resolution_plan

    def _jax_route_plan(self, row, times, profile_c, target):
        """Build a static per-epoch route for rows missing native warm-up."""

        arguments = _arguments(row, times)
        source_x, source_y, separation, mass_ratio, source_radius = arguments
        source_y = jnp.full_like(source_x, source_y)
        mass_ratio = 1.0 / mass_ratio
        point = binary_point_source_batch_ffi(
            source_x, source_y, separation, mass_ratio
        )
        routing = binary_routing_diagnostics_batch_ffi(
            source_x,
            source_y,
            point.magnification,
            separation,
            mass_ratio,
            source_radius,
            absolute_tolerance=0.0,
            relative_tolerance=float(target),
            caustic_bins=1400,
        )
        selections = jax.vmap(
            lambda distance, point_value: select_binary_resolution(
                mass_ratio,
                source_radius,
                distance,
                point_value,
                float(profile_c),
                float(target),
                self.max_source_bins,
            )
        )(routing.caustic_distance, point.magnification)
        prefer_polar, source_bins, point_safe = _block(
            (
                selections.prefer_polar,
                selections.source_bins,
                routing.point_safe,
            )
        )
        prefer_polar = np.asarray(prefer_polar, dtype=bool)
        source_bins = np.asarray(source_bins, dtype=int)
        point_safe = np.asarray(point_safe, dtype=bool)
        route_plan = tuple(
            "polar" if bool(value) else "cartesian" for value in prefer_polar
        )
        # Point-safe rows are normally handled by the native point branch.  A
        # missing native plan means that branch did not produce a complete
        # warm-up record, so retain a finite direct route for measurement.
        route_plan = tuple(
            "polar" if safe else route
            for route, safe in zip(route_plan, point_safe)
        )
        resolution_plan = tuple(
            max(2, min(int(value), self.max_source_bins))
            for value in source_bins
        )
        # A missing native plan has no measured Cartesian bucket to reuse.
        # When the generic routing diagnostic reaches the global Cartesian
        # capacity, that is a bounded signal that the Cartesian support is
        # the expensive branch, not a reason to compile a 400x400 tiled
        # executable blindly.  Use the same polar kernel at that radial
        # resolution; this is a route-capacity policy shared by all such
        # rows, with no q-, d/rho-, or case-specific branch.
        route_plan = tuple(
            "polar"
            if route == "cartesian" and int(resolution) >= self.max_source_bins
            else route
            for route, resolution in zip(route_plan, resolution_plan)
        )
        return route_plan, resolution_plan

    def _probe_plan(
        self,
        profile,
        profile_c,
        candidate_routes,
        candidate_resolutions,
        row,
        times,
        evaluation_indices,
        angular_ratio,
    ):
        """Evaluate only the requested epochs without compiling a batch.

        The production callable remains grouped and JIT-compiled.  This
        scalar probe is deliberately confined to warm-up: it prevents a
        pathological four-epoch vmap compilation from obscuring the actual
        FFI convergence cost while still using exactly the same kernels.
        """

        indices = tuple(int(index) for index in evaluation_indices)
        moment_mode = "linear" if profile == "linear" else "uniform"
        limb_c = float(profile_c) if profile == "linear" else 0.0
        values = np.full(len(indices), np.nan, dtype=float)
        support = np.zeros(len(indices), dtype=bool)
        overflow = np.zeros(len(indices), dtype=bool)
        root_failure = np.zeros(len(indices), dtype=bool)
        cartesian_groups = defaultdict(list)
        for local_index, index in enumerate(indices):
            route = str(candidate_routes[index])
            resolution = int(candidate_resolutions[index])
            if route == "polar":
                radial_target = max(256, 8 * resolution)
                radial_capacity = 1 << (radial_target - 1).bit_length()
                result = binary_inverse_ray_polar_ffi(
                    jnp.asarray(float(times[index]), dtype=jnp.float64),
                    jnp.asarray(float(row["y"]), dtype=jnp.float64),
                    jnp.asarray(float(row["s"]), dtype=jnp.float64),
                    jnp.asarray(1.0 / float(row["q"]), dtype=jnp.float64),
                    jnp.asarray(float(row["rho"]), dtype=jnp.float64),
                    limb_c,
                    0.0,
                    resolution=resolution,
                    angular_bins=int(self.polar_angular_bins),
                    angular_padding_factor=float(angular_ratio),
                    radial_capacity=radial_capacity,
                    limb_samples=64,
                    angular_chunk_size=256,
                    boundary_capacity=_polar_boundary_capacity(resolution),
                    boundary_subdivision=4,
                    moment_mode=moment_mode,
                )
                values[local_index] = float(_block(result.magnification))
                support[local_index] = bool(_block(result.support_valid))
                overflow[local_index] = bool(
                    _block(result.discovery_overflow)
                )
                root_failure[local_index] = bool(_block(result.root_failure))
            else:
                cartesian_groups[resolution].append((local_index, index))
        for resolution, group in cartesian_groups.items():
            cartesian_indices = [local_index for local_index, _ in group]
            cartesian_times = jnp.asarray(
                [float(times[index]) for _, index in group],
                dtype=jnp.float64,
            )
            result = binary_inverse_ray_cartesian_batch_ffi(
                cartesian_times,
                jnp.full_like(cartesian_times, float(row["y"])),
                jnp.asarray(float(row["s"]), dtype=jnp.float64),
                jnp.asarray(1.0 / float(row["q"]), dtype=jnp.float64),
                jnp.asarray(float(row["rho"]), dtype=jnp.float64),
                limb_c,
                0.0,
                cell_size=float(row["rho"]) / int(resolution),
                tile_size=16,
                tile_capacity=int(_tile_capacity(resolution)),
                limb_samples=32,
                moment_mode=moment_mode,
                boundary_subdivision=self.boundary_subdivision,
            )
            cart_values = np.asarray(_block(result.magnification), dtype=float)
            cart_support = np.asarray(_block(result.support_valid), dtype=bool)
            cart_overflow = np.asarray(
                _block(result.discovery_overflow), dtype=bool
            )
            cart_root_failure = np.asarray(
                _block(result.root_failure), dtype=bool
            )
            for local_index, value_index in enumerate(cartesian_indices):
                values[value_index] = cart_values[local_index]
                support[value_index] = cart_support[local_index]
                overflow[value_index] = cart_overflow[local_index]
                root_failure[value_index] = cart_root_failure[local_index]
        return values, support, overflow, root_failure

    def calibrate(self, profile, profile_c, route_plan, resolution_plan, row,
                  times, references, target, forward_timeout):
        arguments = _arguments(row, times)
        started = time.perf_counter()
        with _deadline(forward_timeout):
            function = self._result_function(
                profile,
                profile_c,
                target,
                route_plan,
                resolution_plan,
                arguments,
            )
            result = _block(function(*arguments))
        elapsed = time.perf_counter() - started
        values = np.asarray(result.magnification, dtype=float)
        support = np.asarray(result.support_valid, dtype=bool)
        errors = [
            _relative_error(value, reference)
            for value, reference in zip(values, references)
        ]
        checked = [
            index
            for index, reference in enumerate(references)
            if _finite(reference)
        ]
        passes = bool(
            checked
            and all(
                support[index]
                and _finite(values[index])
                and np.isfinite(errors[index])
                and errors[index] <= float(target)
                for index in checked
            )
        )
        return {
            "resolution": int(max(resolution_plan)),
            "support_all": bool(np.all(support)),
            "max_relative_error": (
                None
                if not checked
                else float(np.nanmax(np.asarray(errors)[checked]))
            ),
            "passes": passes,
            "seconds": float(elapsed),
            "values": values,
            "support": support,
            "errors": errors,
        }

    @staticmethod
    def _expanded_resolution_plan(resolution_plan, maximum):
        return tuple(
            min(maximum, max(int(value) + 1, 2 * int(value)))
            for value in resolution_plan
        )

    def select(self, profile, profile_c, row, times, references, target,
               fallback_resolution, forward_timeout, initial_plan=None,
               initial_values=None, initial_support=None):
        """Select the cheapest statically compiled plan with convergence.

        The native warm-up plan is the first candidate.  Polar candidates use
        a global power-of-two angular ladder; a first-rung target pass is
        accepted immediately, while an initially missed target continues
        until an adjacent pair is numerically stable.  A stable miss is
        recorded as unresolved and stops that angular/radial search, while a
        non-monotone sequence is still allowed to recover a later target pass
        before the stable-miss stop.  This keeps a genuine cross-code
        residual from consuming the global ceiling.
        All limits are global numerical-policy limits; no lens-specific
        fallback is used.
        """

        del fallback_resolution
        epoch_count = len(times)
        saved_plan = (
            initial_plan
            if initial_plan is not None
            else self._saved_route_plan(row, epoch_count)
        )
        if saved_plan is not None:
            route_plan, resolution_plan = saved_plan
            native_plan_mode = (
                "source_jax_plan_start"
                if initial_plan is not None
                else "saved_native_grid_nbin"
            )
            route_plan_source = (
                "source_jax_plan_start"
                if initial_plan is not None
                else "saved_native_grid_nbin"
            )
        else:
            route_plan, resolution_plan = self._jax_route_plan(
                row, times, profile_c, target
            )
            native_plan_mode = "missing_saved_native_plan"
            route_plan_source = "jax_calibrated_route_plan"
        route_plan = tuple(route_plan)
        resolution_plan = tuple(int(value) for value in resolution_plan)
        native_plan, _ = _native_plan(row, None)
        # ``polar_angular_bins`` is mutated while each static candidate is
        # probed.  Keep the constructor value as the start of every row's
        # ladder; a previous row must not silently change the next row's
        # numerical policy.
        angular_base_bins = int(self._polar_angular_base_bins)

        arguments = _arguments(row, times)
        warmup_started = time.perf_counter()
        calibration = []
        best = None
        selected = None
        ratio_ladder = (
            256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0
        )
        checked_references = np.asarray(
            [_finite(reference) for reference in references], dtype=bool
        )
        baseline_entry = None
        if initial_values is not None:
            baseline_values = np.asarray(initial_values, dtype=float)
            if baseline_values.shape == (epoch_count,):
                baseline_support = (
                    np.ones(epoch_count, dtype=bool)
                    if initial_support is None
                    else np.asarray(initial_support, dtype=bool)
                )
                if baseline_support.shape == (epoch_count,):
                    baseline_entry = {
                        "values": baseline_values,
                        "support": baseline_support,
                        "discovery_overflow_by_epoch": np.zeros(
                            epoch_count, dtype=bool
                        ),
                        "root_failure_by_epoch": np.zeros(
                            epoch_count, dtype=bool
                        ),
                    }

        def calibrate_plan(
            candidate_routes,
            candidate_resolutions,
            mode,
            ratio,
            angular_bins,
            active_indices=None,
            base_entry=None,
            required_indices=None,
        ):
            calibration_started = time.perf_counter()
            self.polar_angular_bins = int(angular_bins)
            self.polar_angular_ratio = float(ratio)
            if active_indices is None:
                evaluation_indices = tuple(range(epoch_count))
            else:
                evaluation_indices = tuple(int(index) for index in active_indices)
            timed_out = False
            try:
                with _deadline(forward_timeout):
                    (
                        evaluated_values,
                        evaluated_support,
                        evaluated_overflow,
                        evaluated_root_failure,
                    ) = self._probe_plan(
                        profile,
                        profile_c,
                        candidate_routes,
                        candidate_resolutions,
                        row,
                        times,
                        evaluation_indices,
                        ratio,
                    )
            except _DeadlineExceeded:
                timed_out = True
                evaluated_values = np.full(
                    len(evaluation_indices), np.nan, dtype=float
                )
                evaluated_support = np.zeros(
                    len(evaluation_indices), dtype=bool
                )
                evaluated_overflow = np.ones(
                    len(evaluation_indices), dtype=bool
                )
                evaluated_root_failure = np.ones(
                    len(evaluation_indices), dtype=bool
                )
            if base_entry is None:
                values = np.full(epoch_count, np.nan, dtype=float)
                support = np.zeros(epoch_count, dtype=bool)
                discovery_overflow = np.zeros(epoch_count, dtype=bool)
                root_failure = np.zeros(epoch_count, dtype=bool)
            else:
                values = np.asarray(base_entry["values"], dtype=float).copy()
                support = np.asarray(base_entry["support"], dtype=bool).copy()
                discovery_overflow = np.asarray(
                    base_entry["discovery_overflow_by_epoch"], dtype=bool
                ).copy()
                root_failure = np.asarray(
                    base_entry["root_failure_by_epoch"], dtype=bool
                ).copy()
            values[list(evaluation_indices)] = evaluated_values
            support[list(evaluation_indices)] = evaluated_support
            discovery_overflow[list(evaluation_indices)] = np.asarray(
                evaluated_overflow, dtype=bool
            )
            root_failure[list(evaluation_indices)] = np.asarray(
                evaluated_root_failure, dtype=bool
            )
            errors = np.asarray(
                [
                    _relative_error(value, reference)
                    for value, reference in zip(values, references)
                ],
                dtype=float,
            )
            checked = np.asarray(
                [_finite(reference) for reference in references], dtype=bool
            )
            required = np.ones(epoch_count, dtype=bool)
            if required_indices is not None:
                required[:] = False
                required[list(required_indices)] = True
            checked_required = checked & required
            finite_errors = np.where(checked_required, errors, np.nan)
            entry = {
                "route_plan": list(candidate_routes),
                "resolution_plan": list(candidate_resolutions),
                "selection_mode": mode,
                "polar_angular_bins": int(self.polar_angular_bins),
                "polar_angular_ratio": float(ratio),
                # A support certificate is a separate diagnostic from the
                # numerical comparison.  Near a caustic tangent the shared
                # certificate can remain unresolved even when the finite
                # result is finite and agrees with the VBM reference.  Do
                # not turn that conservative proof status into an expensive
                # global refinement loop: retain it in ``support_all`` and
                # ``certified_passes`` below, while selecting on the actual
                # finite/reference error.
                "passes": bool(
                    np.any(checked_required)
                    and np.all(
                        np.isfinite(values[checked_required])
                        & ~discovery_overflow[checked_required]
                        & np.isfinite(errors[checked_required])
                        & (errors[checked_required] <= float(target))
                    )
                ),
                "support_all": bool(np.all(support[required])),
                "discovery_overflow": bool(
                    np.any(discovery_overflow)
                ),
                "root_failure": bool(
                    np.any(root_failure)
                ),
                "max_relative_error": (
                    None
                    if not np.any(checked_required)
                    or not np.any(np.isfinite(finite_errors))
                    else float(np.nanmax(finite_errors))
                ),
                "failed_indices": [
                    int(index)
                    for index in range(epoch_count)
                    if checked_required[index]
                    and (
                        not support[index]
                        or not np.isfinite(errors[index])
                        or errors[index] > float(target)
                    )
                ],
                "seconds": float(time.perf_counter() - calibration_started),
                "values": values,
                "support": support,
                "discovery_overflow_by_epoch": discovery_overflow,
                "root_failure_by_epoch": root_failure,
                "evaluation_indices": list(evaluation_indices),
                "required_indices": list(np.flatnonzero(required)),
                "timed_out": timed_out,
            }
            entry["certified_passes"] = bool(
                entry["passes"]
                and entry["support_all"]
                and not entry["discovery_overflow"]
                and not entry["root_failure"]
            )
            calibration.append(entry)
            return entry

        def angular_candidate(candidate_routes, candidate_resolutions, mode):
            previous = None
            last = None
            first_passing = None
            base_entry = baseline_entry
            first_level = True
            polar_indices = tuple(
                index
                for index, route in enumerate(candidate_routes)
                if route == "polar"
            )
            partial_polar = bool(polar_indices) and len(polar_indices) < epoch_count
            changed_indices = tuple(
                index
                for index in range(epoch_count)
                if candidate_routes[index] != route_plan[index]
                or int(candidate_resolutions[index])
                != int(resolution_plan[index])
            )
            first_indices = tuple(
                sorted(set(polar_indices) | set(changed_indices))
            )
            if any(route == "polar" for route in candidate_routes):
                if angular_base_bins == 0:
                    angular_levels = tuple(
                        (0, ratio) for ratio in ratio_ladder
                    )
                else:
                    angular_levels = []
                    angle = max(16, angular_base_bins)
                    while angle < MAX_POLAR_ANGULAR_BINS:
                        angular_levels.append(
                            (angle, self.polar_angular_ratio)
                        )
                        angle *= 2
                    angular_levels.append(
                        (min(angle, MAX_POLAR_ANGULAR_BINS),
                         self.polar_angular_ratio)
                    )
                    angular_levels = tuple(angular_levels)
            else:
                angular_levels = (
                    (angular_base_bins, self.polar_angular_ratio),
                )
            for level, (angular_bins, ratio) in enumerate(angular_levels):
                entry = calibrate_plan(
                    candidate_routes,
                    candidate_resolutions,
                    f"{mode}_angular_level_{level}",
                    ratio,
                    angular_bins,
                    active_indices=(
                        first_indices
                        if first_level and baseline_entry is not None
                        and first_indices
                        else (
                            polar_indices
                            if (first_level and partial_polar)
                            or (not first_level and partial_polar)
                            else None
                        )
                    ),
                    base_entry=base_entry,
                    required_indices=(polar_indices if partial_polar else None),
                )
                if first_level:
                    base_entry = entry
                    first_level = False
                if previous is None:
                    pair_errors = None
                    pair_stable = not any(
                        route == "polar" for route in candidate_routes
                    )
                else:
                    pair_errors = np.asarray(
                        [
                            _relative_error(value, other)
                            for value, other in zip(
                                entry["values"], previous["values"]
                            )
                        ],
                        dtype=float,
                    )
                    pair_mask = checked_references & np.asarray(
                        [
                            index in (
                                polar_indices
                                if partial_polar else range(epoch_count)
                            )
                            for index in range(epoch_count)
                        ],
                        dtype=bool,
                    )
                    pair_stable = bool(
                        np.any(pair_mask)
                        and np.all(
                            np.isfinite(pair_errors[pair_mask])
                            & (pair_errors[pair_mask] <= float(target))
                        )
                    )
                entry["angular_pair_errors"] = (
                    None if pair_errors is None else pair_errors.tolist()
                )
                entry["angular_pair_stable"] = pair_stable
                last = entry
                if entry["passes"] and first_passing is None:
                    # The first target-passing rung is the cheapest valid
                    # candidate.  Keep probing only to avoid mistaking a
                    # transient pass for the end of a non-monotone ladder;
                    # if no later certificate appears, this rung remains the
                    # production choice.
                    first_passing = entry
                if entry["passes"] and (pair_stable or previous is None):
                    selected_angle_entry = entry
                    selected_ratio = ratio
                    selected_bins = int(entry["polar_angular_bins"])
                    if previous is not None:
                        pair_mask = checked_references & np.asarray(
                            [
                                index in (
                                    polar_indices
                                    if partial_polar
                                    else range(epoch_count)
                                )
                                for index in range(epoch_count)
                            ],
                            dtype=bool,
                        )
                        pair_max = float(
                            np.nanmax(pair_errors[pair_mask])
                        )
                        # The adjacent fine grid is a certificate for the
                        # coarser grid once their difference is comfortably
                        # below the requested budget.  Keeping that coarser
                        # grid is important: the fine pass is warm-up work,
                        # not a hidden multiplier in the timed kernel.
                        if (
                            previous["passes"]
                            and pair_max <= 0.5 * float(target)
                        ):
                            selected_angle_entry = previous
                            selected_ratio = float(
                                previous["polar_angular_ratio"]
                            )
                            selected_bins = int(
                                previous["polar_angular_bins"]
                            )
                    if partial_polar:
                        # ``passes`` above is intentionally scoped to the
                        # polar epochs while the angular ladder is running.
                        # It is not a whole-plan certificate for a mixed
                        # route.  Always evaluate the complete static plan
                        # once at the selected angle before returning; this
                        # also prevents stale ``initial_values`` from hiding
                        # a Cartesian failure.
                        validated = calibrate_plan(
                            candidate_routes,
                            candidate_resolutions,
                            f"{mode}_full_validation",
                            selected_ratio,
                            selected_bins,
                            active_indices=None,
                            base_entry=None,
                            required_indices=None,
                        )
                        validated["angular_pair_errors"] = entry[
                            "angular_pair_errors"
                        ]
                        validated["angular_pair_stable"] = pair_stable
                        validated["polar_angular_ratio"] = selected_ratio
                        validated["polar_angular_bins"] = selected_bins
                        validated["polar_calibration"] = entry
                        return validated
                    selected_angle_entry["angular_pair_stable"] = pair_stable
                    selected_angle_entry["polar_angular_ratio"] = selected_ratio
                    selected_angle_entry["polar_angular_bins"] = selected_bins
                    return selected_angle_entry
                if previous is not None and pair_stable and not entry["passes"]:
                    # A stable adjacent pair that still misses the requested
                    # target is an unresolved numerical residual.  Stop this
                    # ladder rather than paying for every global rung.  A
                    # non-monotone sequence that had already passed remains
                    # represented by ``first_passing`` and is validated below.
                    entry["angular_stalled"] = True
                    previous = entry
                    break
                previous = entry
            if partial_polar and last is not None:
                # If a mixed plan had an earlier polar pass but the final
                # angular rung missed because of non-monotone quadrature,
                # validate the cheapest passing rung as a complete plan.
                validation_entry = (
                    first_passing if first_passing is not None else last
                )
                validated = calibrate_plan(
                    candidate_routes,
                    candidate_resolutions,
                    f"{mode}_full_validation",
                    float(validation_entry["polar_angular_ratio"]),
                    int(validation_entry["polar_angular_bins"]),
                    active_indices=None,
                    base_entry=None,
                    required_indices=None,
                )
                validated["angular_pair_errors"] = validation_entry.get(
                    "angular_pair_errors"
                )
                validated["angular_pair_stable"] = bool(
                    validation_entry.get("angular_pair_stable", False)
                )
                validated["angular_stalled"] = not bool(
                    validated["passes"]
                )
                validated["polar_angular_ratio"] = float(
                    validation_entry["polar_angular_ratio"]
                )
                validated["polar_angular_bins"] = int(
                    validation_entry["polar_angular_bins"]
                )
                validated["polar_calibration"] = validation_entry
                return validated
            if first_passing is not None:
                return first_passing
            return last

        def consider(entry):
            nonlocal best
            if entry is None:
                return
            score = entry["max_relative_error"]
            score = float("inf") if score is None else float(score)
            if best is None or score < best[0]:
                best = (score, entry)

        current_routes = route_plan
        current_resolutions = resolution_plan
        for radial_level in range(5):
            primary = angular_candidate(
                current_routes,
                current_resolutions,
                f"{route_plan_source}_radial_level_{radial_level}",
            )
            consider(primary)
            if (
                primary.get("angular_stalled", False)
                and not primary["passes"]
                and radial_level >= 1
            ):
                selected = primary
                break
            if primary["passes"]:
                # Once the target is met, do not start a new radial search
                # merely because the final angular rung has not produced a
                # second-grid certificate.  The selected rung is already the
                # global angular ceiling; repeating it at five radial levels
                # would only multiply an expensive Cartesian mixed plan.
                selected = primary
                break

            expanded = self._expanded_resolution_plan(
                current_resolutions, self.max_source_bins
            )
            if expanded == current_resolutions:
                break
            current_resolutions = expanded

        if selected is None:
            selected = best[1] if best is not None else primary
        route_plan = tuple(selected["route_plan"])
        resolution_plan = tuple(selected["resolution_plan"])
        self.polar_angular_bins = int(selected["polar_angular_bins"])
        self.polar_angular_ratio = float(selected["polar_angular_ratio"])
        selection_mode = selected["selection_mode"]
        with _deadline(forward_timeout):
            self._value_function(
                profile,
                profile_c,
                target,
                route_plan,
                resolution_plan,
                arguments,
            )
        warmup_seconds = time.perf_counter() - warmup_started
        return {
            "resolution": int(max(resolution_plan)),
            "max_source_bins": int(self.max_source_bins),
            "route_plan": list(route_plan),
            "resolution_plan": list(resolution_plan),
            "route_plan_source": route_plan_source,
            "native_plan_start": (
                None if native_plan is None else int(native_plan)
            ),
            "native_plan_start_mode": native_plan_mode,
            "native_plan": row.get("chosen_nbin"),
            "warmup_mode": selection_mode,
            "warmup_seconds": float(warmup_seconds),
            "polar_angular_bins": int(self.polar_angular_bins),
            "polar_angular_ratio": float(self.polar_angular_ratio),
            "boundary_subdivision": int(self.boundary_subdivision),
            "target_pass": bool(selected["passes"]),
            "support_valid": bool(selected["support_all"]),
            "certified_pass": bool(selected["certified_passes"]),
            "discovery_overflow": bool(selected["discovery_overflow"]),
            "root_failure": bool(selected["root_failure"]),
            "selected_max_relative_error": (
                float(selected["max_relative_error"])
                if selected["max_relative_error"] is not None
                else float("inf")
            ),
            "convergence_certified": bool(
                selected["certified_passes"]
                and selected["angular_pair_stable"]
            ),
            "calibration": calibration,
            "values": None,
        }

    def timed(self, profile, profile_c, target, route_plan, resolution_plan,
              row, times, repeats, forward_timeout, derivative_timeout,
              skip_derivative=False):
        arguments = _arguments(row, times)
        del skip_derivative
        with _deadline(forward_timeout):
            forward = self._value_function(
                profile,
                profile_c,
                target,
                route_plan,
                resolution_plan,
                arguments,
            )
        warning_messages = []

        def run(function, timeout=None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if timeout is None:
                    value = _block(function(*arguments))
                else:
                    with _deadline(timeout):
                        value = _block(function(*arguments))
            warning_messages.extend(str(item.message) for item in caught)
            return value

        # The derivative executable is compiled above and its compile cost is
        # recorded separately.  This first post-warm call is not included in
        # the steady-state samples either.
        first_forward_started = time.perf_counter()
        forward_values = run(forward, forward_timeout)
        first_forward_seconds = time.perf_counter() - first_forward_started
        first_derivative_started = time.perf_counter()
        try:
            with _deadline(derivative_timeout):
                derivative = self._derivative_function(
                    profile,
                    profile_c,
                    target,
                    route_plan,
                    resolution_plan,
                    arguments,
                )
                derivative_values = run(derivative)
            derivative_timeout_message = None
        except _DeadlineExceeded as error:
            derivative_values = None
            derivative_timeout_message = str(error)
        first_derivative_seconds = (
            time.perf_counter() - first_derivative_started
        )

        forward_samples = []
        derivative_samples = [] if derivative_values is not None else None
        for _ in range(max(1, int(repeats))):
            started = time.perf_counter()
            run(forward, forward_timeout)
            forward_samples.append(time.perf_counter() - started)
        if derivative_values is not None:
            for _ in range(max(1, int(repeats))):
                started = time.perf_counter()
                try:
                    run(derivative, derivative_timeout)
                except _DeadlineExceeded as error:
                    derivative_timeout_message = str(error)
                    derivative_samples = None
                    break
                derivative_samples.append(time.perf_counter() - started)

        return {
            "forward_values": np.asarray(forward_values, dtype=float),
            "dA_dt_values": (
                None
                if derivative_values is None
                else np.asarray(derivative_values, dtype=float)
            ),
            "forward_block_seconds": float(np.median(forward_samples)),
            "forward_samples_seconds": forward_samples,
            "dA_dt_block_seconds": (
                None
                if not derivative_samples
                else float(np.median(derivative_samples))
            ),
            "dA_dt_samples_seconds": derivative_samples,
            "forward_first_after_warmup_seconds": float(
                first_forward_seconds
            ),
            "dA_dt_first_after_warmup_seconds": float(
                first_derivative_seconds
            ),
            "dA_dt_timeout": derivative_values is None
            or derivative_samples is None,
            "dA_dt_timeout_message": derivative_timeout_message,
            "warnings": warning_messages,
            "budget_exhausted": any(
                "space to insert" in message or "max length" in message
                for message in warning_messages
            ),
        }

    def prepare_derivative(self, profile, profile_c, target, route_plan,
                           resolution_plan, row, times, timeout):
        """Compile and execute the selected derivative once during warm-up.

        The steady-state derivative timer must never include the first XLA
        compilation.  The previous path compiled this callable in ``timed``;
        that made a large static Cartesian plan look like a derivative
        timeout even though subsequent calls were fast.
        """

        arguments = _arguments(row, times)
        compile_timeout = max(300.0, float(timeout or 0.0))
        started = time.perf_counter()
        with _deadline(compile_timeout):
            derivative = self._derivative_function(
                profile,
                profile_c,
                target,
                route_plan,
                resolution_plan,
                arguments,
            )
            _block(derivative(*arguments))
        return float(time.perf_counter() - started)


class MicroLuxBatchExecutor:
    """microLUX batched callables with an explicit annulus policy.

    The normal benchmark keeps the public microLUX default of ten annuli.
    A separate fixed-annulus run can request a larger value (for example 80)
    without changing the default result or adding a per-row fallback path.
    """

    def __init__(self, max_annuli=MICROLUX_ANNULI_LADDER[-1],
                 fixed_n_annuli=None):
        from microlux.basic_function import to_lowmass
        from microlux.limb_darkening import LinearLimbDarkening
        from microlux.trajectory import LinearTrajectory
        from microlux.trajectory_model import (
            extended_light_curve_from_trajectory_l,
        )

        self._to_lowmass = to_lowmass
        self._LinearLimbDarkening = LinearLimbDarkening
        self._LinearTrajectory = LinearTrajectory
        self._extended_light_curve = extended_light_curve_from_trajectory_l
        self.max_annuli = int(max_annuli)
        self.fixed_n_annuli = (
            None if fixed_n_annuli is None else int(fixed_n_annuli)
        )
        if self.max_annuli < MICROLUX_DEFAULT_N_ANNULI:
            raise ValueError(
                "max_annuli must be at least the microLUX default "
                f"{MICROLUX_DEFAULT_N_ANNULI}"
            )
        if self.fixed_n_annuli is not None:
            if self.fixed_n_annuli < MICROLUX_DEFAULT_N_ANNULI:
                raise ValueError(
                    "fixed_n_annuli must be at least "
                    f"{MICROLUX_DEFAULT_N_ANNULI}"
                )
            if self.fixed_n_annuli > self.max_annuli:
                raise ValueError(
                    "fixed_n_annuli must not exceed max_annuli"
                )
        self._values = {}
        self._derivatives = {}
        self.compile_records = {}

    @staticmethod
    def _annuli_key(n_annuli):
        return -1 if n_annuli is None else int(n_annuli)

    @classmethod
    def _key(cls, profile, profile_c, target, n_annuli):
        return (
            str(profile),
            float(profile_c),
            float(target),
            cls._annuli_key(n_annuli),
        )

    def _candidates(self, profile):
        if profile != "linear":
            return (None,)
        return (self.n_annuli_for_profile(profile),)

    def n_annuli_for_profile(self, profile):
        if profile != "linear":
            return None
        if self.fixed_n_annuli is not None:
            return self.fixed_n_annuli
        return MICROLUX_DEFAULT_N_ANNULI

    def _raw_value(self, profile, profile_c, target, n_annuli):
        target = float(target)
        profile_c = float(profile_c)
        strategy = _microlux_strategy(target)
        limb_darkening = (
            self._LinearLimbDarkening(profile_c)
            if profile == "linear"
            else None
        )
        def value(times, u0, separation, mass_ratio, source_radius):
            # Match the existing wrapper convention: mirror x and reverse the
            # block, then undo the reversal in the returned chronology.
            mirrored_times = -times[::-1]
            trajectory = self._LinearTrajectory(
                0.0, u0, 1.0, 0.0
            )(mirrored_times)
            trajectory_l = self._to_lowmass(
                separation, mass_ratio, trajectory
            )
            kwargs = {
                "tol": target,
                "retol": target,
                "default_strategy": strategy,
                "limb_darkening": limb_darkening,
            }
            if profile == "linear":
                kwargs["n_annuli"] = int(n_annuli)
            values = self._extended_light_curve(
                trajectory_l,
                separation,
                mass_ratio,
                source_radius,
                **kwargs,
            )
            return values[::-1]

        return value

    def _raw_derivative(self, profile, profile_c, target, n_annuli):
        value = self._raw_value(profile, profile_c, target, n_annuli)

        def derivative(times, u0, separation, mass_ratio, source_radius):
            _, tangent = jax.jvp(
                lambda active_times: value(
                    active_times,
                    u0,
                    separation,
                    mass_ratio,
                    source_radius,
                ),
                (times,),
                (jnp.ones_like(times),),
            )
            return tangent

        return derivative

    def _compile(self, key, kind, function, arguments):
        annuli = "default" if key[3] < 0 else str(key[3])
        record_key = (
            f"{kind}:{key[0]}:{key[1]:g}:target={key[2]:g}:annuli={annuli}"
        )
        if record_key not in self.compile_records:
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _block(function(*arguments))
            self.compile_records[record_key] = {
                "kind": kind,
                "profile": key[0],
                "limb_darkening_c": key[1],
                "target": key[2],
                "n_annuli": None if key[3] < 0 else key[3],
                "default_strategy": list(_microlux_strategy(key[2])),
                "warnings": [str(item.message) for item in caught],
                "compile_and_first_seconds": time.perf_counter() - started,
            }
        return function

    def _function(self, collection, kind, profile, profile_c, target,
                  n_annuli, arguments, factory):
        key = self._key(profile, profile_c, target, n_annuli)
        function = collection.get(key)
        if function is None:
            function = jax.jit(
                factory(profile, profile_c, target, n_annuli)
            )
            collection[key] = function
        return self._compile(key, kind, function, arguments)

    def _value_function(self, profile, profile_c, target, n_annuli,
                        arguments):
        return self._function(
            self._values,
            "microlux_forward",
            profile,
            profile_c,
            target,
            n_annuli,
            arguments,
            self._raw_value,
        )

    def _derivative_function(self, profile, profile_c, target, n_annuli,
                             arguments):
        return self._function(
            self._derivatives,
            "microlux_dA_dt",
            profile,
            profile_c,
            target,
            n_annuli,
            arguments,
            self._raw_derivative,
        )

    @staticmethod
    def _budget_exhausted(warning_messages):
        return any(
            "space to insert" in message or "max length" in message
            for message in warning_messages
        )

    def calibrate(self, profile, profile_c, target, n_annuli, row, times,
                  references, reference_certified, forward_timeout):
        arguments = _arguments(row, times)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            started = time.perf_counter()
            with _deadline(forward_timeout):
                function = self._value_function(
                    profile, profile_c, target, n_annuli, arguments
                )
                values = np.asarray(
                    _block(function(*arguments)), dtype=float
                )
            elapsed = time.perf_counter() - started
        errors = [
            _relative_error(value, reference)
            for value, reference in zip(values, references)
        ]
        checked = [
            index
            for index, reference in enumerate(references)
            if _finite(reference)
        ]
        warning_messages = [str(item.message) for item in caught]
        budget_exhausted = self._budget_exhausted(warning_messages)
        passes = bool(
            reference_certified
            and not budget_exhausted
            and checked
            and all(
                _finite(errors[index]) and errors[index] <= float(target)
                for index in checked
            )
        )
        return {
            "n_annuli": None if n_annuli is None else int(n_annuli),
            "values": values,
            "relative_errors": errors,
            "max_relative_error": (
                None
                if not checked
                else float(np.nanmax(np.asarray(errors)[checked]))
            ),
            "passes_reference": passes,
            "warmup_seconds": float(elapsed),
            "warnings": warning_messages,
            "budget_exhausted": budget_exhausted,
        }

    @staticmethod
    def _pair_stable(left, right, target, references):
        differences = [
            _relative_error(value, other)
            for value, other in zip(left["values"], right["values"])
        ]
        checked = [
            index
            for index, reference in enumerate(references)
            if _finite(reference)
        ]
        return (
            bool(
                checked
                and all(
                    _finite(differences[index])
                    and differences[index] <= float(target)
                    for index in checked
                )
            ),
            differences,
        )

    def select(self, profile, profile_c, target, row, times, references,
               reference_uncertainties, forward_timeout,
               fallback_annuli=None, n_annuli_override=None):
        if fallback_annuli is not None and profile == "linear":
            if int(fallback_annuli) < MICROLUX_DEFAULT_N_ANNULI:
                raise ValueError(
                    "fallback annuli must be at least "
                    f"{MICROLUX_DEFAULT_N_ANNULI}"
                )
        reference_certified = bool(
            all(_finite(value) for value in references)
            and reference_uncertainties is not None
            and len(reference_uncertainties) == len(references)
            and all(
                _finite(value)
                and float(value) <= REFERENCE_UNCERTAINTY_FRACTION * float(target)
                for value in reference_uncertainties
            )
        )
        if n_annuli_override is not None and profile != "linear":
            raise ValueError(
                "n_annuli_override is only valid for linear limb darkening"
            )
        if n_annuli_override is not None:
            selected_annuli = int(n_annuli_override)
            if selected_annuli < 1:
                raise ValueError("n_annuli_override must be positive")
            if selected_annuli > self.max_annuli:
                raise ValueError(
                    "n_annuli_override must not exceed max_annuli"
                )
        else:
            selected_annuli = self.n_annuli_for_profile(profile)
        warmup_started = time.perf_counter()
        arguments = _arguments(row, times)
        with _deadline(forward_timeout):
            # As with JAX, _compile performs the first call only when this
            # static callable is new.  Reusing it avoids a redundant warm-up
            # evaluation for every subsequent row.
            self._value_function(
                profile,
                profile_c,
                target,
                selected_annuli,
                arguments,
            )
        warmup_seconds = float(time.perf_counter() - warmup_started)
        if profile != "linear":
            status = "not_applicable_uniform"
            selection_mode = "not_applicable_uniform"
        elif n_annuli_override is not None:
            status = "event_requested_n_annuli"
            selection_mode = "event_requested_n_annuli"
        elif self.fixed_n_annuli is not None:
            status = "fixed_requested_n_annuli"
            selection_mode = "fixed_requested_n_annuli"
        else:
            status = "fixed_default_n_annuli"
            selection_mode = "fixed_default_n_annuli"
        entry = {
            "n_annuli": selected_annuli,
            "values": None,
            "relative_errors": [None] * len(references),
            "max_relative_error": None,
            "passes_reference": None,
            "warmup_seconds": warmup_seconds,
            "warnings": [],
            "budget_exhausted": False,
            "selection_mode": selection_mode,
        }
        return {
            "status": status,
            "reference_certified": reference_certified,
            "selected_n_annuli": selected_annuli,
            "default_entry": entry,
            "selected_entry": entry,
            "selected_values": None,
            "calibration": [entry],
            "warmup_seconds": warmup_seconds,
        }

    def timed(self, profile, profile_c, target, n_annuli, row, times, repeats,
              forward_timeout, derivative_timeout, skip_derivative=False):
        arguments = _arguments(row, times)
        with _deadline(forward_timeout):
            forward = self._value_function(
                profile, profile_c, target, n_annuli, arguments
            )
        warning_messages = []

        def run(function, timeout=None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if timeout is None:
                    value = _block(function(*arguments))
                else:
                    with _deadline(timeout):
                        value = _block(function(*arguments))
            warning_messages.extend(str(item.message) for item in caught)
            return value

        first_forward_started = time.perf_counter()
        forward_values = run(forward, forward_timeout)
        first_forward_seconds = time.perf_counter() - first_forward_started
        first_derivative_started = time.perf_counter()
        try:
            if skip_derivative:
                derivative_values = None
                derivative_timeout_message = (
                    "skipped pathological low-q caustic case"
                )
            else:
                with _deadline(derivative_timeout):
                    derivative = self._derivative_function(
                        profile, profile_c, target, n_annuli, arguments
                    )
                    derivative_values = run(derivative)
                derivative_timeout_message = None
        except _DeadlineExceeded as error:
            derivative_values = None
            derivative_timeout_message = str(error)
        first_derivative_seconds = (
            time.perf_counter() - first_derivative_started
        )

        forward_samples = []
        derivative_samples = [] if derivative_values is not None else None
        for _ in range(max(1, int(repeats))):
            started = time.perf_counter()
            run(forward, forward_timeout)
            forward_samples.append(time.perf_counter() - started)
        if derivative_values is not None:
            for _ in range(max(1, int(repeats))):
                started = time.perf_counter()
                try:
                    run(derivative, derivative_timeout)
                except _DeadlineExceeded as error:
                    derivative_timeout_message = str(error)
                    derivative_samples = None
                    break
                derivative_samples.append(time.perf_counter() - started)

        return {
            "forward_values": np.asarray(forward_values, dtype=float),
            "dA_dt_values": (
                None
                if derivative_values is None
                else np.asarray(derivative_values, dtype=float)
            ),
            "forward_block_seconds": float(np.median(forward_samples)),
            "forward_samples_seconds": forward_samples,
            "dA_dt_block_seconds": (
                None
                if not derivative_samples
                else float(np.median(derivative_samples))
            ),
            "dA_dt_samples_seconds": derivative_samples,
            "forward_first_after_warmup_seconds": float(
                first_forward_seconds
            ),
            "dA_dt_first_after_warmup_seconds": float(
                first_derivative_seconds
            ),
            "dA_dt_timeout": derivative_values is None
            or derivative_samples is None,
            "dA_dt_timeout_message": derivative_timeout_message,
            "warnings": warning_messages,
            "budget_exhausted": self._budget_exhausted(warning_messages),
        }


def _load_rows(path):
    payload = json.loads(Path(path).read_text())
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(
            f"{path} does not contain a 'results' list; use the controlled "
            "pure-kernel report_v2/results.json"
        )
    rows = [dict(row) for row in rows]

    # Timed-out native/VBM jobs are compact status records.  Their sibling
    # corpus still contains the geometry and high-accuracy reference values,
    # so join those fields back and retain the nominal 12,800 epochs.
    corpus_value = payload.get("corpus") or payload.get("input")
    corpus_path = None
    if corpus_value:
        candidate = Path(corpus_value)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_dir():
            candidate = candidate / "rows.json"
        if candidate.is_file():
            corpus_path = candidate
    if corpus_path is not None:
        corpus_payload = json.loads(corpus_path.read_text())
        corpus_rows = corpus_payload.get("rows", ())
        lookup = {
            (
                int(item["case_id"]),
                str(item["profile"]),
                f"{float(item['x']):.17g}",
                f"{float(item['y']):.17g}",
            ): item
            for item in corpus_rows
        }
        for row in rows:
            if row.get("x") is None or row.get("y") is None:
                continue
            key = (
                int(row["case_id"]),
                str(row["profile"]),
                f"{float(row['x']):.17g}",
                f"{float(row['y']):.17g}",
            )
            source = lookup.get(key)
            if source is None:
                continue
            for name in (
                "s",
                "q",
                "rho",
                "limb_darkening_c",
                "d_over_rho",
                "chosen_nbin",
                "chosen_grid",
            ):
                if row.get(name) is None and source.get(name) is not None:
                    row[name] = source[name]
            references = source.get("references", {})
            values = [
                references.get(str(index), {}).get("value")
                for index in REFERENCE_INDICES
            ]
            uncertainties = [
                references.get(str(index), {}).get("uncertainty")
                for index in REFERENCE_INDICES
            ]
            # The report's ``reference`` is a target-tolerance VBM call.  It
            # is useful for reproducing the old native report, but it is not
            # the oracle for a new annulus warm-up.  The corpus references are
            # the fine VBM values paired with a per-epoch uncertainty.
            if all(_finite(value) for value in values):
                row["accuracy_reference"] = values
                row["accuracy_reference_uncertainty"] = uncertainties
                row["accuracy_reference_floor"] = source.get(
                    "reference_floor"
                )
                row["accuracy_reference_source"] = (
                    "corpus_vbm_fine_reltol_1e-7"
                )
            if row.get("reference") is None and all(
                _finite(value) for value in values
            ):
                row["reference"] = values
                row["reference_source"] = "joined_corpus_rows"
    return payload, rows


def _reference_bundle(row):
    """Return the target-specific VBM oracle, uncertainty, and provenance.

    The corpus stores both a target-tolerance VBM call (``reference``) and a
    finer diagnostic value (``accuracy_reference``).  Warm-up ``nbin`` was
    selected against the former, so it is the correct pass/fail oracle for a
    target lane; the finer value remains available in the input and output
    metadata for a separate accuracy audit.
    """

    values = row.get("reference")
    if values is None:
        values = row.get("accuracy_reference")
    if values is None:
        values = [None] * len(REFERENCE_INDICES)
    uncertainties = row.get("accuracy_reference_uncertainty")
    source = row.get("reference_source")
    if source is None:
        source = (
            "report_target_vbm"
            if row.get("reference") is not None
            else row.get("accuracy_reference_source")
        )
    return (
        [float(value) if _finite(value) else None for value in values],
        (
            [float(value) if _finite(value) else None for value in uncertainties]
            if uncertainties is not None
            else None
        ),
        source,
    )


def _select_rows(rows, profiles, targets, case_id_min, case_id_max, max_jobs):
    profiles = set(profiles) if profiles else None
    targets = {float(value) for value in targets} if targets else None
    selected = []
    for row in rows:
        if profiles is not None and row.get("profile") not in profiles:
            continue
        if targets is not None and float(row["target"]) not in targets:
            continue
        case_id = int(row["case_id"])
        if case_id_min is not None and case_id < case_id_min:
            continue
        if case_id_max is not None and case_id >= case_id_max:
            continue
        reference, _, _ = _reference_bundle(row)
        if reference is not None and (
            len(reference) != len(REFERENCE_INDICES)
            or not all(_finite(value) for value in reference)
        ):
            continue
        if not all(
            row.get(name) is not None
            for name in ("s", "q", "rho", "x", "y")
        ):
            continue
        selected.append(row)
        if max_jobs is not None and len(selected) >= max_jobs:
            break
    return selected


def _stats(values):
    finite = np.asarray(
        [
            float(value)
            for value in values
            if _finite(value) and float(value) > 0.0
        ],
        dtype=float,
    )
    if not finite.size:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "median_seconds": float(np.median(finite)),
        "p10_seconds": float(np.percentile(finite, 10)),
        "p50_seconds": float(np.percentile(finite, 50)),
        "p90_seconds": float(np.percentile(finite, 90)),
        "minimum_seconds": float(np.min(finite)),
    }


def _summary(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["profile"], float(result["target"]))].append(result)
    summary = {}
    for (profile, target), rows in sorted(grouped.items()):
        key = f"{profile}:target={target:g}"
        summary[key] = {
            "jobs": len(rows),
            "epochs": sum(int(row.get("batch_epochs", 0)) for row in rows),
            "status_counts": {
                status: sum(row.get("status") == status for row in rows)
                for status in sorted({row.get("status") for row in rows})
            },
            "jax_forward_block": _stats(
                row.get("jax_forward_block_seconds") for row in rows
            ),
            "jax_forward_seconds_per_epoch": _stats(
                row.get("jax_forward_seconds_per_epoch") for row in rows
            ),
            "jax_dA_dt_block": _stats(
                row.get("jax_dA_dt_block_seconds") for row in rows
            ),
            "jax_dA_dt_seconds_per_epoch": _stats(
                row.get("jax_dA_dt_seconds_per_epoch") for row in rows
            ),
            "jax_dA_dt_timeout_count": sum(
                bool(row.get("jax_dA_dt_timeout")) for row in rows
            ),
            "jax_forward_skipped_hard_case_count": sum(
                bool(row.get("jax_forward_skipped_hard_case"))
                for row in rows
            ),
            "derivative_skipped_hard_case_count": sum(
                bool(row.get("derivative_skipped_hard_case"))
                for row in rows
            ),
            "microlux_forward_block": _stats(
                row.get("microlux_forward_block_seconds") for row in rows
            ),
            "microlux_forward_seconds_per_epoch": _stats(
                row.get("microlux_forward_seconds_per_epoch")
                for row in rows
            ),
            "microlux_dA_dt_block": _stats(
                row.get("microlux_dA_dt_block_seconds") for row in rows
            ),
            "microlux_dA_dt_seconds_per_epoch": _stats(
                row.get("microlux_dA_dt_seconds_per_epoch")
                for row in rows
            ),
            "microlux_default_forward_block": _stats(
                row.get("microlux_default_forward_block_seconds")
                for row in rows
            ),
            "microlux_default_forward_seconds_per_epoch": _stats(
                row.get("microlux_default_forward_seconds_per_epoch")
                for row in rows
            ),
            "microlux_default_dA_dt_block": _stats(
                row.get("microlux_default_dA_dt_block_seconds")
                for row in rows
            ),
            "microlux_default_dA_dt_seconds_per_epoch": _stats(
                row.get("microlux_default_dA_dt_seconds_per_epoch")
                for row in rows
            ),
            "microlux_default_dA_dt_timeout_count": sum(
                bool(row.get("microlux_default_dA_dt_timeout"))
                for row in rows
            ),
            "microlux_over_jax_forward_block": _stats(
                row.get("microlux_over_jax_forward") for row in rows
            ),
            "microlux_over_jax_dA_dt_block": _stats(
                row.get("microlux_over_jax_dA_dt") for row in rows
            ),
            "jax_selected_resolution": _stats(
                row.get("jax_selected_resolution") for row in rows
            ),
            "jax_max_relative_error": _stats(
                row.get("jax_max_relative_error") for row in rows
            ),
            "microlux_max_relative_error": _stats(
                row.get("microlux_max_relative_error") for row in rows
            ),
            "microlux_default_max_relative_error": _stats(
                row.get("microlux_default_max_relative_error")
                for row in rows
            ),
            "microlux_selected_n_annuli": _stats(
                row.get("microlux_selected_n_annuli") for row in rows
            ),
            "matched_speed_eligible": sum(
                bool(row.get("matched_speed_eligible")) for row in rows
            ),
            "matched_dA_dt_eligible": sum(
                bool(row.get("matched_dA_dt_eligible")) for row in rows
            ),
            "microlux_accuracy_status_counts": {
                status: sum(
                    row.get("microlux_accuracy_status") == status
                    for row in rows
                )
                for status in sorted(
                    {row.get("microlux_accuracy_status") for row in rows},
                    key=lambda value: "" if value is None else str(value),
                )
            },
        }
    return summary


def _row_forward_timeout(configured):
    """Use the configured forward timeout uniformly for every row.

    A previous version imposed a hidden 20-second cap after the first JAX
    executable had warmed.  That cap converted legitimate low-q Cartesian
    work into artificial row timeouts, so the retry must use the same explicit
    timeout for compilation, warm-up, and steady-state calls.
    """

    return configured


def _skip_pathological_derivative(row):
    """Compatibility hook: the native-routed backend measures every row."""

    del row
    return False


def _skip_pathological_forward(row):
    """Compatibility hook: the native-routed backend measures every row."""

    del row
    return False


def _skipped_timing(message):
    return {
        "forward_values": None,
        "dA_dt_values": None,
        "forward_block_seconds": None,
        "forward_samples_seconds": [],
        "dA_dt_block_seconds": None,
        "dA_dt_samples_seconds": None,
        "forward_first_after_warmup_seconds": 0.0,
        "dA_dt_first_after_warmup_seconds": 0.0,
        "dA_dt_timeout": True,
        "dA_dt_timeout_message": message,
        "warnings": [],
        "budget_exhausted": False,
    }


def _job(row, jax_executor, micro_executor, repeats, fallback_resolution,
         forward_timeout, derivative_timeout, jax_refined_executor=None):
    profile = str(row["profile"])
    target = float(row["target"])
    profile_c = float(row.get("limb_darkening_c", 0.0))
    times = _times(row)
    references, reference_uncertainties, reference_source = _reference_bundle(
        row
    )
    report_reference = row.get("reference")
    forward_timeout = _row_forward_timeout(forward_timeout)
    # These flags remain in the row schema for compatibility with old result
    # files, but no geometry-based skip policy is used anymore.
    skip_forward = False
    skip_derivative = False

    stage = "jax_select"
    try:
        jax_selection = jax_executor.select(
            profile,
            profile_c,
            row,
            times,
            references,
            target,
            fallback_resolution,
            forward_timeout,
        )
        timing_executor = jax_executor
        if (
            jax_refined_executor is not None
            and not jax_selection["target_pass"]
            and _native_plan_certified(row)
            and any(
                route == "cartesian"
                for route in jax_selection["route_plan"]
            )
        ):
            refined_selection = jax_refined_executor.select(
                profile,
                profile_c,
                row,
                times,
                references,
                target,
                fallback_resolution,
                forward_timeout,
            )
            if (
                refined_selection["target_pass"]
                or refined_selection["selected_max_relative_error"]
                < jax_selection["selected_max_relative_error"]
            ):
                jax_selection = refined_selection
                timing_executor = jax_refined_executor
        stage = "jax_derivative_compile"
        derivative_compile_seconds = timing_executor.prepare_derivative(
            profile,
            profile_c,
            target,
            jax_selection["route_plan"],
            jax_selection["resolution_plan"],
            row,
            times,
            derivative_timeout,
        )
        jax_selection["warmup_seconds"] += derivative_compile_seconds
        stage = "jax_timing"
        jax_timing = timing_executor.timed(
            profile,
            profile_c,
            target,
            jax_selection["route_plan"],
            jax_selection["resolution_plan"],
            row,
            times,
            repeats,
            forward_timeout,
            derivative_timeout,
            skip_derivative,
        )
        stage = "microlux_select"
        micro_selection = micro_executor.select(
            profile,
            profile_c,
            target,
            row,
            times,
            references,
            reference_uncertainties,
            forward_timeout,
        )
        default_n_annuli = (
            MICROLUX_DEFAULT_N_ANNULI if profile == "linear" else None
        )
        stage = "microlux_default_timing"
        default_timing = micro_executor.timed(
            profile,
            profile_c,
            target,
            default_n_annuli,
            row,
            times,
            repeats,
            forward_timeout,
            derivative_timeout,
            skip_derivative,
        )
        selected_n_annuli = micro_selection["selected_n_annuli"]
        matched_timing = None
        if micro_selection["selected_entry"] is not None:
            if selected_n_annuli == default_n_annuli:
                matched_timing = default_timing
            else:
                stage = "microlux_matched_timing"
                matched_timing = micro_executor.timed(
                    profile,
                    profile_c,
                    target,
                    selected_n_annuli,
                    row,
                    times,
                    repeats,
                    forward_timeout,
                    derivative_timeout,
                    skip_derivative,
                )
    except _DeadlineExceeded as error:
        return {
            "status": "timeout",
            "error": f"{type(error).__name__}: {error}",
            "timeout_stage": stage,
            "case_id": int(row["case_id"]),
            "profile": profile,
            "target": target,
            "batch_epochs": len(times),
        }
    except Exception as error:  # noqa: BLE001
        return {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "error_stage": stage,
            "case_id": int(row["case_id"]),
            "profile": profile,
            "target": target,
            "batch_epochs": len(times),
        }

    jax_values = jax_timing["forward_values"]
    jax_derivatives = jax_timing["dA_dt_values"]
    jax_errors = (
        [
            _relative_error(value, reference)
            for value, reference in zip(jax_values, references)
        ]
        if jax_values is not None
        else [None] * len(REFERENCE_INDICES)
    )
    default_values = default_timing["forward_values"]
    default_derivatives = default_timing["dA_dt_values"]
    default_errors = [
        _relative_error(value, reference)
        for value, reference in zip(default_values, references)
    ]
    matched_values = (
        matched_timing["forward_values"] if matched_timing is not None else None
    )
    matched_derivatives = (
        matched_timing["dA_dt_values"] if matched_timing is not None else None
    )
    matched_errors = (
        [
            _relative_error(value, reference)
            for value, reference in zip(matched_values, references)
        ]
        if matched_values is not None
        else [None] * len(REFERENCE_INDICES)
    )
    jax_block = jax_timing["forward_block_seconds"]
    if jax_block is not None:
        jax_block = float(jax_block)
    jax_dadt_block = jax_timing["dA_dt_block_seconds"]
    if jax_dadt_block is not None:
        jax_dadt_block = float(jax_dadt_block)
    default_block = float(default_timing["forward_block_seconds"])
    default_dadt_block = default_timing["dA_dt_block_seconds"]
    if default_dadt_block is not None:
        default_dadt_block = float(default_dadt_block)
    matched_block = (
        float(matched_timing["forward_block_seconds"])
        if matched_timing is not None
        else None
    )
    matched_dadt_block = (
        None
        if matched_timing is None
        or matched_timing["dA_dt_block_seconds"] is None
        else float(matched_timing["dA_dt_block_seconds"])
    )
    epoch_count = len(times)

    def _finite_max(values):
        if values is None or not any(_finite(value) for value in values):
            return None
        return float(np.nanmax(np.asarray(values, dtype=float)))

    def _as_list(values):
        return None if values is None else values.tolist()

    def _per_epoch(seconds):
        return None if seconds is None else seconds / epoch_count

    def _ratio(numerator, denominator):
        if numerator is None or denominator is None or denominator == 0.0:
            return None
        return numerator / denominator

    def _calibration_json(entry):
        result = {}
        for key, value in entry.items():
            if key == "values":
                continue
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, np.generic):
                result[key] = value.item()
            else:
                result[key] = value
        return result

    default_entry = micro_selection["default_entry"]
    default_warnings = (
        [] if default_entry is None else list(default_entry["warnings"])
    )
    default_warnings.extend(default_timing.get("warnings", ()))
    default_budget_exhausted = (
        False
        if default_entry is None
        else bool(default_entry["budget_exhausted"])
    ) or bool(default_timing.get("budget_exhausted"))
    return {
        "status": "completed",
        "case_id": int(row["case_id"]),
        "input_status": row.get("status"),
        "profile": profile,
        "target": target,
        "limb_darkening_c": profile_c,
        "s": float(row["s"]),
        "q": float(row["q"]),
        "q_used_by_lcbinint_jax": 1.0 / float(row["q"]),
        "rho": float(row["rho"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "d_over_rho": float(row.get("d_over_rho", float("nan"))),
        "times": times.tolist(),
        "batch_epochs": int(epoch_count),
        "jax_forward_skipped_hard_case": skip_forward,
        "derivative_skipped_hard_case": skip_derivative,
        "reference": references,
        "report_reference": report_reference,
        "reference_uncertainty": reference_uncertainties,
        "reference_available": all(_finite(value) for value in references),
        "reference_certified_for_target": micro_selection[
            "reference_certified"
        ],
        "reference_source": reference_source,
        "reference_floor": row.get("accuracy_reference_floor"),
        "native_warmup_chosen_grid": row.get("chosen_grid"),
        "native_warmup_chosen_nbin": row.get("chosen_nbin"),
        "jax_warmup_mode": jax_selection["warmup_mode"],
        "jax_convergence_certified": jax_selection[
            "convergence_certified"
        ],
        "jax_support_valid": jax_selection["support_valid"],
        "jax_certified_pass": jax_selection["certified_pass"],
        "jax_discovery_overflow": jax_selection["discovery_overflow"],
        "jax_root_failure": jax_selection["root_failure"],
        "jax_boundary_subdivision": jax_selection[
            "boundary_subdivision"
        ],
        "jax_route_mode": "convergence_selected_static_ffi",
        "jax_route_plan": jax_selection["route_plan"],
        "jax_resolution_plan": jax_selection["resolution_plan"],
        "jax_route_plan_source": jax_selection["route_plan_source"],
        "jax_native_plan_start": jax_selection["native_plan_start"],
        "jax_native_plan_start_mode": jax_selection[
            "native_plan_start_mode"
        ],
        "jax_warmup_seconds": jax_selection["warmup_seconds"],
        "jax_derivative_compile_seconds": derivative_compile_seconds,
        "jax_calibration": [
            _calibration_json(entry)
            for entry in jax_selection["calibration"]
        ],
        "jax_selected_resolution": jax_selection["resolution"],
        "jax_max_source_bins": jax_selection["max_source_bins"],
        "jax_polar_angular_bins": jax_selection["polar_angular_bins"],
        "jax_polar_angular_ratio": jax_selection[
            "polar_angular_ratio"
        ],
        "jax_values": _as_list(jax_values),
        "jax_dA_dt": _as_list(jax_derivatives),
        "jax_relative_errors": jax_errors,
        "jax_max_relative_error": (
            None
            if not any(_finite(value) for value in jax_errors)
            else float(np.nanmax(np.asarray(jax_errors, dtype=float)))
        ),
        "jax_forward_block_seconds": jax_block,
        "jax_forward_seconds_per_epoch": _per_epoch(jax_block),
        "jax_dA_dt_block_seconds": jax_dadt_block,
        "jax_dA_dt_seconds_per_epoch": _per_epoch(jax_dadt_block),
        "jax_forward_samples_seconds": jax_timing[
            "forward_samples_seconds"
        ],
        "jax_dA_dt_samples_seconds": jax_timing[
            "dA_dt_samples_seconds"
        ],
        "jax_forward_first_after_warmup_seconds": jax_timing[
            "forward_first_after_warmup_seconds"
        ],
        "jax_dA_dt_first_after_warmup_seconds": jax_timing[
            "dA_dt_first_after_warmup_seconds"
        ],
        "jax_dA_dt_timeout": jax_timing["dA_dt_timeout"],
        "jax_dA_dt_timeout_message": jax_timing[
            "dA_dt_timeout_message"
        ],
        "microlux_tol": target,
        "microlux_retol": target,
        "microlux_strategy": list(_microlux_strategy(target)),
        "microlux_accuracy_status": micro_selection["status"],
        "microlux_default_n_annuli": default_n_annuli,
        "microlux_selected_n_annuli": selected_n_annuli,
        "microlux_n_annuli": selected_n_annuli,
        "microlux_warmup_seconds": micro_selection["warmup_seconds"],
        "microlux_default_warmup_seconds": (
            None if default_entry is None else default_entry["warmup_seconds"]
        ),
        "microlux_calibration": [
            _calibration_json(entry)
            for entry in micro_selection["calibration"]
        ],
        "microlux_default_values": default_values.tolist(),
        "microlux_default_dA_dt": _as_list(default_derivatives),
        "microlux_default_relative_errors": default_errors,
        "microlux_default_max_relative_error": _finite_max(default_errors),
        "microlux_default_warnings": (
            default_warnings
        ),
        "microlux_default_budget_exhausted": default_budget_exhausted,
        "microlux_default_forward_block_seconds": default_block,
        "microlux_default_forward_seconds_per_epoch": (
            default_block / epoch_count
        ),
        "microlux_default_dA_dt_block_seconds": default_dadt_block,
        "microlux_default_dA_dt_seconds_per_epoch": _per_epoch(
            default_dadt_block
        ),
        "microlux_default_forward_samples_seconds": default_timing[
            "forward_samples_seconds"
        ],
        "microlux_default_dA_dt_samples_seconds": default_timing[
            "dA_dt_samples_seconds"
        ],
        "microlux_default_forward_first_after_warmup_seconds": (
            default_timing["forward_first_after_warmup_seconds"]
        ),
        "microlux_default_dA_dt_first_after_warmup_seconds": (
            default_timing["dA_dt_first_after_warmup_seconds"]
        ),
        "microlux_default_dA_dt_timeout": default_timing[
            "dA_dt_timeout"
        ],
        "microlux_default_dA_dt_timeout_message": default_timing[
            "dA_dt_timeout_message"
        ],
        "microlux_values": (
            None if matched_values is None else matched_values.tolist()
        ),
        "microlux_dA_dt": (
            None if matched_derivatives is None else matched_derivatives.tolist()
        ),
        "microlux_relative_errors": matched_errors,
        "microlux_max_relative_error": _finite_max(matched_errors),
        "microlux_forward_block_seconds": matched_block,
        "microlux_forward_seconds_per_epoch": (
            None if matched_block is None else matched_block / epoch_count
        ),
        "microlux_dA_dt_block_seconds": matched_dadt_block,
        "microlux_dA_dt_seconds_per_epoch": _per_epoch(matched_dadt_block),
        "microlux_forward_samples_seconds": (
            None
            if matched_timing is None
            else matched_timing["forward_samples_seconds"]
        ),
        "microlux_dA_dt_samples_seconds": (
            None
            if matched_timing is None
            else matched_timing["dA_dt_samples_seconds"]
        ),
        "microlux_forward_first_after_warmup_seconds": (
            None
            if matched_timing is None
            else matched_timing["forward_first_after_warmup_seconds"]
        ),
        "microlux_dA_dt_first_after_warmup_seconds": (
            None
            if matched_timing is None
            else matched_timing["dA_dt_first_after_warmup_seconds"]
        ),
        "microlux_dA_dt_timeout": (
            None
            if matched_timing is None
            else matched_timing["dA_dt_timeout"]
        ),
        "microlux_dA_dt_timeout_message": (
            None
            if matched_timing is None
            else matched_timing["dA_dt_timeout_message"]
        ),
        "microlux_over_jax_forward": _ratio(matched_block, jax_block),
        "microlux_over_jax_dA_dt": (
            None
            if matched_dadt_block is None
            else _ratio(matched_dadt_block, jax_dadt_block)
        ),
        "microlux_default_over_jax_forward": _ratio(
            default_block, jax_block
        ),
        "microlux_default_over_jax_dA_dt": _ratio(
            default_dadt_block, jax_dadt_block
        ),
        "matched_speed_eligible": (
            matched_timing is not None
            and matched_block is not None
            and jax_block is not None
        ),
        "matched_dA_dt_eligible": (
            matched_timing is not None and matched_dadt_block is not None
            and jax_dadt_block is not None
        ),
    }


def _split_lane_target(value):
    return f"{float(value):.0e}".replace("+", "p").replace("-", "m")


def _run_split_lanes(args):
    """Run each profile/target lane in a fresh process and merge the rows.

    JAX executable memory is retained by the process even after a callable is
    no longer referenced.  Keeping all four profile/target combinations alive
    can make LLVM fail during a later compile.  A lane boundary therefore
    keeps the process memory bounded without changing any timed call.
    """

    profiles = tuple(args.profiles or ("uniform", "linear"))
    targets = tuple(float(value) for value in args.targets)
    with tempfile.TemporaryDirectory(prefix="jax-microlux-lanes-") as temp:
        temporary = Path(temp)
        part_paths = []
        for profile in profiles:
            for target in targets:
                part = temporary / (
                    f"{profile}-target-{_split_lane_target(target)}.json"
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--input",
                    str(args.input),
                    "--output",
                    str(part),
                    "--profiles",
                    profile,
                    "--targets",
                    str(target),
                    "--repeats",
                    str(args.repeats),
                    "--derivative-timeout",
                    str(args.derivative_timeout),
                    "--forward-timeout",
                    str(args.forward_timeout),
                    "--fallback-resolution",
                    str(args.fallback_resolution),
                    "--max-source-bins",
                    str(args.max_source_bins),
                    "--microlux-max-annuli",
                    str(args.microlux_max_annuli),
                ]
                if args.case_id_min is not None:
                    command.extend(("--case-id-min", str(args.case_id_min)))
                if args.case_id_max is not None:
                    command.extend(("--case-id-max", str(args.case_id_max)))
                if args.max_jobs is not None:
                    command.extend(("--max-jobs", str(args.max_jobs)))
                subprocess.run(command, check=True)
                part_paths.append(part)

        payloads = [json.loads(path.read_text()) for path in part_paths]
        results = [
            result
            for payload in payloads
            for result in payload.get("results", ())
        ]
        merged = dict(payloads[0])
        merged["input_metadata"] = dict(payloads[0].get("input_metadata", {}))
        merged["input_metadata"]["nominal_epochs"] = sum(
            int(payload.get("input_metadata", {}).get("nominal_epochs", 0))
            for payload in payloads
        )
        merged["input_metadata"]["lane_processes"] = len(payloads)
        merged["timing_mode"] = "compiled_warm_batched_reference_epochs_split_lanes"
        merged["configuration"] = dict(payloads[0].get("configuration", {}))
        merged["configuration"]["profiles"] = list(profiles)
        merged["configuration"]["targets"] = list(targets)
        merged["configuration"]["split_lanes"] = True
        merged["compile_records"] = {
            "lcbinint_jax": [
                record
                for payload in payloads
                for record in payload.get("compile_records", {}).get(
                    "lcbinint_jax", ()
                )
            ],
            "microlux": [
                record
                for payload in payloads
                for record in payload.get("compile_records", {}).get(
                    "microlux", ()
                )
            ],
        }
        merged["results"] = results
        merged["summary"] = _summary(results)
        merged["elapsed_seconds"] = sum(
            float(payload.get("elapsed_seconds", 0.0))
            for payload in payloads
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(merged, indent=2) + "\n")
        print(json.dumps(merged["summary"], indent=2), flush=True)
        print(f"saved {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("uniform", "linear"))
    parser.add_argument(
        "--targets", nargs="+", type=float, default=(1.0e-3, 1.0e-4)
    )
    parser.add_argument("--case-id-min", type=int)
    parser.add_argument("--case-id-max", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--derivative-timeout",
        type=float,
        default=60.0,
        help=(
            "maximum seconds for one compiled dA/dt block call; non-positive "
            "disables the guard (default: 60)"
        ),
    )
    parser.add_argument(
        "--forward-timeout",
        type=float,
        default=60.0,
        help=(
            "maximum seconds for one compiled forward/calibration call; "
            "non-positive disables the guard (default: 60)"
        ),
    )
    parser.add_argument(
        "--fallback-resolution",
        type=int,
        default=128,
        help=(
            "legacy compatibility option; missing native plans are handled by "
            "the calibrated native route selector (default: 128)"
        ),
    )
    parser.add_argument(
        "--max-source-bins",
        type=int,
        default=400,
        help=(
            "calibrated maximum source-bin bucket for the native route "
            "selector (default: 400)"
        ),
    )
    parser.add_argument(
        "--microlux-max-annuli",
        type=int,
        default=MICROLUX_DEFAULT_N_ANNULI,
        help=(
            "legacy compatibility option; the benchmark always uses the "
            f"microLUX default n_annuli={MICROLUX_DEFAULT_N_ANNULI}"
        ),
    )
    parser.add_argument(
        "--split-lanes",
        action="store_true",
        help=(
            "run each profile/target in a fresh process before merging; "
            "recommended for the full four-lane run"
        ),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if not np.isfinite(args.derivative_timeout):
        raise SystemExit("--derivative-timeout must be finite")
    if not np.isfinite(args.forward_timeout):
        raise SystemExit("--forward-timeout must be finite")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise SystemExit("--max-jobs must be positive")
    if args.fallback_resolution < 2:
        raise SystemExit("--fallback-resolution must be at least 2")
    if args.microlux_max_annuli < MICROLUX_DEFAULT_N_ANNULI:
        raise SystemExit(
            "--microlux-max-annuli must be at least "
            f"{MICROLUX_DEFAULT_N_ANNULI}"
        )

    profiles = tuple(args.profiles or ("uniform", "linear"))
    targets = tuple(float(value) for value in args.targets)
    if args.split_lanes and len(profiles) * len(targets) > 1:
        _run_split_lanes(args)
        return

    input_payload, source_rows = _load_rows(args.input)
    rows = _select_rows(
        source_rows,
        args.profiles,
        args.targets,
        args.case_id_min,
        args.case_id_max,
        args.max_jobs,
    )
    if not rows:
        raise SystemExit("no rows selected")

    print(
        f"selected {len(rows)} jobs, "
        f"{len(rows) * len(REFERENCE_INDICES)} batched epochs; "
        "native-routed calibrated JAX warm-up + compiled timing, "
        f"repeats={args.repeats}",
        flush=True,
    )
    jax_executor = JaxBatchExecutor(args.max_source_bins)
    jax_refined_executor = JaxBatchExecutor(
        args.max_source_bins,
        boundary_subdivision=CARTESIAN_REFINED_BOUNDARY_SUBDIVISION,
    )
    micro_executor = MicroLuxBatchExecutor(args.microlux_max_annuli)
    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        print(
            f"[{index}/{len(rows)}] case={row['case_id']} "
            f"profile={row['profile']} target={float(row['target']):g}",
            flush=True,
        )
        result = _job(
            row,
            jax_executor,
            micro_executor,
            args.repeats,
            args.fallback_resolution,
            args.forward_timeout,
            args.derivative_timeout,
            jax_refined_executor,
        )
        results.append(result)
        if result["status"] == "completed":
            jax_block = result.get("jax_forward_block_seconds")
            jax_block_text = (
                "skipped" if jax_block is None else f"{jax_block:.6g}s"
            )
            detail = (
                f"jax_resolution={result['jax_selected_resolution']} "
                f"jax_block={jax_block_text} "
                f"micro_default10={result['microlux_default_forward_block_seconds']:.6g}s "
                f"annuli={result.get('microlux_selected_n_annuli')} "
                f"accuracy={result.get('microlux_accuracy_status')}"
            )
        else:
            detail = result.get("error", "")
        print(f"  {result['status']} {detail}", flush=True)

    try:
        import microlux

        microlux_path = str(Path(microlux.__file__).resolve())
        microlux_commit = _checkout(microlux)
    except Exception:  # noqa: BLE001
        microlux_path = ""
        microlux_commit = ""

    payload = {
        "input": str(args.input),
        "input_metadata": {
            "source_payload_keys": sorted(input_payload),
            "reference_indices": list(REFERENCE_INDICES),
            "reference_epoch_count": len(REFERENCE_INDICES),
            "nominal_epochs": len(rows) * len(REFERENCE_INDICES),
            "reference_policy": (
                "corpus fine VBM reference retained for error reporting; "
                "it does not select microLUX annuli"
            ),
        },
        "timing_mode": "compiled_warm_batched_reference_epochs",
        "derivative_mode": "source_trajectory_dA_dt_forward_mode_jvp",
        "coordinate_mode": (
            "VBM corpus; lcbinint_jax q->1/q; microLUX x-mirror with "
            "chronological block reversal"
        ),
        "lcbinint_mode": "native_plan_direct_per_epoch_ffi",
        "jax_batch_resolution_policy": (
            "group equal native route/bin pairs within each four-epoch block; "
            "execute polar or Cartesian direct FFI calls per group"
        ),
        "warmup_mode": (
            "use saved native chosen_grid/chosen_nbin directly; missing native "
            "plans use JAX routing diagnostics; use microLUX library default "
            "n_annuli=10"
        ),
        "compile_policy": (
            "compile-and-first warm-up calls are excluded from "
            "steady-state block samples; forward and dA_dt use separate "
            "synchronized compiled callables"
        ),
        "configuration": {
            "profiles": args.profiles,
            "targets": [float(value) for value in args.targets],
            "reference_indices": list(REFERENCE_INDICES),
            "batch_epochs": len(REFERENCE_INDICES),
            "jax_resolution_policy": (
                "saved native per-epoch chosen_grid/chosen_nbin; missing-plan "
                "rows use the JAX calibrated selector"
            ),
            "jax_fallback_resolution": args.fallback_resolution,
            "jax_max_source_bins": args.max_source_bins,
            "jax_relative_tolerance_policy": "relative_tolerance=target",
            "jax_absolute_tolerance_policy": "absolute_tolerance=0",
            "jax_geometry_skip_policy": "none",
            "jax_angular_policy": (
                "global angular bins 65536,131072,262144,524288,1048576,2097152 "
                "with adjacent-grid convergence; stable misses are recorded "
                "as unresolved and stop the ladder; "
                "no q/d-over-rho branches"
            ),
            "jax_cartesian_boundary_policy": (
                "compiled 4x4 boundary quadrature by default; native-certified "
                "rows whose 4x4 result misses target are rechecked with the "
                "global 8x8 quadrature rule"
            ),
            "jax_cartesian_boundary_subdivision": (
                CARTESIAN_BOUNDARY_SUBDIVISION
            ),
            "jax_cartesian_refined_boundary_subdivision": (
                CARTESIAN_REFINED_BOUNDARY_SUBDIVISION
            ),
            "microlux_tol_policy": "tol=target",
            "microlux_retol_policy": "retol=target",
            "microlux_strategy_policy": (
                "library default strategy at 1e-3; "
                "(60,60,120,240,480) for tighter targets"
            ),
            "microlux_n_annuli_policy": (
                "fixed at the microLUX library default n_annuli=10"
            ),
            "microlux_annuli_ladder": list(MICROLUX_ANNULI_LADDER),
            "microlux_linear_default_n_annuli": MICROLUX_DEFAULT_N_ANNULI,
            "microlux_max_annuli": args.microlux_max_annuli,
            "microlux_reference_uncertainty_fraction": (
                REFERENCE_UNCERTAINTY_FRACTION
            ),
            "derivative_timeout_seconds": args.derivative_timeout,
            "derivative_policy": (
                "JVP measured for every selected row; no low-q geometry skip"
            ),
            "forward_policy": (
                "direct native-plan forward call for every selected row; "
                "low-q Cartesian plans may use polar and no row is skipped"
            ),
            "forward_timeout_seconds": args.forward_timeout,
            "repeats": args.repeats,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "lcbinint_build_root": str(RUNTIME_BUILD_ROOT),
            "lcbinint_extension": str(RUNTIME_EXTENSION),
            "lcbinint_jax_path": str(
                Path(sys.modules["lcbinint_jax"].__file__).resolve()
            ),
            "jax_ffi_capabilities": JAX_FFI_CAPABILITIES,
            "microlux_path": microlux_path,
            "microlux_commit": microlux_commit,
        },
        "results": results,
        "compile_records": {
            "lcbinint_jax": [
                *jax_executor.compile_records.values(),
                *jax_refined_executor.compile_records.values(),
            ],
            "microlux": list(micro_executor.compile_records.values()),
        },
        "summary": _summary(results),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

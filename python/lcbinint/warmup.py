"""Baseline-anchored, per-epoch execution plans for :class:`LightCurve`."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


POINT_SOURCE = 0
HEXADECAPOLE = 1
CARTESIAN = 2
POLAR = 3
SPINE = 4
SOURCE_PLANE = 5

METHOD_NAMES = (
    "point_source",
    "hexadecapole",
    "inverse_ray_cartesian",
    "inverse_ray_polar",
    "inverse_ray_spine",
    "source_plane_quadrature",
)

GRID_SEARCH_MINIMUM = 4
GRID_SEARCH_GROWTH = 1.5
GRID_CONFIRMATION_GROWTH = 1.25
GRID_SEARCH_MAX_ROUNDS = 6
WARMUP_SOURCE_DRIFT_LIMIT = 0.5
WARMUP_LENS_LOG_DRIFT_LIMIT = 0.1
WARMUP_RADIUS_RATIO_LIMIT = 1.5


@dataclass
class WarmupReport:
    """Diagnostic view of the plan retained by a ``LightCurve`` instance."""

    times: np.ndarray
    methods: tuple[str, ...]
    resolutions: np.ndarray
    statuses: tuple[str, ...]
    reference: np.ndarray
    budget: np.ndarray
    cartesian_seconds: np.ndarray
    polar_seconds: np.ndarray
    elapsed_seconds: float
    parameter_fingerprint: tuple
    configuration_fingerprint: tuple
    geometry: object = None

    @property
    def calibrated(self):
        return np.asarray([status == "calibrated" for status in self.statuses])

    @property
    def all_calibrated(self):
        return bool(np.all(self.calibrated))


@dataclass
class JaxWarmupReport(WarmupReport):
    """Native-anchored, JAX-routed plan compiled for the JAX backend.

    ``execution_plan`` is intentionally an implementation detail: it is the
    compiled callable retained by its owning ``LightCurve``.  The public
    report still exposes per-epoch methods and resolutions.  The native
    warm-up remains the numerical reference, while an automatic JAX warm-up
    takes its initial route/bin choices from the same JAX dispatcher that will
    be used for one-off evaluations.  This distinction matters because native
    Cartesian/Polar timings do not predict the relative cost of JAX FFI
    branches.  The inherited ``cartesian_seconds`` and ``polar_seconds``
    fields remain native candidate timings for diagnostics; they are not used
    to select the automatic JAX route.
    """

    execution_plan: object = None
    jax_compile_seconds: float = 0.0
    seed_cache_primed: bool = False

    @property
    def calibrated(self):
        return np.ones(self.times.shape, dtype=bool)

    @property
    def all_calibrated(self):
        return True


@dataclass(frozen=True)
class WarmupGeometry:
    """Root-free lens/source geometry retained for proposal drift checks."""

    source_x: np.ndarray
    source_y: np.ndarray
    separation: np.ndarray
    mass_ratio: np.ndarray
    source_radius: np.ndarray
    caustic_distance: np.ndarray
    topology: np.ndarray


@dataclass(frozen=True)
class WarmupDriftReport:
    """Difference between a proposal and the geometry used for warmup."""

    available: bool
    warn: bool
    reasons: tuple[str, ...]
    topology_changed: bool = False
    maximum_source_drift: float = 0.0
    maximum_lens_drift: float = 0.0
    maximum_radius_drift: float = 0.0


def _binary_topology(separation, mass_ratio):
    separation = np.asarray(separation, dtype=float)
    mass_ratio = np.asarray(mass_ratio, dtype=float)
    mass_product = mass_ratio / np.square(1.0 + mass_ratio)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        close_rhs = np.power(1.0 - np.power(separation, 4), 3) / (
            27.0 * np.power(separation, 8)
        )
        wide_boundary = np.sqrt(
            np.power(1.0 + np.cbrt(mass_ratio), 3) / (1.0 + mass_ratio)
        )
    topology = np.full(separation.shape, "resonant", dtype="<U8")
    topology[(separation < 1.0) & (close_rhs > mass_product)] = "close"
    topology[separation > wide_boundary] = "wide"
    return topology


def _geometry_array(value, *names):
    for name in names:
        if hasattr(value, name):
            return np.asarray(getattr(value, name), dtype=float)
    raise AttributeError(f"geometry has none of {names!r}")


def build_warmup_geometry(curve, times, params, diagnostics):
    """Capture the geometry that makes an epoch plan locally reusable."""

    geometry = curve.finite_source_geometry(times, params)
    source_x = _geometry_array(geometry, "source_x")
    source_y = _geometry_array(geometry, "source_y")
    separation = _geometry_array(geometry, "separation", "separations")
    mass_ratio = _geometry_array(geometry, "mass_ratio", "mass_ratios")
    source_radius = _geometry_array(geometry, "source_radius")
    caustic_distance = _geometry_array(
        diagnostics, "caustic_distances", "caustic_distance"
    )
    return WarmupGeometry(
        source_x=source_x.copy(),
        source_y=source_y.copy(),
        separation=separation.copy(),
        mass_ratio=mass_ratio.copy(),
        source_radius=source_radius.copy(),
        caustic_distance=caustic_distance.copy(),
        topology=_binary_topology(separation, mass_ratio),
    )


def compare_warmup_geometry(curve, times, params, warm):
    """Return a conservative, root-free warning decision for a proposal."""

    if warm is None:
        return WarmupDriftReport(False, False, ("no geometry snapshot",))
    try:
        geometry = curve.finite_source_geometry(times, params)
        source_x = _geometry_array(geometry, "source_x")
        source_y = _geometry_array(geometry, "source_y")
        separation = _geometry_array(geometry, "separation", "separations")
        mass_ratio = _geometry_array(geometry, "mass_ratio", "mass_ratios")
        source_radius = _geometry_array(geometry, "source_radius")
    except Exception:
        # Traced JAX proposals cannot be converted to host NumPy here.  Their
        # compiled execution remains valid; concrete callers can explicitly
        # inspect drift before entering a trace.
        return WarmupDriftReport(False, False, ("geometry is not concrete",))
    if source_x.shape != warm.source_x.shape:
        return WarmupDriftReport(True, True, ("epoch shape changed",))

    topology = _binary_topology(separation, mass_ratio)
    topology_changed = bool(np.any(topology != warm.topology))
    clearance_scale = np.maximum.reduce((
        np.abs(warm.caustic_distance),
        np.abs(warm.source_radius),
        np.full(warm.source_radius.shape, 1.0e-12),
    ))
    source_drift = np.hypot(
        source_x - warm.source_x, source_y - warm.source_y
    ) / clearance_scale
    with np.errstate(divide="ignore", invalid="ignore"):
        lens_drift = np.hypot(
            np.log(separation / warm.separation),
            0.5 * np.log(mass_ratio / warm.mass_ratio),
        )
        radius_drift = np.abs(np.log(
            np.maximum(source_radius, 1.0e-300)
            / np.maximum(warm.source_radius, 1.0e-300)
        ))
    maximum_source_drift = float(np.max(np.nan_to_num(source_drift, nan=np.inf)))
    maximum_lens_drift = float(np.max(np.nan_to_num(lens_drift, nan=np.inf)))
    maximum_radius_drift = float(np.max(np.nan_to_num(radius_drift, nan=np.inf)))
    reasons = []
    if topology_changed:
        reasons.append("binary-caustic topology changed")
    if maximum_source_drift > WARMUP_SOURCE_DRIFT_LIMIT:
        reasons.append("source moved by more than half its warm caustic-clearance scale")
    if maximum_lens_drift > WARMUP_LENS_LOG_DRIFT_LIMIT:
        reasons.append("lens geometry moved by more than 0.1 in log-(s,q) distance")
    if maximum_radius_drift > np.log(WARMUP_RADIUS_RATIO_LIMIT):
        reasons.append("source radius changed by more than a factor of 1.5")
    return WarmupDriftReport(
        True,
        bool(reasons),
        tuple(reasons),
        topology_changed,
        maximum_source_drift,
        maximum_lens_drift,
        maximum_radius_drift,
    )


def _jax_auto_route_plan(curve, times, params):
    """Return the route/bin proposal produced by the automatic JAX path.

    This deliberately asks the public JAX diagnostics for the route instead
    of reproducing the dispatcher in the warm-up module.  The dispatcher has
    several stopped-gradient safety decisions (point source, multipole,
    polar fallback, and Cartesian certification); duplicating those decisions
    here would create a second routing implementation that could drift from
    production.
    """

    from .jax_backend import (
        _integration_tolerances,
        _jax_resolution_bucket,
        _limb_darkening,
    )
    from lcbinint_jax.resolution import select_binary_resolution

    import jax
    import jax.numpy as jnp

    diagnostics = curve.info(times, params)
    methods = np.asarray(
        diagnostics.finite_source_methods, dtype=np.int64
    ).copy()
    resolutions = np.zeros(methods.shape, dtype=np.int64)
    grid_mask = np.isin(methods, (CARTESIAN, POLAR))
    if not np.any(grid_mask):
        return methods, resolutions

    absolute_tolerance, relative_tolerance = _integration_tolerances(
        curve._options, "binary"
    )
    point_magnification = jnp.asarray(
        diagnostics.point_source_magnifications
    )
    limb_c, _ = _limb_darkening(curve._native, params)
    maximum_bins = _jax_resolution_bucket(curve.options.max_source_bins)
    source_radius = jnp.full_like(
        point_magnification,
        jnp.abs(jnp.asarray(params["rho"], dtype=point_magnification.dtype)),
    )
    selection = jax.vmap(
        lambda mass_ratio, source_radius, distance, point: (
            select_binary_resolution(
                mass_ratio,
                source_radius,
                distance,
                point,
                limb_c,
                requested_relative_tolerance=relative_tolerance,
                maximum_bins=maximum_bins,
                requested_absolute_tolerance=absolute_tolerance,
            )
        )
    )(
        jnp.asarray(diagnostics.mass_ratios),
        source_radius,
        jnp.asarray(diagnostics.caustic_distances),
        point_magnification,
    )
    selected_bins = np.asarray(selection.source_bins, dtype=np.int64)
    resolutions[grid_mask] = selected_bins[grid_mask]
    return methods, resolutions


def build_jax_warmup_report(
    curve,
    times,
    params,
    *,
    parameter_fingerprint,
    configuration_fingerprint,
    grid_timing_repeats=1,
):
    """Calibrate, compile, and execute a fixed JAX route/bin plan.

    The native warm-up supplies the numerical reference and the self-converged
    candidate resolutions.  For ``nbin='auto'``, the initial route split is
    taken from the JAX automatic dispatcher itself, rather than from native
    Cartesian/Polar wall-clock measurements.  JAX and native use different
    batching and FFI execution paths, so importing the native timing choice
    can select a JAX-expensive Polar branch for a trajectory that JAX would
    route through Cartesian.  The fixed plan is then certified against the
    native reference and refined in a bounded, backend-independent search.

    JAX executes the retained plan once before returning, so the XLA
    executable and the C++ support/seed caches are both hot for the first
    caller-visible evaluation.  The steady-state callable remains one fixed
    grouped plan.
    """

    from .jax_backend import make_preplanned_magnification

    concrete_times = np.asarray(times, dtype=float)
    if concrete_times.ndim == 0:
        concrete_times = concrete_times.reshape(1)
    if concrete_times.ndim != 1 or concrete_times.size == 0:
        raise ValueError("warm-up times must be a non-empty finite 1-D array")

    started = time.perf_counter()
    native_report, native_methods = build_warmup_report(
        curve,
        concrete_times,
        params,
        parameter_fingerprint=parameter_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        grid_timing_repeats=grid_timing_repeats,
    )

    methods = np.asarray(native_methods, dtype=np.int64).copy()
    resolutions = np.asarray(native_report.resolutions, dtype=np.int64).copy()
    if curve.options.nbin == "auto":
        # Do not let native route timing decide a JAX execution plan.  The
        # diagnostics call below runs the same automatic JAX dispatcher used
        # by an unwarmed evaluation and exposes its actual per-epoch route.
        # The native report remains the accuracy oracle; only the initial
        # route/bin proposal is backend-local.
        jax_methods, jax_resolutions = _jax_auto_route_plan(
            curve, concrete_times, params
        )
        supported = np.isin(
            jax_methods, (POINT_SOURCE, HEXADECAPOLE, CARTESIAN, POLAR)
        )
        methods[supported] = jax_methods[supported]
        resolutions[supported] = jax_resolutions[supported]
    reference = np.asarray(native_report.reference, dtype=float).copy()
    budget = np.asarray(native_report.budget, dtype=float).copy()
    count = int(concrete_times.size)
    cap = int(curve.options.max_source_bins)
    compile_started = time.perf_counter()
    execution_plan = None
    values = None

    def compile_and_execute():
        nonlocal execution_plan
        execution_plan = make_preplanned_magnification(
            curve._native,
            curve._options,
            tuple(int(value) for value in methods),
            tuple(int(value) for value in resolutions),
            params,
        )
        result = execution_plan(concrete_times, params)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        return np.asarray(result, dtype=float).copy()

    values = compile_and_execute()

    grid_mask = np.isin(methods, (CARTESIAN, POLAR))
    # The native report is the baseline-anchored numerical reference.  For a
    # JAX grid, require both a finite support result and agreement with that
    # reference.  Resolution refinement is a generic warm-up operation, not a
    # case-specific fallback.
    def passing(current):
        return (
            np.isfinite(current)
            & np.isfinite(reference)
            & (np.abs(current - reference) <= budget)
        )

    passed = passing(values)

    def refine_grid():
        """Refine only uncertified inverse-ray rows in the current plan."""

        nonlocal values, passed
        for _ in range(GRID_SEARCH_MAX_ROUNDS):
            failing = np.flatnonzero(grid_mask & ~passed)
            if failing.size == 0:
                break
            changed = False
            for index in failing:
                old = int(resolutions[index])
                if old <= 0 or old >= cap:
                    continue
                proposed = min(
                    cap,
                    max(old + 1, int(math.ceil(old * GRID_SEARCH_GROWTH))),
                )
                if proposed != old:
                    resolutions[index] = proposed
                    changed = True
            if not changed:
                break
            values = compile_and_execute()
            passed = passing(values)

    refine_grid()

    # A route selected by the JAX dispatcher can still differ from the native
    # reference route at a borderline epoch.  Try the already-certified native
    # route for those rows as a generic correctness reconciliation.  This is a
    # warm-up-only plan search, not a case-specific production fallback: rows
    # that pass keep the faster JAX-native route, and only rows that fail the
    # same reference/budget certificate are replaced.
    failed = np.flatnonzero(~passed)
    if failed.size and curve.options.nbin == "auto":
        candidate_methods = methods.copy()
        candidate_resolutions = resolutions.copy()
        changed = False
        for index in failed:
            native_method = int(native_methods[index])
            native_resolution = int(native_report.resolutions[index])
            if native_method not in (POINT_SOURCE, HEXADECAPOLE, CARTESIAN, POLAR):
                continue
            if native_resolution < 0:
                continue
            if (
                candidate_methods[index] != native_method
                or candidate_resolutions[index] != native_resolution
            ):
                candidate_methods[index] = native_method
                candidate_resolutions[index] = native_resolution
                changed = True
        if changed:
            methods[:] = candidate_methods
            resolutions[:] = candidate_resolutions
            grid_mask = np.isin(methods, (CARTESIAN, POLAR))
            values = compile_and_execute()
            passed = passing(values)
            refine_grid()

    failed = np.flatnonzero(~passed)
    if failed.size:
        details = ", ".join(
            f"{int(index)}:{METHOD_NAMES[int(methods[index])]}/"
            f"{int(resolutions[index])}"
            for index in failed[:12]
        )
        if failed.size > 12:
            details += ", ..."
        raise RuntimeError(
            "JAX warm-up could not certify every epoch against the native "
            f"warm-up reference ({failed.size} rows: {details})"
        )

    # Keep this warm-up focused on the retained value execution plan.  A
    # parameter Jacobian is benchmark-specific (and may use a different AD
    # transformation), so callers that measure one compile that exact
    # Jacobian explicitly rather than paying for an unrelated trajectory JVP.
    jax_compile_seconds = time.perf_counter() - compile_started
    method_names = tuple(METHOD_NAMES[int(method)] for method in methods)
    has_grid = bool(np.any(grid_mask))
    return JaxWarmupReport(
        times=concrete_times.copy(),
        methods=method_names,
        resolutions=resolutions,
        statuses=("calibrated",) * count,
        reference=reference,
        budget=budget,
        cartesian_seconds=np.asarray(native_report.cartesian_seconds).copy(),
        polar_seconds=np.asarray(native_report.polar_seconds).copy(),
        elapsed_seconds=time.perf_counter() - started,
        parameter_fingerprint=parameter_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        geometry=native_report.geometry,
        execution_plan=execution_plan,
        jax_compile_seconds=jax_compile_seconds,
        seed_cache_primed=has_grid,
    )


def _native_evaluate(curve, times, params, method, resolution):
    count = int(np.asarray(times).size)
    return curve._native._evaluate_preplanned(
        np.asarray(times, dtype=float),
        params,
        [int(method)] * count,
        [int(resolution)] * count,
    )


def _inside(values, support, reference, budget):
    return (
        np.asarray(support, dtype=bool)
        & np.isfinite(values)
        & (np.abs(np.asarray(values) - reference) <= budget)
    )


def _self_convergence_budget(values, atol, reltol):
    """Return the tolerance budget for a set of finite-source values."""

    values = np.asarray(values, dtype=float)
    scale = np.maximum(np.abs(values), 1.0)
    return max(float(atol), float(reltol) * float(np.max(scale)))


def _stable_tail(samples, atol, reltol, count=3):
    """Find a high-resolution tail whose values are mutually consistent.

    ``samples`` is keyed by increasing grid resolution.  The tail rather than
    the first passing run is intentional: an accidentally agreeable coarse
    grid must not become the reference before later refinement is observed.
    The returned reference is the median of the tail, which is insensitive to
    a small non-monotonic quadrature wobble.
    """

    ordered = sorted(samples)
    if len(ordered) < count:
        return None
    tail = ordered[-count:]
    observations = [samples[bins] for bins in tail]
    if not all(
        observation["support"] and np.isfinite(observation["value"])
        for observation in observations
    ):
        return None
    values = np.asarray(
        [observation["value"] for observation in observations], dtype=float
    )
    budget = _self_convergence_budget(values, atol, reltol)
    if float(np.max(values) - np.min(values)) > budget:
        return None
    return tuple(tail), float(np.median(values)), budget


def _interpolated_resolution(n_fail, error_fail, n_pass, error_pass, budget):
    if not (
        n_fail < n_pass
        and error_fail > budget
        and 0.0 < error_pass < budget
        and np.isfinite(error_fail)
        and np.isfinite(error_pass)
    ):
        return n_pass
    denominator = math.log(error_pass / error_fail)
    if denominator == 0.0:
        return n_pass
    fraction = math.log(budget / error_fail) / denominator
    estimate = math.exp(
        math.log(n_fail) + fraction * math.log(n_pass / n_fail)
    )
    return min(n_pass, max(n_fail + 1, int(math.ceil(estimate))))


def _predicted_resolution(
    curve, method, point_magnification, atol, reltol, cap
):
    """Use the frozen native production law as a measured-search hint."""

    return max(
        GRID_SEARCH_MINIMUM,
        int(curve._native._binary_resolution_hint(
            int(method),
            float(point_magnification),
            float(atol),
            float(reltol),
            int(cap),
        )),
    )


def _persistent_pass_run(samples):
    """Return the first run of three increasing passing grids."""

    run = []
    ordered = sorted(samples)
    for bins in ordered:
        if samples[bins]["pass"]:
            run.append(bins)
        else:
            run = []
        if len(run) >= 3:
            return tuple(run[:3])
    return None


def _candidate_batch(predicted, cap):
    values = {
        max(GRID_SEARCH_MINIMUM, int(math.ceil(predicted / 4.0))),
        max(GRID_SEARCH_MINIMUM, int(math.ceil(predicted / 2.0))),
        max(GRID_SEARCH_MINIMUM, int(predicted)),
    }
    return tuple(sorted(value for value in values if value <= cap))


def _next_candidate_batch(samples, cap):
    if not samples or max(samples) >= cap:
        return ()
    current = max(samples)
    values = {
        int(math.ceil(current * GRID_CONFIRMATION_GROWTH)),
        int(math.ceil(current * GRID_SEARCH_GROWTH)),
        int(math.ceil(current * 2.0)),
    }
    return tuple(sorted({
        min(cap, max(current + 1, value))
        for value in values
        if min(cap, max(current + 1, value)) not in samples
    }))


def _evaluate_variable_resolutions(
    curve, times, params, indices, method, resolutions
):
    indices = np.asarray(indices, dtype=int)
    return curve._native._evaluate_preplanned(
        np.asarray(times)[indices],
        params,
        [int(method)] * indices.size,
        [int(value) for value in resolutions],
    )


def _self_converged_resolutions(
    curve,
    times,
    params,
    indices,
    method,
    predicted,
    atol,
    reltol,
    cap,
):
    """Search for an internally self-converged inverse-ray reference.

    The ordinary automatic dispatcher is deliberately not used as a numerical
    oracle here.  It still supplies the empirical starting resolution through
    ``predicted``, but the target value is established from a high-resolution
    tail of the same preplanned route.  A confirmation batch after the first
    stable tail prevents a coarse accidental plateau from being accepted.
    """

    states = {
        int(index): {
            "samples": {},
            "next": _candidate_batch(int(predicted[int(index)]), int(cap)),
            "confirming": False,
        }
        for index in indices
    }
    required = {int(index): None for index in indices}
    references = {int(index): None for index in indices}
    selected_values = {int(index): None for index in indices}

    for _ in range(GRID_SEARCH_MAX_ROUNDS):
        selected = []
        resolutions = []
        for index in sorted(states):
            for bins in states[index]["next"]:
                selected.append(index)
                resolutions.append(bins)
        if not selected:
            break
        selected = np.asarray(selected, dtype=int)
        measured = _evaluate_variable_resolutions(
            curve, times, params, selected, method, resolutions
        )
        values = np.asarray(measured["magnification"], dtype=float)
        support = np.asarray(measured["converged"], dtype=bool)
        for offset, index_value in enumerate(selected):
            index = int(index_value)
            bins = int(resolutions[offset])
            states[index]["samples"][bins] = {
                "value": float(values[offset]),
                "support": bool(support[offset]),
            }
        for index in sorted(states):
            if required[index] is not None:
                continue
            stable = _stable_tail(
                states[index]["samples"], atol, reltol
            )
            if stable is None:
                states[index]["confirming"] = False
                states[index]["next"] = _next_candidate_batch(
                    states[index]["samples"], int(cap)
                )
                continue
            if states[index]["confirming"] or not states[index]["next"]:
                # A stable tail at the hard cap is sufficient even when there
                # is no higher batch available for a separate confirmation.
                states[index]["next"] = ()
            else:
                # Observe one more refinement batch before accepting this
                # tail.  Only the highest candidate is needed for this
                # confirmation; evaluating the entire next batch would add
                # cost without strengthening the plateau check.  This is what
                # distinguishes a genuine plateau from the low-resolution auto
                # value that motivated warm-up.
                states[index]["confirming"] = True
                next_batch = _next_candidate_batch(
                    states[index]["samples"], int(cap)
                )
                states[index]["next"] = next_batch[-1:] if next_batch else ()

        # A state whose tail is stable and has no next batch is ready to be
        # materialized below.  Leave other states in the search loop.
        if all(
            not state["next"]
            for state in states.values()
        ):
            break

    for index, state in states.items():
        stable = _stable_tail(state["samples"], atol, reltol)
        if stable is None:
            continue
        _, reference, budget = stable
        references[index] = reference
        candidates = []
        for bins, observation in state["samples"].items():
            if not observation["support"] or not np.isfinite(observation["value"]):
                continue
            candidate_budget = _self_convergence_budget(
                (reference, observation["value"]), atol, reltol
            )
            if abs(observation["value"] - reference) <= candidate_budget:
                candidates.append((int(bins), float(observation["value"])))
        if candidates:
            bins, value = min(candidates)
            required[index] = bins
            selected_values[index] = value
    return required, references, selected_values


def _time_grid_candidates(
    curve, times, params, method, required, repeats
):
    indices = np.asarray(
        sorted(index for index, bins in required.items() if bins is not None),
        dtype=int,
    )
    if not indices.size:
        return {}
    resolutions = [int(required[int(index)]) for index in indices]
    measurements = []
    for _ in range(repeats):
        result = curve._native._evaluate_preplanned(
            np.asarray(times)[indices],
            params,
            [int(method)] * indices.size,
            resolutions,
        )
        measurements.append(np.asarray(result["seconds"], dtype=float))
    median = np.median(np.asarray(measurements), axis=0)
    return {
        int(index): float(median[offset])
        for offset, index in enumerate(indices)
    }


def _choose_grid(cartesian_bins, polar_bins, cartesian_seconds, polar_seconds):
    if cartesian_bins is None and polar_bins is None:
        return None, None
    choose_polar = cartesian_bins is None or (
        polar_bins is not None and polar_seconds < cartesian_seconds
    )
    if choose_polar:
        return POLAR, polar_bins
    return CARTESIAN, cartesian_bins


def build_warmup_report(
    curve,
    times,
    params,
    *,
    parameter_fingerprint,
    configuration_fingerprint,
    grid_timing_repeats=1,
):
    started = time.perf_counter()
    times = np.asarray(times, dtype=float)
    if times.ndim == 0:
        times = times.reshape(1)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("warm-up times must be a non-empty finite 1-D array")
    cap = int(curve.options.max_source_bins)
    if cap < 1:
        raise ValueError("max_source_bins must be positive")
    grid_timing_repeats = int(grid_timing_repeats)
    if grid_timing_repeats < 1:
        raise ValueError("grid_timing_repeats must be at least one")
    if curve._native.model.lens != "binary":
        raise NotImplementedError("the first warm-up implementation supports binary lenses")
    if curve._native.model.source != "single":
        raise NotImplementedError("warm-up currently supports single-source curves")
    count = times.size

    # The ordinary dispatcher supplies the route split and the cheap point/
    # hexadecapole decisions.  It is deliberately not used as the numerical
    # oracle for inverse-ray rows: the automatic resolution law can itself be
    # the thing that needs correction.
    # A JAX warm-up uses this routine only as the native calibration oracle.
    # Calling the public JAX ``info`` here would calibrate the dispatcher being
    # replaced and would not reproduce the native plan contract.
    baseline = (
        curve._native.info(times, params)
        if curve.options.jax
        else curve.info(times, params)
    )
    baseline_values = np.asarray(
        baseline.finite_source_magnifications, dtype=float
    )
    baseline_methods = np.asarray(baseline.finite_source_methods, dtype=int)
    baseline_refinement = np.asarray(
        baseline.finite_source_refinement_levels, dtype=int
    )
    baseline_support = np.asarray(
        baseline.finite_source_converged, dtype=bool
    )

    reference = baseline_values.copy()
    grid_mask = np.isin(baseline_methods, (CARTESIAN, POLAR))
    scale = np.maximum(np.abs(reference), 1.0)
    atol = max(float(curve.options.finite_source_tol), 0.0)
    reltol = max(float(curve.options.finite_source_reltol), 0.0)
    if atol <= 0.0 and reltol <= 0.0:
        atol = 1.0e-4
        reltol = 1.0e-3
    budget = np.maximum(atol, reltol * scale)
    reference_usable = (
        grid_mask
        & baseline_support
        & np.isfinite(reference)
    )

    methods = np.full(count, POINT_SOURCE, dtype=int)
    resolutions = np.full(count, -1, dtype=int)
    statuses = np.full(count, "reference_limited", dtype=object)
    cartesian_seconds = np.full(count, np.nan)
    polar_seconds = np.full(count, np.nan)

    # Point and hexadecapole are already self-checked by ordinary auto.  The
    # source-plane route already reports whether its 96/192-panel refinement
    # was required, so retain that decision without dragging it through an
    # unrelated Cartesian/polar reference campaign.
    simple_mask = np.isin(baseline_methods, (POINT_SOURCE, HEXADECAPOLE))
    for index in np.flatnonzero(simple_mask & baseline_support & np.isfinite(baseline_values)):
        method = int(baseline_methods[index])
        methods[index] = method
        resolutions[index] = 0
        statuses[index] = "calibrated"

    quadrature_rows = np.flatnonzero(
        (baseline_methods == SOURCE_PLANE)
        & baseline_support
        & np.isfinite(baseline_values)
    )
    for index in quadrature_rows:
        methods[index] = SOURCE_PLANE
        resolutions[index] = 192 if baseline_refinement[index] > 0 else 96
        statuses[index] = "calibrated"

    remaining = np.flatnonzero(
        reference_usable
        & (resolutions < 0)
        & np.isin(baseline_methods, (CARTESIAN, POLAR))
    )
    if remaining.size:
        point = _native_evaluate(
            curve, times[remaining], params, POINT_SOURCE, 0
        )
        point_values = np.asarray(point["magnification"], dtype=float)
        point_by_index = {
            int(index): float(point_values[offset])
            for offset, index in enumerate(remaining)
        }
        predicted_cartesian = {
            int(index): _predicted_resolution(
                curve,
                CARTESIAN,
                point_by_index[int(index)],
                atol,
                reltol,
                cap,
            )
            for index in remaining
        }
        predicted_polar = {
            int(index): _predicted_resolution(
                curve,
                POLAR,
                point_by_index[int(index)],
                atol,
                reltol,
                cap,
            )
            for index in remaining
        }
        (
            required_cartesian,
            cartesian_references,
            cartesian_values,
        ) = _self_converged_resolutions(
            curve,
            times,
            params,
            remaining,
            CARTESIAN,
            predicted_cartesian,
            atol,
            reltol,
            cap,
        )
        (
            required_polar,
            polar_references,
            polar_values,
        ) = _self_converged_resolutions(
            curve,
            times,
            params,
            remaining,
            POLAR,
            predicted_polar,
            atol,
            reltol,
            cap,
        )
        cartesian_timing = _time_grid_candidates(
            curve,
            times,
            params,
            CARTESIAN,
            required_cartesian,
            grid_timing_repeats,
        )
        polar_timing = _time_grid_candidates(
            curve,
            times,
            params,
            POLAR,
            required_polar,
            grid_timing_repeats,
        )
        for index in remaining:
            index = int(index)
            cartesian_bins = required_cartesian[index]
            polar_bins = required_polar[index]
            if index in cartesian_timing:
                cartesian_seconds[index] = cartesian_timing[index]
            if index in polar_timing:
                polar_seconds[index] = polar_timing[index]

            available_references = [
                value
                for value in (
                    cartesian_references[index],
                    polar_references[index],
                )
                if value is not None and np.isfinite(value)
            ]
            if not available_references:
                statuses[index] = "max_source_bins_limited"
                continue

            common_reference = float(np.median(available_references))
            common_budget = _self_convergence_budget(
                available_references, atol, reltol
            )
            if len(available_references) == 2 and (
                abs(available_references[0] - available_references[1])
                > common_budget
            ):
                statuses[index] = "reference_disagreement"
                continue

            # A route may have self-converged while its selected resolution is
            # not inside the common Cartesian/polar reference budget.  Do not
            # install that route merely because its own tail was stable.
            if cartesian_bins is not None:
                cartesian_value = cartesian_values[index]
                if cartesian_value is None or abs(
                    cartesian_value - common_reference
                ) > common_budget:
                    cartesian_bins = None
            if polar_bins is not None:
                polar_value = polar_values[index]
                if polar_value is None or abs(
                    polar_value - common_reference
                ) > common_budget:
                    polar_bins = None
            if cartesian_bins is None and polar_bins is None:
                statuses[index] = "reference_disagreement"
                continue

            reference[index] = common_reference
            budget[index] = common_budget
            method, bins = _choose_grid(
                cartesian_bins,
                polar_bins,
                cartesian_seconds[index],
                polar_seconds[index],
            )
            if method is None:
                statuses[index] = "max_source_bins_limited"
                continue
            methods[index] = method
            resolutions[index] = bins
            statuses[index] = "calibrated"

    method_names = tuple(
        METHOD_NAMES[method] if resolution >= 0 else "auto_fallback"
        for method, resolution in zip(methods, resolutions)
    )
    geometry = build_warmup_geometry(curve, times, params, baseline)
    return WarmupReport(
        times=times.copy(),
        methods=method_names,
        resolutions=resolutions,
        statuses=tuple(str(value) for value in statuses),
        reference=reference.copy(),
        budget=budget.copy(),
        cartesian_seconds=cartesian_seconds,
        polar_seconds=polar_seconds,
        elapsed_seconds=time.perf_counter() - started,
        parameter_fingerprint=parameter_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        geometry=geometry,
    ), methods

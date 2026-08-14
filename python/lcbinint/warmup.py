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
    """Concrete XLA compilation/execution completed by ``LightCurve.warmup``."""

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


def build_jax_warmup_report(
    curve,
    times,
    params,
    *,
    parameter_fingerprint,
    configuration_fingerprint,
):
    """Compile and synchronously execute the actual public JAX path once."""

    from .jax_backend import magnification

    concrete_times = np.asarray(times, dtype=float)
    if concrete_times.ndim == 0:
        concrete_times = concrete_times.reshape(1)
    start = time.perf_counter()
    values = magnification(
        curve._native,
        curve._options,
        concrete_times,
        params,
    )
    if hasattr(values, "block_until_ready"):
        values.block_until_ready()
    elapsed = time.perf_counter() - start
    reference = np.asarray(values, dtype=float).copy()
    diagnostics = curve._native.info(concrete_times, params)
    geometry = build_warmup_geometry(
        curve, concrete_times, params, diagnostics
    )
    count = concrete_times.size
    return JaxWarmupReport(
        times=concrete_times.copy(),
        methods=("jax_compiled",) * count,
        resolutions=np.zeros(count, dtype=np.int64),
        statuses=("compiled",) * count,
        reference=reference,
        budget=np.zeros(count, dtype=float),
        cartesian_seconds=np.zeros(count, dtype=float),
        polar_seconds=np.zeros(count, dtype=float),
        elapsed_seconds=elapsed,
        parameter_fingerprint=parameter_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        geometry=geometry,
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


def _required_resolutions(
    curve,
    times,
    params,
    indices,
    method,
    reference,
    budget,
    predicted,
    cap,
):
    """Search from the empirical hint until three increasing grids pass."""

    states = {
        int(index): {
            "samples": {},
            "next": _candidate_batch(int(predicted[int(index)]), int(cap)),
        }
        for index in indices
    }
    required = {int(index): None for index in indices}

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
        passes = _inside(
            values, support, reference[selected], budget[selected]
        )
        for offset, index_value in enumerate(selected):
            index = int(index_value)
            bins = int(resolutions[offset])
            states[index]["samples"][bins] = {
                "error": float(abs(values[offset] - reference[index])),
                "pass": bool(passes[offset]),
            }
        for index in sorted(states):
            if required[index] is not None or not states[index]["next"]:
                continue
            run = _persistent_pass_run(states[index]["samples"])
            if run is not None:
                required[index] = int(run[0])
                states[index]["next"] = ()
            else:
                states[index]["next"] = _next_candidate_batch(
                    states[index]["samples"], int(cap)
                )

    # Interpolate the measured fail/pass bracket, round upward, and verify the
    # resulting integer.  The two already-passing grids above it provide the
    # persistence confirmation.
    candidate_by_index = {}
    for index, upper in required.items():
        if upper is None:
            continue
        samples = states[index]["samples"]
        lower_failures = [
            bins for bins in samples
            if bins < upper and not samples[bins]["pass"]
        ]
        if not lower_failures:
            continue
        lower = max(lower_failures)
        candidate = _interpolated_resolution(
            lower,
            samples[lower]["error"],
            upper,
            samples[upper]["error"],
            budget[index],
        )
        if lower < candidate < upper:
            candidate_by_index[index] = candidate
    if candidate_by_index:
        selected = np.asarray(sorted(candidate_by_index), dtype=int)
        resolutions = [candidate_by_index[int(index)] for index in selected]
        measured = _evaluate_variable_resolutions(
            curve, times, params, selected, method, resolutions
        )
        values = np.asarray(measured["magnification"], dtype=float)
        support = np.asarray(measured["converged"], dtype=bool)
        passes = _inside(
            values, support, reference[selected], budget[selected]
        )
        for offset, index_value in enumerate(selected):
            if passes[offset]:
                required[int(index_value)] = int(resolutions[offset])
    return required


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
    if curve.options.jax:
        raise NotImplementedError("warm-up execution plans currently require the native backend")

    count = times.size

    # The ordinary dispatcher supplies both A_ref and the cheap first split.
    # Only rows that already entered an inverse-ray route pay for calibration.
    baseline = curve.info(times, params)
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
    grid_indices = np.flatnonzero(grid_mask)

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
        required_cartesian = _required_resolutions(
            curve,
            times,
            params,
            remaining,
            CARTESIAN,
            reference,
            budget,
            predicted_cartesian,
            cap,
        )
        required_polar = _required_resolutions(
            curve,
            times,
            params,
            remaining,
            POLAR,
            reference,
            budget,
            predicted_polar,
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

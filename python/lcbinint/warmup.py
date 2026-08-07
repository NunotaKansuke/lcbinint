"""Reference-certified, per-epoch execution plans for :class:`LightCurve`."""

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

DEFAULT_LADDER = (
    4, 6, 8, 10, 12, 16, 24, 32, 40, 50,
    64, 80, 100, 128, 160, 200, 256, 320, 400,
)
# The campaign also offered 1e-10 as an escalation level.  It is not paid by
# default here: 1e-6/1e-8 already demonstrates the 1e-5 reference floor needed
# by the supported runtime domain, and a row that does not settle is honestly
# reference-limited.  Callers may explicitly append 1e-10.
DEFAULT_CONTOUR_LEVELS = (1.0e-4, 1.0e-6, 1.0e-8)
DEFAULT_QUADRATURE_LADDER = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
REFERENCE_MARGIN = 0.1


@dataclass
class WarmupReport:
    """Diagnostic view of the plan retained by a ``LightCurve`` instance."""

    times: np.ndarray
    methods: tuple[str, ...]
    resolutions: np.ndarray
    statuses: tuple[str, ...]
    reference: np.ndarray
    reference_uncertainty: np.ndarray
    budget: np.ndarray
    cartesian_seconds: np.ndarray
    polar_seconds: np.ndarray
    elapsed_seconds: float
    parameter_fingerprint: tuple
    configuration_fingerprint: tuple

    @property
    def calibrated(self):
        return np.asarray([status == "calibrated" for status in self.statuses])

    @property
    def all_calibrated(self):
        return bool(np.all(self.calibrated))


def _native_evaluate(curve, times, params, method, resolution):
    count = int(np.asarray(times).size)
    return curve._native._evaluate_preplanned(
        np.asarray(times, dtype=float),
        params,
        [int(method)] * count,
        [int(resolution)] * count,
    )


def _contour_witness(geometry, levels):
    try:
        import VBMicrolensing
    except ImportError as error:
        raise ImportError(
            "LightCurve.warmup() requires VBMicrolensing to certify A_ref"
        ) from error

    n = len(geometry.separation)
    values = np.full((len(levels), n), np.nan, dtype=float)
    for row, reltol in enumerate(levels):
        engine = VBMicrolensing.VBMicrolensing()
        engine.Tol = 1.0e-12
        engine.RelTol = float(reltol)
        for index in range(n):
            c = float(geometry.limb_darkening_c[index])
            d = float(geometry.limb_darkening_d[index])
            if d != 0.0:
                raise NotImplementedError(
                    "VBMicrolensing warm-up reference currently supports "
                    "uniform and linear limb darkening only"
                )
            try:
                if c != 0.0:
                    engine.a1 = c
                    engine.SetLDprofile(engine.LDlinear)
                    value = engine.BinaryMagDark(
                        float(geometry.separation[index]),
                        float(geometry.mass_ratio[index]),
                        -float(geometry.source_x[index]),
                        float(geometry.source_y[index]),
                        float(geometry.source_radius[index]),
                        engine.Tol,
                    )
                else:
                    value = engine.BinaryMag2(
                        float(geometry.separation[index]),
                        float(geometry.mass_ratio[index]),
                        -float(geometry.source_x[index]),
                        float(geometry.source_y[index]),
                        float(geometry.source_radius[index]),
                    )
                values[row, index] = float(value)
            except Exception:
                # The polar witness may still certify this row.  A failed VBM
                # call is represented as an absent witness, not a global abort.
                values[row, index] = np.nan
    witness = values[-1]
    if len(levels) < 2:
        self_gap = np.full(n, np.inf)
    else:
        self_gap = np.abs(values[-1] - values[-2]) / np.maximum(
            np.abs(values[-1]), 1.0
        )
    return witness, self_gap, values


def _reference(curve, times, params, geometry, contour_levels):
    cart256 = _native_evaluate(curve, times, params, CARTESIAN, 256)
    cart400 = _native_evaluate(curve, times, params, CARTESIAN, 400)
    polar256 = _native_evaluate(curve, times, params, POLAR, 256)
    polar400 = _native_evaluate(curve, times, params, POLAR, 400)

    reference = np.asarray(cart400["magnification"], dtype=float)
    scale = np.maximum(np.abs(reference), 1.0)
    cart_support = np.asarray(cart400["converged"], dtype=bool)
    cart_previous_support = np.asarray(cart256["converged"], dtype=bool)
    polar_support = np.asarray(polar400["converged"], dtype=bool)
    ladder_gap = np.abs(
        reference - np.asarray(cart256["magnification"], dtype=float)
    ) / scale
    ladder_gap[~cart_previous_support] = np.inf
    polar_gap = np.abs(
        reference - np.asarray(polar400["magnification"], dtype=float)
    ) / scale
    polar_gap[~polar_support] = np.inf

    contour, contour_self_gap, contour_values = _contour_witness(
        geometry, contour_levels
    )
    contour_gap = np.maximum(np.abs(reference - contour) / scale, contour_self_gap)
    contour_gap[~np.isfinite(contour)] = np.inf
    best_witness = np.minimum(polar_gap, contour_gap)
    uncertainty = np.maximum(ladder_gap, best_witness)
    uncertainty[~cart_support | ~np.isfinite(reference)] = np.inf
    return {
        "value": reference,
        "uncertainty": uncertainty,
        "cartesian": {256: cart256, 400: cart400},
        "polar": {256: polar256, 400: polar400},
        "polar_gap": polar_gap,
        "contour_gap": contour_gap,
        "contour_values": contour_values,
    }


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


def _required_resolutions(
    curve, times, params, indices, method, reference, budget, ladder, stored
):
    required = {int(index): None for index in indices}
    active = set(int(index) for index in indices)
    last_pass = {}
    last_pass_error = {}
    first_fail = {}
    first_fail_error = {}

    for bins in reversed(ladder):
        if not active:
            break
        selected = np.asarray(sorted(active), dtype=int)
        if bins in stored:
            full = stored[bins]
            values = np.asarray(full["magnification"], dtype=float)[selected]
            support = np.asarray(full["converged"], dtype=bool)[selected]
        else:
            measured = _native_evaluate(
                curve, np.asarray(times)[selected], params, method, bins
            )
            values = np.asarray(measured["magnification"], dtype=float)
            support = np.asarray(measured["converged"], dtype=bool)
        errors = np.abs(values - reference[selected])
        passes = _inside(values, support, reference[selected], budget[selected])
        for offset, index in enumerate(selected):
            index = int(index)
            if passes[offset]:
                last_pass[index] = bins
                last_pass_error[index] = float(errors[offset])
                continue
            first_fail[index] = bins
            first_fail_error[index] = float(errors[offset])
            required[index] = last_pass.get(index)
            active.remove(index)

    for index in active:
        required[index] = last_pass.get(index)

    # Convert the measured bracket into any positive integer, round upward,
    # then actually evaluate that integer before accepting it.
    candidates = {}
    for index, upper in required.items():
        if upper is None or index not in first_fail:
            continue
        candidate = _interpolated_resolution(
            first_fail[index],
            first_fail_error[index],
            upper,
            last_pass_error[index],
            budget[index],
        )
        if candidate < upper:
            candidates.setdefault(candidate, []).append(index)
    for candidate, candidate_indices in candidates.items():
        selected = np.asarray(candidate_indices, dtype=int)
        measured = _native_evaluate(
            curve, np.asarray(times)[selected], params, method, candidate
        )
        values = np.asarray(measured["magnification"], dtype=float)
        support = np.asarray(measured["converged"], dtype=bool)
        passes = _inside(values, support, reference[selected], budget[selected])
        for offset, index in enumerate(selected):
            if passes[offset]:
                required[int(index)] = candidate
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
    ladder=DEFAULT_LADDER,
    contour_levels=DEFAULT_CONTOUR_LEVELS,
    grid_timing_repeats=3,
):
    started = time.perf_counter()
    times = np.asarray(times, dtype=float)
    if times.ndim == 0:
        times = times.reshape(1)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("warm-up times must be a non-empty finite 1-D array")
    ladder = tuple(sorted({int(value) for value in ladder if int(value) > 0}))
    if not ladder or 256 not in ladder or 400 not in ladder:
        raise ValueError("warm-up ladder must contain the 256 and 400 reference rungs")
    contour_levels = tuple(float(value) for value in contour_levels)
    if len(contour_levels) < 2 or any(value <= 0.0 for value in contour_levels):
        raise ValueError("contour_levels must contain at least two positive values")
    grid_timing_repeats = int(grid_timing_repeats)
    if grid_timing_repeats < 1:
        raise ValueError("grid_timing_repeats must be at least one")
    if curve.lens != "binary":
        raise NotImplementedError("the first warm-up implementation supports binary lenses")
    if curve.options.jax:
        raise NotImplementedError("warm-up execution plans currently require the native backend")
    if curve.options.param_type != "vbm":
        raise NotImplementedError(
            "warm-up VBM cross-reference currently requires coordinates='vbm'"
        )

    geometry = curve.finite_source_geometry(times, params)
    reference_data = _reference(
        curve, times, params, geometry, contour_levels
    )
    reference = reference_data["value"]
    uncertainty = reference_data["uncertainty"]
    scale = np.maximum(np.abs(reference), 1.0)
    atol = max(float(curve.options.finite_source_tol), 0.0)
    reltol = max(float(curve.options.finite_source_reltol), 0.0)
    if atol <= 0.0 and reltol <= 0.0:
        atol = 1.0e-3
        reltol = 1.0e-3
    budget = np.maximum(atol, reltol * scale)
    reference_usable = (
        np.isfinite(reference)
        & np.isfinite(uncertainty)
        & (uncertainty <= REFERENCE_MARGIN * budget / scale)
    )

    count = times.size
    methods = np.full(count, POINT_SOURCE, dtype=int)
    resolutions = np.full(count, -1, dtype=int)
    statuses = np.full(count, "reference_limited", dtype=object)
    cartesian_seconds = np.full(count, np.nan)
    polar_seconds = np.full(count, np.nan)

    # Run ordinary auto once.  Its accepted point/hex/quadrature decisions are
    # retained only when the independently witnessed A_ref confirms them.
    try:
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
    except Exception:
        baseline_values = np.full(count, np.nan)
        baseline_methods = np.full(count, -1, dtype=int)
        baseline_refinement = np.zeros(count, dtype=int)
        baseline_support = np.zeros(count, dtype=bool)

    baseline_pass = _inside(
        baseline_values, baseline_support, reference, budget
    )
    for index in np.flatnonzero(reference_usable & baseline_pass):
        method = int(baseline_methods[index])
        if method in (POINT_SOURCE, HEXADECAPOLE):
            methods[index] = method
            resolutions[index] = 0
            statuses[index] = "calibrated"

    remaining = np.flatnonzero(reference_usable & (resolutions < 0))
    if remaining.size:
        point = _native_evaluate(curve, times[remaining], params, POINT_SOURCE, 0)
        point_values = np.asarray(point["magnification"], dtype=float)
        point_pass = _inside(
            point_values,
            point["converged"],
            reference[remaining],
            budget[remaining],
        )
        for offset, index in enumerate(remaining):
            if point_pass[offset]:
                methods[index] = POINT_SOURCE
                resolutions[index] = 0
                statuses[index] = "calibrated"

    remaining = np.flatnonzero(reference_usable & (resolutions < 0))
    if remaining.size:
        hex_result = _native_evaluate(
            curve, times[remaining], params, HEXADECAPOLE, 0
        )
        hex_values = np.asarray(hex_result["magnification"], dtype=float)
        hex_pass = _inside(
            hex_values,
            hex_result["converged"],
            reference[remaining],
            budget[remaining],
        )
        for offset, index in enumerate(remaining):
            if hex_pass[offset]:
                methods[index] = HEXADECAPOLE
                resolutions[index] = 0
                statuses[index] = "calibrated"

    remaining = np.flatnonzero(reference_usable & (resolutions < 0))
    quadrature_rows = remaining[baseline_methods[remaining] == SOURCE_PLANE]
    for top in (96, 192):
        selected = quadrature_rows[
            np.where(
                np.where(baseline_refinement[quadrature_rows] > 0, 192, 96)
                == top
            )[0]
        ]
        if not selected.size:
            continue
        panel_ladder = tuple(
            value for value in DEFAULT_QUADRATURE_LADDER if value <= top
        )
        required = _required_resolutions(
            curve,
            times,
            params,
            selected,
            SOURCE_PLANE,
            reference,
            budget,
            panel_ladder,
            {},
        )
        for index in selected:
            panels = required[int(index)]
            if panels is None:
                continue
            methods[index] = SOURCE_PLANE
            resolutions[index] = panels
            statuses[index] = "calibrated"

    remaining = np.flatnonzero(reference_usable & (resolutions < 0))
    if remaining.size:
        required_cartesian = _required_resolutions(
            curve,
            times,
            params,
            remaining,
            CARTESIAN,
            reference,
            budget,
            ladder,
            reference_data["cartesian"],
        )
        required_polar = _required_resolutions(
            curve,
            times,
            params,
            remaining,
            POLAR,
            reference,
            budget,
            ladder,
            reference_data["polar"],
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
                statuses[index] = "ladder_limited"
                continue
            methods[index] = method
            resolutions[index] = bins
            statuses[index] = "calibrated"

    method_names = tuple(
        METHOD_NAMES[method] if resolution >= 0 else "auto_fallback"
        for method, resolution in zip(methods, resolutions)
    )
    return WarmupReport(
        times=times.copy(),
        methods=method_names,
        resolutions=resolutions,
        statuses=tuple(str(value) for value in statuses),
        reference=reference.copy(),
        reference_uncertainty=uncertainty.copy(),
        budget=budget.copy(),
        cartesian_seconds=cartesian_seconds,
        polar_seconds=polar_seconds,
        elapsed_seconds=time.perf_counter() - started,
        parameter_fingerprint=parameter_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
    ), methods

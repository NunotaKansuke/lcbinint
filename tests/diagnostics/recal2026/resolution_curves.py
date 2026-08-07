#!/usr/bin/env python3
"""Export resolution curves and fit a common empirical resolution rule.

The resolution sweep already contains the information needed for the paper's
``A(N)`` plots: the magnification at every measured grid size, its wall time,
the high-resolution reference, and the geometry at that source position.  This
module turns that nested table into two deliberately simple artefacts:

* ``resolution_curves.csv`` keeps the measured magnification ``A(N)`` and the
  adjacent changes for Cartesian and polar grids;
* ``resolution_requirements.csv`` records the first *persistent* resolution
  that stays within each tolerance, using the reference definition from
  :mod:`reference`;
* ``feature_fits.json`` compares quantile rules using magnification, ``d/rho``
  and other cheap geometric features, scored on an independent holdout.

The fit is an offline calibration.  It does not change the runtime selector.
In particular, an embedded runtime error estimate is not used as a substitute
for the measured reference curve.

Example::

    python -m tests.diagnostics.recal2026.resolution_curves \
      --discovery tests/diagnostics/results/recal2026/discovery \
      --holdout tests/diagnostics/results/recal2026/holdout \
      --output tests/diagnostics/results/recal2026/resolution_curves
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import tarfile
from pathlib import Path

import numpy as np

from . import analysis, fit_rules
from .engines import BUCKETS
from .sweep_resolution import TARGET_TOLERANCES


GRIDS = ("cartesian", "polar")
FEATURE_NAMES = fit_rules.FEATURE_NAMES
FEATURE_SETS = {
    "A_point": ("log_point",),
    "d_over_rho": ("log_ratio", "proximity"),
    "rho": ("log_rho",),
    "A_point+d_over_rho": ("log_point", "log_ratio", "proximity"),
    "A_point+d_over_rho+rho": (
        "log_point", "log_ratio", "proximity", "log_rho",
    ),
    "all_features": FEATURE_NAMES,
}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _sorted_measurements(ladder):
    """Return measured numeric ladder entries in ascending bucket order."""
    measurements = []
    for bucket in BUCKETS:
        entry = ladder.get(bucket)
        if entry is None:
            entry = ladder.get(str(bucket))
        if not isinstance(entry, dict):
            continue
        value = entry.get("magnification")
        if not _finite(value):
            continue
        measurements.append((bucket, entry))
    return measurements


def _row_geometry(row):
    distance = _as_float(row.get("caustic_distance"))
    rho = _as_float(row.get("rho"))
    ratio = distance / rho if _finite(distance) and _finite(rho) and rho > 0 else float("nan")
    return {
        "case_id": row.get("case_id"),
        "s": _as_float(row.get("s")),
        "q": _as_float(row.get("q")),
        "rho": rho,
        "x": _as_float(row.get("x")),
        "y": _as_float(row.get("y")),
        "profile": row.get("profile", ""),
        "limb_darkening_c": _as_float(row.get("limb_darkening_c")),
        "intended_distance_factor": _as_float(
            row.get("intended_distance_factor")),
        "point_magnification": _as_float(row.get("point_magnification")),
        "caustic_distance": distance,
        "d_over_rho": ratio,
    }


def _curve_records(rows, dataset_name):
    records = []
    requirements = []
    for row_index, row in enumerate(rows):
        geometry = _row_geometry(row)
        reference = row.get("reference") or {}
        reference_value = _as_float(reference.get("value"))
        reference_uncertainty = _as_float(reference.get("uncertainty"))
        for grid in GRIDS:
            measurements = _sorted_measurements(row.get(grid) or {})
            previous = None
            for bucket, entry in measurements:
                value = _as_float(entry.get("magnification"))
                delta = abs(value - previous[1]) if previous is not None else float("nan")
                record = {
                    "dataset": dataset_name,
                    "algorithm_vintage": "20260807_current",
                    "row_index": row_index,
                    "grid": grid,
                    "resolution": bucket,
                    "magnification": value,
                    "seconds": _as_float(entry.get("seconds")),
                    "method": entry.get("method", ""),
                    "support_proven": bool(entry.get("support_proven", False)),
                    "converged": bool(entry.get("converged", False)),
                    "reference": reference_value,
                    "reference_uncertainty": reference_uncertainty,
                    "abs_error_to_reference": (
                        abs(value - reference_value)
                        if _finite(value) and _finite(reference_value)
                        else float("nan")
                    ),
                    "relative_error_to_reference": (
                        abs(value - reference_value) /
                        max(abs(reference_value), 1.0)
                        if _finite(value) and _finite(reference_value)
                        else float("nan")
                    ),
                    "delta_from_previous": delta,
                    "relative_delta_from_previous": (
                        delta / max(abs(value), 1.0)
                        if _finite(delta) else float("nan")
                    ),
                    **geometry,
                }
                records.append(record)
                previous = (bucket, value)

            for tolerance in TARGET_TOLERANCES:
                key = str(tolerance)
                target_entry = (row.get("required") or {}).get(key) or {}
                required = target_entry.get(grid)
                budget = (
                    tolerance * max(abs(reference_value), 1.0)
                    if _finite(reference_value) else float("nan")
                )
                requirements.append({
                    "dataset": dataset_name,
                    "algorithm_vintage": "20260807_current",
                    "row_index": row_index,
                    "grid": grid,
                    "relative_tolerance": tolerance,
                    "usable_reference": bool(target_entry.get("usable", False)),
                    "required_resolution": required,
                    "budget": budget,
                    "reference": reference_value,
                    "reference_uncertainty": reference_uncertainty,
                    **geometry,
                })
    return records, requirements


def _legacy_feature_values(point):
    """Return the current feature names from the old raw-row vocabulary."""
    point_magnification = _as_float(point.get("point_magnification"))
    rho = _as_float(point.get("rho"))
    q = _as_float(point.get("mass_ratio"))
    ratio = _as_float(point.get("caustic_distance_over_rho"))
    q_small = min(abs(q), 1.0 / abs(q)) if _finite(q) and q != 0.0 else float("nan")
    return {
        "log_point": math.log10(max(point_magnification, 1.0)),
        "log_rho": math.log10(max(rho, 1.0e-12)),
        "log_q_small": math.log10(max(q_small, 1.0e-12)),
        "log_ratio": math.log10(max(min(ratio, 1.0e6), 1.0e-3)),
        "proximity": max(0.0, 2.0 - min(ratio, 2.0)),
        "log_swallow": max(0.0, math.log10(
            max(4.0 * rho / max(q_small, 1.0e-12), 1.0))),
        "limb_darkening_c": _as_float(point.get("limb_c")),
    }


def _legacy_curve_records(root):
    """Read the previous large campaign without treating it as current truth.

    The compact CSV has the old reference and required bins.  Its raw tarballs
    retain every measured fixed-grid value, so the old campaign can contribute
    full ``A(N)`` curves as a separately labelled historical dataset.  The old
    and current algorithms are never merged into one holdout fit.
    """
    root = Path(root)
    compact = root / "source-profile-results.csv.gz"
    if not compact.exists():
        return [], [], {"error": f"missing {compact}"}

    references = {}
    with gzip.open(compact, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (
                    row["dataset"], int(row["case_id"]),
                    int(row["point_id"]), round(float(row["limb_c"]), 8),
                )
                references[key] = row
            except (KeyError, TypeError, ValueError):
                continue

    curves = []
    requirements = []
    archives = {
        "discovery": root / "raw" / "discovery.tar.gz",
        "independent_validation": root / "raw" / "independent-validation.tar.gz",
    }
    archive_report = {}
    for dataset_name, archive in archives.items():
        if not archive.exists():
            archive_report[dataset_name] = {"status": "missing"}
            continue
        seen_rows = 0
        matched_rows = 0
        with tarfile.open(archive, "r:gz") as tar:
            members = sorted(
                (
                    member for member in tar.getmembers()
                    if member.isfile() and member.name.endswith(".json")
                ),
                key=lambda member: member.name,
            )
            for member in members:
                payload = json.load(tar.extractfile(member))
                case = payload.get("case") or {}
                if case.get("case_id") is None:
                    continue
                case_id = int(case.get("case_id"))
                separation = _as_float(case.get("separation"))
                mass_ratio = _as_float(case.get("mass_ratio"))
                rho = _as_float(case.get("source_radius"))
                for point in payload.get("rows", []):
                    seen_rows += 1
                    # The compact finalizer preserves the raw point id.  Some
                    # raw point zero rows were intentionally excluded from the
                    # compact reference table, so do not shift the key.
                    key = (
                        dataset_name, case_id, int(point.get("point_id", -1)),
                        round(_as_float(point.get("limb_c")), 8),
                    )
                    reference = references.get(key)
                    if reference is None:
                        continue
                    matched_rows += 1
                    reference_value = _as_float(reference.get("reference"))
                    point = dict(point)
                    point.update({
                        "separation": separation,
                        "mass_ratio": mass_ratio,
                        "rho": rho,
                    })
                    features = _legacy_feature_values(point)
                    geometry = {
                        "case_id": case_id,
                        "s": separation,
                        "q": mass_ratio,
                        "rho": rho,
                        "x": _as_float(point.get("source_x")),
                        "y": _as_float(point.get("source_y")),
                        "profile": "uniform" if _as_float(point.get("limb_c")) == 0.0
                        else "linear",
                        "limb_darkening_c": _as_float(point.get("limb_c")),
                        "intended_distance_factor": _as_float(
                            point.get("requested_distance_factor")),
                        "point_magnification": _as_float(
                            point.get("point_magnification")),
                        "caustic_distance": _as_float(
                            point.get("caustic_distance")),
                        "d_over_rho": _as_float(
                            point.get("caustic_distance_over_rho")),
                    }
                    for grid in GRIDS:
                        previous = None
                        sequence = (point.get("lc_fixed_sequences") or {}).get(grid, [])
                        for entry in sequence:
                            value = _as_float(entry.get("value"))
                            if not _finite(value):
                                continue
                            bucket = int(entry.get("bins"))
                            delta = abs(value - previous[1]) if previous else float("nan")
                            curves.append({
                                "dataset": f"legacy_{dataset_name}",
                                "algorithm_vintage": "20260716_pre_certificate",
                                "row_index": seen_rows - 1,
                                "grid": grid,
                                "resolution": bucket,
                                "magnification": value,
                                "seconds": _as_float(entry.get("elapsed_ns")) / 1.0e9,
                                "method": entry.get("method", ""),
                                "support_proven": bool(entry.get("reported_converged", False)),
                                "converged": bool(entry.get("reported_converged", False)),
                                "reference": reference_value,
                                "reference_uncertainty": float("nan"),
                                "abs_error_to_reference": (
                                    abs(value - reference_value)
                                    if _finite(reference_value) else float("nan")
                                ),
                                "relative_error_to_reference": (
                                    abs(value - reference_value) /
                                    max(abs(reference_value), 1.0)
                                    if _finite(reference_value) else float("nan")
                                ),
                                "delta_from_previous": delta,
                                "relative_delta_from_previous": (
                                    delta / max(abs(value), 1.0)
                                    if _finite(delta) else float("nan")
                                ),
                                **geometry,
                            })
                            previous = (bucket, value)

                        required_key = f"{grid}_required_bins"
                        required = reference.get(required_key)
                        try:
                            required = int(required) if required else None
                        except ValueError:
                            required = None
                        requirements.append({
                            "dataset": f"legacy_{dataset_name}",
                            "algorithm_vintage": "20260716_pre_certificate",
                            "row_index": seen_rows - 1,
                            "grid": grid,
                            "relative_tolerance": 1.0e-3,
                            "usable_reference": reference.get("reference_confidence")
                            in {"two_grid", "contour"},
                            "required_resolution": required,
                            "budget": 1.0e-4 + 1.0e-3 * max(abs(reference_value), 1.0)
                            if _finite(reference_value) else float("nan"),
                            "reference": reference_value,
                            "reference_uncertainty": float("nan"),
                            **features,
                            **geometry,
                        })
        archive_report[dataset_name] = {
            "status": "ok",
            "raw_rows": seen_rows,
            "matched_rows": matched_rows,
        }
    return curves, requirements, archive_report


def _write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    fields = []
    for record in records:
        for field in record:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _read_csv(path):
    """Read an exported table back for plot-only regeneration."""
    numeric = {
        "row_index", "resolution", "magnification", "seconds", "reference",
        "reference_uncertainty", "abs_error_to_reference",
        "relative_error_to_reference", "delta_from_previous",
        "relative_delta_from_previous", "case_id", "s", "q", "rho", "x",
        "y", "limb_darkening_c", "intended_distance_factor",
        "point_magnification", "caustic_distance", "d_over_rho",
        "relative_tolerance", "required_resolution", "budget",
    }
    boolean = {"support_proven", "converged", "usable_reference"}
    records = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            for field in numeric:
                if field not in record:
                    continue
                if record[field] == "":
                    record[field] = float("nan")
                    continue
                try:
                    record[field] = float(record[field])
                except ValueError:
                    record[field] = float("nan")
            for field in boolean:
                if field in record:
                    record[field] = record[field].lower() == "true"
            records.append(record)
    return records


def _feature_dataset(rows, grid, tolerance, feature_names):
    table = analysis.resolution_table(rows, tolerance)
    features = []
    required = []
    for record in table:
        value = record.get(grid)
        if value is None:
            continue
        features.append([record[name] for name in feature_names])
        required.append(fit_rules._nearest_bucket(value))
    return (
        np.asarray(features, dtype=float).reshape(-1, len(feature_names)),
        np.asarray(required, dtype=float),
    )


def _score_rule(predicted, required):
    if required.size == 0:
        return {"rows": 0}
    index = {bucket: position for position, bucket in enumerate(BUCKETS)}
    predicted = np.asarray(predicted, dtype=float)
    required = np.asarray(required, dtype=float)
    predicted_steps = np.asarray([index[int(value)] for value in predicted])
    required_steps = np.asarray([index[int(value)] for value in required])
    return {
        "rows": int(required.size),
        "coverage": float((predicted >= required).mean()),
        "median_bins": float(np.median(predicted)),
        "p90_bins": float(np.percentile(predicted, 90)),
        "p99_bins": float(np.percentile(predicted, 99)),
        "median_overshoot_steps": float(
            np.median(predicted_steps - required_steps)),
        "median_work_vs_required": float(
            np.median((predicted / required) ** 2)),
    }


def _fit_features(discovery, holdout, coverage):
    report = {
        "coverage_target": coverage,
        "feature_sets": FEATURE_SETS,
        "grids": {},
    }
    for grid in GRIDS:
        grid_report = {}
        for tolerance in TARGET_TOLERANCES:
            tolerance_report = {}
            for name, feature_names in FEATURE_SETS.items():
                discovery_x, discovery_y = _feature_dataset(
                    discovery, grid, tolerance, feature_names)
                holdout_x, holdout_y = _feature_dataset(
                    holdout, grid, tolerance, feature_names)
                if discovery_y.size == 0 or holdout_y.size == 0:
                    continue
                model = fit_rules.fit_quantile_linear(
                    discovery_x, np.log2(discovery_y), coverage)
                if model is None:
                    continue
                discovery_prediction = fit_rules.predict_linear(
                    model, discovery_x)
                holdout_prediction = fit_rules.predict_linear(model, holdout_x)
                tolerance_report[name] = {
                    "features": list(feature_names),
                    "model": model,
                    "discovery": _score_rule(
                        discovery_prediction, discovery_y),
                    "holdout": _score_rule(holdout_prediction, holdout_y),
                }

            # Keep a constant baseline beside the fitted models.  This is the
            # essential check: if adding geometry does not improve holdout
            # cost at the same coverage, the feature does not earn hot-path
            # complexity.
            discovery_x, discovery_y = _feature_dataset(
                discovery, grid, tolerance, ("log_point",))
            _, holdout_y = _feature_dataset(
                holdout, grid, tolerance, ("log_point",))
            if discovery_y.size and holdout_y.size:
                constant = fit_rules._round_up(
                    float(np.quantile(discovery_y, coverage)))
                tolerance_report["constant"] = {
                    "bins": constant,
                    "discovery": _score_rule(
                        np.full(discovery_y.size, constant), discovery_y),
                    "holdout": _score_rule(
                        np.full(holdout_y.size, constant), holdout_y),
                }
            grid_report[str(tolerance)] = tolerance_report
        report["grids"][grid] = grid_report
    return report


def _curve_summary(curves, requirements):
    summary = {"curves": {}, "requirements": {}}
    for dataset_name in sorted({row["dataset"] for row in curves}):
        summary["curves"][dataset_name] = {}
        for grid in GRIDS:
            subset = [
                row for row in curves
                if row["dataset"] == dataset_name and row["grid"] == grid
            ]
            values = [row["relative_error_to_reference"] for row in subset
                      if _finite(row["relative_error_to_reference"])]
            deltas = [row["relative_delta_from_previous"] for row in subset
                      if _finite(row["relative_delta_from_previous"])]
            summary["curves"][dataset_name][grid] = {
                "rows": len(subset),
                "finite_reference_errors": len(values),
                "median_relative_error": (
                    float(np.median(values)) if values else None),
                "p90_relative_error": (
                    float(np.percentile(values, 90)) if values else None),
                "median_relative_delta": (
                    float(np.median(deltas)) if deltas else None),
            }

    for dataset_name in sorted({row["dataset"] for row in requirements}):
        summary["requirements"][dataset_name] = {}
        for grid in GRIDS:
            summary["requirements"][dataset_name][grid] = {}
            for tolerance in TARGET_TOLERANCES:
                values = [
                    row["required_resolution"] for row in requirements
                    if row["dataset"] == dataset_name
                    and row["grid"] == grid
                    and row["relative_tolerance"] == tolerance
                    and row["usable_reference"]
                    and row["required_resolution"] is not None
                ]
                if values:
                    summary["requirements"][dataset_name][grid][str(tolerance)] = {
                        "rows": len(values),
                        "median": float(np.median(values)),
                        "p90": float(np.percentile(values, 90)),
                        "p99": float(np.percentile(values, 99)),
                        "max": int(max(values)),
                    }
    return summary


def _plot_curves_from_csv(path, output):
    """Plot curves while keeping only the grouped error arrays in memory."""
    from collections import defaultdict

    groups = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                error = float(row["relative_error_to_reference"])
                if not math.isfinite(error):
                    continue
                groups[(row["dataset"], row["grid"], int(row["resolution"]))].append(error)
            except (KeyError, TypeError, ValueError):
                continue
    _plot_curve_groups(groups, output)


def _plot_curve_groups(groups, output):
    """Plot median and p90 achieved error against resolution."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(TARGET_TOLERANCES), 2, figsize=(9.2, 8.0), squeeze=False,
        sharex=True,
    )
    for row_index, tolerance in enumerate(TARGET_TOLERANCES):
        for col_index, grid in enumerate(GRIDS):
            axis = axes[row_index][col_index]
            datasets = (
                ("discovery", "#1f4e79", "-"),
                ("holdout", "#c45a11", "-"),
                ("legacy_discovery", "#6a3d9a", ":"),
                ("legacy_independent_validation", "#33a02c", ":"),
            )
            for dataset_name, colour, linestyle in datasets:
                by_resolution = {
                    resolution: values
                    for (dataset, plotted_grid, resolution), values in groups.items()
                    if dataset == dataset_name and plotted_grid == grid
                }
                resolutions = sorted(by_resolution)
                if not resolutions:
                    continue
                median = [np.median(by_resolution[value]) for value in resolutions]
                p90 = [np.percentile(by_resolution[value], 90) for value in resolutions]
                axis.plot(resolutions, median, marker="o", color=colour,
                          linestyle=linestyle, label=f"{dataset_name} median")
                axis.plot(resolutions, p90, linestyle="--", color=colour,
                          alpha=0.75, label=f"{dataset_name} p90")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.2)
            axis.set_title(f"{grid}, reltol={tolerance:g}")
            if col_index == 0:
                axis.set_ylabel("|A(N)-A_ref| / max(|A_ref|,1)")
            if row_index == len(TARGET_TOLERANCES) - 1:
                axis.set_xlabel("resolution N")
    axes[0][0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_curves(curves, output):
    """Compatibility wrapper for a freshly generated in-memory table."""
    from collections import defaultdict

    groups = defaultdict(list)
    for point in curves:
        error = point.get("relative_error_to_reference")
        if _finite(error):
            groups[(point["dataset"], point["grid"], point["resolution"])].append(
                error)
    _plot_curve_groups(groups, output)


def _plot_scatter_from_csv(path, output):
    """Plot geometry scatter without materialising the full requirement table."""
    from collections import defaultdict
    import matplotlib.pyplot as plt

    groups = defaultdict(lambda: ([], [], []))
    rng = np.random.default_rng(20260807)
    max_points_per_panel = 4000
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                if row["usable_reference"].lower() != "true":
                    continue
                x = float(row["d_over_rho"])
                y = float(row["required_resolution"])
                colour = float(row["point_magnification"])
                tolerance = float(row["relative_tolerance"])
                if not all(math.isfinite(value) for value in (x, y, colour)):
                    continue
                arrays = groups[(row["grid"], tolerance)]
                count = len(arrays[0])
                if count < max_points_per_panel:
                    arrays[0].append(x)
                    arrays[1].append(y)
                    arrays[2].append(colour)
                else:
                    replacement = int(rng.integers(0, count + 1))
                    if replacement < max_points_per_panel:
                        arrays[0][replacement] = x
                        arrays[1][replacement] = y
                        arrays[2][replacement] = colour
            except (KeyError, TypeError, ValueError):
                continue

    figure, axes = plt.subplots(
        len(TARGET_TOLERANCES), 2, figsize=(9.2, 8.0), squeeze=False,
        sharey="row",
    )
    for row_index, tolerance in enumerate(TARGET_TOLERANCES):
        for col_index, grid in enumerate(GRIDS):
            axis = axes[row_index][col_index]
            x, y, colour = groups.get((grid, tolerance), ([], [], []))
            if x:
                scatter = axis.scatter(
                    x, y, c=np.log10(np.maximum(colour, 1.0)), s=8,
                    alpha=0.45, cmap="viridis")
                if col_index == 1:
                    figure.colorbar(scatter, ax=axis, label="log10 A_point")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.2)
            axis.set_title(f"{grid}, reltol={tolerance:g}")
            if col_index == 0:
                axis.set_ylabel("required resolution")
            if row_index == len(TARGET_TOLERANCES) - 1:
                axis.set_xlabel("d / rho")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_feature_scatter(requirements, output):
    """Show the two most interpretable candidate predictors."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(TARGET_TOLERANCES), 2, figsize=(9.2, 8.0), squeeze=False,
        sharey="row",
    )
    for row_index, tolerance in enumerate(TARGET_TOLERANCES):
        for col_index, grid in enumerate(GRIDS):
            axis = axes[row_index][col_index]
            points = [
                point for point in requirements
                if point["grid"] == grid
                and point["relative_tolerance"] == tolerance
                and point["usable_reference"]
                and point["required_resolution"] is not None
                and _finite(point["d_over_rho"])
                and _finite(point["point_magnification"])
            ]
            if points:
                x = np.asarray([point["d_over_rho"] for point in points])
                y = np.asarray([point["required_resolution"] for point in points])
                colour = np.asarray([point["point_magnification"] for point in points])
                scatter = axis.scatter(x, y, c=np.log10(np.maximum(colour, 1.0)),
                                       s=8, alpha=0.45, cmap="viridis")
                if col_index == 1:
                    figure.colorbar(scatter, ax=axis, label="log10 A_point")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.2)
            axis.set_title(f"{grid}, reltol={tolerance:g}")
            if col_index == 0:
                axis.set_ylabel("required resolution")
            if row_index == len(TARGET_TOLERANCES) - 1:
                axis.set_xlabel("d / rho")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery")
    parser.add_argument("--holdout")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--legacy-root",
        help="Optional finite-source-auto-20260716 artifact to include as a "
        "separately labelled historical dataset.",
    )
    parser.add_argument("--coverage", type=float, default=0.99)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate plots from CSV files already in --output.",
    )
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    if arguments.plot_only:
        _plot_curves_from_csv(
            output / "resolution_curves.csv",
            output / "magnification_convergence.png",
        )
        _plot_scatter_from_csv(
            output / "resolution_requirements.csv",
            output / "required_resolution_geometry.png",
        )
        print(json.dumps({
            "output": str(output),
            "files": sorted(path.name for path in output.iterdir()),
        }, indent=2))
        return
    if not arguments.discovery or not arguments.holdout:
        parser.error("--discovery and --holdout are required unless --plot-only")
    discovery = analysis.load(arguments.discovery)
    holdout = analysis.load(arguments.holdout)

    discovery_curves, discovery_requirements = _curve_records(
        discovery, "discovery")
    holdout_curves, holdout_requirements = _curve_records(holdout, "holdout")
    curves = discovery_curves + holdout_curves
    requirements = discovery_requirements + holdout_requirements
    legacy_report = None
    if arguments.legacy_root:
        legacy_curves, legacy_requirements, legacy_report = _legacy_curve_records(
            arguments.legacy_root)
        curves.extend(legacy_curves)
        requirements.extend(legacy_requirements)
    _write_csv(output / "resolution_curves.csv", curves)
    _write_csv(output / "resolution_requirements.csv", requirements)

    summary = _curve_summary(curves, requirements)
    summary["inputs"] = {
        "discovery": str(arguments.discovery),
        "holdout": str(arguments.holdout),
        "discovery_rows": len(discovery),
        "holdout_rows": len(holdout),
        "ladder": list(BUCKETS),
        "targets": list(TARGET_TOLERANCES),
        "legacy_root": arguments.legacy_root,
        "legacy_archives": legacy_report,
    }
    (output / "curve_summary.json").write_text(json.dumps(summary, indent=2))

    fits = _fit_features(discovery, holdout, arguments.coverage)
    (output / "feature_fits.json").write_text(json.dumps(fits, indent=2))

    if not arguments.no_plots:
        _plot_curves(curves, output / "magnification_convergence.png")
        _plot_feature_scatter(
            requirements, output / "required_resolution_geometry.png")

    print(json.dumps({
        "output": str(output),
        "discovery_rows": len(discovery),
        "holdout_rows": len(holdout),
        "curve_records": len(curves),
        "requirement_records": len(requirements),
        "files": sorted(path.name for path in output.iterdir()),
    }, indent=2))


if __name__ == "__main__":
    main()

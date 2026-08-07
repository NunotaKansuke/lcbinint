#!/usr/bin/env python3
"""Measure reported finite-source error against a common high-resolution value.

This is an audit, not a calibration pass.  The same rows from an existing
holdout are sent through forced Cartesian and forced polar automatic paths.
For each row a fresh 256/400 Cartesian nested pair and a 400-bin polar witness
are also measured with the current native build.  The 400-bin Cartesian value
is the common reference used in

    R = reported_error / abs(magnification - reference)

The raw case files retain the method actually used.  In particular, a forced
grid request can still enter the binary grazing source-plane quadrature; those
rows are not silently presented as inverse-ray evidence.

Run the measurement with the current build first, for example::

    PYTHONPATH=build-final-testing:/path/to/site-packages python -S \
      tests/diagnostics/recal2026/estimator_audit.py measure \
      tests/diagnostics/results/recal2026/holdout \
      --output /tmp/lcbinint-estimator-audit --workers 24

Then print the summary with::

    .../estimator_audit.py analyse /tmp/lcbinint-estimator-audit
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lcbinint  # noqa: E402


TARGETS = (1.0e-2, 1.0e-3, 1.0e-4)
REFERENCE_BINS = (256, 400)


def _scalar(value):
    return float(value[0])


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _curve(grid, nbin, reltol=0.0, reference=False):
    # Disable exits that would bypass the requested image-plane route.  The
    # source-plane grazing arbitration remains enabled because it is part of
    # the current binary algorithm and must be reported, not hidden.
    options = dict(
        coordinates="vbm",
        nbin=nbin,
        caustic_bins=1400,
        inverse_ray_grid=grid,
        max_source_bins=400,
        point_source_threshold=1.0e6 if reference else 0.0,
        hexadecapole_threshold=0.0,
        adaptive_hex_threshold=0.0,
    )
    if reltol:
        options["reltol"] = reltol
    return lcbinint.LightCurve(lens="binary", options=lcbinint.Options(**options))


def _evaluate(curve, row):
    started = time.perf_counter()
    info = curve.info(
        row["x"],
        t0=0.0,
        tE=1.0,
        u0=row["y"],
        alpha=0.0,
        s=row["s"],
        q=row["q"],
        rho=row["rho"],
        limb_darkening_c=row.get("limb_darkening_c", 0.0),
    )
    return {
        "magnification": _scalar(info.finite_source_magnifications),
        "error_estimate": _scalar(info.finite_source_error_estimates),
        "converged": bool(info.finite_source_converged[0]),
        "method": str(info.finite_source_method_names[0]),
        "refinement_level": int(info.finite_source_refinement_levels[0]),
        "seconds": time.perf_counter() - started,
    }


def _compact_row(row, row_index):
    keys = (
        "case_id", "s", "q", "rho", "x", "y", "profile",
        "limb_darkening_c", "intended_distance_factor",
    )
    return {"row_index": row_index, **{key: row.get(key) for key in keys}}


def _measure_case(payload):
    input_path, output_path, reference_mode = payload
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists():
        return input_path.name, "skipped", 0.0

    source = json.loads(input_path.read_text())
    rows = source.get("rows", [])
    # Curves are reused across the rows of one case file.  This preserves the
    # production cache behaviour while keeping each target/grid independent.
    auto = {
        (grid, target): _curve(grid, "auto", target)
        for grid in ("cartesian", "polar")
        for target in TARGETS
    }
    reference = {
        ("cartesian", bins): _curve("cartesian", bins, reference=True)
        for bins in REFERENCE_BINS
    }
    if reference_mode == "full":
        reference[("polar", 400)] = _curve("polar", 400, reference=True)

    measured = []
    started = time.perf_counter()
    for row_index, row in enumerate(rows):
        record = _compact_row(row, row_index)
        record["evaluations"] = {}
        for target in TARGETS:
            target_record = {}
            for grid in ("cartesian", "polar"):
                try:
                    target_record[grid] = _evaluate(auto[(grid, target)], row)
                except Exception as error:  # noqa: BLE001
                    target_record[grid] = {
                        "error": f"{type(error).__name__}: {error}"
                    }
            record["evaluations"][str(target)] = target_record

        try:
            cartesian_256 = _evaluate(reference[("cartesian", 256)], row)
            cartesian_400 = _evaluate(reference[("cartesian", 400)], row)
            value_256 = cartesian_256["magnification"]
            value_400 = cartesian_400["magnification"]
            reference_record = {
                "value": value_400,
                "cartesian_256": cartesian_256,
                "cartesian_400": cartesian_400,
                "nested_gap": (
                    abs(value_400 - value_256)
                    if _finite(value_400) and _finite(value_256)
                    else float("inf")
                ),
            }
            if reference_mode == "full":
                polar_400 = _evaluate(reference[("polar", 400)], row)
                polar_value = polar_400["magnification"]
                reference_record["polar_400"] = polar_400
                reference_record["cross_grid_gap"] = (
                    abs(value_400 - polar_value)
                    if _finite(value_400) and _finite(polar_value)
                    else float("inf")
                )
            else:
                reference_record["cross_grid_gap"] = float("nan")
            record["reference"] = reference_record
        except Exception as error:  # noqa: BLE001
            record["reference"] = {
                "error": f"{type(error).__name__}: {error}"
            }
        measured.append(record)

    result = {
        "source": str(input_path),
        "case": source.get("case"),
        "targets": list(TARGETS),
        "reference_mode": reference_mode,
        "reference_bins": list(REFERENCE_BINS),
        "rows": measured,
        "seconds": time.perf_counter() - started,
    }
    temporary = output_path.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.replace(output_path)
    return input_path.name, "done", result["seconds"]


def measure(arguments):
    input_dir = Path(arguments.input)
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(input_dir.glob("case-*.json"))
    if arguments.offset:
        inputs = inputs[arguments.offset:]
    if arguments.limit:
        inputs = inputs[:arguments.limit]
    payloads = [
        (str(path), str(output_dir / path.name), arguments.reference_mode)
        for path in inputs
    ]
    started = time.perf_counter()
    with multiprocessing.Pool(arguments.workers) as pool:
        for index, (name, status, seconds) in enumerate(
                pool.imap_unordered(_measure_case, payloads), 1):
            print(
                f"[{index}/{len(payloads)}] {name} {status} "
                f"{seconds / 60.0:.2f} min",
                flush=True,
            )
    print(f"finished in {(time.perf_counter() - started) / 60.0:.2f} min")


def _quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _r_summary(records, grid, target, trusted_only=False):
    r_values = []
    under = 0
    eligible = 0
    reference_rejected = 0
    reference_limited = 0
    conservative_reference_limited = 0
    ambiguous_reference_limited = 0
    by_method = {}
    for record in records:
        result = record.get("evaluations", {}).get(str(target), {}).get(grid, {})
        reference = record.get("reference", {})
        estimate = result.get("error_estimate")
        value = result.get("magnification")
        truth = reference.get("value")
        if not (_finite(estimate) and _finite(value) and _finite(truth)):
            continue
        scale = max(abs(truth), 1.0)
        budget = target * scale
        trusted = (
            _finite(reference.get("nested_gap"))
            and _finite(reference.get("cross_grid_gap"))
            and max(reference["nested_gap"], reference["cross_grid_gap"])
            <= 0.1 * budget
        )
        if trusted_only and not trusted:
            reference_rejected += 1
            continue
        eligible += 1
        denominator = abs(value - truth)
        spreads = [
            spread for spread in (
                reference.get("nested_gap"),
                reference.get("cross_grid_gap"),
            ) if _finite(spread)
        ]
        reference_floor = max([1.0e-12 * scale, *spreads])
        if denominator <= reference_floor:
            reference_limited += 1
            if estimate > reference_floor:
                conservative_reference_limited += 1
            else:
                ambiguous_reference_limited += 1
            continue
        ratio = estimate / denominator
        if math.isfinite(ratio):
            r_values.append(ratio)
            if ratio < 1.0:
                under += 1
        method = result.get("method", "<unknown>")
        by_method.setdefault(method, []).append(ratio)

    summary = {
        "rows_with_finite_values": eligible,
        "rows_with_finite_R": len(r_values),
        "reference_rejected": reference_rejected,
        "reference_limited": reference_limited,
        "conservative_reference_limited": conservative_reference_limited,
        "ambiguous_reference_limited": ambiguous_reference_limited,
        "underestimated": under,
        "underestimate_rate": (
            float(under / len(r_values)) if r_values else None
        ),
        "R": {
            "p01": _quantile(r_values, 0.01),
            "p05": _quantile(r_values, 0.05),
            "p50": _quantile(r_values, 0.50),
            "p95": _quantile(r_values, 0.95),
            "p99": _quantile(r_values, 0.99),
            "min": min(r_values) if r_values else None,
            "max": max(r_values) if r_values else None,
        },
        "by_method": {
            method: {
                "n": len(values),
                "underestimate_rate": sum(value < 1.0 for value in values)
                / len(values),
                "p50": _quantile(values, 0.50),
                "p05": _quantile(values, 0.05),
                "p95": _quantile(values, 0.95),
                "min": min(values),
            }
            for method, values in sorted(by_method.items())
            if values
        },
    }
    return summary


def analyse(arguments):
    directory = Path(arguments.input)
    records = []
    for path in sorted(directory.glob("case-*.json")):
        try:
            records.extend(json.loads(path.read_text()).get("rows", []))
        except json.JSONDecodeError:
            continue
    report = {
        "directory": str(directory),
        "records": len(records),
        "grids": {},
    }
    for grid in ("cartesian", "polar"):
        report["grids"][grid] = {str(target): {
            "all": _r_summary(records, grid, target),
            "reference_consistent": _r_summary(
                records, grid, target, trusted_only=True),
        } for target in TARGETS}
    output = json.dumps(report, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(output + "\n")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("input")
    measure_parser.add_argument("--output", required=True)
    measure_parser.add_argument("--workers", type=int, default=1)
    measure_parser.add_argument("--offset", type=int, default=0)
    measure_parser.add_argument("--limit", type=int, default=0,
                                help="number of case files, not rows")
    measure_parser.add_argument(
        "--reference-mode", choices=("full", "nested-cartesian"),
        default="full",
        help="full adds a 400-bin polar witness; nested-cartesian is a "
             "shorter mode whose rows are excluded from the trusted subset",
    )
    measure_parser.set_defaults(function=measure)

    analyse_parser = subparsers.add_parser("analyse")
    analyse_parser.add_argument("input")
    analyse_parser.add_argument("--output", default="")
    analyse_parser.set_defaults(function=analyse)

    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()

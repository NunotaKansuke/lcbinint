#!/usr/bin/env python3
"""Freeze finite-source calibration results into reviewable repository artifacts.

This script consumes the raw discovery, independent-validation, and long-timeout
JSON directories.  It writes a compact per-source table plus the exact constants
used by the runtime heuristic.  VBM values are comparison data only: they are
never used as the sole numerical reference.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from finite_source_calibration_analyze import (
    derive_reference,
    fit_upper_resolution_model,
    required_bins,
)


ABS_TOL = 1.0e-4
REL_TOL = 1.0e-3
CONSENSUS_FACTOR = 2.0
QUANTILE = 0.98
SAFETY_FACTOR = 1.10
BUCKETS = (16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400)
FEATURE_NAMES = (
    "log10_point_magnification",
    "log10_rho",
    "log10_q_small",
    "log10_distance_over_rho",
    "near_caustic_strength",
    "companion_resolution_risk",
    "limb_c",
)


def finite(result: dict[str, Any]) -> bool:
    return "value" in result and math.isfinite(float(result["value"]))


def point_magnification(row: dict[str, Any]) -> float:
    candidates = [row.get("lc_auto", {})]
    for sequence in row.get("lc_fixed_sequences", {}).values():
        candidates.extend(sequence)
    for result in candidates:
        value = result.get("point_magnification")
        if value is not None and math.isfinite(float(value)):
            return abs(float(value))
    return 1.0


def feature_vector(case: dict[str, Any], row: dict[str, Any]) -> np.ndarray:
    q = abs(float(case["mass_ratio"]))
    q_small = min(q, 1.0 / max(q, 1.0e-300))
    rho = float(case["source_radius"])
    distance_ratio = float(row["caustic_distance_over_rho"])
    point_mag = point_magnification(row)
    return np.asarray((
        math.log10(max(point_mag, 1.0)),
        math.log10(max(rho, 1.0e-12)),
        math.log10(max(q_small, 1.0e-12)),
        math.log10(max(distance_ratio, 1.0e-3)),
        max(0.0, 2.0 - min(distance_ratio, 2.0)),
        max(0.0, math.log10(max(4.0 * rho / max(q_small, 1.0e-12), 1.0))),
        float(row["limb_c"]),
    ), dtype=float)


def load_rows(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    errors = 0
    case_count = 0
    configuration = None
    for path in sorted(directory.glob("case-*.json")):
        document = json.loads(path.read_text())
        if "error" in document:
            errors += 1
            continue
        case_count += 1
        configuration = configuration or document.get("configuration")
        case = document["case"]
        for row in document.get("rows", []):
            reference = derive_reference(
                row, ABS_TOL, REL_TOL, CONSENSUS_FACTOR, 1.0e3,
            )
            entry = {"case": case, "row": row, "reference": reference}
            if reference.get("trusted"):
                entry["required"] = {
                    grid: required_bins(sequence, reference["value"], reference["tolerance"])
                    for grid, sequence in row["lc_fixed_sequences"].items()
                }
                entry["features"] = feature_vector(case, row)
            output.append(entry)
    return output, {
        "cases": case_count,
        "rows": len(output),
        "case_errors": errors,
        "configuration": configuration,
    }


def fit_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for entry in rows:
        if not entry["reference"].get("trusted"):
            continue
        needed = entry["required"]["cartesian"]
        records.append({
            "required_bins": needed["bins"],
            "censored": needed["censored"],
            "features": entry["features"],
        })
    # The shared fitter only requires feature vectors and labels.  Its feature
    # names are replaced below because this production model deliberately omits
    # local caustic-component size (not available on the hot path).
    model = fit_upper_resolution_model(records, QUANTILE)
    model["feature_names"] = FEATURE_NAMES
    return model


def raw_prediction(model: dict[str, Any], feature: np.ndarray) -> float:
    mean = np.asarray(model["feature_mean"])
    std = np.asarray(model["feature_std"])
    beta = np.asarray(model["coefficients"])
    design = np.concatenate(([1.0], (feature - mean) / std))
    return 2.0 ** float(design @ beta) * SAFETY_FACTOR


def snap_up(value: float) -> int:
    for bucket in BUCKETS:
        if value <= bucket:
            return bucket
    return BUCKETS[-1]


def select_grid_and_bins(model: dict[str, Any], entry: dict[str, Any]) -> tuple[str, int]:
    row, case = entry["row"], entry["case"]
    a_point = point_magnification(row)
    distance_ratio = float(row["caustic_distance_over_rho"])
    rho = float(case["source_radius"])
    q = abs(float(case["mass_ratio"]))
    q_small = min(q, 1.0 / max(q, 1.0e-300))
    polar = a_point >= 300.0 or (a_point >= 100.0 and distance_ratio < 0.3)
    if polar:
        return "polar", 64
    bins = snap_up(raw_prediction(model, entry["features"]))
    if 0.9 < distance_ratio < 1.1:
        bins = max(bins, 100)
    if 4.0 * rho / max(q_small, 1.0e-300) > 50.0:
        bins = max(bins, 80)
    return "cartesian", bins


def recommends_vbm(entry: dict[str, Any]) -> bool:
    row = entry["row"]
    a_point = point_magnification(row)
    distance_ratio = float(row["caustic_distance_over_rho"])
    limb_c = float(row["limb_c"])
    if limb_c == 0.0:
        return a_point < 1000.0 and not (0.9 < distance_ratio < 1.1)
    return a_point < 5.0 and distance_ratio > 1.05


def evaluate(rows: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    evaluated = under = selected_vbm = vbm_bad = both_accurate = vbm_faster = 0
    selected_bins: list[int] = []
    selected_time = current_time = oracle_time = 0
    by_limb: dict[str, dict[str, int]] = {}
    for entry in rows:
        reference = entry["reference"]
        if not reference.get("trusted"):
            continue
        grid, bins = select_grid_and_bins(model, entry)
        needed = entry["required"][grid]
        if needed["bins"] > 0 and not needed["censored"]:
            evaluated += 1
            selected_bins.append(bins)
            under += bins < needed["bins"]

        row = entry["row"]
        sequences = row["lc_fixed_sequences"]
        # Grid timing uses matching nbin values only and therefore evaluates
        # the switching boundary independently of the nbin predictor.
        cart = {int(x["bins"]): x for x in sequences["cartesian"] if finite(x)}
        polar = {int(x["bins"]): x for x in sequences["polar"] if finite(x)}
        common = sorted(set(cart) & set(polar))
        if common:
            timing_bin = min(common, key=lambda x: abs(x - 64))
            ct, pt = int(cart[timing_bin]["elapsed_ns"]), int(polar[timing_bin]["elapsed_ns"])
            chosen = pt if grid == "polar" else ct
            a_point = point_magnification(row)
            current = pt if a_point >= 100.0 else ct
            selected_time += chosen
            current_time += current
            oracle_time += min(ct, pt)

        if recommends_vbm(entry):
            selected_vbm += 1
            mode = "contour" if float(row["limb_c"]) == 0.0 else "dark"
            vbm = row.get("vbm", {}).get(mode, {})
            accurate = finite(vbm) and abs(float(vbm["value"]) - reference["value"]) <= reference["tolerance"]
            vbm_bad += not accurate
            limb_key = str(row["limb_c"])
            stats = by_limb.setdefault(limb_key, {"selected": 0, "bad": 0, "both_accurate": 0, "vbm_faster": 0})
            stats["selected"] += 1
            stats["bad"] += not accurate
            lc = row.get("lc_auto", {})
            lc_accurate = finite(lc) and abs(float(lc["value"]) - reference["value"]) <= reference["tolerance"]
            if accurate and lc_accurate:
                both_accurate += 1
                stats["both_accurate"] += 1
                faster = int(vbm["elapsed_ns"]) < int(lc["elapsed_ns"])
                vbm_faster += faster
                stats["vbm_faster"] += faster
    return {
        "nbin": {
            "evaluated_rows": evaluated,
            "underpredictions": under,
            "selected_median": float(np.median(selected_bins)) if selected_bins else None,
            "selected_mean": float(np.mean(selected_bins)) if selected_bins else None,
            "fraction_below_50": float(np.mean(np.asarray(selected_bins) < 50)) if selected_bins else None,
            "fraction_above_50": float(np.mean(np.asarray(selected_bins) > 50)) if selected_bins else None,
        },
        "grid": {
            "selected_time_seconds": selected_time / 1.0e9,
            "current_time_seconds": current_time / 1.0e9,
            "oracle_time_seconds": oracle_time / 1.0e9,
        },
        "vbm_recommendation": {
            "selected": selected_vbm,
            "inaccurate_or_failed": vbm_bad,
            "both_accurate": both_accurate,
            "vbm_faster_when_both_accurate": vbm_faster,
            "by_limb_coefficient": by_limb,
        },
    }


def high_magnification_summary(directory: Path) -> dict[str, Any]:
    documents = [json.loads(path.read_text()) for path in sorted(directory.glob("sample-*.json"))]
    errors = sum("error" in doc for doc in documents)
    rows = [doc for doc in documents if "error" not in doc]
    stable_two_grid = stable_polar = 0
    high_64_evaluated = high_64_bad = all_100_evaluated = all_100_bad = 0
    for doc in rows:
        sequences = doc["sequences"]
        cart = [x for x in sequences["cartesian"] if finite(x)]
        polar = [x for x in sequences["polar"] if finite(x)]
        reference = None
        if len(cart) >= 2 and len(polar) >= 2:
            c, p = float(cart[-1]["value"]), float(polar[-1]["value"])
            tol = ABS_TOL + REL_TOL * max(abs(c), 1.0)
            if abs(c - p) <= 2.0 * tol:
                reference = 0.5 * (c + p)
                stable_two_grid += 1
        if reference is None and len(polar) >= 2:
            reference = float(polar[-1]["value"])
            stable_polar += 1
        if reference is None:
            continue
        tol = ABS_TOL + REL_TOL * max(abs(reference), 1.0)
        by_bins = {int(x["bins"]): x for x in polar}
        a_point = doc.get("original_lc_auto", {}).get("point_magnification")
        if a_point is None:
            a_point = next(
                (result["point_magnification"] for result in cart + polar
                 if result.get("point_magnification") is not None),
                1.0,
            )
        a_point = abs(float(a_point))
        if a_point >= 1000.0 and 64 in by_bins:
            high_64_evaluated += 1
            high_64_bad += abs(float(by_bins[64]["value"]) - reference) > tol
        if 100 in by_bins:
            all_100_evaluated += 1
            all_100_bad += abs(float(by_bins[100]["value"]) - reference) > tol
    return {
        "samples": len(documents),
        "sample_errors": errors,
        "two_grid_stable_references": stable_two_grid,
        "polar_tail_references": stable_polar,
        "point_magnification_ge_1000_polar_nbin_64": {
            "evaluated": high_64_evaluated, "tolerance_violations": high_64_bad,
        },
        "all_samples_polar_nbin_100": {
            "evaluated": all_100_evaluated, "tolerance_violations": all_100_bad,
        },
    }


def write_compact_rows(path: Path, datasets: dict[str, list[dict[str, Any]]], model: dict[str, Any]) -> None:
    fields = (
        "dataset", "case_id", "point_id", "source_x", "source_y", "sampling", "limb_c",
        "separation", "mass_ratio", "rho",
        "point_magnification", "distance_over_rho", "reference", "reference_confidence",
        "cartesian_required_bins", "cartesian_censored", "polar_required_bins", "polar_censored",
        "selected_grid", "selected_bins", "recommend_vbm",
    )
    with path.open("wb") as raw_handle, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_handle, mtime=0
    ) as gzip_handle, io.TextIOWrapper(gzip_handle, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, rows in datasets.items():
            for entry in rows:
                if not entry["reference"].get("trusted"):
                    continue
                case, row = entry["case"], entry["row"]
                grid, bins = select_grid_and_bins(model, entry)
                writer.writerow({
                    "dataset": dataset, "case_id": case["case_id"], "point_id": row["point_id"],
                    "source_x": row["source_x"], "source_y": row["source_y"],
                    "sampling": row.get("sampling", ""), "limb_c": row["limb_c"],
                    "separation": case["separation"],
                    "mass_ratio": case["mass_ratio"], "rho": case["source_radius"],
                    "point_magnification": point_magnification(row),
                    "distance_over_rho": row["caustic_distance_over_rho"],
                    "reference": entry["reference"]["value"],
                    "reference_confidence": entry["reference"]["confidence"],
                    "cartesian_required_bins": entry["required"]["cartesian"]["bins"],
                    "cartesian_censored": entry["required"]["cartesian"]["censored"],
                    "polar_required_bins": entry["required"]["polar"]["bins"],
                    "polar_censored": entry["required"]["polar"]["censored"],
                    "selected_grid": grid, "selected_bins": bins,
                    "recommend_vbm": recommends_vbm(entry),
                })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--high-magnification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    discovery, discovery_meta = load_rows(args.discovery)
    validation, validation_meta = load_rows(args.validation)
    model = fit_model(discovery)
    rules = {
        "schema_version": 1,
        "calibration_tolerance": {"absolute": ABS_TOL, "relative": REL_TOL},
        "nbin_model": {
            **model, "safety_factor": SAFETY_FACTOR, "buckets": BUCKETS,
            "cartesian_tangent_floor": {"lower": 0.9, "upper": 1.1, "bins": 100},
            "cartesian_companion_floor": {"four_rho_over_q_threshold": 50.0, "bins": 80},
        },
        "grid_rule": {
            "polar_if_point_magnification_at_least": 300.0,
            "polar_near_caustic_if_point_magnification_at_least": 100.0,
            "polar_near_caustic_distance_over_rho_below": 0.3,
            "polar_bins": 64,
        },
        "vbm_recommendation_rule": {
            "scope": "one source position",
            "uniform": {"point_magnification_below": 1000.0, "excluded_distance_band": [0.9, 1.1]},
            "limb_darkened": {"point_magnification_below": 5.0, "distance_over_rho_above": 1.05},
        },
    }
    summary = {
        "schema_version": 1,
        "discovery": {"metadata": discovery_meta, "evaluation": evaluate(discovery, model)},
        "independent_validation": {"metadata": validation_meta, "evaluation": evaluate(validation, model)},
        "high_magnification": high_magnification_summary(args.high_magnification),
    }
    (args.output / "calibrated-rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_compact_rows(args.output / "source-profile-results.csv.gz", {
        "discovery": discovery, "independent_validation": validation,
    }, model)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

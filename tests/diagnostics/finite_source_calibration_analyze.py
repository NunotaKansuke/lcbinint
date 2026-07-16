#!/usr/bin/env python3
"""Analyze finite-source calibration cases and fit preliminary heuristics.

The analysis never promotes a VBM value to an oracle on its own.  A reference
is trusted only after a fixed-resolution lcbinint sequence converges, with a
second grid or ordinary-magnification VBM result used as corroboration.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


def _finite(result: dict[str, Any]) -> bool:
    return "value" in result and math.isfinite(float(result["value"]))


def _target(value: float, absolute: float, relative: float) -> float:
    return absolute + relative * max(abs(value), 1.0)


def _sequence_tail(sequence: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [result for result in sequence if _finite(result)]
    if not valid:
        return None
    high = valid[-1]
    previous = valid[-2] if len(valid) >= 2 else None
    change = abs(float(high["value"]) - float(previous["value"])) if previous else math.inf
    return {
        "value": float(high["value"]),
        "bins": int(high["bins"]),
        "method": high.get("method"),
        "change": change,
        "valid_count": len(valid),
    }


def _vbm_result(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    mode = "contour" if float(row["limb_c"]) == 0.0 else "dark"
    return mode, row.get("vbm", {}).get(mode, {})


def derive_reference(
    row: dict[str, Any],
    absolute: float,
    relative: float,
    consensus_factor: float,
    vbm_max_point_magnification: float,
) -> dict[str, Any]:
    sequences = row.get("lc_fixed_sequences", {})
    tails = {
        grid: _sequence_tail(sequence)
        for grid, sequence in sequences.items()
    }
    tails = {grid: tail for grid, tail in tails.items() if tail is not None}
    if not tails:
        return {"trusted": False, "reason": "no finite lcbinint tail"}

    # A fast-path result is independent of nbin.  It may still be checked
    # against VBM below, but does not need a resolution label.
    fast = [
        tail for tail in tails.values()
        if tail["method"] in {"point_source", "hexadecapole"}
    ]
    finite_tails = [
        tail for tail in tails.values()
        if tail["method"] not in {"point_source", "hexadecapole"}
    ]
    candidates = finite_tails or fast
    scale_value = float(np.median([tail["value"] for tail in candidates]))
    tolerance = _target(scale_value, absolute, relative)
    stable = [tail for tail in candidates if tail["change"] <= consensus_factor * tolerance]

    if len(stable) >= 2:
        spread = max(tail["value"] for tail in stable) - min(tail["value"] for tail in stable)
        if spread <= consensus_factor * tolerance:
            return {
                "trusted": True,
                "confidence": "two_grid",
                "value": float(np.median([tail["value"] for tail in stable])),
                "tolerance": tolerance,
                "tails": tails,
            }

    mode, vbm = _vbm_result(row)
    point_mag = abs(float(row.get("lc_auto", {}).get("point_magnification", math.inf)))
    if len(stable) == 1 and _finite(vbm) and point_mag <= vbm_max_point_magnification:
        only = stable[0]
        if abs(only["value"] - float(vbm["value"])) <= consensus_factor * tolerance:
            return {
                "trusted": True,
                "confidence": f"one_grid_plus_vbm_{mode}",
                "value": only["value"],
                "tolerance": tolerance,
                "tails": tails,
            }

    # Far-field point/hex results often have identical tails (change=0) on
    # both named grids.  Treat their agreement as a two-path check even though
    # both correctly bypass the inverse-ray backend.
    if len(fast) >= 2:
        spread = max(tail["value"] for tail in fast) - min(tail["value"] for tail in fast)
        if spread <= tolerance:
            return {
                "trusted": True,
                "confidence": "fast_path_grid_independent",
                "value": float(np.median([tail["value"] for tail in fast])),
                "tolerance": tolerance,
                "tails": tails,
            }
    return {"trusted": False, "reason": "high-resolution paths do not corroborate", "tails": tails}


def required_bins(
    sequence: list[dict[str, Any]],
    reference: float,
    tolerance: float,
) -> dict[str, Any]:
    valid = [result for result in sequence if _finite(result)]
    if valid and all(
        result.get("method") in {"point_source", "hexadecapole"} for result in valid
    ):
        return {"bins": 0, "censored": False, "reason": "fast_path"}
    for index, candidate in enumerate(sequence):
        if not _finite(candidate):
            continue
        suffix = sequence[index:]
        if not suffix or not all(_finite(result) for result in suffix):
            continue
        if all(abs(float(result["value"]) - reference) <= tolerance for result in suffix):
            return {"bins": int(candidate["bins"]), "censored": False}
    maximum = max((int(result["bins"]) for result in valid), default=0)
    return {"bins": maximum, "censored": True, "reason": "not converged by maximum bins"}


FEATURE_NAMES = (
    "log10_point_magnification",
    "log10_rho",
    "log10_q_small",
    "log10_distance_over_rho",
    "log10_rho_over_local_caustic",
    "near_caustic_strength",
    "companion_resolution_risk",
    "limb_c",
)


def features(case: dict[str, Any], row: dict[str, Any]) -> np.ndarray:
    q = float(case["mass_ratio"])
    q_small = min(abs(q), 1.0 / max(abs(q), 1e-300))
    rho = float(case["source_radius"])
    distance_ratio = float(row["caustic_distance_over_rho"])
    point_mag = abs(float(row.get("lc_auto", {}).get("point_magnification", 1.0)))
    rho_local = float(row["rho_over_nearest_component_diagonal"])
    return np.asarray([
        math.log10(max(point_mag, 1.0)),
        math.log10(max(rho, 1e-12)),
        math.log10(max(q_small, 1e-12)),
        math.log10(max(distance_ratio, 1e-3)),
        math.log10(max(rho_local, 1e-8)),
        max(0.0, 2.0 - min(distance_ratio, 2.0)),
        max(0.0, math.log10(max(4.0 * rho / max(q_small, 1e-12), 1.0))),
        float(row["limb_c"]),
    ], dtype=float)


def fit_upper_resolution_model(records: list[dict[str, Any]], quantile: float) -> dict[str, Any]:
    usable = [record for record in records if record["required_bins"] > 0 and not record["censored"]]
    if len(usable) < len(FEATURE_NAMES) + 5:
        return {"fitted": False, "reason": "insufficient uncensored rows", "rows": len(usable)}
    x_raw = np.vstack([record["features"] for record in usable])
    y = np.log2(np.asarray([record["required_bins"] for record in usable], dtype=float))
    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std[std < 1e-10] = 1.0
    x = np.column_stack((np.ones(len(x_raw)), (x_raw - mean) / std))

    def objective(beta: np.ndarray) -> float:
        residual = y - x @ beta
        pinball = np.where(residual >= 0.0, quantile * residual, (quantile - 1.0) * residual)
        return float(pinball.mean() + 1e-4 * np.dot(beta[1:], beta[1:]))

    initial = np.linalg.lstsq(x, y, rcond=None)[0]
    result = minimize(objective, initial, method="Powell", options={"maxiter": 3000})
    prediction = x @ result.x
    predicted_bins = np.ceil(2.0 ** prediction)
    actual_bins = 2.0 ** y
    under = predicted_bins < actual_bins
    ratios = predicted_bins / actual_bins
    return {
        "fitted": True,
        "rows": len(usable),
        "quantile": quantile,
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "coefficients": result.x.tolist(),
        "optimizer_success": bool(result.success),
        "training_underprediction_fraction": float(under.mean()),
        "median_predicted_over_required": float(np.median(ratios)),
        "p90_predicted_over_required": float(np.quantile(ratios, 0.9)),
    }


def predict_resolution_bins(model: dict[str, Any], feature: np.ndarray) -> int:
    mean = np.asarray(model["feature_mean"], dtype=float)
    std = np.asarray(model["feature_std"], dtype=float)
    beta = np.asarray(model["coefficients"], dtype=float)
    design = np.concatenate(([1.0], (feature - mean) / std))
    return max(1, int(math.ceil(2.0 ** float(np.dot(design, beta)))))


def grouped_resolution_validation(
    records: list[dict[str, Any]], quantile: float, folds: int = 5
) -> dict[str, Any]:
    predictions: list[tuple[int, int]] = []
    used_folds = 0
    for fold in range(folds):
        training = [record for record in records if record["case_id"] % folds != fold]
        testing = [
            record for record in records
            if record["case_id"] % folds == fold
            and record["required_bins"] > 0
            and not record["censored"]
        ]
        if not testing:
            continue
        model = fit_upper_resolution_model(training, quantile)
        if not model.get("fitted"):
            continue
        used_folds += 1
        predictions.extend(
            (predict_resolution_bins(model, record["features"]), record["required_bins"])
            for record in testing
        )
    if not predictions:
        return {"evaluated": False, "reason": "insufficient grouped folds"}
    predicted = np.asarray([item[0] for item in predictions], dtype=float)
    actual = np.asarray([item[1] for item in predictions], dtype=float)
    return {
        "evaluated": True,
        "folds": used_folds,
        "rows": len(predictions),
        "underprediction_fraction": float((predicted < actual).mean()),
        "median_predicted_over_required": float(np.median(predicted / actual)),
        "p90_predicted_over_required": float(np.quantile(predicted / actual, 0.9)),
    }


def fit_engine_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [record for record in records if record.get("engine_winner") in {"lcbinint", "vbm"}]
    if len(usable) < len(FEATURE_NAMES) + 5 or len({r["engine_winner"] for r in usable}) < 2:
        return {"fitted": False, "reason": "insufficient two-class rows", "rows": len(usable)}
    x_raw = np.vstack([record["features"] for record in usable])
    y = np.asarray([record["engine_winner"] == "vbm" for record in usable], dtype=float)
    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std[std < 1e-10] = 1.0
    x = np.column_stack((np.ones(len(x_raw)), (x_raw - mean) / std))

    def objective(beta: np.ndarray) -> float:
        score = np.clip(x @ beta, -40.0, 40.0)
        return float(np.logaddexp(0.0, score).sum() - np.dot(y, score) + 1e-3 * np.dot(beta[1:], beta[1:]))

    result = minimize(objective, np.zeros(x.shape[1]), method="BFGS")
    probability = 1.0 / (1.0 + np.exp(-np.clip(x @ result.x, -40.0, 40.0)))
    prediction = probability >= 0.5
    return {
        "fitted": True,
        "rows": len(usable),
        "vbm_fraction": float(y.mean()),
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "coefficients": result.x.tolist(),
        "training_accuracy": float((prediction == y).mean()),
        "false_vbm_choice_fraction": float(np.logical_and(prediction, y == 0.0).mean()),
    }


def predict_vbm(model: dict[str, Any], feature: np.ndarray) -> bool:
    mean = np.asarray(model["feature_mean"], dtype=float)
    std = np.asarray(model["feature_std"], dtype=float)
    beta = np.asarray(model["coefficients"], dtype=float)
    design = np.concatenate(([1.0], (feature - mean) / std))
    return float(np.dot(design, beta)) >= 0.0


def grouped_engine_validation(records: list[dict[str, Any]], folds: int = 5) -> dict[str, Any]:
    labels: list[bool] = []
    predictions: list[bool] = []
    used_folds = 0
    for fold in range(folds):
        training = [record for record in records if record["case_id"] % folds != fold]
        testing = [
            record for record in records
            if record["case_id"] % folds == fold
            and record.get("engine_winner") in {"lcbinint", "vbm"}
        ]
        if not testing:
            continue
        model = fit_engine_model(training)
        if not model.get("fitted"):
            continue
        used_folds += 1
        labels.extend(record["engine_winner"] == "vbm" for record in testing)
        predictions.extend(predict_vbm(model, record["features"]) for record in testing)
    if not labels:
        return {"evaluated": False, "reason": "insufficient grouped folds"}
    labels_array = np.asarray(labels, dtype=bool)
    predictions_array = np.asarray(predictions, dtype=bool)
    return {
        "evaluated": True,
        "folds": used_folds,
        "rows": len(labels),
        "accuracy": float((predictions_array == labels_array).mean()),
        "false_vbm_choice_fraction": float(
            np.logical_and(predictions_array, ~labels_array).mean()
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    confidence: Counter[str] = Counter()
    censored: Counter[str] = Counter()
    engine: Counter[str] = Counter()
    cases = 0
    for path in sorted(args.directory.glob("case-*.json")):
        document = json.loads(path.read_text())
        if "error" in document:
            continue
        cases += 1
        case = document["case"]
        for row in document.get("rows", []):
            reference = derive_reference(
                row, args.absolute_tolerance, args.relative_tolerance,
                args.consensus_factor, args.vbm_reference_max_magnification,
            )
            if not reference.get("trusted"):
                confidence["untrusted"] += 1
                continue
            confidence[reference["confidence"]] += 1
            x = features(case, row)
            for grid, sequence in row["lc_fixed_sequences"].items():
                needed = required_bins(sequence, reference["value"], reference["tolerance"])
                censored[f"{grid}:{needed['censored']}"] += 1
                records.append({
                    "case_id": int(case["case_id"]),
                    "grid": grid,
                    "limb_c": float(row["limb_c"]),
                    "reference": reference["value"],
                    "reference_confidence": reference["confidence"],
                    "required_bins": needed["bins"],
                    "censored": needed["censored"],
                    "features": x,
                })

            lc = row.get("lc_auto", {})
            _, vbm = _vbm_result(row)
            lc_accurate = _finite(lc) and abs(float(lc["value"]) - reference["value"]) <= reference["tolerance"]
            vbm_accurate = _finite(vbm) and abs(float(vbm["value"]) - reference["value"]) <= reference["tolerance"]
            winner = "neither"
            if lc_accurate and not vbm_accurate:
                winner = "lcbinint"
            elif vbm_accurate and not lc_accurate:
                winner = "vbm"
            elif lc_accurate and vbm_accurate:
                winner = "lcbinint" if int(lc["elapsed_ns"]) <= int(vbm["elapsed_ns"]) else "vbm"
            engine[winner] += 1
            records[-1]["engine_winner"] = winner

    # Engine labels are attached once per source/profile row.  Copying to the
    # last grid record prevents double-counting while retaining the same
    # feature representation used by the resolution models.
    engine_records = [record for record in records if "engine_winner" in record]
    per_grid = {}
    for grid in sorted({record["grid"] for record in records}):
        grid_records = [record for record in records if record["grid"] == grid]
        model = fit_upper_resolution_model(grid_records, args.resolution_quantile)
        model["grouped_validation"] = grouped_resolution_validation(
            grid_records, args.resolution_quantile
        )
        per_grid[grid] = model
    engine_model = fit_engine_model(engine_records)
    engine_model["grouped_validation"] = grouped_engine_validation(engine_records)
    return {
        "completed_cases": cases,
        "trusted_reference_counts": dict(sorted(confidence.items())),
        "resolution_label_counts": dict(sorted(censored.items())),
        "engine_winner_counts": dict(sorted(engine.items())),
        "resolution_models": per_grid,
        "engine_model": engine_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--consensus-factor", type=float, default=2.0)
    parser.add_argument("--vbm-reference-max-magnification", type=float, default=1e3)
    parser.add_argument("--resolution-quantile", type=float, default=0.95)
    args = parser.parse_args()
    summary = analyze(args)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

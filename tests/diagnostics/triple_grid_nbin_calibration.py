#!/usr/bin/env python3
"""Explore triple nbin fits from fixed Cartesian sweeps.

Only fixed-nbin Cartesian rows are used.  A label is accepted when its two
highest resolutions converge at abs=1e-4 + rel=1e-3 and every later fixed
resolution agrees with that tail.  This deliberately cannot read an auto
result, preventing selector feedback into the calibration labels.  Its
polygonal distance feature is not bit-identical to runtime's refined caustic
distance, so this output is diagnostic rather than a production selector.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


ABS_TOL, REL_TOL, QUANTILE = 1e-4, 1e-3, .985
BUCKETS = (8, 12, 16, 24, 32, 50, 64, 80, 100, 128, 160, 200, 256)
FEATURE_NAMES = (
    "log10_point_magnification", "log10_rho", "log10_q", "log10_q2",
    "log10_sep2_over_s", "cos_angle", "log10_distance_over_rho",
    "near_caustic_strength", "limb_c")


def finite(row: dict[str, Any]) -> bool:
    return "value" in row and math.isfinite(float(row["value"]))


def label(row: dict[str, Any]) -> int | None:
    sequence = [x for x in row["fixed_sequences"]["cartesian"] if finite(x)]
    if len(sequence) < 2:
        return None
    reference = float(sequence[-1]["value"])
    tolerance = ABS_TOL + REL_TOL * max(abs(reference), 1.)
    if abs(reference - float(sequence[-2]["value"])) > 2. * tolerance:
        return None
    for index, candidate in enumerate(sequence):
        if all(abs(float(other["value"]) - reference) <= tolerance for other in sequence[index:]):
            return int(candidate["bins"])
    return None


def features(case: dict[str, Any], row: dict[str, Any]) -> list[float]:
    distance = float(row["caustic_distance_over_rho"])
    return [
        math.log10(max(abs(float(row["point_magnification"])), 1.)),
        math.log10(max(float(case["source_radius"]), 1e-12)),
        math.log10(max(float(case["mass_ratio"]), 1e-12)),
        math.log10(max(float(case["tertiary_mass_ratio"]), 1e-12)),
        math.log10(max(float(case["tertiary_separation"]) / float(case["separation"]), 1e-12)),
        math.cos(float(case["tertiary_angle"])),
        math.log10(max(distance, 1e-3)), max(0., 2. - min(distance, 2.)),
        float(row["limb_c"])]


def load(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("case-*.json")):
        doc = json.loads(path.read_text())
        if "error" in doc:
            continue
        for row in doc["rows"]:
            required = label(row)
            if required is not None:
                records.append({"case": doc["case"], "row": row, "required": required})
    return records


def fit(records: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([features(r["case"], r["row"]) for r in records])
    y = np.log2(np.asarray([r["required"] for r in records], float))
    mean, std = x.mean(axis=0), x.std(axis=0); std[std < 1e-10] = 1.
    design = np.column_stack((np.ones(len(x)), (x - mean) / std))
    def objective(beta: np.ndarray) -> float:
        residual = y - design @ beta
        loss = np.where(residual >= 0., QUANTILE * residual, (QUANTILE - 1.) * residual)
        return float(loss.mean() + 1e-4 * np.dot(beta[1:], beta[1:]))
    result = minimize(objective, np.linalg.lstsq(design, y, rcond=None)[0], method="Powell")
    return {"feature_mean": mean.tolist(), "feature_std": std.tolist(),
            "coefficients": result.x.tolist(), "rows": len(records),
            "optimizer_success": bool(result.success)}


def predict(model: dict[str, Any], record: dict[str, Any], safety: float) -> int:
    x = np.asarray(features(record["case"], record["row"]))
    beta = np.asarray(model["coefficients"])
    z = beta[0] + np.dot(beta[1:], (x - np.asarray(model["feature_mean"])) / np.asarray(model["feature_std"]))
    value = safety * 2. ** float(z)
    chosen = next((bucket for bucket in BUCKETS if value <= bucket), BUCKETS[-1])
    distance = float(record["row"]["caustic_distance_over_rho"])
    if distance <= .5 + 1e-9:
        chosen = max(chosen, 200)
    elif distance <= 1.0 + 1e-9:
        chosen = max(chosen, 128)
    return chosen


def report(model: dict[str, Any], records: list[dict[str, Any]], safety: float) -> dict[str, Any]:
    predicted = np.asarray([predict(model, row, safety) for row in records])
    required = np.asarray([row["required"] for row in records])
    return {"rows": len(records), "underpredictions": int(np.sum(predicted < required)),
            "median_over_required": float(np.median(predicted / required)),
            "p95_over_required": float(np.quantile(predicted / required, .95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, validation = load(args.train_dir), load(args.validation_dir)
    model = fit(train)
    candidates = np.arange(1., 3.01, .05)
    safety = next((float(value) for value in candidates
                   if report(model, validation, float(value))["underpredictions"] == 0), 3.0)
    payload = {"feature_names": FEATURE_NAMES, "buckets": BUCKETS, "quantile": QUANTILE,
               "safety_factor": safety, "model": model,
               "training": report(model, train, safety),
               "validation": report(model, validation, safety)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

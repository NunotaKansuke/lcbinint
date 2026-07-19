#!/usr/bin/env python3
"""Calibrate a one-shot triple-lens finite-source ``nbin`` selector.

The sampler is deliberately independent of ``nbin='auto'``.  It labels each
source position with the lowest fixed Cartesian resolution whose remaining
convergence sequence stays within the frozen tolerance of a high-resolution
tail.  Use distinct seeds for discovery and validation; a fit is accepted only
when the holdout set has no under-predicted uncensored rows.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import lcbinint


BINS = (8, 12, 16, 24, 32, 50, 64, 80, 100, 128, 160, 200, 256)
ABS_TOL = 1.0e-4
REL_TOL = 1.0e-3
SAFETY_FACTOR = 1.40
QUANTILE = 0.985
FEATURE_NAMES = (
    "log10_point_magnification",
    "log10_rho",
    "log10_q",
    "log10_q2",
    "log10_sep2_over_s",
    "cos_angle",
    "log10_distance_over_rho",
    "near_caustic_strength",
    "limb_c",
)


@dataclass(frozen=True)
class Case:
    case_id: int
    separation: float
    mass_ratio: float
    secondary_mass_ratio: float
    secondary_separation: float
    secondary_angle: float
    source_radius: float


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(math.log10(low), math.log10(high)))


def cases(count: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    anchors = (
        (0.7, 1e-3, 1e-4, 0.2, 0.0, 3e-4),
        (0.9, 1e-2, 1e-3, 0.5, 0.7, 1e-3),
        (1.0, 1e-3, 1e-4, 0.5, 1.2, 1e-3),
        (1.2, 1e-3, 1e-3, 0.1, 0.0, 2e-3),
        (1.5, 0.1, 1e-2, 0.7, 2.1, 3e-3),
        (2.0, 1e-2, 1e-4, 1.0, 1.5, 1e-2),
    )
    output: list[Case] = []
    for case_id in range(count):
        if case_id < len(anchors):
            values = anchors[case_id]
        else:
            values = (
                log_uniform(rng, 0.4, 3.0),
                log_uniform(rng, 1e-4, 0.5),
                log_uniform(rng, 1e-5, 0.1),
                log_uniform(rng, 0.08, 1.5),
                float(rng.uniform(0.0, math.pi)),
                log_uniform(rng, 1e-4, 1e-2),
            )
        output.append(Case(case_id, *values))
    return output


def caustic_points(case: Case, bins: int) -> np.ndarray:
    curve = lcbinint.LightCurve(lens="triple", options=lcbinint.Options(caustic_bins=bins))
    branches = curve.caustics(
        s=case.separation, q=case.mass_ratio, q2=case.secondary_mass_ratio,
        sep2=case.secondary_separation, ang=case.secondary_angle, n_points=bins)
    return np.concatenate([
        np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        for x, y in zip(branches.x, branches.y) if len(x) >= 2
    ])


def sample_points(case: Case, points: np.ndarray, count: int, seed: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed + 8191 * (case.case_id + 1))
    factors = (0.0, 0.2, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    output = []
    for index in range(count):
        anchor = points[int(rng.integers(len(points)))]
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        offset = factors[index % len(factors)] * case.source_radius
        output.append((
            float(anchor[0] + offset * math.cos(angle)),
            float(anchor[1] + offset * math.sin(angle)),
        ))
    return output


def nearest_distance(point: np.ndarray, caustics: np.ndarray) -> float:
    # Caustic samples are dense (>= 800 per branch); this is only a feature
    # generator, while the runtime uses its refined distance calculation.
    return float(np.sqrt(np.min(np.sum((caustics - point) ** 2, axis=1))))


def solver(bins: int, limb_c: float) -> lcbinint.LightCurve:
    return lcbinint.LightCurve(
        lens="triple",
        options=lcbinint.Options(
            source_bins=bins, polar_source_bins=bins, nbin=bins,
            inverse_ray_grid="cartesian",
            point_source_threshold=20.0, hexadecapole_threshold=0.0,
            adaptive_hex_threshold=0.0, max_source_bins=bins),
        limb_darkening=lcbinint.LimbDarkening.linear(limb_c))


def _evaluate_direct(case: Case, x: float, y: float, limb_c: float, bins: tuple[int, ...]) -> list[dict]:
    params = dict(
        t0=0.0, tE=1.0, u0=y, alpha=0.0, s=case.separation, q=case.mass_ratio,
        q2=case.secondary_mass_ratio, sep2=case.secondary_separation,
        ang=case.secondary_angle, rho=case.source_radius)
    output = []
    for value in bins:
        info = solver(value, limb_c).info([x], params)
        output.append({
            "bins": value,
            "magnification": float(info.magnifications[0]),
            "method": info.finite_source_method_names[0],
            "point_magnification": float(info.point_source_magnifications[0]),
        })
    return output


def _evaluation_worker(connection, case: Case, x: float, y: float, limb_c: float, bins: tuple[int, ...]) -> None:
    try:
        connection.send({"sequence": _evaluate_direct(case, x, y, limb_c, bins)})
    except Exception as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def evaluate(
    case: Case, x: float, y: float, limb_c: float, bins: tuple[int, ...], timeout: float
) -> list[dict]:
    """Isolate occasional pathological triple solves from the sweep process."""
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_evaluation_worker, args=(child, case, x, y, limb_c, bins), daemon=True)
    process.start()
    child.close()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        parent.close()
        return []
    try:
        payload = parent.recv() if parent.poll() else {"error": f"worker exited {process.exitcode}"}
    finally:
        parent.close()
    return payload.get("sequence", [])


def required_bins(sequence: list[dict]) -> tuple[int, bool]:
    valid = [row for row in sequence if math.isfinite(row["magnification"])]
    if len(valid) < 2:
        return 0, True
    if all(row["method"] in {"point_source", "hexadecapole"} for row in valid):
        return 0, False
    reference = valid[-1]["magnification"]
    previous = valid[-2]["magnification"]
    tolerance = ABS_TOL + REL_TOL * max(abs(reference), 1.0)
    if abs(reference - previous) > 2.0 * tolerance:
        return valid[-1]["bins"], True
    for index, row in enumerate(valid):
        if all(abs(candidate["magnification"] - reference) <= tolerance for candidate in valid[index:]):
            return row["bins"], False
    return valid[-1]["bins"], True


def feature(case: dict, row: dict) -> list[float]:
    distance_ratio = row["distance_over_rho"]
    return [
        math.log10(max(abs(row["point_magnification"]), 1.0)),
        math.log10(max(case["source_radius"], 1e-12)),
        math.log10(max(case["mass_ratio"], 1e-12)),
        math.log10(max(case["secondary_mass_ratio"], 1e-12)),
        math.log10(max(case["secondary_separation"] / case["separation"], 1e-12)),
        math.cos(case["secondary_angle"]),
        math.log10(max(distance_ratio, 1e-3)),
        max(0.0, 2.0 - min(distance_ratio, 2.0)),
        row["limb_c"],
    ]


def run_sweep(args: argparse.Namespace) -> None:
    rows = []
    for case in cases(args.cases, args.seed):
        points = caustic_points(case, args.caustic_bins)
        for x, y in sample_points(case, points, args.points_per_case, args.seed):
            distance = nearest_distance(np.asarray((x, y)), points)
            for limb_c in args.limb_coefficients:
                sequence = evaluate(case, x, y, limb_c, args.bins, args.solve_timeout)
                if not sequence:
                    continue
                required, censored = required_bins(sequence)
                rows.append({
                    "case": asdict(case), "source_x": x, "source_y": y, "limb_c": limb_c,
                    "distance_over_rho": distance / case.source_radius,
                    "point_magnification": sequence[-1]["point_magnification"],
                    "sequence": sequence, "required_bins": required, "censored": censored,
                })
        print(f"case {case.case_id + 1}/{args.cases}", flush=True)
    args.output.write_text(json.dumps({
        "configuration": {
            "cases": args.cases, "points_per_case": args.points_per_case,
            "seed": args.seed, "caustic_bins": args.caustic_bins,
            "bins": args.bins, "limb_coefficients": args.limb_coefficients,
        },
        "rows": rows,
    }, indent=2) + "\n")


def fit(records: list[dict]) -> dict:
    usable = [row for row in records if row["required_bins"] > 0 and not row["censored"]]
    x_raw = np.asarray([feature(row["case"], row) for row in usable], dtype=float)
    y = np.log2(np.asarray([row["required_bins"] for row in usable], dtype=float))
    mean, std = x_raw.mean(axis=0), x_raw.std(axis=0)
    std[std < 1e-10] = 1.0
    design = np.column_stack((np.ones(len(x_raw)), (x_raw - mean) / std))
    def objective(beta: np.ndarray) -> float:
        residual = y - design @ beta
        loss = np.where(residual >= 0.0, QUANTILE * residual, (QUANTILE - 1.0) * residual)
        return float(loss.mean() + 1e-4 * np.dot(beta[1:], beta[1:]))
    result = minimize(objective, np.linalg.lstsq(design, y, rcond=None)[0], method="Powell")
    return {"feature_mean": mean.tolist(), "feature_std": std.tolist(),
            "coefficients": result.x.tolist(), "rows": len(usable),
            "optimizer_success": bool(result.success)}


def predict(model: dict, row: dict) -> int:
    x = np.asarray(feature(row["case"], row))
    y = model["coefficients"][0] + np.dot(
        np.asarray(model["coefficients"][1:]),
        (x - np.asarray(model["feature_mean"])) / np.asarray(model["feature_std"]))
    value = SAFETY_FACTOR * 2.0 ** float(y)
    for bucket in BINS:
        if value <= bucket:
            selected = bucket
            break
    else:
        selected = BINS[-1]
    if row["distance_over_rho"] <= 0.5 + 1.0e-9:
        selected = max(selected, 100)
    return selected


def run_fit(args: argparse.Namespace) -> None:
    train = json.loads(args.train.read_text())["rows"]
    validation = json.loads(args.validation.read_text())["rows"]
    model = fit(train)
    report = {}
    for name, rows in (("training", train), ("validation", validation)):
        usable = [row for row in rows if row["required_bins"] > 0 and not row["censored"]]
        predictions = np.asarray([predict(model, row) for row in usable])
        required = np.asarray([row["required_bins"] for row in usable])
        report[name] = {"rows": len(usable), "underpredictions": int(np.sum(predictions < required)),
                        "median_over_required": float(np.median(predictions / required))}
    payload = {"feature_names": FEATURE_NAMES, "bins": BINS, "safety_factor": SAFETY_FACTOR,
               "model": model, "report": report}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--cases", type=int, default=32)
    sweep.add_argument("--points-per-case", type=int, default=8)
    sweep.add_argument("--seed", type=int, default=20260719)
    sweep.add_argument("--caustic-bins", type=int, default=800)
    sweep.add_argument("--solve-timeout", type=float, default=5.0)
    sweep.add_argument("--bins", type=lambda x: tuple(int(v) for v in x.split(",")), default=BINS)
    sweep.add_argument("--limb-coefficients", type=lambda x: tuple(float(v) for v in x.split(",")), default=(0.0, 0.5))
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--train", type=Path, required=True)
    fit_parser.add_argument("--validation", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sweep":
        run_sweep(args)
    else:
        run_fit(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the balanced parameter corpus for the pure-kernel benchmark.

The physical parameters are independent log-uniform draws.  Source positions
are accepted into equal-width bins of the *measured* caustic distance d/rho,
so the label is not confused with the requested normal-step factor.

The output is deliberately shaped like the existing ``speed_discovery`` rows:
``bench_grid_vs_vbm_pure_kernel.py`` can therefore reuse its warm-up search and
cache-warm timing protocol with ``--search-missing`` and ``--route-filter all``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# This explicitly loads the repository build before importing the helpers or
# lcbinint itself.  The benchmark and the reference-generation pass then use
# the same native extension.
import bench_grid_vs_vbm_pure_kernel as pure  # noqa: E402

import bench_grid_vs_vbm_dark as base  # noqa: E402


CONFIG_S_RANGE = (0.2, 4.0)
CONFIG_Q_RANGE = (1.0e-4, 1.0)
CONFIG_RHO_RANGE = (3.0e-5, 1.0)
D_BINS = (
    (0.0, 0.4),
    (0.4, 0.8),
    (0.8, 1.2),
    (1.2, 1.6),
    (1.6, 2.0),
)
REFERENCE_INDICES = tuple(base.REFERENCE_INDICES)
BLOCK_EPOCHS = int(base.BLOCK_EPOCHS)
BLOCK_SPAN_IN_RADII = float(base.BLOCK_SPAN_IN_RADII)
# The benchmark target is at worst 1e-4.  The 1e-6/1e-7 pair is therefore
# deliberately conservative while avoiding the pathological long-tail cost
# seen for BinaryMagDark at 1e-10 near a caustic.  The pilot corpus measured a
# maximum self-consistency gap of 6.3e-7, well below the 1e-5 acceptance floor.
REFERENCE_RELATIVE_LEVELS = (1.0e-6, 1.0e-7)


def _log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _branches(curve, s, q):
    caustics = curve.caustics(s=float(s), q=float(q))
    branches = []
    for xs, ys in zip(caustics.x, caustics.y):
        points = np.column_stack((
            np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        ))
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) >= 3:
            branches.append(points)
    return branches


def _actual_d_over_rho(curve, s, q, rho, x, y):
    info = curve.info(
        [float(x)],
        t0=0.0,
        tE=1.0,
        u0=float(y),
        alpha=0.0,
        s=float(s),
        q=float(q),
        rho=float(rho),
    )
    distance = float(info.caustic_distances[0])
    return distance / float(rho) if math.isfinite(distance) else None


def _candidate(rng, branches, rho, low, high):
    branch_index = int(rng.integers(len(branches)))
    branch = branches[branch_index]
    vertex_index = int(rng.integers(len(branch)))
    point = branch[vertex_index]
    following = branch[(vertex_index + 1) % len(branch)]
    tangent = following - point
    norm = math.hypot(float(tangent[0]), float(tangent[1]))
    if norm <= 0.0:
        return None
    unit_tangent = tangent / norm
    normal = np.array([-unit_tangent[1], unit_tangent[0]])
    factor = float(rng.uniform(low, high))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    # Keep a small tangential jitter so equal d/rho bins do not reuse the same
    # caustic vertices, while the acceptance test below controls the actual
    # distance that enters the final analysis.
    jitter = float(rng.uniform(-0.25, 0.25)) * float(rho)
    position = point + sign * factor * float(rho) * normal
    position = position + jitter * unit_tangent
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "branch_index": branch_index,
        "vertex_index": vertex_index,
        "requested_factor": factor,
    }


def _sample_positions(curve, s, q, rho, rng, max_attempts):
    positions = []
    branches = _branches(curve, s, q)
    if not branches:
        raise RuntimeError("no caustic branches")
    for bin_index, (low, high) in enumerate(D_BINS):
        accepted = None
        for attempt in range(1, max_attempts + 1):
            candidate = _candidate(rng, branches, rho, low, high)
            if candidate is None:
                continue
            actual = _actual_d_over_rho(
                curve, s, q, rho, candidate["x"], candidate["y"]
            )
            if actual is None:
                continue
            in_bin = low <= actual < high
            if bin_index == len(D_BINS) - 1:
                in_bin = low <= actual <= high
            if in_bin:
                accepted = {
                    **candidate,
                    "d_bin_index": bin_index,
                    "d_bin_low": low,
                    "d_bin_high": high,
                    "actual_d_over_rho": float(actual),
                    "attempt": attempt,
                }
                break
        if accepted is None:
            raise RuntimeError(
                f"could not fill d/rho bin {bin_index} for "
                f"s={s:g}, q={q:g}, rho={rho:g} after {max_attempts} attempts"
            )
        positions.append(accepted)
    return positions


def _times(x, rho):
    return np.linspace(
        float(x) - 0.5 * BLOCK_SPAN_IN_RADII * float(rho),
        float(x) + 0.5 * BLOCK_SPAN_IN_RADII * float(rho),
        BLOCK_EPOCHS,
    )


def _references(row, profile_c):
    times = _times(row["x"], row["rho"])
    reference_times = times[list(REFERENCE_INDICES)]
    values = {level: [] for level in REFERENCE_RELATIVE_LEVELS}
    for level in REFERENCE_RELATIVE_LEVELS:
        vbm = base._new_vbm(float(profile_c), level)
        for x in reference_times:
            values[level].append(
                base._vbm_one(vbm, row, float(x), float(profile_c))
            )
    fine = np.asarray(values[REFERENCE_RELATIVE_LEVELS[-1]], dtype=float)
    coarse = np.asarray(values[REFERENCE_RELATIVE_LEVELS[0]], dtype=float)
    gaps = np.abs(fine - coarse) / np.maximum(np.abs(fine), 1.0)
    return {
        str(index): {
            "value": float(fine[position]),
            "uncertainty": float(gaps[position]),
            "status": "ok",
        }
        for position, index in enumerate(REFERENCE_INDICES)
    }, float(np.max(gaps))


def _make_row(config_id, config, position, profile, profile_c):
    row = {
        "case_id": int(config_id),
        "configuration_id": int(config_id),
        "s": float(config["s"]),
        "q": float(config["q"]),
        "rho": float(config["rho"]),
        "x": float(position["x"]),
        "y": float(position["y"]),
        "u0": float(position["y"]),
        "alpha": 0.0,
        "profile": profile,
        "limb_darkening_c": float(profile_c),
        # The benchmark selector uses this value as the stratum key.  It is
        # the equal-width bin centre, not a claim about achieved geometry.
        "intended_distance_factor": float(
            0.5 * (position["d_bin_low"] + position["d_bin_high"])
        ),
        "actual_d_over_rho": float(position["actual_d_over_rho"]),
        "d_bin_index": int(position["d_bin_index"]),
        "d_bin_low": float(position["d_bin_low"]),
        "d_bin_high": float(position["d_bin_high"]),
        "branch_index": int(position["branch_index"]),
        "vertex_index": int(position["vertex_index"]),
        "requested_factor": float(position["requested_factor"]),
        "sampling_attempt": int(position["attempt"]),
        "block_epochs": BLOCK_EPOCHS,
        # Empty engines deliberately force the pure benchmark to run its
        # warm-up search with --search-missing.  No old corpus ladder is used.
        "engines": [],
    }
    references, floor = _references(row, profile_c)
    row["references"] = references
    row["reference_floor"] = floor
    row["magnification"] = float(np.median([
        entry["value"] for entry in references.values()
    ]))
    return row


def _new_distance_curve():
    """Create the geometry helper inside the process that will use it."""
    return pure.lcbinint.LightCurve(
        lens="binary",
        options=pure.lcbinint.Options(
            coordinates="vbm",
            caustic_bins=1400,
        ),
    )


def _generate_configuration(task):
    """Generate one configuration and all of its source/profile rows.

    Each task owns its RNG and native LightCurve object.  This makes the
    generation embarrassingly parallel without changing the sampled values
    or the order in which the main process writes the final corpus.
    """
    config_id, config, seed, profiles, max_attempts = task
    rng = np.random.default_rng(int(seed))
    distance_curve = _new_distance_curve()
    positions = _sample_positions(
        distance_curve,
        config["s"], config["q"], config["rho"], rng,
        max_attempts,
    )
    rows = []
    for position in positions:
        for profile in profiles:
            profile_c = 0.0 if profile == "uniform" else 0.5
            rows.append(_make_row(
                config_id, config, position, profile, profile_c
            ))
    return {"configuration_id": config_id, **config}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument(
        "--workers", type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel configuration workers used only during corpus generation",
    )
    parser.add_argument("--profiles", nargs="+", default=["uniform", "linear"],
                        choices=["uniform", "linear"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.workers < 1:
        parser.error("--workers must be positive")

    rng = np.random.default_rng(args.seed)
    child_seeds = np.random.SeedSequence(args.seed).spawn(args.cases)
    tasks = []
    for config_id in range(args.cases):
        config = {
            "s": _log_uniform(rng, *CONFIG_S_RANGE),
            "q": _log_uniform(rng, *CONFIG_Q_RANGE),
            "rho": _log_uniform(rng, *CONFIG_RHO_RANGE),
        }
        task_seed = int(child_seeds[config_id].generate_state(
            1, dtype=np.uint64
        )[0])
        tasks.append((
            config_id, config, task_seed, tuple(args.profiles),
            args.max_attempts,
        ))

    configs = []
    rows = []
    pool = None
    if args.workers == 1:
        generated = map(_generate_configuration, tasks)
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(processes=args.workers)
        generated = pool.imap(_generate_configuration, tasks)
    try:
        for config, config_rows in generated:
            configs.append(config)
            rows.extend(config_rows)
            completed = len(configs)
            if completed % 10 == 0 or completed == args.cases:
                print(f"generated {completed}/{args.cases} configurations",
                      flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    else:
        if pool is not None:
            pool.close()
    finally:
        if pool is not None:
            pool.join()

    manifest = {
        "generator": "generate_controlled_pure_kernel.py",
        "seed": args.seed,
        "cases": args.cases,
        "profiles": list(args.profiles),
        "parameter_sampling": {
            "distribution": "independent_log_uniform",
            "s": list(CONFIG_S_RANGE),
            "q": list(CONFIG_Q_RANGE),
            "rho": list(CONFIG_RHO_RANGE),
        },
        "d_over_rho_bins": [list(item) for item in D_BINS],
        "positions_per_configuration": len(D_BINS),
        "source_positions": args.cases * len(D_BINS),
        "reference_indices": list(REFERENCE_INDICES),
        "block_epochs": BLOCK_EPOCHS,
        "block_span_in_radii": BLOCK_SPAN_IN_RADII,
        "reference_relative_levels": list(REFERENCE_RELATIVE_LEVELS),
        "generation_workers": args.workers,
        "build_extension": str(pure.BUILD_EXTENSION),
        "configurations": configs,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "rows.json").write_text(json.dumps({"rows": rows}, indent=2))
    for index, row in enumerate(rows):
        (args.output / f"block-{index:05d}.json").write_text(
            json.dumps({"rows": [row]}, indent=2)
        )

    floors = np.asarray([row["reference_floor"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_d_over_rho"] for row in rows], dtype=float)
    print(json.dumps({
        "rows": len(rows),
        "source_positions": args.cases * len(D_BINS),
        "actual_d_over_rho_min": float(np.min(actual)),
        "actual_d_over_rho_max": float(np.max(actual)),
        "reference_floor_max": float(np.max(floors)),
        "reference_floor_p99": float(np.percentile(floors, 99)),
        "usable_for_1e-3": int(np.sum(floors <= 1.0e-4)),
        "usable_for_1e-4": int(np.sum(floors <= 1.0e-5)),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()

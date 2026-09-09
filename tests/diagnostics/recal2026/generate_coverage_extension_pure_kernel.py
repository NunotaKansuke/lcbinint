#!/usr/bin/env python3
"""Generate a q--rho coverage extension for the pure-kernel corpus.

The canonical corpus is an independent log-uniform sample.  This utility is
for a separate diagnostic extension: it preserves the existing configurations
and adds one configuration in every currently empty q--rho plotting bin,
then fills any requested remainder with independent log-uniform draws.  The
result must not be described as an iid corpus without noting this design.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

import generate_controlled_pure_kernel as canonical


Q_RHO_BINS = 12


def _plot_edges(
    configurations: list[dict], field: str, default_low: float, default_high: float
) -> np.ndarray:
    del configurations, field
    # q and rho are sampled within the protocol ranges.  Do not decade-floor
    # below those ranges: that would create bins that no valid corpus row can
    # occupy and would make the coverage extension chase impossible data.
    return np.geomspace(default_low, default_high, Q_RHO_BINS + 1)


def _bin_index(value: float, edges: np.ndarray) -> int:
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if value == edges[-1]:
        index -= 1
    return index


def _occupied(configurations: list[dict], q_edges, rho_edges) -> set[tuple[int, int]]:
    occupied = set()
    for config in configurations:
        q_index = _bin_index(float(config["q"]), q_edges)
        rho_index = _bin_index(float(config["rho"]), rho_edges)
        if not (0 <= q_index < Q_RHO_BINS and 0 <= rho_index < Q_RHO_BINS):
            raise ValueError("base configuration lies outside q--rho plotting bins")
        occupied.add((q_index, rho_index))
    return occupied


def _draw_in_bin(rng, edges: np.ndarray, index: int) -> float:
    return float(math.exp(rng.uniform(
        math.log(float(edges[index])), math.log(float(edges[index + 1]))
    )))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=96)
    parser.add_argument("--case-id-start", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--fixed-s", type=float,
        help="fix s for a cheap q--rho coverage diagnostic; omit for log-uniform s",
    )
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel workers for corpus generation; does not affect timing",
    )
    parser.add_argument("--profiles", nargs="+", default=["uniform", "linear"],
                        choices=["uniform", "linear"])
    args = parser.parse_args()

    if args.cases < 1:
        parser.error("--cases must be positive")
    if args.case_id_start < 0:
        parser.error("--case-id-start must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.fixed_s is not None and not (
        canonical.CONFIG_S_RANGE[0] <= args.fixed_s <= canonical.CONFIG_S_RANGE[1]
    ):
        parser.error("--fixed-s must lie within the canonical s range")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")

    base_manifest = json.loads(
        (args.base_corpus / "manifest.json").read_text()
    )
    configurations = list(base_manifest.get("configurations", ()))
    if not configurations:
        raise ValueError("base corpus manifest has no configurations")

    # Match the protocol-clamped q--rho plotting range.  The generic plotting
    # helper historically decade-floored rho below 3e-5; the companion
    # protocol-edge plot wrapper avoids that invalid extra range.
    q_edges = _plot_edges(configurations, "q", *canonical.CONFIG_Q_RANGE)
    rho_edges = _plot_edges(configurations, "rho", *canonical.CONFIG_RHO_RANGE)
    occupied = _occupied(configurations, q_edges, rho_edges)
    missing = [
        (q_index, rho_index)
        for rho_index in range(Q_RHO_BINS)
        for q_index in range(Q_RHO_BINS)
        if (q_index, rho_index) not in occupied
    ]
    if args.cases < len(missing):
        raise SystemExit(
            f"need at least {len(missing)} cases to fill the {len(missing)} "
            "empty q--rho bins"
        )

    rng = np.random.default_rng(args.seed)
    child_seeds = np.random.SeedSequence(args.seed).spawn(args.cases)
    tasks = []
    targeted = len(missing)
    for local_index in range(args.cases):
        case_id = args.case_id_start + local_index
        s = (
            float(args.fixed_s)
            if args.fixed_s is not None
            else canonical._log_uniform(rng, *canonical.CONFIG_S_RANGE)
        )
        if local_index < targeted:
            q_index, rho_index = missing[local_index]
            q = _draw_in_bin(rng, q_edges, q_index)
            rho = _draw_in_bin(rng, rho_edges, rho_index)
            extension_mode = "targeted_empty_qrho_bin"
        else:
            q = canonical._log_uniform(rng, *canonical.CONFIG_Q_RANGE)
            rho = canonical._log_uniform(rng, *canonical.CONFIG_RHO_RANGE)
            q_index = _bin_index(q, q_edges)
            rho_index = _bin_index(rho, rho_edges)
            extension_mode = "independent_log_uniform_remainder"
        config = {
            "s": s,
            "q": q,
            "rho": rho,
            "q_rho_extension_mode": extension_mode,
            "q_bin_index": q_index,
            "rho_bin_index": rho_index,
        }
        task_seed = int(child_seeds[local_index].generate_state(
            1, dtype=np.uint64
        )[0])
        tasks.append((
            case_id, config, task_seed, tuple(args.profiles), args.max_attempts,
        ))

    generated_configurations = []
    rows = []
    pool = None
    if args.workers == 1:
        generated = map(canonical._generate_configuration, tasks)
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(processes=args.workers)
        generated = pool.imap(canonical._generate_configuration, tasks)
    try:
        for index, (config, config_rows) in enumerate(generated, 1):
            generated_configurations.append(config)
            rows.extend(config_rows)
            if index % 10 == 0 or index == len(tasks):
                print(f"generated {index}/{len(tasks)} configurations", flush=True)
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
        "generator": "generate_coverage_extension_pure_kernel.py",
        "seed": args.seed,
        "cases": args.cases,
        "case_id_start": args.case_id_start,
        "case_id_end": args.case_id_start + args.cases,
        "profiles": list(args.profiles),
        "parameter_sampling": {
            "distribution": (
                "coverage_completed_extension_fixed_s"
                if args.fixed_s is not None
                else "coverage_completed_extension"
            ),
            "base_distribution": "independent_log_uniform",
            "s": (
                [float(args.fixed_s), float(args.fixed_s)]
                if args.fixed_s is not None
                else list(canonical.CONFIG_S_RANGE)
            ),
            "q": list(canonical.CONFIG_Q_RANGE),
            "rho": list(canonical.CONFIG_RHO_RANGE),
            "q_rho_bins": Q_RHO_BINS,
            "q_edges": q_edges.tolist(),
            "rho_edges": rho_edges.tolist(),
            "edge_policy": "protocol_clamped_range",
        },
        "coverage": {
            "base_cases": len(configurations),
            "base_occupied_bins": len(occupied),
            "base_empty_bins": len(missing),
            "targeted_cases": targeted,
            "remainder_cases": args.cases - targeted,
            "targeted_bins": [list(item) for item in missing],
        },
        "positions_per_configuration": len(canonical.D_BINS),
        "source_positions": args.cases * len(canonical.D_BINS),
        "reference_indices": list(canonical.REFERENCE_INDICES),
        "block_epochs": canonical.BLOCK_EPOCHS,
        "block_span_in_radii": canonical.BLOCK_SPAN_IN_RADII,
        "reference_relative_levels": list(canonical.REFERENCE_RELATIVE_LEVELS),
        "generation_workers": args.workers,
        "build_extension": str(canonical.pure.BUILD_EXTENSION),
        "configurations": generated_configurations,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "rows.json").write_text(json.dumps({"rows": rows}, indent=2))
    for index, row in enumerate(rows):
        (args.output / f"block-{index:05d}.json").write_text(
            json.dumps({"rows": [row]}, indent=2)
        )

    all_occupied = occupied | _occupied(generated_configurations, q_edges, rho_edges)
    print(json.dumps({
        "output": str(args.output),
        "cases": args.cases,
        "rows": len(rows),
        "base_occupied_bins": len(occupied),
        "combined_occupied_bins": len(all_occupied),
        "combined_empty_bins": 144 - len(all_occupied),
        "targeted_cases": targeted,
        "remainder_cases": args.cases - targeted,
    }, indent=2))


if __name__ == "__main__":
    main()

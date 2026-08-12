#!/usr/bin/env python3
"""Generate a controlled corpus after an Nbin=400 VBM cross-check.

VBMicrolensing is used as an external timing/reference implementation, but
its self-convergence can become unreliable at very high magnification.  This
generator therefore keeps a configuration only when at least one lcbinint
grid method at Nbin=400 agrees with the VBM reference at every reference epoch
for both requested tolerances and both source profiles.  Rejected candidate
configurations are replaced by later log-uniform draws.
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

import generate_controlled_pure_kernel as base_generator  # noqa: E402
import bench_grid_vs_vbm_pure_kernel as pure  # noqa: E402


TARGETS = (1.0e-3, 1.0e-4)
REFERENCE_INDICES = tuple(base_generator.REFERENCE_INDICES)
PROFILES = ("uniform", "linear")
AUDIT_SOURCE_BINS = 400
# The stricter requested tolerance implies the looser one for the same fixed
# Nbin value, so the audit evaluates only 1e-4 and uses it as the gate for
# both benchmark tolerances.
AUDIT_TARGET = 1.0e-4


def _audit_row(row, curve_cache):
    """Return audit metadata and whether VBM is usable for this row."""

    times = base_generator._times(row["x"], row["rho"])
    times = times[list(REFERENCE_INDICES)]
    reference = pure.base._reference(row)
    params = pure.base._params(row)
    audit = {}
    accepted = True
    for target in (AUDIT_TARGET,):
        count = len(reference)
        method_errors = {
            "cartesian": [float("inf")] * count,
            "polar": [float("inf")] * count,
        }
        reference_floor = float(row["reference_floor"])
        reference_floor_ok = reference_floor <= 0.1 * target

        def evaluate(method_name, method, indices):
            curve = curve_cache[(float(row["limb_darkening_c"]), target)]
            indices = np.asarray(indices, dtype=int)
            result = pure._evaluate(
                curve,
                times[indices],
                params,
                method,
                [AUDIT_SOURCE_BINS] * len(indices),
            )
            values = np.asarray(result["magnification"], dtype=float)
            errors = np.abs(values - reference[indices]) / np.maximum(
                np.abs(reference[indices]), 1.0
            )
            full = np.asarray(method_errors[method_name], dtype=float)
            full[indices] = errors
            method_errors[method_name] = full.tolist()
            return errors

        # A poor VBM self-consistency floor is already enough to reject the
        # row.  More importantly, screen the epoch with the largest VBM
        # magnification before evaluating all four epochs.  High-magnification
        # candidates are exactly where Nbin=400 can be expensive and where a
        # VBM reference is most likely to fail this compatibility test.
        peak_index = int(np.argmax(np.abs(reference)))
        peak_error = float("inf")
        peak_method = None
        if reference_floor_ok:
            cartesian_peak = evaluate(
                "cartesian", pure.warmup.CARTESIAN, [peak_index]
            )[0]
            if np.isfinite(cartesian_peak) and cartesian_peak <= target:
                peak_error = float(cartesian_peak)
                peak_method = "cartesian"
            else:
                polar_peak = evaluate(
                    "polar", pure.warmup.POLAR, [peak_index]
                )[0]
                if np.isfinite(polar_peak) and polar_peak <= target:
                    peak_error = float(polar_peak)
                    peak_method = "polar"

        if peak_method is not None:
            # Finish with the method that passed the cheap screen.  If it
            # fails at another reference epoch, evaluate the other method as
            # well so that the per-epoch minimum remains exact.
            full_errors = evaluate(
                peak_method,
                (pure.warmup.CARTESIAN if peak_method == "cartesian"
                 else pure.warmup.POLAR),
                range(count),
            )
            if not (np.all(np.isfinite(full_errors))
                    and np.max(full_errors) <= target):
                other = (
                    ("polar", pure.warmup.POLAR)
                    if peak_method == "cartesian"
                    else ("cartesian", pure.warmup.CARTESIAN)
                )
                evaluate(other[0], other[1], range(count))
        # If the peak screen failed for both methods, no other epoch can make
        # the row acceptable; retain only the measured peak errors for audit
        # provenance and avoid an expensive full-grid call.
        for name in ("cartesian", "polar"):
            method_errors.setdefault(name, [float("inf")] * len(reference))
        stacked = np.asarray(list(method_errors.values()), dtype=float)
        best = np.min(stacked, axis=0)
        max_error = float(np.max(best))
        best_method = [
            ("cartesian" if method_errors["cartesian"][index]
             <= method_errors["polar"][index] else "polar")
            for index in range(best.size)
        ]
        target_ok = reference_floor_ok and max_error <= target
        audit[str(target)] = {
            "source_bins": AUDIT_SOURCE_BINS,
            "cartesian_errors": method_errors["cartesian"],
            "polar_errors": method_errors["polar"],
            "best_errors": best.tolist(),
            "best_methods": best_method,
            "max_best_error": max_error,
            "reference_floor": float(row["reference_floor"]),
            "reference_floor_ok": bool(reference_floor_ok),
            "status": (
                "ok" if target_ok
                else ("reference_floor" if not reference_floor_ok
                      else "vbm_grid400_mismatch")
            ),
        }
        accepted = accepted and target_ok
    return audit, accepted


def _generate_candidate(task):
    candidate_id, config, seed, profiles, max_attempts = task
    rng = np.random.default_rng(int(seed))
    distance_curve = base_generator._new_distance_curve()
    positions = base_generator._sample_positions(
        distance_curve,
        config["s"],
        config["q"],
        config["rho"],
        rng,
        max_attempts,
    )
    curves = {
        (0.0 if profile == "uniform" else 0.5, AUDIT_TARGET):
        pure.base._curve(0.0 if profile == "uniform" else 0.5, AUDIT_TARGET)
        for profile in profiles
    }
    rows = []
    audits = []
    accepted = True
    for position in positions:
        position_rows = []
        for profile in profiles:
            profile_c = 0.0 if profile == "uniform" else 0.5
            row = base_generator._make_row(
                candidate_id, config, position, profile, profile_c
            )
            audit, row_ok = _audit_row(row, curves)
            row["reference_audit"] = audit
            position_rows.append(row)
            audits.append(audit)
            accepted = accepted and row_ok
            if not row_ok:
                break
        rows.extend(position_rows)
        if not accepted:
            break
    return {
        "candidate_id": int(candidate_id),
        "configuration": {"configuration_id": int(candidate_id), **config},
        "rows": rows,
        "accepted": bool(accepted),
        "max_audit_error": float(max(
            audit[str(target)]["max_best_error"]
            for audit in audits
        for target in (AUDIT_TARGET,)
        )),
    }


def _audit_existing_configuration(task):
    """Audit one already-generated configuration without resampling it."""

    configuration_id, rows = task
    curves = {
        (0.0 if profile == "uniform" else 0.5, AUDIT_TARGET):
        pure.base._curve(0.0 if profile == "uniform" else 0.5, AUDIT_TARGET)
        for profile in PROFILES
    }
    audited_rows = []
    accepted = True
    max_error = 0.0
    for original in rows:
        row = dict(original)
        audit, row_ok = _audit_row(row, curves)
        row["reference_audit"] = audit
        audited_rows.append(row)
        accepted = accepted and row_ok
        max_error = max(
            max_error,
            float(audit[str(AUDIT_TARGET)]["max_best_error"]),
        )
        if not row_ok:
            break
    return {
        "configuration_id": int(configuration_id),
        "rows": audited_rows,
        "accepted": bool(accepted),
        "max_audit_error": max_error,
    }


def _log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=160)
    parser.add_argument("--candidate-cases", type=int)
    parser.add_argument(
        "--base-corpus", type=Path,
        help=(
            "reuse this corpus first; only configurations failing the "
            "Nbin=400 VBM audit are replaced"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument(
        "--workers", type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES),
                        choices=list(PROFILES))
    args = parser.parse_args()
    if args.cases < 1 or args.workers < 1:
        parser.error("cases and workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    base_manifest = None
    retained = []
    rejected_base_ids = []
    if args.base_corpus is not None:
        base_manifest = json.loads(
            (args.base_corpus / "manifest.json").read_text()
        )
        base_rows = json.loads(
            (args.base_corpus / "rows.json").read_text()
        )["rows"]
        base_configs = base_manifest.get("configurations", [])
        if len(base_configs) != args.cases:
            parser.error(
                f"--base-corpus contains {len(base_configs)} configurations, "
                f"but --cases={args.cases}"
            )
        rows_by_configuration = {}
        for row in base_rows:
            rows_by_configuration.setdefault(
                int(row["configuration_id"]), []
            ).append(row)
        audit_tasks = [
            (int(config["configuration_id"]), rows_by_configuration[int(
                config["configuration_id"]
            )])
            for config in base_configs
        ]
        if args.workers == 1:
            audited = map(_audit_existing_configuration, audit_tasks)
            audit_pool = None
        else:
            context = mp.get_context("spawn")
            audit_pool = context.Pool(processes=args.workers)
            audited = audit_pool.imap(_audit_existing_configuration, audit_tasks)
        try:
            for result in audited:
                if result["accepted"]:
                    config = dict(base_configs[result["configuration_id"]])
                    retained.append((config, result["rows"], "base"))
                else:
                    rejected_base_ids.append(result["configuration_id"])
                completed = len(retained) + len(rejected_base_ids)
                print(
                    f"audit {completed}/{args.cases}: "
                    f"retained={len(retained)} "
                    f"rejected={len(rejected_base_ids)} "
                    f"max_audit_error={result['max_audit_error']:.3e}",
                    flush=True,
                )
        except BaseException:
            if audit_pool is not None:
                audit_pool.terminate()
            raise
        else:
            if audit_pool is not None:
                audit_pool.close()
        finally:
            if audit_pool is not None:
                audit_pool.join()

    needed = args.cases - len(retained)
    candidate_cases = args.candidate_cases
    if candidate_cases is None:
        candidate_cases = max(3 * needed, needed)
    if candidate_cases < needed:
        parser.error("candidate-cases must be at least the number of replacements")

    rng = np.random.default_rng(args.seed)
    child_seeds = np.random.SeedSequence(args.seed).spawn(candidate_cases)
    candidate_tasks = []
    for candidate_id in range(candidate_cases):
        config = {
            "s": _log_uniform(rng, *base_generator.CONFIG_S_RANGE),
            "q": _log_uniform(rng, *base_generator.CONFIG_Q_RANGE),
            "rho": _log_uniform(rng, *base_generator.CONFIG_RHO_RANGE),
        }
        task_seed = int(child_seeds[candidate_id].generate_state(
            1, dtype=np.uint64
        )[0])
        candidate_tasks.append((
            candidate_id,
            config,
            task_seed,
            tuple(args.profiles),
            args.max_attempts,
        ))

    if args.workers == 1:
        generated = map(_generate_candidate, candidate_tasks)
        pool = None
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(processes=args.workers)
        generated = pool.imap(_generate_candidate, candidate_tasks)

    accepted_replacements = []
    rejected_candidates = 0
    try:
        for result in generated:
            if result["accepted"] and len(accepted_replacements) < needed:
                accepted_replacements.append((
                    dict(result["configuration"]),
                    result["rows"],
                    "replacement",
                ))
            else:
                rejected_candidates += 1
            print(
                f"candidate {result['candidate_id'] + 1}/{candidate_cases}: "
                f"replacements={len(accepted_replacements)}/{needed} "
                f"rejected={rejected_candidates} max_audit_error="
                f"{result['max_audit_error']:.3e}",
                flush=True,
            )
            if len(accepted_replacements) >= needed:
                break
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

    if len(accepted_replacements) < needed:
        raise RuntimeError(
            f"only accepted {len(accepted_replacements)} of {needed} replacements; "
            f"increase --candidate-cases"
        )

    accepted_configs = []
    accepted_rows = []
    for config, rows, _source in retained + accepted_replacements:
        new_id = len(accepted_configs)
        config = dict(config)
        config["configuration_id"] = new_id
        accepted_configs.append(config)
        for original in rows:
            row = dict(original)
            row["case_id"] = new_id
            row["configuration_id"] = new_id
            accepted_rows.append(row)

    manifest = {
        "generator": "generate_filtered_controlled_pure_kernel.py",
        "seed": args.seed,
        "cases": args.cases,
        "candidate_cases": candidate_cases,
        "rejected_candidates": len(rejected_base_ids) + rejected_candidates,
        "base_corpus": (
            str(args.base_corpus) if args.base_corpus is not None else None
        ),
        "retained_base_configurations": len(retained),
        "replaced_base_configurations": len(rejected_base_ids),
        "new_candidates_tested": (
            0 if needed == 0 else rejected_candidates + len(accepted_replacements)
        ),
        "rejected_base_configuration_ids": rejected_base_ids,
        "profiles": list(args.profiles),
        "parameter_sampling": {
            "distribution": "independent_log_uniform",
            "s": list(base_generator.CONFIG_S_RANGE),
            "q": list(base_generator.CONFIG_Q_RANGE),
            "rho": list(base_generator.CONFIG_RHO_RANGE),
        },
        "d_over_rho_bins": [list(item) for item in base_generator.D_BINS],
        "positions_per_configuration": len(base_generator.D_BINS),
        "source_positions": args.cases * len(base_generator.D_BINS),
        "reference_indices": list(REFERENCE_INDICES),
        "block_epochs": base_generator.BLOCK_EPOCHS,
        "block_span_in_radii": base_generator.BLOCK_SPAN_IN_RADII,
        "reference_relative_levels": list(
            base_generator.REFERENCE_RELATIVE_LEVELS
        ),
        "reference_audit": {
            "source_bins": AUDIT_SOURCE_BINS,
            "criterion": "min(cartesian, polar) relative error against VBM",
            "targets": [AUDIT_TARGET],
            "support_proven_ignored": True,
        },
        "generation_workers": args.workers,
        "build_extension": str(pure.BUILD_EXTENSION),
        "configurations": accepted_configs,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "rows.json").write_text(
        json.dumps({"rows": accepted_rows}, indent=2)
    )
    for index, row in enumerate(accepted_rows):
        (args.output / f"block-{index:05d}.json").write_text(
            json.dumps({"rows": [row]}, indent=2)
        )
    print(json.dumps({
        "rows": len(accepted_rows),
        "source_positions": args.cases * len(base_generator.D_BINS),
        "accepted_configurations": len(accepted_configs),
        "retained_base_configurations": len(retained),
        "replaced_base_configurations": len(rejected_base_ids),
        "rejected_candidates": len(rejected_base_ids) + rejected_candidates,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()

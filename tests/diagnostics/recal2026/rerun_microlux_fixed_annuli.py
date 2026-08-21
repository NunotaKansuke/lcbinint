#!/usr/bin/env python3
"""Rerun only microLUX with one fixed linear limb-darkening annulus count.

The full JAX/native comparison is already expensive and its rows are
immutable for this experiment.  This runner reuses those input rows and
re-measures only the microLUX callable, including the same compile warm-up,
forward block timing, and source-trajectory ``dA/dt`` timing contract.

For a VBM-matched microLUX audit, use ``--n-annuli 80``.  Uniform sources do
not use annuli; their fixed-annulus field is therefore ``null``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
DEFAULT_INPUT = (
    ROOT
    / "tests/diagnostics/results/recal2026/"
    / "jax_microlux_12800_final_adaptive_v6_20260819/results.json"
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "bench_jax_microlux_12800", SCRIPT.parent / "bench_jax_microlux_12800.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_value(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _finite(value, bench):
    return bench._finite(value)


def _finite_max(values, bench):
    if values is None:
        return None
    finite = [float(value) for value in values if _finite(value, bench)]
    return max(finite) if finite else None


def _pass_status(errors, references, uncertainties, target, bench):
    checked = [
        index
        for index, reference in enumerate(references)
        if _finite(reference, bench)
    ]
    certified = bool(
        checked
        and uncertainties is not None
        and len(uncertainties) == len(references)
        and all(
            _finite(value, bench)
            and float(value)
            <= bench.REFERENCE_UNCERTAINTY_FRACTION * float(target)
            for value in uncertainties
        )
    )
    passed = bool(
        certified
        and all(
            _finite(errors[index], bench)
            and float(errors[index]) <= float(target)
            for index in checked
        )
    )
    return certified, passed


def _row_result(row, executor, bench, args):
    profile = str(row["profile"])
    target = float(row["target"])
    profile_c = float(row.get("limb_darkening_c", 0.0))
    times = bench._times(row)
    references, uncertainties, reference_source = bench._reference_bundle(row)
    # The merged JAX result stores the corpus uncertainty under the compact
    # ``reference_uncertainty`` field.  Prefer the richer source-row field
    # when it was joined, but retain the saved result's field when the input
    # is already a merged report.
    if uncertainties is None:
        uncertainties = row.get("reference_uncertainty")
    if reference_source is None and uncertainties is not None:
        reference_source = "saved_corpus_vbm_fine_reltol_1e-7"
    stage = "select"
    try:
        selection = executor.select(
            profile,
            profile_c,
            target,
            row,
            times,
            references,
            uncertainties,
            args.forward_timeout,
        )
        selected_n_annuli = selection["selected_n_annuli"]
        stage = "timing"
        timing = executor.timed(
            profile,
            profile_c,
            target,
            selected_n_annuli,
            row,
            times,
            args.repeats,
            args.forward_timeout,
            args.derivative_timeout,
            False,
        )
    except bench._DeadlineExceeded as error:
        return {
            "status": "timeout",
            "error": f"{type(error).__name__}: {error}",
            "timeout_stage": stage,
            "case_id": int(row["case_id"]),
            "profile": profile,
            "target": target,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "batch_epochs": len(times),
        }
    except Exception as error:  # noqa: BLE001
        return {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "error_stage": stage,
            "case_id": int(row["case_id"]),
            "profile": profile,
            "target": target,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "batch_epochs": len(times),
        }

    values = timing["forward_values"]
    derivatives = timing["dA_dt_values"]
    errors = [
        bench._relative_error(value, reference)
        for value, reference in zip(values, references)
    ]
    certified, passed = _pass_status(
        errors, references, uncertainties, target, bench
    )
    derivative_errors = (
        [
            bench._relative_error(value, reference)
            for value, reference in zip(derivatives, references)
        ]
        if derivatives is not None
        else [None] * len(references)
    )
    epoch_count = len(times)
    forward_block = timing["forward_block_seconds"]
    derivative_block = timing["dA_dt_block_seconds"]
    selected_calibration = selection.get("selected_entry") or {}
    warnings = list(selected_calibration.get("warnings", ()))
    warnings.extend(timing.get("warnings", ()))
    fixed_n_annuli = (
        selected_n_annuli if profile == "linear" else None
    )
    return {
        "status": "completed",
        "case_id": int(row["case_id"]),
        "input_status": row.get("status"),
        "profile": profile,
        "target": target,
        "limb_darkening_c": profile_c,
        "s": float(row["s"]),
        "q": float(row["q"]),
        "rho": float(row["rho"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "d_over_rho": float(row.get("d_over_rho", float("nan"))),
        "times": times.tolist(),
        "batch_epochs": int(epoch_count),
        "reference": references,
        "reference_uncertainty": uncertainties,
        "reference_available": all(
            _finite(value, bench) for value in references
        ),
        "reference_certified_for_target": certified,
        "reference_source": reference_source,
        "reference_floor": row.get("accuracy_reference_floor"),
        "microlux_tol": target,
        "microlux_retol": target,
        "microlux_strategy": list(bench._microlux_strategy(target)),
        "microlux_accuracy_status": selection["status"],
        "microlux_selection_mode": selected_calibration.get(
            "selection_mode", selection["status"]
        ),
        "microlux_fixed_n_annuli": fixed_n_annuli,
        "microlux_n_annuli": selected_n_annuli,
        "microlux_selected_n_annuli": selected_n_annuli,
        "microlux_warmup_seconds": float(selection["warmup_seconds"]),
        "microlux_values": values.tolist(),
        "microlux_dA_dt": (
            None if derivatives is None else derivatives.tolist()
        ),
        "microlux_relative_errors": errors,
        "microlux_max_relative_error": _finite_max(errors, bench),
        "microlux_dA_dt_relative_errors": derivative_errors,
        "microlux_dA_dt_max_relative_error": _finite_max(
            derivative_errors, bench
        ),
        "microlux_passes_reference": passed,
        "microlux_forward_block_seconds": float(forward_block),
        "microlux_forward_seconds_per_epoch": float(
            forward_block / epoch_count
        ),
        "microlux_dA_dt_block_seconds": (
            None if derivative_block is None else float(derivative_block)
        ),
        "microlux_dA_dt_seconds_per_epoch": (
            None
            if derivative_block is None
            else float(derivative_block / epoch_count)
        ),
        "microlux_forward_samples_seconds": [
            float(value) for value in timing["forward_samples_seconds"]
        ],
        "microlux_dA_dt_samples_seconds": (
            None
            if timing["dA_dt_samples_seconds"] is None
            else [
                float(value)
                for value in timing["dA_dt_samples_seconds"]
            ]
        ),
        "microlux_forward_first_after_warmup_seconds": float(
            timing["forward_first_after_warmup_seconds"]
        ),
        "microlux_dA_dt_first_after_warmup_seconds": float(
            timing["dA_dt_first_after_warmup_seconds"]
        ),
        "microlux_dA_dt_timeout": bool(timing["dA_dt_timeout"]),
        "microlux_dA_dt_timeout_message": timing[
            "dA_dt_timeout_message"
        ],
        "microlux_warnings": warnings,
        "microlux_budget_exhausted": bool(
            selected_calibration.get("budget_exhausted", False)
            or timing.get("budget_exhausted", False)
        ),
    }


def _stats(values, bench):
    finite = [float(value) for value in values if _finite(value, bench)]
    if not finite:
        return {"count": 0}
    finite.sort()
    middle = finite[len(finite) // 2]
    return {
        "count": len(finite),
        "median_seconds": middle,
        "minimum_seconds": min(finite),
        "maximum_seconds": max(finite),
    }


def _summary(results, bench):
    grouped = {}
    for result in results:
        key = f"{result['profile']}:target={float(result['target']):g}"
        grouped.setdefault(key, []).append(result)
    summary = {}
    for key, rows in sorted(grouped.items()):
        completed = [row for row in rows if row.get("status") == "completed"]
        summary[key] = {
            "jobs": len(rows),
            "epochs": sum(int(row.get("batch_epochs", 0)) for row in rows),
            "status_counts": {
                status: sum(row.get("status") == status for row in rows)
                for status in sorted({row.get("status") for row in rows})
            },
            "fixed_n_annuli_counts": {
                str(value): sum(
                    row.get("microlux_fixed_n_annuli") == value
                    for row in completed
                )
                for value in sorted(
                    {
                        row.get("microlux_fixed_n_annuli")
                        for row in completed
                    },
                    key=lambda value: -1 if value is None else int(value),
                )
            },
            "accuracy_certified_count": sum(
                bool(row.get("reference_certified_for_target"))
                for row in completed
            ),
            "accuracy_pass_count": sum(
                bool(row.get("microlux_passes_reference"))
                for row in completed
            ),
            "accuracy_fail_count": sum(
                row.get("reference_certified_for_target") is True
                and row.get("microlux_passes_reference") is False
                for row in completed
            ),
            "forward": _stats(
                [
                    row.get("microlux_forward_block_seconds")
                    for row in completed
                ],
                bench,
            ),
            "dA_dt": _stats(
                [
                    row.get("microlux_dA_dt_block_seconds")
                    for row in completed
                ],
                bench,
            ),
            "dA_dt_timeout_count": sum(
                bool(row.get("microlux_dA_dt_timeout"))
                for row in completed
            ),
            "budget_exhausted_count": sum(
                bool(row.get("microlux_budget_exhausted"))
                for row in completed
            ),
            "max_relative_error": _stats(
                [
                    row.get("microlux_max_relative_error")
                    for row in completed
                ],
                bench,
            ),
            "dA_dt_max_relative_error": _stats(
                [
                    row.get("microlux_dA_dt_max_relative_error")
                    for row in completed
                ],
                bench,
            ),
        }
    return summary


def _split_lane_target(value):
    return f"{float(value):.0e}".replace("+", "p").replace("-", "m")


def _run_split_lanes(args):
    bench = _load_benchmark_module()
    profiles = tuple(args.profiles or ("uniform", "linear"))
    targets = tuple(float(value) for value in args.targets)
    if args.parallel_workers < 1:
        raise SystemExit("--parallel-workers must be positive")
    if args.max_jobs is not None and args.parallel_workers > 1:
        raise SystemExit(
            "--max-jobs cannot be combined with --parallel-workers > 1"
        )
    _, source_rows = bench._load_rows(args.input)

    def _case_chunks(profile, target):
        selected = bench._select_rows(
            source_rows,
            (profile,),
            (target,),
            args.case_id_min,
            args.case_id_max,
            None,
        )
        case_ids = sorted({int(row["case_id"]) for row in selected})
        if not case_ids:
            return ()
        count = min(args.parallel_workers, len(case_ids))
        chunks = []
        for index in range(count):
            start = (index * len(case_ids)) // count
            stop = ((index + 1) * len(case_ids)) // count
            if start == stop:
                continue
            chunks.append((case_ids[start], case_ids[stop - 1] + 1))
        return tuple(chunks)

    with __import__("tempfile").TemporaryDirectory(
        prefix="microlux-fixed-annuli-"
    ) as temp_dir:
        temp = Path(temp_dir)
        tasks = []
        for profile in profiles:
            for target in targets:
                chunks = _case_chunks(profile, target)
                for chunk_index, (case_min, case_max) in enumerate(chunks):
                    part = temp / (
                        f"{profile}-target-{_split_lane_target(target)}-"
                        f"chunk-{chunk_index:02d}.json"
                    )
                    command = [
                        sys.executable,
                        str(SCRIPT),
                        "--input",
                        str(args.input),
                        "--output",
                        str(part),
                        "--profiles",
                        profile,
                        "--targets",
                        str(target),
                        "--n-annuli",
                        str(args.n_annuli),
                        "--repeats",
                        str(args.repeats),
                        "--forward-timeout",
                        str(args.forward_timeout),
                        "--derivative-timeout",
                        str(args.derivative_timeout),
                        "--case-id-min",
                        str(case_min),
                        "--case-id-max",
                        str(case_max),
                    ]
                    tasks.append((profile, target, chunk_index, command, part))

        def _run_task(task):
            subprocess.run(task[3], check=True)
            return task

        if args.parallel_workers == 1:
            completed_tasks = [_run_task(task) for task in tasks]
        else:
            # Each child owns its JAX executable cache.  This keeps the timed
            # callable isolated while allowing independent geometry chunks to
            # make progress concurrently on a multi-core host.
            with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
                completed_tasks = list(pool.map(_run_task, tasks))
        paths = [task[4] for task in completed_tasks]

        payloads = [json.loads(path.read_text()) for path in paths]
        results = [
            result
            for payload in payloads
            for result in payload.get("results", ())
        ]
        merged = dict(payloads[0])
        merged["configuration"] = dict(payloads[0]["configuration"])
        merged["configuration"]["profiles"] = list(profiles)
        merged["configuration"]["targets"] = list(targets)
        merged["configuration"]["split_lanes"] = True
        merged["configuration"]["parallel_workers"] = args.parallel_workers
        merged["results"] = results
        merged["compile_records"] = {
            "microlux": [
                record
                for payload in payloads
                for record in payload.get("compile_records", {}).get(
                    "microlux", ()
                )
            ]
        }
        merged["summary"] = _summary(results, bench)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(merged, indent=2) + "\n")
        print(json.dumps(merged["summary"], indent=2), flush=True)
        print(f"saved {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("uniform", "linear"))
    parser.add_argument(
        "--targets", nargs="+", type=float, default=(1.0e-3, 1.0e-4)
    )
    parser.add_argument("--case-id-min", type=int)
    parser.add_argument("--case-id-max", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--n-annuli", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--forward-timeout", type=float, default=300.0)
    parser.add_argument("--derivative-timeout", type=float, default=300.0)
    parser.add_argument("--split-lanes", action="store_true")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "number of independent case chunks to run concurrently under "
            "--split-lanes; timed samples remain per-process (default: 1)"
        ),
    )
    args = parser.parse_args()

    if args.n_annuli < 10:
        raise SystemExit("--n-annuli must be at least 10")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise SystemExit("--max-jobs must be positive")
    if args.parallel_workers < 1:
        raise SystemExit("--parallel-workers must be positive")

    profiles = tuple(args.profiles or ("uniform", "linear"))
    targets = tuple(float(value) for value in args.targets)
    if args.split_lanes and len(profiles) * len(targets) > 1:
        _run_split_lanes(args)
        return

    bench = _load_benchmark_module()
    input_payload, source_rows = bench._load_rows(args.input)
    rows = bench._select_rows(
        source_rows,
        args.profiles,
        args.targets,
        args.case_id_min,
        args.case_id_max,
        args.max_jobs,
    )
    if not rows:
        raise SystemExit("no rows selected")

    executor = bench.MicroLuxBatchExecutor(
        max_annuli=args.n_annuli,
        fixed_n_annuli=args.n_annuli,
    )
    print(
        f"selected {len(rows)} jobs, "
        f"{len(rows) * len(bench.REFERENCE_INDICES)} batched epochs; "
        f"microLUX fixed linear n_annuli={args.n_annuli}, "
        f"repeats={args.repeats}",
        flush=True,
    )
    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        result = _row_result(row, executor, bench, args)
        results.append(result)
        if index == 1 or index % 25 == 0 or result["status"] != "completed":
            if result["status"] == "completed":
                detail = (
                    f"forward={result['microlux_forward_block_seconds']:.6g}s "
                    f"dA/dt={result['microlux_dA_dt_block_seconds']} "
                    f"maxerr={result['microlux_max_relative_error']:.3g} "
                    f"pass={result['microlux_passes_reference']}"
                )
            else:
                detail = result.get("error", "")
            print(f"[{index}/{len(rows)}] {result['status']} {detail}", flush=True)

    try:
        import microlux

        microlux_path = str(Path(microlux.__file__).resolve())
        microlux_commit = bench._checkout(microlux)
    except Exception:  # noqa: BLE001
        microlux_path = ""
        microlux_commit = ""

    payload = {
        "input": str(args.input),
        "timing_mode": "compiled_warm_microLUX_fixed_annuli_only",
        "derivative_mode": "source_trajectory_dA_dt_forward_mode_jvp",
        "reference_policy": (
            "reuse the saved VBM corpus reference and uncertainty; no JAX or "
            "native call is rerun"
        ),
        "configuration": {
            "profiles": list(profiles),
            "targets": list(targets),
            "reference_indices": list(bench.REFERENCE_INDICES),
            "batch_epochs": len(bench.REFERENCE_INDICES),
            "microlux_fixed_n_annuli": args.n_annuli,
            "microlux_n_annuli_policy": (
                "fixed requested n_annuli for every linear row; uniform rows "
                "do not use annuli"
            ),
            "microlux_tol_policy": "tol=target",
            "microlux_retol_policy": "retol=target",
            "microlux_strategy_policy": (
                "library default strategy at 1e-3; "
                "(60,60,120,240,480) for tighter targets"
            ),
            "forward_timeout_seconds": args.forward_timeout,
            "derivative_timeout_seconds": args.derivative_timeout,
            "repeats": args.repeats,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "split_lanes": False,
            "parallel_workers": args.parallel_workers,
        },
        "input_metadata": {
            "source_payload_keys": sorted(input_payload),
            "reference_indices": list(bench.REFERENCE_INDICES),
            "reference_epoch_count": len(bench.REFERENCE_INDICES),
            "nominal_epochs": len(rows) * len(bench.REFERENCE_INDICES),
        },
        "system": {
            "python": sys.version,
            "platform": __import__("platform").platform(),
            "microlux_path": microlux_path,
            "microlux_commit": microlux_commit,
        },
        "results": results,
        "compile_records": {
            "microlux": list(executor.compile_records.values())
        },
        "summary": _summary(results, bench),
        "elapsed_seconds": time.perf_counter() - started,
        "_bench_module": bench,
    }
    # Do not attempt to serialize the imported module; it is only useful to
    # the in-process summary helper.
    payload.pop("_bench_module", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

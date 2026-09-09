#!/usr/bin/env python3
"""Recompute only the VBM accuracy reference for an existing benchmark.

The input ``plot_parts`` contain the already measured lcbinint values, selected
Nbin values, kernel timings, and VBM timings.  This script leaves all of those
speed-comparison fields untouched and replaces only the VBM values used as the
accuracy reference.  It writes one resumable record per input job and derives
new result parts only after the reference pass is complete.

The work is intentionally job-level: one job contains the four reference
epochs used by the q--rho benchmark.  A small process scheduler gives each
job a point timeout while keeping the number of concurrent VBM evaluations
bounded.  Progress is printed after every finished job so that the launcher
log can be followed with ``tail -f``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np


# The sibling benchmark module installs the explicit repository build and
# exposes the exact VBM API helpers used by the original benchmark.
import bench_grid_vs_vbm_pure_kernel as benchmark  # noqa: E402


REFERENCE_INDICES = tuple(int(value) for value in benchmark.REFERENCE_INDICES)


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_worker(row, reltol, connection):
    """Evaluate one job's four VBM epochs and stream point results."""

    try:
        profile_c = float(row["limb_darkening_c"])
        times = benchmark.base._times(row)
        vbm = benchmark.base._new_vbm(profile_c, float(reltol))
        for local_index, epoch_index in enumerate(REFERENCE_INDICES):
            value = benchmark._forced_vbm_one(
                vbm, row, float(times[epoch_index]), profile_c
            )
            connection.send({
                "kind": "point",
                "index": int(local_index),
                "value": float(value),
            })
        connection.send({"kind": "done"})
    except BaseException as error:  # noqa: BLE001
        try:
            connection.send({
                "kind": "error",
                "error": f"{type(error).__name__}: {error}",
            })
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def _new_active(task, reltol, context, point_timeout):
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_reference_worker,
        args=(task["row"], float(reltol), send),
    )
    process.start()
    send.close()
    return {
        "task": task,
        "receive": receive,
        "process": process,
        "values": [None] * len(REFERENCE_INDICES),
        "statuses": ["unrequested"] * len(REFERENCE_INDICES),
        "next_message_deadline": time.monotonic() + float(point_timeout),
    }


def _finish_active(active, status=None, error=None):
    process = active["process"]
    receive = active["receive"]
    if process.is_alive():
        process.terminate()
    process.join(5.0)
    if status is not None:
        for index, value in enumerate(active["statuses"]):
            if value == "unrequested":
                active["statuses"][index] = status
    payload = {
        "part": active["task"]["part"],
        "result_index": active["task"]["result_index"],
        "case_id": active["task"]["row"].get("case_id"),
        "profile": active["task"]["row"].get("profile"),
        "target": active["task"]["row"].get("target"),
        "d_over_rho": active["task"]["row"].get("d_over_rho"),
        "values": active["values"],
        "statuses": active["statuses"],
        "status": (
            "completed"
            if all(value == "completed" for value in active["statuses"])
            else (status or "error")
        ),
    }
    if error:
        payload["error"] = error
    receive.close()
    return payload


def _poll_active(active, point_timeout):
    """Drain messages and return (done, record) for one active job."""

    receive = active["receive"]
    process = active["process"]
    changed = False
    while receive.poll():
        message = receive.recv()
        kind = message.get("kind")
        if kind == "point":
            index = int(message["index"])
            active["values"][index] = float(message["value"])
            active["statuses"][index] = "completed"
            active["next_message_deadline"] = (
                time.monotonic() + float(point_timeout)
            )
            changed = True
        elif kind == "error":
            return True, _finish_active(
                active, status="error", error=message.get("error")
            )
        elif kind == "done":
            process.join(5.0)
            return True, _finish_active(active)
    if all(value == "completed" for value in active["statuses"]):
        return True, _finish_active(active)
    if not process.is_alive():
        return True, _finish_active(
            active,
            status="error",
            error=f"worker exited with code {process.exitcode}",
        )
    if time.monotonic() > active["next_message_deadline"]:
        return True, _finish_active(
            active,
            status="timeout",
            error=f"point timeout after {point_timeout:g} s",
        )
    return False, None


def _error(value, reference):
    if value is None or reference is None:
        return None
    value = float(value)
    reference = float(reference)
    if not np.isfinite(value) or not np.isfinite(reference):
        return None
    return float(abs(value - reference) / max(abs(reference), 1.0))


def _sample_value(grid, index, required):
    if required is None:
        return None
    sample = grid.get("samples", {}).get(str(index))
    if not sample:
        return None
    for bins, value in zip(sample.get("nbin", []), sample.get("magnification", [])):
        if int(bins) == int(required):
            return value
    return None


def _rewrite_result(original, values, statuses, reltol):
    """Replace accuracy references while preserving all speed measurements."""

    result = copy.deepcopy(original)
    references = [None if value is None else float(value) for value in values]
    result["reference"] = references
    result["reference_status"] = list(statuses)
    result["reference_mode"] = "accuracy_only_vbm_call_at_reltol"
    result["accuracy_reference_reltol"] = float(reltol)

    for grid in result.get("grid", {}).values():
        errors = []
        for index, required in enumerate(grid.get("required", [])):
            value = _sample_value(grid, index, required)
            errors.append(_error(value, references[index]))
        grid["errors"] = errors
        grid["vbm_reference_errors"] = list(errors)

    vbm = result.get("vbm", {})
    vbm["reference_values"] = list(references)
    vbm["reference_status"] = list(statuses)
    vbm["accuracy_reference_reltol"] = float(reltol)
    vbm["timing_reference_errors"] = [
        _error(value, references[index])
        for index, value in enumerate(vbm.get("timing_values", []))
    ]
    result["vbm"] = vbm

    chosen_errors = []
    mismatches = []
    for index, chosen_grid in enumerate(result.get("chosen_grid", [])):
        if chosen_grid is None:
            chosen_errors.append(None)
            mismatches.append(None)
            continue
        error = result["grid"][chosen_grid]["errors"][index]
        chosen_errors.append(error)
        mismatches.append(
            None if error is None else bool(error > float(result["target"]))
        )
    result["chosen_vbm_errors"] = chosen_errors
    result["vbm_mismatch"] = mismatches
    return result


def _load_tasks(input_root: Path, record_root: Path):
    tasks = []
    for input_path in sorted(input_root.glob("*/results.json")):
        payload = json.loads(input_path.read_text())
        part = input_path.parent.name
        for result_index, row in enumerate(payload.get("results", [])):
            if row.get("status") != "completed":
                raise RuntimeError(f"input result is not completed: {input_path}#{result_index}")
            record_path = record_root / part / f"{result_index:04d}.json"
            tasks.append({
                "part": part,
                "input_path": str(input_path),
                "result_index": int(result_index),
                "row": row,
                "record_path": str(record_path),
            })
    return tasks


def _read_completed_record(path: Path):
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if record.get("status") != "completed":
        return None
    if any(value != "completed" for value in record.get("statuses", [])):
        return None
    return record


def _write_progress(path, progress):
    _atomic_json(path, progress)


def _finalize_parts(input_root, derived_root, record_root, reltol):
    derived_root.mkdir(parents=True, exist_ok=True)
    total = 0
    failures = []
    for input_path in sorted(input_root.glob("*/results.json")):
        part = input_path.parent.name
        payload = json.loads(input_path.read_text())
        updated = []
        for index, original in enumerate(payload.get("results", [])):
            record_path = record_root / part / f"{index:04d}.json"
            record = _read_completed_record(record_path)
            if record is None:
                failures.append(f"{part}#{index}")
                updated.append(original)
                continue
            updated.append(
                _rewrite_result(
                    original,
                    record["values"],
                    record["statuses"],
                    reltol,
                )
            )
            total += 1
        derived_payload = copy.deepcopy(payload)
        derived_payload["results"] = updated
        derived_payload["reference_mode"] = "accuracy_only_full_recheck"
        derived_payload["accuracy_reference_reltol"] = float(reltol)
        derived_payload["speed_comparison_preserved"] = True
        derived_payload["source_results"] = str(input_path)
        _atomic_json(derived_root / part / "results.json", derived_payload)
    if failures:
        raise RuntimeError(
            f"cannot finalize {len(failures)} incomplete records; first={failures[:5]}"
        )
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reltol", type=float, default=1.0e-6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--point-timeout", type=float, default=300.0)
    parser.add_argument(
        "--max-jobs", type=int,
        help="debug/smoke-test limit; omit for the complete input root",
    )
    parser.add_argument("--build-dir", type=Path, default=benchmark.BUILD_DIR)
    args = parser.parse_args()

    if args.reltol <= 0.0:
        raise SystemExit("--reltol must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.build_dir.resolve() != benchmark.BUILD_DIR.resolve():
        raise SystemExit("this script requires the repository build selected at import")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    record_root = output_root / "records"
    derived_root = output_root / "plot_parts_vbm_reltol_1e-6_reference"
    log_progress = output_root / "progress.json"
    manifest_path = output_root / "manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)

    extension = Path(benchmark.BUILD_EXTENSION).resolve()
    tasks = _load_tasks(input_root, record_root)
    if args.max_jobs is not None:
        if args.max_jobs < 1:
            raise SystemExit("--max-jobs must be positive")
        tasks = tasks[:args.max_jobs]
    total = len(tasks)
    total_epochs = total * len(REFERENCE_INDICES)
    completed_records = [
        _read_completed_record(Path(task["record_path"])) for task in tasks
    ]
    completed = sum(record is not None for record in completed_records)
    completed_epochs = sum(
        sum(status == "completed" for status in (record or {}).get("statuses", []))
        for record in completed_records
    )
    started = time.time()
    manifest = {
        "status": "running",
        "started_unix": started,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "derived_parts": str(derived_root),
        "reltol": float(args.reltol),
        "workers": int(args.workers),
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "point_timeout": float(args.point_timeout),
        "total_jobs": total,
        "total_reference_epochs": total_epochs,
        "completed_jobs_at_start": completed,
        "max_jobs": args.max_jobs,
        "build_extension": str(extension),
        "build_sha256": _sha256(extension),
        "speed_comparison_preserved": True,
        "reference_indices": list(REFERENCE_INDICES),
    }
    _write_progress(log_progress, {
        "status": "running",
        "completed_jobs": completed,
        "total_jobs": total,
        "completed_reference_epochs": completed_epochs,
        "total_reference_epochs": total_epochs,
        "failed_jobs": 0,
        "started_unix": started,
    })
    _atomic_json(manifest_path, manifest)
    print(
        f"selected {total} jobs / {total_epochs} reference epochs; "
        f"resume={completed}; workers={args.workers}; reltol={args.reltol:g}",
        flush=True,
    )

    pending = [
        task for task in tasks
        if _read_completed_record(Path(task["record_path"])) is None
    ]
    context = mp.get_context("fork")
    active = []
    finished = completed
    finished_epochs = completed_epochs
    failed = 0
    cursor = 0
    try:
        while cursor < len(pending) or active:
            while cursor < len(pending) and len(active) < args.workers:
                active.append(
                    _new_active(
                        pending[cursor], args.reltol, context, args.point_timeout
                    )
                )
                cursor += 1
            progressed = False
            for item in list(active):
                done, record = _poll_active(item, args.point_timeout)
                if not done:
                    continue
                active.remove(item)
                task = item["task"]
                record_path = Path(task["record_path"])
                _atomic_json(record_path, record)
                finished += 1
                finished_epochs += sum(
                    status == "completed" for status in record.get("statuses", [])
                )
                if record.get("status") != "completed":
                    failed += 1
                progress = {
                    "status": "running",
                    "completed_jobs": finished,
                    "total_jobs": total,
                    "completed_reference_epochs": finished_epochs,
                    "total_reference_epochs": total_epochs,
                    "failed_jobs": failed,
                    "last_part": task["part"],
                    "last_result_index": task["result_index"],
                    "last_case_id": task["row"].get("case_id"),
                    "elapsed_seconds": time.time() - started,
                }
                _write_progress(log_progress, progress)
                print(
                    f"[{finished}/{total}] "
                    f"epochs={progress['completed_reference_epochs']}/{total_epochs} "
                    f"case={task['row'].get('case_id')} "
                    f"profile={task['row'].get('profile')} "
                    f"target={float(task['row'].get('target')):g} "
                    f"d/rho={task['row'].get('d_over_rho')} "
                    f"status={record.get('status')}",
                    flush=True,
                )
                progressed = True
            if not progressed:
                time.sleep(0.1)
    except BaseException:
        for item in active:
            _finish_active(item, status="aborted", error="campaign interrupted")
        manifest["status"] = "interrupted"
        manifest["finished_unix"] = time.time()
        _atomic_json(manifest_path, manifest)
        raise

    if failed:
        manifest["status"] = "failed"
        manifest["failed_jobs"] = failed
        manifest["finished_unix"] = time.time()
        _atomic_json(manifest_path, manifest)
        _write_progress(log_progress, {
            "status": "failed",
            "completed_jobs": finished,
            "total_jobs": total,
            "failed_jobs": failed,
            "total_reference_epochs": total_epochs,
        })
        raise SystemExit(f"{failed} reference jobs failed; resume from {output_root}")

    if args.max_jobs is not None:
        manifest.update({
            "status": "partial_smoke",
            "completed_jobs": finished,
            "completed_reference_epochs": finished_epochs,
            "finished_unix": time.time(),
        })
        _atomic_json(manifest_path, manifest)
        _write_progress(log_progress, {
            "status": "partial_smoke",
            "completed_jobs": finished,
            "total_jobs": total,
            "completed_reference_epochs": finished_epochs,
            "total_reference_epochs": total_epochs,
            "failed_jobs": 0,
            "elapsed_seconds": time.time() - started,
        })
        print("partial smoke run complete; no derived parts finalized", flush=True)
        return

    derived_jobs = _finalize_parts(
        input_root, derived_root, record_root, float(args.reltol)
    )
    manifest.update({
        "status": "completed",
        "completed_jobs": finished,
        "completed_reference_epochs": total_epochs,
        "finished_unix": time.time(),
        "derived_jobs": derived_jobs,
    })
    _atomic_json(manifest_path, manifest)
    _write_progress(log_progress, {
        "status": "completed",
        "completed_jobs": finished,
        "total_jobs": total,
        "completed_reference_epochs": total_epochs,
        "total_reference_epochs": total_epochs,
        "failed_jobs": 0,
        "elapsed_seconds": time.time() - started,
    })
    print(f"finalized {derived_jobs} jobs in {derived_root}", flush=True)
    print(f"saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the positioning benchmark over cases and source-brightness profiles.

``benchmark_positioning.py`` intentionally benchmarks one case/profile pair.
This wrapper gives long stress runs small, failure-isolated units of work: the
core JAX/native/VBMicrolensing measurements and microLUX are separate child
processes.  The aggregate JSON is replaced atomically after every operation.
Thus a slow microLUX limb-darkening calculation cannot discard a completed core
measurement for the same case/profile.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


# Keep this deliberately small and explicit rather than importing
# benchmark_positioning: importing that script imports optional benchmark
# dependencies, whereas this orchestration layer should remain testable alone.
BENCHMARK_CASES = (
    "regular",
    "resonant_cusp",
    "planetary_far",
    "planetary_cusp",
)
BENCHMARK_PROFILES = ("uniform", "linear")
BENCHMARK_OPERATIONS = ("core", "microlux")
SCHEMA_VERSION = 2


def comma_values(value: str, allowed: Iterable[str], label: str) -> tuple[str, ...]:
    """Parse a non-empty, duplicate-free comma-separated selection."""

    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    duplicates = sorted({item for item in values if values.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}: {duplicates}")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; allowed values: {list(allowed)}")
    return values


def job_id(case: str, profile: str, operation: str) -> str:
    return f"{case}:{profile}:{operation}"


def summarize(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in jobs:
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "jobs": len(jobs),
        "successful": counts.get("success", 0),
        "timed_out": counts.get("timeout", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "status_counts": counts,
    }


def merge_job(report: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Replace one job row and refresh the aggregate summary without I/O."""

    record_id = record["job_id"]
    jobs = [item for item in report.get("jobs", []) if item.get("job_id") != record_id]
    jobs.append(record)
    jobs.sort(key=lambda item: str(item["job_id"]))
    merged = dict(report)
    merged["jobs"] = jobs
    merged["summary"] = summarize(jobs)
    return merged


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a complete report or leave the preceding report intact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_resume_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema_version") != SCHEMA_VERSION or not isinstance(
        report.get("jobs"), list
    ):
        raise ValueError(f"{path} is not a stress_positioning schema v{SCHEMA_VERSION} report")
    return report


def command_for(
    benchmark: Path,
    case: str,
    profile: str,
    operation: str,
    child_output: Path,
    repeat: int,
    inner: int,
) -> list[str]:
    return [
        sys.executable,
        str(benchmark),
        "--case",
        case,
        "--profile",
        profile,
        "--engine",
        operation,
        "--repeat",
        str(repeat),
        "--inner",
        str(inner),
        "--output",
        str(child_output),
    ]


def text_from_process_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def run_job(
    benchmark: Path,
    case: str,
    profile: str,
    operation: str,
    timeout_seconds: float,
    output_directory: Path,
    repeat: int,
    inner: int,
) -> dict[str, Any]:
    """Execute one isolated benchmark and retain its JSON only on success."""

    child_output = output_directory / f".{job_id(case, profile, operation)}.child.json"
    command = command_for(
        benchmark, case, profile, operation, child_output, repeat, inner
    )
    started = time.perf_counter()
    record: dict[str, Any] = {
        "job_id": job_id(case, profile, operation),
        "case": case,
        "profile": profile,
        "operation": operation,
        "command": command,
        "timeout_seconds": timeout_seconds,
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started
        record.update(
            {
                "status": "timeout",
                "elapsed_seconds": elapsed,
                # The child was alive for at least this long.  Keep this separate
                # from elapsed so censored timings cannot look like exact timings.
                "lower_bound_seconds": max(elapsed, timeout_seconds),
                "stdout": text_from_process_output(error.stdout),
                "stderr": text_from_process_output(error.stderr),
            }
        )
        return record

    elapsed = time.perf_counter() - started
    record.update(
        {
            "elapsed_seconds": elapsed,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if completed.returncode != 0:
        record["status"] = "failed"
        return record
    try:
        with child_output.open(encoding="utf-8") as handle:
            record["benchmark"] = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        record.update({"status": "failed", "error": f"invalid child JSON: {error}"})
        return record
    finally:
        child_output.unlink(missing_ok=True)
    record["status"] = "success"
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(BENCHMARK_CASES))
    parser.add_argument("--profiles", default=",".join(BENCHMARK_PROFILES))
    parser.add_argument("--operations", default=",".join(BENCHMARK_OPERATIONS))
    parser.add_argument(
        "--timeout-core",
        type=float,
        default=180.0,
        help="timeout for the JAX/native/VBMicrolensing child",
    )
    parser.add_argument("--timeout-uniform", type=float, default=180.0)
    parser.add_argument("--timeout-limb", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--inner", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        args.case_values = comma_values(args.cases, BENCHMARK_CASES, "cases")
        args.profile_values = comma_values(args.profiles, BENCHMARK_PROFILES, "profiles")
        args.operation_values = comma_values(
            args.operations, BENCHMARK_OPERATIONS, "operations"
        )
    except ValueError as error:
        parser.error(str(error))
    if min(args.timeout_core, args.timeout_uniform, args.timeout_limb) <= 0.0:
        parser.error("timeouts must be positive")
    if args.repeat <= 0 or args.inner <= 0:
        parser.error("repeat and inner must be positive")
    return args


def initial_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "cases": list(args.case_values),
            "profiles": list(args.profile_values),
            "operations": list(args.operation_values),
            "timeout_core_seconds": args.timeout_core,
            "timeout_uniform_seconds": args.timeout_uniform,
            "timeout_limb_seconds": args.timeout_limb,
            "repeat": args.repeat,
            "inner": args.inner,
        },
        "jobs": [],
        "summary": summarize([]),
    }


def run(args: argparse.Namespace) -> int:
    benchmark = Path(__file__).with_name("benchmark_positioning.py")
    if not benchmark.is_file():
        raise FileNotFoundError(f"benchmark not found: {benchmark}")
    report = load_resume_report(args.output) if args.resume and args.output.exists() else initial_report(args)
    completed_ids = {
        item["job_id"] for item in report["jobs"] if item.get("status") == "success"
    }
    atomic_write_json(args.output, report)
    for case in args.case_values:
        for profile in args.profile_values:
            for operation in args.operation_values:
                identifier = job_id(case, profile, operation)
                if identifier in completed_ids:
                    print(f"skip completed {identifier}", flush=True)
                    continue
                timeout_seconds = (
                    args.timeout_core
                    if operation == "core"
                    else args.timeout_uniform
                    if profile == "uniform"
                    else args.timeout_limb
                )
                print(f"run {identifier} timeout={timeout_seconds:g}s", flush=True)
                record = run_job(
                    benchmark,
                    case,
                    profile,
                    operation,
                    timeout_seconds,
                    args.output.parent,
                    args.repeat,
                    args.inner,
                )
                report = merge_job(report, record)
                atomic_write_json(args.output, report)
                print(
                    f"{record['status']} {identifier} elapsed={record['elapsed_seconds']:.3f}s",
                    flush=True,
                )
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

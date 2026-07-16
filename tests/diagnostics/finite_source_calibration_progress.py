#!/usr/bin/env python3
"""Report progress and health of a finite-source calibration sweep."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def nested_results(row: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    yield "lc_auto", row.get("lc_auto", {})
    for grid, sequence in row.get("lc_fixed_sequences", {}).items():
        for result in sequence:
            yield f"lc_{grid}", result
    for mode, result in row.get("vbm", {}).items():
        yield f"vbm_{mode}", result


def summarize(directory: Path, expected_cases: int | None) -> dict[str, Any]:
    files = sorted(directory.glob("case-*.json"))
    cases: list[dict[str, Any]] = []
    corrupt: list[str] = []
    for path in files:
        try:
            cases.append(json.loads(path.read_text()))
        except Exception:
            corrupt.append(path.name)

    result_counts: Counter[str] = Counter()
    timeout_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    elapsed = []
    rows = 0
    for case in cases:
        if "case_elapsed_seconds" in case:
            elapsed.append(float(case["case_elapsed_seconds"]))
        for row in case.get("rows", []):
            rows += 1
            for family, result in nested_results(row):
                result_counts[family] += 1
                if result.get("timeout"):
                    timeout_counts[family] += 1
                if result.get("skipped_after_timeout"):
                    skipped_counts[family] += 1
                if "error" in result:
                    error_counts[family] += 1
                if family.startswith("lc_") and result.get("method"):
                    method_counts[result["method"]] += 1

    complete = len(cases)
    remaining = max((expected_cases or complete) - complete, 0)
    median_seconds = statistics.median(elapsed) if elapsed else math.nan
    return {
        "directory": str(directory.resolve()),
        "completed_cases": complete,
        "expected_cases": expected_cases,
        "completion_fraction": complete / expected_cases if expected_cases else None,
        "rows": rows,
        "corrupt_files": corrupt,
        "median_case_seconds": median_seconds,
        "remaining_worker_hours_at_median": (
            remaining * median_seconds / 3600.0 if math.isfinite(median_seconds) else None
        ),
        "result_counts": dict(sorted(result_counts.items())),
        "timeout_counts": dict(sorted(timeout_counts.items())),
        "skipped_after_timeout_counts": dict(sorted(skipped_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.directory, args.expected_cases)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    expected = summary["expected_cases"]
    completed = summary["completed_cases"]
    fraction = summary["completion_fraction"]
    progress = f"{completed}" if expected is None else f"{completed}/{expected} ({100*fraction:.1f}%)"
    print(f"cases: {progress}")
    print(f"rows: {summary['rows']}")
    print(f"median case: {summary['median_case_seconds']:.2f} s")
    if summary["remaining_worker_hours_at_median"] is not None:
        print(
            "remaining: "
            f"{summary['remaining_worker_hours_at_median']:.2f} worker-hours at current median"
        )
    print("methods:", json.dumps(summary["method_counts"], sort_keys=True))
    print("timeouts:", json.dumps(summary["timeout_counts"], sort_keys=True))
    print("skipped after timeout:", json.dumps(
        summary["skipped_after_timeout_counts"], sort_keys=True
    ))
    print("errors:", json.dumps(summary["error_counts"], sort_keys=True))
    if summary["corrupt_files"]:
        print("corrupt:", ", ".join(summary["corrupt_files"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

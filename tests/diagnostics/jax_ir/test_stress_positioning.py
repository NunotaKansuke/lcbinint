"""Unit tests for the dependency-free stress benchmark orchestrator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("stress_positioning.py")
SPEC = importlib.util.spec_from_file_location("stress_positioning", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stress_positioning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stress_positioning)


def test_comma_values_validate_empty_duplicate_and_unknown_values():
    assert stress_positioning.comma_values(
        "regular,planetary_far", stress_positioning.BENCHMARK_CASES, "cases"
    ) == ("regular", "planetary_far")
    for value, match in (("", "at least"), ("regular,regular", "duplicate"), ("other", "unknown")):
        with pytest.raises(ValueError, match=match):
            stress_positioning.comma_values(
                value, stress_positioning.BENCHMARK_CASES, "cases"
            )


def test_merge_job_replaces_a_row_and_aggregates_statuses():
    report = {"jobs": []}
    report = stress_positioning.merge_job(
        report, {"job_id": "regular:uniform:microlux", "status": "timeout"}
    )
    report = stress_positioning.merge_job(
        report, {"job_id": "regular:linear:core", "status": "success"}
    )
    report = stress_positioning.merge_job(
        report, {"job_id": "regular:uniform:microlux", "status": "success"}
    )

    assert [job["job_id"] for job in report["jobs"]] == [
        "regular:linear:core",
        "regular:uniform:microlux",
    ]
    assert report["summary"] == {
        "jobs": 2,
        "successful": 2,
        "timed_out": 0,
        "failed": 0,
        "skipped": 0,
        "status_counts": {"success": 2},
    }


def test_parse_args_rejects_non_positive_timeouts(tmp_path):
    with pytest.raises(SystemExit):
        stress_positioning.parse_args(
            ["--output", str(tmp_path / "result.json"), "--timeout-limb", "0"]
        )


def test_default_limb_timeout_allows_slow_microlux_ad(tmp_path):
    args = stress_positioning.parse_args(["--output", str(tmp_path / "result.json")])
    assert args.timeout_limb == 300.0
    assert args.operation_values == ("core", "microlux")


def test_command_routes_each_operation_to_the_matching_benchmark_engine(tmp_path):
    command = stress_positioning.command_for(
        Path("benchmark_positioning.py"),
        "regular",
        "linear",
        "microlux",
        tmp_path / "child.json",
        1,
        1,
    )
    assert command[command.index("--engine") + 1] == "microlux"


def test_resume_keeps_core_measurement_when_microlux_timed_out(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    args = stress_positioning.parse_args(
        [
            "--cases",
            "regular",
            "--profiles",
            "linear",
            "--output",
            str(output),
            "--repeat",
            "1",
            "--inner",
            "1",
        ]
    )
    calls = []

    def first_run(_benchmark, case, profile, operation, timeout, *_unused):
        calls.append((case, profile, operation, timeout))
        status = "success" if operation == "core" else "timeout"
        record = {
            "job_id": stress_positioning.job_id(case, profile, operation),
            "case": case,
            "profile": profile,
            "operation": operation,
            "status": status,
            "elapsed_seconds": 1.0,
        }
        if status == "success":
            record["benchmark"] = {"engine": "core"}
        else:
            record["lower_bound_seconds"] = timeout
        return record

    monkeypatch.setattr(stress_positioning, "run_job", first_run)
    assert stress_positioning.run(args) == 0
    assert [call[2] for call in calls] == ["core", "microlux"]
    saved = json.loads(output.read_text())
    assert saved["summary"]["successful"] == 1
    assert saved["summary"]["timed_out"] == 1

    calls.clear()
    args = stress_positioning.parse_args(
        [
            "--cases",
            "regular",
            "--profiles",
            "linear",
            "--output",
            str(output),
            "--resume",
            "--repeat",
            "1",
            "--inner",
            "1",
        ]
    )
    monkeypatch.setattr(stress_positioning, "run_job", first_run)
    assert stress_positioning.run(args) == 0
    assert [call[2] for call in calls] == ["microlux"]

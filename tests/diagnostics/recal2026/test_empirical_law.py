"""Regression tests for the paper-facing recalibration record."""

from __future__ import annotations

import pytest
import json
from pathlib import Path

from .empirical_law import (
    ABSOLUTE_HOLDOUT_COVERAGE,
    ABSOLUTE_LEVELS,
    ABSOLUTE_LAW,
    B0,
    HOLDOUT_COVERAGE,
    RELATIVE_LEVELS,
    RELATIVE_LAW,
    absolute_supported_table,
    branch_resolution,
    bucket_resolution,
    continuous_resolution,
    effective_budget,
    mixed_bucket_resolution,
    normalized_budget,
    supported_table,
)
from .engines import BUCKETS
from .error_budget_law import _required_outcome
from .absolute_error_law import _records as absolute_records


def test_mixed_budget_normalization_has_one_common_definition():
    assert normalized_budget(0.0, 1.0e-3, 0.2) == pytest.approx(1.0e-3)
    assert normalized_budget(1.0e-4, 1.0e-3, 10.0) == pytest.approx(1.0e-3)
    assert normalized_budget(1.0e-4, 0.0, 10.0) == pytest.approx(1.0e-5)
    assert effective_budget(1.0e-4, 1.0e-3, 10.0) == pytest.approx(1.0e-2)


@pytest.mark.parametrize("grid", ("cartesian", "polar"))
def test_relative_law_is_monotone_and_uses_supported_buckets(grid):
    raw = [continuous_resolution(epsilon, grid) for epsilon in RELATIVE_LEVELS]
    selected = [bucket_resolution(epsilon, grid) for epsilon in RELATIVE_LEVELS]

    assert all(left >= right for left, right in zip(raw[1:], raw[:-1]))
    assert all(bucket in BUCKETS for bucket in selected)
    assert selected == sorted(selected)
    assert continuous_resolution(B0, grid) == pytest.approx(
        RELATIVE_LAW[grid]["C"])


def test_calibrated_table_is_independently_covered():
    for grid in ("cartesian", "polar"):
        table = supported_table(grid)
        assert [row[0] for row in table] == list(RELATIVE_LEVELS)
        assert all(coverage >= 0.99 for _, _, coverage in table)
        assert all(coverage == HOLDOUT_COVERAGE[grid][epsilon]
                   for epsilon, _, coverage in table)


@pytest.mark.parametrize("grid", ("cartesian", "polar"))
def test_absolute_law_is_monotone_and_independently_covered(grid):
    raw = [continuous_resolution(atol, grid, "absolute")
           for atol in ABSOLUTE_LEVELS]
    selected = [bucket_resolution(atol, grid, "absolute")
                for atol in ABSOLUTE_LEVELS]

    assert all(left >= right for left, right in zip(raw[1:], raw[:-1]))
    assert selected == sorted(selected)
    assert continuous_resolution(B0, grid, "absolute") == pytest.approx(
        ABSOLUTE_LAW[grid]["C"])
    assert all(coverage >= 0.99 for _, _, coverage
               in absolute_supported_table(grid))
    assert all(coverage == ABSOLUTE_HOLDOUT_COVERAGE[grid][atol]
               for atol, _, coverage in absolute_supported_table(grid))


@pytest.mark.parametrize("grid", ("cartesian", "polar"))
@pytest.mark.parametrize("magnification", (0.2, 1.0, 10.0, 1000.0))
def test_mixed_selector_uses_the_less_demanding_branch(grid, magnification):
    absolute = branch_resolution(1.0e-4, 1.0e-3, magnification,
                                  grid, "absolute")
    relative = branch_resolution(1.0e-4, 1.0e-3, magnification,
                                 grid, "relative")
    assert mixed_bucket_resolution(1.0e-4, 1.0e-3, magnification, grid) == min(
        absolute, relative)


@pytest.mark.parametrize("grid", ("cartesian", "polar"))
def test_absolute_branch_remains_dimensional(grid):
    assert branch_resolution(1.0e-2, 0.0, 0.2, grid, "absolute") == \
        branch_resolution(1.0e-2, 0.0, 1000.0, grid, "absolute")


def test_mixed_holdout_record_clears_the_coverage_target():
    path = (Path(__file__).resolve().parents[3]
            / "tests/diagnostics/results/recal2026/mixed_error_law.json")
    report = json.loads(path.read_text())
    for grid in ("cartesian", "polar"):
        summary = report["holdout"][grid]["summary"]
        assert summary["pairs"] == 81
        assert summary["pairs_meeting_target"] == 81
        assert summary["minimum_coverage"] >= 0.99
        assert summary["identity_mismatches"] == 0


def test_reference_limited_row_is_retained_as_a_lower_censored_requirement():
    row = {
        "reference": {"value": 2.0, "uncertainty": 1.0e-2},
        "cartesian": {4: {
            "magnification": 2.0,
            "support_proven": True,
        }},
    }
    outcome = _required_outcome(row, "cartesian", 1.0e-3, 0.0)
    assert outcome["status"] == "lower_censored"
    assert outcome["reason"] == "reference_uncertainty"
    assert outcome["required"] is None
    assert outcome["lower_bound"] == 4


def test_certified_row_keeps_an_exact_persistent_crossing():
    row = {
        "reference": {"value": 2.0, "uncertainty": 1.0e-8},
        "cartesian": {4: {
            "magnification": 2.0,
            "support_proven": True,
        }},
    }
    outcome = _required_outcome(row, "cartesian", 1.0e-3, 0.0)
    assert outcome == {
        "status": "observed",
        "required": 4,
        "lower_bound": 4,
        "reason": None,
    }


def test_absolute_record_builder_keeps_reference_limited_rows():
    row = {
        "case_id": 7,
        "reference": {"value": 2.0, "uncertainty": 1.0e-2},
        "point_magnification": 2.0,
        "cartesian": {4: {
            "magnification": 2.0,
            "support_proven": True,
        }},
        "polar": {4: {
            "magnification": 2.0,
            "support_proven": True,
        }},
    }
    records = absolute_records([row], "holdout")
    assert len(records) == 2 * len(ABSOLUTE_LEVELS)
    assert all(record["censored"] for record in records)
    assert all(record["required_resolution"] == 4
               for record in records)
    assert all(record["censor_reason"] == "reference_uncertainty"
               for record in records)

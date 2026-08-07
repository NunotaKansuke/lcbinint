"""Regression tests for the paper-facing recalibration record."""

from __future__ import annotations

import pytest

from .empirical_law import (
    B0,
    HOLDOUT_COVERAGE,
    RELATIVE_LEVELS,
    RELATIVE_LAW,
    bucket_resolution,
    continuous_resolution,
    normalized_budget,
    supported_table,
)
from .engines import BUCKETS


def test_mixed_budget_normalization_has_one_common_definition():
    assert normalized_budget(0.0, 1.0e-3, 0.2) == pytest.approx(1.0e-3)
    assert normalized_budget(1.0e-4, 1.0e-3, 10.0) == pytest.approx(1.01e-3)
    assert normalized_budget(1.0e-4, 0.0, 10.0) == pytest.approx(1.0e-5)


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

import jax
import numpy as np

from lcbinint_jax import select_binary_resolution

jax.config.update("jax_enable_x64", True)


def test_native_calibrated_polar_resolution_and_tolerance_scaling():
    regular = select_binary_resolution(1.0e-3, 1.0e-3, 2.0e-4, 1000.0)
    tight = select_binary_resolution(
        1.0e-3,
        1.0e-3,
        2.0e-4,
        1000.0,
        requested_relative_tolerance=1.0e-5,
    )
    assert bool(regular.prefer_polar)
    # Native's continuous 105.3-bin result gets the one backend-wide safety
    # margin and is then rounded up to JAX's 128-bin static bucket.
    assert int(regular.source_bins) == 128
    assert bool(tight.prefer_polar)
    assert int(tight.source_bins) == 400


def test_native_calibrated_cartesian_power_law_and_cap():
    regular = select_binary_resolution(0.1, 1.0e-3, 1.0e-3, 10.0)
    tight = select_binary_resolution(
        0.1,
        1.0e-3,
        1.0e-3,
        10.0,
        requested_relative_tolerance=1.0e-5,
    )
    capped = select_binary_resolution(
        0.1,
        1.0e-3,
        1.0e-3,
        10.0,
        requested_relative_tolerance=1.0e-5,
        maximum_bins=128,
    )
    assert not bool(regular.prefer_polar)
    assert int(regular.source_bins) == 64
    assert not bool(tight.prefer_polar)
    # 1e-5 is below native's calibrated support range, so the selector
    # deliberately fails closed at the cap rather than inventing an
    # extrapolated power law.
    assert int(tight.source_bins) == 400
    assert int(capped.source_bins) == 128


def test_absolute_tolerance_is_passed_as_a_separate_native_branch():
    relative_only = select_binary_resolution(
        0.1,
        1.0e-3,
        1.0e-3,
        10.0,
        requested_relative_tolerance=1.0e-3,
        requested_absolute_tolerance=0.0,
    )
    absolute_only = select_binary_resolution(
        0.1,
        1.0e-3,
        1.0e-3,
        10.0,
        requested_relative_tolerance=0.0,
        requested_absolute_tolerance=1.0e-3,
    )
    assert int(relative_only.source_bins) == 64
    assert int(absolute_only.source_bins) == 400


def test_selection_is_jittable_and_returns_stopped_integer_routing_data():
    evaluate = jax.jit(select_binary_resolution)
    result = evaluate(1.0e-4, 2.0e-3, 4.0e-3, 20.0, 0.5)
    assert result.source_bins.dtype == np.int32
    assert result.prefer_polar.dtype == np.bool_

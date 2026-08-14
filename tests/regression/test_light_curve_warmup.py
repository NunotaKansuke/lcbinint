import importlib
import warnings

import numpy as np

import lcbinint

warmup_module = importlib.import_module("lcbinint.warmup")


PARAMETERS = {
    "t0": 0.0,
    "tE": 1.0,
    "u0": 0.01,
    "alpha": 0.0,
    "s": 1.2,
    "q": 0.1,
    "rho": 0.02,
}
TIMES = np.asarray([-0.02, 0.0, 0.02])


def _curve(reltol=1.0e-3, max_source_bins=400):
    return lcbinint.LightCurve(
        options=lcbinint.Options(
            nbin="auto", reltol=reltol, max_source_bins=max_source_bins
        )
    )


def _baseline_reference(curve, times=TIMES, params=PARAMETERS):
    return np.asarray(curve(times, params), dtype=float)


def test_warmup_retains_and_automatically_uses_execution_plan():
    curve = _curve()
    reference = _baseline_reference(curve)

    report = curve.warmup(TIMES, PARAMETERS)

    assert curve.warmup_profile is report
    assert curve.warmup_plan is report
    assert report.all_calibrated
    assert all(method != "auto_fallback" for method in report.methods)
    actual = np.asarray(curve(TIMES, PARAMETERS))
    np.testing.assert_array_less(
        np.abs(actual - reference),
        report.budget + np.finfo(float).eps,
    )
    assert not hasattr(curve, "_warmup_values")
    assert report.geometry is not None


def test_warmup_plan_is_reused_for_nearby_parameters_without_warning():
    curve = _curve()
    report = curve.warmup(TIMES, PARAMETERS)

    changed = dict(PARAMETERS, u0=0.011)
    expected = curve._native._magnification_preplanned(
        TIMES,
        changed,
        curve._warmup_methods.tolist(),
        report.resolutions.tolist(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = curve(TIMES, changed)
    assert caught == []
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    curve.clear_warmup()
    assert curve.warmup_profile is None
    assert not hasattr(curve, "_warmup_values")


def test_large_geometry_drift_warns_but_keeps_the_plan():
    curve = _curve()
    report = curve.warmup(TIMES, PARAMETERS)
    changed = dict(PARAMETERS, s=3.0)

    drift = curve.warmup_drift(TIMES, changed)
    assert drift.warn
    assert drift.topology_changed
    with np.testing.assert_warns(RuntimeWarning):
        actual = curve(TIMES, changed)
    expected = curve._native._magnification_preplanned(
        TIMES,
        changed,
        curve._warmup_methods.tolist(),
        report.resolutions.tolist(),
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_shared_model_changes_invalidate_warmup_plan():
    model = lcbinint.Model()
    curve = lcbinint.LightCurve(
        model=model,
        options=lcbinint.Options(nbin="auto", reltol=1.0e-3),
    )
    curve.warmup(TIMES, PARAMETERS)

    model.sky = lcbinint.obs.SkyCoord(270.0, -30.0)
    assert not curve._matching_warmup(TIMES, PARAMETERS)

    model.sky = None
    model.finite_source = False
    assert not curve._matching_warmup(TIMES, PARAMETERS)
    expected = lcbinint.LightCurve(
        model=model,
        options=lcbinint.Options(nbin="auto", reltol=1.0e-3),
    )(TIMES, PARAMETERS)
    np.testing.assert_allclose(
        curve(TIMES, PARAMETERS), expected, rtol=0.0, atol=0.0
    )


def test_incomplete_warmup_is_rejected_instead_of_retaining_fallback_rows():
    # Four cells are below the minimum useful automatic grid and correctly
    # fail closed in the production API.  Eight cells are still deliberately
    # too small for warm-up's three-pass certification, while normal auto can
    # return its bounded result; this exercises the per-epoch fallback without
    # making the baseline itself an intentional numerical-error case.
    curve = _curve(max_source_bins=8)
    with np.testing.assert_raises_regex(
        RuntimeError, "could not calibrate every epoch"
    ):
        curve.warmup(TIMES, PARAMETERS)
    assert curve.warmup_profile is None


def test_native_preplanned_route_skips_auto_and_preserves_requested_method():
    curve = _curve()
    result = curve._native._evaluate_preplanned(
        TIMES, PARAMETERS, [1] * TIMES.size, [0] * TIMES.size
    )
    assert result["method"] == [1, 1, 1]
    assert all(result["converged"])
    assert np.all(np.isfinite(result["magnification"]))


def test_warmup_hint_uses_frozen_native_resolution_law():
    curve = _curve()
    hint = curve._native._binary_resolution_hint

    assert hint(warmup_module.CARTESIAN, 10.0, 0.0, 1.0e-3, 400) == 50
    assert hint(warmup_module.POLAR, 10.0, 0.0, 1.0e-3, 400) == 106
    assert hint(warmup_module.CARTESIAN, 10.0, 1.0e-3, 0.0, 400) == 303
    assert hint(warmup_module.CARTESIAN, 10.0, 1.0e-2, 1.0e-4, 400) == 114
    assert hint(warmup_module.POLAR, 10.0, 0.0, 1.0e-3, 80) == 80


def test_grid_choice_uses_measured_time_and_only_qualified_candidates():
    choose = warmup_module._choose_grid
    assert choose(32, 64, 0.004, 0.002) == (3, 64)
    assert choose(32, 64, 0.001, 0.003) == (2, 32)
    assert choose(None, 64, np.nan, 0.003) == (3, 64)
    assert choose(32, None, 0.003, np.nan) == (2, 32)
    assert choose(None, None, np.nan, np.nan) == (None, None)


def test_binary_topology_uses_the_analytic_equal_mass_boundaries():
    topology = warmup_module._binary_topology(
        np.asarray([0.7, 1.0, 2.1]),
        np.ones(3),
    )
    assert topology.tolist() == ["close", "resonant", "wide"]


def test_grid_search_requires_three_increasing_passes():
    run = warmup_module._persistent_pass_run
    samples = {
        16: {"pass": False},
        32: {"pass": True},
        48: {"pass": True},
    }
    assert run(samples) is None
    samples[64] = {"pass": True}
    assert run(samples) == (32, 48, 64)
    assert warmup_module._candidate_batch(64, 400) == (16, 32, 64)

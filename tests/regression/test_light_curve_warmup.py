import importlib

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


def _curve(reltol=1.0e-3):
    return lcbinint.LightCurve(
        options=lcbinint.Options(nbin="auto", reltol=reltol)
    )


def _install_exact_contour(monkeypatch, curve, times=TIMES, params=PARAMETERS):
    native = curve._native._evaluate_preplanned(
        times, params, [2] * len(times), [400] * len(times)
    )
    reference = np.asarray(native["magnification"], dtype=float)

    def contour(_geometry, levels):
        values = np.repeat(reference[None, :], len(levels), axis=0)
        return reference.copy(), np.zeros_like(reference), values

    monkeypatch.setattr(warmup_module, "_contour_witness", contour)
    return reference


def test_warmup_retains_and_automatically_uses_execution_plan(monkeypatch):
    curve = _curve()
    reference = _install_exact_contour(monkeypatch, curve)

    report = curve.warmup(
        TIMES,
        PARAMETERS,
        ladder=(16, 64, 256, 400),
        contour_levels=(1.0e-4, 1.0e-6),
    )

    assert curve.warmup_profile is report
    assert curve.warmup_plan is report
    assert report.all_calibrated
    assert all(method != "auto_fallback" for method in report.methods)
    actual = np.asarray(curve(TIMES, PARAMETERS))
    np.testing.assert_array_less(
        np.abs(actual - reference),
        report.budget + np.finfo(float).eps,
    )


def test_warmup_is_not_reused_for_different_parameters(monkeypatch):
    curve = _curve()
    _install_exact_contour(monkeypatch, curve)
    curve.warmup(
        TIMES,
        PARAMETERS,
        ladder=(16, 64, 256, 400),
        contour_levels=(1.0e-4, 1.0e-6),
    )

    changed = dict(PARAMETERS, u0=0.011)
    expected = _curve()(TIMES, changed)
    np.testing.assert_allclose(curve(TIMES, changed), expected, rtol=0.0, atol=0.0)

    curve.clear_warmup()
    assert curve.warmup_profile is None


def test_native_preplanned_route_skips_auto_and_preserves_requested_method():
    curve = _curve()
    result = curve._native._evaluate_preplanned(
        TIMES, PARAMETERS, [1] * TIMES.size, [0] * TIMES.size
    )
    assert result["method"] == [1, 1, 1]
    assert all(result["converged"])
    assert np.all(np.isfinite(result["magnification"]))


def test_grid_choice_uses_measured_time_and_only_qualified_candidates():
    choose = warmup_module._choose_grid
    assert choose(32, 64, 0.004, 0.002) == (3, 64)
    assert choose(32, 64, 0.001, 0.003) == (2, 32)
    assert choose(None, 64, np.nan, 0.003) == (3, 64)
    assert choose(32, None, 0.003, np.nan) == (2, 32)
    assert choose(None, None, np.nan, np.nan) == (None, None)

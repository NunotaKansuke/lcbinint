import numpy as np
import pytest


def test_parameters_accept_keyword_overrides_and_geometry_helpers():
    lcbinint = pytest.importorskip("lcbinint")
    params = lcbinint.Parameters(t0=0.0, tE=1.0, u0=0.2, s=1.1, q=0.2, rho=0.0)
    curve = lcbinint.LightCurve()

    overridden = curve([0.0], params, u0=0.8)
    expected = curve([0.0], {"t0": 0.0, "tE": 1.0, "u0": 0.8, "s": 1.1, "q": 0.2, "rho": 0.0})
    assert np.allclose(overridden, expected)

    trajectory = curve.source_trajectory([0.0], params)
    assert trajectory.times == [0.0]
    assert len(trajectory.x) == 1
    assert curve.separation(params) == pytest.approx(1.1)


def test_model_parallax_false_disables_option_flag():
    lcbinint = pytest.importorskip("lcbinint")
    sky = lcbinint.obs.SkyCoord(270.0, -30.0)
    model = lcbinint.Model(parallax=False, sky=sky, t_ref=2459000.0)
    options = lcbinint.Options()
    options.parallax_mode = 1
    curve = lcbinint.LightCurve(options=options, model=model)
    baseline = lcbinint.LightCurve(model=model)
    params = dict(
        t0=2459000.0, tE=20.0, u0=0.1, s=1.1, q=0.2, rho=0.0,
        piEN=0.1, piEE=0.05, ra=270.0, dec=-30.0,
    )

    assert np.allclose(curve([2459020.0], params), baseline([2459020.0], params))

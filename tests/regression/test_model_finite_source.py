"""Regression coverage for Model.finite_source."""

import numpy as np
import pytest


def _lcbinint():
    return pytest.importorskip("lcbinint")


def test_point_profile_forces_single_source_radius_to_zero():
    lcbinint = _lcbinint()
    times = np.linspace(7499.5, 7500.5, 9)
    parameters = {
        "s": 0.9, "q": 0.1, "alpha": 0.7, "tE": 30.0,
        "t0": 7500.0, "u0": 0.02, "rho": 0.004,
    }
    point = lcbinint.LightCurve(model=lcbinint.Model(finite_source=False))
    explicit = lcbinint.LightCurve(model=lcbinint.Model(finite_source=True))
    zero_radius = dict(parameters, rho=0.0)

    assert point._native.model.finite_source is False
    np.testing.assert_allclose(point(times, parameters), explicit(times, zero_radius))


def test_point_profile_forces_both_binary_source_radii_to_zero():
    lcbinint = _lcbinint()
    times = np.linspace(7499.5, 7500.5, 9)
    parameters = {
        "s": 0.9, "q": 0.1, "alpha": 0.7, "tE": 30.0,
        "t0": 7500.0, "u0": 0.02, "rho1": 0.004,
        "t0_2": 7501.2, "u0_2": -0.06, "rho2": 0.003,
        "flux_ratio": 0.4,
    }
    point = lcbinint.LightCurve(
        model=lcbinint.Model(source="binary", finite_source=False)
    )
    explicit = lcbinint.LightCurve(
        model=lcbinint.Model(source="binary", finite_source=True)
    )
    zero_radii = dict(parameters, rho1=0.0, rho2=0.0)

    np.testing.assert_allclose(point(times, parameters), explicit(times, zero_radii))

"""Public binary-source component API regression coverage."""

import numpy as np
import pytest


def _lcbinint():
    return pytest.importorskip("lcbinint")


TIMES = np.linspace(7470.0, 7530.0, 31)


@pytest.mark.parametrize(
    "model,parameters",
    [
        (
            {},
            dict(
                s=0.9, q=0.1, alpha=0.7, tE=30.0, t0=7500.0, u0=0.35,
                t0_2=7501.2, u0_2=0.32, rho1=0.004, rho2=0.002,
                flux_ratio=0.4,
            ),
        ),
        (
            dict(
                xallarap="circular_velocity",
                source_orbit_coordinates="xallarap",
                t_ref=7500.0,
            ),
            dict(
                s=0.9, q=0.1, alpha=0.7, tE=30.0, t0=7500.0, u0=0.35,
                rho1=0.004, rho2=0.002, flux_ratio=0.4,
                source_mass_ratio=0.7, xi_1=0.006, xi_2=-0.003,
                w1=0.001, w2=0.12, w3=0.02,
            ),
        ),
        (
            dict(
                xallarap="circular_velocity",
                source_orbit_coordinates="trajectory_offset",
                t_ref=7500.0,
            ),
            dict(
                s=0.9, q=0.1, alpha=0.7, tE=30.0,
                t0=7499.82, u0=0.347, t0_2=7500.257142857,
                u0_2=0.354285714, rho1=0.004, rho2=0.002,
                flux_ratio=0.4, source_mass_ratio=0.7,
                w1=0.001, w2=0.12, w3=0.02,
            ),
        ),
    ],
)
def test_binary_source_components_match_combined_curve(model, parameters):
    curve = _lcbinint().LightCurve(source="binary", **model)

    components = curve.binary_source_components(TIMES, parameters)

    np.testing.assert_allclose(components.total, curve(TIMES, parameters))
    np.testing.assert_allclose(
        components.total,
        (
            np.asarray(components.source1.magnification)
            + parameters["flux_ratio"] * np.asarray(components.source2.magnification)
        ) / (1.0 + parameters["flux_ratio"]),
    )
    assert len(components.source1.trajectory.x) == len(TIMES)
    assert len(components.source2.trajectory.y) == len(TIMES)


def test_binary_source_components_reject_single_source_curve():
    curve = _lcbinint().LightCurve()
    with pytest.raises(ValueError, match="source='binary'"):
        curve.binary_source_components(
            TIMES,
            dict(s=0.9, q=0.1, alpha=0.7, tE=30.0, t0=7500.0, u0=0.35, rho=0.004),
        )

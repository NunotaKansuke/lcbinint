"""Regression tests for xallarap (source orbital motion) modes.

Covers all 4 supported modes:
  orbital_elements  – Kepler orbit, elements parameterization
  circular_elements – circular orbit (ecc=0), elements parameterization
  circular_velocity – circular orbit, position+velocity at tref
  kepler_velocity   – Kepler orbit, position+velocity at tref

Also covers:
  - orbital_elements(ecc=0, peri=0) == circular_elements identity
  - xi_1=xi_2=0 degenerates to no-xallarap case
"""
import math
import sys
from pathlib import Path

import pytest
import numpy as np


def _lc():
    return pytest.importorskip("lcbinint")


TIMES = np.array([-15.0, -5.0, 0.0, 5.0, 15.0])
COMMON = dict(t0=0.0, tE=20.0, u0=0.3, s=1.0, q=0.1, alpha=0.5)


# ---------------------------------------------------------------------------
# Regression values (generated from reference implementation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,kwargs,expected", [
    pytest.param(
        "orbital_elements",
        dict(xi_1=0.25, xi_2=-0.1, period_xa=35.0, ecc_xa=0.3, peri_xa=0.8, inc_xa=1.1),
        [1.0951242498, 1.7918681347, 2.8680974931, 2.2761981412, 1.1267079521],
        id="orbital_elements",
    ),
    pytest.param(
        "circular_elements",
        dict(xi_1=0.25, xi_2=-0.1, period_xa=35.0, inc_xa=1.1),
        [1.1455635838, 2.1297043229, 2.8680974931, 3.0138925494, 1.2326004581],
        id="circular_elements",
    ),
    pytest.param(
        "circular_velocity",
        dict(xi_1=0.25, xi_2=-0.1, w1=0.02, w2=1.1, w3=0.3),
        [1.50989037, 2.5975735221, 7.4576095196, 2.399784891, 2.0642836428],
        id="circular_velocity",
    ),
    pytest.param(
        "kepler_velocity",
        dict(xi_1=0.25, xi_2=-0.1, w1=0.02, w2=1.1, w3=0.3, xa_szs=0.2, xa_ar=1.5),
        [1.244662934, 1.59211377, 7.4576095196, 1.5468668614, 1.3328897521],
        id="kepler_velocity",
    ),
])
def test_xallarap_regression(mode, kwargs, expected):
    lc = _lc().LightCurve(xallarap=mode)
    mag = lc(TIMES, **COMMON, **kwargs)
    np.testing.assert_allclose(mag, expected, rtol=1e-7)


# ---------------------------------------------------------------------------
# Physical identities
# ---------------------------------------------------------------------------

def test_orbital_elements_ecc0_equals_circular_elements():
    """orbital_elements with ecc=0, peri=0 must be identical to circular_elements."""
    lc = _lc()
    lc_oe = lc.LightCurve(xallarap="orbital_elements")
    lc_ce = lc.LightCurve(xallarap="circular_elements")
    kw = dict(xi_1=0.25, xi_2=-0.1, period_xa=35.0, inc_xa=1.1)
    m_oe = lc_oe(TIMES, **COMMON, **kw, ecc_xa=0.0, peri_xa=0.0)
    m_ce = lc_ce(TIMES, **COMMON, **kw)
    np.testing.assert_allclose(m_oe, m_ce, rtol=1e-10)


def test_xi_zero_is_no_xallarap():
    """xi_1=xi_2=0 must give the same magnification as no xallarap."""
    lc = _lc()
    lc_no = lc.LightCurve()
    m_no = lc_no(TIMES, **COMMON)
    for mode, extra in [
        ("orbital_elements",  dict(period_xa=35.0, ecc_xa=0.3, peri_xa=0.8, inc_xa=1.1)),
        ("circular_elements", dict(period_xa=35.0, inc_xa=1.1)),
        ("circular_velocity", dict(w1=0.02, w2=1.1, w3=0.3)),
        ("kepler_velocity",   dict(w1=0.02, w2=1.1, w3=0.3, xa_szs=0.2, xa_ar=1.5)),
    ]:
        m = lc.LightCurve(xallarap=mode)(TIMES, **COMMON, xi_1=0.0, xi_2=0.0, **extra)
        np.testing.assert_allclose(m, m_no, rtol=1e-10,
                                   err_msg=f"{mode}: xi=0 should equal no-xallarap")


def test_binary_velocity_xallarap_requires_explicit_coordinate_system():
    lc = _lc()
    with pytest.raises(ValueError, match="requires source_orbit_coordinates"):
        lc.LightCurve(source="binary", xallarap="circular_velocity", t_ref=0.0)
    with pytest.raises(ValueError, match="requires binary source xallarap"):
        lc.LightCurve(source_orbit_coordinates="xallarap")
    with pytest.raises(ValueError, match="only with binary velocity xallarap"):
        lc.LightCurve(
            source="binary", xallarap="circular_elements",
            source_orbit_coordinates="xallarap", t_ref=0.0,
        )


def _binary_xallarap_base():
    return dict(
        s=1.0, q=0.1, alpha=0.5, tE=20.0,
        rho1=0.0, rho2=0.0, flux_ratio=0.4,
        source_mass_ratio=0.5,
    )


def test_binary_velocity_xallarap_coordinates_match_two_single_sources():
    """The xi path uses the supplied source-one CoM state and q fixes source two."""
    lc = _lc()
    params = dict(
        _binary_xallarap_base(), t0=0.0, u0=0.3,
        xi_1=0.25, xi_2=-0.1, w1=0.02, w2=1.1, w3=0.3,
    )
    binary = lc.LightCurve(
        source="binary", xallarap="circular_velocity",
        source_orbit_coordinates="xallarap", t_ref=0.0,
    )
    got = binary(TIMES, params)
    one = lc.LightCurve(xallarap="circular_velocity", t_ref=0.0)
    first = one(TIMES, dict(params, rho=0.0))
    second = one(TIMES, dict(
        params, rho=0.0,
        xi_1=-params["xi_1"] / params["source_mass_ratio"],
        xi_2=-params["xi_2"] / params["source_mass_ratio"],
    ))
    np.testing.assert_allclose(got, (first + 0.4 * second) / 1.4, rtol=1e-12)


def test_binary_trajectory_offset_xallarap_coordinates_match_two_single_sources():
    """The t0/u0 path is converted to CoM plus the relative state at t_ref."""
    lc = _lc()
    params = dict(
        _binary_xallarap_base(), t0=-0.4, u0=0.3, t0_2=0.8, u0_2=-0.1,
        w1=0.02, w2=1.1, w3=0.3,
    )
    binary = lc.LightCurve(
        source="binary", xallarap="circular_velocity",
        source_orbit_coordinates="trajectory_offset", t_ref=0.0,
    )
    got = binary(TIMES, params)

    q = params["source_mass_ratio"]
    t0_com = (params["t0"] + q * params["t0_2"]) / (1.0 + q)
    u0_com = (params["u0"] + q * params["u0_2"]) / (1.0 + q)
    xi_1 = -q / (1.0 + q) * (params["t0"] - params["t0_2"]) / params["tE"]
    xi_2 = -q / (1.0 + q) * (params["u0_2"] - params["u0"])
    common = dict(params, t0=t0_com, u0=u0_com, rho=0.0, xi_1=xi_1, xi_2=xi_2)
    one = lc.LightCurve(xallarap="circular_velocity", t_ref=0.0)
    first = one(TIMES, common)
    second = one(TIMES, dict(common, xi_1=-xi_1 / q, xi_2=-xi_2 / q))
    np.testing.assert_allclose(got, (first + 0.4 * second) / 1.4, rtol=1e-12)


def test_binary_element_xallarap_uses_xi_coordinates_without_switch():
    lc = _lc()
    params = dict(
        _binary_xallarap_base(), t0=0.0, u0=0.3,
        xi_1=0.25, xi_2=-0.1, period_xa=35.0, inc_xa=1.1,
    )
    binary = lc.LightCurve(
        source="binary", xallarap="circular_elements", t_ref=0.0,
    )
    got = binary(TIMES, params)
    one = lc.LightCurve(xallarap="circular_elements", t_ref=0.0)
    first = one(TIMES, dict(params, rho=0.0))
    second = one(TIMES, dict(
        params, rho=0.0,
        xi_1=-params["xi_1"] / params["source_mass_ratio"],
        xi_2=-params["xi_2"] / params["source_mass_ratio"],
    ))
    np.testing.assert_allclose(got, (first + 0.4 * second) / 1.4, rtol=1e-12)


@pytest.mark.parametrize("orbital_motion", ["circular", "kepler"])
def test_xallarap_is_not_discarded_when_lens_orbital_motion_is_active(orbital_motion):
    """The LOM frame rotation must use the xallarap-perturbed trajectory."""
    lc = _lc()
    common = dict(
        **COMMON,
        rho=0.0,
        g1=0.004,
        g2=0.011,
        g3=0.006,
    )
    if orbital_motion == "kepler":
        common.update(lom_szs=0.2, lom_ar=1.4)
    plain = lc.LightCurve(orbital_motion=orbital_motion, t_ref=0.0)
    perturbed = lc.LightCurve(
        orbital_motion=orbital_motion,
        xallarap="circular_elements",
        t_ref=0.0,
    )
    times = np.asarray([-15.0, -5.0, 5.0, 15.0])
    no_xallarap = plain(times, common)
    with_xallarap = perturbed(
        times,
        dict(
            common,
            xi_1=0.25,
            xi_2=-0.1,
            period_xa=35.0,
            inc_xa=1.1,
        ),
    )

    assert np.max(np.abs(with_xallarap - no_xallarap)) > 1.0e-5

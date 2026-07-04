"""Regression tests for xallarap (source orbital motion) modes.

Covers all 4 supported modes:
  orbital_elements  – Kepler orbit, elements parameterization
  circular_elements – circular orbit (ecc=0), elements parameterization
  circular_velocity – circular orbit, position+velocity at tref
  kepler_velocity   – Kepler orbit, position+velocity at tref

Also covers:
  - orbital_elements(ecc=0, peri=0) == circular_elements identity
  - xi_1=xi_2=0 degenerates to no-xallarap case
  - coupled binary source (q_mass scaling)
"""
import math
import sys
from pathlib import Path

import pytest
import numpy as np


def _lc():
    return pytest.importorskip("lcbinint").lc


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


# ---------------------------------------------------------------------------
# Coupled binary source
# ---------------------------------------------------------------------------

def test_coupled_binary_regression():
    """circular_elements + coupled binary source regression."""
    lc = _lc().LightCurve(xallarap="circular_elements", source="binary")
    mag = lc(TIMES, **COMMON, xi_1=0.25, xi_2=-0.1, period_xa=35.0, inc_xa=1.1,
             q_source=0.5, q_mass=2.0)
    expected = [1.3233486146, 2.1688701857, 2.8680974931, 3.0173340503, 1.3013206685]
    np.testing.assert_allclose(mag, expected, rtol=1e-7)


def test_coupled_binary_qmass_scaling():
    """Source 2 xi is -xi_1/q_mass: manual construction must match LightCurve q_mass path."""
    lc = _lc()
    lc_s = lc.LightCurve(xallarap="circular_elements")
    lc_b = lc.LightCurve(xallarap="circular_elements", source="binary")
    kw = dict(**COMMON, xi_1=0.3, xi_2=0.0, period_xa=40.0, inc_xa=0.9)

    for q_mass in [1.0, 3.0]:
        m_coupled = lc_b(TIMES, **kw, q_source=1.0, q_mass=q_mass)
        m_s1 = lc_s(TIMES, **kw)
        kw2 = dict(**COMMON, xi_1=-kw["xi_1"]/q_mass, xi_2=0.0,
                   period_xa=kw["period_xa"], inc_xa=kw["inc_xa"])
        m_s2 = lc_s(TIMES, **kw2)
        m_manual = (m_s1 + m_s2) / 2.0
        np.testing.assert_allclose(m_coupled, m_manual, rtol=1e-10,
                                   err_msg=f"q_mass={q_mass}")


def test_coupled_binary_all_modes_run():
    """All 4 modes with binary source must compute without error."""
    lc = _lc()
    modes_kwargs = [
        ("orbital_elements",
         dict(xi_1=0.2, xi_2=0.0, period_xa=30.0, ecc_xa=0.1, peri_xa=0.5, inc_xa=1.0)),
        ("circular_elements",
         dict(xi_1=0.2, xi_2=0.0, period_xa=30.0, inc_xa=1.0)),
        ("circular_velocity",
         dict(xi_1=0.2, xi_2=0.0, w1=0.01, w2=1.0, w3=0.2)),
        ("kepler_velocity",
         dict(xi_1=0.2, xi_2=0.0, w1=0.01, w2=1.0, w3=0.2, xa_szs=0.1, xa_ar=1.3)),
    ]
    for mode, extra in modes_kwargs:
        lb = lc.LightCurve(xallarap=mode, source="binary")
        mag = lb(TIMES, **COMMON, **extra, q_source=1.0, q_mass=2.0)
        assert np.all(np.isfinite(mag)), f"{mode}: non-finite magnification"
        assert np.all(mag > 0), f"{mode}: non-positive magnification"

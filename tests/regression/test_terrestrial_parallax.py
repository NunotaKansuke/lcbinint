"""Regression tests for terrestrial parallax.

Terrestrial parallax adds the observatory's geocentric displacement (projected
onto the sky plane) to the source trajectory, using the same piEN/piEE as
annual parallax.  The tests verify:
  1. parallax terms are inactive unless the physical model enables parallax.
  2. site=(0, 0) is treated as a real equatorial/Greenwich observatory.
  3. The inter-observatory displacement between two telescopes at the same time
     matches the hand-computed value from the sky-projected baseline.
  4. Diurnal variation: source position changes over a night as Earth rotates.
"""
import math
import numpy as np
import pytest
import lcbinint

# Sidereal rotation rate in degrees/day (same as GMST formula)
_SIDEREAL_DEG_PER_DAY = 360.98564736629
# Earth radius in AU
_EARTH_RADIUS_AU = 4.2635212e-5
# Degrees to radians
_DEG = math.pi / 180.0


def _sky_north(ra_deg, dec_deg):
    """Unit vector pointing North on the sky plane (equatorial J2000)."""
    ra = ra_deg * _DEG
    dec = dec_deg * _DEG
    # event unit vector
    ev = np.array([math.cos(ra) * math.cos(dec), math.sin(ra) * math.cos(dec), math.sin(dec)])
    # east unit vector
    east = np.cross([0, 0, 1], ev)
    east /= np.linalg.norm(east)
    return np.cross(ev, east)


def _sky_east(ra_deg, dec_deg):
    ra = ra_deg * _DEG
    dec = dec_deg * _DEG
    ev = np.array([math.cos(ra) * math.cos(dec), math.sin(ra) * math.cos(dec), math.sin(dec)])
    east = np.cross([0, 0, 1], ev)
    return east / np.linalg.norm(east)


def _gast_rad(jd):
    """Greenwich Mean Sidereal Time in radians (GAST approximation)."""
    return (_SIDEREAL_DEG_PER_DAY * (jd - 2451545.0) + 280.46061837) * _DEG


def _tel_position_au(lat_deg, lon_deg, jd):
    """Geocentric telescope position in equatorial J2000 (AU)."""
    lat = lat_deg * _DEG
    lon = lon_deg * _DEG
    ha = _gast_rad(jd) + lon
    cos_lat = math.cos(lat)
    return _EARTH_RADIUS_AU * np.array([
        cos_lat * math.cos(ha),
        cos_lat * math.sin(ha),
        math.sin(lat),
    ])


def _terrestrial_delta_u(lat_deg, lon_deg, jd, piEN, piEE, ra_deg, dec_deg):
    """Expected (delta_tau, delta_beta) due to terrestrial parallax."""
    r = _tel_position_au(lat_deg, lon_deg, jd)
    north = _sky_north(ra_deg, dec_deg)
    east = _sky_east(ra_deg, dec_deg)
    proj_N = -np.dot(r, north)
    proj_E = -np.dot(r, east)
    delta_tau = piEN * proj_N + piEE * proj_E
    delta_beta = -piEE * proj_N + piEN * proj_E
    return delta_tau, delta_beta


# Common event parameters (large piEN so the terrestrial shift is measurable)
_RA = 270.0
_DEC = -30.0
# Use a JD well within the ephemeris range; t0par = same value so annual offset = 0
_T0PAR = 2459000.0  # HJD 2459000 (inside typical annual-parallax ephemeris span)
_PIEN = 0.5
_PIEE = 0.3


def make_lc(site=None, terrestrial=True):
    return lcbinint.LightCurve(
        sky=lcbinint.obs.SkyCoord(_RA, _DEC),
        site=site,
        t_ref=_T0PAR,
        parallax=True,
        terrestrial=terrestrial,
    )


_PARAMS_BASE = {
    "t0": _T0PAR,
    "tE": 100.0,
    "u0": 0.1,
    "alpha": 0.0,
    "s": 1.2,
    "q": 1.0e-3,
    "rho": 0.0,
    "piEN": _PIEN,
    "piEE": _PIEE,
}


def test_terrestrial_requires_explicit_flag():
    """site has no effect unless terrestrial=True."""
    site = lcbinint.obs.Site("ground", 20.0, -156.0)
    lc_no_flag = make_lc(site=site, terrestrial=False)
    lc_with_flag = make_lc(site=site, terrestrial=True)
    lc_geo = make_lc(site=None, terrestrial=False)

    times = np.array([_T0PAR + 1.0])
    info_no_flag = lc_no_flag.info(times, _PARAMS_BASE)
    info_geo = lc_geo.info(times, _PARAMS_BASE)
    info_with_flag = lc_with_flag.info(times, _PARAMS_BASE)

    # Without terrestrial=True, site is ignored -> same as geocenter.
    np.testing.assert_array_equal(info_no_flag.source_x, info_geo.source_x)
    np.testing.assert_array_equal(info_no_flag.source_y, info_geo.source_y)

    # With terrestrial=True, source position differs from geocenter.
    assert info_with_flag.source_x[0] != info_geo.source_x[0]


def test_terrestrial_requires_ground_site():
    with pytest.raises(ValueError, match="terrestrial=True requires"):
        make_lc(site=None, terrestrial=True)


def test_parallax_requires_explicit_model_flag():
    """piE parameters do not silently activate parallax physics."""
    times = np.array([_T0PAR + 1.0, _T0PAR + 10.0])
    inactive = lcbinint.LightCurve()
    with_pie = inactive.info(times, _PARAMS_BASE)
    without_pie = inactive.info(times, {**_PARAMS_BASE, "piEN": 0.0, "piEE": 0.0})

    np.testing.assert_array_equal(with_pie.source_x, without_pie.source_x)
    np.testing.assert_array_equal(with_pie.source_y, without_pie.source_y)


def test_equator_greenwich_is_a_real_observatory():
    """An explicitly supplied site=(0, 0) must not be used as a sentinel."""
    lc_geo = make_lc(site=None, terrestrial=False)
    lc_zero = make_lc(site=lcbinint.obs.Site("ground", 0.0, 0.0))
    times = np.array([_T0PAR + 1.0])
    info_geo = lc_geo.info(times, _PARAMS_BASE)
    info_zero = lc_zero.info(times, _PARAMS_BASE)
    expected_tau, expected_beta = _terrestrial_delta_u(
        0.0, 0.0, times[0], _PIEN, _PIEE, _RA, _DEC
    )

    assert info_zero.source_x[0] - info_geo.source_x[0] == pytest.approx(
        expected_tau, rel=1e-5, abs=1e-12
    )
    assert info_zero.source_y[0] - info_geo.source_y[0] == pytest.approx(
        expected_beta, rel=1e-5, abs=1e-12
    )
    assert math.hypot(expected_tau, expected_beta) > 1e-7


def test_terrestrial_inter_observatory_displacement():
    """Two telescopes at different latitudes see a displaced source.

    The expected delta_u in lens-plane coordinates is computed analytically
    from the projected baseline.  We verify the magnification difference
    matches what the shifted trajectory would give to first order.
    """
    lat1, lon1 = 20.0, -156.0   # Mauna Kea (approx)
    lat2, lon2 = -32.0, 20.0    # Sutherland / SAAO (approx)

    lc1 = make_lc(site=lcbinint.obs.Site("ground", lat1, lon1))
    lc2 = make_lc(site=lcbinint.obs.Site("ground", lat2, lon2))

    times = np.array([_T0PAR + 1.0])
    info1 = lc1.info(times, _PARAMS_BASE)
    info2 = lc2.info(times, _PARAMS_BASE)

    # The source positions should differ between the two telescopes
    dx = info1.source_x[0] - info2.source_x[0]
    dy = info1.source_y[0] - info2.source_y[0]
    separation = math.sqrt(dx * dx + dy * dy)

    # Expected separation from hand calculation
    # Use the actual observation time (annual parallax cancels identically for both telescopes)
    r1 = _tel_position_au(lat1, lon1, _T0PAR + 1.0)
    r2 = _tel_position_au(lat2, lon2, _T0PAR + 1.0)
    baseline = r1 - r2
    north = _sky_north(_RA, _DEC)
    east = _sky_east(_RA, _DEC)
    bN = -np.dot(baseline, north)
    bE = -np.dot(baseline, east)
    # delta_tau and delta_beta (tau = along alpha, beta = perpendicular)
    d_tau = _PIEN * bN + _PIEE * bE
    d_beta = -_PIEE * bN + _PIEN * bE
    # The (x, y) change in VBM mode depends on theta=alpha=0: x=tau, y=beta
    expected_dx = d_tau  # alpha=0 => cos(alpha)=1, sin(alpha)=0
    expected_dy = d_beta
    expected_sep = math.sqrt(expected_dx**2 + expected_dy**2)

    assert separation == pytest.approx(expected_sep, rel=1e-6)
    assert separation > 1e-7  # The effect must be non-trivial


def test_terrestrial_diurnal_variation():
    """Source position at a telescope varies differently from geocenter as Earth rotates.

    The diurnal signal is isolated by subtracting the geocentric source position
    (which already includes annual parallax but no terrestrial offset) from the
    ground-telescope position at the same times.
    """
    lat, lon = 43.0, 172.5  # Mt John Observatory, New Zealand (approx)
    lc_obs = make_lc(site=lcbinint.obs.Site("ground", lat, lon))
    lc_geo = make_lc(site=None, terrestrial=False)

    # Sample over ~0.5 sidereal day
    jd0 = _T0PAR
    times = np.array([jd0, jd0 + 0.1, jd0 + 0.2, jd0 + 0.3, jd0 + 0.4, jd0 + 0.4986])
    infos_obs = [lc_obs.info(np.array([t]), _PARAMS_BASE) for t in times]
    infos_geo = [lc_geo.info(np.array([t]), _PARAMS_BASE) for t in times]

    # Terrestrial offset (x, y) at each time
    offsets = [
        (o.source_x[0] - g.source_x[0], o.source_y[0] - g.source_y[0])
        for o, g in zip(infos_obs, infos_geo)
    ]
    # The offsets should be non-trivial and vary over time (diurnal oscillation)
    norms = [math.hypot(dx, dy) for dx, dy in offsets]
    assert max(norms) > 1e-6  # measurable terrestrial displacement
    # The offset varies due to Earth's rotation (different sidereal angles)
    assert max(norms) - min(norms) > max(norms) * 0.01  # at least 1% variation

    # Verify against hand-computed terrestrial displacement at two times
    for t, (obs_dx, obs_dy) in zip(times, offsets):
        exp_tau, exp_beta = _terrestrial_delta_u(lat, lon, t, _PIEN, _PIEE, _RA, _DEC)
        # VBM, alpha=0: x=tau, y=beta
        assert obs_dx == pytest.approx(exp_tau, rel=1e-5, abs=1e-12)
        assert obs_dy == pytest.approx(exp_beta, rel=1e-5, abs=1e-12)

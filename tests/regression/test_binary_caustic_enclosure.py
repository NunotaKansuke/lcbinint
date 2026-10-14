"""Independent critical quartics check the continuous-caustic routing bound."""

import math

import numpy as np
import pytest


def _independent_caustic(s, q):
    a = -abs(s) if q < 1 else abs(s)
    q = min(q, 1/q)
    m1 = 1/(1+q)
    m2 = q*m1
    points = []
    # Offset phases deliberately avoid the centres/edges of native intervals.
    for angle in 2*np.pi*(np.arange(257)+.371)/257:
        phase = np.exp(1j*angle)
        coefficients = [m2*a*a, -2*m2*a, 1-phase*a*a, 2*phase*a, -phase]
        for root in np.roots(coefficients[::-1]):
            points.append(root-m1/(np.conj(root)-a)-m2/np.conj(root)-a*m1)
    return np.asarray(points)


@pytest.mark.parametrize("s,q", [
    (.45, .3), (.7, .3), (1., .1), (1.2, .01), (2.4, 1e-4),
    (.7, 1/.3), (1.2, 100.), (2.4, 1e4),
    # Exact topology boundaries can have colliding critical roots. An
    # unproved enclosure must return zero and preserve the ordinary fallback.
    ((1+.1**(1/3))**1.5/math.sqrt(1.1), .1),
])
def test_continuous_enclosure_contains_independent_critical_phases(s, q):
    lcbinint = pytest.importorskip("lcbinint")
    bound = lcbinint._native._binary_caustic_clearance_lower_bound
    caustic = _independent_caustic(s, q)
    # A caustic point cannot have a strictly positive distance lower bound.
    for z in caustic[::7]:
        assert bound(s, q, float(z.real), float(z.imag)) <= 2e-10
    # For remote sources the lower bound must never exceed even the nearest
    # independently computed caustic sample (whose distance is an upper bound
    # on true continuous clearance).
    for source in [.2+.3j, -.23+.3j, 2.+.7j, -.8-.5j]:
        lower = bound(s, q, source.real, source.imag)
        assert 0 <= lower <= np.min(abs(caustic-source)) + 2e-10


def test_remote_ghost_veto_is_resolved_without_detailed_caustic():
    lcbinint = pytest.importorskip("lcbinint")
    vbm_module = pytest.importorskip("VBMicrolensing")
    params = dict(t0=0., u0=.3, tE=1., alpha=0., rho=.003, q=.01, s=1.2)
    curve = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        param_type="vbm", nbin="auto", finite_source_tol=1e-3, finite_source_reltol=1e-3))
    times = np.linspace(-1, 1, 201)
    info = curve.info(times, params)
    assert set(info.finite_source_method_names) == {"point_source"}
    # Preserve the calibrated ghost veto. The alternate topology witness,
    # rather than a weakened constant, permits this remote point.
    assert info.point_source_safety_flags[77] & 2 == 0
    assert math.isinf(info.caustic_distances[77])
    assert lcbinint._native._binary_caustic_clearance_lower_bound(
        1.2, 100., -.23, .3) > 10*params["rho"]
    reference = vbm_module.VBMicrolensing()
    reference.Tol = reference.RelTol = 1e-3
    vp = [math.log(1.2), math.log(.01), .3, 0., math.log(.003), 0., 0.]
    expected = reference.BinaryLightCurve(vp, times.tolist())[0]
    np.testing.assert_allclose(info.magnifications, expected, rtol=2e-12, atol=2e-12)


def test_scalar_holonomic_api_shares_selector_and_rejects_uncertified_support():
    lcbinint = pytest.importorskip("lcbinint")
    from lcbinint.holonomic_fused import lightcurve_value

    curve = lcbinint.LightCurve(lens="binary", options=lcbinint.Options(
        param_type="vbm", nbin="auto", finite_source_tol=1e-3, finite_source_reltol=1e-3))
    params = dict(t0=0., u0=.3, tE=1., alpha=0., rho=.003, q=.01, s=1.2)
    times = np.linspace(-1, 1, 201)
    scalar = curve.holonomic_magnification(times, params)
    diagnostic = lightcurve_value(curve, times, params)
    np.testing.assert_allclose(scalar, diagnostic["mu"], rtol=2e-12, atol=2e-12)
    assert isinstance(curve.holonomic_magnification(0., params), float)
    # Changing the source/lens parameters must refresh the root and geometry
    # state. A last iterate lacking support must still fail in the scalar API.
    crossing = dict(params, u0=.03, alpha=-.08)
    crossing_times = np.linspace(-.2, .8, 201)
    diagnostic = lightcurve_value(curve, crossing_times, crossing)
    assert np.any(diagnostic["integrator_used"])
    assert not np.all(diagnostic["fully_converged"])
    with pytest.raises(RuntimeError):
        curve.holonomic_magnification(crossing_times, crossing)

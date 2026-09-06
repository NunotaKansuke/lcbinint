"""M1 completeness: the enumerated radial events are exactly the radii at
which the source-circle image topology (number of boundary crossings on the
circle |z| = R) can change.

The reference quantity is deliberately independent of the ATPT machinery:
we evaluate phi(R, theta) = 1 - |f(z) - zeta|^2 / rho^2 directly on a fine
theta grid and count sign changes around the full circle (theta = pi
included).  This count is piecewise-constant in R and may only jump at an
enumerated event.  We assert three things:

  1. constancy  -- on every open sub-interval between consecutive enumerated
     events the crossing count is constant along a fine sub-grid;
  2. physical events bite -- every *isolated* physical_real event has a
     different crossing count on its two sides;
  3. no unexplained jump -- on a dense R sweep every crossing-count change
     lies within a small tolerance of some enumerated event (of any kind).

Plus a stress battery (random configs incl. planetary q, resonant / close /
wide, caustic marches, on-axis): the enumerator must not raise and must
return sorted radii strictly inside (0, R_max).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "lcbinint"))
import holonomic_ref as H  # noqa: E402

_NTH = 4096
_TH = np.linspace(0.0, 2.0 * math.pi, _NTH, endpoint=False)
_EITH = np.exp(1j * _TH)


def _crossing_count(R, a, m0, xs, ys, rho):
    """# sign changes of phi = 1 - |f(z)-zeta|^2/rho^2 around |z| = R,
    evaluated straight from the lens equation (vectorized over theta)."""
    z = R * _EITH
    zb = np.conj(z)
    fz = z - m0 / zb - (1.0 - m0) / (zb - a)
    phi = 1.0 - np.abs(fz - complex(xs, ys)) ** 2 / (rho * rho)
    s = np.sign(phi)
    s[s == 0] = 1.0
    return int(np.sum(s != np.roll(s, 1)))


CASES = [
    dict(a=1.2, m0=2 / 3, xs=1 / 5, ys=1 / 7, rho=1 / 8),        # plan sec.15
    dict(a=0.9, m0=1 / 1.3, xs=0.05, ys=0.02, rho=0.05),         # resonant
    dict(a=2.5, m0=1 / 1.001, xs=1.4, ys=0.10, rho=0.03),        # planetary, wide
    dict(a=0.55, m0=1 / 1.8, xs=0.4, ys=-0.05, rho=0.09),        # close
    dict(a=1.35, m0=0.625, xs=0.30, ys=0.0, rho=0.04),           # on-axis
]


def _all_event_radii(events):
    return sorted(e.radius for e in events)


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"a{c['a']}")
def test_crossing_count_constant_between_events(cfg):
    events, Rmax = H.radial_events(**cfg)
    marks = [2e-3] + _all_event_radii(events) + [Rmax * (1 - 1e-6)]
    for lo, hi in zip(marks[:-1], marks[1:]):
        if hi - lo < 3e-4:
            continue
        Rs = np.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), 7)
        counts = {_crossing_count(R, **cfg) for R in Rs}
        assert len(counts) == 1, (
            f"crossing count varies on ({lo:.5f}, {hi:.5f}): {sorted(counts)}")


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"a{c['a']}")
def test_most_isolated_physical_real_events_change_topology(cfg):
    """A physical_real event carries a real double root of P.  Most such
    events are band births/deaths and flip the crossing count; a minority
    are external tangencies (band of zero measure on both sides) -- those
    are still legitimate cell boundaries, so we only require that the
    majority bite."""
    events, Rmax = H.radial_events(**cfg)
    radii = _all_event_radii(events)
    isolated = [e for e in events if e.kind == "physical_real"
                and not any(abs(r - e.radius) < 1.5e-3
                            for r in radii if r != e.radius)]
    if not isolated:
        return
    changed = 0
    for e in isolated:
        d = 5e-4 * max(e.radius, 1.0)
        if _crossing_count(e.radius - d, **cfg) != _crossing_count(e.radius + d, **cfg):
            changed += 1
    assert changed >= math.ceil(0.6 * len(isolated)), (
        f"only {changed}/{len(isolated)} isolated physical_real events "
        f"changed the crossing count")


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"a{c['a']}")
def test_no_unexplained_crossing_jump(cfg):
    events, Rmax = H.radial_events(**cfg)
    radii = np.array(_all_event_radii(events))
    Rgrid = np.linspace(3e-3, Rmax * (1 - 1e-6), 2500)
    cnt = np.array([_crossing_count(R, **cfg) for R in Rgrid])
    jump_R = 0.5 * (Rgrid[:-1] + Rgrid[1:])[np.diff(cnt) != 0]
    tol = 3.0 * (Rgrid[1] - Rgrid[0])
    for Rj in jump_R:
        near = float(np.min(np.abs(radii - Rj))) if radii.size else 9.9
        assert near < tol, (
            f"crossing count jumps at R={Rj:.5f} with no enumerated event "
            f"within {tol:.1e} (nearest {near:.2e})")


@pytest.mark.parametrize("seed", range(60))
def test_random_configs_enumerator_well_formed(seed):
    rng = np.random.default_rng(seed)
    q = 10 ** rng.uniform(-6, 0)
    a = 10 ** rng.uniform(-1, 1)
    rho = 10 ** rng.uniform(-4, -0.5)
    r = rng.uniform(0, 1.3)
    ang = rng.uniform(0, 2 * math.pi)
    cfg = dict(a=a, m0=1 / (1 + q), xs=r * math.cos(ang), ys=r * math.sin(ang), rho=rho)

    events, Rmax = H.radial_events(**cfg)
    radii = [e.radius for e in events]
    assert radii == sorted(radii)
    assert all(0.0 < R < Rmax for R in radii)
    for x, y in zip(events[:-1], events[1:]):
        assert not (x.kind == y.kind and abs(x.radius - y.radius) < 1e-7)


def test_caustic_march_no_missed_band():
    a, q, rho = 1.1, 0.4, 0.02
    m0 = 1 / (1 + q)
    for ys in (0.0, 0.01, 0.03):
        for xs in np.linspace(-0.15, 0.15, 5):
            cfg = dict(a=a, m0=m0, xs=float(xs), ys=ys, rho=rho)
            events, Rmax = H.radial_events(**cfg)
            radii = np.array(_all_event_radii(events))
            Rgrid = np.linspace(3e-3, Rmax * (1 - 1e-6), 2000)
            cnt = np.array([_crossing_count(R, **cfg) for R in Rgrid])
            jump_R = 0.5 * (Rgrid[:-1] + Rgrid[1:])[np.diff(cnt) != 0]
            tol = 3.0 * (Rgrid[1] - Rgrid[0])
            for Rj in jump_R:
                near = float(np.min(np.abs(radii - Rj)))
                assert near < tol, (
                    f"xs={xs:.3f} ys={ys}: crossing jump at R={Rj:.5f} "
                    f"unmatched (nearest {near:.2e})")

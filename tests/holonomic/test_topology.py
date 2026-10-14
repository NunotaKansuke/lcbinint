"""M2: radial-cell topology classification.

Checks, against phi taken straight from the lens equation:

  * every cell's ``(kind, n_crossings)`` is stable across the whole cell;
  * ``n_crossings`` is even and ``<= 4`` (``deg_t P = 4`` => ``<= 2`` arcs);
  * arc endpoints are genuine boundary points (phi = 0, region just inside);
  * the arc count only changes at an enumerated event, and matches the
    brute-force crossing count / 2;
  * the birth/continue/death lineage conserves arc count across every event;
  * ``empty`` / ``full`` cells really are empty / full.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import holonomic_ref as H
from holonomic_ref.topology import _phi, _phi_grid

_TWO_PI = 2.0 * math.pi

CASES = [
    dict(xs=1 / 5, ys=1 / 7, rho=1 / 8, q=0.5, a=1.2),          # plan sec.15
    dict(xs=0.05, ys=0.02, rho=0.05, q=0.3, a=0.9),             # resonant
    dict(xs=1.4, ys=0.10, rho=0.03, q=1e-3, a=2.5),             # planetary wide
    dict(xs=0.4, ys=-0.05, rho=0.09, q=0.8, a=0.55),            # close
    dict(xs=0.30, ys=0.0, rho=0.04, q=0.6, a=1.35),             # on-axis
    dict(xs=0.0, ys=0.0, rho=0.25, q=0.5, a=1.1),               # big source, on centre
]


def _params(cfg):
    return H.LensParams(barycentric=False, **cfg)


def _brute_crossings(R, cfg, n=4096):
    th = np.linspace(0.0, _TWO_PI, n, endpoint=False)
    v = _phi_grid(R, th, cfg["a"], 1 / (1 + cfg["q"]), cfg["xs"], cfg["ys"], cfg["rho"])
    s = np.sign(v)
    s[s == 0] = 1.0
    return int(np.sum(s != np.roll(s, 1)))


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"a{c['a']}_q{c['q']}")
def test_cells_are_internally_consistent(cfg):
    res = H.classify_cells(_params(cfg), samples_per_cell=6)
    assert res.status == "OK", [c for c in res.cells if c.status != "OK"]
    m0 = 1 / (1 + cfg["q"])
    for c in res.cells:
        assert c.n_crossings in (0, 2, 4)
        assert len(c.arcs) <= 2
        assert (c.kind == "arcs") == (c.n_crossings > 0)
        # stable across the cell
        for fr in (0.05, 0.35, 0.65, 0.95):
            R = c.r_lo + fr * (c.r_hi - c.r_lo)
            k, n, _ = H.arcs_at(R, cfg["a"], m0, cfg["xs"], cfg["ys"], cfg["rho"])
            assert (k, n) == (c.kind, c.n_crossings), (c.index, R, k, n)


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"a{c['a']}_q{c['q']}")
def test_arc_endpoints_are_boundary_points(cfg):
    res = H.classify_cells(_params(cfg), samples_per_cell=3)
    m0 = 1 / (1 + cfg["q"])
    args = (cfg["a"], m0, cfg["xs"], cfg["ys"], cfg["rho"])
    for c in res.cells:
        if not c.arcs:
            continue
        R = c.r_mid
        # smallest gap between consecutive endpoints bounds a safe probe offset
        pts = sorted(t for arc in c.arcs
                     for t in (arc.theta_enter, arc.theta_leave))
        gaps = [(b - a2) for a2, b in zip(pts, pts[1:])] + \
               [pts[0] + _TWO_PI - pts[-1]]
        eps = 0.2 * min(gaps)
        for arc in c.arcs:
            for th in (arc.theta_enter, arc.theta_leave):
                assert abs(_phi(R, th, *args)) < 1e-7
            assert _phi(R, arc.midpoint, *args) > 0.0            # region interior
            assert _phi(R, arc.theta_enter - eps, *args) < 0.0   # just outside
            assert _phi(R, arc.theta_leave + eps, *args) < 0.0


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"a{c['a']}_q{c['q']}")
def test_arc_count_matches_brute_force(cfg):
    res = H.classify_cells(_params(cfg), samples_per_cell=3)
    for c in res.cells:
        assert self_consistent(c, cfg)


def self_consistent(c, cfg):
    if c.kind == "empty":
        return _brute_crossings(c.r_mid, cfg) == 0
    if c.kind == "full":
        return _brute_crossings(c.r_mid, cfg) == 0
    return _brute_crossings(c.r_mid, cfg) == c.n_crossings


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"a{c['a']}_q{c['q']}")
def test_transitions_conserve_arc_count(cfg):
    res = H.classify_cells(_params(cfg), samples_per_cell=3)
    by_lower: dict[int, list] = {}
    by_upper: dict[int, list] = {}
    for t in res.transitions:
        by_lower.setdefault(t.lower_cell, []).append(t)
        by_upper.setdefault(t.upper_cell, []).append(t)
    for c in res.cells:
        outgoing = by_lower.get(c.index, [])
        if outgoing:
            k = sum(t.kind in ("continue", "death") for t in outgoing)
            assert k == len(c.arcs), (c.index, "lower", k, len(c.arcs))
        incoming = by_upper.get(c.index, [])
        if incoming:
            k = sum(t.kind in ("continue", "birth") for t in incoming)
            assert k == len(c.arcs), (c.index, "upper", k, len(c.arcs))


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"a{c['a']}_q{c['q']}")
def test_adjacent_cells_differ_only_across_a_real_change(cfg):
    res = H.classify_cells(_params(cfg), samples_per_cell=3)
    for lo, hi in zip(res.cells[:-1], res.cells[1:]):
        if (lo.kind, lo.n_crossings) == (hi.kind, hi.n_crossings):
            continue
        # the shared boundary must be a real crossing-count change nearby
        R = 0.5 * (lo.r_hi + hi.r_lo)
        d = 1e-4 * max(R, 1.0)
        assert _brute_crossings(R - d, cfg) != _brute_crossings(R + d, cfg) \
            or lo.kind != hi.kind


def test_full_circle_state_detected():
    # a large source sitting on the primary: some inner radius is fully imaged
    p = H.LensParams(xs=0.0, ys=0.0, rho=0.4, q=0.4, a=1.6, barycentric=False)
    res = H.classify_cells(p, samples_per_cell=4)
    kinds = {c.kind for c in res.cells}
    assert "full" in kinds
    full = next(c for c in res.cells if c.kind == "full")
    rng = np.random.default_rng(0)
    for th in rng.uniform(0, _TWO_PI, 200):
        assert _phi(full.r_mid, th, 1.6, 1 / 1.4, 0.0, 0.0, 0.4) >= 0.0


@pytest.mark.parametrize("seed", range(30))
def test_random_configs_classify_without_error(seed):
    rng = np.random.default_rng(seed)
    cfg = dict(
        q=float(10 ** rng.uniform(-5, 0)),
        a=float(10 ** rng.uniform(-0.7, 0.7)),
        rho=float(10 ** rng.uniform(-3, -0.6)),
    )
    r = rng.uniform(0, 1.2)
    ang = rng.uniform(0, _TWO_PI)
    cfg["xs"], cfg["ys"] = r * math.cos(ang), r * math.sin(ang)
    res = H.classify_cells(_params(cfg), samples_per_cell=4)
    assert res.cells
    assert res.cells[0].r_lo == 0.0
    assert abs(res.cells[-1].r_hi - res.r_max) < 1e-9
    for lo, hi in zip(res.cells[:-1], res.cells[1:]):
        assert abs(lo.r_hi - hi.r_lo) < 1e-9
    bad = [c.index for c in res.cells if c.status != "OK"]
    assert not bad, f"uncertain cells {bad} in {cfg}"

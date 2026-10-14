"""M3: period reduction -- 7 monomial forms -> residue-free 6 forms -> observable.

Checks, per plan sec. 5-7:

  * the residue coefficients b1,b2,b3 from the closed formula equal the
    Laurent series of (Q/q8/t^8)^{-1/2} at infinity  (exact);
  * the Gauss-Manin coefficient identity
        -1/2 t^k Q_R == C_k Q + S_k' Q - 1/2 S_k Q_t
    holds exactly for k = 0..6  (exact rational, at sample t);
  * the residue conditions:  h3' := h3 + b1 h4 + b2 h5 + b3 h6 == 0 exactly,
    and every psi row is residue-free at infinity  (exact);
  * the reduced period values -- both the 7D monomial sum sum_k h_k I_k and
    the 6D residue-free form c^T (W I) -- reproduce the bare angular integral
    (rho R / 2) int sqrt(phi) dtheta to rtol ~1e-10;
  * a soft end-to-end check that the closed-contour period vector satisfies
    d/dR Pi = C Pi (contour quadrature -- loose tolerance).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import sympy as sp

import holonomic_ref as H
from holonomic_ref.connection import gm_polynomials, q_coeffs_exact
from holonomic_ref.period_reduction import (
    arc_chart,
    closed_period_eta,
    h_coeffs_exact,
    half_period_obs_angular,
    half_period_obs_reduced,
    laurent_b,
    observed_covector_exact,
    residue_at_infinity,
    residue_b_exact,
)

CASES = [
    dict(xs=1 / 5, ys=1 / 7, rho=1 / 8, q=0.5, a=1.2),      # plan sec.15
    dict(xs=0.05, ys=0.02, rho=0.05, q=0.3, a=0.9),         # resonant
    dict(xs=1.4, ys=0.10, rho=0.03, q=1e-3, a=2.5),         # planetary wide
    dict(xs=0.4, ys=-0.05, rho=0.09, q=0.8, a=0.55),        # close
    dict(xs=0.30, ys=0.0, rho=0.04, q=0.6, a=1.35),         # on-axis
    dict(xs=0.0, ys=0.0, rho=0.25, q=0.5, a=1.1),           # big source on centre
]

_IDS = [f"a{c['a']}_q{c['q']}" for c in CASES]


def _rat(x):
    """Exact rational for a Python float / int / sympy number (dyadic for floats)."""
    return sp.Rational(x)


@functools.lru_cache(maxsize=None)
def _classify(cfg_id):
    cfg = CASES[_IDS.index(cfg_id)]
    return H.classify_cells(H.LensParams(barycentric=False, **cfg),
                            samples_per_cell=3)


def _arc_samples(cfg):
    """(R, chart-params, arc) for every arc of every 'arcs' cell."""
    m0 = 1.0 / (1.0 + cfg["q"])
    res = _classify(_IDS[CASES.index(cfg)])
    out = []
    for c in res.cells:
        if c.kind != "arcs":
            continue
        for arc in c.arcs:
            params, _tt, _s = arc_chart(cfg["a"], m0, cfg["xs"], cfg["ys"],
                                        cfg["rho"], arc)
            out.append((c.index, c.r_mid, params, arc,
                        c.r_hi - c.r_lo))
    return m0, out


# --------------------------------------------------------------- residue b ---
@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_residue_b_formula_equals_series(cfg):
    m0, samples = _arc_samples(cfg)
    assert samples
    for _idx, R, params, _arc, _w in samples:
        qc = q_coeffs_exact(_rat(R), *(_rat(x) for x in params))
        b_formula = residue_b_exact(qc)                     # (b1, b2, b3)
        b_series = laurent_b(qc, n=3)                       # [b0, b1, b2, b3]
        for j in range(3):
            assert sp.simplify(b_formula[j] - b_series[j + 1]) == 0


# ---------------------------------------------------- Gauss-Manin identity ---
@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_gauss_manin_coefficient_identity(cfg):
    """-1/2 t^k Q_R == C_k Q + S_k' Q - 1/2 S_k Q_t  for k=0..6.

    ``connection`` builds ``C_k``/``S_k`` exactly and raises unless the exact
    rational residual is 0; here we re-check the returned (float) polynomials
    at sample points as an independent guard.
    """
    m0, samples = _arc_samples(cfg)
    seen = 0
    tt = np.array([-1.3, -0.4, 0.25, 0.9, 1.7])
    for _idx, R, params, _arc, _w in samples:
        try:
            gp = gm_polynomials(_rat(R), *(_rat(x) for x in params))
        except ValueError:
            continue                            # branch collision -> fail closed
        seen += 1
        Q = np.polynomial.Polynomial(gp["Q"])
        QR = np.polynomial.Polynomial(gp["Q_R"])
        Qt = np.polynomial.Polynomial(gp["Q_t"])
        scale = max(abs(gp["Q"]).max(), abs(gp["Q_R"]).max())
        for k in range(7):
            S = np.polynomial.Polynomial(gp["S"][k])
            C = np.polynomial.Polynomial(gp["C"][k])
            lhs = -0.5 * tt**k * QR(tt)
            rhs = C(tt) * Q(tt) + S.deriv()(tt) * Q(tt) - 0.5 * S(tt) * Qt(tt)
            assert np.max(np.abs(lhs - rhs)) < 1e-6 * scale, (cfg["a"], R, k)

    if cfg["xs"] == 0.0 and cfg["ys"] == 0.0:
        # exactly-symmetric source on the axis: Q carries a structural
        # repeated factor at every R, so the connection is degenerate
        # everywhere -- the module fails closed (documented in checkpoint M3).
        assert seen == 0
    else:
        assert seen                              # not every sample degenerate


# ------------------------------------------------------ residue conditions ---
@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_residue_conditions_exact(cfg):
    m0, samples = _arc_samples(cfg)
    for _idx, R, params, _arc, _w in samples:
        c, h3p = observed_covector_exact(_rat(R), *(_rat(x) for x in params))
        assert sp.simplify(h3p) == 0                        # P/(A Y) is 2nd kind

        qc = q_coeffs_exact(_rat(R), *(_rat(x) for x in params))
        b1, b2, b3 = residue_b_exact(qc)
        psi_rows = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, -b1, 1, 0, 0],
            [0, 0, 0, -b2, 0, 1, 0],
            [0, 0, 0, -b3, 0, 0, 1],
        ]
        for row in psi_rows:
            assert sp.simplify(residue_at_infinity(row, qc)) == 0
        # eta_3 alone must carry a non-zero residue (it is the odd one out)
        assert sp.simplify(residue_at_infinity([0, 0, 0, 1, 0, 0, 0], qc)) != 0


# ------------------------------------------------------ reduced period value -
@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_reduced_period_matches_angular_integral(cfg):
    m0, samples = _arc_samples(cfg)
    worst = 0.0
    for idx, R, _params, arc, _w in samples:
        base = (R, cfg["a"], m0, cfg["xs"], cfg["ys"], cfg["rho"], arc)
        ang = half_period_obs_angular(*base)
        red7 = half_period_obs_reduced(*base, basis="7D")
        red6 = half_period_obs_reduced(*base, basis="6D")
        e7 = abs(red7 - ang) / abs(ang)
        e6 = abs(red6 - ang) / abs(ang)
        worst = max(worst, e7, e6)
        # tiny arcs near a birth/death lose relative precision (ang -> 0);
        # give them a floor on the absolute error instead
        tol = 1e-10 if ang > 1e-3 else 5e-8
        assert e7 < tol, (idx, R, "7D", e7, ang)
        assert e6 < tol, (idx, R, "6D", e6, ang)
    assert worst < 5e-8


# ------------------------------------------- soft end-to-end connection check -
def test_closed_period_satisfies_connection_ode():
    """d/dR oint psi == C @ oint psi -- contour quadrature, loose tolerance."""
    cfg = CASES[0]
    m0 = 1.0 / (1.0 + cfg["q"])
    res = _classify(_IDS[0])
    from holonomic_ref.connection import connection_matrix

    checked = 0
    for c in res.cells:
        if c.kind != "arcs" or (c.r_hi - c.r_lo) < 0.05:
            continue
        arc = c.arcs[0]
        params, _tt, _s = arc_chart(cfg["a"], m0, cfg["xs"], cfg["ys"],
                                    cfg["rho"], arc)
        R = c.r_mid
        h = 5e-4
        try:
            stncl = [closed_period_eta(R + d * h, cfg["a"], m0, cfg["xs"],
                                       cfg["ys"], cfg["rho"], arc)[0]
                     for d in (-2, -1, 1, 2)]
            P0 = closed_period_eta(R, cfg["a"], m0, cfg["xs"], cfg["ys"],
                                   cfg["rho"], arc)[0]
            M = connection_matrix(R, *params)
        except ValueError:
            continue
        dP = (stncl[0] - 8 * stncl[1] + 8 * stncl[2] - stncl[3]) / (12 * h)
        pred = M @ P0
        rel = float(np.max(np.abs(dP - pred) / (np.abs(pred) + 1e-12)))
        assert rel < 1e-4, (c.index, rel)
        checked += 1
    assert checked >= 2

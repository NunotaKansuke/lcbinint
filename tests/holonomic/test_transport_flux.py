"""M4: root pair, numeric fast paths, pointwise transport, and flux assembly.

Covers, per plan sec. 5-8 and the M4 gate:

  * the numeric fast paths (``boundary_quartic_dR``, ``q_coeffs_numeric`` and
    its ``d/dR``, ``h_coeffs_numeric``, ``connection_matrix_numeric``) agree
    with the sympy oracles / finite differences they shadow;
  * the boundary root pair ``(m, v)``: endpoint round-trip, ``delta_theta``
    vs ``arc.measure``, ``E = O = 0`` at real boundary roots, the analytic
    ``eo_jacobian`` vs finite differences, the tangency determinant, and the
    radial-motion solves ``root_pair_dR`` / ``endpoint_dR`` vs finite
    differences of re-solved roots;
  * ``psi_connection`` closure residual ~ 0 and agreement with the exact
    oracle ``psi_connection_exact``;
  * the pointwise connection ODE ``dI/dR == C_eta(R) I(R)`` for the
    moving-endpoint segment period (the flux re-anchors this at every node);
  * ``epoch_flux`` F0 / F_1/2 vs the fully independent nested quadrature
    ``image_plane_flux``, and the derived magnifications; per-cell
    Gauss-Manin condition numbers are recorded and finite.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import holonomic_ref as H
from holonomic_ref import LensParams
from holonomic_ref.connection import (
    connection_matrix,
    connection_matrix_numeric,
    q_coeffs_exact,
    q_coeffs_numeric,
)
from holonomic_ref.period_reduction import (
    arc_chart,
    h_coeffs_exact,
    h_coeffs_numeric,
)
from holonomic_ref.polynomial_family import boundary_quartic, boundary_quartic_dR
from holonomic_ref.root_pair import (
    RootPair,
    endpoint_dR,
    eo_jacobian,
    eo_residuals,
    from_endpoints,
    root_pair_dR,
    tangency_determinant,
)
from holonomic_ref.seed import seed_eta
from holonomic_ref.topology import arcs_at
from holonomic_ref.transport import (
    cell_conditioning,
    psi_connection,
    psi_connection_exact,
)

CASES = [
    dict(xs=1 / 5, ys=1 / 7, rho=1 / 8, q=0.5, a=1.2),      # plan sec.15
    dict(xs=0.05, ys=0.02, rho=0.05, q=0.3, a=0.9),         # resonant
    dict(xs=0.4, ys=-0.05, rho=0.09, q=0.8, a=0.55),        # close
    dict(xs=1.4, ys=0.10, rho=0.03, q=1e-3, a=2.5),         # planetary wide
    dict(xs=0.30, ys=1e-6, rho=0.04, q=0.6, a=1.35),        # near on-axis
]
_IDS = [f"a{c['a']}_q{c['q']}" for c in CASES]

_FLUX_CASES = CASES[:3]              # the three well-resolved, non-grazing configs
_FLUX_IDS = _IDS[:3]

# the near-on-axis config (ys = 1e-6) sits next to the xs = ys = 0 singular
# locus where Q nearly carries a repeated factor -- the connection is
# genuinely ill-conditioned there, so the float surrogate and the pointwise
# transport identity are only exercised on the four off-axis configs.
_CONN_CASES = CASES[:4]
_CONN_IDS = _IDS[:4]


def _params(cfg):
    return LensParams(barycentric=False, **cfg)


def _m0(cfg):
    return 1.0 / (1.0 + cfg["q"])


def _arc_cells(cfg):
    """[(cell_index, R_mid, chart_params, arc, width), ...] over 'arcs' cells."""
    p = _params(cfg)
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    out = []
    for c in H.classify_cells(p, samples_per_cell=3).cells:
        if c.kind != "arcs":
            continue
        for arc in c.arcs:
            cp, _tt, _s = arc_chart(a, m0, xs, ys, rho, arc)
            out.append((c.index, c.r_mid, cp, arc, c.r_hi - c.r_lo))
    return out


# ------------------------------------------------------- numeric fast paths ---
@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_boundary_quartic_dR_vs_fd(cfg):
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    h = 1e-6
    for R in (0.4, 0.8, 1.3, 2.1):
        an = np.array(boundary_quartic_dR(R, a, m0, xs, ys, rho))
        fd = (np.array(boundary_quartic(R + h, a, m0, xs, ys, rho))
              - np.array(boundary_quartic(R - h, a, m0, xs, ys, rho))) / (2 * h)
        scale = np.abs(fd).max() + 1.0
        assert np.max(np.abs(an - fd)) < 1e-6 * scale, (cfg["a"], R)


@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_q_and_h_numeric_vs_exact(cfg):
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    import sympy as sp

    for R in (0.5, 1.1, 1.9):
        qn = np.asarray(q_coeffs_numeric(R, a, m0, xs, ys, rho), dtype=float)
        qe = np.array([float(x) for x in
                       q_coeffs_exact(sp.Rational(R), *(sp.Rational(v)
                       for v in (a, m0, xs, ys, rho)))])
        assert np.allclose(qn, qe, rtol=1e-11, atol=1e-11 * np.abs(qe).max())

        hn = np.asarray(h_coeffs_numeric(R, a, m0, xs, ys, rho), dtype=float)
        he = np.array([float(x) for x in
                       h_coeffs_exact(sp.Rational(R), *(sp.Rational(v)
                       for v in (a, m0, xs, ys, rho)))])
        assert np.allclose(hn, he, rtol=1e-10,
                           atol=1e-10 * (np.abs(he).max() + 1.0))


@pytest.mark.parametrize("cfg", _CONN_CASES, ids=_CONN_IDS)
def test_connection_matrix_numeric_vs_exact(cfg):
    """Float Gauss-Manin surrogate vs the sympy oracle, on the cells where it
    is meant to be usable -- ``|q8|`` not near the ``theta -> pi`` degree drop
    (there it hands off to the oracle, tested separately below)."""
    checked = 0
    for _idx, R, cp, _arc, _w in _arc_cells(cfg):
        q = np.asarray(q_coeffs_numeric(R, *cp), dtype=float)
        if abs(q[8]) < 1e-1 * np.abs(q).max():
            continue
        try:
            Ce = connection_matrix(R, *cp)
        except ValueError:
            continue                          # branch collision -> oracle fails closed
        Cn = connection_matrix_numeric(R, *cp)
        scale = np.abs(Ce).max() + 1.0
        assert np.max(np.abs(Ce - Cn)) < 1e-4 * scale, (cfg["a"], R)
        checked += 1
    assert checked


def test_connection_matrix_numeric_hands_off_on_degree_drop():
    """When ``|q8|`` is tiny the surrogate returns the exact oracle verbatim."""
    cfg = CASES[1]                            # resonant -- has theta~pi cells
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    seen = 0
    for _idx, R, cp, _arc, _w in _arc_cells(cfg):
        q = np.asarray(q_coeffs_numeric(R, *cp), dtype=float)
        if abs(q[8]) >= 1e-6 * np.abs(q).max():
            continue
        try:
            Ce = connection_matrix(R, *cp)
        except ValueError:
            continue
        assert np.array_equal(connection_matrix_numeric(R, *cp), Ce)
        seen += 1
    if not seen:
        pytest.skip("no near-degenerate cell in this config")


# ------------------------------------------------------------- root pair -----
def test_root_pair_endpoint_roundtrip():
    rp = from_endpoints(-0.37, 1.42)
    assert math.isclose(rp.t_minus, -0.37, rel_tol=0, abs_tol=1e-14)
    assert math.isclose(rp.t_plus, 1.42, rel_tol=0, abs_tol=1e-14)
    rp2 = RootPair(rp.m, rp.v)
    assert (rp2.t_minus, rp2.t_plus) == (rp.t_minus, rp.t_plus)


def test_tangency_determinant_double_root():
    pc = [1.0, 0.0, -2.0, 0.0, 1.0]                 # (t^2 - 1)^2
    for m in (-1.0, 1.0):
        (E_m, E_v), (O_m, O_v) = eo_jacobian(RootPair(m, 0.0), pc)
        det = E_m * O_v - E_v * O_m
        assert math.isclose(det, tangency_determinant(m, pc), rel_tol=1e-12)
        assert math.isclose(det, -32.0, rel_tol=1e-12)


def test_delta_theta_matches_arctan_span():
    """``RootPair.delta_theta`` == ``2 (arctan t_+ - arctan t_-)`` (theta = 2 arctan t)."""
    for t_lo, t_hi in [(-0.3, 1.4), (-2.1, -0.4), (0.2, 5.0), (-4.0, 4.0)]:
        rp = from_endpoints(t_lo, t_hi)
        want = 2.0 * (math.atan(t_hi) - math.atan(t_lo))
        assert math.isclose(rp.delta_theta, want, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_eo_zero_at_boundary_roots(cfg):
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    seen = 0
    for _idx, R, _cp, _arc, _w in _arc_cells(cfg):
        kind, _n, arcs = arcs_at(R, a, m0, xs, ys, rho)
        if kind != "arcs":
            continue
        pc = boundary_quartic(R, a, m0, xs, ys, rho)
        pscale = max(abs(c) for c in pc) + 1.0
        for arc in arcs:
            te, tl = arc.theta_enter, arc.theta_leave
            # skip arcs straddling theta = pi (t = tan(theta/2) blows up)
            if min(abs(((te - math.pi) % (2 * math.pi)) - math.pi),
                   abs(((tl - math.pi) % (2 * math.pi)) - math.pi)) < 0.05:
                continue
            t_lo = math.tan(0.5 * (te % (2 * math.pi)))
            t_hi = math.tan(0.5 * (tl % (2 * math.pi)))
            rp = from_endpoints(*sorted((t_lo, t_hi)))
            E, O = eo_residuals(rp, pc)
            assert abs(E) < 1e-7 * pscale, (cfg["a"], R)
            assert abs(O) < 1e-7 * pscale, (cfg["a"], R)
            seen += 1
    assert seen


@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_eo_jacobian_vs_fd(cfg):
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    checked = 0
    for _idx, R, cp, _arc, _w in _arc_cells(cfg):
        pc = boundary_quartic(R, *cp)
        roots = np.roots(pc[::-1])
        real = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
        if len(real) < 2:
            continue
        rp = from_endpoints(real[0], real[-1])
        (E_m, E_v), (O_m, O_v) = eo_jacobian(rp, pc)
        hm = 1e-6 * (abs(rp.m) + 1.0)
        hv = 1e-6 * (abs(rp.v) + 1.0)
        Em = ((eo_residuals(RootPair(rp.m + hm, rp.v), pc)[0]
               - eo_residuals(RootPair(rp.m - hm, rp.v), pc)[0]) / (2 * hm))
        Ev = ((eo_residuals(RootPair(rp.m, rp.v + hv), pc)[0]
               - eo_residuals(RootPair(rp.m, rp.v - hv), pc)[0]) / (2 * hv))
        Om = ((eo_residuals(RootPair(rp.m + hm, rp.v), pc)[1]
               - eo_residuals(RootPair(rp.m - hm, rp.v), pc)[1]) / (2 * hm))
        Ov = ((eo_residuals(RootPair(rp.m, rp.v + hv), pc)[1]
               - eo_residuals(RootPair(rp.m, rp.v - hv), pc)[1]) / (2 * hv))
        sc = max(abs(E_m), abs(E_v), abs(O_m), abs(O_v)) + 1.0
        assert abs(E_m - Em) < 1e-5 * sc
        assert abs(E_v - Ev) < 1e-5 * sc
        assert abs(O_m - Om) < 1e-5 * sc
        assert abs(O_v - Ov) < 1e-5 * sc
        checked += 1
    assert checked


def _nearest_real_root(R, args, target):
    pc = boundary_quartic(R, *args)
    roots = np.roots(np.asarray(pc)[::-1])
    real = [r.real for r in roots if abs(r.imag) < 1e-6]
    return min(real, key=lambda r: abs(r - target))


@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_root_pair_dR_and_endpoint_dR_vs_fd(cfg):
    """``root_pair_dR`` / ``endpoint_dR`` vs finite differences of re-solved roots."""
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    args = (a, m0, xs, ys, rho)
    checked = 0
    seen_R = set()
    for _idx, R, _cp, _arc, _w in _arc_cells(cfg):
        if _w < 0.02:                          # steep root motion; FD ref too noisy
            continue
        Rk = round(R, 6)
        if Rk in seen_R:
            continue
        seen_R.add(Rk)
        kind, _n, arcs = arcs_at(R, *args)
        if kind != "arcs":
            continue
        for arc in arcs:
            te, tl = arc.theta_enter % (2 * math.pi), arc.theta_leave % (2 * math.pi)
            if min(abs(((te - math.pi) % (2 * math.pi)) - math.pi),
                   abs(((tl - math.pi) % (2 * math.pi)) - math.pi)) < 0.3:
                continue                       # endpoint racing to the theta=pi pole
            t_lo, t_hi = sorted((math.tan(0.5 * te), math.tan(0.5 * tl)))
            rp = from_endpoints(t_lo, t_hi)
            h = 1e-5
            lo_p = _nearest_real_root(R + h, args, t_lo)
            lo_m = _nearest_real_root(R - h, args, t_lo)
            hi_p = _nearest_real_root(R + h, args, t_hi)
            hi_m = _nearest_real_root(R - h, args, t_hi)
            rp_p = from_endpoints(lo_p, hi_p)
            rp_m = from_endpoints(lo_m, hi_m)
            dm_fd = (rp_p.m - rp_m.m) / (2 * h)
            dv_fd = (rp_p.v - rp_m.v) / (2 * h)
            try:
                dm, dv = root_pair_dR(rp, R, *args)
            except ValueError:
                continue
            # dm, dv can be very steep in narrow cells -- the central
            # difference of np.roots then carries ~1e-4 relative noise, so
            # this only guards against sign / factor errors in the 2x2 solve.
            assert abs(dm - dm_fd) < 5e-4 * (abs(dm_fd) + 1.0), (cfg["a"], R)
            assert abs(dv - dv_fd) < 5e-4 * (abs(dv_fd) + 1.0), (cfg["a"], R)
            dt_fd = (hi_p - hi_m) / (2 * h)
            dt = endpoint_dR(t_hi, R, *args)
            assert abs(dt - dt_fd) < 5e-4 * (abs(dt_fd) + 1.0), (cfg["a"], R)
            checked += 1
    if not checked:
        pytest.skip("no cell wide enough for a clean radial finite difference")


# ------------------------------------------------------- psi connection ------
@pytest.mark.parametrize("cfg", _CONN_CASES, ids=_CONN_IDS)
def test_psi_connection_second_kind_closure(cfg):
    """The residue-free basis closes under d/dR: ``M[:,3] + sum b_j M[:,3+j] ~ 0``
    (exact oracle -- this is the second-kind statement, must hold everywhere)."""
    checked = 0
    for _idx, R, cp, _arc, _w in _arc_cells(cfg):
        try:
            Cp, resid = psi_connection_exact(R, cp)
        except ValueError:
            continue
        assert np.max(np.abs(resid)) < 1e-7 * (np.abs(Cp).max() + 1.0), (cfg["a"], R)
        checked += 1
    assert checked


@pytest.mark.parametrize("cfg", _CONN_CASES, ids=_CONN_IDS)
def test_psi_connection_numeric_vs_oracle(cfg):
    """Numeric ``psi_connection`` matches the oracle on benign (non-degree-drop) cells."""
    checked = 0
    for _idx, R, cp, _arc, _w in _arc_cells(cfg):
        q = np.asarray(q_coeffs_numeric(R, *cp), dtype=float)
        if abs(q[8]) < 1e-1 * np.abs(q).max():
            continue
        try:
            Ce, _re = psi_connection_exact(R, cp)
        except ValueError:
            continue
        Cp, resid = psi_connection(R, cp)
        assert np.max(np.abs(resid)) < 1e-5 * (np.abs(Cp).max() + 1.0)
        assert np.max(np.abs(Cp - Ce)) < 1e-4 * (np.abs(Ce).max() + 1.0), (cfg["a"], R)
        checked += 1
    assert checked


# -------------------------------------------- pointwise connection ODE ------
@pytest.mark.parametrize("cfg", _CONN_CASES, ids=_CONN_IDS)
def test_moving_endpoint_period_satisfies_connection(cfg):
    """dI/dR == C_eta(R) I(R) for the moving-endpoint segment period.

    This is the identity the reference flux relies on -- the arc endpoints
    are re-found at every radius (``seed_eta`` -> ``arcs_at``) so the
    segment period obeys the exact Gauss-Manin connection.
    """
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    checked = 0
    for _idx, R, cp, arc, w in _arc_cells(cfg):
        if w < 0.04:
            continue
        h = min(1e-4, 0.15 * w)
        try:
            stncl = [seed_eta(R + d * h, a, m0, xs, ys, rho, arc)[0]
                     for d in (-2, -1, 1, 2)]
            I0, _arc0 = seed_eta(R, a, m0, xs, ys, rho, arc)
            Ce = connection_matrix(R, *cp)
        except ValueError:
            continue
        dI = (stncl[0] - 8 * stncl[1] + 8 * stncl[2] - stncl[3]) / (12 * h)
        pred = Ce @ I0
        rel = np.max(np.abs(dI - pred)) / (np.max(np.abs(pred)) + 1e-14)
        assert rel < 2e-5, (cfg["a"], _idx, rel)
        checked += 1
    if not checked:
        pytest.skip("no cell wide enough for a clean radial finite difference")


# --------------------------------------------------------- flux assembly ----
@pytest.mark.parametrize("cfg", _FLUX_CASES, ids=_FLUX_IDS)
def test_epoch_flux_matches_independent_quadrature(cfg):
    p = _params(cfg)
    topo = H.classify_cells(p)
    ev = sorted({c.r_lo for c in topo.cells} | {c.r_hi for c in topo.cells})

    fr = H.epoch_flux(p)
    F0i, Fhi = H.image_plane_flux(p, event_radii=ev)

    assert abs(fr.F0 - F0i) / F0i < 1e-4, fr.status
    assert abs(fr.F_half - Fhi) / Fhi < 1e-4, fr.status
    assert fr.status in ("OK", "OK_ESTIMATED", "TOPOLOGY_UNCERTAIN")

    # mu_uniform = F0 / (pi rho^2) and the linear-LD blend are self-consistent
    assert math.isclose(fr.mu_uniform, F0i / (math.pi * cfg["rho"] ** 2),
                        rel_tol=1e-4)
    assert fr.mu_linear_ld(0.0) == pytest.approx(fr.mu_uniform, rel=1e-14)
    mu_half = fr.mu_linear_ld(0.6)
    assert 1.0 < mu_half < fr.mu_uniform * 1.5


@pytest.mark.parametrize("cfg", _FLUX_CASES, ids=_FLUX_IDS)
def test_cell_condition_numbers_recorded(cfg):
    """The M4 gate record: exact per-cell Gauss-Manin condition numbers are
    finite and the residue-free basis closure residual is ~0."""
    p = _params(cfg)
    cells = [c for c in H.classify_cells(p).cells if c.kind == "arcs"]
    assert cells
    a, m0, xs, ys, rho = cfg["a"], _m0(cfg), cfg["xs"], cfg["ys"], cfg["rho"]
    got = 0
    for c in cells[:2]:
        cp, _tt, _s = arc_chart(a, m0, xs, ys, rho, c.arcs[0])
        cond = cell_conditioning(c.r_lo + 0.05 * (c.r_hi - c.r_lo),
                                 c.r_hi - 0.05 * (c.r_hi - c.r_lo), cp, n=5)
        for key in ("cond_eta_max", "cond_psi_max", "psi_closure_resid"):
            assert math.isfinite(cond[key]) and cond[key] >= 0.0
        assert cond["psi_closure_resid"] < 1e-6
        assert cond["cond_eta_max"] >= 1.0
        got += 1
    assert got


def test_epoch_flux_records_conditioning_when_asked():
    fr = H.epoch_flux(_params(_FLUX_CASES[0]), record_conditioning=True)
    assert any(c.cond and "error" not in c.cond
               for c in fr.cells if c.kind == "arcs")

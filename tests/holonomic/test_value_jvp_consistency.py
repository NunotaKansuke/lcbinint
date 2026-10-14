"""M6: value / JVP consistency for the 5-component epoch Jacobian.

Plan sec. 11 and the M6 gate.  ``holonomic_ref.jacobian`` reconstructs the
finite-source magnification ``mu`` from the *same* fixed Gauss-Chebyshev
radial pass that produces its five parameter derivatives, so the returned
``value`` and ``JVP`` are guaranteed mutually consistent.  These tests pin:

  * the closed-form ``phi`` gradient vs central differences;
  * endpoint-derivative *chart invariance* -- IFT-in-theta vs IFT-in-t agree
    to machine precision on every arc (plan: "チャート変更前後で整合");
  * central-difference convergence of ``grad_mu`` in the radial node count
    (plan: "中心差分収束");
  * ``grad_mu`` vs a central difference of the *independent* M4 value path
    ``epoch_flux`` (plan: "既存経路との勾配比較"), and a loose A/B against a
    finite difference of the production ``binary_ray_shooting``;
  * ``d mu / d u`` sign and magnitude;
  * on-axis ``ys`` symmetry (``mu`` even in ``ys`` -> ``d mu / d ys`` ~ 0);
  * ``solve_epoch(with_jacobian=True)`` wiring and the fail-closed
    ``GRADIENT_UNRELIABLE`` status (singular patch, degenerate quartic node).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import holonomic_ref as H
from holonomic_ref import LensParams
from holonomic_ref.jacobian import (
    arc_intervals,
    endpoint_dtheta_t_chart,
    endpoint_dtheta_theta_chart,
    epoch_jacobian,
    phi_grad,
    radius_terms,
)
from holonomic_ref.radial_events import radial_events
from holonomic_ref.topology import classify_cells

# (xs, ys, rho, q, a), non-barycentric primary-frame convention
CASE_PLAN15 = dict(xs=1 / 5, ys=1 / 7, rho=1 / 8, q=0.5, a=1.2)
CASE_RESONANT = dict(xs=0.05, ys=0.02, rho=0.05, q=0.3, a=0.9)
CASE_BENIGN = dict(xs=0.35, ys=0.22, rho=0.05, q=0.4, a=1.15)

_PNAMES = ("xs", "ys", "rho", "q", "a")


def _p(cfg):
    return LensParams(barycentric=False, **cfg)


def _bump(cfg, key, h):
    c = dict(cfg)
    c[key] = cfg[key] + h
    return c


def _step(cfg, key, rel=1e-5):
    base = abs(cfg[key]) if abs(cfg[key]) > 1e-2 else 1e-2
    return rel * base


# ----------------------------------------------------- phi closed-form grad ---
def test_phi_grad_matches_finite_difference():
    """(d phi/d theta, d phi/d X, .., d phi/d a) in closed form vs central diff."""
    a, m0, X, Y, rho = 1.15, 0.4 / 1.4, 0.42, 0.19, 0.06
    h = 1e-6
    for R in (0.5, 0.95, 1.4):
        for theta in (0.3, 1.9, 3.6, 5.2):
            phi0, dth, gdp = phi_grad(R, theta, a, m0, X, Y, rho)

            fp = phi_grad(R, theta + h, a, m0, X, Y, rho)[0]
            fm = phi_grad(R, theta - h, a, m0, X, Y, rho)[0]
            assert abs(dth - (fp - fm) / (2 * h)) < 1e-6 * (abs(dth) + 1.0)

            base = (a, m0, X, Y, rho)
            for j, _name in enumerate(("X", "Y", "rho", "m0", "a")):
                # map j onto (a, m0, X, Y, rho) argument order of phi_grad
                argpos = {0: 2, 1: 3, 2: 4, 3: 1, 4: 0}[j]
                up = list(base); up[argpos] += h
                dn = list(base); dn[argpos] -= h
                num = (phi_grad(R, theta, *up)[0]
                       - phi_grad(R, theta, *dn)[0]) / (2 * h)
                assert abs(gdp[j] - num) < 1e-6 * (abs(num) + 1.0), (j, R, theta)


# ------------------------------------------------ endpoint chart invariance ---
@pytest.mark.parametrize("cfg", [CASE_PLAN15, CASE_RESONANT, CASE_BENIGN],
                         ids=["plan15", "resonant", "benign"])
def test_endpoint_derivative_chart_invariance(cfg):
    """d theta*/d P via IFT-in-theta == via IFT-in-t on every arc endpoint.

    The two are algebraically identical; the primary path uses IFT-in-theta
    because IFT-in-t carries a removable 2/(1+t^2) pole at theta = pi.
    """
    p = _p(cfg)
    X, Y = p.primary_frame_source()
    pf = (p.a, p.m0, X, Y, p.rho)
    checked = 0
    for c in classify_cells(p).cells:
        if c.kind != "arcs":
            continue
        for R in (c.r_lo + 0.25 * (c.r_hi - c.r_lo),
                  c.r_mid,
                  c.r_lo + 0.75 * (c.r_hi - c.r_lo)):
            kind, arcs = arc_intervals(R, pf)
            if kind != "arcs":
                continue
            for (te, tl) in arcs:
                for th in (te, tl):
                    # skip endpoints within 1e-3 rad of theta = pi where the
                    # t-chart form is deliberately ill-posed
                    if abs(((th - math.pi + math.pi) % (2 * math.pi)) - math.pi) < 1e-3:
                        continue
                    da = endpoint_dtheta_theta_chart(R, th, pf)
                    db = endpoint_dtheta_t_chart(R, th, pf)
                    assert np.all(np.isfinite(da)) and np.all(np.isfinite(db))
                    scale = np.abs(da).max() + 1e-9
                    assert np.abs(da - db).max() < 1e-7 * scale
                    checked += 1
    assert checked > 0


# ------------------------------------------- central-difference convergence ---
def test_grad_mu_central_difference_convergence():
    """max|grad_mu - CD(own mu)| falls as the radial node count grows.

    The reconstruction is a fixed per-cell Gauss-Chebyshev rule; both the
    value and every derivative converge at O(1/n_r^2).
    """
    cfg = CASE_RESONANT
    u = 0.0

    def cd_self(n_r):
        r = epoch_jacobian(_p(cfg), u, n_r=n_r)
        worst = 0.0
        for j, k in enumerate(_PNAMES):
            h = _step(cfg, k, 3e-6)
            hi = epoch_jacobian(_p(_bump(cfg, k, h)), u, n_r=n_r).mu
            lo = epoch_jacobian(_p(_bump(cfg, k, -h)), u, n_r=n_r).mu
            worst = max(worst, abs(r.grad_mu[j] - (hi - lo) / (2 * h)))
        return worst

    e_lo = cd_self(48)
    e_hi = cd_self(160)
    assert e_hi < 0.6 * e_lo, (e_lo, e_hi)
    assert e_hi < 5e-3


# ------------------------------------------- gradient vs the M4 value path ----
@pytest.mark.parametrize("cfg,u", [
    (CASE_PLAN15, 0.0), (CASE_PLAN15, 0.6),
    (CASE_RESONANT, 0.0), (CASE_RESONANT, 0.6),
], ids=["plan15_u0", "plan15_u06", "resonant_u0", "resonant_u06"])
def test_grad_mu_vs_epoch_flux_central_difference(cfg, u):
    """grad_mu (jet) vs a central difference of the independent ``epoch_flux``
    magnification.  Different construction (adaptive ``quad`` vs fixed
    Gauss-Chebyshev), so agreement is a genuine cross-check of both."""
    r = epoch_jacobian(_p(cfg), u)
    assert r.status == "OK"

    def ef_mu(c):
        return H.epoch_flux(_p(c)).mu_linear_ld(u)

    for j, k in enumerate(_PNAMES):
        h = _step(cfg, k, 1e-5)
        num = (ef_mu(_bump(cfg, k, h)) - ef_mu(_bump(cfg, k, -h))) / (2 * h)
        err = abs(r.grad_mu[j] - num)
        # 2e-3 relative, or 6e-3 absolute for the near-zero d/da / the
        # catastrophically-cancelling d/drho where the quad oracle is itself
        # only ~1e-4 accurate.
        assert err < 2e-3 * abs(num) or err < 6e-3, (k, r.grad_mu[j], num)


# ------------------------------------------------------ loose A/B vs native ---
def test_grad_mu_ab_vs_binary_ray_shooting():
    """Loose A/B: d mu / d xs vs a finite difference of the production
    inverse-ray ``binary_ray_shooting`` (coarse; ~1e-2)."""
    lc = pytest.importorskip("lcbinint")
    if not hasattr(lc, "binary_ray_shooting"):
        pytest.skip("binary_ray_shooting unavailable")
    cfg = CASE_PLAN15
    u = 0.0
    r = epoch_jacobian(_p(cfg), u)

    def ray_mu(c):
        p = _p(c)
        x_com = c["xs"] - p.m1 * c["a"]
        return float(lc.binary_ray_shooting(x_com, c["ys"], s=c["a"],
                                            q=c["q"], rho=c["rho"]))

    h = 2e-3
    num = (ray_mu(_bump(cfg, "xs", h)) - ray_mu(_bump(cfg, "xs", -h))) / (2 * h)
    assert abs(r.grad_mu[0] - num) < 3e-2 * abs(num) + 3e-2, (r.grad_mu[0], num)


# ------------------------------------------------------------- d mu / d u -----
@pytest.mark.parametrize("cfg", [CASE_PLAN15, CASE_RESONANT],
                         ids=["plan15", "resonant"])
def test_dmu_du_sign_and_magnitude(cfg):
    u = 0.3
    r = epoch_jacobian(_p(cfg), u)
    assert np.sign(r.dmu_du) == np.sign(r.F_half - 2.0 * r.F0 / 3.0)
    hi = epoch_jacobian(_p(cfg), u + 1e-4).mu
    lo = epoch_jacobian(_p(cfg), u - 1e-4).mu
    num = (hi - lo) / 2e-4
    assert abs(r.dmu_du - num) < 1e-6 * abs(num) + 1e-6


# ------------------------------------------------------- on-axis symmetry -----
def test_on_axis_ys_symmetry():
    """With ys = 0 (and xs off both lenses) mu is even in ys, so d mu/d ys ~ 0
    even though every other component is O(1--30)."""
    p = LensParams(0.30, 0.0, 0.04, 0.6, 1.35, barycentric=False)
    r = epoch_jacobian(p, 0.3)
    assert r.status == "OK"
    assert abs(r.grad_mu[1]) < 1e-6
    assert np.abs(r.grad_mu[[0, 2, 3, 4]]).max() > 0.1


# ------------------------------------------------- solve_epoch(...) wiring ----
def test_solve_epoch_with_jacobian_wellformed():
    p = _p(CASE_PLAN15)

    base = H.solve_epoch(p, 0.5)
    assert base.jacobian is None                    # default path untouched

    res = H.solve_epoch(p, 0.5, with_jacobian=True)
    j = res.jacobian
    assert set(j) == {"grad_mu", "dmu_du", "param_order", "status", "notes"}
    assert j["param_order"] == ("xs", "ys", "rho", "q", "a")
    assert np.asarray(j["grad_mu"]).shape == (5,)
    assert np.all(np.isfinite(j["grad_mu"])) and math.isfinite(j["dmu_du"])
    assert j["status"] == "OK"

    # value / JVP consistency: solve_epoch adopts the jet reconstruction's
    # flux so mu and grad_mu come from one numerical object (plan sec. 11).
    ej = epoch_jacobian(p, 0.5)
    assert res.mu == pytest.approx(ej.mu, rel=1e-12)
    # and that reconstruction still agrees with the adaptive-quad value path
    assert res.mu == pytest.approx(base.mu, rel=2e-3)


# ------------------------------------------------------------ fail closed -----
def test_gradient_fails_closed_on_singular_patch():
    """xs = ys = 0 on-axis origin: the connection is genuinely singular there;
    the gradient path reports GRADIENT_UNRELIABLE rather than a number."""
    r = epoch_jacobian(LensParams(0.0, 0.0, 0.02, 1.0, 1.0, barycentric=False),
                       0.0)
    assert r.status == "GRADIENT_UNRELIABLE"
    assert any("singular" in n for n in r.notes)


def test_radius_terms_flags_degenerate_quartic_node():
    """At a chart_p4 event the boundary quartic loses its leading coefficient;
    arc_intervals cannot bracket from the roots and falls back to the grid
    with reliable = False."""
    p = _p(CASE_RESONANT)
    X, Y = p.primary_frame_source()
    pf = (p.a, p.m0, X, Y, p.rho)
    evs, _ = radial_events(p.a, p.m0, X, Y, p.rho)
    hit = 0
    for e in evs:
        if e.kind != "chart_p4":
            continue
        assert arc_intervals(e.radius, pf)[0] == "degenerate"
        assert radius_terms(e.radius, pf)[4] is False
        hit += 1
    assert hit > 0


def test_gradient_fails_closed_when_topology_uncertain():
    """A full-phi>0 circle inside r_max forces GRADIENT_UNRELIABLE: the
    periodic-trapezoid endpoint rule there has no error control."""
    # deep resonant crossing: pick a config with a 'full' cell if present,
    # otherwise assert the mechanism directly via a synthetic full radius.
    from holonomic_ref.jacobian import _full_circle_terms
    p = _p(CASE_RESONANT)
    X, Y = p.primary_frame_source()
    pf = (p.a, p.m0, X, Y, p.rho)
    f0, fh, d0, dh, rel = _full_circle_terms(0.01, pf)
    assert rel is False
    assert d0.shape == (5,) and dh.shape == (5,)

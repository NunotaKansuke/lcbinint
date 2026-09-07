"""M5: the full-epoch solver -- invariants, the singular patch, and a small
audit slice that must contain zero silent misses.

Covers the M5 gate ("監査領域で精度と失敗率を別々に報告、silent miss なし"):

  * :func:`holonomic_ref.solve_epoch` returns a well-formed
    :class:`EpochResult` with a status from the plan sec. 12.2 taxonomy on
    every M4 config, in both ``fast`` and ``validated`` mode;
  * ``EpochResult.mu`` at ``u = 0`` equals ``mu_uniform`` and equals the
    ``epoch_flux`` uniform magnification; the linear-LD blend is monotone;
  * the ``xs = ys = 0`` on-axis-origin locus (and its ``|zeta| < tol``
    neighbourhood) routes through :func:`on_axis_origin_flux`, returns a
    non-``OK`` status (``LOCAL_REFERENCE_USED``), and lands within 1e-2 of an
    independent source-plane quadrature -- it is never silently wrong;
  * ``representation_events`` reports ``R = a`` / ``R = sqrt(m0)`` when they
    fall inside the scan and nothing otherwise;
  * a curated audit slice: every config whose status is clean (``OK`` /
    ``OK_VALIDATED``) agrees with the production inverse-ray solver to better
    than 5e-3 -- i.e. no silent miss.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import holonomic_ref as H
from holonomic_ref import LensParams
from holonomic_ref.direct_quadrature import image_plane_flux_grid
from holonomic_ref.singular import near_origin_source, representation_events

try:
    import lcbinint
except Exception:                                               # pragma: no cover
    lcbinint = None

_STATUS_TAXONOMY = {
    "OK", "OK_VALIDATED", "OK_ESTIMATED", "LOCAL_REFERENCE_USED",
    "TOPOLOGY_UNCERTAIN", "BASIS_DEGENERATE", "CONNECTION_ILL_CONDITIONED",
    "TRANSPORT_TOLERANCE_FAILED", "GRADIENT_UNRELIABLE",
    "ARC_TOPOLOGY_INVALID", "EVENT_UNRESOLVED", "RESOURCE_LIMIT",
}
_CLEAN = {"OK", "OK_VALIDATED"}

CASES = [
    dict(xs=1 / 5, ys=1 / 7, rho=1 / 8, q=0.5, a=1.2),
    dict(xs=0.05, ys=0.02, rho=0.05, q=0.3, a=0.9),
    dict(xs=0.4, ys=-0.05, rho=0.09, q=0.8, a=0.55),
]
_IDS = [f"a{c['a']}_q{c['q']}" for c in CASES]


def _p(cfg):
    return LensParams(barycentric=False, **cfg)


def _ray_mu(cfg, u=0.0):
    if lcbinint is None:
        return None
    q, a = cfg["q"], cfg["a"]
    m1a = (q / (1.0 + q)) * a
    ld = lcbinint.LimbDarkening.linear(u) if u else None
    return float(lcbinint.binary_ray_shooting(
        cfg["xs"] - m1a, cfg["ys"], s=a, q=q, rho=cfg["rho"], limb_darkening=ld))


# --------------------------------------------------------------- invariants ---
@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
@pytest.mark.parametrize("mode", ["fast", "validated"])
def test_epoch_result_well_formed(cfg, mode):
    er = H.solve_epoch(_p(cfg), 0.0, mode=mode)
    assert er.status in _STATUS_TAXONOMY
    assert math.isfinite(er.mu) and er.mu > 1.0
    assert math.isfinite(er.F0) and er.F0 > 0.0
    assert math.isfinite(er.F_half) and er.F_half > 0.0
    assert er.mode == mode
    # u = 0 : mu, mu_uniform and the LD blend at u=0 all coincide
    assert er.mu == pytest.approx(er.mu_uniform, rel=1e-12)
    assert er.mu_linear_ld(0.0) == pytest.approx(er.mu_uniform, rel=1e-12)


@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_limb_darkening_blend_monotone(cfg):
    """``mu_linear_ld(u)`` is strictly monotone in ``u`` and stays > 1.

    d/du of ``((1-u)F0 + u F_1/2) / (pi rho^2 (1 - u/3))`` is
    ``pi rho^2 (F_1/2 - 2 F0/3)`` -- *constant* in ``u`` -- so the blend is
    always monotone.  Its **sign is not fixed**: near (but outside) a caustic
    the source centre is more magnified than its limb, so weighting the
    centre up (larger ``u``) raises ``mu`` -- this is the plan15 case.  The
    physical invariant is monotonicity + positivity, not a direction."""
    er = H.solve_epoch(_p(cfg), 0.0, mode="fast")
    vals = [er.mu_linear_ld(u) for u in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    assert all(v > 1.0 for v in vals), vals
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    assert all(d > 0 for d in diffs) or all(d < 0 for d in diffs), vals


@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
def test_magnification_helper_matches_solve_epoch(cfg):
    assert H.magnification(_p(cfg), 0.0, mode="fast") == pytest.approx(
        H.solve_epoch(_p(cfg), 0.0, mode="fast").mu, rel=1e-12)


# ----------------------------------------------------- singular patch --------
@pytest.mark.parametrize("zeta", [0.0, 5e-4])
def test_on_axis_origin_routes_to_patch_and_is_not_silently_wrong(zeta):
    cfg = dict(xs=zeta, ys=0.0, rho=0.1, q=0.4, a=1.1)
    p = _p(cfg)
    assert near_origin_source(p)

    er = H.solve_epoch(p, 0.0, mode="fast")
    # fail-closed: the holonomic connection is undefined here, the status
    # must say a local reference was used -- never a bare OK
    assert er.status == "LOCAL_REFERENCE_USED"
    assert er.status not in _CLEAN

    # ... but the value is still correct: within 1e-2 of an independent
    # image-plane dense-grid quadrature (source-plane Witt & Mao is
    # unreliable here -- a lens sits inside the source disk)
    f0, fh = image_plane_flux_grid(p, n_r=1200, n_theta=18000)
    mu_ref = f0 / (math.pi * cfg["rho"] ** 2)
    assert abs(er.mu - mu_ref) / mu_ref < 1e-2, (er.mu, mu_ref)


def test_off_axis_source_does_not_route_to_patch():
    assert not near_origin_source(_p(dict(xs=0.2, ys=0.0, rho=0.05, q=0.4, a=1.1)))
    er = H.solve_epoch(_p(dict(xs=0.2, ys=0.0, rho=0.05, q=0.4, a=1.1)), 0.0,
                       mode="fast")
    assert er.status != "LOCAL_REFERENCE_USED"


# ------------------------------------------------ representation events ------
def test_representation_events_reported_inside_scan():
    # a = 1.1 and sqrt(m0) = sqrt(1/1.4) ~ 0.845 both sit inside R_max here
    p = _p(dict(xs=0.3, ys=0.05, rho=0.04, q=0.4, a=1.1))
    evs = representation_events(p)
    kinds = {k for _R, k in evs}
    assert kinds == {"R=a", "R=sqrt(m0)"}
    got = dict((k, R) for R, k in evs)
    assert got["R=a"] == pytest.approx(1.1)
    assert got["R=sqrt(m0)"] == pytest.approx(math.sqrt(1.0 / 1.4))


def test_representation_events_is_cheap():
    """Must not call the sympy radial-event solve (it is on the solve_epoch
    hot path) -- a closed-form R_max bound only."""
    import time as _t
    p = _p(dict(xs=0.21, ys=0.13, rho=0.05, q=0.35, a=1.3))
    t0 = _t.perf_counter()
    for _ in range(50):
        representation_events(p)
    assert _t.perf_counter() - t0 < 0.5


# --------------------------------------------------- audit: no silent miss --
@pytest.mark.skipif(lcbinint is None, reason="production reference unavailable")
@pytest.mark.parametrize("cfg", CASES, ids=_IDS)
@pytest.mark.parametrize("u", [0.0, 0.5])
def test_audit_slice_no_silent_miss(cfg, u):
    """Clean status  =>  agrees with the production solver to < 5e-3."""
    er = H.solve_epoch(_p(cfg), u, mode="validated")
    ref = _ray_mu(cfg, u)
    rel = abs(er.mu - ref) / abs(ref)
    if er.status in _CLEAN:
        assert rel < 5e-3, (
            f"SILENT MISS: status {er.status} but rel {rel:.2e} "
            f"(mu={er.mu:.6f} ref={ref:.6f})")
    else:
        # a flagged status is allowed to be inaccurate -- that is the point
        # of flagging it -- but record it so a regression to silently-wrong
        # is visible in -rs output
        print(f"{cfg} u={u}: flagged {er.status}, rel={rel:.2e}")


@pytest.mark.skipif(lcbinint is None, reason="production reference unavailable")
def test_validated_mode_promotes_a_well_resolved_config():
    """At least one benign config must reach OK_VALIDATED -- otherwise the
    cross-check gate is not actually exercising anything."""
    statuses = [H.solve_epoch(_p(cfg), 0.0, mode="validated").status
                for cfg in CASES]
    assert any(s == "OK_VALIDATED" for s in statuses), statuses

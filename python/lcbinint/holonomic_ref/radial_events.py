"""M1: radial event enumeration (plan sec. 3).

The circle |z| = R meets the source-image boundary phi(z) = 0 in a number
of arcs that is piecewise-constant in R.  It can only change at a
*radial event*.  Two kinds, kept distinct:

  * PHYSICAL        -- Disc_t P(t; R) = 0.  Verified identity
                       Disc_t P = R^4 * D14(R^2) with deg_v D14 = 14
                       (checks/holonomic/symbolic_checks.py).  A positive
                       real root v of D14 with a *real* double root t* is an
                       image-band birth/death ("physical_real"); one whose
                       double root t* is complex touches nothing on the real
                       circle ("physical_complex", representation-only but
                       still a safe cell boundary).
  * REPRESENTATION  -- R = a, R = sqrt(m0), the chart event p4(R) = 0
                       (a boundary point crossing theta = pi), and, on the
                       binary axis (ys = 0), the roots of L(v) from
                       Res(P, B).  No band is created or destroyed but the
                       period basis / connection representation degenerates.

Correctness first: D14 and every auxiliary polynomial are built in exact
rational arithmetic (sympy) from the exact binary values of the inputs, and
their roots are found at 30-digit precision.  The monomial basis is very
ill-conditioned (plan sec. 15, kappa ~ 1e9); float64 companion-matrix
root-finding silently turns real double roots into spurious complex pairs
and misses band births -- see the regression notes in
tests/holonomic/test_radial_event_completeness.py.  A fast float path lives
in :func:`d14_coeffs_fast` for later benchmarking only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp

_R, _V = sp.symbols("R v", real=True)
_DPS = 30
_IM_TOL = 1e-12          # |Im(root)| below this  -> treat as real
_REAL_MERGE = 1e-9       # dedupe identical roots (e.g. exact double roots)


# --------------------------------------------------------------------------
# exact polynomial family in R  (coefficients of P in t, each a poly in R)
# --------------------------------------------------------------------------
def _P_coeffs_in_R_exact(a, m0, xs, ys, rho):
    a, m0, xs, ys, rho = (sp.Rational(x) for x in (a, m0, xs, ys, rho))
    zeta = xs + sp.I * ys
    n0 = -zeta * _R**2
    n1 = _R * (_R**2 - 1 + a * zeta)
    n2 = a * (m0 - _R**2)
    cT0 = n0 + n1 + n2
    cT1 = 2 * sp.I * (n2 - n0)
    cT2 = n1 - n0 - n2
    bm, bp = (_R - a)**2, (_R + a)**2
    k = rho**2 * _R**2
    p0 = sp.expand(k * bm - cT0 * sp.conjugate(cT0))
    p1 = sp.expand(-(cT0 * sp.conjugate(cT1) + cT1 * sp.conjugate(cT0)))
    p2 = sp.expand(k * (bm + bp)
                   - (cT1 * sp.conjugate(cT1)
                      + cT0 * sp.conjugate(cT2) + cT2 * sp.conjugate(cT0)))
    p3 = sp.expand(-(cT1 * sp.conjugate(cT2) + cT2 * sp.conjugate(cT1)))
    p4 = sp.expand(k * bp - cT2 * sp.conjugate(cT2))
    return [sp.expand(sp.re(p)) for p in (p0, p1, p2, p3, p4)]


@lru_cache(maxsize=256)
def _d14_poly(a, m0, xs, ys, rho):
    """sympy.Poly D14(v), v = R^2, from the quartic invariants of P."""
    p0, p1, p2, p3, p4 = _P_coeffs_in_R_exact(a, m0, xs, ys, rho)
    Iq = sp.expand(12 * p4 * p0 - 3 * p3 * p1 + p2**2)
    Jq = sp.expand(72 * p4 * p2 * p0 + 9 * p3 * p2 * p1
                   - 27 * p4 * p1**2 - 27 * p3**2 * p0 - 2 * p2**3)
    disc = sp.expand((4 * Iq**3 - Jq**2) / 27)
    coeffs_asc = sp.Poly(disc, _R).all_coeffs()[::-1]
    if any(c != 0 for c in coeffs_asc[:4]):
        raise ValueError("Disc_t P is not divisible by R^4 as expected")
    coeffs_asc = coeffs_asc[4:]
    if any(coeffs_asc[i] != 0 for i in range(1, len(coeffs_asc), 2)):
        raise ValueError("Disc_t P / R^4 is not even in R")
    return sp.Poly(list(reversed(coeffs_asc[0::2])), _V)


@lru_cache(maxsize=256)
def _p4_poly(a, m0, xs, ys, rho):
    return sp.Poly(_P_coeffs_in_R_exact(a, m0, xs, ys, rho)[4], _R)


def d14_coeffs(a, m0, xs, ys, rho):
    """Ascending float coefficients of D14(v) and a normalising scale."""
    c = np.asarray(
        [float(x) for x in reversed(_d14_poly(a, m0, xs, ys, rho).all_coeffs())],
        dtype=float)
    return c, float(np.abs(c).max() + 1e-300)


def d14_coeffs_fast(a, m0, xs, ys, rho):
    """float64-only D14 via numpy polynomial arithmetic.  APPROXIMATE --
    for benchmarking the exact path, never for enumeration."""
    pm = np.polynomial.polynomial
    zeta = complex(xs, ys)
    R1 = np.array([0.0, 1.0], complex)
    R2 = np.array([0.0, 0.0, 1.0], complex)
    n0 = -zeta * R2
    n1 = pm.polyadd([0, 0, 0, 1.0], (a * zeta - 1.0) * R1)
    n2 = pm.polyadd(-a * R2, [a * m0])
    cT0 = pm.polyadd(pm.polyadd(n0, n1), n2)
    cT1 = 2j * pm.polyadd(n2, -n0)
    cT2 = pm.polyadd(pm.polyadd(n1, -n0), -n2)
    bm = np.array([a * a, -2 * a, 1.0])
    bp = np.array([a * a, 2 * a, 1.0])
    kR2 = (rho * rho) * R2
    lin0 = pm.polymul(kR2, bm)
    lin2 = pm.polymul(kR2, pm.polyadd(bm, bp))
    lin4 = pm.polymul(kR2, bp)
    cj = np.conjugate
    a0 = pm.polymul(cT0, cj(cT0))
    a1 = pm.polyadd(pm.polymul(cT0, cj(cT1)), pm.polymul(cT1, cj(cT0)))
    a2 = pm.polyadd(pm.polyadd(pm.polymul(cT1, cj(cT1)),
                               pm.polymul(cT0, cj(cT2))),
                    pm.polymul(cT2, cj(cT0)))
    a3 = pm.polyadd(pm.polymul(cT1, cj(cT2)), pm.polymul(cT2, cj(cT1)))
    a4 = pm.polymul(cT2, cj(cT2))
    p0 = pm.polyadd(lin0, -a0)
    p1 = -a1
    p2 = pm.polyadd(lin2, -a2)
    p3 = -a3
    p4 = pm.polyadd(lin4, -a4)
    Iq = pm.polyadd(pm.polyadd(12 * pm.polymul(p4, p0),
                               -3 * pm.polymul(p3, p1)),
                    pm.polymul(p2, p2))
    Jq = pm.polyadd(pm.polyadd(pm.polyadd(
        72 * pm.polymul(pm.polymul(p4, p2), p0),
        9 * pm.polymul(pm.polymul(p3, p2), p1)),
        pm.polyadd(-27 * pm.polymul(pm.polymul(p4, p1), p1),
                   -27 * pm.polymul(pm.polymul(p3, p3), p0))),
        -2 * pm.polymul(pm.polymul(p2, p2), p2))
    disc = np.real(pm.polyadd(4 * pm.polymul(pm.polymul(Iq, Iq), Iq),
                              -pm.polymul(Jq, Jq)) / 27.0)
    return disc[4:][0::2]


# --------------------------------------------------------------------------
@dataclass
class RadialEvent:
    radius: float
    kind: str
    physically_real: bool
    detail: str = ""


def _pos_roots(poly, im_tol=_IM_TOL):
    """(real_positive, complex_positive_realpart) roots of a sympy Poly.

    Real roots come from exact real-root isolation (robust for the
    ill-conditioned degree-14 D14); complex roots are a best-effort numeric
    pass and are only ever used as soft cell boundaries.
    """
    real: list[float] = []
    try:
        for r in poly.real_roots():
            rv = float(r.evalf(_DPS))
            if rv > 0.0:
                real.append(rv)
    except (sp.PolynomialError, NotImplementedError, ValueError):
        for r in poly.nroots(n=_DPS, maxsteps=800):
            re, im = float(sp.re(r)), float(sp.im(r))
            if re > 0.0 and abs(im) < im_tol:
                real.append(re)

    cplx: list[float] = []
    try:
        for r in poly.nroots(n=_DPS, maxsteps=800):
            re, im = float(sp.re(r)), float(sp.im(r))
            if re > 0.0 and abs(im) >= im_tol:
                cplx.append(re)
    except Exception:
        pass
    return sorted(real), sorted(cplx)


def _dedupe(vals, tol):
    out = []
    for x in sorted(vals):
        if not out or abs(x - out[-1]) > tol:
            out.append(x)
    return out


def _double_root_is_real(R, a, m0, xs, ys, rho, tol=1e-6):
    """At a discriminant zero P has a multiple root t*; is it real?"""
    from .polynomial_family import boundary_quartic
    P = np.asarray(boundary_quartic(R, a, m0, xs, ys, rho)[::-1], dtype=float)
    scale = float(np.abs(P).sum()) + 1e-300
    best = None
    for z in np.roots(np.polyder(P)):                 # cubic -- accurate
        resid = abs(np.polyval(P, z)) / scale
        if best is None or resid < best[1]:
            best = (z, resid)
    z, resid = best
    return resid < tol and abs(z.imag) < 1e-4 * (abs(z.real) + 1.0)


# --------------------------------------------------------------------------
def radial_events(a, m0, xs, ys, rho, *, merge_tol=1e-7):
    """Return ``(sorted [RadialEvent], R_max)`` for the interval (0, R_max)."""
    W = math.hypot(xs, ys) + rho
    Rmax = 0.5 * (a + W + math.hypot(a - W, 2.0)) + 1e-12

    ev: list[RadialEvent] = []

    real_v, cplx_v = _pos_roots(_d14_poly(a, m0, xs, ys, rho))
    for v in _dedupe(real_v, _REAL_MERGE):
        R = math.sqrt(v)
        if not (0.0 < R < Rmax):
            continue
        is_real = _double_root_is_real(R, a, m0, xs, ys, rho)
        ev.append(RadialEvent(
            R, "physical_real" if is_real else "physical_complex",
            is_real, "D14 real root"))
    for v in _dedupe(cplx_v, _REAL_MERGE):
        R = math.sqrt(v)
        if 0.0 < R < Rmax:
            ev.append(RadialEvent(R, "physical_complex", False,
                                  "D14 complex root (Re v)"))

    if 0.0 < a < Rmax:
        ev.append(RadialEvent(a, "R_eq_a", False, "P<->B factor clash"))
    rm0 = math.sqrt(m0)
    if 0.0 < rm0 < Rmax:
        ev.append(RadialEvent(rm0, "R_eq_sqrt_m0", False,
                              "P acquires factor A ; Q double factor"))

    if abs(ys) < 1e-14:
        aR, m0R, xsR = sp.Rational(a), sp.Rational(m0), sp.Rational(xs)
        Lpoly = sp.Poly((aR - xsR) * _V * (_V - aR**2)
                        - aR * _V + aR**3 * m0R, _V)
        Lreal, _ = _pos_roots(Lpoly)
        for v in _dedupe(Lreal, _REAL_MERGE):
            R = math.sqrt(v)
            if 0.0 < R < Rmax:
                ev.append(RadialEvent(R, "L_root", False,
                                      "Res(P, B) axis root"))

    p4real, _ = _pos_roots(_p4_poly(a, m0, xs, ys, rho))
    for R in _dedupe(p4real, _REAL_MERGE):
        if 0.0 < R < Rmax:
            ev.append(RadialEvent(R, "chart_p4", False,
                                  "p4(R)=0 : boundary point at theta = pi"))

    ev.sort(key=lambda e: e.radius)
    merged: list[RadialEvent] = []
    for e in ev:
        if merged and e.kind == merged[-1].kind \
           and abs(e.radius - merged[-1].radius) < merge_tol:
            continue
        merged.append(e)
    return merged, Rmax

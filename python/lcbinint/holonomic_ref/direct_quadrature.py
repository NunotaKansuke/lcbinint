"""M4: fully independent reference for the epoch flux (plan sec. 16).

Nothing here touches the holonomic machinery (no ``P(t;R)``, no periods, no
connection).  Two independent 2-D quadratures of the *source* disk:

    F0     = integral_disk  A_pt(w)                       dA(w)
    F_1/2  = integral_disk  A_pt(w) sqrt(1 - |w-w_c|^2/rho^2)  dA(w)
    mu_u   = ( (1-u) F0 + u F_1/2 ) / ( pi rho^2 (1 - u/3) )

where ``A_pt(w)`` is the binary point-source magnification obtained from the
degree-5 complex lens polynomial (Witt & Mao), summed over the physical
images.  ``sqrt(1 - p^2) = sqrt(phi)`` ties ``F_1/2`` to the image-plane
half-moment used by :mod:`flux`.
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq

from .polynomial_family import LensParams

_TWO_PI = 2.0 * math.pi


def _phi(R, theta, a, m0, xs, ys, rho):
    z = R * complex(math.cos(theta), math.sin(theta))
    zb = z.conjugate()
    fz = z - m0 / zb - (1.0 - m0) / (zb - a)
    return 1.0 - abs(fz - complex(xs, ys)) ** 2 / (rho * rho)


def _circle_arcs(R, a, m0, xs, ys, rho, *, n_scan=2048):
    """``phi >= 0`` sub-intervals of ``[0, 2pi)`` on ``|z| = R``.

    Independent brentq bracketing straight from ``phi`` -- no import from
    :mod:`topology`.  Returns a list of ``(theta_enter, theta_leave)`` with
    ``theta_leave`` possibly wrapped past ``2 pi``; ``"full"`` / ``"empty"``
    for the degenerate cases.
    """
    th = np.linspace(0.0, _TWO_PI, n_scan, endpoint=False)
    val = np.array([_phi(R, t, a, m0, xs, ys, rho) for t in th])
    pos = val >= 0.0
    if pos.all():
        return "full"
    if not pos.any():
        return "empty"
    step = _TWO_PI / n_scan
    nxt = np.roll(pos, -1)
    idx = np.nonzero(pos != nxt)[0]
    f = lambda t: _phi(R, t, a, m0, xs, ys, rho)
    roots, rising = [], []
    for i in idx:
        r = brentq(f, th[i], th[i] + step, xtol=1e-14, rtol=8.9e-16)
        roots.append(r % _TWO_PI)
        rising.append(not pos[i])
    order = np.argsort(roots)
    roots = [roots[k] for k in order]
    rising = [rising[k] for k in order]
    n = len(roots)
    if n % 2:
        return "odd"
    start = 0 if rising[0] else -1
    out = []
    for k in range(0, n, 2):
        te = roots[(start + k) % n]
        tl = roots[(start + k + 1) % n]
        if tl <= te:
            tl += _TWO_PI
        out.append((te, tl))
    return out


# ---- degree-5 lens polynomial, built once symbolically then lambdified ----
def _build_coeff_fn():
    z, zc, a, m0, m1 = sp.symbols("z zc a m0 m1")
    # z̄ from the conjugated lens equation, substituted into the lens equation
    zbar = zc + m0 / z + m1 / (z - a)
    lhs = z - m0 / zbar - m1 / (zbar - a) - sp.Symbol("zeta")
    num = sp.together(lhs).as_numer_denom()[0]
    poly = sp.Poly(sp.expand(num), z)
    coeffs = poly.all_coeffs()               # descending, length 6
    zeta = sp.Symbol("zeta")
    fn = sp.lambdify((zeta, zc, a, m0, m1), coeffs, "numpy")
    return fn


_COEFFS = _build_coeff_fn()


def point_source_magnification(w, a, m0, *, tol=1e-6):
    """Binary point-source magnification at source position ``w`` (complex).

    Lenses at ``0`` (mass ``m0``) and ``a`` (mass ``1 - m0``).
    """
    m1 = 1.0 - m0
    zeta = complex(w)
    coeffs = _COEFFS(zeta, np.conj(zeta), a, m0, m1)
    roots = np.roots([complex(c) for c in coeffs])
    mu = 0.0
    for zr in roots:
        zb = np.conj(zr)
        f = zr - m0 / zb - m1 / (zb - a)
        if abs(f - zeta) > tol * max(1.0, abs(zeta)):
            continue                                  # spurious root
        dfz_bar = m0 / zb**2 + m1 / (zb - a) ** 2      # d f / d zbar
        detJ = 1.0 - abs(dfz_bar) ** 2
        if detJ == 0.0:
            continue
        mu += 1.0 / abs(detJ)
    return mu


def source_plane_flux(params: LensParams, *, n_r=64, n_theta=256):
    """``(F0, F_1/2)`` by Gauss-Legendre x midpoint quadrature of the disk.

    Radial rule on ``[0, rho]`` with the plain measure ``r dr`` (Gauss-
    Legendre, exact-ish for the smooth ``A_pt`` away from caustics); azimuth
    by the midpoint rule (spectral for the periodic integrand).
    """
    xs, ys = params.primary_frame_source()
    a, m0, rho = params.a, params.m0, params.rho
    wc = complex(xs, ys)

    xg, wg = leggauss(n_r)
    r = 0.5 * rho * (xg + 1.0)
    wr = 0.5 * rho * wg
    th = (np.arange(n_theta) + 0.5) * (2.0 * math.pi / n_theta)
    dth = 2.0 * math.pi / n_theta

    F0 = Fh = 0.0
    for ri, wri in zip(r, wr):
        p = ri / rho
        sq = math.sqrt(max(1.0 - p * p, 0.0))
        row = wc + ri * np.exp(1j * th)
        acc = 0.0
        for w in row:
            acc += point_source_magnification(w, a, m0)
        cell = wri * ri * dth * acc          # r dr dtheta
        F0 += cell
        Fh += cell * sq
    return F0, Fh


def _r_max(xs, ys, rho, a):
    W = abs(complex(xs, ys)) + rho
    return (a + W + math.sqrt((a - W) ** 2 + 4.0)) / 2.0


def _circle_moments(R, a, m0, xs, ys, rho):
    """``(R * m(R), R * s(R))`` -- the image-annulus radial integrands."""
    arcs = _circle_arcs(R, a, m0, xs, ys, rho)
    if arcs in ("empty", "odd"):
        return 0.0, 0.0
    if arcs == "full":
        s_ang = quad(lambda t: math.sqrt(max(_phi(R, t, a, m0, xs, ys, rho),
                                             0.0)),
                     0.0, _TWO_PI, limit=200)[0]
        return R * _TWO_PI, R * s_ang
    m_ang = sum(hi - lo for lo, hi in arcs)
    s_ang = 0.0
    for lo, hi in arcs:
        half = 0.5 * (hi - lo)
        mid = lo + half

        def g(u, mid=mid, half=half):
            v = _phi(R, mid + half * u, a, m0, xs, ys, rho)
            s = 1.0 - u * u
            return math.sqrt(v / s) if (v > 0.0 and s > 0.0) else 0.0

        s_ang += half * quad(g, -1.0, 1.0, weight="alg", wvar=(0.5, 0.5),
                             limit=200)[0]
    return R * m_ang, R * s_ang


def image_plane_flux(params: LensParams, *, r_max=None, event_radii=None):
    """``(F0, F_1/2)`` by nested adaptive quadrature of the image annulus.

    Inner: for each ``R`` the ``phi >= 0`` arcs are found by independent
    brentq bracketing; ``m(R) = sum_arcs width``, ``s(R) = sum_arcs
    int sqrt(phi) dtheta``.  Outer: ``scipy.integrate.quad`` in ``R`` on
    ``[0, r_max]``, subdivided at ``event_radii`` if given (the radial
    integrand has sqrt kinks at arc births/deaths).  Fully independent of
    the holonomic path (no ``P(t;R)``, periods, connection or charts).
    """
    xs, ys = params.primary_frame_source()
    a, m0, rho = params.a, params.m0, params.rho
    if r_max is None:
        r_max = _r_max(xs, ys, rho, a)
    pts = None
    if event_radii is not None:
        pts = sorted(r for r in event_radii if 0.0 < r < r_max) or None

    F0 = quad(lambda R: _circle_moments(R, a, m0, xs, ys, rho)[0],
              0.0, r_max, points=pts, limit=400, epsabs=1e-12, epsrel=1e-11)[0]
    Fh = quad(lambda R: _circle_moments(R, a, m0, xs, ys, rho)[1],
              0.0, r_max, points=pts, limit=400, epsabs=1e-12, epsrel=1e-11)[0]
    return F0, Fh


def image_plane_flux_grid(params: LensParams, *, r_max=None, n_r=800,
                          n_theta=12000):
    """Cheap dense-grid cross-check of :func:`image_plane_flux` (~1e-3)."""
    xs, ys = params.primary_frame_source()
    a, m0, rho = params.a, params.m0, params.rho
    if r_max is None:
        r_max = _r_max(xs, ys, rho, a)
    R = np.linspace(0.5 * r_max / n_r, r_max, n_r)
    th = np.linspace(0.0, _TWO_PI, n_theta, endpoint=False)
    dth = _TWO_PI / n_theta
    dR = R[1] - R[0]
    TH, RR = np.meshgrid(th, R)
    z = RR * np.exp(1j * TH)
    zb = np.conj(z)
    fz = z - m0 / zb - (1.0 - m0) / (zb - a)
    phi = 1.0 - np.abs(fz - complex(xs, ys)) ** 2 / (rho * rho)
    mask = phi >= 0.0
    F0 = np.sum(RR[mask]) * dR * dth
    Fh = np.sum(RR[mask] * np.sqrt(np.clip(phi[mask], 0.0, None))) * dR * dth
    return F0, Fh


def magnification_reference(params: LensParams, u=0.0, **kw):
    """``mu_u`` from :func:`source_plane_flux` (plan sec. 1 formula)."""
    F0, Fh = source_plane_flux(params, **kw)
    rho2 = params.rho ** 2
    return (((1.0 - u) * F0 + u * Fh)
            / (math.pi * rho2 * (1.0 - u / 3.0)))

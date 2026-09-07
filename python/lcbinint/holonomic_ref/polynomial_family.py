"""M1: the single source of the boundary polynomial family P(t;R), T(t), B(t;R).

Plan sec. 2.  All other modules must obtain coefficients from here (no
per-solver re-derivation).  Conventions:

    lenses at 0 and a ; masses m0, m1 = 1 - m0
    zeta = xs + i ys
    z = R e^{i th},  t = tan(th/2),  A(t) = 1 + t^2
    B(t;R) = (R-a)^2 + (R+a)^2 t^2
    n0 = -zeta R^2 ,  n1 = R (R^2 - 1 + a zeta) ,  n2 = a (m0 - R^2)
    T(t) = n0 (1-it)^2 + n1 (1+t^2) + n2 (1+it)^2 = cT0 + cT1 t + cT2 t^2
    P(t;R) = rho^2 R^2 A B - T conj(T)          (real, degree 4 in t)
    phi    = P / (rho^2 R^2 A B)

Verified by checks/holonomic/symbolic_checks.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _T_complex_coeffs(R, a, m0, xs, ys):
    zeta = complex(xs, ys)
    n0 = -zeta * R * R
    n1 = R * (R * R - 1.0 + a * zeta)
    n2 = a * (m0 - R * R)
    cT0 = n0 + n1 + n2
    cT1 = 2j * (n2 - n0)
    cT2 = n1 - n0 - n2
    return cT0, cT1, cT2


def T_coeffs(R, a, m0, xs, ys):
    return list(_T_complex_coeffs(R, a, m0, xs, ys))


def B_coeffs(R, a):
    return [(R - a) ** 2, 0.0, (R + a) ** 2]


def boundary_quartic(R, a, m0, xs, ys, rho):
    """P coefficients [p0, p1, p2, p3, p4] (ascending powers of t)."""
    rho2 = rho * rho
    R2 = R * R
    bm = (R - a) ** 2
    bp = (R + a) ** 2
    cT0, cT1, cT2 = _T_complex_coeffs(R, a, m0, xs, ys)

    k = rho2 * R2
    lin0, lin2, lin4 = k * bm, k * (bm + bp), k * bp

    a0 = abs(cT0) ** 2
    a1 = 2.0 * (cT0 * cT1.conjugate()).real
    a2 = abs(cT1) ** 2 + 2.0 * (cT0 * cT2.conjugate()).real
    a3 = 2.0 * (cT1 * cT2.conjugate()).real
    a4 = abs(cT2) ** 2

    return [lin0 - a0, -a1, lin2 - a2, -a3, lin4 - a4]


def boundary_quartic_dR(R, a, m0, xs, ys, rho):
    """d/dR of :func:`boundary_quartic` -- [dp0, .., dp4] (ascending), exact.

    Closed form; no finite differencing.  P's coefficients are explicit
    polynomials in ``R`` through ``k = rho^2 R^2``, ``bm = (R-a)^2``,
    ``bp = (R+a)^2`` and the complex ``T`` coefficients ``cT0, cT1, cT2``.
    """
    rho2 = rho * rho
    k = rho2 * R * R
    dk = 2.0 * rho2 * R
    bm, bp = (R - a) ** 2, (R + a) ** 2
    dbm, dbp = 2.0 * (R - a), 2.0 * (R + a)

    dlin0 = dk * bm + k * dbm
    dlin2 = dk * (bm + bp) + k * (dbm + dbp)
    dlin4 = dk * bp + k * dbp

    cT0, cT1, cT2 = _T_complex_coeffs(R, a, m0, xs, ys)
    zeta = complex(xs, ys)
    dn0 = -2.0 * zeta * R
    dn1 = 3.0 * R * R - 1.0 + a * zeta
    dn2 = -2.0 * a * R
    dcT0 = dn0 + dn1 + dn2
    dcT1 = 2j * (dn2 - dn0)
    dcT2 = dn1 - dn0 - dn2

    da0 = 2.0 * (dcT0 * cT0.conjugate()).real
    da1 = 2.0 * (dcT0 * cT1.conjugate() + cT0 * dcT1.conjugate()).real
    da2 = (2.0 * (dcT1 * cT1.conjugate()).real
           + 2.0 * (dcT0 * cT2.conjugate() + cT0 * dcT2.conjugate()).real)
    da3 = 2.0 * (dcT1 * cT2.conjugate() + cT1 * dcT2.conjugate()).real
    da4 = 2.0 * (dcT2 * cT2.conjugate()).real

    return [dlin0 - da0, -da1, dlin2 - da2, -da3, dlin4 - da4]


def _poly_eval(coeffs, t):
    r = 0.0
    for c in reversed(coeffs):
        r = r * t + c
    return r


def phi_value(R, theta, a, m0, xs, ys, rho):
    """(phi_lens, phi_from_P) -- two independent evaluations, must agree."""
    z = R * complex(math.cos(theta), math.sin(theta))
    zb = z.conjugate()
    fz = z - m0 / zb - (1.0 - m0) / (zb - a)
    phi_lens = 1.0 - abs(fz - complex(xs, ys)) ** 2 / (rho * rho)

    t = math.tan(theta / 2.0)
    A = 1.0 + t * t
    B = (R - a) ** 2 + (R + a) ** 2 * t * t
    P = _poly_eval(boundary_quartic(R, a, m0, xs, ys, rho), t)
    return phi_lens, P / (rho * rho * R * R * A * B)


@dataclass(frozen=True)
class LensParams:
    xs: float
    ys: float
    rho: float
    q: float
    a: float
    barycentric: bool = True

    @property
    def m0(self):
        return 1.0 / (1.0 + self.q)

    @property
    def m1(self):
        return self.q / (1.0 + self.q)

    def primary_frame_source(self):
        if self.barycentric:
            return self.xs + self.m1 * self.a, self.ys
        return self.xs, self.ys

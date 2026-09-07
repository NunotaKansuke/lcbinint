"""M3: the Gauss-Manin connection of the period family (plan sec. 7).

For a fixed source radius the boundary polynomial is ``P(t; R)`` and the
period integrand carries ``Y = sqrt(Q)``, ``Q = P * A * B`` (degree 8 in
``t``).  The seven forms

    eta_k = t^k dt / Y ,   k = 0 .. 6

close under ``d/dR`` modulo exact forms: there are polynomials ``S_k`` (deg
<= 7) and ``C_k`` (deg <= 6) with the *exact* identity

    -1/2 t^k Q_R = C_k Q + S_{k,t} Q - 1/2 S_k Q_t                    (GM)

(verified symbolically in checks/holonomic/symbolic_checks.py), which is the
statement

    d/dR ( t^k dt / Y ) = C_k dt / Y + d_t( S_k / Y ) dt .

Integrated over a cycle that encloses two branch points (so ``S_k / Y`` is
single-valued), the exact piece drops and

    d/dR  oint eta_k  =  sum_j (C_k)_j  oint eta_j .

This module builds the 7x7 matrix ``(C_k)_j`` exactly (sympy, from the exact
binary values of the inputs) and returns it as float64.  Speed never; this
is the oracle for the M7 C++ transport.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp

_t, _R = sp.symbols("t R", real=True)


def _q_expr_in_tR(a, m0, xs, ys, rho):
    """Q(t, R) = P A B as an exact sympy expression (a, m0, ... rational)."""
    a, m0, xs, ys, rho = (sp.Rational(x) for x in (a, m0, xs, ys, rho))
    zeta = xs + sp.I * ys
    n0 = -zeta * _R**2
    n1 = _R * (_R**2 - 1 + a * zeta)
    n2 = a * (m0 - _R**2)
    cT0 = n0 + n1 + n2
    cT1 = 2 * sp.I * (n2 - n0)
    cT2 = n1 - n0 - n2
    A = 1 + _t**2
    B = (_R - a) ** 2 + (_R + a) ** 2 * _t**2
    T = cT0 + cT1 * _t + cT2 * _t**2
    P = sp.expand(sp.re(sp.expand(rho**2 * _R**2 * A * B - T * sp.conjugate(T))))
    return sp.expand(P * A * B)


def q_coeffs_exact(R, a, m0, xs, ys, rho):
    """[q0 .. q8] of Q(.; R) as exact sympy numbers."""
    Qe = _q_expr_in_tR(a, m0, xs, ys, rho).subs(_R, sp.Rational(R))
    p = sp.Poly(sp.expand(Qe), _t)
    c = p.all_coeffs()[::-1]                       # ascending
    c += [sp.Integer(0)] * (9 - len(c))
    return c[:9]


def q_coeffs(R, a, m0, xs, ys, rho):
    return np.array([float(x) for x in q_coeffs_exact(R, a, m0, xs, ys, rho)])


@lru_cache(maxsize=64)
def _ck_sk_exact(a, m0, xs, ys, rho, R_str):
    """(C matrix rows, S polys) as sympy, for one rational R."""
    Rr = sp.Rational(R_str)
    base = _q_expr_in_tR(a, m0, xs, ys, rho)
    Qe = sp.expand(base.subs(_R, Rr))
    QRe = sp.expand(sp.diff(base, _R).subs(_R, Rr))
    Q = sp.Poly(Qe, _t)
    QR = sp.Poly(QRe, _t)
    Qt = sp.Poly(sp.diff(Qe, _t), _t)
    if Q.degree() != 8:
        raise ValueError(f"deg_t Q = {Q.degree()} != 8 (representation degenerate)")
    if sp.gcd(Q, Qt).degree() != 0:
        raise ValueError("Q has a repeated t-root at this R (branch collision)")
    inv = Qt.invert(Q)                             # (Q_t)^{-1} mod Q
    C_rows, S_polys = [], []
    two_Q = sp.Poly(sp.expand(2 * Qe), _t)
    for k in range(7):
        Sk = (sp.Poly(_t**k, _t) * QR * inv).rem(Q)
        num = sp.Poly(sp.expand(Sk.as_expr() * Qt.as_expr() - _t**k * QRe), _t)
        Tk, rem = sp.div(num, two_Q)
        if not rem.is_zero:
            raise ValueError(f"GM division has nonzero remainder at k={k}")
        Ck = sp.expand(Tk.as_expr() - sp.diff(Sk.as_expr(), _t))
        idr = sp.expand(-sp.Rational(1, 2) * _t**k * QRe
                        - (Ck * Qe + sp.diff(Sk.as_expr(), _t) * Qe
                           - sp.Rational(1, 2) * Sk.as_expr() * Qt.as_expr()))
        if idr != 0:
            raise ValueError(f"GM identity residual != 0 at k={k}: {idr}")
        cp = sp.Poly(Ck, _t) if Ck != 0 else None
        if cp is not None and cp.degree() > 6:
            raise ValueError(f"deg C_{k} = {cp.degree()} > 6")
        row = [sp.Integer(0)] * 7
        if cp is not None:
            for (e,), coef in cp.terms():
                row[e] = coef
        C_rows.append(row)
        S_polys.append(Sk)
    return C_rows, S_polys


def _r_key(R):
    return repr(sp.Rational(R))


def connection_matrix(R, a, m0, xs, ys, rho):
    """7x7 float64 ``M`` with  d/dR oint eta_k = sum_j M[k, j] oint eta_j."""
    rows, _ = _ck_sk_exact(a, m0, xs, ys, rho, _r_key(R))
    return np.array([[float(x) for x in row] for row in rows])


def gm_polynomials(R, a, m0, xs, ys, rho):
    """Float coefficient arrays (ascending) for one rational R:

        Q, Q_R, Q_t  and  S_k, C_k (k = 0..6)

    satisfying, exactly,   -1/2 t^k Q_R = C_k Q + S_k' Q - 1/2 S_k Q_t .
    """
    Rr = sp.Rational(R)
    base = _q_expr_in_tR(a, m0, xs, ys, rho)
    Qe = sp.expand(base.subs(_R, Rr))
    QRe = sp.expand(sp.diff(base, _R).subs(_R, Rr))
    Qte = sp.expand(sp.diff(Qe, _t))

    def asc(expr, n):
        c = sp.Poly(expr, _t).all_coeffs()[::-1] if expr != 0 else [0.0]
        return np.array([float(x) for x in c] + [0.0] * (n - len(c)))[:n]

    rows, s_list = _ck_sk_exact(a, m0, xs, ys, rho, _r_key(R))
    two_Q = sp.Poly(sp.expand(2 * Qe), _t)
    C_polys = []
    for k in range(7):
        Sk = s_list[k]
        num = sp.Poly(sp.expand(Sk.as_expr() * Qte - _t**k * QRe), _t)
        Tk, _r = sp.div(num, two_Q)
        Ck = sp.expand(Tk.as_expr() - sp.diff(Sk.as_expr(), _t))
        C_polys.append(asc(Ck, 8))
    return {
        "Q": asc(Qe, 9), "Q_R": asc(QRe, 9), "Q_t": asc(Qte, 9),
        "S": [asc(s.as_expr(), 8) for s in s_list],
        "C": C_polys,
    }


def s_polynomials(R, a, m0, xs, ys, rho):
    """The S_k (k=0..6) as ascending-coefficient float lists (deg <= 7)."""
    _, sp_list = _ck_sk_exact(a, m0, xs, ys, rho, _r_key(R))
    out = []
    for Sk in sp_list:
        expr = Sk.as_expr()
        c = sp.Poly(expr, _t).all_coeffs()[::-1] if expr != 0 else [0.0]
        c = [float(x) for x in c] + [0.0] * (8 - len(c))
        out.append(c[:8])
    return out

"""M4: period transport across a normal radial cell (plan sec. 7-8).

The seven monomial periods ``eta_k = t^k dt / sqrt(Q)`` obey the exact
Gauss-Manin connection ``d/dR oint eta = C_eta(R) oint eta`` built in
:mod:`connection`.  The residue-free basis (plan sec. 7)

    psi = W(R) @ eta ,   W = [[I_3 | 0], [ * | I_3]]  (b1,b2,b3 in column 3)

is closed under ``d/dR`` too, with the 6x6 connection

    C_psi = ( W'(R) + W(R) @ C_eta(R) )[:, (0,1,2,4,5,6)]

*provided* the eta_3 column of ``M = W' + W C_eta`` cancels against
``b1,b2,b3`` -- i.e.

    M[:, 3] + b1 M[:, 4] + b2 M[:, 5] + b3 M[:, 6]  ==  0                (closure)

which is the second-kind statement.  :func:`psi_connection` returns that
residual; the caller fails closed (``BASIS_DEGENERATE``) if it is not tiny.

Transport itself (:func:`transport_psi`) is a stiff linear ODE integration.
The raw monomial / psi bases are ill-conditioned across a full cell (plan
sec. 15, "monomial 基底の悪条件, 要対策"): ``cond(C) ~ 1e3-1e8`` per cell and
a single midpoint anchor loses 2-4 digits by the cell edge.  This module
therefore *records* the per-cell condition numbers (the M4 gate) and the
reference flux (:mod:`flux`) re-anchors the period at every radial
quadrature node instead of trusting a single transported seed.  A
well-conditioned flux-priority basis is deferred to M7/M8 (plan sec. 8).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from .connection import (
    _q_expr_in_tR,
    connection_matrix,
    connection_matrix_numeric,
    q_coeffs_dR_numeric,
    q_coeffs_numeric,
)

_t, _R = sp.symbols("t R", real=True)

# indices of eta kept in psi (eta_3 is the lone residue-carrying form)
_PSI_IDX = (0, 1, 2, 4, 5, 6)


def eta_connection(R, params):
    """7x7 float ``C_eta(R)`` -- alias of :func:`connection.connection_matrix`."""
    return connection_matrix(R, *params)


def _b_and_bprime_numeric(R, params):
    """(b1,b2,b3) and (b1',b2',b3') at ``R`` from the float ``q``/``q_R``."""
    q = q_coeffs_numeric(R, *params)
    dq = q_coeffs_dR_numeric(R, *params)
    q5, q6, q7, q8 = q[5], q[6], q[7], q[8]
    d5, d6, d7, d8 = dq[5], dq[6], dq[7], dq[8]

    def _dquot(u, du, v, dv):
        return (du * v - u * dv) / (v * v)

    b1 = -q7 / (2 * q8)
    b1p = -0.5 * _dquot(q7, d7, q8, d8)
    b2 = -q6 / (2 * q8) + 3 * q7**2 / (8 * q8**2)
    b2p = (-0.5 * _dquot(q6, d6, q8, d8)
           + (3.0 / 8.0) * _dquot(q7**2, 2 * q7 * d7, q8**2, 2 * q8 * d8))
    b3 = (-q5 / (2 * q8) + 3 * q7 * q6 / (4 * q8**2)
          - 5 * q7**3 / (16 * q8**3))
    b3p = (-0.5 * _dquot(q5, d5, q8, d8)
           + 0.75 * _dquot(q7 * q6, d7 * q6 + q7 * d6, q8**2, 2 * q8 * d8)
           - (5.0 / 16.0) * _dquot(q7**3, 3 * q7**2 * d7, q8**3, 3 * q8**2 * d8))
    return (b1, b2, b3), (b1p, b2p, b3p)


@lru_cache(maxsize=32)
def _b_exprs(params):
    """(b1, b2, b3) and their R-derivatives as sympy expressions in ``_R``."""
    base = sp.Poly(sp.expand(_q_expr_in_tR(*params)), _t)
    cd = {e[0]: c for e, c in base.terms()}
    q8, q7, q6, q5 = (cd.get(k, sp.Integer(0)) for k in (8, 7, 6, 5))
    b = [
        -q7 / (2 * q8),
        -q6 / (2 * q8) + 3 * q7**2 / (8 * q8**2),
        -q5 / (2 * q8) + 3 * q7 * q6 / (4 * q8**2) - 5 * q7**3 / (16 * q8**3),
    ]
    bp = [sp.diff(x, _R) for x in b]
    return b, bp


def _W_and_Wprime(bval, bpval):
    W = np.zeros((6, 7))
    W[0, 0] = W[1, 1] = W[2, 2] = 1.0
    W[3, 4] = W[4, 5] = W[5, 6] = 1.0
    W[3, 3], W[4, 3], W[5, 3] = -bval[0], -bval[1], -bval[2]
    Wp = np.zeros((6, 7))
    Wp[3, 3], Wp[4, 3], Wp[5, 3] = -bpval[0], -bpval[1], -bpval[2]
    return W, Wp


def _assemble_psi(Ceta, bval, bpval):
    W, Wp = _W_and_Wprime(bval, bpval)
    M = Wp + W @ Ceta                                   # 6x7
    resid = (M[:, 3] + bval[0] * M[:, 4]
             + bval[1] * M[:, 5] + bval[2] * M[:, 6])
    return M[:, _PSI_IDX].copy(), resid


def psi_connection(R, params):
    """``(C_psi, closure_residual)`` -- float polynomial Gauss-Manin path.

    ``C_psi`` is the 6x6 connection of the residue-free basis;
    ``closure_residual`` (6-vector) is ``M[:,3] + sum_j b_j M[:,3+j]`` and
    must be ~0 (second-kind closure -- the caller fails closed otherwise).
    """
    bval, bpval = _b_and_bprime_numeric(R, params)
    Ceta = connection_matrix_numeric(R, *params)
    return _assemble_psi(Ceta, bval, bpval)


def psi_connection_exact(R, params):
    """Sympy oracle for :func:`psi_connection` (exact ``C_eta`` and ``b_j``)."""
    Rr = sp.Rational(R)
    b, bp = _b_exprs(params)
    bval = [float(x.subs(_R, Rr)) for x in b]
    bpval = [float(x.subs(_R, Rr)) for x in bp]
    Ceta = connection_matrix(R, *params)
    return _assemble_psi(Ceta, bval, bpval)


def transport_psi(R0, R1, Pi0, params, *, rtol=1e-11, atol=1e-14, n_nodes=17):
    """Transport the closed-contour psi-period vector from ``R0`` to ``R1``.

    ``C_psi`` is sampled on ``n_nodes`` Chebyshev points of ``[R0, R1]`` and
    Chebyshev-interpolated; the linear system is integrated with DOP853.

    Returns ``(Pi1, cond_max)`` -- the transported 6-vector and the largest
    ``cond(C_psi)`` seen on the node grid (recorded per the M4 gate).
    """
    Pi0 = np.asarray(Pi0, dtype=float)
    lo, hi = (R0, R1) if R0 <= R1 else (R1, R0)
    mid, hw = 0.5 * (lo + hi), 0.5 * (hi - lo)
    xg = np.cos(np.pi * np.arange(n_nodes) / (n_nodes - 1))
    Rg = mid + hw * xg
    Cs = np.empty((n_nodes, 6, 6))
    cond_max = 0.0
    for i, Rn in enumerate(Rg):
        Cs[i], _res = psi_connection(float(Rn), params)
        cond_max = max(cond_max, float(np.linalg.cond(Cs[i])))
    deg = min(n_nodes - 1, 12)
    coef = np.empty((6, 6, deg + 1))
    for i in range(6):
        for j in range(6):
            coef[i, j] = np.polynomial.chebyshev.chebfit(xg, Cs[:, i, j], deg)

    def rhs(R, y):
        x = (R - mid) / hw
        C = np.empty((6, 6))
        for i in range(6):
            for j in range(6):
                C[i, j] = np.polynomial.chebyshev.chebval(x, coef[i, j])
        return C @ y

    sol = solve_ivp(rhs, (R0, R1), Pi0, method="DOP853",
                    rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"transport_psi: {sol.message}")
    return sol.y[:, -1], cond_max


def cell_conditioning(r_lo, r_hi, params, *, n=7, exact=True):
    """Per-cell condition-number record (plan M4 gate: '条件数記録').

    Samples ``cond(C_eta)`` and ``cond(C_psi)`` on ``n`` interior radii and
    returns the max / median of each plus the worst psi-closure residual.

    ``exact`` (default) builds ``C_eta`` and ``b_j`` from the sympy oracle:
    the float monomial Gauss-Manin is itself unreliable near the theta -> pi
    degree drop and would report spurious 1e20 condition numbers.  Pass
    ``exact=False`` for a fast, benign-config-only preview.
    """
    w = r_hi - r_lo
    Rs = r_lo + w * np.linspace(0.06, 0.94, n)
    ce, cp, res = [], [], 0.0
    for R in Rs:
        R = float(R)
        if exact:
            Ceta = connection_matrix(R, *params)
            Cp, r = psi_connection_exact(R, params)
        else:
            Ceta = connection_matrix_numeric(R, *params)
            Cp, r = psi_connection(R, params)
        ce.append(float(np.linalg.cond(Ceta)))
        cp.append(float(np.linalg.cond(Cp)))
        res = max(res, float(np.max(np.abs(r))))
    return {
        "cond_eta_max": max(ce), "cond_eta_med": float(np.median(ce)),
        "cond_psi_max": max(cp), "cond_psi_med": float(np.median(cp)),
        "psi_closure_resid": res,
    }

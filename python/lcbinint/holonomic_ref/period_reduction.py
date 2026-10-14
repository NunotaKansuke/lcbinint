"""M3: period reduction -- 7 forms -> residue-free 6 forms -> observable
(plan sec. 5-7).

The linear limb-darkening half-moment of one image arc is a *period*

    Phi_arc(R) = integral_{t1}^{t2} (P / A)(t) dt / sqrt(Q(t))         (obs)

with ``Q = P A B`` and ``t1 < t2`` the two boundary roots of the arc.  The
substitution ``t = tan(theta/2)`` turns it into a bare angular integral
straight from the lens equation,

    Phi_arc(R) = (rho R / 2) * integral_arc sqrt(phi(R, theta)) dtheta ,

which is this module's ground truth.

The design reduces (obs) algebraically:

  * LD reduction (plan sec. 5-6):  P/A == H + d_t( S Y / A )  as forms,
    with S = -t/(4 a R), deg_t H <= 6, so

        Phi_arc = sum_{k=0}^{6} h_k I_k ,   I_k = integral t^k dt / sqrt(Q) .

  * residue-free basis (plan sec. 7):  eta_k = t^k dt / sqrt(Q).  Of
    eta_0..eta_6 only eta_3 carries a residue at infinity; the six

        psi = (eta0, eta1, eta2, eta4-b1 eta3, eta5-b2 eta3, eta6-b3 eta3)

    are residue-free (b1,b2,b3 from the top coefficients q5..q8 of Q).  As
    P/(A Y) dt is itself second kind, the eta_3 part of H vanishes:

        h3'  :=  h3 + b1 h4 + b2 h5 + b3 h6  ==  0                (residue cond.)

    and  Phi_arc = c^T Pi ,  c = (h0,h1,h2,h4,h5,h6),  Pi = integral psi .
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy import integrate

from .connection import (
    _q_expr_in_tR,
    q_coeffs,
    q_coeffs_exact,
    q_coeffs_numeric,
)
from .polynomial_family import boundary_quartic


def _asc_add(*polys):
    n = max(len(p) for p in polys)
    out = np.zeros(n)
    for p in polys:
        out[: len(p)] += np.asarray(p, dtype=float)
    return out


def h_coeffs_numeric(R, a, m0, xs, ys, rho):
    """[h0 .. h6] of ``H(.; R)`` by float polynomial arithmetic (no sympy).

    ``H = (2 (R+a)^2 P + t (B P_t + B_t P)) / (8 a R)`` (plan sec. 6).  The
    fast path for the flux assembly; :func:`h_coeffs_exact` is the oracle.
    """
    P = np.asarray(boundary_quartic(R, a, m0, xs, ys, rho), dtype=float)
    Pt = np.array([P[1], 2.0 * P[2], 3.0 * P[3], 4.0 * P[4]])
    B = np.array([(R - a) ** 2, 0.0, (R + a) ** 2])
    Bt = np.array([0.0, 2.0 * (R + a) ** 2])
    inner = _asc_add(np.convolve(B, Pt), np.convolve(Bt, P))   # deg 5
    t_inner = np.concatenate([[0.0], inner])                   # * t  -> deg 6
    H = _asc_add(2.0 * (R + a) ** 2 * P, t_inner) / (8.0 * a * R)
    out = np.zeros(7)
    out[: min(7, len(H))] = H[:7]
    if len(H) > 7 and np.any(np.abs(H[7:]) > 1e-6 * (np.abs(H).max() + 1e-300)):
        raise ValueError("deg_t H > 6 (numeric)")
    return out

_t = sp.symbols("t", real=True)
_Rs = sp.Symbol("R", real=True)


# ---------------------------------------------------------------- H(t; R) ---
def _H_expr(a, m0, xs, ys, rho):
    a_, m0_, xs_, ys_, rho_ = (sp.Rational(x) for x in (a, m0, xs, ys, rho))
    zeta = xs_ + sp.I * ys_
    n0 = -zeta * _Rs**2
    n1 = _Rs * (_Rs**2 - 1 + a_ * zeta)
    n2 = a_ * (m0_ - _Rs**2)
    cT0, cT1, cT2 = n0 + n1 + n2, 2 * sp.I * (n2 - n0), n1 - n0 - n2
    At = 1 + _t**2
    Bt = (_Rs - a_) ** 2 + (_Rs + a_) ** 2 * _t**2
    T = cT0 + cT1 * _t + cT2 * _t**2
    P = sp.expand(sp.re(sp.expand(rho_**2 * _Rs**2 * At * Bt - T * sp.conjugate(T))))
    H = (2 * (_Rs + a_) ** 2 * P
         + _t * (Bt * sp.diff(P, _t) + sp.diff(Bt, _t) * P)) / (8 * a_ * _Rs)
    return sp.expand(H)


def h_coeffs_exact(R, a, m0, xs, ys, rho):
    """[h0 .. h6] of H(.; R), exact."""
    He = sp.expand(_H_expr(a, m0, xs, ys, rho).subs(_Rs, sp.Rational(R)))
    p = sp.Poly(He, _t)
    if p.degree() > 6:
        raise ValueError(f"deg_t H = {p.degree()} > 6")
    c = p.all_coeffs()[::-1]
    c += [sp.Integer(0)] * (7 - len(c))
    return c[:7]


def h_coeffs(R, a, m0, xs, ys, rho):
    return np.array([float(x) for x in h_coeffs_exact(R, a, m0, xs, ys, rho)])


# ------------------------------------------------- residue-free reduction ---
def residue_b_exact(qc):
    """(b1, b2, b3) from q5..q8 (plan sec. 7); exact if qc is exact."""
    q8, q7, q6, q5 = qc[8], qc[7], qc[6], qc[5]
    b1 = -q7 / (2 * q8)
    b2 = -q6 / (2 * q8) + 3 * q7**2 / (8 * q8**2)
    b3 = -q5 / (2 * q8) + 3 * q7 * q6 / (4 * q8**2) - 5 * q7**3 / (16 * q8**3)
    return b1, b2, b3


def laurent_b(qc, n=6):
    """[b0=1, b1, .., bn]: coeffs of (Q / q8 / t^8)^{-1/2} at t = infinity.

    Independent series cross-check of :func:`residue_b_exact`.
    """
    q8 = qc[8]
    s = sp.symbols("s")                       # s = 1/t
    u = sum((qc[8 - j] / q8) * s**j for j in range(1, 9))
    ser = sp.series((1 + u) ** sp.Rational(-1, 2), s, 0, n + 1).removeO()
    poly = sp.Poly(sp.expand(ser), s)
    return [poly.coeff_monomial(s**j) for j in range(n + 1)]


def psi_reduction_matrix(qc):
    """6x7 float ``W`` with  psi = W @ (eta0, .., eta6)."""
    b1, b2, b3 = (float(x) for x in residue_b_exact(qc))
    W = np.zeros((6, 7))
    W[0, 0] = W[1, 1] = W[2, 2] = 1.0
    W[3, 4], W[3, 3] = 1.0, -b1
    W[4, 5], W[4, 3] = 1.0, -b2
    W[5, 6], W[5, 3] = 1.0, -b3
    return W


def residue_at_infinity(num_coeffs, qc):
    """Residue at t = infinity of  N(t) dt / sqrt(Q)  (N ascending, deg <= 6)."""
    bb = laurent_b(qc, n=6)                   # [b0=1, b1, ..., b6]
    N = list(num_coeffs) + [sp.Integer(0)] * (7 - len(num_coeffs))
    acc = sp.Integer(0)
    for i in range(7):
        j = i - 3
        if 0 <= j < len(bb):
            acc += N[i] * bb[j]
    return acc / sp.sqrt(qc[8])


def observed_covector_exact(R, a, m0, xs, ys, rho):
    """(c, h3prime).  c = (h0,h1,h2,h4,h5,h6); h3prime must vanish."""
    hc = h_coeffs_exact(R, a, m0, xs, ys, rho)
    qc = q_coeffs_exact(R, a, m0, xs, ys, rho)
    b1, b2, b3 = residue_b_exact(qc)
    h3p = sp.expand(hc[3] + b1 * hc[4] + b2 * hc[5] + b3 * hc[6])
    c = [hc[0], hc[1], hc[2], hc[4], hc[5], hc[6]]
    return c, h3p


# ---------------------------------------------------------- numeric periods -
_TWO_PI = 2.0 * math.pi


def _wrap_pi(x):
    return (x + math.pi) % _TWO_PI - math.pi


def arc_chart(a, m0, xs, ys, rho, arc):
    """Pick a chart in which ``arc`` is a *bounded* ``t``-interval.

    ``t = tan(theta / 2)`` is singular at ``theta = pi``.  If the arc does not
    contain ``theta = pi`` the natural chart (chart 1) works.  If it does, use
    chart 2: ``u = -1/t = tan((theta - pi) / 2)``, which is the same boundary
    family with the lens configuration reflected through the origin
    (``a -> -a``, ``zeta -> -zeta``) -- verified in
    checks: ``P_u(u) == u^4 P(-1/u)``.

    The chart is chosen so the arc sits near ``t = 0`` (well conditioned):
    chart 2 whenever the arc midpoint is in the left half plane
    (``cos(theta_mid) < 0``), chart 1 otherwise.

    Returns ``(params, (t_lo, t_hi), shift)`` with ``params`` the 5-tuple
    ``(a, m0, xs, ys, rho)`` for the chosen chart, ``t_lo < t_hi`` the arc's
    endpoints in that chart, and ``shift`` in ``{0, pi}``.

    Raises ``ValueError`` if the arc still contains the chosen chart's
    singular angle (arc longer than a half turn straddling both ``0`` and
    ``pi``) -- fail closed.
    """
    meas = arc.measure
    use_chart2 = math.cos(arc.midpoint) < 0.0
    if use_chart2:
        params, shift, sing = (-a, m0, -xs, -ys, rho), math.pi, 0.0
    else:
        params, shift, sing = (a, m0, xs, ys, rho), 0.0, math.pi
    d_sing = (sing - arc.theta_enter) % _TWO_PI
    if 0.0 < d_sing < meas:
        raise ValueError("arc spans both theta=0 and theta=pi; no regular "
                         "tan chart (arc too long -- near a full circle)")
    w1 = math.tan(_wrap_pi(arc.theta_enter - shift) / 2.0)
    w2 = math.tan(_wrap_pi(arc.theta_leave - shift) / 2.0)
    t_lo, t_hi = sorted((w1, w2))
    return params, (t_lo, t_hi), shift


def _deflated_quadratic(pc, t1, t2):
    """p4(t-t3)(t-t4) = p4 t^2 + d1 t + d0, by dividing P by (t-t1)(t-t2)."""
    p4 = pc[4]
    beta, gamma = -(t1 + t2), t1 * t2
    d1 = pc[3] - beta * p4
    d0 = pc[2] - beta * d1 - gamma * p4
    return p4, d1, d0


def arc_t_endpoints(R, a, m0, xs, ys, rho, arc):
    """(t_lo, t_hi) -- the arc's chart coordinates (chart chosen automatically)."""
    _p, (t_lo, t_hi), _s = arc_chart(a, m0, xs, ys, rho, arc)
    return t_lo, t_hi


def half_period_eta(R, a, m0, xs, ys, rho, arc, kmax=6):
    """[I_0 .. I_kmax],  I_k = integral_{t1}^{t2} t^k dt / sqrt(Q(t)).

    ``t1 < t2`` are the arc endpoints in the chart :func:`arc_chart` picks; the
    integrand and ``Q`` are evaluated with that chart's (possibly reflected)
    parameters, so a ``theta = pi`` straddling arc becomes bounded first.

    Deflate the boundary pair from ``P``:

        g(t) = Q(t) / ((t - t1)(t2 - t)) = -(p4 t^2 + d1 t + d0) A(t) B(t)

    is smooth and positive on ``[t1, t2]``; with ``t = mid + half u``,

        I_k = integral_{-1}^{1} (mid+half u)^k / sqrt(g)  (1-u^2)^-1/2 du ,

    a smooth integrand against the Chebyshev weight -- QAWSE is spectral.
    """
    (ca, cm0, cxs, cys, crho), (t1, t2), _shift = \
        arc_chart(a, m0, xs, ys, rho, arc)
    pc = boundary_quartic(R, ca, cm0, cxs, cys, crho)
    p4, d1, d0 = _deflated_quadratic(pc, t1, t2)
    half = 0.5 * (t2 - t1)
    mid = 0.5 * (t1 + t2)
    bm, bp = (R - ca) ** 2, (R + ca) ** 2

    def g(t):
        A = 1.0 + t * t
        B = bm + bp * t * t
        return -(d0 + t * (d1 + t * p4)) * A * B

    if g(mid) <= 0.0:
        raise ValueError("deflated integrand non-positive; arc/chart mismatch")

    out = []
    for k in range(kmax + 1):
        def f(u, k=k):
            t = mid + half * u
            return t**k / math.sqrt(g(t))
        val, _e = integrate.quad(f, -1.0, 1.0, weight="alg", wvar=(-0.5, -0.5),
                                 limit=200, epsabs=1e-13, epsrel=1e-13)
        out.append(val)
    return np.array(out)


def half_period_obs_angular(R, a, m0, xs, ys, rho, arc):
    """(rho R / 2) integral_arc sqrt(phi) dtheta -- ground truth for Phi_arc.

    ``phi`` vanishes linearly at each end, so ``sqrt(phi)`` carries a
    square-root edge; integrate against the Chebyshev weight
    ``(1-u)^{1/2}(1+u)^{1/2}`` after mapping the arc to ``[-1, 1]``.
    """
    lo = arc.theta_enter
    half = 0.5 * arc.measure
    mid = lo + half

    def integrand(u):
        theta = mid + half * u
        z = R * complex(math.cos(theta), math.sin(theta))
        zb = z.conjugate()
        fz = z - m0 / zb - (1.0 - m0) / (zb - a)
        phi = 1.0 - abs(fz - complex(xs, ys)) ** 2 / (rho * rho)
        s = 1.0 - u * u
        if phi <= 0.0 or s <= 0.0:
            return 0.0
        return math.sqrt(phi / s)

    val, _e = integrate.quad(integrand, -1.0, 1.0, weight="alg",
                             wvar=(0.5, 0.5), limit=400,
                             epsabs=1e-13, epsrel=1e-13)
    return 0.5 * rho * R * half * val


def _chart_h_c(R, params):
    """(h_coeffs float, c float, W) for the given chart params."""
    ca, cm0, cxs, cys, crho = params
    Rr = sp.Rational(R) if not isinstance(R, sp.Basic) else R
    hc = h_coeffs_exact(Rr, *(sp.Rational(x) for x in params))
    qc_e = q_coeffs_exact(Rr, *(sp.Rational(x) for x in params))
    b1, b2, b3 = residue_b_exact(qc_e)
    h3p = sp.expand(hc[3] + b1 * hc[4] + b2 * hc[5] + b3 * hc[6])
    hcf = np.array([float(x) for x in hc])
    cf = np.array([float(hc[i]) for i in (0, 1, 2, 4, 5, 6)])
    W = psi_reduction_matrix(np.array([float(x) for x in qc_e]))
    return hcf, cf, W, h3p


def half_period_obs_reduced(R, a, m0, xs, ys, rho, arc, *, basis="6D"):
    """Algebraic reduction of Phi_arc.

    ``basis="7D"``  -> sum_k h_k I_k          (monomial forms)
    ``basis="6D"``  -> c^T (W @ I)            (residue-free forms)

    Both are evaluated in the chart :func:`arc_chart` selects, so they are
    valid for ``theta = pi`` straddling arcs too.
    """
    params, _tt, _s = arc_chart(a, m0, xs, ys, rho, arc)
    I = half_period_eta(R, a, m0, xs, ys, rho, arc)
    hcf, cf, W, _h3p = _chart_h_c(R, params)
    if basis == "7D":
        return float(hcf @ I)
    return float(cf @ (W @ I))


def phi_arc_reduced_numeric(R, a, m0, xs, ys, rho, arc, *, basis="6D"):
    """Fast (sympy-free) evaluation of ``Phi_arc(R)`` for the flux assembly.

    Same result as :func:`half_period_obs_reduced` but every coefficient
    comes from the float polynomial fast paths (:func:`h_coeffs_numeric`,
    :func:`connection.q_coeffs_numeric`).  Returns ``(value, h3_residual)``;
    ``h3_residual`` is the second-kind check ``h3 + b1 h4 + b2 h5 + b3 h6``
    and the caller fails closed if it is not small relative to ``|h|``.
    """
    params, _tt, _s = arc_chart(a, m0, xs, ys, rho, arc)
    I = half_period_eta(R, a, m0, xs, ys, rho, arc)
    hcf = h_coeffs_numeric(R, *params)
    qc = q_coeffs_numeric(R, *params)
    b1, b2, b3 = residue_b_exact(qc)
    h3res = hcf[3] + b1 * hcf[4] + b2 * hcf[5] + b3 * hcf[6]
    if basis == "7D":
        val = float(hcf @ I)
    else:
        W = psi_reduction_matrix(qc)
        cf = hcf[[0, 1, 2, 4, 5, 6]]
        val = float(cf @ (W @ I))
    return val, float(h3res)


def closed_period_eta(R, a, m0, xs, ys, rho, arc, kmax=6, n=8000, pad=1.35):
    """oint_Gamma t^k dt / sqrt(Q)  over an ellipse enclosing [t1,t2] only.

    Complex contour, no endpoint singularity -- validates the Gauss-Manin
    connection (d/dR of this vector = C @ this vector).  The minor axis is
    shrunk until exactly the two branch points t1, t2 lie inside.

    Secondary (soft) check only -- the rigorous connection test is the
    pointwise polynomial Gauss-Manin identity in :mod:`connection`."""
    params, (w1, w2), _s = arc_chart(a, m0, xs, ys, rho, arc)
    qc = q_coeffs(R, *params)
    pc = boundary_quartic(R, *params)
    Q = np.polynomial.Polynomial(qc)
    qroots = np.roots(qc[::-1])
    proots = np.roots(pc[::-1])

    # true branch points: the two P-roots bracketing the arc
    order = sorted(range(len(proots)),
                   key=lambda i: min(abs(proots[i] - w1), abs(proots[i] - w2)))
    t1, t2 = sorted((proots[order[0]].real, proots[order[1]].real))

    span = t2 - t1
    mid = 0.5 * (t1 + t2)
    ax = 0.5 * span * pad
    ay = 0.5 * span * 0.45
    for _ in range(40):
        inside = int(np.sum(((qroots.real - mid) / ax) ** 2
                            + (qroots.imag / ay) ** 2 < 1.0))
        others = [r for r in qroots
                  if min(abs(r - t1), abs(r - t2)) > 1e-6 * span]
        near = min((abs(((r.real - mid) / ax) ** 2 + (r.imag / ay) ** 2 - 1.0)
                    for r in others), default=1.0)
        if inside == 2 and near > 0.15:
            break
        ay *= 0.6
        if ay < 1e-9 * span:
            raise ValueError("closed_period_eta: cannot isolate the branch pair")
    ph = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    z = mid + ax * np.cos(ph) + 1j * ay * np.sin(ph)
    dz = (-ax * np.sin(ph) + 1j * ay * np.cos(ph)) * (2.0 * math.pi / n)

    qv = Q(z)
    Y = np.sqrt(np.abs(qv)) * np.exp(0.5j * np.unwrap(np.angle(qv)))
    return np.array([np.sum(z**k / Y * dz) for k in range(kmax + 1)]), (t1, t2)

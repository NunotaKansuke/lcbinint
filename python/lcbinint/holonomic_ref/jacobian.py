"""M6: the 5-component Jacobian of the epoch finite-source magnification.

Plan sec. 11.  ``value`` and ``JVP`` are produced by the *same* finite
numerical reconstruction: a per-cell fixed Gauss-Chebyshev (1st kind)
radial rule, one radial pass that -- at every node -- enumerates the arc
endpoints, then accumulates

    F0    = int R * sum_arcs ( theta_leave - theta_enter ) dR
    F_1/2 = int R * sum_arcs int_arc sqrt(phi) dtheta       dR

together with all five parameter derivatives.  The internal parameter
vector is the *primary frame*

    P = (X, Y, rho, m0, a)

(``X, Y`` = primary-frame source position, ``m0`` = mass fraction of the
lens at the origin, ``a`` = separation); :func:`_internal_to_user_jac`
maps a covector back to the user parameters ``p = (xs, ys, rho, q, a)``.

Endpoint derivatives.  Each arc endpoint ``theta*`` satisfies
``phi(R, theta*; P) = 0``, so by the implicit function theorem

    d theta* / d P_j = - (d phi / d P_j) / (d phi / d theta) .          (IFT-theta)

This is chart-free: it never parametrises the endpoint by ``t = tan(theta/2)``
and so has no pole at ``theta = pi``.  The alternative ``t``-chart form

    d t* / d P_j = - (d P / d P_j) / (d P / d t) ,
    d theta* / d P_j = 2 / (1 + t*^2) * d t* / d P_j                    (IFT-t)

is used only by :func:`endpoint_dtheta_t_chart` for the representation-
invariance cross-check (it blows up for a ``theta = pi``-straddling arc,
where IFT-theta stays finite).

The ``F_1/2`` boundary terms ``sqrt(phi) * d theta*/d P`` vanish because
``sqrt(phi) = 0`` at every endpoint, so only the interior
``d phi / d P / (2 sqrt(phi))`` integral contributes -- itself handled by
the Gauss-Chebyshev angular weight that absorbs the ``1/sqrt`` edge.

Arc enumeration is done from the *real roots of the boundary quartic*
``P(t; R)`` directly (:func:`boundary_quartic`), not from
:func:`topology.arcs_at`.  The grid sign-scan in ``arcs_at`` misses a
newborn arc thinner than its step over a finite sub-interval above a
tangency (``physical_real`` / D14) cell edge; the value tolerates that
(``int sqrt(R - R_e) dR ~ w^{3/2}``) but the derivative does not
(``int R_e' / sqrt(R - R_e) dR ~ sqrt(w)``, an O(few %) miss).  The
quartic roots resolve the arc as soon as its half-gap exceeds ~1e-9 in
``t``.  ``arcs_at`` remains the fallback for the rare degenerate node
(leading coefficient ~ 0 at a ``theta = pi`` crossing, or an odd real-root
count).

Fail closed: any node that hits a near-tangency (``|d phi/d theta|`` below
a relative floor), a degenerate quartic, a ``full`` cell, or the on-axis
singular patch marks the whole Jacobian ``GRADIENT_UNRELIABLE`` rather
than returning a silently wrong number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .polynomial_family import LensParams, boundary_quartic, boundary_quartic_dp
from .topology import _phi, arcs_at, classify_cells

_TWO_PI = 2.0 * math.pi

# relative floor on |d phi / d theta| at an endpoint, scaled by rho / R:
# below this the endpoint is a near-tangency and its IFT derivative is
# not trustworthy.
_TAN_REL = 1e-4
# |Im t| tolerance (relative) for calling a quartic root real
_ROOT_IM_REL = 1e-8
# |p4| / max|p_i| below which the quartic is treated as degenerate
_P4_DEGEN_REL = 1e-11


# ---- Chebyshev-1st nodes/weights: int_{-1}^{1} f dx ~ sum W_i f(x_i) ------
def _cheb1(n: int):
    i = np.arange(1, n + 1)
    x = np.cos((2.0 * i - 1.0) * math.pi / (2.0 * n))
    w = (math.pi / n) * np.sqrt(1.0 - x * x)   # carries the sqrt(1-x^2) factor
    return x, w


_ANG_X, _ANG_W = _cheb1(64)                       # angular rule
_FULL_TH = np.linspace(0.0, _TWO_PI, 256, endpoint=False)   # full-circle rule


# ---- phi and its closed-form gradient wrt the internal params ------------
def phi_grad(R, theta, a, m0, X, Y, rho):
    """``(phi, d phi/d theta, (d phi/d X, .., d phi/d a))`` in closed form.

    ``phi = 1 - |f(z) - zeta|^2 / rho^2`` with ``z = R e^{i theta}``,
    ``f(z) = z - m0 / zbar - m1 / (zbar - a)``, ``m1 = 1 - m0``,
    ``zeta = X + i Y``.  All five derivatives are analytic.
    """
    ct, st = math.cos(theta), math.sin(theta)
    z = complex(R * ct, R * st)
    zb = z.conjugate()
    m1 = 1.0 - m0
    f = z - m0 / zb - m1 / (zb - a)
    g = f - complex(X, Y)
    r2 = rho * rho
    g2 = g.real * g.real + g.imag * g.imag
    phi = 1.0 - g2 / r2

    fzb = m0 / zb**2 + m1 / (zb - a) ** 2
    dg_dtheta = 1j * z - 1j * zb * fzb
    dphi_dtheta = -(2.0 / r2) * (g.conjugate() * dg_dtheta).real

    dphi_dX = (2.0 / r2) * g.real
    dphi_dY = (2.0 / r2) * g.imag
    dphi_drho = 2.0 * g2 / (rho * r2)
    df_dm0 = -1.0 / zb + 1.0 / (zb - a)
    dphi_dm0 = -(2.0 / r2) * (g.conjugate() * df_dm0).real
    df_da = -m1 / (zb - a) ** 2
    dphi_da = -(2.0 / r2) * (g.conjugate() * df_da).real
    return phi, dphi_dtheta, (dphi_dX, dphi_dY, dphi_drho, dphi_dm0, dphi_da)


def polish_endpoint(R, theta, pf, iters=6):
    """Newton on ``phi(R, theta; P) = 0``.

    Returns ``(theta*, d phi/d theta at theta*, reliable)``.  ``reliable``
    is ``False`` when the angular slope collapses (a tangency) so the
    caller can fail closed.
    """
    a, m0, X, Y, rho = pf
    th = theta
    dth = 1.0
    for _ in range(iters):
        ph, dth, _ = phi_grad(R, th, a, m0, X, Y, rho)
        if abs(dth) < 1e-13:
            return th, dth, False
        step = ph / dth
        th -= step
        if abs(step) < 1e-15:
            break
    ph, dth, _ = phi_grad(R, th, a, m0, X, Y, rho)
    return th, dth, abs(dth) >= 1e-13


# ---- arc enumeration from the real roots of the boundary quartic --------
def _real_root_thetas(pc):
    """Sorted ``theta in [0, 2 pi)`` of the real roots of ``P(t)`` (t = tan(th/2))."""
    roots = np.roots(pc[::-1])                      # np.roots wants descending
    out = []
    for r in roots:
        if abs(r.imag) <= _ROOT_IM_REL * (1.0 + abs(r.real)):
            out.append((2.0 * math.atan(r.real)) % _TWO_PI)
    out.sort()
    # merge a near-double root (near-tangency) into one endpoint
    merged = []
    for x in out:
        if not merged or x - merged[-1] > 1e-11:
            merged.append(x)
    return merged


def arc_intervals(R, pf):
    """``(kind, arcs)`` for the circle ``|z| = R`` from the boundary quartic.

    ``kind`` is one of ``"arcs"`` (``arcs`` = list of ``(theta_enter,
    theta_leave)`` with ``theta_leave`` possibly ``> 2 pi`` for a
    ``theta = pi``-straddling arc), ``"full"``, ``"empty"``, or
    ``"degenerate"`` (leading coefficient ~ 0, or an odd real-root count --
    the caller falls back to :func:`topology.arcs_at`).
    """
    a, m0, X, Y, rho = pf
    pc = boundary_quartic(R, a, m0, X, Y, rho)
    amax = max(abs(c) for c in pc)
    if amax == 0.0 or abs(pc[4]) < _P4_DEGEN_REL * amax:
        return "degenerate", []
    th = _real_root_thetas(pc)
    if not th:
        return ("full" if pc[4] > 0.0 else "empty"), []
    if len(th) % 2 != 0:
        return "degenerate", []
    arcs = []
    n = len(th)
    for i in range(n):
        lo = th[i]
        hi = th[(i + 1) % n]
        if hi <= lo:
            hi += _TWO_PI
        if _phi(R, 0.5 * (lo + hi), a, m0, X, Y, rho) > 0.0:
            arcs.append((lo, hi))
    return "arcs", arcs


def _grid_intervals(R, pf):
    """Fallback arc list from :func:`topology.arcs_at` (grid sign-scan)."""
    a, m0, X, Y, rho = pf
    kind, _n, arcs = arcs_at(R, a, m0, X, Y, rho, n_grid=4096)
    if kind in ("full", "empty"):
        return kind, []
    out = []
    for arc in arcs:
        tl = arc.theta_leave
        if tl <= arc.theta_enter:
            tl += _TWO_PI
        out.append((arc.theta_enter, tl))
    return "arcs", out


# ---- one radius: value + derivative integrands --------------------------
def _full_circle_terms(R, pf):
    """``(f0, fh, df0, dfh, reliable)`` for a full ``phi > 0`` circle.

    ``f0 = R * 2 pi`` (topologically constant -> ``df0 = 0``); ``fh`` and
    ``dfh`` are the smooth periodic trapezoid.  Flagged unreliable: near
    the cell edge ``phi -> 0`` somewhere and the periodic rule degrades.
    """
    a, m0, X, Y, rho = pf
    dth = _TWO_PI / len(_FULL_TH)
    fh = 0.0
    dfh = np.zeros(5)
    for t in _FULL_TH:
        ph, _d, gdp = phi_grad(R, t, a, m0, X, Y, rho)
        if ph <= 0.0:
            continue
        sq = math.sqrt(ph)
        fh += sq
        for j in range(5):
            dfh[j] += gdp[j] / (2.0 * sq)
    return R * _TWO_PI, R * dth * fh, np.zeros(5), R * dth * dfh, False


def radius_terms(R, pf, tan_rel=_TAN_REL):
    """``(f0, fh, df0[5], dfh[5], reliable)`` -- integrands in ``R`` at ``R``.

    ``df0 = R * sum_arcs ( d theta_leave/d P - d theta_enter/d P )`` via
    IFT-theta; ``dfh = R * sum_arcs int_arc  d phi/d P / (2 sqrt phi) dtheta``.
    """
    a, m0, X, Y, rho = pf
    f0 = 0.0
    fh = 0.0
    df0 = np.zeros(5)
    dfh = np.zeros(5)
    reliable = True

    kind, arcs = arc_intervals(R, pf)
    if kind == "empty":
        return f0, fh, df0, dfh, reliable
    if kind == "full":
        return _full_circle_terms(R, pf)
    if kind == "degenerate":
        gkind, arcs = _grid_intervals(R, pf)
        reliable = False
        if gkind == "empty":
            return f0, fh, df0, dfh, reliable
        if gkind == "full":
            return _full_circle_terms(R, pf)

    tan_thresh = tan_rel * rho / max(R, 1e-9)
    for te0, tl0 in arcs:
        te, td, rel_e = polish_endpoint(R, te0, pf)
        tl, tdl, rel_l = polish_endpoint(R, tl0, pf)
        if tl <= te:
            tl += _TWO_PI
        reliable = reliable and rel_e and rel_l
        if abs(td) < tan_thresh or abs(tdl) < tan_thresh:
            reliable = False

        # ---- F0 pieces (IFT-theta endpoint derivatives)
        f0 += R * (tl - te)
        _, _, ge = phi_grad(R, te, a, m0, X, Y, rho)
        _, _, gl = phi_grad(R, tl, a, m0, X, Y, rho)
        for j in range(5):
            dte = -ge[j] / td if td != 0.0 else 0.0
            dtl = -gl[j] / tdl if tdl != 0.0 else 0.0
            df0[j] += R * (dtl - dte)

        # ---- F_half pieces, one Gauss-Chebyshev angular pass
        half = 0.5 * (tl - te)
        mid = 0.5 * (te + tl)
        thn = mid + half * _ANG_X
        acc_val = 0.0
        acc_der = np.zeros(5)
        for k in range(len(thn)):
            ph, _d, gdp = phi_grad(R, thn[k], a, m0, X, Y, rho)
            if ph <= 0.0:
                continue
            sq = math.sqrt(ph)
            wk = _ANG_W[k]
            acc_val += wk * sq
            inv = wk / (2.0 * sq)
            for j in range(5):
                acc_der[j] += inv * gdp[j]
        fh += R * half * acc_val
        for j in range(5):
            dfh[j] += R * half * acc_der[j]

    return f0, fh, df0, dfh, reliable


# ---- endpoint derivative in the t-chart (representation cross-check) ----
def endpoint_dtheta_theta_chart(R, theta_star, pf):
    """``d theta*/d P`` (length-5) via IFT-theta -- the primary form."""
    a, m0, X, Y, rho = pf
    _, dphi_dtheta, gdp = phi_grad(R, theta_star, a, m0, X, Y, rho)
    if dphi_dtheta == 0.0:
        return np.full(5, np.nan)
    return np.array([-gdp[j] / dphi_dtheta for j in range(5)])


def endpoint_dtheta_t_chart(R, theta_star, pf):
    """``d theta*/d P`` (length-5) via IFT-t: differentiate ``P(t*; R) = 0``.

    Diverges as ``theta* -> pi`` (``t* -> inf``); used only to cross-check
    IFT-theta away from that pole.
    """
    a, m0, X, Y, rho = pf
    t = math.tan(theta_star / 2.0)
    pc = boundary_quartic(R, a, m0, X, Y, rho)
    p0, p1, p2, p3, p4 = pc
    P_t = p1 + t * (2.0 * p2 + t * (3.0 * p3 + t * 4.0 * p4))
    if P_t == 0.0:
        return np.full(5, np.nan)
    dps = boundary_quartic_dp(R, a, m0, X, Y, rho)
    jac = 2.0 / (1.0 + t * t)
    out = np.zeros(5)
    for j in range(5):
        d = dps[j]
        dP_j = d[0] + t * (d[1] + t * (d[2] + t * (d[3] + t * d[4])))
        out[j] = jac * (-dP_j / P_t)
    return out


# ---- per-cell radial pass ---------------------------------------------
@dataclass
class FluxJacobian:
    F0: float
    F_half: float
    dF0: np.ndarray                  # d/d(xs, ys, rho, q, a)   -- user params
    dF_half: np.ndarray
    dF0_internal: np.ndarray        # d/d(X, Y, rho, m0, a)     -- primary frame
    dF_half_internal: np.ndarray
    r_max: float
    status: str
    notes: list = field(default_factory=list)


def _internal_to_user_jac(dvec_internal, params: LensParams):
    """Map a primary-frame covector ``d/d(X, Y, rho, m0, a)`` to the user
    covector ``d/d(xs, ys, rho, q, a)``.

    ``X = xs + m1 a`` (barycentric) or ``X = xs`` ; ``Y = ys`` ;
    ``m0 = 1/(1+q)`` -> ``dm0/dq = -1/(1+q)^2`` , ``dm1/dq = 1/(1+q)^2`` ;
    the primary-frame ``a`` and the user ``a`` differ (barycentric) by the
    ``X`` shift ``m1 a`` -> ``dX/da = m1 = q/(1+q)``.
    """
    q = params.q
    a = params.a
    dm0_dq = -1.0 / (1.0 + q) ** 2
    dm1_dq = 1.0 / (1.0 + q) ** 2
    dX, dY, drho, dm0, da_pf = dvec_internal
    out = np.zeros(5)
    out[0] = dX
    out[1] = dY
    out[2] = drho
    if params.barycentric:
        out[3] = dm0 * dm0_dq + dX * (a * dm1_dq)
        out[4] = da_pf + dX * (q / (1.0 + q))
    else:
        out[3] = dm0 * dm0_dq
        out[4] = da_pf
    return out


def flux_jacobian(params: LensParams, *, n_r: int = 64) -> FluxJacobian:
    """``F0``, ``F_1/2`` and both 5-covectors for one source position.

    Same per-cell fixed Gauss-Chebyshev reconstruction as the value; the
    returned ``F0 / F_half`` are that reconstruction's (so value and
    Jacobian are mutually consistent -- plan sec. 11), not
    :func:`flux.epoch_flux`'s adaptive quad (which agrees to ~1e-4).
    """
    X, Y = params.primary_frame_source()
    pf = (params.a, params.m0, X, Y, params.rho)
    topo = classify_cells(params)

    xr, wr = _cheb1(n_r)
    F0 = 0.0
    F_half = 0.0
    dF0i = np.zeros(5)
    dFhi = np.zeros(5)
    status = "OK" if topo.status == "OK" else "GRADIENT_UNRELIABLE"
    notes: list = []
    if topo.status != "OK":
        notes.append(f"topology status {topo.status!r} -> gradient unreliable")

    for c in topo.cells:
        w = c.r_hi - c.r_lo
        if w <= 0.0 or c.kind == "empty":
            continue
        if c.kind == "full":
            status = "GRADIENT_UNRELIABLE"
            notes.append(f"cell {c.index}: full-circle radius -- periodic "
                         f"trapezoid gradient, flagged")
        ins = 1e-9 * w
        lo = c.r_lo + ins
        hi = c.r_hi - ins
        rmid = 0.5 * (lo + hi)
        rhalf = 0.5 * (hi - lo)
        for k in range(n_r):
            R = rmid + rhalf * xr[k]
            f0, fh, d0, dh, rel = radius_terms(R, pf)
            if not rel:
                status = "GRADIENT_UNRELIABLE"
            Wk = rhalf * wr[k]
            F0 += Wk * f0
            F_half += Wk * fh
            dF0i += Wk * d0
            dFhi += Wk * dh

    if status == "GRADIENT_UNRELIABLE" and "gradient unreliable" not in " ".join(notes):
        notes.append("a radial node hit a near-tangency / degenerate arc")

    dF0u = _internal_to_user_jac(tuple(dF0i), params)
    dFhu = _internal_to_user_jac(tuple(dFhi), params)
    return FluxJacobian(F0=F0, F_half=F_half, dF0=dF0u, dF_half=dFhu,
                        dF0_internal=dF0i, dF_half_internal=dFhi,
                        r_max=topo.r_max, status=status, notes=notes)


# ---- magnification Jacobian -----------------------------------------
@dataclass
class EpochJacobian:
    params: LensParams
    u: float
    mu: float
    grad_mu: np.ndarray             # d mu / d(xs, ys, rho, q, a)
    dmu_du: float                   # d mu / d u
    F0: float
    F_half: float
    r_max: float
    status: str
    notes: list = field(default_factory=list)

    @property
    def mu_uniform(self) -> float:
        return self.F0 / (math.pi * self.params.rho ** 2)


def epoch_jacobian(params: LensParams, u: float = 0.0, *,
                   n_r: int = 64) -> EpochJacobian:
    """``mu`` and its 5-component gradient for one source position (plan sec. 11).

    ``mu(u) = ((1-u) F0 + u F_1/2) / (pi rho^2 (1 - u/3))``.  For a user
    parameter ``p_j != rho`` the gradient is
    ``((1-u) dF0/dp_j + u dF_1/2/dp_j) / D`` with ``D = pi rho^2 (1-u/3)``;
    ``rho`` also enters ``D`` -> an extra ``-2 mu / rho``.
    ``d mu / d u = pi rho^2 (F_1/2 - 2 F0 / 3) / D^2`` -- the numerator is
    constant in ``u`` so ``mu`` is monotone in ``u`` (sign geometry-set).
    """
    from .singular import near_origin_source

    fj = flux_jacobian(params, n_r=n_r)
    status = fj.status
    notes = list(fj.notes)
    if near_origin_source(params):
        status = "GRADIENT_UNRELIABLE"
        notes.append("on-axis-origin singular patch -- gradient not "
                     "supported by this path")

    rho = params.rho
    r2 = rho * rho
    D = math.pi * r2 * (1.0 - u / 3.0)
    F0, Fh = fj.F0, fj.F_half
    mu = ((1.0 - u) * F0 + u * Fh) / D
    grad = ((1.0 - u) * fj.dF0 + u * fj.dF_half) / D
    grad = np.array(grad, dtype=float)
    grad[2] -= 2.0 * mu / rho                         # rho enters D
    dmu_du = math.pi * r2 * (Fh - 2.0 * F0 / 3.0) / (D * D)

    return EpochJacobian(params=params, u=u, mu=mu, grad_mu=grad,
                         dmu_du=dmu_du, F0=F0, F_half=Fh, r_max=fj.r_max,
                         status=status, notes=notes)

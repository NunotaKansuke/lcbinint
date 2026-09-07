"""M5: singular patches (plan sec. 9, 12).

Most of the radial line is *normal*: the boundary quartic has four simple
roots, the Gauss-Manin connection is regular, and :mod:`flux` re-anchors the
period at every quadrature node.  A handful of loci need dedicated care:

* **source at the origin, on axis** (``xs = ys = 0``).  ``Q(t; R)`` then
  carries a structural degree-2 repeated factor at *every* ``R`` (M3 sec. 4)
  -- the connection is genuinely undefined, not merely stiff.  We fail
  closed on the holonomic path and fall back to a connection-free direct
  angular integral of the flux, reported as ``LOCAL_REFERENCE_USED``.  This
  also covers the near-origin neighbourhood ``|zeta| < SINGULAR_ZETA_TOL``
  where the connection is arbitrarily ill-conditioned.

* **representation events** ``R = a`` and ``R = sqrt(m0)`` (plan sec. 3, 9).
  Empirically (``checkpoint_M5.md`` sec. 3) the QAWSE re-anchor already
  crosses these cleanly -- the ``t``-chart degree drop at ``R = a`` and the
  ``Res(P, A) = 0`` touch at ``R = sqrt(m0)`` are integrable -- so this
  module only *recognises* them (:func:`representation_events`) for the
  status record; no bridging is needed for the value.

* **birth / death tangencies** (``v -> 0``).  :func:`tangency_flux_series`
  exposes the plan sec. 9 closed-contour period limit so the transport
  stage (and M6 gradients) can start an arc at its tangency without the
  deflated quadrature going singular.  The M4/M5 *flux* does not need it --
  ``scipy`` QAWSE integrates the ``sqrt`` birth profile directly -- but it
  is provided and checked.

Nothing here deletes or bypasses a normal-cell path; :mod:`solver` decides
per epoch whether a patch applies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .polynomial_family import LensParams, boundary_quartic
from .seed import tangency_seed_eta
from .topology import _phi, arcs_at, classify_cells

_TWO_PI = 2.0 * math.pi

# |zeta| below this -> treat the epoch as the on-axis-origin singular patch.
# Set from checkpoint_M4 sec. 5: connection_matrix_numeric is already ~2e-2
# wrong at |zeta| ~ 1e-6 and the exact connection's psi-closure degrades as
# the b_j blow up; 1e-3 keeps a safe margin below the normal regime.
SINGULAR_ZETA_TOL = 1.0e-3


@dataclass
class PatchResult:
    F0: float
    F_half: float
    status: str
    note: str


# ----------------------------------------------------------------- predicates
def near_origin_source(params: LensParams, tol: float = SINGULAR_ZETA_TOL) -> bool:
    """True when the source centre sits on the ``xs = ys = 0`` singular patch.

    Uses the *primary-frame* source position (lens 0 at the origin), which is
    the frame in which the degeneracy is ``zeta = 0`` exactly.
    """
    xs, ys = params.primary_frame_source()
    return math.hypot(xs, ys) < tol


def _r_max(params: LensParams) -> float:
    """Closed-form image-radius bound (plan sec. 3) -- no event solve."""
    xs, ys = params.primary_frame_source()
    W = math.hypot(xs, ys) + params.rho
    return (params.a + W + math.sqrt((params.a - W) ** 2 + 4.0)) / 2.0


def representation_events(params: LensParams):
    """``[(R, kind), ...]`` for ``R = a`` and ``R = sqrt(m0)`` inside ``(0, R_max)``.

    Informational -- these radii get a status note but need no bridging for
    the flux value (``checkpoint_M5.md`` sec. 3).  Uses the closed-form
    ``R_max`` bound so it costs nothing (no sympy event solve).
    """
    r_max = _r_max(params)
    out = []
    for R, kind in ((params.a, "R=a"), (math.sqrt(params.m0), "R=sqrt(m0)")):
        if 0.0 < R < r_max:
            out.append((float(R), kind))
    return out


# --------------------------------------------------- on-axis-origin flux patch
def _circle_flux_integrands(R, a, m0, xs, ys, rho):
    """``(R * sum_arcs dtheta, R * sum_arcs int sqrt(phi) dtheta)`` at ``R``.

    Connection-free: the arcs come straight from the lens equation and each
    ``int sqrt(phi) dtheta`` is a QAWSE integral against the ``(1-u^2)^{-1/2}``
    endpoint weight (``sqrt(phi)`` vanishes linearly at each arc end).
    """
    kind, _n, arcs = arcs_at(R, a, m0, xs, ys, rho)
    if kind == "empty":
        return 0.0, 0.0
    if kind == "full":
        s = quad(lambda th: math.sqrt(max(_phi(R, th, a, m0, xs, ys, rho), 0.0)),
                 0.0, _TWO_PI, limit=200)[0]
        return R * _TWO_PI, R * s
    m_tot = s_tot = 0.0
    for arc in arcs:
        lo, hi = arc.theta_enter, arc.theta_leave
        if hi <= lo:
            hi += _TWO_PI
        m_tot += hi - lo
        half = 0.5 * (hi - lo)
        mid = lo + half

        def g(u, mid=mid, half=half):
            v = _phi(R, mid + half * u, a, m0, xs, ys, rho)
            w = 1.0 - u * u
            return math.sqrt(v / w) if (v > 0.0 and w > 0.0) else 0.0

        s_tot += half * quad(g, -1.0, 1.0, weight="alg", wvar=(0.5, 0.5),
                             limit=200)[0]
    return R * m_tot, R * s_tot


def on_axis_origin_flux(params: LensParams, *, topo=None,
                        quad_limit: int = 200) -> PatchResult:
    """Direct (connection-free) ``(F0, F_1/2)`` for the ``|zeta| ~ 0`` patch.

    The radial cells come from :func:`topology.classify_cells`; each
    image-bearing cell is integrated on its own with an adaptive
    ``scipy.integrate.quad`` (the ``empty`` cells contribute nothing and are
    skipped -- integrating them globally makes QUADPACK hunt the thin bands
    for tens of thousands of evaluations).  This is the same quadrature
    :mod:`direct_quadrature` uses as the independent oracle, so on this
    measure-zero locus the "reference" and the "solver" coincide by
    construction -- hence the ``LOCAL_REFERENCE_USED`` status.

    ``topo`` may be a pre-computed :func:`classify_cells` result (the
    ``classify_cells`` sympy event solve dominates the cost here, and
    :mod:`solver` already needs it for ``r_max``); it is recomputed only when
    not supplied.
    """
    xs, ys = params.primary_frame_source()
    a, m0, rho = params.a, params.m0, params.rho
    if topo is None:
        topo = classify_cells(params)

    F0 = Fh = 0.0
    for c in topo.cells:
        w = c.r_hi - c.r_lo
        if w <= 0.0 or c.kind == "empty":
            continue
        lo = c.r_lo + 1e-9 * w
        hi = c.r_hi - 1e-9 * w
        F0 += quad(lambda R: _circle_flux_integrands(R, a, m0, xs, ys, rho)[0],
                   lo, hi, limit=quad_limit, epsabs=1e-12, epsrel=1e-10)[0]
        Fh += quad(lambda R: _circle_flux_integrands(R, a, m0, xs, ys, rho)[1],
                   lo, hi, limit=quad_limit, epsabs=1e-12, epsrel=1e-10)[0]

    exact = math.hypot(xs, ys) == 0.0
    note = ("source exactly on the axis at the origin -- Q has a structural "
            "repeated t-root at every R (plan sec. 9 / checkpoint_M3 sec. 4); "
            "holonomic connection undefined, direct angular integral used"
            if exact else
            f"|zeta| = {math.hypot(xs, ys):.2e} < {SINGULAR_ZETA_TOL:g} -- "
            "connection ill-conditioned near the on-axis-origin locus; "
            "direct angular integral used")
    return PatchResult(F0=F0, F_half=Fh, status="LOCAL_REFERENCE_USED", note=note)


# -------------------------------------------------------- tangency (birth/death)
def tangency_flux_series(params: LensParams, R_star: float, m_star: float):
    """Closed-contour period vector at a simple boundary tangency (``v -> 0``).

    Wraps :func:`seed.tangency_seed_eta` with the plan sec. 9 normalisation

        Pi_k^{(closed)}  ->  2 pi m_*^k / sqrt(C(m_*, R_*)) ,
        C(m_*, R_*) = (1/2) P''(m_*) A(m_*) B(m_*) .

    Returned so the transport / gradient stages can anchor an arc at its
    birth without the deflated QAWSE seed degenerating.  Raises (fail
    closed) if ``m_star`` is not a near-double root of ``P(.; R_star)`` or
    the leading coefficient is non-positive.
    """
    xs, ys = params.primary_frame_source()
    pc = boundary_quartic(R_star, params.a, params.m0, xs, ys, params.rho)
    p = float(np.polyval(pc[::-1], m_star))
    dp = float(np.polyval(np.polyder(np.asarray(pc[::-1], dtype=float)), m_star))
    scale = max(abs(c) for c in pc) + 1.0
    if abs(p) > 1e-6 * scale or abs(dp) > 1e-4 * scale:
        raise ValueError(
            f"tangency_flux_series: m_star={m_star!r} is not a double root of "
            f"P(.;{R_star!r})  (P={p:.3e}, P'={dp:.3e}, scale={scale:.3e})")
    return tangency_seed_eta(m_star, R_star, params.a, params.m0, xs, ys,
                             params.rho)

"""M4: period seeds for the transport (plan sec. 8-9).

Two ways to anchor the period vector at a radius ``R``:

* :func:`seed_eta` / :func:`seed_psi` -- *robust* seed from the deflated
  QAWSE quadrature of :func:`period_reduction.half_period_eta`, with the arc
  endpoints re-found at ``R`` (``arcs_at(R)``) and tracked by nearest
  midpoint.  Validated against the bare angular integral to ~3e-12 in M3.
  The closed-contour period is twice the real-segment period, so
  ``Pi_psi = 2 * W(R) @ I(R)``.

* :func:`tangency_seed_eta` -- the ``v -> 0`` series of plan sec. 9,

      Pi_k^{(closed)}  ->  2 pi m_*^k / sqrt(C(m_*, R_*)) ,
      C(m_*, R_*) = (1/2) P''(m_*) A(m_*) B(m_*)  (leading coeff of Q at m_*),

  used to start an arc at its birth tangency where the quadrature seed
  degenerates.  (Plan allows deferring the birth handling to M5; this is
  provided and lightly exercised.)
"""
from __future__ import annotations

import math

import numpy as np

from .connection import q_coeffs_numeric
from .period_reduction import (
    arc_chart,
    half_period_eta,
    psi_reduction_matrix,
)
from .polynomial_family import boundary_quartic
from .topology import arcs_at

_TWO_PI = 2.0 * math.pi


def _track_arc(R, a, m0, xs, ys, rho, mid0):
    """The arc at radius ``R`` whose midpoint is closest to ``mid0``."""
    kind, _ncr, arcs = arcs_at(R, a, m0, xs, ys, rho)
    if not arcs:
        raise ValueError(f"no arc at R={R!r} (kind={kind})")
    return min(arcs, key=lambda A: abs(
        ((A.midpoint - mid0 + math.pi) % _TWO_PI) - math.pi))


def seed_eta(R, a, m0, xs, ys, rho, ref_arc):
    """``([I_0 .. I_6], arc)`` -- real-segment monomial periods at ``R``.

    ``ref_arc`` only supplies the midpoint lineage; the endpoints are
    re-solved at ``R``.
    """
    arc = _track_arc(R, a, m0, xs, ys, rho, ref_arc.midpoint)
    return half_period_eta(R, a, m0, xs, ys, rho, arc), arc


def seed_psi(R, a, m0, xs, ys, rho, ref_arc):
    """``(Pi_psi, arc)`` -- the closed-contour residue-free period 6-vector
    ``Pi_psi = 2 W(R) @ I(R)`` at ``R`` (chart chosen by :func:`arc_chart`)."""
    arc = _track_arc(R, a, m0, xs, ys, rho, ref_arc.midpoint)
    params, _tt, _s = arc_chart(a, m0, xs, ys, rho, arc)
    I = half_period_eta(R, a, m0, xs, ys, rho, arc)
    W = psi_reduction_matrix(q_coeffs_numeric(R, *params))
    return 2.0 * (W @ I), arc


def tangency_seed_eta(m_star, R_star, a, m0, xs, ys, rho, kmax=6):
    """Closed-contour period vector at a simple tangency (``v -> 0``)."""
    pc = boundary_quartic(R_star, a, m0, xs, ys, rho)
    p2 = 2 * pc[2] + m_star * (6 * pc[3] + m_star * 12 * pc[4])   # P''(m_*)
    A = 1.0 + m_star * m_star
    B = (R_star - a) ** 2 + (R_star + a) ** 2 * m_star * m_star
    C = 0.5 * p2 * A * B
    if C <= 0.0:
        raise ValueError("tangency seed: leading Q coefficient non-positive")
    pref = _TWO_PI / math.sqrt(C)
    return np.array([pref * m_star**k for k in range(kmax + 1)])

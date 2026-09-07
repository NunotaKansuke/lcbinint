"""M4: the boundary root pair ``(m, v)`` of one image arc (plan sec. 8, 11).

An image arc on the circle ``|z| = R`` is bounded by two real roots
``t_- < t_+`` of the boundary quartic ``P(t; R)``.  Instead of the roots
themselves the transport carries the *symmetric* pair

    m = (t_+ + t_-) / 2 ,      v = ((t_+ - t_-) / 2)^2 ,

so that a tangency (``t_+ -> t_-``) is the smooth boundary ``v -> 0`` rather
than a collision of two coordinates.  For a quartic

    E(m, v) := (P(m + s) + P(m - s)) / 2 = P(m) + (v/2) P''(m) + (v^2/24) P''''(m)
    O(m, v) := (P(m + s) - P(m - s)) / (2 s) = P'(m) + (v/6) P'''(m)

with ``s = sqrt(v)`` and *no* truncation (higher derivatives of a quartic
vanish), so ``E = O = 0`` iff ``P(t_-) = P(t_+) = 0``.  The Jacobian at a
genuine tangency (``P'(m) = 0``, ``v = 0``) is

    det d(E, O)/d(m, v) |_{v=0} = -1/2 P''(m)^2 .

The arc's angular width is ``Delta theta = 2 atan2(2 sqrt(v), 1 + m^2 - v)``
(from ``theta = 2 arctan t``).

Radial motion of the pair is the 2x2 implicit-function solve

    [[E_m, E_v], [O_m, O_v]] [dm/dR, dv/dR]^T = -[E_R, O_R]^T ,

and of a single endpoint ``dt/dR = -P_R(t) / P_t(t)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .polynomial_family import boundary_quartic, boundary_quartic_dR


@dataclass(frozen=True)
class RootPair:
    """Symmetric coordinates of a boundary root pair: ``m`` = midpoint,
    ``v`` = squared half-gap (``>= 0``; ``v = 0`` is a tangency)."""
    m: float
    v: float

    @property
    def t_minus(self) -> float:
        return self.m - math.sqrt(max(self.v, 0.0))

    @property
    def t_plus(self) -> float:
        return self.m + math.sqrt(max(self.v, 0.0))

    @property
    def delta_theta(self) -> float:
        """Angular width of the arc this pair bounds (``theta = 2 arctan t``)."""
        s = math.sqrt(max(self.v, 0.0))
        return math.atan2(2.0 * s, 1.0 + self.m * self.m - self.v) * 2.0


def from_endpoints(t_lo: float, t_hi: float) -> RootPair:
    m = 0.5 * (t_lo + t_hi)
    half = 0.5 * (t_hi - t_lo)
    return RootPair(m, half * half)


# --------------------------------------------------------- P and derivatives
def _p_derivs(pc, t):
    """(P, P', P'', P''', P'''') of the quartic ``pc`` (ascending) at ``t``."""
    p0, p1, p2, p3, p4 = pc
    P = p0 + t * (p1 + t * (p2 + t * (p3 + t * p4)))
    P1 = p1 + t * (2 * p2 + t * (3 * p3 + t * 4 * p4))
    P2 = 2 * p2 + t * (6 * p3 + t * 12 * p4)
    P3 = 6 * p3 + t * 24 * p4
    P4 = 24 * p4
    return P, P1, P2, P3, P4


def eo_residuals(rp: RootPair, pc):
    """``(E, O)`` -- both zero iff ``t_-`` and ``t_+`` are roots of ``P``."""
    m, v = rp.m, rp.v
    P, P1, P2, P3, P4 = _p_derivs(pc, m)
    E = P + 0.5 * v * P2 + (v * v / 24.0) * P4
    O = P1 + (v / 6.0) * P3
    return E, O


def eo_jacobian(rp: RootPair, pc):
    """``[[E_m, E_v], [O_m, O_v]]`` at ``(m, v)`` (exact for a quartic)."""
    m, v = rp.m, rp.v
    p4 = pc[4]
    _P, P1, P2, P3, _P4 = _p_derivs(pc, m)
    E_m = P1 + 0.5 * v * P3          # d/dm [P + v/2 P'' + v^2 p4]
    E_v = 0.5 * P2 + 2.0 * v * p4
    O_m = P2 + 4.0 * v * p4          # d/dm [P' + v/6 P''']
    O_v = P3 / 6.0
    return ((E_m, E_v), (O_m, O_v))


def tangency_determinant(m: float, pc) -> float:
    """The boxed identity value ``-1/2 P''(m)^2`` (Jacobian det at ``v = 0``)."""
    _P, _P1, P2, _P3, _P4 = _p_derivs(pc, m)
    return -0.5 * P2 * P2


# ----------------------------------------------------------- radial motion
def endpoint_dR(t: float, R, a, m0, xs, ys, rho) -> float:
    """``dt/dR = -P_R(t) / P_t(t)`` for one boundary root ``t`` at radius ``R``."""
    pc = boundary_quartic(R, a, m0, xs, ys, rho)
    pcR = boundary_quartic_dR(R, a, m0, xs, ys, rho)
    _P, P_t, _2, _3, _4 = _p_derivs(pc, t)
    P_R = pcR[0] + t * (pcR[1] + t * (pcR[2] + t * (pcR[3] + t * pcR[4])))
    return -P_R / P_t


def root_pair_dR(rp: RootPair, R, a, m0, xs, ys, rho):
    """``(dm/dR, dv/dR)`` by the 2x2 implicit-function solve.

    Raises ``ValueError`` if the pair Jacobian is singular (tangency /
    degenerate arc) -- fail closed, the caller should switch to the
    tangency series (plan sec. 9).
    """
    m, v = rp.m, rp.v
    pc = boundary_quartic(R, a, m0, xs, ys, rho)
    pcR = boundary_quartic_dR(R, a, m0, xs, ys, rho)
    (E_m, E_v), (O_m, O_v) = eo_jacobian(rp, pc)
    # E_R = P_R(m) + v/2 P''_R(m) + v^2 p4_R ;  O_R = P'_R(m) + v/6 P'''_R(m)
    PR0 = pcR[0] + m * (pcR[1] + m * (pcR[2] + m * (pcR[3] + m * pcR[4])))
    PR1 = pcR[1] + m * (2 * pcR[2] + m * (3 * pcR[3] + m * 4 * pcR[4]))
    PR2 = 2 * pcR[2] + m * (6 * pcR[3] + m * 12 * pcR[4])
    PR3 = 6 * pcR[3] + m * 24 * pcR[4]
    E_R = PR0 + 0.5 * v * PR2 + v * v * pcR[4]
    O_R = PR1 + (v / 6.0) * PR3
    det = E_m * O_v - E_v * O_m
    if abs(det) < 1e-300 or not math.isfinite(det):
        raise ValueError("root-pair Jacobian singular (tangency)")
    dm = -(O_v * E_R - E_v * O_R) / det
    dv = -(-O_m * E_R + E_m * O_R) / det
    return dm, dv

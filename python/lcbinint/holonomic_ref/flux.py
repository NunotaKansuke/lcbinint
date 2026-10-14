"""M4: epoch flux assembly and magnification (plan sec. 1, 5-8).

Two radial flux integrals feed the linear limb-darkening magnification:

    F0     = integral_0^Rmax  R * ( sum_arcs  Delta theta_arc(R) )  dR
           = area of the image region  =  mu_uniform * pi rho^2 ,

    F_1/2  = integral_0^Rmax  ( sum_arcs  (2/rho) Phi_arc(R) )  dR ,
             Phi_arc(R) = (rho R / 2) integral_arc sqrt(phi) dtheta ,

    mu_u   = ( (1-u) F0 + u F_1/2 ) / ( pi rho^2 (1 - u/3) )        (plan sec. 1).

``Phi_arc`` is evaluated through the holonomic reduction
(:func:`period_reduction.phi_arc_reduced_numeric`, i.e. ``c^T (W @ I)`` in
the residue-free basis).  The arc topology at each radial node comes from
:func:`topology.arcs_at` straight from the lens equation, and *every*
returned arc is summed -- no lineage matching is needed for the total.

Per the M4 gate the per-cell Gauss-Manin condition numbers are recorded
(:func:`transport.cell_conditioning`); full-cell transport in the raw basis
is ill-conditioned (plan sec. 15) so the reference re-anchors the period at
every node rather than transporting a single seed.  ``full`` cells fall back
to a direct angular integral (a genuine singular-patch case, deferred to M5
for the algebraic treatment); ``empty`` cells contribute nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad

from .period_reduction import phi_arc_reduced_numeric
from .polynomial_family import LensParams
from .topology import _phi, arcs_at, classify_cells
from .transport import cell_conditioning

_TWO_PI = 2.0 * math.pi
_H3_TOL = 1e-6           # relative second-kind residual before fail-closed


@dataclass
class CellFlux:
    index: int
    r_lo: float
    r_hi: float
    kind: str
    n_crossings: int
    dF0: float
    dF_half: float
    status: str = "OK"
    cond: dict = field(default_factory=dict)
    detail: str = ""


@dataclass
class FluxResult:
    params: LensParams
    r_max: float
    F0: float
    F_half: float
    status: str
    cells: list
    notes: list = field(default_factory=list)

    @property
    def mu_uniform(self) -> float:
        return self.F0 / (math.pi * self.params.rho ** 2)

    def mu_linear_ld(self, u: float) -> float:
        rho2 = self.params.rho ** 2
        return (((1.0 - u) * self.F0 + u * self.F_half)
                / (math.pi * rho2 * (1.0 - u / 3.0)))


# ---------------------------------------------------------------- integrands
def _arc_measure_sum(R, a, m0, xs, ys, rho):
    kind, _n, arcs = arcs_at(R, a, m0, xs, ys, rho)
    if kind == "full":
        return _TWO_PI
    return sum(arc.measure for arc in arcs)


def _full_circle_sqrt_phi(R, a, m0, xs, ys, rho):
    def g(theta):
        v = _phi(R, theta, a, m0, xs, ys, rho)
        return math.sqrt(v) if v > 0.0 else 0.0
    val, _e = quad(g, 0.0, _TWO_PI, limit=200, epsabs=1e-12, epsrel=1e-12)
    return val


def _reduced_half_sum(R, a, m0, xs, ys, rho):
    """sum_arcs (2/rho) Phi_arc(R) through the holonomic reduction."""
    kind, _n, arcs = arcs_at(R, a, m0, xs, ys, rho)
    if kind == "full":
        return R * _full_circle_sqrt_phi(R, a, m0, xs, ys, rho), "FULL"
    tot, worst = 0.0, 0.0
    for arc in arcs:
        val, h3 = phi_arc_reduced_numeric(R, a, m0, xs, ys, rho, arc)
        scale = abs(val) * rho + 1.0
        worst = max(worst, abs(h3) / scale)
        tot += (2.0 / rho) * val
    status = "OK" if worst < _H3_TOL else "BASIS_DEGENERATE"
    return tot, status


# ------------------------------------------------------------------- driver
def epoch_flux(params: LensParams, *, record_conditioning=False,
               quad_limit=200) -> FluxResult:
    """Assemble ``F0`` and ``F_1/2`` for one source position (plan sec. 5-8).

    ``record_conditioning`` attaches the exact per-cell Gauss-Manin condition
    numbers (:func:`transport.cell_conditioning`, the M4 gate record).  It is
    off by default because the flux itself never transports a seed -- it
    re-anchors the period at every quadrature node -- and the exact
    conditioning sweep is far more expensive than the flux.
    """
    xs, ys = params.primary_frame_source()
    a, m0, rho = params.a, params.m0, params.rho
    topo = classify_cells(params)
    F0 = F_half = 0.0
    cells: list = []
    notes: list = []
    status = "OK" if topo.status == "OK" else topo.status

    for c in topo.cells:
        w = c.r_hi - c.r_lo
        if w <= 0.0:
            continue
        lo = c.r_lo + 1e-7 * w
        hi = c.r_hi - 1e-7 * w
        cell_status = "OK" if c.status == "OK" else c.status
        detail = c.detail
        cond = {}

        if c.kind == "empty":
            cells.append(CellFlux(c.index, c.r_lo, c.r_hi, c.kind,
                                  c.n_crossings, 0.0, 0.0, cell_status, cond,
                                  detail))
            continue

        d0, _e0 = quad(lambda R: R * _arc_measure_sum(R, a, m0, xs, ys, rho),
                       lo, hi, limit=quad_limit, epsabs=1e-13, epsrel=1e-10)

        seen = {"s": "OK"}

        def half_integrand(R):
            v, st = _reduced_half_sum(R, a, m0, xs, ys, rho)
            if st not in ("OK", "FULL"):
                seen["s"] = st
            return v

        dh, _eh = quad(half_integrand, lo, hi, limit=quad_limit,
                       epsabs=1e-13, epsrel=1e-10)

        if c.kind == "full":
            cell_status = _worst(cell_status, "LOCAL_REFERENCE_USED")
            notes.append(f"cell {c.index}: full-circle radius -- direct "
                         f"angular integral (M5 singular patch)")
        elif seen["s"] != "OK":
            cell_status = _worst(cell_status, seen["s"])

        if record_conditioning and c.kind == "arcs":
            try:
                arc0 = c.arcs[0]
                from .period_reduction import arc_chart
                cparams, _tt, _s = arc_chart(a, m0, xs, ys, rho, arc0)
                cond = cell_conditioning(lo, hi, cparams, exact=True)
                # the flux re-anchors the period at every node -- a stiff
                # connection does NOT corrupt it -- so this only annotates.
                if cond["cond_psi_max"] > 1e9:
                    cell_status = _worst(cell_status, "OK_ESTIMATED")
                    notes.append(
                        f"cell {c.index}: cond(C_psi)~{cond['cond_psi_max']:.1e}"
                        f" -- single-seed transport ill-conditioned here "
                        f"(plan sec. 15); flux re-anchored per node")
            except Exception as exc:                       # pragma: no cover
                cond = {"error": repr(exc)}

        F0 += d0
        F_half += dh
        status = _worst(status, cell_status)
        cells.append(CellFlux(c.index, c.r_lo, c.r_hi, c.kind, c.n_crossings,
                              d0, dh, cell_status, cond, detail))

    return FluxResult(params=params, r_max=topo.r_max, F0=F0, F_half=F_half,
                      status=status, cells=cells, notes=notes)


_SEVERITY = {
    "OK": 0, "OK_ESTIMATED": 1, "LOCAL_REFERENCE_USED": 2,
    "TOPOLOGY_UNCERTAIN": 3, "CONNECTION_ILL_CONDITIONED": 4,
    "BASIS_DEGENERATE": 5, "ARC_TOPOLOGY_INVALID": 6,
}


def _worst(a, b):
    return a if _SEVERITY.get(a, 9) >= _SEVERITY.get(b, 9) else b


def mu_uniform(params: LensParams) -> float:
    return epoch_flux(params).mu_uniform


def mu_linear_ld(params: LensParams, u: float) -> float:
    return epoch_flux(params).mu_linear_ld(u)

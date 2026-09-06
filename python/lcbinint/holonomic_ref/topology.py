"""M2: radial-cell topology classification (plan sec. 4).

The radial events of :mod:`radial_events` cut ``(0, R_max)`` into open
*cells*.  Inside one cell the image of the source circle -- the set
``{theta : phi(R, theta) >= 0}`` on ``|z| = R`` -- has a fixed combinatorial
type:

  * ``empty``  -- ``phi < 0`` on the whole circle (no image at this radius);
  * ``full``   -- ``phi >= 0`` on the whole circle (the radius lies entirely
                  inside one image region -- a full-circle state, plan sec. 9);
  * ``arcs``   -- a positive even number (2 or 4) of boundary crossings, i.e.
                  1 or 2 image arcs.  ``deg_t P = 4`` bounds this at 2 arcs.

For each cell we record the arcs at a representative radius, sampling a few
radii to catch a mis-placed event (``status = TOPOLOGY_UNCERTAIN``), and we
thread an *incidence* lineage that says how each arc is born, continues, or
dies across the bounding events -- the structure the period-transport stage
(M4) walks.

Reference quality: arcs are found by dense sign scanning of ``phi`` straight
from the lens equation plus Brent refinement.  No algebraic short-cuts; the
production backend will instead transport arc endpoints (plan sec. 8, 11).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .polynomial_family import LensParams
from .radial_events import radial_events

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
def _phi(R, theta, a, m0, xs, ys, rho):
    z = R * complex(math.cos(theta), math.sin(theta))
    zb = z.conjugate()
    fz = z - m0 / zb - (1.0 - m0) / (zb - a)
    return 1.0 - abs(fz - complex(xs, ys)) ** 2 / (rho * rho)


def _phi_grid(R, thetas, a, m0, xs, ys, rho):
    z = R * np.exp(1j * thetas)
    zb = np.conj(z)
    fz = z - m0 / zb - (1.0 - m0) / (zb - a)
    return 1.0 - np.abs(fz - complex(xs, ys)) ** 2 / (rho * rho)


@dataclass(frozen=True)
class Arc:
    """A maximal ``phi >= 0`` interval on the circle, oriented so that the
    region is entered at ``theta_enter`` and left at ``theta_leave`` when
    theta increases (mod 2 pi)."""
    theta_enter: float
    theta_leave: float

    @property
    def measure(self) -> float:
        return (self.theta_leave - self.theta_enter) % _TWO_PI

    @property
    def midpoint(self) -> float:
        return (self.theta_enter + 0.5 * self.measure) % _TWO_PI


@dataclass
class CellPlan:
    """Topology of one radial cell.  Transport / flux fields are added at M4."""
    index: int
    r_lo: float
    r_hi: float
    r_mid: float
    kind: str                       # "empty" | "full" | "arcs"
    n_crossings: int                # 0, 2, 4
    arcs: tuple[Arc, ...]
    lo_event: str                   # event kind at r_lo ("origin" for the first)
    hi_event: str                   # event kind at r_hi ("R_max" for the last)
    status: str = "OK"              # "OK" | "TOPOLOGY_UNCERTAIN"
    detail: str = ""


@dataclass
class ArcTransition:
    event_radius: float
    event_kind: str
    kind: str                       # "continue" | "birth" | "death"
    lower_cell: int
    lower_arc: "int | None"
    upper_cell: int
    upper_arc: "int | None"


@dataclass
class TopologyResult:
    params: LensParams
    r_max: float
    cells: list
    transitions: list = field(default_factory=list)
    status: str = "OK"              # worst cell status


# --------------------------------------------------------------------------
def arcs_at(R, a, m0, xs, ys, rho, *, n_grid=3072):
    """``(kind, n_crossings, arcs)`` for the circle ``|z| = R``."""
    th = np.linspace(0.0, _TWO_PI, n_grid, endpoint=False)
    val = _phi_grid(R, th, a, m0, xs, ys, rho)

    pos = val >= 0.0
    if pos.all():
        return "full", 0, ()
    if (~pos).all():
        return "empty", 0, ()

    step = _TWO_PI / n_grid
    nxt = np.roll(pos, -1)
    idx = np.nonzero(pos != nxt)[0]

    def f(t):
        return _phi(R, t, a, m0, xs, ys, rho)

    roots, rising = [], []
    for i in idx:
        t0, t1 = th[i], th[i] + step
        r = brentq(f, t0, t1, xtol=1e-14, rtol=8.9e-16, maxiter=200)
        roots.append(r % _TWO_PI)
        rising.append(not pos[i])          # False->True crossing == entering
    order = np.argsort(roots)
    roots = [roots[k] for k in order]
    rising = [rising[k] for k in order]

    n = len(roots)
    if n % 2 != 0:
        return "arcs", n, ()               # odd count -> report, don't guess

    start = 0 if rising[0] else -1
    arcs = []
    for k in range(0, n, 2):
        te = roots[(start + k) % n]
        tl = roots[(start + k + 1) % n]
        arcs.append(Arc(te, tl))
    arcs.sort(key=lambda arc: arc.theta_enter)
    return "arcs", n, tuple(arcs)


# --------------------------------------------------------------------------
def _match_arcs(lo, hi, ev_radius, ev_kind):
    """Greedy nearest-midpoint matching of arcs across a shared event."""
    used_hi = set()
    out = []
    for i, alo in enumerate(lo.arcs):
        best, bestd = None, math.inf
        for j, ahi in enumerate(hi.arcs):
            if j in used_hi:
                continue
            d = abs(((ahi.midpoint - alo.midpoint + math.pi) % _TWO_PI) - math.pi)
            if d < bestd:
                best, bestd = j, d
        if best is not None and bestd < 0.6:
            used_hi.add(best)
            out.append(ArcTransition(ev_radius, ev_kind, "continue",
                                     lo.index, i, hi.index, best))
        else:
            out.append(ArcTransition(ev_radius, ev_kind, "death",
                                     lo.index, i, hi.index, None))
    for j in range(len(hi.arcs)):
        if j not in used_hi:
            out.append(ArcTransition(ev_radius, ev_kind, "birth",
                                     lo.index, None, hi.index, j))
    return out


def classify_cells(params: LensParams, *, samples_per_cell=3) -> TopologyResult:
    xs, ys = params.primary_frame_source()
    a, m0, rho = params.a, params.m0, params.rho
    events, r_max = radial_events(a, m0, xs, ys, rho)

    # collapse events sharing a radius (degenerate radii carry several kinds)
    merged: list = []
    for e in events:
        if merged and abs(e.radius - merged[-1][0]) <= max(1e-9, 1e-7 * e.radius):
            merged[-1][1].append(e.kind)
        else:
            merged.append([e.radius, [e.kind]])

    bounds = [0.0] + [m[0] for m in merged] + [r_max]
    kinds = ["origin"] + ["+".join(sorted(set(m[1]))) for m in merged] + ["R_max"]

    cells = []
    worst = "OK"
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < 1e-11:
            continue
        fr = np.linspace(0.18, 0.82, max(1, samples_per_cell))
        rs = lo + fr * (hi - lo)
        seen = [arcs_at(R, a, m0, xs, ys, rho) for R in rs]
        mid = seen[len(seen) // 2]
        sig = {(s[0], s[1]) for s in seen}
        status = "OK"
        if len(sig) != 1 or mid[1] % 2 != 0:
            status = "TOPOLOGY_UNCERTAIN"
            worst = "TOPOLOGY_UNCERTAIN"
        cells.append(CellPlan(
            index=len(cells), r_lo=lo, r_hi=hi,
            r_mid=float(rs[len(rs) // 2]),
            kind=mid[0], n_crossings=mid[1], arcs=mid[2],
            lo_event=kinds[k], hi_event=kinds[k + 1],
            status=status,
            detail="" if len(sig) == 1 else f"samples disagree: {sorted(sig)}"))

    transitions = []
    for lo, hi in zip(cells[:-1], cells[1:]):
        transitions.extend(_match_arcs(lo, hi, hi.r_lo, lo.hi_event))

    return TopologyResult(params=params, r_max=r_max, cells=cells,
                          transitions=transitions, status=worst)

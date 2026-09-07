"""Dump Python holonomic_ref values the C++ M7 port is checked against.

Writes evidence/holonomic/m7_reference.tsv -- a flat, whitespace-free-value
TSV so the C++ harness needs no JSON dependency.  One record per line:

  PF     <case> a m0 X Y rho rmax
  EV     <case> radius kind physically_real
  SAMPLE <case> R theta phi dphi_dtheta  dP0..dP4  bq0..bq4  dbq(5x5, row-major)
  RT     <case> R f0 fh df0(5) dfh(5) reliable kind
  EPOCH  <case> u mu grad_mu(5) dmu_du F0 F_half status

Run: taskset -c 0-7 python tests/holonomic_cpp/dump_reference.py
"""
from __future__ import annotations

import math
import os
import sys

_REF = os.path.join(os.path.dirname(__file__), "..", "..", "python", "lcbinint")
sys.path.insert(0, os.path.abspath(_REF))

import numpy as np                                            # noqa: E402
import holonomic_ref as H                                     # noqa: E402
from holonomic_ref import LensParams                           # noqa: E402
from holonomic_ref.polynomial_family import (                   # noqa: E402
    boundary_quartic, boundary_quartic_dp)
from holonomic_ref.jacobian import (                            # noqa: E402
    phi_grad, radius_terms, arc_intervals)
from holonomic_ref.radial_events import radial_events            # noqa: E402

CASES = {
    "benign": (0.35, 0.22, 0.05, 0.4, 1.15),
    "close": (0.4, -0.05, 0.09, 0.8, 0.55),
    "resonant": (0.05, 0.02, 0.05, 0.3, 0.9),
    "plan15": (1 / 5, 1 / 7, 1 / 8, 0.5, 1.2),
    "wide-planet": (1.4, 0.10, 0.03, 1e-3, 2.5),
    "cusp": (0.9, -0.3, 0.05, 0.25, 0.7),
    "tiny-rho": (0.1, 0.02, 5e-3, 0.3, 1.0),
}


def g(*xs):
    return " ".join(repr(float(x)) if not isinstance(x, str) else x for x in xs)


def dump(fh):
    for name, (xs, ys, rho, q, a) in CASES.items():
        p = LensParams(xs=xs, ys=ys, rho=rho, q=q, a=a, barycentric=False)
        X, Y = p.primary_frame_source()
        pf = (a, p.m0, X, Y, rho)
        evs, r_max = radial_events(a, p.m0, X, Y, rho)
        print("PF", name, g(a, p.m0, X, Y, rho, r_max), file=fh)
        for e in evs:
            print("EV", name, g(e.radius), e.kind,
                  int(e.physically_real), file=fh)

        for R in np.linspace(0.15 * r_max, 0.95 * r_max, 7):
            for th in np.linspace(0.2, 2 * math.pi - 0.2, 5):
                gr = phi_grad(R, th, a, p.m0, X, Y, rho)
                bq = boundary_quartic(R, a, p.m0, X, Y, rho)
                dbq = boundary_quartic_dp(R, a, p.m0, X, Y, rho)
                flat = []
                for row in dbq:
                    flat += list(row)
                print("SAMPLE", name, g(R, th, gr[0], gr[1]),
                      g(*gr[2]), g(*bq), g(*flat), file=fh)

        for R in np.linspace(0.2 * r_max, 0.9 * r_max, 11):
            f0, fh_, d0, dh, rel = radius_terms(float(R), pf)
            kind, _ = arc_intervals(float(R), pf)
            print("RT", name, g(R, f0, fh_), g(*d0), g(*dh),
                  int(bool(rel)), kind, file=fh)

        for u in (0.0, 0.5):
            ej = H.epoch_jacobian(p, u, n_r=64)
            print("EPOCH", name, g(u, ej.mu), g(*ej.grad_mu),
                  g(ej.dmu_du, ej.F0, ej.F_half), ej.status, file=fh)


if __name__ == "__main__":
    dest = os.path.join(os.path.dirname(__file__), "..", "..", "evidence",
                        "holonomic", "m7_reference.tsv")
    dest = os.path.abspath(dest)
    with open(dest, "w") as fh:
        dump(fh)
    os.remove(os.path.join(os.path.dirname(dest), "m7_reference.json")) \
        if os.path.exists(os.path.join(os.path.dirname(dest),
                                       "m7_reference.json")) else None
    print("wrote", dest)

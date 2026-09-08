"""Dump holonomic_ref/root_pair.py values the C++ root_pair.hpp port checks.

Writes evidence/holonomic/root_pair_ref.tsv -- one record per boundary root
pair, whitespace-separated so the C++ harness needs no JSON dependency:

  name R  pc0..pc4  pcR0..pcR4  m v  E O  E_m E_v O_m O_v  dm_dR dv_dR ok

`pc`  = boundary_quartic(R) coefficients (ascending),
`pcR` = boundary_quartic_dR(R) coefficients,
`(m, v)` = symmetric coordinates of a real root pair of `pc`,
`E, O`   = eo_residuals, `E_m..O_v` = eo_jacobian, `dm_dR, dv_dR` = root_pair_dR
(`ok = 0` if the pair Jacobian is singular -- tangency).

Run: taskset -c 0-7 python tests/holonomic_cpp/gen_root_pair_ref.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(_HERE, "..", "..", "python", "lcbinint", "holonomic_ref")
_OUT = os.path.join(_HERE, "..", "..", "evidence", "holonomic",
                    "root_pair_ref.tsv")
_BENCH = os.environ.get("BENCH_TSV", "/tmp/bench_cases.tsv")

# The installed `lcbinint` package can shadow the in-tree holonomic_ref, so
# load the two modules we need straight from their files.
_pkg = types.ModuleType("holonomic_ref")
_pkg.__path__ = [_REF]
sys.modules["holonomic_ref"] = _pkg


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"holonomic_ref.{name}", os.path.join(_REF, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"holonomic_ref.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_pf = _load("polynomial_family")
rp = _load("root_pair")
boundary_quartic = _pf.boundary_quartic
boundary_quartic_dR = _pf.boundary_quartic_dR


def real_roots(pc):
    c = list(reversed(pc))
    while len(c) > 1 and c[0] == 0.0:
        c = c[1:]
    if len(c) <= 1:
        return []
    z = np.roots(c)
    return sorted(r.real for r in z
                  if abs(r.imag) <= 1e-8 * (1.0 + abs(r.real)))


def main():
    rows = []
    with open(_BENCH) as f:
        for line in f:
            p = line.split()
            if len(p) < 9:
                continue
            xs, ys, rho, q, a = map(float, p[:5])
            bary = int(p[5])
            name = p[8]
            m0 = 1.0 / (1.0 + q)
            m1 = q / (1.0 + q)
            X = xs + m1 * a if bary else xs
            Y = ys
            for R in np.linspace(0.15, 1.0 + a, 9):
                pc = boundary_quartic(R, a, m0, X, Y, rho)
                pcR = boundary_quartic_dR(R, a, m0, X, Y, rho)
                rr = real_roots(pc)
                for i in range(len(rr) - 1):
                    pair = rp.from_endpoints(rr[i], rr[i + 1])
                    E, O = rp.eo_residuals(pair, pc)
                    (E_m, E_v), (O_m, O_v) = rp.eo_jacobian(pair, pc)
                    try:
                        dm, dv = rp.root_pair_dR(pair, R, a, m0, X, Y, rho)
                        ok = 1
                    except ValueError:
                        dm = dv = 0.0
                        ok = 0
                    vals = ([float(R)] + [float(x) for x in pc]
                            + [float(x) for x in pcR]
                            + [float(pair.m), float(pair.v), float(E), float(O),
                               float(E_m), float(E_v), float(O_m), float(O_v),
                               float(dm), float(dv)])
                    rows.append([name] + vals + [ok])

    with open(_OUT, "w") as f:
        for r in rows:
            f.write(" ".join(f"{x:.17g}" if isinstance(x, float) else str(x)
                             for x in r) + "\n")
    print(f"{len(rows)} rows -> {os.path.relpath(_OUT)}")


if __name__ == "__main__":
    main()

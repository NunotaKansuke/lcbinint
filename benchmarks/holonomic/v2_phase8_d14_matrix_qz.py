#!/usr/bin/env python3
"""Research-only D14 matrix-polynomial/QZ probe.

This script is intentionally outside the production C++ build.  It uses
SciPy's QZ implementation to test the 16x16 first companion pencil described
in d14_matrix.hpp.  The exact coefficient identity is checked against the
same C3/G4/Z3 formula, and beta==0 generalized eigenvalues are counted as the
two grade-infinite roots.  A QZ result is never fed to the V2 router.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eig


def cases(path: Path):
    for line in path.read_text().splitlines():
        row = line.split()
        if len(row) >= 9:
            yield tuple(float(x) for x in row[:5]) + (int(row[5]), row[8])


def blocks(xs, ys, rho, q, a, bary):
    m1 = q / (1.0 + q)
    m = 1.0 / (1.0 + q)
    x = xs + m1 * a if bary else xs
    y = ys
    h = rho * rho
    a2 = a * a
    beta = x * x + y * y - h

    c = np.array([
        a2 * m * m,
        1.0 - 2.0 * a2 * m + a2 * beta + 2.0 * a * x * (3.0 * m - 1.0),
        -2.0 + a2 + beta - 4.0 * a * x,
        1.0,
    ])
    l = np.array([-a2 * m, a2 - 1.0, 1.0])
    u = np.array([
        a * m - a2 * m * x,
        -a * (m + 1.0) + x * (a2 - 1.0) + a * beta,
        a + x,
    ])
    g = np.polynomial.polynomial.polymul(u, u)
    g += y * y * np.polynomial.polynomial.polymul(l, l)
    g += -4.0 * a * x * np.polynomial.polynomial.polymul([-m, 1.0], c)
    g += -4.0 * a2 * (4.0 * x * x + y * y) * np.array([0.0, m * m, -2.0 * m, 1.0, 0.0])

    b2 = np.array([a2 * m * m, 2.0 * a * m * (x - a),
                   (x - a) ** 2 + y * y - h])
    z = a2 * (1.0 - m) * y * y * np.polynomial.polynomial.polymul([-m, 1.0], b2)
    return c, g, z


def matrix_poly(c, g, z):
    a = np.polynomial.polynomial.polymul(c, c)
    a[1:1 + len(g)] -= 3.0 * g
    b = np.polynomial.polynomial.polymul(c, g)
    b[1:1 + len(z)] += 36.0 * z
    e = np.polynomial.polynomial.polymul(g, g)
    cz = np.polynomial.polynomial.polymul(c, z)
    e[:len(cz)] += 12.0 * cz
    m = np.zeros((9, 2, 2))
    m[:, 0, 0] = 2.0 * e
    m[:len(b), 0, 1] = b
    m[:len(b), 1, 0] = b
    m[:len(a), 1, 1] = 2.0 * a
    return m


def pencil(m):
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    for j in range(8):
        k = 7 - j
        a[:2, 2 * j:2 * j + 2] = -m[k]
        if j == 0:
            b[:2, :2] = m[8]
    for j in range(1, 8):
        a[2 * j:2 * j + 2, 2 * (j - 1):2 * j] = np.eye(2)
        b[2 * j:2 * j + 2, 2 * j:2 * j + 2] = np.eye(2)
    return a, b


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "evidence/holonomic/gm_coverage_cases.tsv")
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    timings = []
    finite_counts = []
    identity_worst = 0.0
    failures = 0
    for xs, ys, rho, q, a, bary, _name in cases(path):
        c, g, z = blocks(xs, ys, rho, q, a, bary != 0)
        A = np.polynomial.polynomial.polymul(c, c)
        A[1:1 + len(g)] -= 3.0 * g
        B = np.polynomial.polynomial.polymul(c, g)
        B[1:1 + len(z)] += 36.0 * z
        E = np.polynomial.polynomial.polymul(g, g)
        cz = np.polynomial.polynomial.polymul(c, z)
        E[:len(cz)] += 12.0 * cz
        det = 4.0 * np.polynomial.polynomial.polymul(A, E)
        det[:len(np.polynomial.polynomial.polymul(B, B))] -= np.polynomial.polynomial.polymul(B, B)
        # The scalar identity is Dhat = det/3, D14 = 4096 Dhat.
        # Reconstruct Dhat directly from the same blocks for an independent
        # coefficient-level check of the matrix construction.
        f6 = np.polynomial.polynomial.polymul(c, c)
        f6[1:1 + len(g)] -= 4.0 * g
        dhat = np.polynomial.polynomial.polymul(f6, np.polynomial.polynomial.polymul(g, g))
        inner = 2.0 * np.polynomial.polynomial.polymul(c, c)
        inner[1:1 + len(g)] -= 9.0 * g
        term2 = 8.0 * np.polynomial.polynomial.polymul(
            np.polynomial.polynomial.polymul(c, inner), z)
        term3 = -432.0 * np.pad(np.polynomial.polynomial.polymul(z, z), (2, 0))
        dhat[:len(term2)] += term2
        dhat[:len(term3)] += term3
        identity_worst = max(identity_worst,
                             float(np.max(np.abs(det / 3.0 - dhat)) /
                                   (1.0 + np.max(np.abs(dhat)))))
        if not np.all(np.isfinite(det)) or len(det) < 15:
            failures += 1
            continue
        pencil_a, pencil_b = pencil(matrix_poly(c, g, z))
        finite = 0
        for _ in range(reps):
            start = time.perf_counter()
            alpha, beta = eig(pencil_a, pencil_b, left=False, right=False,
                              homogeneous_eigvals=True)
            timings.append((time.perf_counter() - start) * 1e3)
            alpha = np.asarray(alpha)
            beta = np.asarray(beta)
            finite = int(np.count_nonzero(np.abs(beta) >
                                          1e-12 * np.maximum(1.0, np.abs(alpha) + np.abs(beta))))
        finite_counts.append(finite)
        if finite != 14:
            failures += 1

    def pct(p):
        return float(np.percentile(timings, p)) if timings else 0.0

    print(f"cases={len(finite_counts)} reps={reps}")
    print(f"qz_ms_p50={pct(50):.6f} qz_ms_p90={pct(90):.6f} "
          f"qz_ms_p99={pct(99):.6f} qz_ms_max={pct(100):.6f}")
    print(f"finite_count_min={min(finite_counts, default=0)} "
          f"finite_count_max={max(finite_counts, default=0)}")
    print(f"matrix_identity_worst={identity_worst:.3e} failures={failures}")
    print("production_router=unchanged")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render q--rho maps with target-matched, profile-specific colour scales.

The speed scale is shared between the ``1e-3`` and ``1e-4`` targets within a
profile, so target-to-target comparisons do not silently change colour
meaning.  The lower error row uses the lower-error scale for ``1e-4`` and the
standard scale for ``1e-3``.
"""

from __future__ import annotations

import math
import sys

import numpy as np

import plot_controlled_pure_kernel_protocol_edges as protocol


SPEED_SCALES = {
    "uniform": (
        (0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0),
        (
            r"$<0.01\times$", r"$0.01\times$", r"$0.1\times$",
            r"$1\times$", r"$2\times$", r"$3\times$",
            r"$4\times$", r"$6\times$", r"$8\times$",
            r"$\geq16\times$",
        ),
    ),
    "linear": (
        (0.03, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0),
        (
            r"$<0.03\times$", r"$0.03\times$", r"$0.1\times$",
            r"$1\times$", r"$2\times$", r"$4\times$",
            r"$6\times$", r"$8\times$", r"$12\times$",
            r"$\geq16\times$",
        ),
    ),
}

LOW_ERROR_BOUNDARIES = np.asarray(
    (1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3),
    dtype=float,
)
LOW_ERROR_LABELS = (
    r"$\leq10^{-5}$", r"$2\times10^{-5}$", r"$5\times10^{-5}$",
    r"$10^{-4}$", r"$2\times10^{-4}$", r"$5\times10^{-4}$",
    r"$10^{-3}$", r"$\geq10^{-3}$",
)


def _argument(name, default):
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


profile = _argument("--profile", "uniform")
target = float(_argument("--target", "0.001"))
if profile not in SPEED_SCALES:
    raise SystemExit(f"unsupported profile: {profile}")

protocol.legacy.SPEED_BOUNDARIES = np.asarray(
    SPEED_SCALES[profile][0], dtype=float
)
protocol.legacy.SPEED_LABELS = SPEED_SCALES[profile][1]
if math.isclose(target, 1.0e-4, rel_tol=0.0, abs_tol=1.0e-15):
    protocol.legacy.ERROR_BOUNDARIES = LOW_ERROR_BOUNDARIES
    protocol.legacy.ERROR_LABELS = LOW_ERROR_LABELS


if __name__ == "__main__":
    protocol.legacy.main()

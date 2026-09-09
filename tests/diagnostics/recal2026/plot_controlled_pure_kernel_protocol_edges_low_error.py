#!/usr/bin/env python3
"""Render protocol-clamped maps with a lower-resolution error colour scale.

The ordinary map starts its p95 error colour bins at ``1e-4``.  That is useful
for the looser target, but it saturates the low-error half of the ``1e-4``
map.  This wrapper keeps the same records, binning, runtime colours, and
minimum-count rule while moving only the lower-row error colour boundaries.
"""

from __future__ import annotations

import numpy as np

import plot_controlled_pure_kernel_protocol_edges as protocol


protocol.legacy.ERROR_BOUNDARIES = np.asarray(
    (1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3),
    dtype=float,
)
protocol.legacy.ERROR_LABELS = (
    r"$\leq10^{-5}$", r"$2\times10^{-5}$", r"$5\times10^{-5}$",
    r"$10^{-4}$", r"$2\times10^{-4}$", r"$5\times10^{-4}$",
    r"$10^{-3}$", r"$\geq10^{-3}$",
)


if __name__ == "__main__":
    protocol.legacy.main()

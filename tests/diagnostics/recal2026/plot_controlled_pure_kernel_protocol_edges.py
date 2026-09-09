#!/usr/bin/env python3
"""Render the pure-kernel map with q--rho axes clamped to protocol ranges.

The historical plotting helper decade-floors observed lower bounds.  For
q--rho this can create a rho bin below the corpus protocol range
``[3e-5, 1]``.  This wrapper preserves the helper for all other axes while
using the physical protocol range for q and rho.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_SCRIPT_DIR = Path(
    "/home/nunota/.codex/worktrees/8d2b/lcbinint/tests/diagnostics/recal2026"
)
sys.path.insert(0, str(LEGACY_SCRIPT_DIR))
import plot_controlled_pure_kernel_speed_accuracy_map as legacy  # noqa: E402


_LEGACY_LOG_EDGES = legacy._log_edges


def _protocol_edges(records, field, default_low, default_high, bins=12):
    if field in {"q", "rho"}:
        return np.geomspace(default_low, default_high, bins + 1)
    return _LEGACY_LOG_EDGES(
        records, field, default_low, default_high, bins=bins
    )


legacy._log_edges = _protocol_edges


if __name__ == "__main__":
    legacy.main()

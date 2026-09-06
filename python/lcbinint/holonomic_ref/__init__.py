"""Pure-Python reference implementation of the ATPT holonomic finite-source solver.

Correctness first, speed never.  Used as the oracle that the future C++
production backend (M7) must reproduce.  See docs/holonomic/.
"""
from .polynomial_family import (
    boundary_quartic,
    T_coeffs,
    B_coeffs,
    phi_value,
    LensParams,
)
from .radial_events import radial_events, RadialEvent
from .topology import (
    Arc,
    CellPlan,
    ArcTransition,
    TopologyResult,
    arcs_at,
    classify_cells,
)

__all__ = [
    "boundary_quartic",
    "T_coeffs",
    "B_coeffs",
    "phi_value",
    "LensParams",
    "radial_events",
    "RadialEvent",
    "Arc",
    "CellPlan",
    "ArcTransition",
    "TopologyResult",
    "arcs_at",
    "classify_cells",
]

"""Pure-Python reference implementation of the ATPT holonomic finite-source solver.

Correctness first, speed never.  Used as the oracle that the future C++
production backend (M7) must reproduce.  See docs/holonomic/.
"""
from .polynomial_family import boundary_quartic, T_coeffs, B_coeffs, phi_value
from .radial_events import radial_events, RadialEvent

__all__ = [
    "boundary_quartic",
    "T_coeffs",
    "B_coeffs",
    "phi_value",
    "radial_events",
    "RadialEvent",
]

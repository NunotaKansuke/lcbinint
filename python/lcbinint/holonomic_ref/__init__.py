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
from .connection import (
    q_coeffs,
    q_coeffs_exact,
    connection_matrix,
    gm_polynomials,
    s_polynomials,
)
from .period_reduction import (
    h_coeffs,
    h_coeffs_exact,
    residue_b_exact,
    laurent_b,
    psi_reduction_matrix,
    residue_at_infinity,
    observed_covector_exact,
    arc_chart,
    arc_t_endpoints,
    half_period_eta,
    half_period_obs_angular,
    half_period_obs_reduced,
    closed_period_eta,
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
    "q_coeffs",
    "q_coeffs_exact",
    "connection_matrix",
    "gm_polynomials",
    "s_polynomials",
    "h_coeffs",
    "h_coeffs_exact",
    "residue_b_exact",
    "laurent_b",
    "psi_reduction_matrix",
    "residue_at_infinity",
    "observed_covector_exact",
    "arc_chart",
    "arc_t_endpoints",
    "half_period_eta",
    "half_period_obs_angular",
    "half_period_obs_reduced",
    "closed_period_eta",
]

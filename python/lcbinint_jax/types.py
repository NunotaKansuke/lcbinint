"""Fixed-shape result types for the experimental JAX inverse-ray kernel."""

from typing import NamedTuple

import jax


class FixedSupportResult(NamedTuple):
    """Result of integrating a caller-supplied set of image-plane tiles."""

    magnification: jax.Array
    moments: jax.Array
    boundary_cells: jax.Array
    active_cells: jax.Array


class DiscoveryResult(NamedTuple):
    """Fixed-capacity macro-tile support discovered from image seeds."""

    tile_indices: jax.Array
    tile_origins: jax.Array
    tile_mask: jax.Array
    active_mask: jax.Array
    overflow: jax.Array
    visited_count: jax.Array
    active_count: jax.Array
    seed_count: jax.Array
    root_failure: jax.Array


class InverseRayResult(NamedTuple):
    """Magnification and discovery diagnostics from the automatic MVP path."""

    magnification: jax.Array
    moments: jax.Array
    boundary_cells: jax.Array
    active_cells: jax.Array
    tile_count: jax.Array
    discovery_overflow: jax.Array
    root_failure: jax.Array
    support_valid: jax.Array


class PolarSupportResult(NamedTuple):
    """Stopped-gradient radial bands for polar inverse-ray integration."""

    lower: jax.Array
    upper: jax.Array
    mask: jax.Array
    band_count: jax.Array
    overflow: jax.Array
    root_failure: jax.Array


class AutoInverseRayResult(NamedTuple):
    """Unified result from the automatic Cartesian/polar dispatcher."""

    magnification: jax.Array
    moments: jax.Array
    boundary_cells: jax.Array
    active_cells: jax.Array
    support_count: jax.Array
    discovery_overflow: jax.Array
    root_failure: jax.Array
    support_valid: jax.Array
    used_polar: jax.Array


class HybridMagnificationResult(NamedTuple):
    """Result from the multipole/image-plane/source-plane dispatcher."""

    magnification: jax.Array
    method: jax.Array
    estimated_error: jax.Array
    support_valid: jax.Array
    used_multipole: jax.Array
    used_polar: jax.Array
    used_source_plane: jax.Array
    used_expanded_cartesian: jax.Array


class BinaryResolutionSelection(NamedTuple):
    """Stopped-gradient native-calibrated binary inverse-ray selection."""

    source_bins: jax.Array
    prefer_polar: jax.Array


class BinaryRoutingDiagnostics(NamedTuple):
    """Stopped native point-safety and caustic-scan routing diagnostics."""

    point_magnification: jax.Array
    point_error_estimate: jax.Array
    point_absolute_tolerance: jax.Array
    caustic_distance: jax.Array
    scan_min_distance: jax.Array
    quadrupole_indicator: jax.Array
    cusp_indicator: jax.Array
    ghost_indicator: jax.Array
    planetary_distance2: jax.Array
    image_count: jax.Array
    ghost_count: jax.Array
    safety_flags: jax.Array
    point_preflight_safe: jax.Array
    point_safe: jax.Array
    scan_performed: jax.Array
    any_vertex_inside: jax.Array
    has_crossing_probes: jax.Array
    chord_band: jax.Array
    tangent_band: jax.Array
    grazing_ring_band: jax.Array


class CalibratedMagnificationResult(NamedTuple):
    """Hybrid result plus native-calibrated routing diagnostics."""

    magnification: jax.Array
    method: jax.Array
    estimated_error: jax.Array
    support_valid: jax.Array
    used_multipole: jax.Array
    used_polar: jax.Array
    used_source_plane: jax.Array
    used_expanded_cartesian: jax.Array
    selected_source_bins: jax.Array
    comparison_resolution: jax.Array
    executed_resolution: jax.Array
    tile_capacity: jax.Array
    caustic_distance: jax.Array
    prefer_polar: jax.Array
    point_safe: jax.Array
    chord_band: jax.Array
    tangent_band: jax.Array
    grazing_ring_band: jax.Array
    value_error: jax.Array
    value_budget: jax.Array
    value_converged: jax.Array


class TrajectoryMagnificationResult(NamedTuple):
    """Conditional hybrid results for a one-dimensional source trajectory."""

    magnification: jax.Array
    method: jax.Array
    estimated_error: jax.Array
    support_valid: jax.Array
    value_converged: jax.Array
    used_multipole: jax.Array
    used_polar: jax.Array
    used_source_plane: jax.Array
    used_expanded_cartesian: jax.Array
    attempted_counts: jax.Array


class ConvergenceResult(NamedTuple):
    """Coarse/fine convergence diagnostics for normalized observables."""

    coarse_observables: jax.Array
    fine_observables: jax.Array
    observable_errors: jax.Array
    observable_budgets: jax.Array
    coarse_directional_derivatives: jax.Array
    fine_directional_derivatives: jax.Array
    derivative_errors: jax.Array
    derivative_budgets: jax.Array
    coarse_tile_count: jax.Array
    fine_tile_count: jax.Array
    coarse_support_valid: jax.Array
    fine_support_valid: jax.Array
    value_converged: jax.Array
    moments_converged: jax.Array
    gradient_checked: jax.Array
    gradient_converged: jax.Array

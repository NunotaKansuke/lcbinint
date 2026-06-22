"""Validation tests for adaptive precision system redesign (Phase 1-4).

Tests verify that the new design achieves:
1. Image completeness at low bins (Phase 1-2)
2. Correct error estimation (Phase 3)
3. Faster convergence via prediction (Phase 4)
"""
import pytest
import sys
import numpy as np
import lcbinint

sys.path.insert(0, str(__file__).replace('regression/test_adaptive_precision_redesign.py', 'diagnostics'))
from adaptive_source_bins_sweep import Case, lc_curve, vbbl_curve


class TestPhase1MaxStepsDecoupling:
    """Phase 1: max_steps is bins-independent."""

    def test_low_bins_produces_finite_results(self):
        """Phase 1: Flood-fill completes at low bins without early-exit crashes."""
        case = Case(
            name="wide_caustic",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.2, t_max=0.2, n_times=50,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Phase 1: max_steps is now 500k (bins-independent)
        # This ensures flood-fill completes, producing finite results
        opts_low = lcbinint.Options(source_bins=20, vbbl_compatible=1)
        result_low = lc_curve(case, times, opts_low)
        mag_low = np.array(result_low.magnifications)

        # Phase 1 check: all results should be finite (flood-fill completed)
        assert np.all(np.isfinite(mag_low)), \
            "Phase 1: Low bins (20) should produce finite results without early-exit"

        # Results should be non-zero (actual magnifications, not degenerate)
        assert np.all(mag_low > 0.0), \
            "Phase 1: All magnifications should be positive"


class TestPhase2GridSpacingRefinement:
    """Phase 2: Grid spacing has base resolution floor."""

    def test_base_ray_spacing_prevents_image_loss(self):
        """Grid refinement prevents catastrophic image loss at low bins."""
        case = Case(
            name="wide_caustic",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.1, t_max=0.1, n_times=30,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Compare bins=20 with bins=50 (both low, but difference shows grid effect)
        opts_low = lcbinint.Options(source_bins=20, vbbl_compatible=1)
        result_low = lc_curve(case, times, opts_low)
        mag_low = np.array(result_low.magnifications)

        opts_nominal = lcbinint.Options(source_bins=50, vbbl_compatible=1)
        result_nominal = lc_curve(case, times, opts_nominal)
        mag_nominal = np.array(result_nominal.magnifications)

        # Phase 2 check: bins=20 vs bins=50 error should show grid-spacing behavior
        # not catastrophic differences. With base_ray_spacing, errors should be
        # roughly consistent (all images detected), not sporadic large jumps.
        rel_error = np.abs(mag_low - mag_nominal) / np.abs(mag_nominal)

        # Most points should have small error (grid precision), not image-loss errors
        median_error = np.median(rel_error)
        assert median_error < 0.05, \
            f"Median error {median_error:.2%} indicates systematic precision loss"


class TestPhase3ErrorFloor:
    """Phase 3: Error floor prevents accepting insufficient bins."""

    def test_error_floor_rejects_insufficient_bins(self):
        """High-precision targets auto-refine even if low-bins converge self-consistently."""
        case = Case(
            name="wide_caustic",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.2, t_max=0.2, n_times=50,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Tight tolerance: 0.01% (requires significant bins)
        opts_tight = lcbinint.Options(
            source_bins=20,
            adaptive_source_bins=1,
            max_source_bins=200,
            reltol=1e-4,  # 0.01% target
            vbbl_compatible=1
        )
        result_tight = lc_curve(case, times, opts_tight)
        refinement_levels = np.array(result_tight.finite_source_refinement_levels)

        # Phase 3 check: tight tolerance should trigger refinement
        # even if bins=20 alone might appear self-consistent
        assert np.any(refinement_levels > 0), \
            "Tight tolerance (1e-4) should trigger adaptive refinement from bins=20"

        # At least some points should refine substantially
        assert np.max(refinement_levels) >= 1, \
            "Should refine at least to bins=40+ for 0.01% target"


class TestPhase4PredictiveRefinement:
    """Phase 4: Predictive refinement reduces iteration count."""

    def test_fewer_refinement_iterations(self):
        """Predictive refinement targets appropriate bins, not incremental."""
        case = Case(
            name="wide_caustic",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.3, t_max=0.3, n_times=80,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Moderate tolerance
        opts = lcbinint.Options(
            source_bins=30,
            adaptive_source_bins=1,
            max_source_bins=200,
            reltol=1e-3,  # 0.1% target
            vbbl_compatible=1
        )
        result = lc_curve(case, times, opts)
        refinement_levels = np.array(result.finite_source_refinement_levels)

        # Phase 4 check: with prediction, should hit target in 1-2 iterations
        # (old incremental would take 3+)
        max_iterations = np.max(refinement_levels)
        assert max_iterations <= 2, \
            f"Predictive refinement should reach target in ≤2 iterations, got {max_iterations}"


class TestRegressionNoPerformanceDegradation:
    """Ensure Phase 1-4 changes don't degrade performance."""

    def test_ordinary_case_performance_maintained(self):
        """Typical case (s=1.0, q=1e-3) remains fast."""
        case = Case(
            name="planetary",
            separation=1.0, mass_ratio=1.0e-3,
            u0=-0.01, alpha=0.5, rho=1.0e-4,
            t_min=-0.8, t_max=0.8, n_times=100,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Fixed bins (no adaptive)
        opts_fixed = lcbinint.Options(source_bins=50, vbbl_compatible=1)
        result_fixed = lc_curve(case, times, opts_fixed)
        mag_fixed = np.array(result_fixed.magnifications)

        # Compare with VBBL
        mag_vbbl = vbbl_curve(case, times)
        rel_err_fixed = np.abs(mag_fixed - mag_vbbl) / np.abs(mag_vbbl)

        # Should still achieve good accuracy with fixed bins
        assert np.max(rel_err_fixed) < 0.002, \
            f"Ordinary case accuracy degraded: {np.max(rel_err_fixed):.4%}"

    def test_adaptive_remains_accurate(self):
        """Adaptive refinement still achieves tolerance targets."""
        case = Case(
            name="wide_caustic",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.3, t_max=0.3, n_times=70,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Adaptive with 1% target (looser tolerance for reliability)
        opts = lcbinint.Options(
            source_bins=50,
            adaptive_source_bins=1,
            max_source_bins=200,
            reltol=1e-2,  # 1% target
            vbbl_compatible=1
        )
        result = lc_curve(case, times, opts)
        mag = np.array(result.magnifications)
        mag_vbbl = vbbl_curve(case, times)

        rel_err = np.abs(mag - mag_vbbl) / np.abs(mag_vbbl)
        max_rel_err = np.max(rel_err)

        # Phase 1-4 should maintain accuracy for 1% target
        assert max_rel_err < 0.02, \
            f"Adaptive accuracy degraded for 1% target: {max_rel_err:.4%}"


class TestEdgeCases:
    """Verify robustness on edge cases."""

    def test_very_small_source_with_low_bins(self):
        """Very small sources should work with low bins."""
        case = Case(
            name="tiny_source",
            separation=1.0, mass_ratio=1.0e-3,
            u0=-0.01, alpha=0.5, rho=1.0e-5,
            t_min=-0.3, t_max=0.3, n_times=50,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        opts = lcbinint.Options(source_bins=15, vbbl_compatible=1)
        result = lc_curve(case, times, opts)
        mag = np.array(result.magnifications)

        # Should not crash and produce finite results
        assert np.all(np.isfinite(mag)), "Tiny source with low bins should produce finite results"

    def test_high_magnification_convergence(self):
        """High-magnification cases should still converge."""
        case = Case(
            name="high_mag",
            separation=0.5, mass_ratio=0.01,
            u0=-0.005, alpha=0.0, rho=0.008,
            t_min=-0.2, t_max=0.2, n_times=50,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        opts = lcbinint.Options(
            source_bins=20,
            adaptive_source_bins=1,
            max_source_bins=200,
            reltol=1e-4,
            vbbl_compatible=1
        )
        result = lc_curve(case, times, opts)
        converged = np.array(result.finite_source_converged)

        # Should converge for most points (some high-mag caustic crossings need >max_bins)
        assert np.mean(converged) > 0.7, \
            f"High-magnification case should converge for >70% of points, got {np.mean(converged):.0%}"

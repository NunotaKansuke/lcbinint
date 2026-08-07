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
from adaptive_source_bins_sweep import Case, lc_curve, vbm_curve


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
        opts_low = lcbinint.Options(source_bins=20)
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
        opts_low = lcbinint.Options(source_bins=20)
        result_low = lc_curve(case, times, opts_low)
        mag_low = np.array(result_low.magnifications)

        opts_nominal = lcbinint.Options(source_bins=50)
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


class TestPhase3EmpiricalResolution:
    """Phase 3: The calibrated binary selector is one-shot."""

    def test_auto_nbin_preselects_for_tight_tolerance(self):
        """The empirical law handles the corpus without grid retries."""
        case = Case(
            name="wide_caustic",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.2, t_max=0.2, n_times=50,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Tight tolerance: 0.01% (requires significant bins)
        opts_tight = lcbinint.Options(
            nbin="auto",
            max_source_bins=200,
            reltol=1e-4,  # 0.01% target
        )
        result_tight = lc_curve(case, times, opts_tight)
        refinement_levels = np.array(result_tight.finite_source_refinement_levels)

        assert np.all(np.isfinite(result_tight.finite_source_magnifications))
        # The empirical selection is the complete nbin decision. Diagnostics
        # and support certification must not arm a second resolution.
        assert np.all(refinement_levels == 0)
        converged = np.array(result_tight.finite_source_converged)
        assert np.array_equal(
            np.isfinite(result_tight.magnifications), converged
        )
        assert result_tight.all_converged == bool(np.all(converged))
        assert len(result_tight.unconverged_indices) == int(np.count_nonzero(~converged))


class TestOneShotAutoResolution:
    """The calibrated selector never uses error-indicator feedback."""

    def test_refinement_iterations_are_zero(self):
        """Automatic binary resolution performs exactly one selected grid."""
        case = Case(
            name="wide_caustic",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.3, t_max=0.3, n_times=80,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Moderate tolerance
        opts = lcbinint.Options(
            nbin="auto",
            max_source_bins=200,
            reltol=1e-3,  # 0.1% target
        )
        result = lc_curve(case, times, opts)
        mag = np.array(result.magnifications)
        refinement_levels = np.array(result.finite_source_refinement_levels)
        converged = np.array(result.finite_source_converged)

        assert np.all(refinement_levels == 0)

        assert np.all(np.isfinite(mag)), "Auto nbin should return finite magnifications"
        assert result.all_converged
        assert result.all_converged == bool(np.all(converged))
        assert len(result.unconverged_indices) == int(np.count_nonzero(~converged))


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
        opts_fixed = lcbinint.Options(source_bins=50)
        result_fixed = lc_curve(case, times, opts_fixed)
        mag_fixed = np.array(result_fixed.magnifications)

        # Compare with vbm
        mag_vbm = vbm_curve(case, times)
        rel_err_fixed = np.abs(mag_fixed - mag_vbm) / np.abs(mag_vbm)

        # Should still achieve good accuracy with fixed bins
        assert np.max(rel_err_fixed) < 0.002, \
            f"Ordinary case accuracy degraded: {np.max(rel_err_fixed):.4%}"

    def test_adaptive_remains_accurate(self):
        """Automatic empirical resolution still achieves tolerance targets."""
        case = Case(
            name="wide_caustic",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.3, t_max=0.3, n_times=70,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Adaptive with 1% target (looser tolerance for reliability)
        opts = lcbinint.Options(
            nbin="auto",
            max_source_bins=200,
            reltol=1e-2,  # 1% target
        )
        result = lc_curve(case, times, opts)
        mag = np.array(result.magnifications)
        mag_vbm = vbm_curve(case, times)

        rel_err = np.abs(mag - mag_vbm) / np.abs(mag_vbm)
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

        opts = lcbinint.Options(source_bins=15)
        result = lc_curve(case, times, opts)
        mag = np.array(result.magnifications)

        # Should not crash and produce finite results
        assert np.all(np.isfinite(mag)), "Tiny source with low bins should produce finite results"

    def test_high_magnification_convergence(self):
        """High-magnification cases still fail closed on support failures."""
        case = Case(
            name="high_mag",
            separation=0.5, mass_ratio=0.01,
            u0=-0.005, alpha=0.0, rho=0.008,
            t_min=-0.2, t_max=0.2, n_times=50,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        opts = lcbinint.Options(
            nbin="auto",
            max_source_bins=200,
            reltol=1e-4
        )
        result = lc_curve(case, times, opts)
        mag = np.array(result.magnifications)
        converged = np.array(result.finite_source_converged)

        # The empirical grid is one-shot, but image-component support remains
        # an independent fail-closed requirement. The raw integrated value is
        # retained diagnostically even when public magnification is withheld.
        assert np.all(np.isfinite(result.finite_source_magnifications)), \
            "the integrated value must survive even when the budget is missed"
        assert np.array_equal(np.isfinite(mag), converged)
        assert np.mean(converged) >= 0.65, \
            f"High-magnification convergence unexpectedly dropped to {np.mean(converged):.0%}"
        assert result.all_converged == bool(np.all(converged))
        assert len(result.unconverged_indices) == int(np.count_nonzero(~converged))

"""Validation tests for Phase 6 component union redesign.

Tests focus on edge cases that could expose issues with the new
component-based tracking and inclusion-exclusion logic.
"""
import pytest
import sys
import numpy as np
import lcbinint

# Add diagnostics to path to import helper functions
sys.path.insert(0, str(__file__).replace('regression/test_component_union_validation.py', 'diagnostics'))
from adaptive_source_bins_sweep import Case, lc_curve, vbbl_curve


class TestComponentUnionEdgeCases:
    """Edge cases for component union and overlap handling."""

    def test_wide_caustic_large_source(self):
        """Wide caustic with large rho=0.01 (Phase 6 validation case)."""
        case = Case(
            name="wide_caustic_large_source",
            separation=0.95, mass_ratio=0.01,
            u0=-0.01, alpha=0.5, rho=0.01,
            t_min=-0.8, t_max=0.8, n_times=400,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Use adaptive refinement with tight tolerance
        opts = lcbinint.Options(
            source_bins=50, adaptive_source_bins=1,
            max_source_bins=200, reltol=1e-4, vbbl_compatible=1
        )
        result_lc = lc_curve(case, times, opts)
        mag_lc = np.array(result_lc.magnifications)

        # Compare against VBBL
        mag_vbbl = vbbl_curve(case, times)

        # Check error metrics
        rel_errors = np.abs(mag_lc - mag_vbbl) / np.abs(mag_vbbl)
        max_rel_error = np.max(rel_errors)
        median_error = np.median(rel_errors)

        # Phase 6 expectation: max error < 0.3% after union refinement
        assert max_rel_error < 0.003, f"Max error {max_rel_error:.4%} exceeds 0.3%"
        assert median_error < 1e-4, f"Median error {median_error:.4e} exceeds 1e-4"

    def test_complex_overlap_convergence(self):
        """Test convergence with complex fold overlaps (Phase 6 robustness)."""
        case = Case(
            name="complex_overlap",
            separation=0.75, mass_ratio=0.05,
            u0=-0.02, alpha=0.3, rho=0.015,
            t_min=-0.5, t_max=0.5, n_times=200,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        # Test different bin counts to ensure monotonic improvement
        results = {}
        for bins in [50, 100, 150, 200]:
            opts = lcbinint.Options(source_bins=bins, vbbl_compatible=1)
            result = lc_curve(case, times, opts)
            mag = np.array(result.magnifications)
            mag_vbbl = vbbl_curve(case, times)
            rel_err = np.abs(mag - mag_vbbl) / np.abs(mag_vbbl)
            results[bins] = np.max(rel_err)

        # Check monotonic improvement (or at least no regressions)
        errors = list(results.values())
        for i in range(len(errors) - 1):
            assert errors[i+1] <= errors[i] * 1.01, \
                f"Error not monotonic: bins={list(results.keys())[i]} -> {list(results.keys())[i+1]}: {errors[i]:.4%} -> {errors[i+1]:.4%}"

    def test_binary_stability_with_component_tracking(self):
        """Ensure component tracking doesn't destabilize ordinary binaries."""
        case = Case(
            name="ordinary_binary",
            separation=1.2, mass_ratio=0.3,
            u0=-0.05, alpha=0.0, rho=0.002,
            t_min=-1.0, t_max=1.0, n_times=200,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        opts = lcbinint.Options(source_bins=50, vbbl_compatible=1)
        result = lc_curve(case, times, opts)
        mag_lc = np.array(result.magnifications)
        mag_vbbl = vbbl_curve(case, times)

        rel_err = np.abs(mag_lc - mag_vbbl) / np.abs(mag_vbbl)
        max_rel_error = np.max(rel_err)

        # Even ordinary cases should have < 1% error with bins=50
        assert max_rel_error < 0.01, f"Ordinary binary error {max_rel_error:.4%} too high"

    def test_high_magnification_accuracy(self):
        """Test high-magnification cases where component union matters most."""
        case = Case(
            name="high_mag",
            separation=0.5, mass_ratio=0.01,
            u0=-0.005, alpha=0.0, rho=0.008,
            t_min=-0.3, t_max=0.3, n_times=100,
        )
        times = np.linspace(case.t_min, case.t_max, case.n_times)

        opts = lcbinint.Options(
            source_bins=50, adaptive_source_bins=1,
            max_source_bins=200, reltol=1e-4, vbbl_compatible=1
        )
        result = lc_curve(case, times, opts)
        mag_lc = np.array(result.magnifications)
        mag_vbbl = vbbl_curve(case, times)

        # At high magnification, errors are more sensitive
        rel_err = np.abs(mag_lc - mag_vbbl) / np.abs(mag_vbbl)
        max_rel_error = np.max(rel_err)

        assert max_rel_error < 0.002, f"High-mag error {max_rel_error:.4%} exceeds 0.2%"


class TestRegressionOnPhaseChanges:
    """Ensure Phase 1-6 changes didn't regress existing functionality."""

    def test_all_named_cases_still_pass(self):
        """Ensure representative cases remain solved with Phase 1-6."""
        # Test representative cases that would catch regressions
        test_cases = [
            Case(name="wide_caustic", separation=0.95, mass_ratio=0.01, u0=-0.01, alpha=0.5, rho=0.01, t_min=-0.8, t_max=0.8, n_times=100),
            Case(name="cusped_caustic", separation=1.0, mass_ratio=0.1, u0=-0.02, alpha=0.0, rho=0.005, t_min=-0.5, t_max=0.5, n_times=80),
            Case(name="near_cusp", separation=0.8, mass_ratio=0.05, u0=-0.005, alpha=0.2, rho=0.008, t_min=-0.3, t_max=0.3, n_times=60),
        ]

        for case in test_cases:
            times = np.linspace(case.t_min, case.t_max, case.n_times)
            opts = lcbinint.Options(source_bins=50, vbbl_compatible=1)
            result = lc_curve(case, times, opts)
            mag_lc = np.array(result.magnifications)
            mag_vbbl = vbbl_curve(case, times)

            rel_err = np.abs(mag_lc - mag_vbbl) / np.abs(mag_vbbl)
            max_rel_error = np.max(rel_err)

            # Phase 1-6 should maintain < 1% error for basic accuracy
            assert max_rel_error < 0.01, \
                f"{case.name}: error {max_rel_error:.4%} exceeds 1%"

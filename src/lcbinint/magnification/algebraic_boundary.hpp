#pragma once

#include "lcbinint/types.hpp"

#include <string>
#include <utility>
#include <vector>

namespace lcbinint::magnification {

struct AlgebraicBoundarySettings {
    double absolute_tolerance = 1.0e-8;
    double relative_tolerance = 1.0e-6;
    int maximum_radial_depth = 26;
    int maximum_band_search_steps = 64;
    double unit_circle_tolerance = 2.0e-7;
    double boundary_residual_tolerance = 2.0e-9;
};

struct AlgebraicBoundaryDiagnostics {
    int root_solve_count = 0;
    int lens_evaluation_count = 0;
    int radial_band_count = 0;
    int radial_subdivision_count = 0;
    int rejected_root_count = 0;
    int tangent_root_count = 0;
    int warm_root_solve_count = 0;
    int cold_root_solve_count = 0;
    int cold_retry_count = 0;
    int newton_prediction_count = 0;
    int recovered_seed_support_count = 0;
    int real_quartic_solve_count = 0;
    int real_quartic_fallback_count = 0;
    int real_sextic_solve_count = 0;
    int real_sextic_fallback_count = 0;
    int tangency_solve_count = 0;
    int tangency_fallback_count = 0;
    // Set when the VBM-style multipole test resolved the epoch and the band
    // discovery / radial quadrature were skipped entirely.
    int multipole_shortcut_count = 0;
};

struct AlgebraicBoundaryResult {
    double magnification = 0.0;
    double error_estimate = 0.0;
    bool success = false;
    std::string unsafe_reason;
    AlgebraicBoundaryDiagnostics diagnostics;
};

struct AlgebraicBoundaryLens {
    // Native binary coordinates: lenses at 0 and `lens_position`, with the
    // native source-frame shift -lens_position*m1 included in the map.
    double lens_position = 0.0;
    double m1 = 0.0;
    double m2 = 0.0;
};

struct AlgebraicBoundaryIntervals {
    bool success = false;
    bool full_circle = false;
    std::string unsafe_reason;
    std::vector<double> boundary_angles;
    // Cyclic intervals are unwrapped: the upper endpoint may exceed 2*pi.
    std::vector<std::pair<double, double>> inside_intervals;
};

struct AlgebraicCartesianIntervals {
    bool success = false;
    std::string unsafe_reason;
    std::vector<double> boundary_coordinates;
    std::vector<std::pair<double, double>> inside_intervals;
};

// Constant-first coefficients of the fixed-radius self-inversive quartic.
std::vector<Complex> algebraic_boundary_quartic(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_radius);

AlgebraicBoundaryIntervals algebraic_boundary_intervals_at_radius(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_radius,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics = nullptr);

struct AlgebraicBoundaryRadialBands {
    bool success = false;
    std::string unsafe_reason;
    // Merged, sorted [R_lo, R_hi] image-radius support bands.
    std::vector<std::pair<double, double>> bands;
};

// Additive read-only export of the primal's frozen band decomposition. Runs
// exactly the internal `find_radial_bands` the integral uses; no effect on the
// primal numerics. Consumed by the forward-mode Jacobian driver.
AlgebraicBoundaryRadialBands experimental_algebraic_boundary_radial_bands(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    const std::vector<SourcePosition>& certified_image_seeds,
    const AlgebraicBoundarySettings& settings = {});

AlgebraicBoundaryResult experimental_algebraic_boundary_integral(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double limb_darkening_c,
    double limb_darkening_d,
    const std::vector<SourcePosition>& certified_image_seeds,
    const AlgebraicBoundarySettings& settings = {});

// Constant-first coefficients of F(x_fixed, y). Coefficients are real up to
// floating-point roundoff; Complex is retained for the shared root solver.
std::vector<Complex> algebraic_cartesian_boundary_sextic(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x);

AlgebraicCartesianIntervals algebraic_cartesian_intervals_at_x(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double image_x,
    const AlgebraicBoundarySettings& settings,
    AlgebraicBoundaryDiagnostics* diagnostics = nullptr);

AlgebraicBoundaryResult experimental_algebraic_cartesian_integral(
    const AlgebraicBoundaryLens& lens,
    SourcePosition source,
    double source_radius,
    double limb_darkening_c,
    double limb_darkening_d,
    const std::vector<SourcePosition>& certified_image_seeds,
    const AlgebraicBoundarySettings& settings = {});


} // namespace lcbinint::magnification

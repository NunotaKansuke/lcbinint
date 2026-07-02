#pragma once

#include "lcbinint/model/triple_lens_geometry.hpp"
#include "lcbinint/types.hpp"

#include <limits>
#include <string>
#include <vector>

namespace lcbinint::magnification {

class PointSourceMagnifier;

enum class FiniteSourceMethod {
    point_source,
    hexadecapole,
    inverse_ray_cartesian,
    inverse_ray_polar,
    inverse_ray_spine,
    source_plane_quadrature,
};

struct FiniteSourceSettings {
    int source_bins = 50;
    int caustic_bins = 1400;
    double grid_ratio = 4.0;
    int polar_source_bins = 0;
    double polar_grid_ratio = 0.0;
    int finite_mode = 1;       // 1 = cartesian, 2 = polar, 3 = experimental spine, 4 = auto cartesian/polar
    double kinji_threshold = 20.0;   // bbox margin for fast-PS exit (in units of rho)
    double hex_threshold = 3.0;      // unused when adaptive_hex_threshold > 0
    double adaptive_hex_threshold = 0.001;  // VBM-style: |a4 correction|/mag > this → IR
    double limb_darkening_c = 0.0;
    double limb_darkening_d = 0.0;
    int adaptive_source_bins = 0;
    int max_source_bins = 400;
    double finite_source_tol = 0.0;
    double finite_source_reltol = 0.0;
};

struct FiniteSourceDecision {
    FiniteSourceMethod method = FiniteSourceMethod::point_source;
    int estimated_evaluations = 0;
    std::string reason;
};

struct FiniteSourceResult {
    double magnification = 0.0;
    int image_count = 0;
    FiniteSourceDecision decision;
    double error_estimate = 0.0;
    int refinement_level = 0;
    bool converged = true;
};

struct HexadecapoleDiagnosticResult {
    double magnification = 0.0;
    double relative_error = 0.0;
    double derivative_relative_error = 0.0;
};

class FiniteSourceMagnifier {
public:
    explicit FiniteSourceMagnifier(FiniteSourceSettings settings);

    const FiniteSourceSettings& settings() const { return settings_; }

    FiniteSourceResult binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double point_source_magnification,
        const std::vector<SourcePosition>* center_image_seeds = nullptr,
        bool point_source_magnification_is_exact = false,
        const PointSourceMagnifier* point_magnifier_hint = nullptr) const;
    FiniteSourceResult triple_mag(
        const model::TripleLensGeometry& geometry,
        SourcePosition source,
        double source_radius,
        double point_source_magnification,
        const PointSourceMagnifier* point_magnifier_hint = nullptr) const;
    void ensure_limb_darkening_table() const;
    void augment_seeds_from_caustic_branches(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        std::vector<SourcePosition>& seeds) const;
    double limb_darkening_table_brightness(double normalized_radius2) const;
    // Phase-ordered caustic branch polylines for (separation, mass_ratio),
    // built once per lens geometry.  Seed generation walks these instead of
    // re-solving the critical-curve polynomial per source position.
    const std::vector<std::vector<SourcePosition>>& binary_caustic_branches(
        double separation,
        double mass_ratio) const;

private:
    void ensure_binary_caustic_cache(double separation, double mass_ratio) const;
    double binary_caustic_distance(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double hint_nearest_point_dist = std::numeric_limits<double>::infinity()) const;
    double binary_sampled_caustic_distance(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double search_radius) const;
    FiniteSourceResult inverse_ray_polar_binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double caustic_distance) const;

    FiniteSourceSettings settings_;
    mutable bool caustic_cache_valid_ = false;
    mutable double caustic_cache_separation_ = 0.0;
    mutable double caustic_cache_mass_ratio_ = 0.0;
    mutable int caustic_cache_bins_ = 0;
    mutable std::vector<std::vector<SourcePosition>> caustic_cache_branches_;
    mutable std::vector<SourcePosition> caustic_cache_points_;
    mutable double caustic_cache_min_x_ = 0.0;
    mutable double caustic_cache_max_x_ = 0.0;
    mutable double caustic_cache_min_y_ = 0.0;
    mutable double caustic_cache_max_y_ = 0.0;
    mutable double caustic_cache_grid_step_x_ = 1.0;
    mutable double caustic_cache_grid_step_y_ = 1.0;
    mutable int caustic_cache_grid_size_ = 128;
    mutable std::vector<std::vector<int>> caustic_cache_grid_;
    struct CausticSegRef { int branch; int pos; };
    mutable std::vector<std::vector<CausticSegRef>> caustic_cache_branch_grid_;
    mutable double caustic_cache_max_seg_len_ = 0.0;
    mutable bool result_cache_valid_ = false;
    mutable double result_cache_separation_ = 0.0;
    mutable double result_cache_mass_ratio_ = 0.0;
    mutable double result_cache_source_x_ = 0.0;
    mutable double result_cache_source_y_ = 0.0;
    mutable double result_cache_source_radius_ = 0.0;
    mutable double result_cache_point_magnification_ = 0.0;
    mutable FiniteSourceResult result_cache_;
    mutable bool limb_darkening_table_valid_ = false;
    mutable double limb_darkening_table_c_ = 0.0;
    mutable double limb_darkening_table_d_ = 0.0;
    mutable std::vector<double> limb_darkening_table_;
};

const char* finite_source_method_name(FiniteSourceMethod method);
HexadecapoleDiagnosticResult diagnostic_hexadecapole_binary(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings);

} // namespace lcbinint::magnification

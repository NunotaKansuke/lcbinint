#pragma once

#include "lcbinint/model/triple_lens_geometry.hpp"
#include "lcbinint/magnification/algebraic_boundary.hpp"
#include "lcbinint/types.hpp"

#include <cstddef>
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

// Canonical, engine-neutral finite-source geometry for one source position:
// the trajectory-resolved binary-lens configuration, with no root-solving
// and no knowledge of any integration strategy. Useful to any external
// caller that wants to implement its own finite-source integration.
struct FiniteSourceGeometry {
    double separation = 0.0;
    double mass_ratio = 0.0;
    SourcePosition source;
    double source_radius = 0.0;
    double limb_darkening_c = 0.0;
    double limb_darkening_d = 0.0;
    double absolute_tolerance = 0.0;
    double relative_tolerance = 0.0;
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
    bool automatic_source_bins = true;
    int max_source_bins = 400;
    double finite_source_tol = 0.0;
    double finite_source_reltol = 0.0;
};

struct BinaryResolutionSelection {
    int source_bins = 50;
    bool prefer_polar = false;
};

// Internal calibration helpers.  They operate per source position and do not
// call or depend on any external finite-source implementation.
BinaryResolutionSelection calibrated_binary_resolution(
    double mass_ratio,
    double source_radius,
    double caustic_distance,
    double point_source_magnification,
    double limb_darkening_c,
    double requested_relative_tolerance,
    int maximum_bins);

// Conservative triple-lens auto resolution selector.  It uses a fixed,
// bounded resolution; "auto" never performs probes or hidden refinement.
BinaryResolutionSelection calibrated_triple_resolution(
    const model::TripleLensGeometry& geometry,
    double source_radius,
    double caustic_distance,
    double point_source_magnification,
    double limb_darkening_c,
    double requested_relative_tolerance,
    int maximum_bins);

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
    double point_source_quadrupole_indicator = 0.0;
    double point_source_cusp_indicator = 0.0;
    double point_source_ghost_indicator = 0.0;
    double point_source_planetary_distance2 = std::numeric_limits<double>::infinity();
    double point_source_safety_tolerance = 0.0;
    int point_source_ghost_count = 0;
    int point_source_safety_flags = 0;
    double caustic_distance = std::numeric_limits<double>::infinity();
};

struct BinaryRoutingDiagnostics {
    double point_magnification = 0.0;
    double point_error_estimate = std::numeric_limits<double>::infinity();
    double point_absolute_tolerance = 0.0;
    double caustic_distance = std::numeric_limits<double>::infinity();
    double scan_min_distance = std::numeric_limits<double>::infinity();
    double quadrupole_indicator = 0.0;
    double cusp_indicator = 0.0;
    double ghost_indicator = 0.0;
    double planetary_distance2 = std::numeric_limits<double>::infinity();
    int image_count = 0;
    int ghost_count = 0;
    int safety_flags = 0;
    bool point_preflight_safe = false;
    bool point_safe = false;
    bool scan_performed = false;
    bool any_vertex_inside = false;
    bool has_crossing_probes = false;
    bool chord_band = false;
    bool tangent_band = false;
    bool grazing_ring_band = false;
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
    // Phase-ordered physical caustic curves for (separation, mass_ratio),
    // reconstructed from the root monodromy and built once per lens geometry.
    // Seed generation walks these instead of re-solving the critical-curve
    // polynomial per source position.
    const std::vector<std::vector<SourcePosition>>& binary_caustic_branches(
        double separation,
        double mass_ratio) const;
    std::vector<std::vector<SourcePosition>> binary_critical_curve_branches(
        double separation,
        double mass_ratio) const;
    std::vector<std::vector<SourcePosition>> triple_caustic_branches(
        const model::TripleLensGeometry& geometry) const;
    std::vector<std::vector<SourcePosition>> triple_critical_curve_branches(
        const model::TripleLensGeometry& geometry) const;
    double triple_caustic_distance_for_source(
        const model::TripleLensGeometry& geometry,
        SourcePosition source,
        double refine_within = std::numeric_limits<double>::infinity()) const;
    double binary_caustic_distance_for_source(
        double separation,
        double mass_ratio,
        SourcePosition source) const;
    BinaryRoutingDiagnostics binary_routing_diagnostics_for_source(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double point_source_magnification,
        const PointSourceMagnifier* point_magnifier_hint = nullptr) const;
    // Isolated native experiment. It is deliberately not wired into binary_mag,
    // the public C API, JAX, or automatic routing.
    AlgebraicBoundaryResult experimental_algebraic_boundary_binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        const AlgebraicBoundarySettings& algebraic_settings = {}) const;
    double experimental_raster_polar_binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius) const;
    AlgebraicBoundaryResult experimental_algebraic_cartesian_binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        const AlgebraicBoundarySettings& algebraic_settings = {}) const;

private:
    void ensure_binary_caustic_cache(
        double separation, double mass_ratio, double separation_tolerance = 0.0) const;
    double binary_caustic_distance(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double hint_nearest_point_dist = std::numeric_limits<double>::infinity(),
        double separation_tolerance = 0.0) const;
    double binary_sampled_caustic_distance(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double search_radius,
        double separation_tolerance = 0.0) const;
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

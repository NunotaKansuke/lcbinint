#pragma once

#include "lcbinint/model/triple_lens_geometry.hpp"
#include "lcbinint/types.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lcbinint::holonomic {
struct PreparedEpochGeometry;
}

namespace lcbinint::magnification {

class PointSourceMagnifier;

// Allocate an empty rolling PreparedEpochGeometry for a trajectory that will
// evaluate FiniteSourceMethod::holonomic_binary across many nearby epochs.
// Attach the returned handle to every holonomic MagnificationExecutionPlan
// entry for that trajectory (MagnificationExecutionPlan::prepared_geometry) so
// consecutive epochs reuse the D14 root set / cell plan (L2 warm-seeded D14,
// exact-parity, fail-closed -- checkpoint sec.21 / Phase C step 2 part 3).
std::shared_ptr<holonomic::PreparedEpochGeometry> make_prepared_epoch_geometry();

enum class FiniteSourceMethod {
    point_source,
    hexadecapole,
    inverse_ray_cartesian,
    inverse_ray_polar,
    inverse_ray_spine,
    source_plane_quadrature,
    // ATPT holonomic binary finite-source engine (fused value + 5-Jacobian
    // radial pass; header-only, src/lcbinint/magnification/holonomic/).  Wired
    // through binary_mag_preplanned only -- selected by LightCurve.warmup()
    // where the engine is reference-validated and faster (Phase C step 2).
    // The ordinary routing cascade never picks it; a non-OK holonomic Status
    // fails closed (converged = false), never a silent approximation.
    holonomic_binary,
    // Result method for the caustic-seeded radial-frontier value integrator.
    // holonomic_auto is a request-only sentinel that preserves the ordinary
    // point/hex/integral selector and replaces only its integral branch.
    holonomic_seeded_radial,
    holonomic_auto,
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
bool binary_auto_tolerance_supported(
    double requested_absolute_tolerance,
    double requested_relative_tolerance);

BinaryResolutionSelection calibrated_binary_resolution(
    double mass_ratio,
    double source_radius,
    double caustic_distance,
    double point_source_magnification,
    double limb_darkening_c,
    int finite_mode,
    double requested_absolute_tolerance,
    double requested_relative_tolerance,
    int maximum_bins);

// Conservative triple-lens auto resolution selector.  It supplies a
// continuous, bounded initial grid; the finite-source caller independently
// checks a nested grid and may grow it up to maximum_bins.
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
    bool seed_support_proven = false;
    bool frontier_discovery_completed = false;
    bool integral_support_certified = false;
    bool value_converged = false;
    int center_image_count = 0;
    int augmented_seed_count = 0;
    std::string diagnostic_stop;
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
        const PointSourceMagnifier* point_magnifier_hint = nullptr,
        bool use_seeded_radial_frontier = false,
        double holonomic_mu_atol = std::numeric_limits<double>::quiet_NaN(),
        double holonomic_mu_rtol = std::numeric_limits<double>::quiet_NaN(),
        bool barycentric_frame = false) const;
    // Execute a route selected and reference-validated by LightCurve.warmup().
    // The caller may reuse the plan across a nearby inference neighbourhood;
    // this entry deliberately skips the normal routing cascade.
    FiniteSourceResult binary_mag_preplanned(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double point_source_magnification,
        FiniteSourceMethod method,
        int resolution,
        const std::vector<SourcePosition>* center_image_seeds = nullptr,
        const PointSourceMagnifier* point_magnifier_hint = nullptr,
        holonomic::PreparedEpochGeometry* rolling_prepared = nullptr) const;
    FiniteSourceResult triple_mag(
        const model::TripleLensGeometry& geometry,
        SourcePosition source,
        double source_radius,
        double point_source_magnification,
        const PointSourceMagnifier* point_magnifier_hint = nullptr) const;
    void ensure_limb_darkening_table() const;
    // Retain the complete, phase-ordered image seed set for an exact binary
    // epoch. Repeated LightCurve calls commonly revisit the same trajectory;
    // those calls can skip all centre, caustic, boundary, and certificate
    // probes while preserving the original seed order.
    std::vector<SourcePosition> cached_binary_image_seeds(
        const PointSourceMagnifier& point_magnifier,
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double caustic_distance,
        const std::vector<SourcePosition>* center_image_seeds,
        bool* support_proven) const;
    // Start a cold binary-epoch diagnostic while retaining the reusable
    // limb-darkening table. The normal production path keeps these caches.
    void clear_binary_epoch_caches() const;
    double limb_darkening_table_brightness(double normalized_radius2) const;
    const double* limb_darkening_table_data() const
    {
        return limb_darkening_table_.data();
    }
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
    // Cached view used by JAX support discovery.  The returned reference is
    // valid until this thread requests a different geometry or bin count.
    const std::vector<std::vector<SourcePosition>>&
    triple_caustic_branches_cached(
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
    // Lower bound from a cover of every continuous critical-curve phase.
    // Zero means no clearance was proved; it is never an estimated distance.
    double binary_caustic_clearance_lower_bound(
        double separation, double mass_ratio, SourcePosition source) const;

private:
    bool binary_point_topology_clearance(
        double separation, double mass_ratio, SourcePosition source,
        double source_radius) const;
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
    mutable bool caustic_enclosure_attempted_ = false;
    mutable double caustic_enclosure_separation_ = 0.0;
    mutable double caustic_enclosure_mass_ratio_ = 0.0;
    mutable std::vector<std::pair<SourcePosition, double>> caustic_enclosure_;
    struct BinarySeedCacheEntry {
        double separation;
        double mass_ratio;
        SourcePosition source;
        double source_radius;
        double caustic_distance;
        bool has_center_image_seeds;
        std::vector<SourcePosition> seeds;
        bool support_proven;
    };
    mutable std::vector<BinarySeedCacheEntry> binary_seed_cache_;
    mutable bool result_cache_valid_ = false;
    mutable double result_cache_separation_ = 0.0;
    mutable double result_cache_mass_ratio_ = 0.0;
    mutable double result_cache_source_x_ = 0.0;
    mutable double result_cache_source_y_ = 0.0;
    mutable double result_cache_source_radius_ = 0.0;
    mutable double result_cache_point_magnification_ = 0.0;
    mutable bool result_cache_seeded_radial_frontier_ = false;
    mutable double result_cache_holonomic_mu_atol_ = 0.0;
    mutable double result_cache_holonomic_mu_rtol_ = 0.0;
    mutable bool result_cache_barycentric_frame_ = false;
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

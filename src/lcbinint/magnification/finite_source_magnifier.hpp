#pragma once

#include "lcbinint/model/triple_lens_geometry.hpp"
#include "lcbinint/types.hpp"
#include "lcbinint/magnification/algebraic_boundary.hpp"
#include "lcbinint/magnification/algebraic_boundary_autodiff.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
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
    // Result method for the algebraic image-boundary finite-source integrator.
    // The request sentinel preserves point/hex routing and replaces the terminal
    // inverse-ray solve with the exact image-boundary integral.
    algebraic_boundary = 7, // Keep diagnostic method IDs stable; 6 is retired.
    algebraic_boundary_auto,
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

// Physical tangent, with the same coordinates as triple::Frame::pullback.
struct TripleAlgebraicGradientResult {
    FiniteSourceResult value;
    double point_magnification = std::numeric_limits<double>::quiet_NaN();
    std::array<double,11> gradient{};
    bool gradient_valid = false;
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

    // Radial-origin choice of the binary algebraic-boundary route.  The
    // kernel's radial variable is v = |z - lens0|^2 with lens0 the mass
    // 1/(1+q) lens, so q < 1 puts the origin at the heavy lens and q > 1 at
    // the light lens.  Both give the same mu through the exact symmetry
    //   mu(x, y, rho, q, s) = mu(-x, -y, rho, 1/q, s)   (centre-of-mass frame),
    // and the route picks the origin itself, reflecting the call and pulling
    // the Jacobian back.  The two origins cost differently:
    //  - heavy origin: images are round about the heavy lens, so the radial
    //    quadrature needs 5-30% fewer nodes (up to 1.6x faster at large rho);
    //  - but when the heavy-lens critical loop is nearly a circle (small q)
    //    and rho is small, the images hug that Einstein ring, the D14 roots
    //    are near-multiple and root isolation explodes (4-10x slower, some
    //    "full radial retries produced no finite ..." throws).
    // The measured switch (two surveys, 724 (s, q, rho) cells, 40
    // near-caustic epochs each) is
    //   heavy origin  iff  max(S_heavy(s, q), rho) >= 0.03,
    // S_heavy = relative spread (max - min)/max of |z - heavy|^2 over the
    // critical-curve points nearer the heavy lens.  It had no failures and
    // was within 3-5% of the per-cell best (always-light: 10-12%).  Over the
    // epochs this picks, the heavy origin also had fewer Jacobian outliers
    // (error > 1e-3 of the largest component at rtol 1e-4: 16 vs 29 of
    // 3440).  The exception is a source limb passing over the heavy lens:
    // there the heavy origin's Jacobian is worse (15 vs 9 of 64 limb-crossing
    // epochs, q <= 1e-3), so epochs with
    //   | |source - heavy| - rho | < 0.05 rho
    // use the light origin.  When the disk covers the heavy lens, some
    // image-plane circles about the heavy lens map entirely inside the
    // source.  For a uniform source their angular integral is exactly 2 pi;
    // a limb-darkened one integrates sqrt(phi) over them with a periodic
    // trapezoid capped at 256 nodes (mapped_full_circle).  With the planet
    // near the heavy lens's Einstein ring (s ~ 1) these circles pass close to
    // it, sqrt(phi) gets a narrow bump the cap cannot resolve (~1e-3 per
    // circle), and the heavy-origin integral stops InnerAccuracyLimited
    // (value_converged false; 2-3x the light-origin time), so limb-darkened
    // epochs with
    //   |source - heavy| < 1.05 rho
    // use the light origin up front.  A frame switch rebuilds the caustic
    // cache (~1.3 ms), and a light curve crosses that band at most a few
    // times.  A heavy-origin call that throws, returns a non-finite value,
    // or returns an algebraic-boundary value that did not converge is
    // retried at the light origin (the heavy result is kept if the light one
    // does not converge either).
    bool binary_uses_heavy_origin(
        double separation, double mass_ratio, SourcePosition source,
        double source_radius) const;
    // S_heavy(s, q) above (cached for the last (|s|, min(q, 1/q))).
    double binary_heavy_origin_spread(double separation, double mass_ratio) const;

    bool has_binary_caustic_geometry(double separation, double mass_ratio) const {
        return caustic_cache_valid_ && caustic_cache_separation_ == separation &&
            caustic_cache_mass_ratio_ == mass_ratio;
    }

    FiniteSourceResult binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double point_source_magnification,
        const std::vector<SourcePosition>* center_image_seeds = nullptr,
        bool point_source_magnification_is_exact = false,
        const PointSourceMagnifier* point_magnifier_hint = nullptr,
        bool use_algebraic_boundary = false,
        double algebraic_boundary_atol = std::numeric_limits<double>::quiet_NaN(),
        double algebraic_boundary_rtol = std::numeric_limits<double>::quiet_NaN(),
        bool barycentric_frame = false,
        bool multipole_only_probe = false,
        autodiff::AlgebraicBoundaryJacobian* jacobian_out = nullptr,
        std::array<std::vector<BinaryImage>, 13>* image_stencil = nullptr) const;
    AlgebraicBoundaryResult algebraic_boundary_binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        const AlgebraicBoundarySettings& algebraic_settings = {},
        FiniteSourceResult* value_result = nullptr,
        double* point_source_magnification = nullptr,
        bool barycentric_frame = false) const;
    autodiff::AlgebraicBoundaryJacobian algebraic_boundary_binary_jacobian(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        const AlgebraicBoundarySettings& algebraic_settings = {},
        FiniteSourceResult* value_result = nullptr,
        double* point_source_magnification = nullptr,
        bool barycentric_frame = false,
        const PointSourceMagnifier* point_hint = nullptr) const;
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
        const PointSourceMagnifier* point_magnifier_hint = nullptr) const;
    FiniteSourceResult algebraic_boundary_triple_mag(
        const model::TripleLensGeometry& geometry, SourcePosition source,
        double source_radius, const AlgebraicBoundarySettings& settings,
        const PointSourceMagnifier* point_hint = nullptr,
        double known_point_magnification = std::numeric_limits<double>::quiet_NaN()) const;
    TripleAlgebraicGradientResult algebraic_boundary_triple_jacobian(
        const model::TripleLensGeometry& geometry, SourcePosition source,
        double source_radius, const AlgebraicBoundarySettings& settings,
        const PointSourceMagnifier* point_hint = nullptr) const;

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
        bool* support_proven,
        bool continue_probe_roots = false) const;
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
        double separation, double mass_ratio, SourcePosition source,
        double separation_half_width = 0.0,
        double* enclosure_clearance = nullptr) const;

private:
    // binary_mag / algebraic_boundary_binary_jacobian in the frame given,
    // without the radial-origin choice.
    FiniteSourceResult binary_mag_in_frame(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double point_source_magnification,
        const std::vector<SourcePosition>* center_image_seeds,
        bool point_source_magnification_is_exact,
        const PointSourceMagnifier* point_magnifier_hint,
        bool use_algebraic_boundary,
        double algebraic_boundary_atol,
        double algebraic_boundary_rtol,
        bool barycentric_frame,
        bool multipole_only_probe,
        autodiff::AlgebraicBoundaryJacobian* jacobian_out,
        std::array<std::vector<BinaryImage>, 13>* image_stencil) const;
    autodiff::AlgebraicBoundaryJacobian algebraic_boundary_binary_jacobian_in_frame(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        const AlgebraicBoundarySettings& algebraic_settings,
        FiniteSourceResult* value_result,
        double* point_source_magnification,
        bool barycentric_frame,
        const PointSourceMagnifier* point_hint) const;
    bool binary_caustic_enclosure_matches(double separation, double mass_ratio) const;
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
    FiniteSourceResult algebraic_boundary_deep_solve(
        const PointSourceMagnifier& point_magnifier,
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        const std::vector<SourcePosition>& center_image_seeds,
        const AlgebraicBoundarySettings& algebraic_settings,
        bool barycentric_frame,
        autodiff::AlgebraicBoundaryJacobian* jacobian_out) const;
    FiniteSourceResult inverse_ray_polar_binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double caustic_distance) const;

    FiniteSourceSettings settings_;
    mutable bool origin_spread_cache_valid_ = false;
    mutable double origin_spread_cache_separation_ = 0.0;
    mutable double origin_spread_cache_mass_ratio_ = 0.0;
    mutable double origin_spread_cache_value_ = 0.0;
    mutable bool binary_separation_seen_ = false;
    mutable bool binary_separation_changes_ = false;
    mutable double binary_last_separation_ = 0.0;
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
    mutable double caustic_enclosure_half_width_ = 0.0;
    mutable bool caustic_interval_disabled_ = false;
    mutable std::vector<std::pair<SourcePosition, double>> caustic_enclosure_;
    mutable SourcePosition caustic_clearance_source_ {};
    mutable double caustic_clearance_witness_ = 0.0;
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
    // Each certificate probe continues from the corresponding previous-epoch
    // solve. Every warm root set still passes the ordinary validation/cold retry.
    mutable std::vector<PointSourceMagnifier> binary_probe_magnifiers_;
    mutable bool result_cache_valid_ = false;
    mutable double result_cache_separation_ = 0.0;
    mutable double result_cache_mass_ratio_ = 0.0;
    mutable double result_cache_source_x_ = 0.0;
    mutable double result_cache_source_y_ = 0.0;
    mutable double result_cache_source_radius_ = 0.0;
    mutable double result_cache_point_magnification_ = 0.0;
    mutable bool result_cache_algebraic_boundary_ = false;
    mutable double result_cache_algebraic_boundary_atol_ = 0.0;
    mutable double result_cache_algebraic_boundary_rtol_ = 0.0;
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

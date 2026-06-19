#pragma once

#include "lcbinint/types.hpp"

#include <string>
#include <vector>

namespace lcbinint::magnification {

enum class FiniteSourceMethod {
    point_source,
    hexadecapole,
    inverse_ray_cartesian,
    inverse_ray_polar,
};

enum class InverseRayMethod {
    auto_select,
    cartesian,
    polar,
};

struct FiniteSourceSettings {
    double tolerance = 1.0e-3;
    double relative_tolerance = 0.0;
    int source_bins = 20;
    int caustic_bins = 1400;
    double grid_ratio = 4.0;
    InverseRayMethod inverse_ray_method = InverseRayMethod::auto_select;
    bool legacy_mode = false;
    int legacy_finite_mode = 4;
    double legacy_kinji = 9.0;
    double legacy_hex = 2.0;
    double limb_darkening_c = 0.0;
    double limb_darkening_d = 0.0;
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

class FiniteSourceMagnifier {
public:
    explicit FiniteSourceMagnifier(FiniteSourceSettings settings);

    const FiniteSourceSettings& settings() const { return settings_; }

    FiniteSourceDecision choose_binary_method(
        SourcePosition source,
        double source_radius,
        double point_source_magnification) const;

    FiniteSourceResult binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius,
        double point_source_magnification) const;
    void ensure_limb_darkening_table() const;
    double limb_darkening_table_brightness(double normalized_radius2) const;

private:
    void ensure_legacy_caustic_cache(double separation, double mass_ratio) const;
    double legacy_binary_caustic_distance(
        double separation,
        double mass_ratio,
        SourcePosition source) const;
    double legacy_binary_sampled_caustic_distance(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double search_radius) const;
    void ensure_legacy_polar_map_cache(
        double separation,
        double mass_ratio,
        double source_radius) const;
    FiniteSourceResult legacy_polar_memory_binary_mag(
        double separation,
        double mass_ratio,
        SourcePosition source,
        double source_radius) const;

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
    mutable bool result_cache_valid_ = false;
    mutable double result_cache_separation_ = 0.0;
    mutable double result_cache_mass_ratio_ = 0.0;
    mutable double result_cache_source_x_ = 0.0;
    mutable double result_cache_source_y_ = 0.0;
    mutable double result_cache_source_radius_ = 0.0;
    mutable double result_cache_point_magnification_ = 0.0;
    mutable FiniteSourceResult result_cache_;
    mutable bool polar_map_cache_valid_ = false;
    mutable double polar_map_cache_separation_ = 0.0;
    mutable double polar_map_cache_mass_ratio_ = 0.0;
    mutable double polar_map_cache_source_radius_ = 0.0;
    mutable int polar_map_cache_source_bins_ = 0;
    mutable double polar_map_cache_grid_ratio_ = 0.0;
    mutable double polar_map_cache_dr_ = 1.0;
    mutable double polar_map_cache_dphi_ = 1.0;
    mutable int polar_map_cache_phi_bins_ = 0;
    mutable int polar_map_cache_radial_offset_min_index_ = 0;
    mutable std::vector<int> polar_map_cache_radial_offsets_;
    mutable std::vector<SourcePosition> polar_map_cache_;
    mutable bool limb_darkening_table_valid_ = false;
    mutable double limb_darkening_table_c_ = 0.0;
    mutable double limb_darkening_table_d_ = 0.0;
    mutable std::vector<double> limb_darkening_table_;
};

const char* finite_source_method_name(FiniteSourceMethod method);

} // namespace lcbinint::magnification

#pragma once

#include "lcbinint/types.hpp"

#include <string>

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

private:
    FiniteSourceSettings settings_;
};

const char* finite_source_method_name(FiniteSourceMethod method);

} // namespace lcbinint::magnification

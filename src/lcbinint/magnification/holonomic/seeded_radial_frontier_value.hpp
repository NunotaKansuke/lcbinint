#pragma once

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"

#include <limits>
#include <vector>

namespace lcbinint::holonomic {

struct CausticAwareBinarySeeds {
    std::vector<SourcePosition> center_images;
    std::vector<Cplx<double>> radial_seeds;
    bool support_proven = false;
};

// Shared by the radial-frontier value and gradient routes. The finite-source
// seed scan adds caustic images that can be absent at the source center.
inline CausticAwareBinarySeeds make_caustic_aware_binary_seeds(
    const magnification::FiniteSourceMagnifier& finite,
    const magnification::PointSourceMagnifier& point,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const std::vector<SourcePosition>* supplied_center_images = nullptr)
{
    CausticAwareBinarySeeds result;
    if (supplied_center_images != nullptr) {
        result.center_images = *supplied_center_images;
    } else {
        for (const auto& image : point.binary_images(
                 separation, mass_ratio, source)) {
            result.center_images.push_back(image.position);
        }
    }

    const auto positions = finite.cached_binary_image_seeds(
        point, separation, mass_ratio, source, source_radius,
        std::numeric_limits<double>::infinity(), &result.center_images,
        &result.support_proven);
    result.radial_seeds.reserve(positions.size());
    const double frame_shift = mass_ratio < 1.0 ? separation : 0.0;
    for (const auto& image : positions) {
        // The point solver's VBM-compatible frame has lenses at 0 and -s;
        // the holonomic kernel uses lenses at 0 and +s.
        result.radial_seeds.push_back({image.x + frame_shift, image.y});
    }
    return result;
}

inline SeededFrontierResult epoch_seeded_radial_frontier_value(
    const LensParams& params,
    double limb_darkening,
    const std::vector<Cplx<double>>& seeds,
    double absolute_tolerance,
    double relative_tolerance,
    AdaptiveWorkspace& workspace)
{
    AdaptiveConfig config;
    config.tol.mu_atol = absolute_tolerance;
    config.tol.mu_rtol = relative_tolerance;
    config.gradient_policy = GradientPolicy::None;
    config.with_jacobian = false;
    return epoch_seeded_radial_frontier(
        params, limb_darkening, seeds, config, workspace);
}

} // namespace lcbinint::holonomic

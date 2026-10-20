#pragma once

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/binary_caustic_tree.hpp"
#include "lcbinint/magnification/algebraic_boundary/seeded_radial_frontier.hpp"

#include "lcbinint/magnification/binary_local_seeds.hpp"

#include "lcbinint/magnification/algebraic_boundary/radial_support_plan.hpp"

#include <limits>
#include <vector>

namespace lcbinint::algebraic_boundary {

struct CausticAwareBinarySeeds {
    std::vector<SourcePosition> center_images;
    std::vector<Cplx<double>> radial_seeds;
    bool support_proven = false;
    RadialSupportPlan support_plan;
};

// Shared physical image seeds and optional radial support for both API routes.
inline CausticAwareBinarySeeds make_caustic_aware_binary_seeds(
    const magnification::FiniteSourceMagnifier& finite,
    const magnification::PointSourceMagnifier& point,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const std::vector<SourcePosition>* supplied_center_images = nullptr,
    bool allow_radial_plan = true)
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

    const bool regular=!finite.has_binary_caustic_geometry(separation,mass_ratio) &&
        magnification::detail::binary_caustic_disk_clearance(
        separation,mass_ratio,source,source_radius,source_radius)>source_radius;
    const auto positions = [&]() {
        if(regular)return result.center_images;
        if(allow_radial_plan && !finite.has_binary_caustic_geometry(separation,mass_ratio)) {
            result.support_plan=exact_support_detail::radial_plan(PrimaryFrame::from(
                {source.x,source.y,source_radius,mass_ratio,separation,true}));
            if(result.support_plan.completed) {
                result.support_proven=true;
                return result.center_images;
            }
        }
        if(!finite.has_binary_caustic_geometry(separation,mass_ratio)) {
            auto local=magnification::detail::binary_local_image_seeds(
                separation,mass_ratio,source,source_radius);
            if(local.completed) {
                auto positions=result.center_images;
                positions.insert(positions.end(),local.seeds.begin(),local.seeds.end());
                result.support_proven=true;
                return positions;
            }
        }
        if(!finite.has_binary_caustic_geometry(separation,mass_ratio)) {
            const auto pf=PrimaryFrame::from({source.x,source.y,source_radius,
                mass_ratio,separation,true});
            auto support=exact_support_detail::seeds(pf);
            if(support.completed) {
                auto positions=result.center_images;
                const double shift=mass_ratio<1.0?separation:0.0;
                for(const auto& z:support.seeds)positions.push_back({z[0]-shift,z[1]});
                result.support_proven=true;
                return positions;
            }
        }
        return finite.cached_binary_image_seeds(
        point, separation, mass_ratio, source, source_radius,
        std::numeric_limits<double>::infinity(), &result.center_images,
        &result.support_proven, true);
    }();
    if(regular)result.support_proven=true;
    result.radial_seeds.reserve(positions.size());
    const double frame_shift = mass_ratio < 1.0 ? separation : 0.0;
    for (const auto& image : positions) {
        // The point solver's VBM-compatible frame has lenses at 0 and -s;
        // the algebraic-boundary kernel uses lenses at 0 and +s.
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
    AdaptiveWorkspace& workspace,
    const RadialSupportPlan* support_plan=nullptr)
{
    AdaptiveConfig config;
    config.tol.mu_atol = absolute_tolerance;
    config.tol.mu_rtol = relative_tolerance;
    config.gradient_policy = GradientPolicy::None;
    config.with_jacobian = false;
    return epoch_seeded_radial_frontier(
        params, limb_darkening, seeds, config, workspace, false, support_plan);
}

} // namespace lcbinint::algebraic_boundary

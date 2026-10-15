#pragma once
#include "lcbinint/magnification/adaptive_route_features.hpp"
#include "lcbinint/magnification/adaptive_route_model.hpp"

namespace lcbinint::magnification::adaptive_detail {
struct AdaptiveRouteDecision {
    bool polar = true;
    bool within_training_domain = false;
};

// The centre roots should be shared with the subsequent preplanned call.
// Outside the fitted configuration use Polar without claiming a fitted speed.
inline AdaptiveRouteDecision choose_adaptive_route(
    double separation, double mass_ratio, double rho, double limb,
    double relative_tolerance, const std::vector<BinaryImage>& images)
{
    if (!(std::isfinite(separation) && separation>0 && std::isfinite(mass_ratio) && mass_ratio>0 &&
          std::isfinite(rho) && rho>0 && (limb==0 || limb==0.5) &&
          (relative_tolerance==1e-3 || relative_tolerance==1e-4) &&
          (images.size()==3 || images.size()==5))) return {};
    const double normalized_q=std::min(mass_ratio,1.0/mass_ratio);
    if(separation<0.20159474280522882 || separation>3.949369483067623 ||
       normalized_q<0.0001084095167119894 || normalized_q>0.9221703406929264 ||
       rho<3.135563210556899e-05 || rho>0.9509862925080927) return {};
    for(const auto& image:images)
        if(!(std::isfinite(image.position.x) && std::isfinite(image.position.y) &&
             std::isfinite(image.jacobian_determinant) && image.jacobian_determinant!=0)) return {};
    const auto features=adaptive_route_features(separation,mass_ratio,rho,limb,relative_tolerance,images);
    for(double f:features)if(!std::isfinite(f))return {};
    return {predict_adaptive_polar(features),true};
}
}

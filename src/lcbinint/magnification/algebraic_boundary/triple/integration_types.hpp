#pragma once
#include "radial_events.hpp"
#include "radial_panel.hpp"
namespace lcbinint::algebraic_boundary::triple {
struct IntegrationOptions {
    double absolute_tolerance=1e-6,relative_tolerance=1e-5;
    int event_box_budget=1000000,maximum_refinements=160;
    AngularOptions angular{};
    bool automatic_endpoint_tolerance=true;
    bool separate_components=true;
    bool moving_radial_gradients=true;
};
struct IntegrationDiagnostics {
    double magnification=std::numeric_limits<double>::quiet_NaN(),error_estimate=INFINITY;
    bool converged=false,support_complete=false;
    bool gradient_valid=false;
    std::string reason="not_started";
    std::size_t value_evaluations=0,parameter_derivative_evaluations=0,quad_samples=0;
    std::size_t continued_radii=0,global_radii=0;
    bool smooth_source_used=false;std::size_t smooth_source_samples=0,smooth_source_reused_samples=0;
    int refinements=0,h_splits=0,component_domains=1,moving_radial_panels=0;
    RadialEventPlan support;
};
template<bool Gradient=false> struct IntegrationResult:IntegrationDiagnostics {
    RadiusIntegral<Gradient> normalized;
};

}

#pragma once
#include "radial_panel.hpp"
namespace lcbinint::algebraic_boundary::triple {
// Optional value tape for diagnostics and historical replay comparisons.
// The production fused gradient path does not construct or consume it.
struct ValueIntegralTrace {
    struct Panel {
        std::unique_ptr<RadialPanel<false>> primal;
        AngularOptions options;
        std::optional<ImageExclusion> inclusion;
        std::vector<ImageExclusion> exclusions;
        explicit Panel(std::unique_ptr<RadialPanel<false>> value):primal(std::move(value)),options(primal->angular_options()) {
            if(options.inclusion)inclusion=*options.inclusion;
            if(options.exclusions)exclusions=*options.exclusions;
            options.inclusion=nullptr;options.exclusions=nullptr;
        }
        AngularOptions angular_options() const {
            auto out=options;out.inclusion=inclusion?&*inclusion:nullptr;
            out.exclusions=exclusions.empty()?nullptr:&exclusions;return out;
        }
    };
    double relative_tolerance=1e-5;
    std::vector<Panel> panels;
};
}

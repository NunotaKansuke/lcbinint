#include "fused_gradient.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
#include <algorithm>
#include <cmath>

namespace lcbinint::holonomic {
FusedGradientResult epoch_fused_gradient(const LensParams& p,double u,
    const std::vector<std::array<double,2>>& image_seeds,const FusedGradientOptions& options) {
    namespace kernel=lcbinint::holonomic_fused;
    const kernel::LensParams params{p.xs,p.ys,p.rho,p.q,p.a,p.barycentric};
    std::vector<kernel::Cplx<double>> seeds;
    seeds.reserve(image_seeds.size());
    for(const auto& z:image_seeds)seeds.push_back({z[0],z[1]});
    kernel::AdaptiveConfig cfg;
    cfg.tol.mu_atol=options.mu_atol;cfg.tol.mu_rtol=options.mu_rtol;
    cfg.with_jacobian=true;cfg.fused_value_mesh=true;
    cfg.gradient_policy=kernel::GradientPolicy::ValueFirst;
    cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;
    kernel::AdaptiveWorkspace workspace;
    kernel::fused_gradient::reliability_retry_used=false;
    auto result=kernel::epoch_seeded_radial_frontier(params,u,seeds,cfg,workspace);
    auto gradients_usable=[](const kernel::AdaptiveResult& integral) {
        if(!integral.value_converged)return false;
        for(int j=0;j<5;++j) {
            if(integral.grad_quality[j]==kernel::GradientQuality::Invalid ||
               integral.grad_quality[j]==kernel::GradientQuality::NotRequested ||
               !std::isfinite(integral.grad_mu[j]))return false;
        }
        return true;
    };
    bool unresolved_outer_edge_pair=false;
    const auto edge_pairs=kernel::geometry_experiment::known_upper_edge_pairs(
        kernel::PrimaryFrame::from(params),result.topology,
        std::max(1,(1<<cfg.initial_level)-1),&unresolved_outer_edge_pair);
    const bool interval_certified_edge_pair=
        !edge_pairs.empty() && !unresolved_outer_edge_pair;
    const auto initial_stop=result.integral.stop;
    const bool event_location_failure=
        initial_stop==kernel::AdaptiveStop::EventLocationLimited;
    // The certified D14 edge pair is already carried by the completed
    // frontier. Requiring the coarser frontier event list to rediscover it
    // loses precisely the close pair this route is meant to resolve. Keep
    // exact topology failures fail-closed unless the frontier itself provides
    // an interval-certified upper-edge pair. A usable incumbent or an
    // event-location-limited derivative can also use the regular chart.
    const bool isolated_edge_contact=result.frontier.discovery_completed &&
        result.frontier.isolated_real_edge_pair &&
        (gradients_usable(result.integral) || event_location_failure ||
         interval_certified_edge_pair);
    bool no_fold_retry=false;
    std::size_t total_nodes=result.integral.stats.node_evaluations;
    const bool topology_failure=event_location_failure ||
        initial_stop==kernel::AdaptiveStop::TopologyUnresolved;
    // A TopologyUnresolved retry can only recover the value; its gradient is
    // deliberately kept invalid. Skip that work when the original value has
    // already converged. EventLocationLimited still retries because the
    // regular map may recover a usable gradient.
    const bool retry_can_help=event_location_failure ||
        !result.integral.value_converged;
    if(isolated_edge_contact) {
        // A certified real close pair at the outer support edge makes the
        // fold-mapped derivative sharply localized. Resolve that rare geometry
        // on a regular radial chart at the caller's unchanged value tolerance.
        auto contact_cfg=cfg;
        contact_cfg.fold_maps=false;
        contact_cfg.fused_value_mesh=false;
        contact_cfg.material_coordinate_motion=false;
        contact_cfg.initial_level=8;
        // The regular radial chart uses the same value-driven replay policy as
        // epoch_adaptive; no gradient-error adaptation is enabled here.
        contact_cfg.gradient_adaptive=false;
        kernel::AdaptiveWorkspace contact_workspace;
        auto contact=kernel::epoch_adaptive(
            params,u,contact_cfg,contact_workspace);
        total_nodes+=contact.stats.node_evaluations;
        no_fold_retry=true;
        if(contact.value_converged && gradients_usable(contact) &&
           contact.stop==kernel::AdaptiveStop::Converged)
            result.integral=std::move(contact);
    }
    if(!no_fold_retry && topology_failure && retry_can_help &&
       !gradients_usable(result.integral)) {
        // Retry rare unresolved geometry with complete D14 topology and the
        // regular radial coordinate. The value can be recovered without a
        // certified event coordinate. A gradient replaces the first result
        // only for EventLocationLimited, where the fold endpoint was the sole
        // blocker; topology failures remain fail-closed for derivatives.
        auto fallback_cfg=cfg;
        fallback_cfg.fold_maps=false;
        // This route is entered only after an unresolved epoch. Start at the
        // finest supported rule to resolve the narrow event neighborhood
        // without changing the caller's value tolerances.
        fallback_cfg.initial_level=8;
        kernel::AdaptiveWorkspace fallback_workspace;
        auto fallback=kernel::epoch_adaptive(
            params,u,fallback_cfg,fallback_workspace);
        total_nodes+=fallback.stats.node_evaluations;
        no_fold_retry=true;
        if(event_location_failure && !unresolved_outer_edge_pair &&
           gradients_usable(fallback)) {
            result.integral=std::move(fallback);
        } else if(fallback.value_converged) {
            // Preserve the original gradient status and values. A finite
            // no-fold derivative at a topology failure is not enough to
            // establish a classical derivative at a caustic-limb contact.
            result.integral.mu=fallback.mu;
            result.integral.value_converged=true;
        }
    }
    FusedGradientResult out;
    out.reliability_retried=no_fold_retry ||
        kernel::fused_gradient::reliability_retry_used;
    out.mu=result.integral.mu;out.grad_mu=result.integral.grad_mu;
    out.value_converged=result.integral.value_converged;
    out.stop=kernel::adaptive_stop_name(result.integral.stop);
    out.support_certified=result.frontier.coverage_certified;
    out.node_evaluations=total_nodes;
    for(int j=0;j<5;++j)out.gradient_valid[j]=
        result.integral.grad_quality[j]!=kernel::GradientQuality::Invalid &&
        result.integral.grad_quality[j]!=kernel::GradientQuality::NotRequested &&
        std::isfinite(out.grad_mu[j]);
    return out;
}
}

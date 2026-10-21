#include "fused_gradient.hpp"
#include "tangent/seeded_radial_frontier.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace lcbinint::algebraic_boundary {
PreciseValueResult epoch_precise_value(const LensParams& p,double u,
    const std::vector<std::array<double,2>>& image_seeds,
    double absolute_tolerance,double relative_tolerance) {
    namespace kernel=lcbinint::algebraic_boundary::tangent;
    const kernel::LensParams params{p.xs,p.ys,p.rho,p.q,p.a,p.barycentric};
    std::vector<kernel::Cplx<double>> seeds;
    seeds.reserve(image_seeds.size());
    for(const auto& image:image_seeds)seeds.push_back({image[0],image[1]});
    kernel::AdaptiveConfig config;
    config.tol.mu_atol=absolute_tolerance;config.tol.mu_rtol=relative_tolerance;
    config.with_jacobian=false;
    config.gradient_policy=kernel::GradientPolicy::None;
    config.fold_maps=false;
    kernel::AdaptiveWorkspace workspace;
    kernel::fused_gradient::PreciseTangentScope precision;
    const auto result=kernel::epoch_seeded_radial_frontier(
        params,u,seeds,config,workspace,false,nullptr,true);
    PreciseValueResult out;
    out.mu=result.integral.mu;
    out.value_error_estimate=result.integral.estimated_abs_error_mu;
    out.value_converged=result.integral.value_converged;
    out.support_certified=result.frontier.coverage_certified;
    out.frontier_discovery_completed=result.frontier.discovery_completed;
    out.node_evaluations=result.integral.stats.node_evaluations;
    out.stop=kernel::adaptive_stop_name(result.integral.stop);
    if(!result.frontier.discovery_completed) {
        out.stop+="; frontier=";
        out.stop+=result.frontier.reason;
    }
    return out;
}

FusedGradientResult epoch_fused_gradient(const LensParams& p,double u,
    const std::vector<std::array<double,2>>& image_seeds,const FusedGradientOptions& options,
    const std::function<bool(FusedGradientResult&)>& smooth_repair,
    const RadialSupportPlan* support_plan,
    const std::vector<std::array<double,2>>* inherited_edge_pairs) {
    namespace kernel=lcbinint::algebraic_boundary::tangent;
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
    std::vector<kernel::geometry_experiment::Pair> retry_edge_pairs;
    if(inherited_edge_pairs)for(const auto& radii:*inherited_edge_pairs) {
        kernel::geometry_experiment::Pair pair;
        pair.left=radii[0];pair.right=radii[1];
        pair.center=.5*(pair.left+pair.right);pair.scale=.5*(pair.right-pair.left);
        pair.real=true;pair.upper_edge=true;
        retry_edge_pairs.push_back(pair);
    }
    const auto remember_edge_pairs=[&](const auto& pairs) {
        for(const auto& pair:pairs) {
            bool known=false;
            for(const auto& old:retry_edge_pairs)
                if(old.left==pair.left && old.right==pair.right)known=true;
            if(!known)retry_edge_pairs.push_back(pair);
        }
    };
    auto result=kernel::epoch_seeded_radial_frontier(params,u,seeds,cfg,workspace,false,
        support_plan,!retry_edge_pairs.empty(),&retry_edge_pairs);
    remember_edge_pairs(result.edge_pairs);
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
        !retry_edge_pairs.empty() ||
        (!edge_pairs.empty() && !unresolved_outer_edge_pair);
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
    const bool retry_can_help=event_location_failure ||
        !result.integral.value_converged;
    bool uncertified_candidate_used=false;
    std::string candidate_reason;
    const auto export_result=[&]() {
        FusedGradientResult out;
        out.reliability_retried=no_fold_retry ||
            kernel::fused_gradient::reliability_retry_used;
        out.mu=result.integral.mu;out.grad_mu=result.integral.grad_mu;
        out.value_error_estimate=result.integral.estimated_abs_error_mu;
        out.frontier_discovery_completed=result.frontier.discovery_completed;
        out.value_converged=result.integral.value_converged;
        out.stop=kernel::adaptive_stop_name(result.integral.stop);
        if(!result.frontier.discovery_completed) {
            out.stop+="; frontier=";
            out.stop+=result.frontier.reason;
        }
        if(uncertified_candidate_used) {
            out.stop+="; gradient_candidate=regular_radial; gradient_uncertified="+candidate_reason;
            out.stop+="; initial_stop=";
            out.stop+=kernel::adaptive_stop_name(initial_stop);
        }
        out.support_certified=result.frontier.coverage_certified;
        out.node_evaluations=total_nodes;
        for(const auto& pair:retry_edge_pairs)out.radial_edge_pairs.push_back({pair.left,pair.right});
        for(int j=0;j<5;++j)out.gradient_valid[j]=
            result.frontier.discovery_completed &&
            !uncertified_candidate_used &&
            result.integral.grad_quality[j]!=kernel::GradientQuality::Invalid &&
            result.integral.grad_quality[j]!=kernel::GradientQuality::NotRequested &&
            std::isfinite(out.grad_mu[j]);
        return out;
    };
    const auto retain_uncertified_candidate=[&](const kernel::AdaptiveResult& candidate,
                                                const char* reason) {
        if(candidate.stats.node_evaluations==0 || !std::isfinite(candidate.mu) ||
           candidate.mu<=0.0)return false;
        for(int j=0;j<5;++j)
            if(!std::isfinite(candidate.grad_mu[j]) ||
               candidate.grad_quality[j]==kernel::GradientQuality::NotRequested)return false;
        result.integral=candidate;
        uncertified_candidate_used=true;
        candidate_reason=reason;
        return true;
    };
    // Use the same smooth-disk certificate and quadrature as the caller's
    // late repair. If it succeeds, no discarded regular-chart solve is needed.
    if(smooth_repair && (isolated_edge_contact ||
       kernel::fused_gradient::reliability_retry_used ||
       !gradients_usable(result.integral))) {
        auto candidate=export_result();
        if(smooth_repair(candidate))return candidate;
    }
    if(isolated_edge_contact && !(support_plan && support_plan->completed &&
       result.integral.value_converged && gradients_usable(result.integral))) {
        // A certified real close pair at the outer support edge makes the
        // fold-mapped derivative sharply localized. Resolve that rare geometry
        // on a regular radial chart at the caller's unchanged value tolerance.
        auto contact_cfg=cfg;
        contact_cfg.fold_maps=false;
        contact_cfg.fused_value_mesh=false;
        contact_cfg.material_coordinate_motion=false;
        contact_cfg.initial_level=8;
        if(!retry_edge_pairs.empty())contact_cfg.initial_level=cfg.initial_level;
        // The regular radial chart uses the same value-driven replay policy as
        // epoch_adaptive; no gradient-error adaptation is enabled here.
        contact_cfg.gradient_adaptive=false;
        kernel::AdaptiveWorkspace contact_workspace;
        kernel::AdaptiveResult contact;
        if(retry_edge_pairs.empty())
            contact=kernel::epoch_adaptive(params,u,contact_cfg,contact_workspace);
        else {
            auto frontier=kernel::epoch_seeded_radial_frontier(params,u,seeds,contact_cfg,
                contact_workspace,false,support_plan,true,&retry_edge_pairs);
            remember_edge_pairs(frontier.edge_pairs);
            contact=std::move(frontier.integral);
        }
        total_nodes+=contact.stats.node_evaluations;
        no_fold_retry=true;
        if(contact.value_converged && gradients_usable(contact) &&
           contact.stop==kernel::AdaptiveStop::Converged)
            result.integral=std::move(contact);
        else
            retain_uncertified_candidate(contact,"contact_retry_unverified");
    }
    const bool missing_candidate=!(result.integral.mu>0.0) ||
        !std::isfinite(result.integral.mu) || result.integral.stats.node_evaluations==0 ||
        !std::all_of(result.integral.grad_mu.begin(),result.integral.grad_mu.end(),
            [](double component){return std::isfinite(component);});
    if((!no_fold_retry && topology_failure && retry_can_help &&
        !gradients_usable(result.integral)) || missing_candidate) {
        auto fallback_cfg=cfg;
        fallback_cfg.fold_maps=false;
        // This route is entered only after an unresolved epoch. Start at the
        // finest supported rule to resolve the narrow event neighborhood
        // without changing the caller's value tolerances.
        fallback_cfg.initial_level=8;
        kernel::AdaptiveWorkspace fallback_workspace;
        auto fallback=kernel::epoch_adaptive(params,u,fallback_cfg,fallback_workspace);
        if(!(fallback.mu>0.0) || !std::isfinite(fallback.mu) ||
           !std::all_of(fallback.grad_mu.begin(),fallback.grad_mu.end(),
               [](double component){return std::isfinite(component);})) {
            const auto previous_nodes=fallback.stats.node_evaluations;
            fallback_cfg.initial_level=cfg.initial_level;
            fallback_cfg.fused_value_mesh=false;
            fallback_cfg.material_coordinate_motion=false;
            fallback_cfg.gradient_adaptive=false;
            fallback_workspace.reset();
            auto frontier=kernel::epoch_seeded_radial_frontier(
                params,u,seeds,fallback_cfg,fallback_workspace,false,nullptr,true,
                &retry_edge_pairs);
            remember_edge_pairs(frontier.edge_pairs);
            fallback=std::move(frontier.integral);
            fallback.stats.node_evaluations+=previous_nodes;
        }
        total_nodes+=fallback.stats.node_evaluations;
        no_fold_retry=true;
        if(event_location_failure && !unresolved_outer_edge_pair &&
           gradients_usable(fallback)) {
            result.integral=std::move(fallback);
        } else if(!retain_uncertified_candidate(fallback,unresolved_outer_edge_pair
                    ? "unresolved_outer_edge_pair" : "topology_or_gradient_unverified") &&
                  fallback.value_converged) {
            result.integral.mu=fallback.mu;
            result.integral.estimated_abs_error_mu=fallback.estimated_abs_error_mu;
            result.integral.value_converged=true;
        }
    }
    if(!(result.integral.mu>0.0) || !std::isfinite(result.integral.mu) ||
       !std::all_of(result.integral.grad_mu.begin(),result.integral.grad_mu.end(),
           [](double component){return std::isfinite(component);})) {
        char context[2048];
        std::snprintf(context,sizeof(context),
            "algebraic_boundary: full radial retries produced no finite integral/derivative candidate; xs=%.17g ys=%.17g rho=%.17g q=%.17g s=%.17g barycentric=%d limb=%.17g mu=%.17g grad=[%.17g,%.17g,%.17g,%.17g,%.17g] initial_stop=%s final_stop=%s nodes=%zu frontier=%s seeds=%zu",
            p.xs,p.ys,p.rho,p.q,p.a,int(p.barycentric),u,result.integral.mu,
            result.integral.grad_mu[0],result.integral.grad_mu[1],result.integral.grad_mu[2],
            result.integral.grad_mu[3],result.integral.grad_mu[4],
            kernel::adaptive_stop_name(initial_stop),kernel::adaptive_stop_name(result.integral.stop),
            total_nodes,result.frontier.reason,seeds.size());
        throw std::runtime_error(context);
    }
    return export_result();
}
}

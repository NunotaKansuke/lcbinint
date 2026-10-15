#include "fused_gradient.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"

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
    const auto result=kernel::epoch_seeded_radial_frontier(params,u,seeds,cfg,workspace);
    FusedGradientResult out;
    out.reliability_retried=kernel::fused_gradient::reliability_retry_used;
    out.mu=result.integral.mu;out.grad_mu=result.integral.grad_mu;
    out.value_converged=result.integral.value_converged;
    out.stop=kernel::adaptive_stop_name(result.integral.stop);
    out.support_certified=result.frontier.coverage_certified;
    out.node_evaluations=result.integral.stats.node_evaluations;
    for(int j=0;j<5;++j)out.gradient_valid[j]=
        result.integral.grad_quality[j]!=kernel::GradientQuality::Invalid &&
        result.integral.grad_quality[j]!=kernel::GradientQuality::NotRequested &&
        std::isfinite(out.grad_mu[j]);
    return out;
}
}

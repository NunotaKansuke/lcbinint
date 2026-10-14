#pragma once
#ifndef HOLO_FUSED_GRADIENT_EXPERIMENT
#error "Use the isolated fused-gradient build; this header is not a production entry point."
#endif
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"

namespace lcbinint::holonomic::fused_gradient {
enum class Refinement { ValueMesh, JointTolerance };
struct Result {
    SeededFrontierResult epoch;
    bool gradient_precision_checked=false;
    bool all_gradient_tolerances_met=false;
};
inline Result evaluate(const LensParams& p,double u,const std::vector<Cplx<double>>& seeds,
    AdaptiveConfig cfg,AdaptiveWorkspace& workspace,Refinement refinement=Refinement::ValueMesh) {
    cfg.with_jacobian=true;
    cfg.gradient_policy=refinement==Refinement::ValueMesh?GradientPolicy::ValueFirst:GradientPolicy::Strict;
    cfg.fused_value_mesh=refinement==Refinement::ValueMesh;
    if(cfg.fused_value_mesh) {
        cfg.value_first_gradient_node_budget=0;
        cfg.value_first_gradient_round_budget=0;
    }
    Result result;
    result.epoch=epoch_seeded_radial_frontier(p,u,seeds,cfg,workspace);
    result.gradient_precision_checked=!cfg.fused_value_mesh;
    result.all_gradient_tolerances_met=result.gradient_precision_checked&&std::all_of(result.epoch.integral.grad_converged.begin(),
        result.epoch.integral.grad_converged.end(),[](bool met){return met;});
    return result;
}
}

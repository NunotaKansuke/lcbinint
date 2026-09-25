#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace lcbinint::holonomic;
int main(){
  LensParams p; p.xs=-0.07045642231021985; p.ys=-0.14887100926364968;
  p.rho=0.07530925479707781; p.q=2.1433640368724767e-8;
  p.a=3.035303804422572; p.barycentric=false;
  std::cout<<std::setprecision(17)<<"u\trtol\tpolicy\twall_ms\tvalue_ok\tvalue_stop\tmu\tmu_error\tnodes\tpanels\tsplits";
#ifdef SOFTCUTFOLD_CANDIDATE
  std::cout<<"\tmerges";
#endif
  for(int j=0;j<5;++j)std::cout<<"\tgrad"<<j<<"\tgrad_err"<<j<<"\tgrad_quality"<<j<<"\tgrad_reason"<<j<<"\tradial"<<j;
    std::cout<<"\n";
  for(double u:{0.0,.5})for(double rtol:{1e-3,1e-4})for(auto pol:{GradientPolicy::None,GradientPolicy::ValueFirst}){
    AdaptiveConfig c;c.gradient_policy=pol;c.with_jacobian=pol!=GradientPolicy::None;
    c.tol.mu_rtol=rtol;c.tol.grad_atol.fill(1e-4);c.tol.grad_rtol.fill(1e-2);
    c.value_first_gradient_round_budget=64;c.value_first_gradient_node_budget=8192;
    c.max_node_evals=16384;c.max_level=8;c.max_depth=8;
    AdaptiveWorkspace w;auto t0=std::chrono::steady_clock::now();
    auto r=epoch_adaptive(p,u,c,w);
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    std::cout<<u<<'\t'<<rtol<<'\t'<<gradient_policy_name(pol)<<'\t'<<ms<<'\t'<<r.value_converged<<'\t'<<adaptive_stop_name(r.value_stop_reason)<<'\t'<<r.mu<<'\t'<<r.estimated_abs_error_mu<<'\t'<<r.stats.unique_nodes<<'\t'<<r.stats.panels<<'\t'<<r.stats.splits;
#ifdef SOFTCUTFOLD_CANDIDATE
    std::cout<<'\t'<<r.stats.softcut_fold_merges;
#endif
    for(int j=0;j<5;++j)std::cout<<'\t'<<r.grad_mu[j]<<'\t'<<r.estimated_abs_error_grad[j]<<'\t'<<gradient_quality_name(r.grad_quality[j])<<'\t'<<gradient_reason_name(r.grad_reason[j])<<'\t'<<r.radial_error[j+1];
    std::cout<<'\n';
  }
}

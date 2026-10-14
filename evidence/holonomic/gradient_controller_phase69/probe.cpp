#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"
#include <iomanip>
#include <iostream>
int main(int argc,char**argv){
 std::ifstream f(argv[1]);std::string line;std::cout<<std::setprecision(17);
 std::cout<<"max_level grad_rtol kind name u parameter rounds tol level nodes converged mu grad error quality n h stop grad_reason radial inner geometry\n";
 while(std::getline(f,line)){
 LensParams p;double u,dummy;int bary;std::string name;std::istringstream ss(line);
 if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>dummy>>name))continue;
 int j=-1;
 if(name=="caustic-cross")j=u==0?4:0;
 if(name=="rand030"&&u==0)j=2;
 if(j<0)continue;p.barycentric=bary;
 for(int ml:{0,4,5})for(double gt:{1e-2,1e-3})for(double safety:{8.,32.,64.})for(double tol:{1e-3}){
 AdaptiveConfig cfg;cfg.gradient_local_refinement=ml!=0;cfg.gradient_split_min_level=ml?ml:4;cfg.tol.grad_atol.fill(1e-4);cfg.tol.grad_rtol.fill(gt);cfg.nested_difference_safety=2.;cfg.value_first_gradient_round_budget=int(safety);cfg.tol.mu_rtol=tol;cfg.gradient_policy=GradientPolicy::ValueFirst;
 // Node budget stays 4096; round limit is swept independently.
 AdaptiveWorkspace w;PreparedEpochGeometry state;(void)epoch_adaptive_prepared(p,u,cfg,w,state);
 auto r=epoch_adaptive_prepared(p,u,cfg,w,state);
 std::cout<<ml<<" "<<gt<<" budget "<<name<<" "<<u<<" "<<j<<" "<<safety<<" "<<tol<<" 3 "<<r.stats.unique_nodes<<" "<<r.value_converged<<" "<<r.mu<<" "<<r.grad_mu[j]<<" "<<r.estimated_abs_error_grad[j]<<" "<<int(r.grad_quality[j])<<" 0 0 "<<adaptive_stop_name(r.value_stop_reason)<<" "<<int(r.grad_reason[j])<<" "<<r.radial_error[j+1]<<" "<<r.inner_error[j+1]<<" "<<r.geometry_error[j+1]<<"\n";
 }
 }
}

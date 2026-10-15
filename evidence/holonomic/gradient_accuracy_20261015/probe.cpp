#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
using namespace lcbinint::holonomic_fused;
int main(int argc,char**argv) {
 if(argc<3 || argc>4)return 2;
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);
 out<<std::setprecision(17)<<"t\tarm\tj\tmu\tg\terror\tradial\tinner\tevent\tgeometry\troundoff\tquality\tvalue_converged\tstop\tgrad_stop\tnodes\tms\n";
 double t,x,y,rho,q,a,u;
 while(in>>t>>x>>y>>rho>>q>>a>>u){
  lcbinint::magnification::FiniteSourceSettings fs;fs.caustic_bins=1400;
  lcbinint::magnification::FiniteSourceMagnifier fm(fs);
  lcbinint::magnification::PointSourceMagnifier pm;
  auto seeds=make_caustic_aware_binary_seeds(fm,pm,a,q,{x,y},rho).radial_seeds;
  for(int arm=argc==4?std::stoi(argv[3]):0;arm<10;++arm){
#ifdef ONLY_VALUE_MESH
   if(arm!=0)continue;
#endif
   AdaptiveConfig cfg;cfg.with_jacobian=true;cfg.tol.mu_atol=cfg.tol.mu_rtol=std::getenv("LCBININT_PROBE_MU_TOL")?std::atof(std::getenv("LCBININT_PROBE_MU_TOL")):1e-5;
   cfg.tol.grad_atol.fill(1e-6);cfg.tol.grad_rtol.fill(1e-4);
   std::string name;
   if(arm<=1){
    cfg.fused_value_mesh=true;cfg.gradient_policy=GradientPolicy::ValueFirst;
    cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;
    name=arm==0?"value_mesh":"value_mesh_tight";
    if(arm==1){cfg.tol.mu_atol=1e-9;cfg.tol.mu_rtol=1e-8;}
   }else if(arm<=3){
    cfg.gradient_policy=GradientPolicy::Strict;name=arm==2?"joint":"joint_loose";
    if(arm==3)cfg.tol.grad_rtol.fill(1e-3);
   }else if(arm<=5){
    cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.value_first_gradient_node_budget=16384;
    cfg.value_first_gradient_round_budget=64;name=arm==4?"post_gradient":"post_hsplit";
    cfg.gradient_local_refinement=arm==5;
   }else if(arm<=8){
    cfg.fused_value_mesh=true;cfg.gradient_policy=GradientPolicy::ValueFirst;
    cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;
    cfg.initial_level=arm;name="radial"+std::to_string((1<<arm)-1);
   }else{
    cfg.gradient_policy=GradientPolicy::Strict;name="coordinate_scaled";
    cfg.tol.grad_atol[3]=1e-14;
    cfg.tol.grad_atol[0]=cfg.tol.grad_atol[1]=1e-7;
    cfg.tol.grad_rtol[0]=cfg.tol.grad_rtol[1]=1e-7;
   }
   AdaptiveWorkspace w;auto start=std::chrono::steady_clock::now();
   auto r=epoch_seeded_radial_frontier({x,y,rho,q,a,true},u,seeds,cfg,w).integral;
   double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   for(int j=0;j<5;++j)out<<t<<'\t'<<name<<'\t'<<j<<'\t'<<r.mu<<'\t'<<r.grad_mu[j]<<'\t'<<r.estimated_abs_error_grad[j]<<'\t'<<r.radial_error[j+1]<<'\t'<<r.inner_error[j+1]<<'\t'<<r.event_error[j+1]<<'\t'<<r.geometry_error[j+1]<<'\t'<<r.roundoff_error[j+1]<<'\t'<<gradient_quality_name(r.grad_quality[j])<<'\t'<<r.value_converged<<'\t'<<adaptive_stop_name(r.stop)<<'\t'<<adaptive_stop_name(r.gradient_stop_reason)<<'\t'<<r.stats.node_evaluations<<'\t'<<ms<<'\n';
   out.flush();std::cerr<<t<<' '<<name<<' '<<ms<<" ms\n";
  }
 }
}

#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic_fused;
struct Case{const char*name;double t;};
int main(){constexpr double s=.95,q=1e7,rho=1e-6,u0=9.846481110931334e-7,alpha=.5,tE=10.;const double co=std::cos(alpha),si=std::sin(alpha);std::cout<<std::setprecision(17);for(double tol:{1e-3,1e-5})for(const Case c: {Case{"negative",-1.7455113096375089e-6},Case{"center",0.}}){auto value=[&](double dt0){double tau=(c.t-dt0)/tE;LensParams p{tau*co-u0*si,tau*si+u0*co,rho,q,s,true};AdaptiveConfig cfg;cfg.tol.mu_atol=cfg.tol.mu_rtol=tol;cfg.with_jacobian=true;cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.fold_maps=false;cfg.fused_value_mesh=false;cfg.material_coordinate_motion=false;cfg.initial_level=8;cfg.gradient_adaptive=false;cfg.value_first_gradient_node_budget=cfg.value_first_gradient_round_budget=0;AdaptiveWorkspace w;auto r=epoch_adaptive(p,.5,cfg,w);return r;};for(double h:{1e-8,3e-9,1e-9}){auto plus=value(h),minus=value(-h);double fd=(plus.mu-minus.mu)/(2*h);std::cout<<c.name<<" tol="<<tol<<" h="<<h<<" fd="<<fd<<" plus="<<plus.mu<<" minus="<<minus.mu<<" conv="<<plus.value_converged<<minus.value_converged<<" stops="<<adaptive_stop_name(plus.stop)<<"/"<<adaptive_stop_name(minus.stop)<<" nodes="<<plus.stats.node_evaluations+minus.stats.node_evaluations<<"\n";}}}

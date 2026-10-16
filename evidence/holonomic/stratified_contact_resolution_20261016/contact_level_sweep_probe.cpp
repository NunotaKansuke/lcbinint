#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic_fused;
int main(){constexpr double u0=9.846481110931334e-7,alpha=.5,tE=10.,rho=1e-6,q=1e7,s=.95;const double c=std::cos(alpha),si=std::sin(alpha);std::cout<<std::setprecision(17);
for(auto [name,t]:{std::pair<const char*,double>{"negative",-1.7455113096375089e-6},{"center",0.}}){double tau=t/tE;LensParams p{tau*c-u0*si,tau*si+u0*c,rho,q,s,true};
for(bool fold:{true,false})for(int level:{3,4,5,6,7,8}){AdaptiveConfig cfg;cfg.tol.mu_atol=1e-3;cfg.tol.mu_rtol=1e-3;cfg.with_jacobian=true;cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.fold_maps=fold;cfg.initial_level=level;cfg.collect_diagnostics=false;AdaptiveWorkspace w;auto r=epoch_adaptive(p,.5,cfg,w);double t0g=(-c*r.grad_mu[0]-si*r.grad_mu[1])/tE;std::cout<<name<<" fold="<<fold<<" level="<<level<<" val="<<r.mu<<" t0="<<t0g<<" nodes="<<r.stats.node_evaluations<<" panels="<<r.stats.panels<<" stop="<<adaptive_stop_name(r.stop)<<" gvalid=";for(auto x:r.grad_converged)std::cout<<x;std::cout<<"\n";}}
}

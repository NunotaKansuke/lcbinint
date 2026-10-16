#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
int main(){using namespace lcbinint::holonomic;constexpr double rho=1e-6,u0=9.846481110931334e-7,alpha=.5,tE=10.,s=.95,q=1e7;
 std::cout<<std::setprecision(17);for(auto [name,t]:{std::pair<const char*,double>{"neg",-1.7455113096375089e-6},{"center",0.}}){double tau=t/tE,x=tau*cos(alpha)-u0*sin(alpha),y=tau*sin(alpha)+u0*cos(alpha);LensParams p{x,y,rho,q,s,true};
  for(double tol:{1e-3,1e-5})for(bool adapt:{false,true}){AdaptiveConfig cfg;cfg.tol.mu_atol=tol;cfg.tol.mu_rtol=tol;cfg.with_jacobian=true;cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.fold_maps=false;cfg.initial_level=3;cfg.collect_diagnostics=true;for(int j=0;j<5;++j)cfg.tol.grad_rtol[j]=tol;
   if(adapt){cfg.gradient_adaptive=true;cfg.value_first_gradient_node_budget=8192;cfg.value_first_gradient_round_budget=16;cfg.gradient_adaptive_node_budget=8192;cfg.gradient_adaptive_round_budget=16;}
   else {cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;}
   AdaptiveWorkspace w;auto a=epoch_adaptive(p,.5,cfg,w);double t0grad=(-cos(alpha)*a.grad_mu[0]-sin(alpha)*a.grad_mu[1])/tE;
   std::cout<<"CASE "<<name<<" tol="<<tol<<" adapt="<<adapt<<" mu="<<a.mu<<" valerr="<<a.estimated_abs_error_mu<<" conv="<<a.value_converged<<" stop="<<adaptive_stop_name(a.stop)<<" nodes="<<a.stats.node_evaluations<<" panels="<<a.stats.panels<<" t0="<<t0grad<<" gerr=";for(auto z:a.estimated_abs_error_grad)std::cout<<z<<',';std::cout<<" gconv=";for(auto z:a.grad_converged)std::cout<<z<<',';std::cout<<" gq=";for(auto z:a.grad_quality)std::cout<<int(z)<<',';std::cout<<" angular="<<a.stats.angular_upgrades<<"\n";
  }
 }
}

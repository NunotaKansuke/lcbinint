#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
int main(){using namespace lcbinint::holonomic;constexpr double rho=1e-6,u0=9.846481110931334e-7,alpha=.5,tE=10.,s=.95,q=1e7;
 std::cout<<std::setprecision(17);for(auto [name,t]:{std::pair<const char*,double>{"neg",-1.7455113096375089e-6},{"center",0.},{"pos",1.7455113096375089e-6}}){double tau=t/tE,x=tau*cos(alpha)-u0*sin(alpha),y=tau*sin(alpha)+u0*cos(alpha);LensParams p{x,y,rho,q,s,true};
  for(double tol:{1e-3,1e-5,1e-7}){AdaptiveConfig cfg;cfg.tol.mu_atol=tol;cfg.tol.mu_rtol=tol;cfg.with_jacobian=true;cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;cfg.fold_maps=false;cfg.initial_level=8;cfg.collect_diagnostics=true;AdaptiveWorkspace w;auto a=epoch_adaptive(p,.5,cfg,w);
   std::cout<<"CASE "<<name<<" tol="<<tol<<" mu="<<a.mu<<" err="<<a.estimated_abs_error_mu<<" conv="<<a.value_converged<<" stop="<<adaptive_stop_name(a.stop)<<" nodes="<<a.stats.node_evaluations<<" panels="<<w.panels.size()<<" reject=";for(auto z:a.stats.sample_reject_counts)std::cout<<z<<',';std::cout<<" cold=";for(auto z:a.stats.sample_cold_reject_counts)std::cout<<z<<',';std::cout<<" graderr=";for(auto z:a.estimated_abs_error_grad)std::cout<<z<<',';std::cout<<"\n";
   std::vector<const AdaptivePanel*> ps;for(const auto&z:w.panels)if(z.active)ps.push_back(&z);std::sort(ps.begin(),ps.end(),[](auto*a,auto*b){return a->error[0]>b->error[0];});for(size_t j=0;j<std::min<size_t>(3,ps.size());++j){auto&z=*ps[j];std::cout<<" P "<<z.cell<<' '<<z.map.a<<' '<<z.map.b<<" lv="<<z.level<<" dep="<<z.depth<<" err="<<z.error[0]<<" radial="<<z.radial[0]<<" inner="<<z.inner[0]<<" event="<<z.event[0]<<"\n";}
   for(size_t j=0;j<std::min<size_t>(6,a.stats.sample_diagnostics.size());++j){auto&d=a.stats.sample_diagnostics[j];std::cout<<" REJ "<<adaptive_sample_reject_reason_name(d.reason)<<" R="<<d.R<<" cell="<<d.cell<<" level="<<d.level<<" cold="<<d.cold_retry<<" ok="<<d.cold_retry_succeeded<<"\n";}
  }
 }
}

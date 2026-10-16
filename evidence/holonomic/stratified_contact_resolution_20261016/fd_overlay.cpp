#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
int main(){using namespace lcbinint::holonomic_fused;constexpr double rho=1e-6,u0=9.846481110931334e-7,alpha=.5,tE=10.,s=.95,q=1e7;std::cout<<std::setprecision(17);
 for(auto [name,t]:{std::pair<const char*,double>{"neg",-1.7455113096375089e-6},{"center",0.},{"pos",1.7455113096375089e-6}}){double tau=t/tE,x=tau*cos(alpha)-u0*sin(alpha),y=tau*sin(alpha)+u0*cos(alpha);LensParams p{x,y,rho,q,s,true};AdaptiveConfig cfg;cfg.tol.mu_atol=1e-5;cfg.tol.mu_rtol=1e-5;cfg.fold_maps=false;cfg.initial_level=8;
  auto eval=[&](double delta){auto pp=p;pp.xs-=cos(alpha)*delta/tE;pp.ys-=sin(alpha)*delta/tE;AdaptiveWorkspace w;auto a=epoch_value_adaptive(pp,.5,cfg,w);return a;};auto base=eval(0);std::cout<<"CASE "<<name<<" base="<<base.mu<<" err="<<base.estimated_abs_error_mu<<" ok="<<base.value_converged<<" stop="<<adaptive_stop_name(base.stop)<<"\n";
  for(double h:{1e-7,3e-8,1e-8,3e-9,1e-9}){auto plus=eval(h),minus=eval(-h);double central=(plus.mu-minus.mu)/(2*h),left=(base.mu-minus.mu)/h,right=(plus.mu-base.mu)/h;std::cout<<" h="<<h<<" plus="<<plus.mu<<"("<<plus.value_converged<<","<<plus.estimated_abs_error_mu<<") minus="<<minus.mu<<"("<<minus.value_converged<<","<<minus.estimated_abs_error_mu<<") d="<<central<<" sides="<<left<<","<<right<<" nodes="<<plus.stats.node_evaluations<<','<<minus.stats.node_evaluations<<"\n";}
 }
}

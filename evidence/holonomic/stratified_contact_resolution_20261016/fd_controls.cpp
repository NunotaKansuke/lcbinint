#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic_fused;
void run(const char* name,double t,double u0,double alpha,double tE,double rho,double q,double s,double limb,const double* hs,int nh){
 const double c=std::cos(alpha),si=std::sin(alpha),tau=t/tE;
 LensParams p{tau*c-u0*si,tau*si+u0*c,rho,q,s,true};
 std::cout<<"CASE "<<name<<"\n";
 for(double tol:{1e-3,1e-5}){
  AdaptiveConfig cfg;cfg.tol.mu_atol=tol;cfg.tol.mu_rtol=tol;cfg.fold_maps=false;cfg.initial_level=8;
  auto eval=[&](double dt){auto pp=p;pp.xs-=c*dt/tE;pp.ys-=si*dt/tE;AdaptiveWorkspace w;auto z=epoch_value_adaptive(pp,limb,cfg,w);return z;};
  auto z0=eval(0);std::cout<<"tol="<<tol<<" base="<<z0.mu<<" ok="<<z0.value_converged<<" stop="<<adaptive_stop_name(z0.stop)<<"\n";
  for(int i=0;i<nh;++i){auto a=eval(hs[i]),b=eval(-hs[i]);double fd=(a.mu-b.mu)/(2*hs[i]);
   std::cout<<" h="<<hs[i]<<" plus="<<a.mu<<"("<<a.value_converged<<","<<adaptive_stop_name(a.stop)<<") minus="<<b.mu<<"("<<b.value_converged<<","<<adaptive_stop_name(b.stop)<<") fd="<<fd<<"\n";}
 }
}
int main(){std::cout<<std::setprecision(17);
 const double hs_s[]={1e-8,3e-9,1e-9};
 run("strat_neg",-1.7455113096375089e-6,9.846481110931334e-7,.5,10.,1e-6,1e7,.95,.5,hs_s,3);
 run("strat_center",0.,9.846481110931334e-7,.5,10.,1e-6,1e7,.95,.5,hs_s,3);
 const double hs_o[]={1e-4,3e-5,1e-5};
 run("ordinary",0.,.3,0.,10.,.003,100.,1.2,.5,hs_o,3);
}

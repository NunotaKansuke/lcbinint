#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include "fused_k.hpp"
using namespace lcbinint::holonomic_fused;
TopologyResult topo(double t,double u0,double alpha,double tE,double rho,double q,double s){double tau=t/tE,c=cos(alpha),z=sin(alpha);return classify_cells(PrimaryFrame::from(LensParams{tau*c-u0*z,tau*z+u0*c,rho,q,s,true}),nullptr,nullptr,true);}
int main(){constexpr double t=0,u0=9.846481110931334e-7,alpha=.5,tE=10,rho=1e-6,q=1e7,s=.95,h=1e-9;auto base=topo(t,u0,alpha,tE,rho,q,s),plus=topo(t+h,u0,alpha,tE,rho,q,s),minus=topo(t-h,u0,alpha,tE,rho,q,s);auto pf=PrimaryFrame::from(LensParams{-u0*sin(alpha),u0*cos(alpha),rho,q,s,true});std::cout<<std::setprecision(17)<<"event counts="<<base.events.size()<<","<<plus.events.size()<<","<<minus.events.size()<<"\n";for(size_t i=0;i<base.events.size();++i){const auto&e=base.events[i];if(!e.physically_real)continue;auto loc=adaptive_detail::topology_event_estimate(e,pf,1e-3,false);adaptive_detail::EventLocation fixed;fixed.radius=3;auto map=fused_gradient::make_material_map(loc,fixed,pf,false,true);double analytic=(-cos(alpha)*map.da[0]-sin(alpha)*map.da[1])/tE;double rp=i<plus.events.size()?plus.events[i].radius+plus.events[i].radius_lo:NAN,rm=i<minus.events.size()?minus.events[i].radius+minus.events[i].radius_lo:NAN;double fd=(rp-rm)/(2*h);std::cout<<"i="<<i<<" R="<<e.radius+e.radius_lo<<" dRdt0="<<analytic<<" fd="<<fd<<" map="<<map.valid<<" seeds="<<e.fold_t_seed_valid<<","<<e.fold_u_seed_valid<<" t="<<e.fold_t_seed<<"\n";}}

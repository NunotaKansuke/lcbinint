#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic_fused;
TopologyResult topo(double x,double y,double rho,double q,double s){return classify_cells(PrimaryFrame::from(LensParams{x,y,rho,q,s,true}),nullptr,nullptr,true);}
int main(){constexpr double rho=1e-6,u0=9.846481110931334e-7,a=.5,tE=10.,s=.95,q=1e7,h=1e-7;std::cout<<std::setprecision(17);
 for(auto [name,t]:{std::pair<const char*,double>{"neg",-1.7455113096375089e-6},{"center",0.}}){double tau=t/tE,x=tau*cos(a)-u0*sin(a),y=tau*sin(a)+u0*cos(a);auto z=topo(x,y,rho,q,s),zp=topo(x-cos(a)*h/tE,y-sin(a)*h/tE,rho,q,s),zm=topo(x+cos(a)*h/tE,y+sin(a)*h/tE,rho,q,s);auto pf=PrimaryFrame::from(LensParams{x,y,rho,q,s,true});
  std::cout<<"CASE "<<name<<"\n";for(const auto&e:z.events)if(e.physically_real&&e.radius>1.7&&e.radius<1.95){auto loc=adaptive_detail::topology_event_estimate(e,pf,1e-3,false);auto other=adaptive_detail::EventLocation{};other.radius=1.95;other.t_seed_valid=false;auto mm=fused_gradient::make_material_map(loc,other,pf,false,true);double analytic=(-cos(a)*mm.da[0]-sin(a)*mm.da[1])/tE;
   auto nearest=[&](const TopologyResult&v){const RadialEvent*b=nullptr;for(const auto&k:v.events)if(k.physically_real&&k.detail==e.detail&&(!b||fabs(k.radius-e.radius)<fabs(b->radius-e.radius)))b=&k;return b?b->radius:std::numeric_limits<double>::quiet_NaN();};double rp=nearest(zp),rm=nearest(zm),fd=(rp-rm)/(2*h);
   std::cout<<" event="<<e.radius<<" dRdt0="<<analytic<<" fd="<<fd<<" rp="<<rp<<" rm="<<rm<<" t="<<e.fold_t_seed<<" valid="<<mm.valid<<" da="<<mm.da[0]<<','<<mm.da[1]<<"\n";}}
}

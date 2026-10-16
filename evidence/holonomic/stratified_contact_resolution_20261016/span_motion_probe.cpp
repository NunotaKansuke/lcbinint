#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include "fused_k.hpp"
using namespace lcbinint::holonomic_fused;
void run(const char* name,double t,double u0,double alpha,double tE,double rho,double q,double s) {
 double co=std::cos(alpha),si=std::sin(alpha),tau=t/tE;
 LensParams p{tau*co-u0*si,tau*si+u0*co,rho,q,s,true};
 auto pf=PrimaryFrame::from(p);auto topo=classify_cells(pf,nullptr,nullptr,true);
 AdaptiveConfig cfg;auto plan=adaptive_detail::make_adaptive_integration_plan(topo,topo.cells,cfg.initial_level,cfg.fold_maps);
 std::cout<<"CASE "<<name<<" rmax="<<topo.r_max<<" events="<<topo.events.size()<<" cells="<<topo.cells.size()<<" spans="<<plan.spans.size()<<"\n";
 for(const auto& span:plan.spans){const auto& c=span.cell;std::cout<<" span "<<c.index<<" ["<<c.r_lo<<","<<c.r_hi<<"] kind="<<int(c.kind)<<" crossings="<<c.n_crossings;
  for(int side=0;side<2;++side){double r=side?c.r_hi:c.r_lo;const RadialEvent* event=nullptr;for(const auto&e:topo.events)if(e.radius==r&&adaptive_detail::adaptive_physical_fold_event(e)){event=&e;break;}if(!event)continue;
   auto loc=adaptive_detail::topology_event_estimate(*event,pf,1e-3,false);adaptive_detail::EventLocation fixed;fixed.radius=side?1.0:0.0;fixed.t_seed_valid=false;
   auto m=side?fused_gradient::make_material_map(fixed,loc,pf,true,false):fused_gradient::make_material_map(loc,fixed,pf,false,true);
   const auto& d=side?m.db:m.da;double dRdt0=(-co*d[0]-si*d[1])/tE;
   std::cout<<" "<<(side?"R":"L")<<"event{"<<event->kind<<","<<event->detail<<",R="<<event->radius<<",Rlo="<<event->radius_lo<<",seed="<<event->fold_t_seed<<",loc="<<loc.radius+loc.radius_lo<<",valid="<<m.valid<<",dRdt0="<<dRdt0<<"}";
  }
  std::cout<<"\n";
 }
}
int main(){std::cout<<std::setprecision(17);
 run("ordinary",0.,.3,0.,10.,.003,100.,1.2);
 run("case117",-1.671007233467463,1e-9,1.186247434824958,10.,.1671007233467463,1268048.7165272941,.710053684596912);
 run("strat_neg",-1.7455113096375089e-6,9.846481110931334e-7,.5,10.,1e-6,1e7,.95);
 run("strat_center",0.,9.846481110931334e-7,.5,10.,1e-6,1e7,.95);
}

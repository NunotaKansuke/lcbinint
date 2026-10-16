#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include "fused_k.hpp"
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::adaptive_detail;
TopologyResult topo(double t,double u0,double alpha,double tE,double rho,double q,double s){double tau=t/tE,c=cos(alpha),z=sin(alpha);return classify_cells(PrimaryFrame::from(LensParams{tau*c-u0*z,tau*z+u0*c,rho,q,s,true}),nullptr,nullptr,true);}
int main(){constexpr double u0=9.846481110931334e-7,alpha=.5,tE=10,rho=1e-6,q=1e7,s=.95;auto base=topo(0,u0,alpha,tE,rho,q,s);auto pf=PrimaryFrame::from(LensParams{-u0*sin(alpha),u0*cos(alpha),rho,q,s,true});std::cout<<std::setprecision(17);
 for(size_t i=0;i<base.events.size();++i){const auto&e=base.events[i];if(!e.physically_real || (i!=8&&i!=9&&i!=10))continue;auto loc=topology_event_estimate(e,pf,1e-3,true);double m=loc.pair_motion_valid?loc.pair_m:loc.t_seed;bool reciprocal=loc.pair_motion_valid?loc.pair_reciprocal:loc.seed_reciprocal;double theta=reciprocal?2*std::atan(m)+kPi:2*std::atan(m);auto c=phi_detail::core(loc.radius+loc.radius_lo,theta,pf);auto g=phi_detail::dP_from(c,pf);using namespace phi_detail;C radial=(c.z+C{pf.m0,0}*c.inv_zb+C{c.m1,0}*(c.inv_zba*c.inv_zba)*c.zb)/C{loc.radius+loc.radius_lo,0};double slope=-2*(c.g.re*radial.re+c.g.im*radial.im)*c.inv_r2;double angular=phi_detail::dphi_dtheta_from(c,pf);auto map=lcbinint::holonomic::fused_gradient::make_material_map(loc,EventLocation{.radius=3,.uncertainty=0},pf,false,true);double analytic=(-cos(alpha)*map.da[0]-sin(alpha)*map.da[1])/tE;
 std::cout<<"EVENT i="<<i<<" R="<<loc.radius+loc.radius_lo<<" t="<<e.fold_t_seed<<" loc_t="<<loc.t_seed<<" loc_valid="<<loc.t_seed_valid<<" tier="<<loc.precision_tier<<" pair="<<loc.pair_motion_valid<<" pair_m="<<loc.pair_m<<" dm="<<loc.pair_dm<<" dv="<<loc.pair_dv<<" ddres="<<loc.dd_residual<<" map="<<map.valid<<" phi="<<c.phi<<" angular="<<angular<<" slope="<<slope<<" dRdt0="<<analytic<<" dP=";for(auto x:g)std::cout<<x<<",";std::cout<<"\n";
 for(double h:{1e-8,3e-9,1e-9,3e-10,1e-10}){auto plus=topo(h,u0,alpha,tE,rho,q,s),minus=topo(-h,u0,alpha,tE,rho,q,s);double r0=loc.radius+loc.radius_lo,rp=NAN,rm=NAN,dp=INFINITY,dm=INFINITY;for(const auto&v:plus.events)if(v.physically_real){double r=v.radius+v.radius_lo;if(std::abs(r-r0)<dp){dp=std::abs(r-r0);rp=r;}}for(const auto&v:minus.events)if(v.physically_real){double r=v.radius+v.radius_lo;if(std::abs(r-r0)<dm){dm=std::abs(r-r0);rm=r;}}std::cout<<" FD h="<<h<<" rp="<<rp<<" rm="<<rm<<" dRdt0="<<-(rp-rm)/(2*h)<<" match="<<dp<<","<<dm<<"\n";}
 }
}

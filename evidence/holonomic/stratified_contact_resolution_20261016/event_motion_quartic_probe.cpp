#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::adaptive_detail;
TopologyResult topo(double t,double u0,double alpha,double tE,double rho,double q,double s){double tau=t/tE,c=cos(alpha),z=sin(alpha);return classify_cells(PrimaryFrame::from(LensParams{tau*c-u0*z,tau*z+u0*c,rho,q,s,true}),nullptr,nullptr,true);}
double eval(const std::array<double,5>& c,double x){double v=0;for(int i=4;i>=0;--i)v=v*x+c[i];return v;}
int main(){constexpr double u0=9.846481110931334e-7,alpha=.5,tE=10,rho=1e-6,q=1e7,s=.95;auto base=topo(0,u0,alpha,tE,rho,q,s);auto pf=PrimaryFrame::from(LensParams{-u0*sin(alpha),u0*cos(alpha),rho,q,s,true});std::cout<<std::setprecision(17);for(size_t i=0;i<base.events.size();++i){const auto&e=base.events[i];if(!e.physically_real||i<8||i>9)continue;auto loc=topology_event_estimate(e,pf,1e-3,true);double R=loc.radius+loc.radius_lo,t=loc.pair_m;auto pr=boundary_quartic_dR(R,pf).p;double P_R=eval(pr,t);auto dp=boundary_quartic_dp(R,pf);double dX=-eval(dp.dp[0],t)/P_R,dY=-eval(dp.dp[1],t)/P_R;double dt0=(-cos(alpha)*dX-sin(alpha)*dY)/tE;auto P=boundary_quartic(R,pf).p;double pval=eval(P,t),pt=0;for(int k=1;k<=4;k++)pt+=k*P[k]*std::pow(t,k-1);std::cout<<"i="<<i<<" R="<<R<<" t="<<t<<" PR="<<P_R<<" P="<<pval<<" Pt="<<pt<<" dX="<<dX<<" dY="<<dY<<" dt0="<<dt0<<"\n";}}

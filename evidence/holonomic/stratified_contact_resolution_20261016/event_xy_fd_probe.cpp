#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic;using namespace lcbinint::holonomic::adaptive_detail;
TopologyResult topo(double xs,double ys,double a=.95){return classify_cells(PrimaryFrame::from(LensParams{xs,ys,1e-6,1e7,a,true}),nullptr,nullptr,true);}
int main(){double u0=9.846481110931334e-7,alpha=.5,a=.95,m1=1e7/(1+1e7),xs=-u0*sin(alpha),ys=u0*cos(alpha),baseX=xs+m1*a;auto b=topo(xs,ys);std::cout<<std::setprecision(17)<<"base X="<<baseX<<"\n";for(double target:{.28497402197043031,1.8161301771116094}){double r0=0,bd=INFINITY;for(auto&e:b.events)if(e.physically_real){double r=e.radius+e.radius_lo,d=fabs(r-target);if(d<bd){bd=d;r0=r;}}std::cout<<"event R="<<r0<<"\n";for(int axis=0;axis<2;++axis)for(double h:{1e-9,3e-10,1e-10}){auto bp=topo(xs+(axis==0?h:0),ys+(axis==1?h:0)),bm=topo(xs-(axis==0?h:0),ys-(axis==1?h:0));auto near=[&](const TopologyResult&t){double best=INFINITY,r=0;for(auto&e:t.events)if(e.physically_real){double x=e.radius+e.radius_lo,d=fabs(x-r0);if(d<best){best=d;r=x;}}return r;};double rp=near(bp),rm=near(bm);std::cout<<" axis="<<axis<<" h="<<h<<" dR="<<(rp-rm)/(2*h)<<" delta="<<fabs(rp-r0)<<","<<fabs(rm-r0)<<"\n";}}}

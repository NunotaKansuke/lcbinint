#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include "geometry.hpp"
using namespace lcbinint::holonomic_fused;
void run(const char*name,double t,double u0,double alpha,double tE,double rho,double q,double s){double c=cos(alpha),z=sin(alpha),tau=t/tE;LensParams p{tau*c-u0*z,tau*z+u0*c,rho,q,s,true};auto pf=PrimaryFrame::from(p);auto topo=classify_cells(pf,nullptr,nullptr,true);bool complete=false;auto known=geometry_experiment::known_internal_pairs(pf,topo,7);auto all=geometry_experiment::detect(pf,topo,{},&complete,7);std::cout<<std::setprecision(17)<<"CASE "<<name<<" rmax="<<topo.r_max<<" events="<<topo.events.size()<<" cells="<<topo.cells.size()<<" known="<<known.size()<<" detected="<<all.size()<<" complete="<<complete<<" screen_special="<<geometry_experiment::screen_stats.special_epochs<<" fine="<<geometry_experiment::screen_stats.fine_epochs<<"\n";for(size_t i=0;i<topo.events.size();++i){const auto&e=topo.events[i];if(e.physically_real&&e.radius>1.7)std::cout<<" event["<<i<<"] R="<<e.radius<<" kind="<<e.kind<<" detail="<<e.detail<<"\n";}for(auto x:all)std::cout<<" pair center="<<x.center<<" scale="<<x.scale<<" left="<<x.left<<" right="<<x.right<<" band="<<x.band_lo<<","<<x.band_hi<<"\n";}
int main(){run("pos",1.7455113096375089e-6,9.846481110931334e-7,.5,10,1e-6,1e7,.95);run("neg",-1.7455113096375089e-6,9.846481110931334e-7,.5,10,1e-6,1e7,.95);run("center",0,9.846481110931334e-7,.5,10,1e-6,1e7,.95);run("ordinary",0,.3,0,10,.003,100,1.2);}

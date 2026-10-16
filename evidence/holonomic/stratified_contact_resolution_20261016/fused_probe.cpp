#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
#include "lcbinint/magnification/holonomic/fused_gradient.hpp"
int main(){using namespace lcbinint;using namespace lcbinint::holonomic;constexpr double s=.95,q=1e7,rho=1e-6,u0=9.846481110931334e-7,alpha=.5,tE=10.;magnification::FiniteSourceSettings st;st.caustic_bins=1400;magnification::FiniteSourceMagnifier f(st);magnification::PointSourceMagnifier pt;f.binary_caustic_branches(s,q);std::cout<<std::setprecision(17);
 for(auto [name,t]:{std::pair<const char*,double>{"neg",-1.7455113096375089e-6},{"center",0.},{"pos",1.7455113096375089e-6}}){double tau=t/tE,x=tau*cos(alpha)-u0*sin(alpha),y=tau*sin(alpha)+u0*cos(alpha);auto seed=make_caustic_aware_binary_seeds(f,pt,s,q,{x,y},rho);std::vector<std::array<double,2>> sv;for(auto z:seed.radial_seeds)sv.push_back({z.re,z.im});for(double tol:{1e-3,1e-5}){FusedGradientOptions o;o.mu_atol=tol;o.mu_rtol=tol;auto r=epoch_fused_gradient({x,y,rho,q,s,true},.5,sv,o);double t0g=(-cos(alpha)*r.grad_mu[0]-sin(alpha)*r.grad_mu[1])/tE;std::cout<<name<<" tol="<<tol<<" mu="<<r.mu<<" conv="<<r.value_converged<<" stop="<<r.stop<<" retry="<<r.reliability_retried<<" nodes="<<r.node_evaluations<<" t0="<<t0g<<" valid=";for(bool v:r.gradient_valid)std::cout<<v;std::cout<<" raw=";for(auto z:r.grad_mu)std::cout<<z<<',';std::cout<<"\n";}}
}

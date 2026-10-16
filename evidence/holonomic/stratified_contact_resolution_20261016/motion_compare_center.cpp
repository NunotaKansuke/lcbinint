#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "../../../src/lcbinint/magnification/holonomic/fused_gradient.hpp"
#include "../../../src/lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
using namespace lcbinint::holonomic;
struct Case { const char* name; double u0,alpha,tE,rho,q,s,limb; std::vector<double> times; };
void run_case(const Case& c) {
    lcbinint::magnification::FiniteSourceSettings settings; settings.caustic_bins=1400;
    lcbinint::magnification::FiniteSourceMagnifier finite(settings);
    finite.binary_caustic_branches(c.s,c.q);
    lcbinint::magnification::PointSourceMagnifier point;
    for(double t:c.times) for(double tol:{1e-3,1e-5}) {
        const double tau=t/c.tE, co=std::cos(c.alpha), si=std::sin(c.alpha);
        const double x=tau*co-c.u0*si, y=tau*si+c.u0*co;
        LensParams p{x,y,c.rho,c.q,c.s,true};
        auto seed=make_caustic_aware_binary_seeds(finite,point,c.s,c.q,{x,y},c.rho);
        std::vector<std::array<double,2>> seeds;
        for(const auto& z:seed.radial_seeds)seeds.push_back({z.re,z.im});
        auto r=epoch_fused_gradient(p,c.limb,seeds,{tol,tol});
        const double t0=(-co*r.grad_mu[0]-si*r.grad_mu[1])/c.tE;
        std::cout<<c.name<<" t="<<std::setprecision(17)<<t<<" tol="<<tol
          <<" mu="<<r.mu<<" value="<<r.value_converged<<" stop="<<r.stop
          <<" retry="<<r.reliability_retried<<" support="<<r.support_certified
          <<" seeds="<<seed.radial_seeds.size()<<" t0="<<t0
          <<" gx="<<r.grad_mu[0]<<" gy="<<r.grad_mu[1]<<" rho="<<r.grad_mu[2]
          <<" valid=";
        for(bool v:r.gradient_valid)std::cout<<v;
        std::cout<<" nodes="<<r.node_evaluations<<"\n";
    }
}
int main(){
  run_case({"stratified09_center",9.846481110931334e-7,.5,10.,1e-6,1e7,.95,.5,{0.}});
}

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "lcbinint/magnification/holonomic/fused_gradient.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
int main(int argc,char**argv){
 double s=std::atof(argv[1]),q=std::atof(argv[2]),rho=std::atof(argv[3]),x=std::atof(argv[4]),y=std::atof(argv[5]),u=std::atof(argv[6]),tol=std::atof(argv[7]);
 lcbinint::magnification::FiniteSourceMagnifier finite(lcbinint::magnification::FiniteSourceSettings{});
 lcbinint::magnification::PointSourceMagnifier point;
 auto seed=lcbinint::holonomic::make_caustic_aware_binary_seeds(finite,point,s,q,{x,y},rho);
 std::vector<std::array<double,2>> seeds;for(auto z:seed.radial_seeds)seeds.push_back({z.re,z.im});
 auto r=lcbinint::holonomic::epoch_fused_gradient({x,y,rho,q,s,true},u,seeds,{tol,tol});
 std::cout<<std::setprecision(17)<<r.mu;for(auto v:r.grad_mu)std::cout<<" "<<v;std::cout<<"\n";
}

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
using namespace lcbinint::holonomic;
int main(int argc,char**argv){if(argc!=2)return 2;const double u0=9.846481110931334e-7,alpha=.5,tE=10,rho=1e-6,q=1e7,s=.95;double t=std::stod(argv[1]);const double tau=t/tE,co=cos(alpha),si=sin(alpha),x=tau*co-u0*si,y=tau*si+u0*co;lcbinint::magnification::FiniteSourceSettings settings;settings.caustic_bins=1400;lcbinint::magnification::FiniteSourceMagnifier finite(settings);finite.binary_caustic_branches(s,q);lcbinint::magnification::PointSourceMagnifier point;auto seed=make_caustic_aware_binary_seeds(finite,point,s,q,{x,y},rho);std::cout<<std::setprecision(17)<<"count="<<seed.radial_seeds.size()<<"\n";for(const auto&z:seed.radial_seeds)std::cout<<z.re<<" "<<z.im<<"\n";}

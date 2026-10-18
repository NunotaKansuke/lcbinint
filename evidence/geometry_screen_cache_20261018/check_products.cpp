#include "lcbinint/magnification/algebraic_boundary/tangent/fused_geometry_screen.hpp"
#include <random>
#include <iostream>
using namespace lcbinint::algebraic_boundary::tangent::geometry_experiment;
int main(){std::mt19937_64 rng(42);for(int k=0;k<200000;++k){double x[4];for(auto&v:x)v=std::ldexp(std::uniform_real_distribution<double>(-1,1)(rng),int(rng()%2001)-1000);ScreenInterval a{std::min(x[0],x[1]),std::max(x[0],x[1])},b{std::min(x[2],x[3]),std::max(x[2],x[3])};auto c=a*b;__float128 q[4]={(__float128)a.lo*b.lo,(__float128)a.lo*b.hi,(__float128)a.hi*b.lo,(__float128)a.hi*b.hi};for(auto v:q)if(!((__float128)c.lo<=v&&(__float128)c.hi>=v)){std::cerr<<k;return 1;}}std::cout<<"200000 interval products enclose all quad endpoint products\n";}

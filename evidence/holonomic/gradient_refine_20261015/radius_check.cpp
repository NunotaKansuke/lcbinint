#include <fused_geometry_screen.hpp>
#include <random>
#include <iostream>
using namespace lcbinint::holonomic_fused::geometry_experiment;
int main(){std::mt19937_64 r(90423);int checked=0;for(int k=0;k<100000;++k){__float128 ex=ldexpq(__float128(1)+(r()%1000000)/1000000.Q,int(r()%1800)-900),im=ldexpq(__float128(1)+(r()%1000000)/1000000.Q,int(r()%1800)-900);double x=screen_up(double(ex)),y=screen_up(double(im));double rr=2*screen_up(std::sqrt(screen_up(screen_up(x*x)+screen_up(y*y))));__float128 exact=2*sqrtq(ex*ex+im*im);if(std::isfinite(rr)&&__float128(rr)<exact){std::cerr<<"radius underbound\n";return 1;}++checked;}std::cout<<checked<<" outward radius checks passed\n";}

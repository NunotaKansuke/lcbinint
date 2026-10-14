#include <complex>
#include <iostream>
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
using namespace lcbinint::holonomic;
int main(){int checks=0,fail=0,empty=0;
 for(double q:{1e-8,1e-4,.1,1.})for(double a:{.6,1.,2.})for(double rho:{1e-5,.1}){
 LensParams lens{.3,-.2,rho,q,a,true};auto pf=PrimaryFrame::from(lens);
 const double center=std::hypot(pf.X,pf.Y);
 for(double R:{.01,.2,.5,.9,1.,1.4,2.,4.}){
 if(!seeded_frontier_detail::radial_circle_outside_source(R,pf,center))continue;
 ++empty;
 // Independent direct lens-map evaluation in extended arithmetic.
 for(int i=0;i<256;++i){const long double theta=2*acosl(-1.L)*i/256;
 const std::complex<long double> z=(long double)R*std::complex<long double>(cosl(theta),sinl(theta));
 const auto w=z-(long double)pf.m0/std::conj(z)-(1.L-pf.m0)/(std::conj(z)-(long double)pf.a);
 ++checks;if(!(std::abs(w-std::complex<long double>(pf.X,pf.Y))>pf.rho))++fail;
 }
 }
 ++checks;if(seeded_frontier_detail::radial_circle_outside_source(0,pf,center))++fail;
 ++checks;if(seeded_frontier_detail::radial_circle_outside_source(pf.a,pf,center))++fail;
 }
 ++checks;if(empty<10)++fail;
 std::cout<<checks<<" checks "<<fail<<" failures; excluded circles "<<empty<<'\n';return fail?1:0;
}

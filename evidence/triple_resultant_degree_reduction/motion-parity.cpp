#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include "motion-before.hpp"
#include <random>
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
int main(){std::mt19937_64 rng(8623194);std::uniform_real_distribution<double>u(-1,1);int checked=0;
for(int k=0;k<5000;++k){Geometry p{u(rng),u(rng),.01+.2*std::abs(u(rng)),.2,.08,1.,-.4,.6};
 regular_support_detail::Disk disk{{2*u(rng),2*u(rng)},{},.2*std::abs(u(rng)),0};
 regular_support_detail::ArcMotionModel model(disk,p);
 for(int j=0;j<3;++j){double phi=3.2*u(rng),half=.7*std::abs(u(rng));auto a=model.evaluate(phi,half),b=motion_before::reference(disk,phi,half,p);
 if(bool(a)!=bool(b))return 1;
 if(a){if(a->first.lo!=b->first.lo || a->first.hi!=b->first.hi || a->second.lo!=b->second.lo || a->second.hi!=b->second.hi)return 2;checked+=4;}
 }
}
std::cout<<"PASS: "<<checked<<" cached motion enclosure endpoints exactly match prior arithmetic\n";}

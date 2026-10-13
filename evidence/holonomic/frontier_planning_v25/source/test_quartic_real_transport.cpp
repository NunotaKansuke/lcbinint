#include <iostream>
#include <random>
#include "lcbinint/magnification/holonomic/radius_terms.hpp"
using namespace lcbinint::holonomic;
int main(){std::mt19937_64 rng(17291);std::uniform_real_distribution<double> uniform(-8,8);int checks=0,fail=0;double maxerr=0;
for(int k=0;k<3000;++k){std::array<Cplx<double>,4> roots{};int n=(k%2)?2:4;
for(int i=0;i<n;++i)roots[i]={uniform(rng),0};if(n==2){double x=uniform(rng),y=.2+std::fabs(uniform(rng));roots[2]={x,y};roots[3]={x,-y};}
std::array<Cplx<double>,5> coef{};coef[0]={1,0};int degree=0;
for(auto root:roots){std::array<Cplx<double>,5> next{};for(int j=0;j<=degree;++j){next[j]=next[j]-coef[j]*root;next[j+1]=next[j+1]+coef[j];}coef=next;++degree;}
std::array<double,5> pc{};double scale=std::pow(10.,(k%7-3)*40.);for(int i=0;i<5;++i)pc[i]=coef[i].re*scale;
std::vector<double> expected;for(int i=0;i<n;++i){double h=std::fmod(2*std::atan(roots[i].re),kTwoPi);if(h<0)h+=kTwoPi;expected.push_back(h);}std::sort(expected.begin(),expected.end());
QuarticWarm warm;auto actual=real_root_thetas_warm(pc,warm,n);++checks;
if(actual.size()!=expected.size()){++fail;continue;}for(int i=0;i<n;++i){double err=std::fabs(actual[i]-expected[i]);maxerr=std::max(maxerr,err);if(err>1e-8)++fail;}
}
std::cout<<"checks="<<checks<<" failures="<<fail<<" max_theta_error="<<maxerr<<"\n";return fail?1:0;}

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <iostream>
using namespace lcbinint::holonomic;
int main(){
 int checks=0,failures=0,accepted=0;
 auto check=[&](bool ok,const char* msg){++checks;if(!ok){++failures;std::cerr<<msg<<'\n';}};
 for(double m:{0.,.1,1.,3.,8.})for(double width:{.001,.05,.3,1.,3.})
 for(double b0:{.1,1.,3.})for(double b2:{.01,1.,9.})for(double gate:{5e-5,5e-6}){
  const DeflatedQuad q{-.01,0.,-1.};double value=0,error=0;int evaluations=0;
  const bool ok=k_rule_budgeted(m,width*width,width,q,b0,b2,2.,value,nullptr,&error,gate,&evaluations);
  check(evaluations==3||evaluations==7||evaluations==15||evaluations==31,
        "nested refinement evaluates each point only once");
  if(!ok)continue;++accepted;
  // Independent long-double Simpson integration after x=cos(theta).
  constexpr int N=8192;long double sum=0;
  for(int i=1;i<N;++i){
   long double theta=acosl(-1.L)*i/N,si=sinl(theta),t=m+width*cosl(theta);
   long double S2=1+.01L*t*t,A=1+t*t,B=b0*b0+b2*t*t;
   sum+=(i%2?4:2)*si*si*sqrtl(S2)/(A*sqrtl(A)*sqrtl(B));
  }
  const double reference=double(2*width*width*sum*acosl(-1.L)/(3*N));
  check(std::fabs(value-reference)<=gate*std::fabs(reference)+1e-12,
        "budgeted rule agrees with independent Simpson oracle");
  check(error<=gate*std::fabs(value),"accepted difference satisfies reserved budget");
 }
 check(accepted>100,"smooth and wide arcs exercised");
 double value,error;int work;
 check(!k_rule_budgeted(0,1,1,DeflatedQuad{0,0,1},1,1,1,value,nullptr,&error,5e-5,&work),
       "negative deflated radicand rejects");
 check(!k_rule_budgeted(0,1,1,DeflatedQuad{0,0,-1},0,0,1,value,nullptr,&error,5e-5,&work),
       "zero physical denominator rejects");
 check(!k_rule_budgeted(0,1,1,DeflatedQuad{0,0,-1},1,1,1,value,nullptr,&error,NAN),
       "nonfinite budget rejects");
 AdaptiveConfig cfg;check(adaptive_detail::valid_config(cfg),"default remains valid");
 cfg.angular_value_budget_fraction=-1;check(!adaptive_detail::valid_config(cfg),"negative fraction rejected");
 cfg.angular_value_budget_fraction=NAN;check(!adaptive_detail::valid_config(cfg),"nonfinite fraction rejected");
 cfg.angular_value_budget_fraction=2;check(!adaptive_detail::valid_config(cfg),"fraction over one rejected");
 std::cout<<checks<<" checks "<<failures<<" failures; "<<accepted<<" accepted arcs\n";
 return failures?1:0;
}

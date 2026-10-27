#pragma once
#include "root_pair.hpp"
#include "patterson_rule.hpp"
#include "../nested_fejer2.hpp"
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// Global endpoint regularization after the complex-pair sinh map. Nested
// nodes retain the full local sensitivity packet. Only scalar flux chooses
// the angular order; canonical parameter RHS are pulled back afterwards.
template<int N,class Work> std::optional<std::array<double,16>> global_angular_moments(
 const root_pair_detail::Arc<N>& active,double ulo,double uhi,bool left,bool right,Work& work,double value_gate,double& error){
 constexpr int maximum=6;const double span=uhi-ulo,h=active.half_width,v=active.pair.square.value;
 const auto& q=active.pair.quotient;const FoldRadialMap normalized{0,1,left,right};
 std::array<std::array<double,16>,1<<maximum> cache;std::array<unsigned char,1<<maximum> ready{};double previous=0;
 for(int level=2;level<=maximum;++level){const auto& rule=patterson_rule(level);const int count=1<<level;std::array<double,16> total{};
  for(int k=1;k<count;++k){const int slot=k<<(maximum-level);auto& point=cache[slot];
   if(!ready[slot]){const double x=rule.x[k-1],t0=(1+x)*.5;const auto node=left&&right?std::array<double,2>{t0<=.5?t0*t0*(3-2*t0):1-(1-t0)*(1-t0)*(1+2*t0),3*t0*(1-t0)}:normalized(x);const double alpha=node[0],u=ulo+span*alpha,sh=std::sinh(u),ch=std::cosh(u),t=active.pair.mean.value+h*sh,one=1+t*t;
    double Q=q[4].value;for(int d=3;d>=0;--d)Q=Q*t+q[d].value;const double Qt=((4*q[4].value*t+3*q[3].value)*t+2*q[2].value)*t+q[1].value;
    std::array<double,2> pole{},inverse{};double denominator_slope=6*t/one;for(int d=0;d<2;++d){const auto& p=active.poles[d];pole[d]=(p[2].value*t+p[1].value)*t+p[0].value;inverse[d]=1/pole[d];denominator_slope+=(2*p[2].value*t+p[1].value)*inverse[d];}
    const double ratio=Q/(one*one*one*pole[0]*pole[1]);if(!(ratio>0)||!std::isfinite(ratio))return std::nullopt;
    const double density=2*active.inverse_rho*(-v)*ch*ch*std::sqrt(ratio),weight=span*node[1]*density,slope=.5*(Qt/Q-denominator_slope),uslope=2*sh/ch+slope*h*ch;
    point[0]=weight;point[1]=weight*slope;point[2]=weight*(1/v-slope*sh/(2*h));double power=1;for(int d=0;d<5;++d){point[3+d]=.5*weight*power/Q;power*=t;}
    for(int d=0;d<2;++d){double power=1;for(int e=0;e<3;++e){point[8+3*d+e]=-.5*weight*power*inverse[d];power*=t;}}
    point[14]=weight*(uslope*(1-alpha)-1/span);point[15]=weight*(uslope*alpha+1/span);ready[slot]=1;++work.evaluations;
   }
   for(int j=0;j<16;++j)total[j]+=rule.w[k-1]*point[j];
  }
  if(level>=4&&std::abs(total[0]-previous)<=.5*value_gate*std::abs(total[0])){error=2*std::abs(total[0]-previous);return total;}
  previous=total[0];
 }return std::nullopt;
}
}

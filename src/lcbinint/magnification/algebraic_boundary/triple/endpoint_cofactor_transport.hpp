#pragma once
#include "root_pair.hpp"
#include "../nested_fejer2.hpp"
#include "patterson_rule.hpp"
#include "dyadic_packet_cache.hpp"
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
// After the complex pair, remove the known real image-boundary factors as
// well. Their endpoint motion belongs to the integration map. Keeping these
// factors explicit avoids subtracting divergent coefficient sensitivities.
template<int N> struct EndpointCofactorIntegral {double flux=0;std::array<double,N> gradient{};};
template<int N,class Work> std::optional<EndpointCofactorIntegral<N>> endpoint_cofactor_transport(
 const root_pair_detail::Arc<N>& active,const root_pair_detail::Jet<N>& lower,const root_pair_detail::Jet<N>& upper,
 bool physical_left,bool physical_right,Work& work,int order=8,double gate=1e-8){
 using J=root_pair_detail::Jet<N>;
 const double v=active.pair.square.value,h=active.half_width,lo=lower.value,hi=upper.value,span=hi-lo;
 if(!(v<0&&h>0&&span>0&&span<=64))return std::nullopt;
 auto quotient=active.pair.quotient;int degree=4;
 const auto divide=[&](const J& root)->bool{
  std::array<J,5> next{};next[degree-1]=quotient[degree];for(int k=degree-2;k>=0;--k)next[k]=quotient[k+1]+root*next[k+1];
  const J remainder=quotient[0]+root*next[0];double scale=0,power=1;for(int k=0;k<=degree;++k){scale+=std::abs(quotient[k].value)*power;power*=std::abs(root.value);}
  if(!(std::abs(remainder.value)<=1e-8*std::max(scale,1e-300)))return false;
  quotient=next;--degree;return true;
 };
 const auto boundary=[&](const J& u){const double sh=std::sinh(u.value),ch=std::cosh(u.value);J root(active.pair.mean.value+h*sh);for(int k=0;k<N;++k)root.dp[k]=active.pair.mean.dp[k]-sh*active.pair.square.dp[k]/(2*h)+h*ch*u.dp[k];return root;};
 if(physical_left&&!divide(boundary(lower)))return std::nullopt;
 if(physical_right&&!divide(boundary(upper)))return std::nullopt;
 std::array<double,16> moments{};const int cells=std::max(1,int(std::ceil(span)));
 for(int cell=0;cell<cells;++cell){const FoldRadialMap normalized{0,1,physical_left&&cell==0,physical_right&&cell+1==cells};
  using Moment=std::array<double,16>;packet_cache_detail::DyadicPacketCache<Moment> cache;double previous=0;bool accepted=false;
  for(int level=1;level<=6;++level){const auto& rule=patterson_rule(level);cache.prepare(level);Moment total{};
  for(std::size_t k=0;k<rule.x.size();++k){const unsigned slot=packet_cache_detail::slot(level,unsigned(k+1));auto& packet=cache[slot];
   if(!cache.contains(slot)){packet.fill(0);auto& moments=packet;const auto node=normalized(rule.x[k]);const double alpha=(cell+node[0])/cells,u=lo+span*alpha,sh=std::sinh(u),ch=std::cosh(u),t=active.pair.mean.value+h*sh,one=1+t*t;
   double Q=quotient[degree].value,Qt=0;for(int d=degree-1;d>=0;--d){Qt=Qt*t+Q;Q=Q*t+quotient[d].value;}if(Q==0)return std::nullopt;
   double factor=1,dl=0,dh=0;int ends=0;
   if(physical_left){const double half=span*alpha*.5,s=std::sinh(half),c=std::cosh(half),mid=lo+half,tanh_mid=std::tanh(mid);if(s==0)return std::nullopt;factor*=2*h*s*std::cosh(mid);const double coth=c/s;dl+=-.5*alpha*coth+(1-.5*alpha)*tanh_mid;dh+=.5*alpha*(coth+tanh_mid);++ends;}
   if(physical_right){const double half=span*(alpha-1)*.5,s=std::sinh(half),c=std::cosh(half),mid=hi+half,tanh_mid=std::tanh(mid);if(s==0)return std::nullopt;factor*=2*h*s*std::cosh(mid);const double coth=c/s;dl+=.5*(1-alpha)*(coth+tanh_mid);dh+=.5*(alpha-1)*coth+.5*(alpha+1)*tanh_mid;++ends;}
   double denominator=one*one*one,denominator_slope=6*t/one;std::array<double,2> inverse{};
   for(int d=0;d<2;++d){const auto& p=active.poles[d];const double pole=(p[2].value*t+p[1].value)*t+p[0].value;if(!(pole>0))return std::nullopt;inverse[d]=1/pole;denominator*=pole;denominator_slope+=(2*p[2].value*t+p[1].value)*inverse[d];}
   const double ratio=Q*factor/denominator;if(!(ratio>0)||!std::isfinite(ratio))return std::nullopt;
   const double weight=span*node[1]/cells*2*active.inverse_rho*(-v)*ch*ch*std::sqrt(ratio),slope=.5*(Qt/Q-denominator_slope),uslope=2*sh/ch+slope*h*ch;
   moments[0]+=weight;moments[1]+=weight*slope;moments[2]+=weight*(1/v-slope*sh/(2*h)+ends/(4*v));
   double power=1;for(int d=0;d<=degree;++d){moments[3+d]+=.5*weight*power/Q;power*=t;}
   for(int d=0;d<2;++d){power=1;for(int e=0;e<3;++e){moments[8+3*d+e]-=.5*weight*power*inverse[d];power*=t;}}
   moments[14]+=weight*(uslope*(1-alpha)-1/span+.5*dl);moments[15]+=weight*(uslope*alpha+1/span+.5*dh);++work.evaluations;
   }for(int d=0;d<16;++d)total[d]+=rule.w[k]*packet[d];
  }cache.finish(level);
  if(level>=3&&std::abs(total[0]-previous)<=gate*std::abs(total[0])){for(int d=0;d<16;++d)moments[d]+=total[d];accepted=true;break;}previous=total[0];
  }if(!accepted)return std::nullopt;
 }
 EndpointCofactorIntegral<N> result;result.flux=moments[0];for(int k=0;k<N;++k){double d=-moments[0]*active.rho.dp[k]*active.inverse_rho+moments[1]*active.pair.mean.dp[k]+moments[2]*active.pair.square.dp[k]+moments[14]*lower.dp[k]+moments[15]*upper.dp[k];for(int j=0;j<=degree;++j)d+=moments[3+j]*quotient[j].dp[k];for(int j=0;j<2;++j)for(int e=0;e<3;++e)d+=moments[8+3*j+e]*active.poles[j][e].dp[k];if(!std::isfinite(d))return std::nullopt;result.gradient[k]=d;}return result;
}
}

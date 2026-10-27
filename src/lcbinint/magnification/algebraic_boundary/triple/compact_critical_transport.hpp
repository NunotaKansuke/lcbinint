#pragma once
#include "root_pair.hpp"
#include "patterson_rule.hpp"
#include "dyadic_packet_cache.hpp"
#include "../nested_fejer2.hpp"
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
template<int N> struct CompactCriticalIntegral {double flux=0,error=0;std::array<double,N> gradient{};};
// t=m+h*tan(phi). The complex-pair square root and the Jacobian supply
// sec(phi)^3, exactly cancelled by homogenizing the remaining factors.
// This integrates a bounded angular interval instead of many sinh cells.
template<int N,class Work> std::optional<CompactCriticalIntegral<N>> compact_critical_transport(
 const root_pair_detail::Arc<N>& active,const root_pair_detail::Jet<N>& old_lo,const root_pair_detail::Jet<N>& old_hi,
 bool physical_left,bool physical_right,Work& work,double gate=1e-8){
#ifdef LCBININT_TRIPLE_HYBRID_COMPACT_ANGLE
 if(std::max(std::abs(old_lo.value),std::abs(old_hi.value))>std::acosh(16.))return std::nullopt;
#endif
 using J=root_pair_detail::Jet<N>;const double h=active.half_width,v=active.pair.square.value;if(!(v<0&&h>0))return std::nullopt;
 const auto compact=[](const J& u){J phi(std::atan(std::sinh(u.value)));const double inverse=1/std::cosh(u.value);for(int j=0;j<N;++j)phi.dp[j]=inverse*u.dp[j];return phi;};
 const J lo=compact(old_lo),hi=compact(old_hi);const double span=hi.value-lo.value;if(!(span>1e-10&&span<3.15))return std::nullopt;
 using Moment=std::array<double,16>;packet_cache_detail::DyadicPacketCache<Moment> cache;Moment previous{};const FoldRadialMap normalized{0,1,physical_left,physical_right};
 for(int level=1;level<=6;++level){const auto& rule=patterson_rule(level);cache.prepare(level);Moment total{};
  for(std::size_t k=0;k<rule.x.size();++k){const auto slot=packet_cache_detail::slot(level,unsigned(k+1));auto& packet=cache[slot];
   if(!cache.contains(slot)){packet.fill(0);const auto node=normalized(rule.x[k]);const double alpha=node[0],phi=lo.value+span*alpha,c=std::cos(phi),s=std::sin(phi),y=active.pair.mean.value*c+h*s,yp=-active.pair.mean.value*s+h*c;
    std::array<double,5> cp{1,c},ypower{1,y};for(int j=2;j<5;++j){cp[j]=cp[j-1]*c;ypower[j]=ypower[j-1]*y;}
    double Q=0,Qy=0,Qc=0;for(int j=0;j<5;++j){const double q=active.pair.quotient[j].value;Q+=q*ypower[j]*cp[4-j];if(j)Qy+=j*q*ypower[j-1]*cp[4-j];if(j<4)Qc+=(4-j)*q*ypower[j]*cp[3-j];}
    const double one=c*c+y*y;if(!(Q>0&&one>0))return std::nullopt;double denominator=one*one*one,sy=-3*y/one,sc=-3*c/one;std::array<double,2> inverse{};
    for(int d=0;d<2;++d){const auto& p=active.poles[d];const double value=p[0].value*c*c+p[1].value*y*c+p[2].value*y*y;if(!(value>0))return std::nullopt;inverse[d]=1/value;denominator*=value;sy-=.5*(p[1].value*c+2*p[2].value*y)*inverse[d];sc-=.5*(2*p[0].value*c+p[1].value*y)*inverse[d];}
    const double ratio=Q/denominator;if(!(ratio>0&&std::isfinite(ratio)))return std::nullopt;const double weight=span*node[1]*2*active.inverse_rho*(-v)*std::sqrt(ratio);sy+=.5*Qy/Q;sc+=.5*Qc/Q;const double angular=sy*yp-sc*s;
    packet[0]=weight;packet[1]=weight*sy*c;packet[2]=weight*(1/v-sy*s/(2*h));for(int j=0;j<5;++j)packet[3+j]=.5*weight*ypower[j]*cp[4-j]/Q;
    for(int d=0;d<2;++d)for(int j=0;j<3;++j)packet[8+3*d+j]=-.5*weight*ypower[j]*cp[2-j]*inverse[d];
    packet[14]=weight*(angular*(1-alpha)-1/span);packet[15]=weight*(angular*alpha+1/span);++work.evaluations;
   }for(int j=0;j<16;++j)total[j]+=rule.w[k]*packet[j];
  }cache.finish(level);const double error=std::abs(total[0]-previous[0]);
  if(level>=3&&error<=gate*std::abs(total[0])){CompactCriticalIntegral<N> out;out.flux=total[0];out.error=error;for(int j=0;j<N;++j){double g=-out.flux*active.rho.dp[j]*active.inverse_rho+total[1]*active.pair.mean.dp[j]+total[2]*active.pair.square.dp[j]+total[14]*lo.dp[j]+total[15]*hi.dp[j];for(int d=0;d<5;++d)g+=total[3+d]*active.pair.quotient[d].dp[j];for(int d=0;d<2;++d)for(int e=0;e<3;++e)g+=total[8+3*d+e]*active.poles[d][e].dp[j];if(!std::isfinite(g))return std::nullopt;out.gradient[j]=g;}return out;}previous=total;
 }return std::nullopt;
}
}

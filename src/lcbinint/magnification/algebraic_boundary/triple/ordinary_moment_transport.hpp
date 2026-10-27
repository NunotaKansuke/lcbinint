#pragma once
#include "root_pair_angular.hpp"
#include "root_pair_reverse.hpp"
#include "dyadic_packet_cache.hpp"
namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail {
inline thread_local std::array<std::size_t,8> ordinary_moment_counts{};
// Ordinary arc: integrate scalar local coefficient sensitivities together
// with flux; apply the eight canonical RHS once, after angular integration.
// A failed scalar/physical consistency guard keeps the incumbent kernel.
template<class EndpointType,class Core,class Work> std::optional<PairIntegral<8>> ordinary_moment_transport(
 double R,const Geometry& p,const EndpointType& lo,const EndpointType& hi,const std::array<double,8>& motion,const Core& midpoint,double gate,Work& work,
 std::optional<root_pair_detail::Coefficients<8>>& canonical,std::optional<root_pair_detail::Coefficients<0>>& scalar,std::unique_ptr<root_pair_detail::RadialCache<8>>* tape,bool concave=true){
 ++ordinary_moment_counts[0];using namespace root_pair_detail;const double width=hi.theta-lo.theta;if(!(width>0&&width<1.9*std::acos(-1.)&&midpoint.phi>0&&midpoint.roundoff<1e-7*midpoint.phi))return std::nullopt;
 if(!canonical){++ordinary_moment_counts[1];if(tape&&!*tape)*tape=std::make_unique<RadialCache<8>>(p);canonical.emplace(R,p,motion,tape?tape->get():nullptr);}
 if(!scalar){Coefficients<0> data;data.R=R;data.rho=p.rho;data.lx1=p.a;data.lx2=p.bx;data.ly2=p.by;for(int k=0;k<7;++k)data.harmonics[k].value=canonical->harmonics[k].value;scalar.emplace(std::move(data));}
 Arc<0> model;const double basis=(lo.theta+hi.theta)*.5,h=std::tan(width*.25),v=h*h;if(!(h>0&&h<16&&v>0)||!model.configure(R,p,basis,{},&scalar)||!model.pair.finish(model.polynomial,0,v))return std::nullopt;model.half_width=h;
 ++ordinary_moment_counts[2];const double predicted=model.phi_from_factor(0,-v);if(std::abs(predicted-midpoint.phi)+midpoint.roundoff>1e-7*midpoint.phi)return std::nullopt;
 ++ordinary_moment_counts[3];std::array<double,7> polynomial;for(int k=0;k<7;++k)polynomial[k]=model.polynomial[k].value;const auto residual=equations(polynomial,0,v);if(std::abs(residual.e)+h*std::abs(residual.o)>1e-9*v*std::abs(model.pair.quotient[0].value))return std::nullopt;
 ++ordinary_moment_counts[4];++ordinary_moment_counts[5];using Moments=std::array<double,14>;struct Point{Moments moments;};packet_cache_detail::DyadicPacketCache<Point> cache;
 const int minimum=concave||model.disk_resolved(4)?3:model.disk_resolved(2)?4:5;
 double previous=0;for(int level=2;level<=8;++level){const int n=1<<level;const auto& rule=angular_rule_extended(level);cache.prepare(level);Moments total{};
  for(int k=1;k<n;++k){const auto slot=packet_cache_detail::slot(level,k);auto& point=cache[slot];if(!cache.contains(slot)){const double x=rule.x[k-1],t=h*x,one=1+t*t;const auto& q=model.pair.quotient;double Q=q[4].value;for(int d=3;d>=0;--d)Q=Q*t+q[d].value;const double Qt=((4*q[4].value*t+3*q[3].value)*t+2*q[2].value)*t+q[1].value,iq=1/Q;if(!std::isfinite(iq))return std::nullopt;
    std::array<double,2> poles,inverse;double denominator=one*one*one,slope=6*t/one;for(int d=0;d<2;++d){const auto& b=model.poles[d];poles[d]=(b[2].value*t+b[1].value)*t+b[0].value;inverse[d]=1/poles[d];denominator*=poles[d];slope+=(2*b[2].value*t+b[1].value)*inverse[d];}
    const double ratio=-Q/denominator;if(!(ratio>0)||!std::isfinite(ratio))return std::nullopt;const double f=2*model.inverse_rho*v*std::sqrt(ratio),s=.5*(Qt*iq-slope);auto& m=point.moments;m[0]=f;m[1]=f*s;m[2]=f*(1/v+s*x/(2*h));double power=1;for(int d=0;d<5;++d){m[3+d]=.5*f*iq*power;power*=t;}for(int d=0;d<2;++d){power=1;for(int e=0;e<3;++e){m[8+3*d+e]=-.5*f*inverse[d]*power;power*=t;}}++work.geometry_evaluations;++ordinary_moment_counts[6];
   }for(int d=0;d<14;++d)total[d]+=rule.norm_w[k-1]*point.moments[d];}
  cache.finish(level);const double error=std::abs(total[0]-previous);if(level>=minimum&&error<=gate*std::abs(total[0])){std::array<double,5> aq;for(int d=0;d<5;++d)aq[d]=total[3+d];const auto adjoint=quotient_adjoint(model,total[1],total[2],aq);std::array<std::array<double,3>,2> pole;for(int d=0;d<2;++d)for(int e=0;e<3;++e)pole[d][e]=total[8+3*d+e];PairIntegral<8> result;result.flux=total[0];result.error=error;result.gradient=coefficient_adjoint(*canonical,basis,adjoint.polynomial,pole,-total[0]*model.inverse_rho);
   const double endpoint_scale=.5*(1+v);for(int j=0;j<8;++j){const double l=endpoint_scale*(lo.motion[j]+lo.radial_motion*motion[j]),r=endpoint_scale*(hi.motion[j]+hi.radial_motion*motion[j]);result.gradient[j]+=adjoint.mean*.5*(l+r)+adjoint.square*h*(r-l);}
   for(double d:result.gradient)if(!std::isfinite(d))return std::nullopt;++work.evaluations;++ordinary_moment_counts[7];return result;}previous=total[0];
 }return std::nullopt;
}
}

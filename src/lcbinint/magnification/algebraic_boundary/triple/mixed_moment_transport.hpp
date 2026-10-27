#pragma once
#include "fused_pair.hpp"
#include "dyadic_packet_cache.hpp"
#include "../nested_fejer2.hpp"
namespace lcbinint::algebraic_boundary::triple::fused_detail {
// The integration nodes carry one spatial direction and local coefficient
// sensitivities. Canonical parameters and their mixed spatial derivatives
// are contracted only after integration, sharing the scalar factorization.
class MixedMomentTransport {
 using J=root_pair_detail::Jet<1>;using Local=std::array<J,16>;
 struct Point {Local moments;};
 const MixedPair* model_;bool critical_;int side_;Mixed lower_,upper_;
 J rho_,mean_,square_;std::array<J,5> quotient_;std::array<std::array<J,3>,2> poles_;J lower_jet_,span_;
 packet_cache_detail::DyadicPacketCache<Point> cache_;
 static J spatial(const Mixed& x){J j(x.value);j.dp[0]=x.r;return j;}
 static J scale(const J& x,double s){return root_pair_detail::linear_scale(x,s);}
 static J reciprocal(const J& x){return root_pair_detail::inverse(x);}
 static const Fejer2Rule& rule(int level){if(level<=8)return fejer_rule(level);if(level==9){static const Fejer2Rule q(9);return q;}if(level==10){static const Fejer2Rule q(10);return q;}throw std::invalid_argument("mixed moment level");}
public:
 MixedMomentTransport(const MixedPair& model):MixedMomentTransport(model,Mixed(),Mixed(1),false,false,0,false){}
 MixedMomentTransport(const MixedPair& model,Mixed lower,Mixed upper,bool left,bool right,int side=0,bool critical=true):model_(&model),critical_(critical),side_(side),lower_(lower),upper_(upper),rho_(spatial(model.rho)),mean_(spatial(model.mean)),square_(spatial(model.square)),lower_jet_(spatial(lower)),span_(spatial(upper-lower)),left_(left),right_(right){for(int k=0;k<5;++k)quotient_[k]=spatial(model.quotient[k]);for(int d=0;d<2;++d)for(int k=0;k<3;++k)poles_[d][k]=spatial(model.poles[d][k]);}
 bool left_,right_;
 bool sample_local(double x,Local& moments)const{
  const auto node=critical_?FoldRadialMap{0,1,left_,right_}(x):std::array<double,2>{0,1};
  J sh,ch,t,h,weight,moment_square,u_slope;
  if(critical_){const auto u=lower_jet_+scale(span_,node[0]);sh=J(std::sinh(u.value));ch=J(std::cosh(u.value));sh.dp[0]=ch.value*u.dp[0];ch.dp[0]=sh.value*u.dp[0];h=root_pair_detail::root(side_?square_:-square_);t=side_?mean_+scale(h*ch,side_):mean_+h*sh;}
  else{h=root_pair_detail::root(square_);t=mean_+scale(h,x);}
  const auto one=J(1)+t*t;auto q=quotient_[4];for(int k=3;k>=0;--k)q=q*t+quotient_[k];
  const auto qt=((scale(quotient_[4],4)*t+scale(quotient_[3],3))*t+scale(quotient_[2],2))*t+quotient_[1];
  const auto iq=reciprocal(q);auto denominator=one*one*one;auto denominator_slope=scale(t*reciprocal(one),6);
  std::array<J,2> inverse_pole;for(int d=0;d<2;++d){const auto& p=poles_[d];const auto pole=(p[2]*t+p[1])*t+p[0];denominator=denominator*pole;inverse_pole[d]=reciprocal(pole);denominator_slope=denominator_slope+(scale(p[2]*t,2)+p[1])*inverse_pole[d];}
  const auto ratio=(critical_?q:-q)*reciprocal(denominator);if(!(ratio.value>0)||!std::isfinite(ratio.value))return false;
  auto f=scale(reciprocal(rho_)*root_pair_detail::root(ratio),2);
  const auto slope=scale(qt*iq-denominator_slope,.5);
  if(critical_){f=f*(side_?square_*sh*sh:(-square_)*ch*ch);weight=f*scale(span_,node[1]);
   moment_square=reciprocal(square_)+(side_?scale(slope*ch*reciprocal(h),.5*side_):-scale(slope*sh*reciprocal(h),.5));
   u_slope=side_?scale(ch*reciprocal(sh),2)+scale(slope*h*sh,side_):scale(sh*reciprocal(ch),2)+slope*h*ch;
  }else{f=f*square_;weight=f;moment_square=reciprocal(square_)+scale(slope*reciprocal(h),x*.5);}
  moments={};moments[0]=weight;moments[1]=weight*slope;moments[2]=weight*moment_square;auto power=J(1);const auto qw=scale(weight*iq,.5);
  for(int k=0;k<5;++k){moments[3+k]=qw*power;power=power*t;}
  for(int d=0;d<2;++d){power=J(1);const auto pw=scale(weight*inverse_pole[d],-.5);for(int k=0;k<3;++k){moments[8+3*d+k]=pw*power;power=power*t;}}
  if(critical_){moments[14]=weight*(scale(u_slope,1-node[0])-reciprocal(span_));moments[15]=weight*(scale(u_slope,node[0])+reciprocal(span_));}
  for(const auto& m:moments)if(!std::isfinite(m.value)||!std::isfinite(m.dp[0]))return false;return true;
 }
 Mixed pullback(const Local& moments)const{
  Mixed out;out.value=moments[0].value;out.r=moments[0].dp[0];
  const auto add=[&](const J& adjoint,const Mixed& coefficient){for(int k=0;k<8;++k){out.p[k]+=adjoint.value*coefficient.p[k];out.rp[k]+=adjoint.dp[0]*coefficient.p[k]+adjoint.value*coefficient.rp[k];}};
  add(-moments[0]*reciprocal(rho_),model_->rho);add(moments[1],model_->mean);add(moments[2],model_->square);
  for(int k=0;k<5;++k)add(moments[3+k],model_->quotient[k]);for(int d=0;d<2;++d)for(int k=0;k<3;++k)add(moments[8+3*d+k],model_->poles[d][k]);
  if(critical_){add(moments[14],lower_);add(moments[15],upper_);}return out;
 }
 bool integrate(int level,Mixed& out,std::size_t& evaluations){
  if(level<1||level>10)return false;cache_.prepare(level);Local total{};const auto& q=rule(level);const int n=1<<level;
  for(int k=1;k<n;++k){const unsigned slot=packet_cache_detail::slot(level,k);auto& point=cache_[slot];if(!cache_.contains(slot)){if(!sample_local(q.x[k-1],point.moments))return false;++evaluations;}
   const double weight=critical_?q.w[k-1]:q.norm_w[k-1];for(int j=0;j<16;++j){total[j].value+=weight*point.moments[j].value;total[j].dp[0]+=weight*point.moments[j].dp[0];}}
  cache_.finish(level);out=pullback(total);return std::isfinite(out.value)&&std::isfinite(out.r);
 }
};
}

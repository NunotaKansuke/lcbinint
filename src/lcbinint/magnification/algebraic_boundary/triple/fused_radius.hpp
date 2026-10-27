#pragma once
#include "coupled_profile.hpp"
#include "geometry.hpp"
#include "angular_topology.hpp"
#include "fused_pair.hpp"
#ifdef LCBININT_TRIPLE_MIXED_MOMENTS
#include "mixed_moment_transport.hpp"
#endif
#include "local_precision.hpp"
#include "../nested_fejer2.hpp"
#include <memory>
#include <type_traits>
#include <vector>
namespace lcbinint::algebraic_boundary::triple::fused_detail {
using Q=local_precision::Q;
inline thread_local std::array<std::size_t,6> transport_counts{};
inline const Fejer2Rule& angular_rule(int level){if(level<=8)return fejer_rule(level);if(level==9){static const Fejer2Rule r(9);return r;}if(level==10){static const Fejer2Rule r(10);return r;}throw std::invalid_argument("fused angular level");}
template<class T> T square_root(T x){if constexpr(std::is_same_v<T,Q>)return sqrtq(x);else return std::sqrt(x);}
template<class T> T absolute(T x){if constexpr(std::is_same_v<T,Q>)return fabsq(x);else return std::abs(x);}
template<class T> T epsilon(){if constexpr(std::is_same_v<T,Q>)return ldexpq(Q(1),-112);else return std::numeric_limits<T>::epsilon();}
template<class T> struct Jet {
 T phi=0,r=0,t=0,rr=0,rt=0,tt=0,roundoff=0;
 std::array<T,8> p{},rp{},tp{};
};
// One physical field evaluation supplies value, parameter RHS and the radial
// spatial jet. No completed value integral or subsequent tangent pass exists.
template<class T> class Ray {
 using C=coefficient_detail::C<T>;
 T R,m0,m1,m2,rho,inv_rho2,radial,primary,source_size;
 C source,l1,l2;
 static T norm(C z){return z.re*z.re+z.im*z.im;}
 static T dot(C a,C b){return a.re*b.re+a.im*b.im;}
 static C conjugate(C z){return {z.re,-z.im};}
 static C inverse_conj(C z){const T s=norm(z);return {z.re/s,z.im/s};}

public:
 Ray(double radius,const Geometry& p):R(radius),m0(T(1)-T(p.m1)-T(p.m2)),m1(p.m1),m2(p.m2),rho(p.rho),inv_rho2(T(1)/(rho*rho)),radial(R-m0/R),primary(m0/R),source_size(square_root(T(p.X)*T(p.X)+T(p.Y)*T(p.Y))),source{T(p.X),T(p.Y)},l1{T(p.a)},l2{T(p.bx),T(p.by)}{}
 template<bool Mixed=true> Jet<T> sample(T theta)const {
  T sn,cs;if constexpr(std::is_same_v<T,Q>)sincosq(theta,&sn,&cs);else{sn=std::sin(theta);cs=std::cos(theta);}
  const C w{cs,sn},z{R*cs,R*sn},I{0,1},wb=conjugate(w),zb=conjugate(z);
  const C i0{cs/R,sn/R},i1=inverse_conj(z-l1),i2=inverse_conj(z-l2);
  const C q0=i0*i0,q1=i1*i1,q2=i2*i2;
  const C g=C{radial*cs-source.re,radial*sn-source.im}-C{m1}*i1-C{m2}*i2;
  const C shear=C{m0}*q0+C{m1}*q1+C{m2}*q2,gr=w+wb*shear,gt=I*(z-zb*shear);
  const T inv2=T(-2)*inv_rho2,normg=norm(g);
  Jet<T> out;out.phi=T(1)-normg*inv_rho2;out.r=inv2*dot(g,gr);out.t=inv2*dot(g,gt);
  const T deflection=primary+m1*square_root(norm(i1))+m2*square_root(norm(i2));
  const T shear_bound=m0/(R*R)+m1*norm(i1)+m2*norm(i2),delta=T(32)*epsilon<T>()*(R+source_size+deflection+R*shear_bound);
  out.roundoff=(T(2)*square_root(normg)*delta+delta*delta)*inv_rho2+T(16)*epsilon<T>()*(T(1)+normg*inv_rho2);
  if constexpr(Mixed) {
   const C c0=q0*i0,c1=q1*i1,c2=q2*i2,cubic=C{m0}*c0+C{m1}*c1+C{m2}*c2;
   const C grr=C{T(-2)}*wb*wb*cubic,grt=I*(w-wb*shear+C{T(2)}*zb*wb*cubic),gtt=C{T(-1)}*z-zb*shear+C{T(2)}*zb*zb*cubic;
   out.rr=inv2*(norm(gr)+dot(g,grr));out.rt=inv2*(dot(gr,gt)+dot(g,grt));out.tt=inv2*(norm(gt)+dot(g,gtt));
   out.p[0]=T(2)*g.re*inv_rho2;out.p[1]=T(2)*g.im*inv_rho2;out.p[2]=T(2)*normg*inv_rho2/rho;
   out.rp[0]=T(2)*gr.re*inv_rho2;out.rp[1]=T(2)*gr.im*inv_rho2;out.rp[2]=T(-2)*out.r/rho;
   out.tp[0]=T(2)*gt.re*inv_rho2;out.tp[1]=T(2)*gt.im*inv_rho2;out.tp[2]=T(-2)*out.t/rho;
   const std::array<C,5> gp{i0-i1,i0-i2,C{-m1}*q1,C{-m2}*q2,I*C{m2}*q2};
   const std::array<C,5> grp{C{T(-1)}*wb*(q0-q1),C{T(-1)}*wb*(q0-q2),C{T(2)*m1}*wb*c1,C{T(2)*m2}*wb*c2,C{T(-2)}*I*C{m2}*wb*c2};
   for(int j=0;j<5;++j){out.p[j+3]=inv2*dot(g,gp[j]);out.rp[j+3]=inv2*(dot(gr,gp[j])+dot(g,grp[j]));out.tp[j+3]=inv2*(dot(gt,gp[j])+R*(g.re*grp[j].im-g.im*grp[j].re));}
  }
  return out;
 }
};
template<class T> struct End {T theta=0,radial=0,slope=0;std::array<T,8> parameter{},mixed{};};
template<class T> bool endpoint(const Ray<T>& ray,double initial,double width,const std::array<double,8>& v,const std::array<double,8>& slope,End<T>& out) {
 T theta=initial;bool resolved=false;
 for(int k=0;k<12;++k) {
  const auto q=ray.template sample<false>(theta);const T step=q.phi/q.t;
  if(!std::isfinite(double(step)) || absolute(theta-T(initial))+absolute(step)>T(.25*width))return false;
  theta-=step;
  if(absolute(step)<std::max(T(32)*epsilon<T>()*(1+absolute(theta)),T(2)*q.roundoff/absolute(q.t))){resolved=true;break;}
 }
 if(!resolved)return false;
 const auto q=ray.sample(theta);
 if(!(q.roundoff<absolute(q.t)*T(width)*T(1e-7)) || q.t==0)return false;
 out.theta=theta;out.slope=q.t;const T invt=T(1)/q.t,tr=-q.r*invt;out.radial=tr;
 for(int j=0;j<8;++j) {
  const T tp=-(q.p[j]+q.r*T(v[j]))*invt;
  out.parameter[j]=tp;
  out.mixed[j]=-(q.rp[j]+q.tp[j]*tr+(q.rr+q.rt*tr)*T(v[j])+q.r*T(slope[j])+(q.rt+q.tt*tr)*tp)*invt;
 }
 return true;
}
struct Density {double flux=0,radial=0,error=0;std::array<double,8> p{},rp{};bool ready=false;};
template<class T> Density density(const Jet<T>& q,T R,const End<T>& lo,const End<T>& hi,double x,bool full,const std::array<double,8>& velocity,const std::array<double,8>& slope) {
 const T u=T((1+x)*.5),shape=full?T(1):T((1-x)*(1+x)),half=(hi.theta-lo.theta)/2,hr=(hi.radial-lo.radial)/2;
 const T B=lo.radial*(1-u)+hi.radial*u,qr=q.r+q.t*B,root=square_root(q.phi/shape),ir=T(.5)/(root*shape),rootr=qr*ir;
 const T U=R*half,Ur=half+R*hr;
 Density out;out.flux=double(U*root);out.radial=double(Ur*root+U*rootr);out.error=double(absolute(U*root)*q.roundoff/q.phi);
 for(int j=0;j<8;++j) {
  const T hp=(hi.parameter[j]-lo.parameter[j])/2,hrp=(hi.mixed[j]-lo.mixed[j])/2;
  const T A=lo.parameter[j]*(1-u)+hi.parameter[j]*u,C=lo.mixed[j]*(1-u)+hi.mixed[j]*u;
  const T qp=q.p[j]+q.r*T(velocity[j])+q.t*A;
  const T qrp=q.rp[j]+q.tp[j]*B+(q.rr+q.rt*B)*T(velocity[j])+q.r*T(slope[j])+(q.rt+q.tt*B)*A+q.t*C;
  const T rootp=qp*ir,rootrp=qrp*ir-qr*qp/(4*root*root*root*shape*shape),Up=T(velocity[j])*half+R*hp;
  const T Urp=T(slope[j])*half+T(velocity[j])*hr+hp+R*hrp;
  out.p[j]=double(Up*root+U*rootp);out.rp[j]=double(Urp*root+Up*rootr+Ur*rootp+U*rootrp);
 }
 out.ready=true;return out;
}
struct Result {double f0=0,fh=0,r0=0,rh=0,error=0,angular_error=0,radial_angular_error=0;std::array<double,8> p0{},ph{},rp0{},rph{};};
class Radius {
 Geometry geometry_;double R_;std::array<double,8> velocity_,slope_;Ray<double> ray_;
 std::unique_ptr<Ray<Q>> precise_;
 std::shared_ptr<root_pair_detail::PreparedFrames> frames_;double relative_gate_=0,radial_scale_=1;
 struct Cell {double center,scale,lo,hi;bool left,right;std::vector<Density> samples;std::shared_ptr<MixedPair> critical;Mixed moving_lo,moving_hi;int real_side=0;
#ifdef LCBININT_TRIPLE_MIXED_MOMENTS
  std::unique_ptr<MixedMomentTransport> moments;bool moments_declined=false;
#endif
 };
 struct Arc {double seed_lo=0,seed_hi=0;End<double> lo,hi;std::unique_ptr<std::array<End<Q>,2>> wide;std::vector<Density> samples;bool chart_checked=false;std::vector<Cell> cells;std::unique_ptr<MixedPair> pair;std::array<Density,11> totals{};
#ifdef LCBININT_TRIPLE_MIXED_MOMENTS
  std::unique_ptr<MixedMomentTransport> moments;bool moments_declined=false;
#endif
 };
 std::vector<Arc> arcs_;bool full_=false,ready_=false;
 struct Witness {double theta,phi,r,t,error;};std::array<Witness,48> witnesses_{};std::size_t witness_count_=0;
 bool precise(Arc& a) {
  if(a.wide)return true;
  if(!precise_)precise_=std::make_unique<Ray<Q>>(R_,geometry_);
  a.wide=std::make_unique<std::array<End<Q>,2>>();
  if(full_){(*a.wide)[0].theta=Q(a.lo.theta);(*a.wide)[1].theta=Q(a.hi.theta);return true;}
  const double width=a.hi.theta-a.lo.theta;
  if(!endpoint(*precise_,a.seed_lo,width,velocity_,slope_,(*a.wide)[0]) || !endpoint(*precise_,a.seed_hi,width,velocity_,slope_,(*a.wide)[1])){a.wide.reset();return false;}
  return (*a.wide)[0].slope>0 && (*a.wide)[1].slope<0;
 }
 bool strictly_concave(double lo,double hi)const {
  const double half=(hi-lo)*.5,mid=lo+half,pad=1e-12;
  if(!(half>0&&half<1))return false;
  const Complex w(std::cos(mid),std::sin(mid)),z=R_*w;
  const double radial=R_-(1-geometry_.m1-geometry_.m2)/R_;
  Complex g=radial*w-Complex(geometry_.X,geometry_.Y),shear{};
  double first=std::abs(radial),second=first;
  for(const auto& lens:std::array<std::pair<Complex,double>,2>{{{Complex(geometry_.a),geometry_.m1},{Complex(geometry_.bx,geometry_.by),geometry_.m2}}}) {
   if(lens.second==0)continue;const Complex delta=z-lens.first;
   const double distance=std::abs(delta)-R_*half-pad*(1+R_+std::abs(lens.first));if(!(distance>0))return false;
   const Complex inverse=inverse_conjugate(delta);g-=lens.second*inverse;shear+=lens.second*inverse*inverse;
   const double bound=lens.second*R_/(distance*distance);first+=bound;second+=bound*(1+2*R_/distance);
  }
  const Complex gt=Complex(0,1)*(radial*w-std::conj(z)*shear);first*=1+pad;second*=1+pad;
  const double lower=(1-pad)*std::abs(gt)-half*second-pad,upper=(1+pad)*std::abs(g)+half*first+pad;
  return lower>0&&lower*lower>(1+pad)*upper*second;
 }
 bool angular_chart(Arc& a) {
  if(a.chart_checked)return true;LCBININT_COUPLED_SCOPE(chart);a.chart_checked=true;
  const double lo=a.lo.theta,hi=a.hi.theta,width=hi-lo;
  struct Minimum {double center,scale;};std::vector<Minimum> minima;
  double left=lo,previous=ray_.sample<false>(lo).t;
  for(int k=1,scans=strictly_concave(lo,hi)?0:32;k<=scans;++k){const double right=lo+width*k/32;const auto field=ray_.sample<false>(right);++evaluations;
   if(previous<0&&field.t>0){double low=left,high=right,c=(low+high)*.5;Jet<double> q;
    for(int iteration=0;iteration<24;++iteration){q=ray_.sample(c);++evaluations;if(q.t<0)low=c;else high=c;const double next=c-q.t/q.tt;if(std::abs(next-c)<1e-13*(1+std::abs(c)))break;c=std::isfinite(next)&&next>low&&next<high?next:(low+high)*.5;}
    q=ray_.sample(c);++evaluations;double phi=q.phi;
    if(!(phi>0)||q.roundoff>1e-5*phi){if(!precise_)precise_=std::make_unique<Ray<Q>>(R_,geometry_);phi=double(precise_->sample(Q(c)).phi);++promoted;}
    const double scale=std::sqrt(2*phi/q.tt);if(phi>0&&q.tt>0&&scale<width*.15&&std::isfinite(scale))minima.push_back({c,scale});
   }left=right;previous=field.t;
  }
  for(std::size_t j=0;j<minima.size();++j){const auto m=minima[j];const double lower=j?(minima[j-1].center+m.center)*.5:lo,upper=j+1<minima.size()?(m.center+minima[j+1].center)*.5:hi;
   ++transport_counts[0];auto critical=std::make_shared<MixedPair>();
   if(critical->initialize(R_,geometry_,m.center-m.scale,m.center+m.scale,velocity_,slope_,nullptr,m.scale)){
    const auto endpoint_jet=[](const End<double>& e){Mixed q(e.theta);q.r=e.radial;q.p=e.parameter;q.rp=e.mixed;return q;};
    const auto low=critical->critical_coordinate(j?Mixed(lower):endpoint_jet(a.lo),m.center),high=critical->critical_coordinate(j+1<minima.size()?Mixed(upper):endpoint_jet(a.hi),m.center);
    if(high.value>low.value&&high.value-low.value<64){const int pieces=std::max(1,int(std::ceil(high.value-low.value)));bool valid=true;
     for(double x:{-.5,0.,.5}){Mixed q;if(!critical->critical_density(low+(high-low)*Mixed((1+x)*.5),q))valid=false;}
     if(valid){++transport_counts[1];for(int k=0;k<pieces;++k){Cell cell{m.center,m.scale,0,0,!full_&&j==0&&k==0,!full_&&j+1==minima.size()&&k+1==pieces,{}};cell.critical=critical;cell.moving_lo=low+(high-low)*Mixed(double(k)/pieces);cell.moving_hi=low+(high-low)*Mixed(double(k+1)/pieces);a.cells.push_back(std::move(cell));}continue;}
    }
   }
   const double tl=std::asinh((lower-m.center)/m.scale),th=std::asinh((upper-m.center)/m.scale);if(!(th>tl)||th-tl>64)return false;const int n=std::max(1,int(std::ceil(th-tl)));
   for(int k=0;k<n;++k)a.cells.push_back({m.center,m.scale,tl+(th-tl)*k/n,tl+(th-tl)*(k+1)/n,!full_&&j==0&&k==0,!full_&&j+1==minima.size()&&k+1==n,{}});
  }
  // A nearby *real* gap is the other side of the same fold. Pair its
  // two boundaries, rather than the distant ends of the positive arc.
  // cosh transport removes that exterior square root before differentiation.
  if(minima.empty()&&!full_&&arcs_.size()>1){const std::size_t index=&a-arcs_.data(),previous=(index+arcs_.size()-1)%arcs_.size(),next=(index+1)%arcs_.size();
   const double pi=std::acos(-1.);double previous_hi=arcs_[previous].hi.theta,next_lo=arcs_[next].lo.theta;
   previous_hi+=2*pi*std::floor((lo-previous_hi)/(2*pi));next_lo+=2*pi*std::ceil((hi-next_lo)/(2*pi));
   const double left_gap=lo-previous_hi,right_gap=next_lo-hi;const bool left_close=left_gap>0&&left_gap<width*.5,right_close=right_gap>0&&right_gap<width*.5;
   const auto end_jet=[](const End<double>& e){Mixed q(e.theta);q.r=e.radial;q.p=e.parameter;q.rp=e.mixed;return q;};
   const auto build=[&](double lower,double upper,bool left,bool right,int side,double gl,double gh){
    const double basis=(gl+gh)*.5;if(std::max(std::abs(lower-basis),std::abs(upper-basis))>pi-.05)return false;
    auto model=std::make_shared<MixedPair>();if(!model->initialize(R_,geometry_,gl,gh,velocity_,slope_,nullptr,0,true))return false;
    Mixed low=side>0?Mixed(0):model->critical_coordinate(lower==lo?end_jet(a.lo):Mixed(lower),basis,side);
    Mixed high=side<0?Mixed(0):model->critical_coordinate(upper==hi?end_jet(a.hi):Mixed(upper),basis,side);
    if(side<0){std::swap(low,high);std::swap(left,right);}
    if(!(high.value>low.value&&high.value-low.value<64)||!std::isfinite(high.r)||!std::isfinite(low.r))return false;
    const int pieces=std::max(1,int(std::ceil(high.value-low.value)));for(int k=0;k<pieces;++k){Cell cell{basis,0,0,0,left&&k==0,right&&k+1==pieces,{}};cell.critical=model;cell.real_side=side;cell.moving_lo=low+(high-low)*Mixed(double(k)/pieces);cell.moving_hi=low+(high-low)*Mixed(double(k+1)/pieces);Mixed probe;if(!model->critical_density((cell.moving_lo+cell.moving_hi)*Mixed(.5),probe,side))return false;a.cells.push_back(std::move(cell));}++transport_counts[4];return true;
   };
   bool okay=false;if(left_close&&right_close){const double mid=(lo+hi)*.5;okay=build(lo,mid,true,false,1,previous_hi,lo)&&build(mid,hi,false,true,-1,hi,next_lo);}
   else if(left_close)okay=build(lo,hi,true,true,1,previous_hi,lo);else if(right_close)okay=build(lo,hi,true,true,-1,hi,next_lo);
   if(okay)return true;a.cells.clear();
  }
  #ifndef LCBININT_TRIPLE_DISABLE_FUSED_PAIR
  if(minima.empty()&&!full_){++transport_counts[2];auto pair=std::make_unique<MixedPair>();if(pair->initialize(R_,geometry_,lo,hi,velocity_,slope_,frames_.get())){
   // Arithmetic consistency guard, not a parameter-error mesh controller:
   // the polynomial and physical endpoint forms represent the same area jet.
   // A cancellation-damaged coefficient packet retains the physical kernel.
   const auto area=pair->uniform();const double wr=a.hi.radial-a.lo.radial;
   const auto close=[](double x,double y){return std::abs(x-y)<=1e-7*(1+std::abs(y));};
   bool okay=close(area.value,R_*width)&&close(area.r,width+R_*wr);
   for(int j=0;j<8&&okay;++j){const double wp=a.hi.parameter[j]-a.lo.parameter[j],wrp=a.hi.mixed[j]-a.lo.mixed[j];okay=close(area.p[j],velocity_[j]*width+R_*wp)&&close(area.rp[j],slope_[j]*width+velocity_[j]*wr+wp+R_*wrp);}
   if(okay){++transport_counts[3];a.pair=std::move(pair);}
  }}
  #endif
  return true;
 }
 bool point(Arc& a,double x,Density& out) {
  if(out.ready)return true;
  const double theta=a.lo.theta+(a.hi.theta-a.lo.theta)*((1+x)*.5);const auto q=ray_.sample(theta);++evaluations;
  if(q.phi>0&&q.roundoff<1e-5*q.phi)out=density(q,R_,a.lo,a.hi,x,full_,velocity_,slope_);
  else{if(!precise(a))return false;const auto& lo=(*a.wide)[0];const auto& hi=(*a.wide)[1];const Q tq=lo.theta+(hi.theta-lo.theta)*Q((1+x)*.5);const auto pq=precise_->sample(tq);++promoted;if(!(pq.phi>0))return false;out=density(pq,Q(R_),lo,hi,x,full_,velocity_,slope_);}return true;
 }
 static void grow(std::vector<Density>& samples,int n){if(samples.size()<std::size_t(n)){const auto old=samples.size();samples.resize(n);for(std::size_t j=old;j-->0;){samples[2*j]=std::move(samples[j]);if(2*j!=j)samples[j]={};}}}
 bool limb_integral(Arc& a,int level,Density& out){
  if(a.totals[level].ready){out=a.totals[level];return true;}
  out={};const auto& rule=angular_rule(level);const int n=1<<level;const double width=a.hi.theta-a.lo.theta;
  const auto add=[&](const Density& point,double weight){out.flux+=weight*point.flux;out.radial+=weight*point.radial;out.error+=weight*point.error;for(int j=0;j<8;++j){out.p[j]+=weight*point.p[j];out.rp[j]+=weight*point.rp[j];}};
   if(a.cells.empty()) {
#ifdef LCBININT_TRIPLE_MIXED_MOMENTS
    bool used_moments=false;
    if(a.pair&&!a.moments_declined){if(!a.moments)a.moments=std::make_unique<MixedMomentTransport>(*a.pair);Mixed q;if(a.moments->integrate(level,q,evaluations)){Density packet;packet.flux=q.value;packet.radial=q.r;packet.p=q.p;packet.rp=q.rp;packet.error=64*std::numeric_limits<double>::epsilon()*std::abs(q.value);add(packet,1);used_moments=true;}else a.moments_declined=true;}
    if(!used_moments){
#endif
    grow(a.samples,n);
    for(int k=1;k<n;++k){const double x=rule.x[k-1];auto& sample=a.samples[k*(int(a.samples.size())/n)];if(!sample.ready&&a.pair){Mixed q;if(!a.pair->density(x,q))return false;sample.flux=q.value;sample.radial=q.r;sample.p=q.p;sample.rp=q.rp;sample.error=64*std::numeric_limits<double>::epsilon()*std::abs(q.value);sample.ready=true;++evaluations;}if(!sample.ready&&!point(a,x,sample))return false;add(sample,full_?rule.w[k-1]:rule.norm_w[k-1]);}
#ifdef LCBININT_TRIPLE_MIXED_MOMENTS
    }
#endif
   } else for(auto& cell:a.cells){
#ifdef LCBININT_TRIPLE_MIXED_MOMENTS
    if(cell.critical&&!cell.moments_declined){if(!cell.moments)cell.moments=std::make_unique<MixedMomentTransport>(*cell.critical,cell.moving_lo,cell.moving_hi,cell.left,cell.right,cell.real_side);Mixed q;if(cell.moments->integrate(level,q,evaluations)){Density packet;packet.flux=q.value;packet.radial=q.r;packet.p=q.p;packet.rp=q.rp;packet.error=128*std::numeric_limits<double>::epsilon()*std::abs(q.value);add(packet,1);continue;}cell.moments_declined=true;}
#endif
    grow(cell.samples,n);
    if(cell.critical){const FoldRadialMap normalized{0,1,cell.left,cell.right};const auto span=cell.moving_hi-cell.moving_lo;
     for(int k=1;k<n;++k){auto& sample=cell.samples[k*(int(cell.samples.size())/n)];if(!sample.ready){const auto node=normalized(rule.x[k-1]);const auto u=cell.moving_lo+span*Mixed(node[0]);Mixed q;if(!cell.critical->critical_density(u,q,cell.real_side))return false;q=q*span*Mixed(node[1]);sample.flux=q.value;sample.radial=q.r;sample.p=q.p;sample.rp=q.rp;sample.error=128*std::numeric_limits<double>::epsilon()*std::abs(q.value);sample.ready=true;++evaluations;}add(sample,rule.w[k-1]);}continue;
    }
    const FoldRadialMap map{cell.lo,cell.hi,cell.left,cell.right};
    for(int k=1;k<n;++k){const auto t=map(rule.x[k-1]);const double theta=cell.center+cell.scale*std::sinh(t[0]),x=2*(theta-a.lo.theta)/width-1;const double shape=full_?1:(1-x)*(1+x);if(!(shape>0))return false;auto& sample=cell.samples[k*(int(cell.samples.size())/n)];if(!point(a,x,sample))return false;const double weight=rule.w[k-1]*cell.scale*std::cosh(t[0])*t[1]*2/width*std::sqrt(shape);add(sample,weight);}
   }
  out.ready=true;a.totals[level]=out;return true;
 }

public:
 Jet<double> spatial_witness(double theta,double gate,double width,std::size_t* promotions){
  Witness* retained=nullptr;for(std::size_t i=0;i<witness_count_;++i)if(witnesses_[i].theta==theta){retained=&witnesses_[i];break;}
  Jet<double> q;
  if(retained){q.phi=retained->phi;q.r=retained->r;q.t=retained->t;q.roundoff=retained->error;}else q=ray_.sample<false>(theta);
  if(q.roundoff>.125*gate*(width>0?std::abs(q.t)*width:std::abs(q.phi))){if(!precise_)precise_=std::make_unique<Ray<Q>>(R_,geometry_);const auto w=precise_->sample<false>(Q(theta));q.phi=double(w.phi);q.r=double(w.r);q.t=double(w.t);q.roundoff=double(w.roundoff);if(promotions)++*promotions;}
  if(!retained&&witness_count_<witnesses_.size())retained=&witnesses_[witness_count_++];
  if(retained)*retained={theta,q.phi,q.r,q.t,q.roundoff};return q;
 }
 int minimum_level(){if(relative_gate_>0)return 3;int level=3;for(auto& a:arcs_){if(!angular_chart(a))return 10;level=std::max(level,a.pair?a.pair->minimum_level:5);}return level;}
 void endpoint_geometry(AngularTopology& topology,std::array<std::array<double,2>,3>* radial=nullptr)const{for(std::size_t j=0;j<arcs_.size();++j){topology.arcs[j].lo=arcs_[j].lo.theta;topology.arcs[j].hi=arcs_[j].hi.theta;if(radial&&j<3)(*radial)[j]={arcs_[j].lo.radial,arcs_[j].hi.radial};}}
 std::size_t evaluations=0,promoted=0;
 Radius(double R,const Geometry& p,const std::array<double,8>& v,const std::array<double,8>& slope,std::shared_ptr<root_pair_detail::PreparedFrames> frames=nullptr):geometry_(p),R_(R),velocity_(v),slope_(slope),ray_(R,p),frames_(std::move(frames)){}
 bool initialize(const AngularTopology& topology) {
  LCBININT_COUPLED_SCOPE(endpoint);
  full_=topology.roots.empty();arcs_.resize(topology.arcs.size());
  for(std::size_t j=0;j<arcs_.size();++j) {
   auto& a=arcs_[j];a.lo.theta=a.seed_lo=topology.arcs[j].lo;a.hi.theta=a.seed_hi=topology.arcs[j].hi;
   if(full_)continue;
   const double width=a.hi.theta-a.lo.theta;
   const bool okay=endpoint(ray_,a.lo.theta,width,velocity_,slope_,a.lo)&&endpoint(ray_,a.hi.theta,width,velocity_,slope_,a.hi)&&a.lo.slope>0&&a.hi.slope<0;evaluations+=2;
   if(!okay) {
    if(!precise(a))return false;promoted+=2;
    const auto copy=[](const End<Q>& from,End<double>& to){to.theta=double(from.theta);to.radial=double(from.radial);to.slope=double(from.slope);for(int k=0;k<8;++k){to.parameter[k]=double(from.parameter[k]);to.mixed[k]=double(from.mixed[k]);}};
    copy((*a.wide)[0],a.lo);copy((*a.wide)[1],a.hi);
   }
  }
  // Continuation may only consume the retained root-count proof if its
  // physical arcs remain distinct, disjoint and positively oriented.
  if(!full_)for(std::size_t j=0;j<arcs_.size();++j){const auto& a=arcs_[j];if(!(a.hi.theta>a.lo.theta&&ray_.sample<false>((a.lo.theta+a.hi.theta)*.5).phi>0))return false;
   for(std::size_t k=0;k<j;++k)for(int turn:{-1,0,1}){const double shift=turn*2*std::acos(-1.);if(std::max(a.lo.theta,arcs_[k].lo.theta+shift)<std::min(a.hi.theta,arcs_[k].hi.theta+shift))return false;}}
  ready_=true;return true;
 }
 void set_accuracy(double relative,double radial_scale){relative_gate_=relative;radial_scale_=radial_scale;}
 bool evaluate(int level,bool uniform,Result& out) {
  LCBININT_COUPLED_SCOPE(angular);
  if(!ready_)return false;out={};
  for(auto& a:arcs_){const double width=a.hi.theta-a.lo.theta,wr=a.hi.radial-a.lo.radial;
   out.f0+=R_*width;out.r0+=width+R_*wr;
   for(int j=0;j<8;++j){const double wp=a.hi.parameter[j]-a.lo.parameter[j],wrp=a.hi.mixed[j]-a.lo.mixed[j];out.p0[j]+=velocity_[j]*width+R_*wp;out.rp0[j]+=slope_[j]*width+velocity_[j]*wr+wp+R_*wrp;}
   if(uniform)continue;if(!angular_chart(a))return false;
   Density sum;double error=INFINITY,radial_error=INFINITY;
   if(relative_gate_<=0){if(!limb_integral(a,level,sum))return false;}
   else{int minimum=a.pair?a.pair->minimum_level:5;
    if(!a.cells.empty()&&std::all_of(a.cells.begin(),a.cells.end(),[](const Cell& c){return bool(c.critical);}))minimum=4;
    const int maximum=std::max(level,minimum);
    Density previous;for(int lev=1;lev<=maximum;++lev){if(!limb_integral(a,lev,sum))return false;
     error=lev>1?std::abs(sum.flux-previous.flux):INFINITY;radial_error=lev>1?std::abs(sum.radial-previous.radial):INFINITY;
     if(lev>=minimum&&std::max(error,radial_scale_*radial_error)<=relative_gate_*std::abs(sum.flux))break;previous=sum;
    }
    out.angular_error+=error;out.radial_angular_error+=radial_error;
   }
   out.fh+=sum.flux;out.rh+=sum.radial;out.error+=sum.error;
   for(int j=0;j<8;++j){out.ph[j]+=sum.p[j];out.rph[j]+=sum.rp[j];}
  }return std::isfinite(out.f0)&&std::isfinite(out.fh);
 }

};
template<bool>struct Storage {};
template<>struct Storage<true>{std::unique_ptr<Radius> fused;std::shared_ptr<root_pair_detail::PreparedFrames> coefficient_frames;double spatial_scale=1,angular_relative=0;};
}

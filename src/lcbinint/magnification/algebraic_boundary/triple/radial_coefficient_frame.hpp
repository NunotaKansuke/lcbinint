#pragma once
#include "root_pair.hpp"
#include <memory>
#include <vector>
namespace lcbinint::algebraic_boundary::triple::root_pair_detail {
// Extended accumulation is confined to a chart's once-only coefficient
// construction. The pointwise transport and all eight RHS remain double.
struct Wide {
 long double value=0;std::array<long double,8> dp{};
 Wide()=default;Wide(long double v):value(v){}
};
inline Wide operator+(Wide a,const Wide& b){a.value+=b.value;for(int j=0;j<8;++j)a.dp[j]+=b.dp[j];return a;}
inline Wide operator-(Wide a){a.value=-a.value;for(auto& x:a.dp)x=-x;return a;}
inline Wide operator-(Wide a,const Wide& b){return a+(-b);}
inline Wide operator*(const Wide& a,const Wide& b){Wide r(a.value*b.value);for(int j=0;j<8;++j)r.dp[j]=a.dp[j]*b.value+a.value*b.dp[j];return r;}
struct WidePolynomial {
 std::array<Wide,9> c{};int degree=0;
 WidePolynomial()=default;WidePolynomial(double v){c[0]=Wide(v);}explicit WidePolynomial(Wide v){c[0]=v;}
};
inline WidePolynomial operator+(WidePolynomial a,const WidePolynomial& b){a.degree=std::max(a.degree,b.degree);for(int j=0;j<=b.degree;++j)a.c[j]=a.c[j]+b.c[j];return a;}
inline WidePolynomial operator-(WidePolynomial a){for(int j=0;j<=a.degree;++j)a.c[j]=-a.c[j];return a;}
inline WidePolynomial operator-(WidePolynomial a,const WidePolynomial& b){return a+(-b);}
inline WidePolynomial operator*(const WidePolynomial& a,const WidePolynomial& b){WidePolynomial r;r.degree=a.degree+b.degree;if(r.degree>8)throw std::logic_error("stream radial degree");for(int i=0;i<=a.degree;++i)for(int j=0;j<=b.degree;++j)r.c[i+j]=r.c[i+j]+a.c[i]*b.c[j];return r;}
// One frozen chart per continued arc. Geometry coefficients and their eight
// parameter RHS are constructed once, then evaluated radially with Horner.
struct PreparedFrame {
 using J=root_pair_detail::Jet<8>;using P=root_pair_detail::RadialPolynomial<8>;
 using C=coefficient_detail::C<P>;
 bool verified=false;double basis,origin;std::array<P,7> coefficients;J rho,lx1,ly1,lx2,ly2;
 PreparedFrame(const Geometry& p,double angle,double radius):basis(angle),origin(radius),rho(p.rho){
  J X(p.X),Y(p.Y),m1(p.m1),m2(p.m2),a(p.a),bx(p.bx),by(p.by);
  X.dp[0]=1;Y.dp[1]=1;rho.dp[2]=1;m1.dp[3]=1;m2.dp[4]=1;a.dp[5]=1;bx.dp[6]=1;by.dp[7]=1;
  const J co(std::cos(angle)),si(std::sin(angle));
  lx1=co*a;ly1=-si*a;lx2=co*bx+si*by;ly2=co*by-si*bx;
  const J sx=co*X+si*Y,sy=co*Y-si*X;
  P radial;radial.degree=1;radial.c[0]=J(radius);radial.c[1]=J(1);
  coefficients=root_pair_detail::spectrum_coefficients(radial,std::array<C,3>{C{},C{P(lx1),P(ly1)},C{P(lx2),P(ly2)}},std::array<P,3>{P(J(1)-m1-m2),P(m1),P(m2)},C{P(sx),P(sy)},P(rho));
 }
 void promote(const Geometry& p){
  const double angle=basis,radius=origin;
  using W=Wide;using WP=WidePolynomial;using WC=coefficient_detail::C<WP>;
  W wx(p.X),wy(p.Y),wrho(p.rho),wm1(p.m1),wm2(p.m2),wa(p.a),wbx(p.bx),wby(p.by);
  wx.dp[0]=1;wy.dp[1]=1;wrho.dp[2]=1;wm1.dp[3]=1;wm2.dp[4]=1;wa.dp[5]=1;wbx.dp[6]=1;wby.dp[7]=1;
  const W wc(cosl((long double)angle)),ws(sinl((long double)angle));
  WP radial;radial.degree=1;radial.c[0]=W(radius);radial.c[1]=W(1);
  const auto built=root_pair_detail::spectrum_coefficients(radial,std::array<WC,3>{WC{},WC{WP(wc*wa),WP(-ws*wa)},WC{WP(wc*wbx+ws*wby),WP(wc*wby-ws*wbx)}},std::array<WP,3>{WP(W(1)-wm1-wm2),WP(wm1),WP(wm2)},WC{WP(wc*wx+ws*wy),WP(wc*wy-ws*wx)},WP(wrho));
  for(int k=0;k<7;++k){coefficients[k].degree=built[k].degree;for(int i=0;i<=built[k].degree;++i){coefficients[k].c[i].value=double(built[k].c[i].value);for(int j=0;j<8;++j)coefficients[k].c[i].dp[j]=double(built[k].c[i].dp[j]);}}

 }
 template<class M> std::array<M,7> at(double R,const std::array<double,8>& velocity,const std::array<double,8>& slope)const{
  std::array<M,7> out;const double delta=R-origin;
  for(int k=0;k<7;++k){const auto& p=coefficients[k];J value=p.c[p.degree],radial;double second=0;
   for(int j=p.degree-1;j>=0;--j){second=second*delta+2*radial.value;radial=radial*J(delta)+value;value=value*J(delta)+p.c[j];}
   auto& m=out[k];m.value=value.value;m.r=radial.value;
   for(int j=0;j<8;++j){m.p[j]=value.dp[j]+radial.value*velocity[j];m.rp[j]=radial.dp[j]+second*velocity[j]+radial.value*slope[j];}
  }return out;
 }
 bool prepare(double R,const std::array<double,8>& motion,double lo,double hi,root_pair_detail::Arc<8>& out,double& lr,double& hr)const{
  const double l=lo-basis,h=hi-basis,pi=std::acos(-1.);
  if(!(l>-pi+.05&&h<pi-.05&&h>l))return false;
  std::array<double,7> pc{},pr{};
  for(int k=0;k<7;++k){const auto& p=coefficients[k];J value=p.c[p.degree];double radial=0;
   for(int j=p.degree-1;j>=0;--j){radial=radial*(R-origin)+value.value;value.value=value.value*(R-origin)+p.c[j].value;for(int i=0;i<8;++i)value.dp[i]=value.dp[i]*(R-origin)+p.c[j].dp[i];}
   for(int i=0;i<8;++i)value.dp[i]+=radial*motion[i];out.polynomial[k]=value;pc[k]=value.value;pr[k]=radial;
  }
  const double tl=std::tan(l*.5),th=std::tan(h*.5),initial=(tl+th)*.5,width=th-tl;
  double m=initial,v=width*width*.25;
  for(int n=0;n<12;++n){auto f=root_pair_detail::equations(pc,m,v);const double det=f.em*f.ov-f.ev*f.om;
   if(!std::isfinite(det)||std::abs(det)<1e-300)return false;
   const double dm=(f.ev*f.o-f.ov*f.e)/det,dv=(f.om*f.e-f.em*f.o)/det;
   m+=dm;v+=dv;if(!(v>0)||std::abs(m-initial)>width*.35||v>width*width)return false;
   if(std::abs(dm)<1e-13*(1+std::abs(m))&&std::abs(dv)<1e-13*(1+m*m))break;
   if(n==11)return false;
  }
  if(!out.pair.finish(out.polynomial,m,v))return false;
  out.center=basis;out.half_width=std::sqrt(v);out.R=J(R);out.R.dp=motion;out.rho=rho;out.inverse_rho=1/rho.value;
  out.lx1=lx1;out.ly1=ly1;out.lx2=lx2;out.ly2=ly2;
  const auto pole=[&](J lx,J ly){return std::array<J,3>{(out.R-lx)*(out.R-lx)+ly*ly,J(-4)*out.R*ly,(out.R+lx)*(out.R+lx)+ly*ly};};
  out.poles={pole(lx1,ly1),pole(lx2,ly2)};
  const auto f=root_pair_detail::equations(pc,m,v),r=root_pair_detail::equations(pr,m,v);const double det=f.em*f.ov-f.ev*f.om;
  const double mr=(f.ev*r.o-f.ov*r.e)/det,vr=(f.om*r.e-f.em*r.o)/det;
  const double dl=m-out.half_width,dh=m+out.half_width;
  lr=2*(mr-vr/(2*out.half_width))/(1+dl*dl);hr=2*(mr+vr/(2*out.half_width))/(1+dh*dh);return true;
 }
};
struct PreparedFrames {
 std::vector<std::shared_ptr<PreparedFrame>> frames;
 std::shared_ptr<PreparedFrame> get(const Geometry& p,double center,double radius){
  for(const auto& f:frames)if(std::abs(center-f->basis)<.3)return f;
  auto result=std::make_shared<PreparedFrame>(p,center,radius);frames.push_back(result);return result;
 }
};
}

#pragma once
// Isolated degree <=14 enclosure Sturm kernel. The target is exact dyadic
// PrimaryFrame input, not a previously rounded coefficient vector.
#include <array>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <limits>
#include <quadmath.h>
#include "d14_real.hpp"
#include "lens_frame.hpp"
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::holonomic::positive_detail {
template<class T> T up(T x);
template<class T> T down(T x);
template<> inline double up(double x) { return std::nextafter(x, INFINITY); }
template<> inline double down(double x) { return std::nextafter(x, -INFINITY); }
template<> inline __float128 up(__float128 x) { return nextafterq(x,HUGE_VALQ); }
template<> inline __float128 down(__float128 x) { return nextafterq(x,-HUGE_VALQ); }
inline bool environment_ok() {
#ifdef __FAST_MATH__
 return false;
#endif
#if defined(__SSE__)
 if (_mm_getcsr() & ((1u<<15)|(1u<<6))) return false;
#endif
 return std::fegetround()==FE_TONEAREST;
}
// Primitive bounds are widened after EACH correctly-rounded scalar operation.
// Exact zero/identity operations are preserved separately.
template<class T> struct Interval {
 T lo=0,hi=0;
 Interval()=default; Interval(double x):lo(x),hi(x) {}
 Interval(T l,T h):lo(l),hi(h) {}
 bool zero() const { return lo==0 && hi==0; }
 bool valid() const { return lo<=hi && lo>-HUGE_VALQ && hi<HUGE_VALQ; }
 int sign() const { return !valid()?2:lo>0?1:hi<0?-1:zero()?0:2; }
};
template<class T> Interval<T> operator-(Interval<T> a) { return {-a.hi,-a.lo}; }
template<class T> Interval<T> operator+(Interval<T> a,Interval<T> b) {
 if(a.zero())return b; if(b.zero())return a;
 return {down(a.lo+b.lo),up(a.hi+b.hi)};
}
template<class T> Interval<T> operator-(Interval<T> a,Interval<T> b) {return a+(-b);}
template<class T> Interval<T> operator*(Interval<T> a,Interval<T> b) {
 if(a.zero()||b.zero())return {};
 T v[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
 return {down(*std::min_element(v,v+4)),up(*std::max_element(v,v+4))};
}
template<class T> Interval<T> scale(Interval<T> a,int e) {
 if(a.zero())return a;
 if constexpr(std::is_same_v<T,double>) return {down(std::ldexp(a.lo,e)),up(std::ldexp(a.hi,e))};
 else return {down(scalbnq(a.lo,e)),up(scalbnq(a.hi,e))};
}
// Twofold center with a rigorously accumulated binary64 radius. Each center
// operation is assembled from EFT terms; no one-ULP bound for DD arithmetic.
struct Ball {
 D14Real c{}; double r=0;
 Ball()=default; Ball(double x):c(x) {}
 bool zero()const{return c.hi==0 && c.lo==0 && r==0;}
 bool valid()const{return c.finite() && std::isfinite(r) && r>=0;}
 double mag()const{return up(std::abs(c.hi)+std::abs(c.lo));}
 int sign()const { if(!valid())return 2; if(zero())return 0;
  double l=down(c.hi+c.lo),h=up(c.hi+c.lo);
  return l>r?1:h<-r?-1:2;
 }
 void term(double x) {
  auto s=d14_two_sum(c.hi,x); auto t=d14_two_sum(c.lo,s.lo);
  auto u=d14_two_sum(s.hi,t.hi); auto v=d14_two_sum(u.lo,t.lo);
  c=d14_two_sum(u.hi,v.hi);
  if(v.lo!=0)r=up(r+std::abs(v.lo));
 }
};
inline Ball operator-(Ball a){a.c=-a.c;return a;}
inline Ball operator+(Ball a,Ball b){if(a.zero())return b;if(b.zero())return a;
 a.r=up(a.r+b.r);a.term(b.c.hi);a.term(b.c.lo);return a;}
inline Ball operator-(Ball a,Ball b){return a+(-b);}
inline Ball operator*(Ball a,Ball b){if(a.zero()||b.zero())return {};
 Ball z; z.r=up(up(a.mag()*b.r)+up(b.mag()*a.r));z.r=up(z.r+up(a.r*b.r));
 for(double x:{a.c.hi,a.c.lo})for(double y:{b.c.hi,b.c.lo}) {
  if(x==0||y==0)continue; double p=x*y; double e=std::fma(x,y,-p);
  z.term(p);z.term(e);
  // FMA residual can underflow: bound absolute lost residual by min subnormal.
  z.r=up(z.r+std::numeric_limits<double>::denorm_min());
 }
 return z;
}
inline Ball scale(Ball a,int e){if(a.zero())return a;
 a.c.hi=std::ldexp(a.c.hi,e);a.c.lo=std::ldexp(a.c.lo,e);
 a.r=up(std::ldexp(a.r,e));
 // Scaling subnormals can round either limb.
 a.r=up(a.r+2*std::numeric_limits<double>::denorm_min()); return a;
}
template<class I> __float128 magnitude(I a) {
 if constexpr(std::is_same_v<I,Ball>)return (__float128)a.mag()+a.r;
 else return fmaxq(fabsq(a.lo),fabsq(a.hi));
}
template<class I> struct Poly { std::array<I,15> c{}; int degree=0; };
template<class I> Poly<I> add(Poly<I> a,Poly<I> b) {
 a.degree=std::max(a.degree,b.degree);for(int i=0;i<=b.degree;++i)a.c[i]=a.c[i]+b.c[i];return a;
}
template<class I> Poly<I> neg(Poly<I> a){for(auto& x:a.c)x=-x;return a;}
template<class I> Poly<I> mul(Poly<I> a,Poly<I> b) {
 Poly<I> r;r.degree=a.degree+b.degree;
 if(r.degree>14){r.degree=14;r.c[0]=I(INFINITY);return r;}
 for(int i=0;i<=a.degree;++i)for(int j=0;j<=b.degree;++j)r.c[i+j]=r.c[i+j]+a.c[i]*b.c[j];return r;
}
template<class I> Poly<I> times(Poly<I> a,double n){for(auto&x:a.c)x=x*I(n);return a;}
template<class I> Poly<I> shift(Poly<I> a,int n){Poly<I> r;r.degree=a.degree+n;for(int i=0;i<=a.degree;++i)r.c[i+n]=a.c[i];return r;}
template<class I> Poly<I> constant(I x){Poly<I> p;p.c[0]=x;return p;}
template<class I> Poly<I> polynomial(const PrimaryFrame& pf) {
 I a(pf.a),m(pf.m0),x(pf.X),y(pf.Y),rho(pf.rho),one(1),two(2);
 I a2=a*a, beta=x*x+y*y-rho*rho;
 Poly<I> C,L,U,E,Z;
 C.degree=3;C.c[0]=a2*m*m;
 C.c[1]=one-two*a2*m+a2*beta+two*a*x*(I(3)*m-one);
 C.c[2]=I(-2)+a2+beta-I(4)*a*x; C.c[3]=one;
 L.degree=2;L.c[0]=-a2*m;L.c[1]=a2-one;L.c[2]=one;
 U.degree=2;U.c[0]=a*m-a2*m*x;
 U.c[1]=-a*(m+one)+x*(a2-one)+a*beta;U.c[2]=a+x;
 E.degree=1;E.c[0]=-m;E.c[1]=one;
 auto G=add(add(mul(U,U),mul(constant(y*y),mul(L,L))),
   add(mul(constant(I(-4)*a*x),mul(E,C)),
       mul(constant(I(-4)*a2*(I(4)*x*x+y*y)),shift(mul(E,E),1))));
 Poly<I> b;b.degree=2;b.c[0]=a2*m*m;b.c[1]=two*a*m*(x-a);
 b.c[2]=(x-a)*(x-a)+y*y-rho*rho;
 Z=mul(constant(a2*(one-m)*y*y),mul(E,b));
 auto CC=mul(C,C), VG=shift(G,1);
 return add(add(mul(add(CC,times(VG,-4)),mul(G,G)),
    times(mul(mul(C,add(times(CC,2),times(VG,-9))),Z),8)),
    times(shift(mul(Z,Z),2),-432));
}
template<class I> bool trim(Poly<I>& p){while(p.degree>0&&p.c[p.degree].zero())--p.degree;return p.c[p.degree].sign()!=2;}
template<class I> bool normalize(Poly<I>& p){
 __float128 m=0;for(int i=0;i<=p.degree;++i){if(!p.c[i].valid())return false;m=fmaxq(m,magnitude(p.c[i]));}
 if(m==0)return true;int e;frexpq(m,&e);for(int i=0;i<=p.degree;++i)p.c[i]=scale(p.c[i],-e);return trim(p);
}
template<class I> Poly<I> derivative(Poly<I> p){Poly<I> d;d.degree=std::max(0,p.degree-1);for(int i=1;i<=p.degree;++i)d.c[i-1]=p.c[i]*I(double(i));return d;}
template<class I> I eval(const Poly<I>& p,I x){I r=p.c[p.degree];for(int i=p.degree-1;i>=0;--i)r=r*x+p.c[i];return r;}
template<class I> I point(__float128 x){
 if constexpr(std::is_same_v<I,Ball>){Ball b;b.c=d14_from_qf(x);
  __float128 diff=fabsq(x-d14_to_qf(b.c));b.r=diff==0?0:up(double(diff));return b;}
 else {using T=decltype(I{}.lo);T t=T(x); if((__float128)t==x)return {t,t};return {down(t),up(t)};}
}
template<class I> struct Chain {
 std::array<Poly<I>,15> s{};int size=0;bool ok=false;
 bool build(Poly<I> p,int scale_exponent=0){
  size=0;ok=false;
  for(int i=0;i<=p.degree;++i)p.c[i]=scale(p.c[i],scale_exponent*i);
  if(!trim(p)||!normalize(p)||p.c[p.degree].zero())return false;
  s[size++]=p;s[size++]=derivative(p);if(!normalize(s[1]))return false;
  while(s[size-1].degree>0){
   auto r=s[size-2];const auto& b=s[size-1];int sig=b.c[b.degree].sign();if(sig!=1&&sig!=-1)return false;
   I pivot=sig>0?b.c[b.degree]:-b.c[b.degree];
   while(r.degree>=b.degree){int d=r.degree,k=d-b.degree;I lead=r.c[d];
    for(int i=0;i<d;++i)r.c[i]=pivot*r.c[i]-(i>=k?(sig>0?lead:-lead)*b.c[i-k]:I(0));
    r.c[d]=I(0);--r.degree;
    if(!trim(r)||!normalize(r))return false;
   }
   r=neg(r);if(r.c[r.degree].zero())break;
   if(size>=15)return false;s[size++]=r;
  }
  ok=true;return true;
 }
 // One-sided signs only after exact zero; an uncertain sign never becomes zero.
 int variation(__float128 x,int side=1)const {
  if(!ok)return -1;int prev=0,v=0;
  for(int i=0;i<size;++i){auto p=s[i];int k=0,sg;
   for(;;){sg=eval(p,point<I>(x)).sign();if(sg==2)return -1;
    if(sg||p.degree==0)break;p=derivative(p);++k;}
   if(side<0&&(k&1))sg=-sg;if(sg){if(prev&&prev!=sg)++v;prev=sg;}
  }return v;
 }
};
} // namespace
#if defined(__GNUC__)
#pragma GCC pop_options
#endif

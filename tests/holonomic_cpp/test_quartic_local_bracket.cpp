#include "lcbinint/magnification/holonomic/quartic_local_bracket.hpp"
#include <quadmath.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif
using namespace lcbinint::holonomic::local_bracket_detail;
void checked(bool b,int line){if(!b){std::fprintf(stderr,"local bracket check failed at line %d\n",line);std::exit(1);}}
#define check(b) checked((b),__LINE__)
int main(){
 std::mt19937_64 gen(7116);int accepted=0;
 for(int i=0;i<20000;++i){
  double a=std::ldexp(double(int64_t(gen()%2000001)-1000000),int(gen()%201)-100);
  double b=std::ldexp(double(int64_t(gen()%2000001)-1000000),int(gen()%201)-100);
  auto s=add(point(a),point(b)),p=mul(point(a),point(b));
  __float128 qs=(__float128)a+b,qp=(__float128)a*b;
  if(s.ok){check((__float128)s.lo<=qs&&qs<=(__float128)s.hi);++accepted;}
  if(p.ok){check((__float128)p.lo<=qp&&qp<=(__float128)p.hi);++accepted;}
 }
 check(accepted>39000);
 // Independent qf evaluations inside random polynomial intervals.
 int polynomial_accepts=0;
 for(int i=0;i<1000;++i){
  std::array<double,5> c{};
  for(auto& v:c)v=double(int(gen()%201)-100)/32;
  double lo=double(int(gen()%201)-100)/32,hi=lo+0.03125;
  auto pv=value(c,{lo,hi,true}),dv=derivative(c,{lo,hi,true});
  // Zero coefficients get outward min-normal widths; subsequent products
  // can conservatively reject underflow. Only accepted enclosures assert
  // inclusion, and the acceptance count below prevents a vacuous test.
  if(!pv.ok||!dv.ok)continue;
  ++polynomial_accepts;
  for(double t:{lo,0.5*(lo+hi),hi}){
   __float128 v=c[4],d=4*(__float128)c[4];
   for(int k=3;k>=0;--k)v=v*t+c[k];
   for(int k=3;k>=1;--k)d=d*t+k*(__float128)c[k];
   check((__float128)pv.lo<=v&&v<=(__float128)pv.hi);
   check((__float128)dv.lo<=d&&d<=(__float128)dv.hi);
  }
 }

 check(polynomial_accepts>900);
 std::printf("polynomial enclosure checks: %d accepted / 1000\n",polynomial_accepts);
#if defined(__SSE2__)
 const unsigned csr=_mm_getcsr();_mm_setcsr(csr|0x8040);
 check(!point(std::numeric_limits<double>::denorm_min()).ok);
 check(!point(-std::numeric_limits<double>::denorm_min()).ok);
 check(regular(0.0)&&regular(-0.0));
 _mm_setcsr(csr);
#endif
 check(!mul(point(1e-250),point(1e-250)).ok);
 check(!mul(point(1e250),point(1e250)).ok);
 std::array<double,5> p{4,0,-5,0,1};
 check(certify(p,{-2,-1,1,2},4));
 check(!certify(p,{-2,-2,1,2},4));
 check(!certify({0,-2,-1,2,1},{-2,-1,0,1},4));
 check(!certify(p,{-2,-1,1,3},4));
 check(!certify({1,0,-2,0,1},{-1,1,0,0},2));
 check(!certify({1,0,0,0,1},{-1,1,0,0},2));
 std::fesetround(FE_UPWARD);check(!certify(p,{-2,-1,1,2},4));std::fesetround(FE_TONEAREST);
 std::printf("local bracket PASS; interval comparisons %d\n",accepted);
}

#include "lcbinint/magnification/algebraic_boundary/triple/support_resultant.hpp"
#include "resultant_reference.hpp"
#include <iostream>
#include <random>
using namespace lcbinint::algebraic_boundary::triple;
void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
int main(){
 // Bounds generated from the exact Gaussian-rational discriminant of Q,
 // then converted to Bernstein form using rational arithmetic. Input
 // rationals are the exact binary64 values, not decimal approximations.
 for(const auto&fixture:resultant_references){
  const auto&p=fixture.p;Geometry g{p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]};
  auto check=[&]() {
   const auto b=resultant_detail::bernstein(g,fixture.lo,fixture.hi);
   for(int j=0;j<29;++j)require(b[j].valid()&&b[j].lo<=fixture.lower[j]&&b[j].hi>=fixture.upper[j],
     "cold/warm support coefficient misses exact rational enclosure");
  };
  // Exercise both one-shot formation and the reused lens-only matrix basis.
  for(int repeat=0;repeat<6;++repeat)check();
  // Every lens coordinate/mass belongs to cache identity. Populate a
  // different geometry, then require the exact original coefficients again.
  for(int key=0;key<5;++key){
   Geometry changed=g;
   if(key==0)changed.m1*=.9;
   if(key==1)changed.m2*=.9;
   if(key==2)changed.a*=1.1;
   if(key==3)changed.bx+=.03;
   if(key==4)changed.by+=.04;
   for(int repeat=0;repeat<4;++repeat)(void)resultant_detail::bernstein(changed,fixture.lo,fixture.hi);
   for(int repeat=0;repeat<4;++repeat)check();
  }
 }
 using Ball=coefficient_ball_detail::Ball<long double>;
 std::mt19937_64 random(372906);int corners=0;
 for(int trial=0;trial<4000;++trial){
  auto draw=[&](){return std::ldexp(double(int(random()%2001)-1000)/1000,int(random()%1201)-600);};
  double a=draw(),b=draw();if(trial%31==0)a=-0.;
  const double ar=std::abs(a)*std::ldexp(1.,-30),br=std::abs(b)*std::ldexp(1.,-30);
  const Ball x((long double)a,(long double)ar),y((long double)b,(long double)br);
  const auto sum=(x+y).interval(),product=(x*y).interval();
  for(int sa:{-1,1})for(int sb:{-1,1}){
   const __float128 xx=(__float128)a+sa*(__float128)ar,yy=(__float128)b+sb*(__float128)br;
   require((__float128)sum.lo<=xx+yy&&xx+yy<=(__float128)sum.hi,"ball sum enclosure");
   require((__float128)product.lo<=xx*yy&&xx*yy<=(__float128)product.hi,"ball product enclosure");++corners;
  }
 }
 std::cout<<"PASS: cold/warm and lens-invalidation exact resultant coefficients and "<<corners<<" promoted arithmetic corners\n";
}

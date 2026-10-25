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
  const auto b=resultant_detail::bernstein(g,fixture.lo,fixture.hi);
  for(int j=0;j<29;++j)require(b[j].valid()&&b[j].lo<=fixture.lower[j]&&b[j].hi>=fixture.upper[j],
    "support coefficient misses exact rational enclosure");
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
 std::cout<<"PASS: 87 exact resultant coefficients and "<<corners<<" promoted arithmetic corners\n";
}

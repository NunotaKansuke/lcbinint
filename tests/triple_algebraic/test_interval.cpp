#include "lcbinint/magnification/algebraic_boundary/triple/interval.hpp"
#include <random>
#include <iostream>
#include <stdexcept>
using namespace lcbinint::algebraic_boundary;
int main(){
 using namespace triple::interval_detail;
 auto same=[](double a,double b){std::uint64_t x,y;std::memcpy(&x,&a,8);std::memcpy(&y,&b,8);return x==y;};
 std::mt19937_64 random(20261024);
 for(int i=0;i<1000000;++i){
  std::uint64_t bits=random();double x;std::memcpy(&x,&bits,8);
  if(std::isnan(x))continue;
  if(!same(up(x),std::nextafter(x,INFINITY)) || !same(down(x),std::nextafter(x,-INFINITY)))
   throw std::runtime_error("outward neighbour differs");
 }
 for(double x:{0.,-0.,double(INFINITY),-double(INFINITY),std::numeric_limits<double>::max(),std::numeric_limits<double>::denorm_min()})
  if(!same(up(x),std::nextafter(x,INFINITY)) || !same(down(x),std::nextafter(x,-INFINITY)))
   throw std::runtime_error("boundary neighbour differs");
 using Q=__float128;using U=unsigned __int128;
 auto sameq=[](Q a,Q b){return std::memcmp(&a,&b,16)==0;};
 for(int i=0;i<100000;++i){
  U bits=U(random())<<64|random();Q x;std::memcpy(&x,&bits,16);
  if(isnanq(x))continue;
  if(!sameq(up(x),nextafterq(x,HUGE_VALQ)) || !sameq(down(x),nextafterq(x,-HUGE_VALQ)))
   throw std::runtime_error("quad outward neighbour differs");
 }
 for(Q x:std::array<Q,6>{Q(0),-Q(0),HUGE_VALQ,-HUGE_VALQ,nextafterq(HUGE_VALQ,0),nextafterq(Q(0),Q(1))})
  if(!sameq(up(x),nextafterq(x,HUGE_VALQ)) || !sameq(down(x),nextafterq(x,-HUGE_VALQ)))
   throw std::runtime_error("quad boundary neighbour differs");
 std::uniform_real_distribution<double> dist(-1e10,1e10);
 for(int i=0;i<100000;++i){
  double a=dist(random),b=dist(random),c=dist(random),d=dist(random);if(a>b)std::swap(a,b);if(c>d)std::swap(c,d);
  auto x=Interval<double>(a,b)*Interval<double>(c,d);
  auto y=positive_detail::Interval<double>(a,b)*positive_detail::Interval<double>(c,d);
  if(x.lo!=y.lo || x.hi!=y.hi)throw std::runtime_error("product enclosure differs");
 }
 std::cout<<"IEEE neighbours and signed interval products match incumbent bounds\n";
}

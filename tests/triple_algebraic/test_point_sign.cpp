#include "lcbinint/magnification/algebraic_boundary/triple/point_polynomial.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/angular_topology.hpp"
#include <random>
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
using IQ=interval_detail::Interval<__float128>;
void check(bool b,const char*s){if(!b)throw std::runtime_error(s);}
template<class T> int exercise() {
 std::mt19937_64 rng(723978);std::uniform_real_distribution<double> u(-1,1);int checks=0;
 for(int i=0;i<240;++i) {
  Geometry p{u(rng),u(rng),.001+.5*std::abs(u(rng)),.2,.08,1.,-.4,.6};
  point_polynomial_detail::Cache<T> cache(p);
  for(int j=0;j<12;++j) {
   const double R=.2+2*std::abs(u(rng)),t=2*u(rng);const bool chart=j%2;
   const auto polynomial=cache.at(R);const auto exact=boundary_as<IQ>(R,p).coefficients;
   for(int k=0;k<7;++k) {
    auto enclosure=polynomial.c[k].interval();
    check(__float128(enclosure.lo)<=exact[k].lo && exact[k].hi<=__float128(enclosure.hi),"coefficient enclosure");++checks;
   }
   const int sign=polynomial.sign(t,chart);
   const auto reference=physical_detail::Ray<IQ>(IQ(R),p).value(IQ(t),chart);
   if(sign && reference){check(reference->sign()==sign,"independent physical sign");++checks;}
  }
 }
 Geometry contact{.1,.15,.15,.2,.08,1.,-.4,.6};point_polynomial_detail::Cache<T> cache(contact);
 for(double R:{.82,.89,.93,1.04,1.115,1.19}) {
  const auto topology=angular_topology(R,contact);check(topology.completed,"contact topology");const auto polynomial=cache.at(R);
  for(auto root:topology.roots) {
   double t=std::tan((root.lo+(root.hi-root.lo)*.5)*.5);bool chart=std::abs(t)>1;if(chart)t=-1/t;
   for(double width:{1e-6,1e-10,1e-12,1e-14})for(int side:{-1,1}) {
    const double point=t+side*width;const int sign=polynomial.sign(point,chart);
    const auto reference=physical_detail::Ray<IQ>(IQ(R),contact).value(IQ(point),chart);
    if(sign && reference){check(reference->sign()==sign,"near-root sign");++checks;}
   }
  }
 }
 auto invalid=cache.at(.93);invalid.c[0].error=INFINITY;check(invalid.sign(.2,false)==0,"ambiguous coefficients fail closed");
 return checks;
}
int main(){auto a=exercise<double>(),b=exercise<long double>();std::cout<<"PASS point signs and coefficient enclosures: "<<a+b<<"\n";}

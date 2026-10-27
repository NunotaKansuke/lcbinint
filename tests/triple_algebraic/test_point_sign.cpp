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
 // Independent promoted interval evaluation of uncertain polynomials,
 // including cancellation, both coordinate signs and every radial degree.
 using B=coefficient_ball_detail::Ball<T>;
 for(int trial=0;trial<2000;++trial) {
  const int degree=int(rng()%9);std::array<B,9> coefficients{};
  for(int j=0;j<=degree;++j) {
   const T centre=std::ldexp(T(int(rng()%2001)-1000)/T(1024),int(rng()%601)-300);
   coefficients[j]=B(centre,std::abs(centre)*std::ldexp(T(1),-40));
  }
  const double x=std::ldexp(double(int(rng()%2001)-1000)/1024,int(rng()%7)-3);
  const auto result=point_polynomial_detail::horner_ball<T>(degree,T(x),[&](int j){return coefficients[j];});
  auto promoted=[&](int j){const auto&a=coefficients[j];
   return IQ((__float128)a.value-(__float128)a.error,(__float128)a.value+(__float128)a.error);
  };
  IQ reference=promoted(degree);
  for(int j=degree-1;j>=0;--j)reference=reference*IQ(x)+promoted(j);
  check((__float128)result.value-(__float128)result.error<=reference.lo &&
      reference.hi<=(__float128)result.value+(__float128)result.error,"Horner input and rounding enclosure");
  ++checks;
 }
 // Underflow padding and unresolved overflow must retain fail-closed behavior.
 {
  const T tiny=std::numeric_limits<T>::min();
  std::array<B,9> coefficients;
  for(int j=0;j<9;++j)coefficients[j]=B(j%2?-tiny:tiny,T(0));
  const auto result=point_polynomial_detail::horner_ball<T>(8,T(.25),[&](int j){return coefficients[j];});
  const __float128 leading=coefficients[8].value;
  IQ reference(leading,leading);
  for(int j=7;j>=0;--j){const __float128 value=coefficients[j].value;
   reference=reference*IQ(.25)+IQ(value,value);
  }
  check((__float128)result.value-result.error<=reference.lo &&
      reference.hi<=(__float128)result.value+result.error,"Horner underflow enclosure");
  auto overflow=cache.at(.93);
  for(auto& a:overflow.c)a=B(std::numeric_limits<T>::max(),T(0));
  check(overflow.sign(4.,false)==0,"Horner overflow fails closed");
 }
 return checks;
}
int main(){auto a=exercise<double>(),b=exercise<long double>();std::cout<<"PASS point signs and coefficient enclosures: "<<a+b<<"\n";}

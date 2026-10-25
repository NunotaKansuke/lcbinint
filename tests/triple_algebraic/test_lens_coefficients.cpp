#include "lcbinint/magnification/algebraic_boundary/triple/point_polynomial.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/radial_polynomial.hpp"
#include <thread>
#include <random>
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
using IQ=interval_detail::Interval<__float128>;
void check(bool b,const char*s){if(!b)throw std::runtime_error(s);}
template<class T> int exercise(){
 std::mt19937_64 rng(7486734);std::uniform_real_distribution<double> u(-1,1);int checks=0;
 for(int i=0;i<80;++i){Geometry p{u(rng),u(rng),.001+.2*std::abs(u(rng)),.01+.3*std::abs(u(rng)),.001+.2*std::abs(u(rng)),.2+1.8*std::abs(u(rng)),u(rng),u(rng)};
  point_polynomial_detail::LensCache<T> lens(p);
  for(int j=0;j<8;++j){p.X=u(rng);p.Y=u(rng);p.rho=.001+.4*std::abs(u(rng));auto cache=lens.query(p);double R=.2+1.8*std::abs(u(rng));auto q=cache.at(R);auto reference=boundary_from_radius(IQ(R),p).coefficients;
   for(int k=0;k<7;++k){auto b=q.c[k].interval();check(__float128(b.lo)<=reference[k].lo && reference[k].hi<=__float128(b.hi),"source-basis coefficient enclosure");++checks;}
  }
  auto bad=p;bad.m1=std::nextafter(p.m1,1.);bool rejected=false;try{lens.query(bad);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"changed lens identity rejected");
 }
 return checks;
}
int main(){int n=exercise<double>()+exercise<long double>();Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};point_polynomial_detail::LensContext context(p);check(!context.reuse_ready(),"new context cold");for(int i=0;i<3;++i)context.note_query();check(!context.reuse_ready(),"short sequence cold");context.note_query();check(context.reuse_ready(),"reuse sequence");
 bool ok[2]={false,false};auto worker=[&](int i){try{auto q=p;q.X+=i*.02;auto cache=context.table<long double>(q);auto actual=cache.at(.93);auto ref=boundary_from_radius(IQ(.93),q).coefficients;ok[i]=true;for(int k=0;k<7;++k){auto b=actual.c[k].interval();ok[i]=ok[i] && __float128(b.lo)<=ref[k].lo && ref[k].hi<=__float128(b.hi);}}catch(...){ok[i]=false;}};
 std::thread a(worker,0),b(worker,1);a.join();b.join();check(ok[0]&&ok[1],"shared lazy basis concurrent queries");
 std::cout<<"PASS source coefficient enclosures "<<n<<" and identity/lazy/concurrent use\n";
}

#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/radius_terms.hpp"
#include <quadmath.h>
#include <random>
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
void check(bool b,const char*s){if(!b)throw std::runtime_error(s);}
int main(){std::mt19937_64 rng(9237624);std::uniform_real_distribution<double> u(-1,1);int n=0;
 auto test=[&](Complex z){const double value=regular_support_detail::fast_abs(z),original=std::abs(z);
  const double norm=std::norm(z),upper=radius_value_detail::bound_magnitude(z,norm);
  if(std::isnan(original)){check(std::isnan(value)&&std::isnan(upper),"nan magnitude");return;}
  if(std::isinf(original)){check(std::isinf(value)&&std::isinf(upper),"infinite magnitude");return;}
  const __float128 x=z.real(),y=z.imag(),reference=sqrtq(x*x+y*y);
  const __float128 bound=__float128(4*std::numeric_limits<double>::epsilon())*reference+std::numeric_limits<double>::denorm_min();
  check(fabsq(__float128(value)-reference)<=bound,"magnitude roundoff exceeds planner margin");
  if(norm>=32*std::numeric_limits<double>::min() && std::isfinite(norm)){
   check(__float128(upper)>=reference,"value magnitude must round upward conservatively");
   check(__float128(upper)-reference<=2*bound,"value magnitude padding exceeds roundoff margin");
  }else check(fabsq(__float128(upper)-reference)<=bound,"scaled value magnitude fallback");
  ++n;
 };
 for(int i=0;i<24000;++i){int e=int(std::floor(-1070+2090*std::abs(u(rng))));test({std::ldexp(u(rng),e),std::ldexp(u(rng),e)});}
 for(double a:{0.,std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::min(),std::sqrt(32*std::numeric_limits<double>::min()),1.,1e150,1e300,std::numeric_limits<double>::max()})for(double b:{0.,a,-a})test({a,b});
 test({INFINITY,NAN});test({NAN,1.});check(regular_support_detail::fast_abs(-7)==7,"scalar abs unchanged");
 std::cout<<"PASS magnitude independent quad comparisons "<<n<<" including subnormal/overflow fallback\n";
}

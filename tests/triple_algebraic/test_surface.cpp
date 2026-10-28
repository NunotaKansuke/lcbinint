#include "lcbinint/magnification/algebraic_boundary/triple/radial_events.hpp"
#include <iostream>
#include <random>
#include <cstring>
using namespace lcbinint::algebraic_boundary::triple;
int main(){
 for(int n=0;n<=16;++n)for(int k=0;k<=n;++k) {
  int reference=1;for(int j=1;j<=k;++j)reference=reference*(n-j+1)/j;
  if(angular_detail::choose(n,k)!=reference)throw std::runtime_error("cached exact binomial coefficient");
 }
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};
 for(bool reciprocal:{false,true}) {
  auto tensor=surface_detail::make(.25,2.,reciprocal,p);
  auto sub=surface_detail::restrict(tensor,.25,2.,-1,1,.5,1.5,-.5,.5);
  for(int i=0;i<=8;++i)for(int j=0;j<=8;++j){
   double x=i/8.,y=j/8.,R=.5+x,t=-.5+y;
   auto enclosure=surface_detail::evaluate(sub,surface_detail::I(x),surface_detail::I(y));
   auto direct=event_detail::evaluate((__float128)R,(__float128)t,p,reciprocal).v;
   if(direct<(__float128)enclosure.lo || direct>(__float128)enclosure.hi)
    throw std::runtime_error("Bernstein surface does not enclose point polynomial");
  }
 }
 // Check dyadic split coefficients directly in quad, including cancellation
 // and subnormal arithmetic. This tests the bound, not just sampled curves.
 std::mt19937_64 random(18713);
 for(int trial=0;trial<2000;++trial) {
  surface_detail::Tensor parent;parent.nr=8;parent.nt=6;
  const int exponent=trial%7==0?-1074:trial%11==0?1023:int(random()%1800)-900;
  for(auto& row:parent.b)for(auto& entry:row) {
   const double value=std::ldexp(double(int(random()%2049)-1024)/1024,exponent);
   entry={std::nextafter(value,-INFINITY),std::nextafter(value,INFINITY)};
  }
  // Independent quad extrema check the early sign test, including
  // cancellation and subnormal differences that round to zero in double.
  for(bool difference:{false,true}) {
   __float128 lo=HUGE_VALQ,hi=-HUGE_VALQ;
   for(int i=0;i<=parent.nr;++i)for(int j=0;j<=parent.nt-int(difference);++j) {
    const auto a=parent.b[i][j];
    const auto b=parent.b[i][j+int(difference)];
    const __float128 lower=difference?(__float128)b.lo-a.hi:(__float128)a.lo;
    const __float128 upper=difference?(__float128)b.hi-a.lo:(__float128)a.hi;
    lo=std::min(lo,lower);hi=std::max(hi,upper);
   }
   if(surface_detail::excludes_zero(parent,difference)!=(lo>0 || hi<0))
    throw std::runtime_error("early surface exclusion disagrees with quad coefficient signs");
  }
  for(int axis:{0,1}) {
   auto h=surface_detail::difference_hull(parent,axis);
   for(int i=0;i<=parent.nr-(axis==0);++i)for(int j=0;j<=parent.nt-(axis==1);++j) {
    auto a=axis?parent.b[i][j+1]:parent.b[i+1][j],b=parent.b[i][j];
    if((__float128)a.lo-b.hi<(__float128)h.lo || (__float128)a.hi-b.lo>(__float128)h.hi)
     throw std::runtime_error("difference hull lost an endpoint enclosure");
   }
  }
  for(double fraction:{.5,.4375,.3})for(int axis:{0,1}) {
   const surface_detail::I fraction_box=fraction==.3?
       surface_detail::I(std::nextafter(fraction,0.),std::nextafter(fraction,1.)):surface_detail::I(fraction);
   auto children=surface_detail::split(parent,axis,fraction_box);
   const int degree=axis?parent.nt:parent.nr,other=axis?parent.nr:parent.nt;
   for(double exact_fraction:{fraction_box.lo,fraction_box.hi})for(int side:{0,1})for(int line=0;line<=other;++line)for(int endpoint:{0,1}) {
    std::array<__float128,9> values{};
    for(int j=0;j<=degree;++j) {
     const auto a=axis?parent.b[line][j]:parent.b[j][line];
     values[j]=endpoint?a.hi:a.lo;
    }
    for(int level=0;level<=degree;++level) {
     const int index=side?degree-level:level;
     const auto& child=side?children.second:children.first;
     const auto bound=axis?child.b[line][index]:child.b[index][line];
     const auto exact=values[side?degree-level:0];
     if(exact<(__float128)bound.lo || exact>(__float128)bound.hi)
      throw std::runtime_error("dyadic split lost an endpoint enclosure");
     for(int j=0;j<degree-level;++j)values[j]=(1-(__float128)exact_fraction)*values[j]+(__float128)exact_fraction*values[j+1];
    }
   }
  }
 }
 // Sign-consistent, monotone, mixed, and invalid tensors exercise both
 // successful exclusion branches; random tensors mostly exercise refusal.
 for(double scale:{1.,std::numeric_limits<double>::denorm_min()})for(int sign:{-1,1}) {
  surface_detail::Tensor t;
  for(int i=0;i<=t.nr;++i)for(int j=0;j<=t.nt;++j)t.b[i][j]=surface_detail::I(sign*(j+1)*scale);
  if(!surface_detail::excludes_zero(t) || !surface_detail::excludes_zero(t,true))
   throw std::runtime_error("strict coefficient signs should exclude a zero");
  t.b[0][0]=surface_detail::I(-sign*scale);
  if(surface_detail::excludes_zero(t))throw std::runtime_error("mixed signs must remain unresolved");
  t.b[0][0]={NAN,NAN};
  if(surface_detail::excludes_zero(t) || surface_detail::excludes_zero(t,true))
   throw std::runtime_error("invalid coefficient must fail closed");
 }
 std::cout<<"PASS: bivariate restriction and 2000 midpoint, 7/16 and general split plus difference hull enclosure trials\n";
}

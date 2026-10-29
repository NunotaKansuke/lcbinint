#include "lcbinint/magnification/algebraic_boundary/triple/exterior_caustic_bound.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include <cfenv>
#include <iostream>
using namespace lcbinint;
namespace tr=algebraic_boundary::triple;
namespace rs=tr::regular_support_detail;
void check(bool condition,const char* text){if(!condition)throw std::runtime_error(text);}
int main(){
 int critical_checks=0,counts=0;
 for(auto p:std::array<tr::Geometry,3>{{
     {.1,.2,.01,.2,.08,1.,-.4,.6},
     {.1,.2,.01,.25,.1,2.4,-.3,.8},
     {.1,.2,.01,.1,.2,3.,-2.,1.3}}}) {
  const auto lenses=p.lenses();const auto masses=p.masses();
  const tr::ExteriorCausticBound bound(lenses,masses);
  check(bound.valid(),"positive separated triple has an exterior bound");
  for(int k=0;k<12;++k) {
   std::vector<tr::Complex> roots;
   const auto critical=rs::cover(p,.173+2*std::acos(-1.)*k/12,0.,roots);
   check(critical.size()==6,"independent critical-root cover");
   for(auto disk:critical) {
    double nearest=INFINITY;
    for(auto lens:lenses)nearest=std::min(nearest,std::abs(disk.w-lens));
    check(nearest+disk.h<bound.radius(),"mapped critical enclosure stays in caustic bound");
    ++critical_checks;
   }
  }
  rs::Tree independent(p);
  for(auto source:std::array<tr::Complex,4>{{{8,6},{-8,6},{-8,-6},{8,-6}}}) {
   const auto exterior=bound.clearance(source);
   check(exterior.first>.1 && exterior.second==4,"escaping ray proves exterior image count");
   check(independent.count(source,.1)==4,"independent critical-curve winding agrees");
   ++counts;
  }
  check(bound.clearance(lenses[0]).second<0,"no false certificate inside bounding disk");
 }
 const double total=2.0000001;
 std::array<tr::Complex,3> wide{{{50,0},{-50,0},{-50.1,0}}};
 std::array<double,3> mass{{1/total,1/total,1e-7/total}};
 std::array<int,3> order{{0,1,2}};
 do {
  const tr::ExteriorCausticBound bound({wide[order[0]],wide[order[1]],wide[order[2]]},
      {mass[order[0]],mass[order[1]],mass[order[2]]});
  check(bound.valid() && bound.radius()>5 && bound.radius()<25,"clustered wide-lens caustic radius");
  const auto exterior=bound.clearance({0,.001});
  check(exterior.second==4 && exterior.first>25,"wide flat source escapes all caustics");
 }while(std::next_permutation(order.begin(),order.end()));
 check(!tr::ExteriorCausticBound(wide,{.5,.5,0.}).valid(),"zero mass is not a positive triple");
 auto coincident=wide;coincident[2]=coincident[1];
 check(!tr::ExteriorCausticBound(coincident,mass).valid(),"coincident poles refuse exterior proof");
 // Three overlapping disks can surround a hole. Being outside every
 // disk is insufficient; all four escape rays must remain blocked here.
 {
  const double side=3.7;
  const std::array<tr::Complex,3> triangle{{{0,0},{side,0},
      {side*.5,side*std::sqrt(3.)*.5}}};
  const tr::Complex centre(side*.5,side*std::sqrt(3.)/6);
  const tr::ExteriorCausticBound ring(triangle,{1./3,1./3,1./3});
  check(ring.radius()>side*.5 && ring.radius()<side/std::sqrt(3.),
        "bounding disks have an interior hole");
  for(auto lens:triangle)check(std::abs(centre-lens)>ring.radius(),
      "hole is outside every individual disk");
  check(ring.clearance(centre).second<0,"outside-only test must not certify winding zero");
 }
 const tr::ExteriorCausticBound bound(wide,mass);
 const int rounding=std::fegetround();
 std::fesetround(FE_UPWARD);
 check(bound.clearance({0,.001}).second<0,"cached proof obeys floating environment gate");
 std::fesetround(rounding);
 check(bound.clearance({0,.001}).second==4,"rounding restoration re-enables valid proof");
 std::cout<<"PASS: "<<critical_checks<<" critical enclosures, "<<counts<<" independent winding counts and six lens orderings\n";
}

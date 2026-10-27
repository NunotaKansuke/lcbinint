#include "lcbinint/magnification/algebraic_boundary/triple/radial_events.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
void check(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
int main(){
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};auto plan=radial_event_plan(p,200000);
 check(plan.completed,plan.reason);check(plan.events.size()==12,"event count fixture");
 check(angular_topology(plan.rmin,p).arcs.empty() && angular_topology(plan.rmax,p).arcs.empty(),"exterior domain bounds");
 for(std::size_t j=0;j<plan.events.size();++j){
  const auto e=plan.events[j];const double theta=(e.reciprocal?std::acos(-1.):0)+2*std::atan(e.coordinate);
  const auto f=evaluate<true>(e.radius,theta,p);
  check(std::abs(f.phi)<1e-8 && std::abs(f.dtheta)<1e-7,"independent physical event residual");
  double gap=std::min(e.radius-(j?plan.events[j-1].radius:plan.rmin),
       (j+1<plan.events.size()?plan.events[j+1].radius:plan.rmax)-e.radius);
  auto before=angular_topology(e.radius-.1*gap,p),after=angular_topology(e.radius+.1*gap,p);
  check(before.completed && after.completed,"event side topology");
  check(std::abs(int(before.roots.size())-int(after.roots.size()))==2,"simple fold changes root count by two");
 }
 // Each fast centre remains inside a unique physical interval proof and
 // agrees with the independently forced promoted proof. Strict callers can
 // retain the original quad route, with truthful precision diagnostics.
 for(const auto& original:plan.events) {
  auto fast=original,wide=original;bool fast_quad=false,wide_quad=false;
  check(refine_event_center(fast,p,true,&fast_quad),"adaptive event precision");
  check(refine_event_center(wide,p,false,&wide_quad) && wide_quad,"forced quad event precision");
  check(fast.radius_lo<=wide.radius_hi && wide.radius_lo<=fast.radius_hi &&
        fast.coordinate_lo<=wide.coordinate_hi && wide.coordinate_lo<=fast.coordinate_hi,
        "double and quad event enclosures overlap");
  check(fast.radius_lo>=original.radius_lo && fast.radius_hi<=original.radius_hi &&
        fast.coordinate_lo>=original.coordinate_lo && fast.coordinate_hi<=original.coordinate_hi,
        "event precision retains original enclosure");
 }
 // Check coverage through independent full angular isolation, including
 // a source-limb crossing where the number of radial events changes.
 for(const auto& fixture:std::array<std::pair<Geometry,int>,3>{{
       {p,12},
       {{-.087243824247718094,-.52604606110908503,.03,.2,.08,1.,-.4,.6},10},
       {{-.087211699298757425,-.5259953857930596,.03,.2,.08,1.,-.4,.6},12}}}) {
  const auto packet=radial_event_plan(fixture.first,200000);
  check(packet.completed && int(packet.events.size())==fixture.second,"limb crossing event coverage");
  for(std::size_t band=0;band<=packet.events.size();++band) {
   const double lo=band?packet.events[band-1].radius_hi:packet.rmin;
   const double hi=band<packet.events.size()?packet.events[band].radius_lo:packet.rmax;
   if(!(hi>lo))continue;
   int expected=-1;
   for(double fraction:{.2,.5,.8}) {
    const auto topology=angular_topology(lo+(hi-lo)*fraction,fixture.first);
    check(topology.completed,"independent inter-event angular isolation");
    const int count=int(topology.roots.size());
    if(expected<0)expected=count;
    check(count==expected,"no unreported root-count change inside event band");
   }
  }
 }
 auto budget=radial_event_plan(p,1);check(!budget.completed,"event budget fails closed");
 std::cout<<"PASS: global interval event coverage, "<<plan.events.size()<<" physical folds; boxes="<<plan.boxes<<"\n";
}

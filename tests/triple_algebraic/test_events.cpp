#include "lcbinint/magnification/algebraic_boundary/triple/radial_events.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
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
 // A 1e-8 companion at separation 100 has two radial folds only
 // about 1e-13 apart. A fixed 32*eps*(1+r) proposal cannot prove either
 // one unique, even when promoted Newton has already converged.
 {
  const double q=1e-8,s=100.,q2=1.,sep=100.1,ang=std::acos(-1.);
  const double off=q*s/(1+q),den=1+q+q2,alpha=-.9190931997774809;
  const Complex source(-std::cos(alpha)-.5*std::sin(alpha),
                       -std::sin(alpha)+.5*std::cos(alpha));
  const auto frame=primary_frame({Complex(-s/(1+q),0),Complex(off,0),
      Complex(off,0)-sep*std::polar(1.,ang)},source,.1,1/den,q2/den);
  RadialEvent event{5.0055296119529426e-11,.005503362362115934,
      4.9827893990889966e-11,5.0282698248168885e-11,true,
      .0055030781356412801,.0055036465885914777};
  const auto original=event;bool quad=false;
  check(refine_event_center(event,frame.geometry,false,&quad) && quad,
        "pole-local event needs a narrower proof proposal");
  check(event.radius_lo>=original.radius_lo && event.radius_hi<=original.radius_hi &&
        event.coordinate_lo>=original.coordinate_lo && event.coordinate_hi<=original.coordinate_hi,
        "local repair stays inside retained event enclosure");
  // The contracted Krawczyk image itself need not be strictly invariant;
  // independently enclose the tiny neighbourhood around the retained centre.
  const double w=std::ldexp(32*std::numeric_limits<double>::epsilon(),-16);
  const event_detail::Box box{event.radius-w,event.radius+w,
      std::nextafter(event.coordinate-w,-INFINITY),
      std::nextafter(event.coordinate+w,INFINITY),event.reciprocal};
  check(event_detail::prove_quad(box,frame.geometry).kind==2,
        "independent local physical uniqueness proof");
 }
 auto budget=radial_event_plan(p,1);check(!budget.completed,"event budget fails closed");
 std::cout<<"PASS: global interval event coverage, "<<plan.events.size()<<" physical folds; boxes="<<plan.boxes<<"\n";
}

#include "lcbinint/magnification/algebraic_boundary/triple/radial_events.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
namespace lcbinint::algebraic_boundary::triple::physical_detail {
template<class T> std::optional<Jet<T>> physical_reference(T radius,T coordinate,const Geometry& p,bool reciprocal_chart) {
    using J=Jet<T>;using C=coefficient_detail::C<J>;
    const J R(radius,T(1),T(0)),t(coordinate,T(0),T(1)),one(1);
    const J den=one+t*t;
    if(!positive_value(den.v) || !positive_value(radius))return std::nullopt;
    const J inverse=reciprocal_jet(den),sign(reciprocal_chart?-1:1);
    const C unit{sign*(one-t*t)*inverse,sign*J(2)*t*inverse};
    const C z=C{R}*unit;
    // At a fixed polar radius, z-m0/conj(z)=(R-m0/R)*unit.
    // This identity also holds for the R/t jets and their interval extensions.
    const J radial=R-(J(1)-J(p.m1)-J(p.m2))*reciprocal_jet(R);
    C g=C{radial}*unit-C{J(p.X),J(p.Y)};
    const std::array<C,3> lenses{C{},C{J(p.a)},C{J(p.bx),J(p.by)}};
    const std::array<J,3> masses{J(1)-J(p.m1)-J(p.m2),J(p.m1),J(p.m2)};
    for(int j=1;j<3;++j) {
        const C d=coefficient_detail::conj(z-lenses[j]);
        const J norm=d.re*d.re+d.im*d.im;
        if(!positive_value(norm.v))return std::nullopt;
        const J inv=reciprocal_jet(norm);
        g=g-C{masses[j]}*coefficient_detail::conj(d)*C{inv};
    }
    return J(p.rho)*J(p.rho)-g.re*g.re-g.im*g.im;
}
}
int main(){
 // Independent endpoint squares, including signed zero and extreme ranges.
 for(int e:{-1074,-600,-1,0,500,1023})for(double lo:{-1.,-.5,0.,.25})for(double width:{.25,.75}) {
  const double a=std::ldexp(lo,e),b=std::ldexp(lo+width,e);
  auto squared=physical_detail::square_value(angular_detail::I<double>(a,b));
  for(double x:{a,a+(b-a)*.5,b}) {
   const __float128 q=(__float128)x*x;
   if(q<(__float128)squared.lo || q>(__float128)squared.hi)
    throw std::runtime_error("dependent square lost promoted point");
  }
 }
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};
 for(int k=0;k<100;++k){double R=.2+.0147*k,t=-.91+.018*k;
  for(bool chart:{false,true}){
   auto f=physical_detail::physical(R,t,p,chart);auto a=physical_detail::angular(R,t,p,chart);
   auto v=physical_detail::boundary_value(R,t,p,chart);
   if(!f || !a || !v)throw std::runtime_error("ordinary physical evaluation failed");
   const auto direct=evaluate<true>(R,(chart?std::acos(-1.):0)+2*std::atan(t),p);
   const double expected=direct.phi*p.rho*p.rho;
   if(std::abs(f->v-expected)>1e-10*(1+std::abs(expected)) ||
      std::abs(f->r-direct.dR*p.rho*p.rho)>1e-9*(1+std::abs(f->r)) ||
      std::abs(f->t-direct.dtheta*p.rho*p.rho*2/(1+t*t))>1e-9*(1+std::abs(f->t)) ||
      std::abs(a->v-f->v)+std::abs(a->t-f->t)+std::abs(*v-f->v)>1e-10*(1+std::abs(f->v)+std::abs(f->t)))
    throw std::runtime_error("physical jet disagrees with original lens map");
   using I=angular_detail::I<double>;
   auto box=physical_detail::physical(I(R-1e-6,R+1e-6),I(t-1e-6,t+1e-6),p,chart);
   auto q=physical_detail::physical_reference((__float128)R,(__float128)t,p,chart);
   auto fast=physical_detail::Ray<I>(I(R-1e-6,R+1e-6),p).event_jet(I(t-1e-6,t+1e-6),chart);
   if(!fast)throw std::runtime_error("direct event enclosure failed");
   const physical_detail::Ray<I> ray(I(R-1e-6,R+1e-6),p);
   const auto angular_box=ray.angular(I(t-1e-6,t+1e-6),chart);
   const auto value_box=ray.value(I(t-1e-6,t+1e-6),chart);
   if(!angular_box || !value_box)throw std::runtime_error("angular enclosure failed");

   if(!box || !q)throw std::runtime_error("physical enclosure failed");
   auto encloses=[](I x,__float128 y){return (__float128)x.lo<=y && y<=(__float128)x.hi;};
   if(!encloses(box->v,q->v)||!encloses(box->r,q->r)||!encloses(box->t,q->t)||!encloses(box->rt,q->rt)||!encloses(box->tt,q->tt))
    throw std::runtime_error("physical interval jet lost promoted point");
   for(double rr:{R-1e-6,R+1e-6})for(double tt:{t-1e-6,t+1e-6}) {
    auto corner=physical_detail::physical_reference((__float128)rr,(__float128)tt,p,chart);
    if(!corner || !encloses(*value_box,corner->v) ||
       !encloses(angular_box->v,corner->v) || !encloses(angular_box->t,corner->t))
     throw std::runtime_error("real-scaled angular enclosure lost independent promoted corner");
    if(!corner || !encloses(fast->v,corner->v)||!encloses(fast->r,corner->r)||!encloses(fast->t,corner->t)||
       !encloses(fast->rt,corner->rt)||!encloses(fast->tt,corner->tt))
     throw std::runtime_error("direct event jet lost independent promoted corner");

    if(!corner || !encloses(box->v,corner->v)||!encloses(box->r,corner->r)||!encloses(box->t,corner->t)||
       !encloses(box->rt,corner->rt)||!encloses(box->tt,corner->tt))
     throw std::runtime_error("physical square jet lost promoted corner");
   }
  }
 }
 auto plan=radial_event_plan(p,1000000);
 if(!plan.completed)throw std::runtime_error(plan.reason);
 for(auto e:plan.events) {
  event_detail::Box b{e.radius_lo,e.radius_hi,e.coordinate_lo,e.coordinate_hi,e.reciprocal};
  if(event_detail::physical_exclusion(b,p))throw std::runtime_error("disk screen excluded certified event");
  // A screen must also retain every larger box containing a true event.
  // These straddle fold extrema and exercise cancellation in the stationary
  // numerator, independently of the planner's chosen subdivisions.
  for(double width:{1e-8,1e-5,.001,.02,.2}) {
   event_detail::Box expanded{std::max(plan.rmin,e.radius_lo-width),e.radius_hi+width,
       e.coordinate_lo-width,e.coordinate_hi+width,e.reciprocal};
   if(event_detail::physical_exclusion(expanded,p))throw std::runtime_error("disk screen excluded event neighbourhood");
  }
 }
 std::cout<<"physical jets, interval enclosures, and pole/derivative screen verified\n";
}

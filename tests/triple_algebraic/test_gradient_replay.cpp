#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/gradient_replay.hpp"
#include <random>
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
// Generic automatic spatial jets provide an independent arithmetic check of
// the hand-derived complex event kernel, including off-real chart points.
static physical_detail::Jet<Complex> reference_complex_event(Complex radius,Complex coordinate,const Geometry& p,bool chart) {
 using namespace physical_detail;using J=Jet<Complex>;using C=coefficient_detail::C<J>;
 const J R(radius,Complex(1),Complex(0)),t(coordinate,Complex(0),Complex(1)),one(1);
 const J t2=square_jet(t),inverse=reciprocal_jet(one+t2),sign(chart?-1:1);
 const C unit{sign*(one-t2)*inverse,sign*J(2)*t*inverse},z=C{R}*unit;
 const J radial=R-(J(1)-J(p.m1)-J(p.m2))*reciprocal_jet(R);
 C g=C{radial}*unit-C{J(p.X),J(p.Y)};
 const std::array<C,2> lens{C{J(p.a)},C{J(p.bx),J(p.by)}};
 const std::array<J,2> mass{J(p.m1),J(p.m2)};
 for(int k=0;k<2;++k){const auto d=coefficient_detail::conj(z-lens[k]);const auto inv=reciprocal_jet(square_jet(d.re)+square_jet(d.im));g=g-C{mass[k]}*coefficient_detail::conj(d)*C{inv};}
 return J(p.rho)*J(p.rho)-square_jet(g.re)-square_jet(g.im);
}
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){
 // Independent polynomial moments catch weight/node errors, including the
 // highest integrated degree and both signs. Tables are shared by all calls.
 for(int n:{8,12,16,24,32,64,128}) {
  const auto& rule=shape_gradient_detail::gauss_rule(n);
  for(int power=0;power<2*n;++power){long double sum=0;
   for(int k=0;k<n;++k){require(rule.w[k]>0 && std::abs(rule.x[k])<1,"Gaussian interior positive rule");sum+=(long double)rule.w[k]*std::pow((long double)rule.x[k],power);}
   const long double exact=power%2?0:2.L/(power+1);
   require(std::abs(sum-exact)<3e-14L,"Gaussian moment");
  }
 }
 std::mt19937 random(1703);std::uniform_real_distribution<double> uniform(-1,1);
 int concave_arcs=0;
 for(int i=0;i<200;++i) {
  Geometry p{uniform(random),uniform(random),.1+.2*std::abs(uniform(random)),.2,.08,1.,-.4,.6};
  const double R=.3+1.4*std::abs(uniform(random)),theta=3*uniform(random);
  shape_gradient_detail::Ray ray(p,R);double error=0,curvature=0;
  const auto actual=ray.sample(theta,error,&curvature),reference=evaluate<true>(R,theta,p);
  auto close=[](double a,double b){return std::abs(a-b)<2e-11*(1+std::abs(b));};
  require(close(actual.phi,reference.phi)&&close(actual.dR,reference.dR)&&close(actual.dtheta,reference.dtheta),"replay independent geometry jet");
  for(int j=0;j<8;++j)require(close(actual.dp[j],reference.dp[j]),"replay parameter jet");
  for(double h:{1e-5,3e-6}) {
   const double fd=(evaluate<true>(R,theta+h,p).dtheta-evaluate<true>(R,theta-h,p).dtheta)/(2*h);
   require(std::abs(curvature-fd)<2e-5*(1+std::abs(fd)),"angular curvature independent difference");
  }
  // The skip condition bounds the entire arc, not just its center. Check
  // its implication against the independent physical curvature at 33 points.
  const double half=.001+.03*std::abs(uniform(random));
  if(ray.strictly_concave(theta-half,theta+half)) {
   ++concave_arcs;
   for(int k=0;k<=32;++k) {
    const double angle=theta-half+2*half*k/32,h=1e-5;
    const double fd=(evaluate<true>(R,angle+h,p).dtheta-evaluate<true>(R,angle-h,p).dtheta)/(2*h);
    require(fd<0,"whole-arc concavity guard");
   }
  }
  for(bool chart:{false,true}) {
   const double t=.7*uniform(random);
   const auto real=physical_detail::physical(R,t,p,chart);
   const auto complex=shape_gradient_detail::complex_event(Complex(R),Complex(t),p,chart);
   require(bool(real),"real event jet exists");
   const Complex rc(R,.03*uniform(random)),tc(t,.03*uniform(random));
   const auto fast_complex=shape_gradient_detail::complex_event(rc,tc,p,chart);
   const auto reference_complex=reference_complex_event(rc,tc,p,chart);
   for(auto pair:{std::pair<Complex,Complex>{fast_complex.v,reference_complex.v},
       {fast_complex.r,reference_complex.r},{fast_complex.t,reference_complex.t},
       {fast_complex.rt,reference_complex.rt},{fast_complex.tt,reference_complex.tt}})
    require(std::abs(pair.first-pair.second)<2e-10*(1+std::abs(pair.second)),"direct complex jet versus automatic jet");
   for(auto pair:{std::pair<Complex,double>{complex.v,real->v},{complex.r,real->r},
       {complex.t,real->t},{complex.rt,real->rt},{complex.tt,real->tt}})
    require(std::abs(pair.first-Complex(pair.second))<2e-11*(1+std::abs(pair.second)),"complex chart agrees on real axis");
  }
 }
 require(concave_arcs>10,"concavity guard exercised");
 // A retained trace owns inclusion/exclusion data after the primal returns.
 // Compare the optional replay to independent high-accuracy value differences.
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};auto plan=radial_event_plan(p,200000);
 require(plan.completed,"replay support");IntegrationOptions o;o.absolute_tolerance=0;o.relative_tolerance=1e-4;
 const auto value=integrate(p,.5,o,&plan);require(value.converged,"primal convergence");
 require(value.parameter_derivative_evaluations==0,"value-only derivative isolation");
 o.gradient_value_mesh=true;const auto tangent=integrate<true>(p,.5,o,&plan);
 require(tangent.converged && tangent.gradient_valid,"replay convergence");
 require(!tangent.gradient_precision_checked && !tangent.gradient_reliable,"finite replay is not certified");
 require(std::abs(tangent.magnification-value.magnification)<=value.error_estimate+tangent.error_estimate,"primal flux contract preserved");
 double Geometry::* member[]={&Geometry::X,&Geometry::Y,&Geometry::rho,&Geometry::m1,&Geometry::m2,&Geometry::a,&Geometry::bx,&Geometry::by};
 IntegrationOptions reference;reference.absolute_tolerance=0;reference.relative_tolerance=1e-8;
 for(int j=0;j<8;++j)for(double h:{3e-5,1e-5}) {
  auto plus=p,minus=p;plus.*member[j]+=h;minus.*member[j]-=h;
  const auto vp=integrate(plus,.5,reference),vm=integrate(minus,.5,reference);
  require(vp.converged && vm.converged,"difference values converged");
  const double fd=(vp.magnification-vm.magnification)/(2*h);
  if(!(std::abs(fd-tangent.normalized.df0[j])<1e-4+2e-4*std::abs(fd))) {
   std::cerr<<j<<" replay="<<tangent.normalized.df0[j]<<" fd="<<fd<<"\n";
   throw std::runtime_error("retained shape replay finite difference");
  }
 }
}

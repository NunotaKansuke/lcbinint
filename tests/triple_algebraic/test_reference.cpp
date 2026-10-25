#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <iomanip>
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
using GK=boost::math::quadrature::gauss_kronrod<double,61>;
int main(int argc,char** argv){
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};
 const std::string name=argc>1?argv[1]:"contact";
 if(name=="large")p.rho=.3;
 else if(name=="regular"){p.X=.5;p.Y=.4;p.rho=.03;}
 else if(name=="asymmetric")p=primary_frame(std::array<Complex,3>{Complex(.3,-.2),Complex(.8,.9),Complex(-.7,.6)},Complex(.2,.4),.12,.25,.1).geometry;
 auto plan=radial_event_plan(p,1000000);
 if(!plan.completed)throw std::runtime_error(plan.reason);
 std::vector<double> cuts{plan.rmin};for(auto e:plan.events)cuts.push_back(e.radius);
 cuts.push_back(p.a);cuts.push_back(std::hypot(p.bx,p.by));cuts.push_back(plan.rmax);std::sort(cuts.begin(),cuts.end());
 double uniform=0,limb=0,outer_error=0;
 for(size_t j=1;j<cuts.size();++j){
  double lo=cuts[j-1],hi=cuts[j];auto witness=angular_topology((lo+hi)/2,p);
  if(!witness.completed)throw std::runtime_error(witness.reason);
  if(witness.arcs.empty())continue;
  for(int component=0;component<2;++component){
   auto integrand=[&](double x){
    double pi=std::acos(-1.0),t=(x+1)*.5,half=(hi-lo)*.5;
    double R=lo+(hi-lo)*std::pow(std::sin(pi*.5*t),2),jac=half*pi*.5*std::sin(pi*t);
    auto topology=angular_topology(R,p,{1e-13,8192,true});
    if(!topology.completed)throw std::runtime_error(topology.reason);
    double sum=0;
    for(auto arc:topology.arcs){
     if(component==0)sum+=arc.hi-arc.lo;
     else {
      auto intensity=[&](double theta){double phi=evaluate(R,theta,p).phi;
       if(phi< -1e-8)throw std::runtime_error("reference outside arc");
       return std::sqrt(std::max(0.,phi));};
      double error=0;
      sum+=GK::integrate(intensity,arc.lo,arc.hi,12,1e-9,&error);
     }
    }
    return R*jac*sum;
   };
   double error=0,value=GK::integrate(integrand,-1.,1.,10,1e-8,&error);
   if(component)limb+=value;else uniform+=value;outer_error+=error;
  }
 }
 const double pi=std::acos(-1.),mu_uniform=uniform/(pi*p.rho*p.rho);
 const double mu_limb=(.5*uniform+.5*limb)/(pi*p.rho*p.rho*(1-.5/3));
 IntegrationOptions options;options.absolute_tolerance=1e-9;options.relative_tolerance=1e-8;options.maximum_refinements=1024;
 auto actual=integrate(p,.5,options,&plan);
 std::cout<<std::setprecision(16)<<name<<" independent_GK uniform="<<mu_uniform<<" limb="<<mu_limb
  <<" outer_error="<<outer_error<<" actual="<<actual.magnification<<" difference="<<actual.magnification-mu_limb<<" converged="<<actual.converged
  <<" error="<<actual.error_estimate<<" stop="<<actual.reason<<std::endl;
 if(!actual.converged || std::abs(actual.magnification-mu_limb)>2e-7)throw std::runtime_error("independent quadrature mismatch");
}

#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include <iostream>
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
using namespace lcbinint::algebraic_boundary::triple;
int main(){
 const double pi=std::acos(-1.);
 const auto symmetric=primary_frame(std::array<Complex,3>{Complex(1,0),std::polar(1.,2*pi/3),std::polar(1.,4*pi/3)},Complex(0,0),.1,1./3,1./3).geometry;
 IntegrationOptions strict;strict.absolute_tolerance=1e-7;strict.relative_tolerance=1e-6;
 for(double u:{0.,.5}) {
  auto r=integrate<true>(symmetric,u,strict);
  if(!r.converged || !r.gradient_reliable)throw std::runtime_error(r.reason);
  if(std::abs(r.normalized.df0[0])+std::abs(r.normalized.df0[1])>1e-6)
   throw std::runtime_error("threefold source symmetry gradient");
 }
 int index=0;
 for(auto p:{Geometry{.1,0.,.15,.2,.08,1.,-.4,0.},Geometry{0.,0.,.15,.1,.1,1.,-1.,0.},
       Geometry{.1,.05,.01,.001/1.0011,.0001/1.0011,1.2,.7,.3}}){
  IntegrationOptions options;options.absolute_tolerance=1e-6;options.relative_tolerance=1e-5;
  auto value=integrate<true>(p,.5,options);
  std::cout<<index++<<" "<<value.reason<<" mu="<<value.magnification<<" boxes="<<value.support.boxes
           <<" quad_samples="<<value.quad_samples<<" nodes="<<value.value_evaluations<<std::endl;
  if(!value.converged || !value.gradient_reliable)throw std::runtime_error(value.reason);
  if(p.Y==0 && std::abs(value.normalized.df0[1])>1e-6)
      throw std::runtime_error("axial source reflection derivative");
 }
}

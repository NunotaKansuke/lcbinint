#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
int main(){Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};
 auto plan=radial_event_plan(p,200000);
 auto v=integrate(p,.5,{},&plan);
 std::cout<<v.reason<<" A="<<v.magnification<<" err="<<v.error_estimate<<" calls="<<v.value_evaluations<<" refinements="<<v.refinements<<"\n";
 auto g=integrate<true>(p,.5,{},&plan);
 std::cout<<g.reason<<" A="<<g.magnification<<" err="<<g.error_estimate<<" calls="<<g.value_evaluations<<" refinements="<<g.refinements<<"\n";
 for(int j=0;j<8;++j)std::cout<<g.normalized.df0[j]<<" ";std::cout<<"\n";
 return v.converged&&g.gradient_valid?0:1;}

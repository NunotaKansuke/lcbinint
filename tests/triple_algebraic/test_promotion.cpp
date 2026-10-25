#include "lcbinint/magnification/algebraic_boundary/triple/radius_terms.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
int main(){
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};bool exercised=false;
 for(int k=0;k<60 && !exercised;++k){double R=.3+.018*k;
  RadiusWorkspace<true> late,early;
  if(!late.initialize(R,p) || late.topology.arcs.empty() || !late.refine(2) || late.quad_samples)continue;
  const auto before=late.value_evaluations;
  if(!late.prepare_precise_arc(0) || !late.refine(5))throw std::runtime_error(late.reason);
  if(!early.initialize(R,p) || !early.prepare_precise_arc(0) || !early.refine(5))throw std::runtime_error(early.reason);
  const auto a=late.result(),b=early.result();
  for(int j=0;j<8;++j)if(std::abs(a.df0[j]-b.df0[j])+std::abs(a.dfh[j]-b.dfh[j])>1e-11*(1+std::abs(b.df0[j])+std::abs(b.dfh[j])))
   throw std::runtime_error("late endpoint promotion left stale derivatives");
  if(late.value_evaluations<before || late.reused_samples==0)throw std::runtime_error("samples discarded during promotion");
  exercised=true;
 }
 if(!exercised)throw std::runtime_error("no late-promotion case exercised");
 std::cout<<"late endpoint promotion agrees with early promotion and reuses nodes\n";
}

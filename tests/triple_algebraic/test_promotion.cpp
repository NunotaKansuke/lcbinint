#include "lcbinint/magnification/algebraic_boundary/triple/radius_terms.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
int main(){
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};bool exercised=false;
 for(int k=0;k<60 && !exercised;++k){double R=.3+.018*k;
  RadiusWorkspace<> late,early;
  if(!late.initialize(R,p) || late.topology.arcs.empty() || !late.refine(2) || late.quad_samples)continue;
  const auto before=late.value_evaluations;
  if(!late.prepare_precise_arc(0) || !late.refine(5))throw std::runtime_error(late.reason);
  if(!early.initialize(R,p) || !early.prepare_precise_arc(0) || !early.refine(5))throw std::runtime_error(early.reason);
  const auto a=late.result(),b=early.result();
  if(std::abs(a.f0-b.f0)+std::abs(a.fh-b.fh)>1e-11*(1+std::abs(b.f0)+std::abs(b.fh)))
   throw std::runtime_error("late endpoint promotion left stale values");
  if(late.value_evaluations<before || late.reused_samples==0)throw std::runtime_error("samples discarded during promotion");
  exercised=true;
 }
 if(!exercised)throw std::runtime_error("no late-promotion case exercised");
 // The derivative workspace owns its endpoint precision and transported
 // packets. Incremental and fresh refinement must evaluate each packet once.
 for(double R:{.82,.89,.93,1.04,1.115,1.19}){
  RadiusWorkspace<true> retained,fresh;
  if(!retained.initialize(R,p)||!fresh.initialize(R,p)||!retained.refine(2))throw std::runtime_error("transport initialization");
  const auto before=retained.value_evaluations;
  if(!retained.refine(7)||!fresh.refine(7))throw std::runtime_error("transport refinement");
  const auto count=retained.value_evaluations;
  if(count<before||count!=fresh.value_evaluations||!retained.refine(7)||count!=retained.value_evaluations)
   throw std::runtime_error("transport discarded retained packets");
  for(int j=0;j<8;++j)if(retained.result().df0[j]!=fresh.result().df0[j]||retained.result().dfh[j]!=fresh.result().dfh[j])
   throw std::runtime_error("transport cache changed derivatives");
 }
 std::cout<<"scalar endpoint promotion and derivative packet retention\n";
}

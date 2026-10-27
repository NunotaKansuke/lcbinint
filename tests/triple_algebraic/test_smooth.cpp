#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include <iostream>
namespace tr=lcbinint::algebraic_boundary::triple;
void check(bool x,const char*s){if(!x)throw std::runtime_error(s);}
int main(){
 tr::Geometry p{.5,.4,.03,.2,.08,1.,-.4,.6};tr::PreparedRegularSupport prepared;
 auto plan=prepared.plan(p,1e-4,1000000,false);check(plan.completed,"complete full source support");
 tr::IntegrationOptions reference;reference.absolute_tolerance=0;reference.relative_tolerance=1e-7;
 auto radial=tr::integrate<true>(p,.5,reference,&plan);
 check(radial.converged&&radial.gradient_valid,"tight image-plane reference");
 for(double u:{0.,.5,1.}){
  tr::IntegrationOptions opt;opt.absolute_tolerance=0;opt.relative_tolerance=1e-4;
  auto result=tr::integrate<true>(p,u,opt,&plan);
  check(result.converged&&result.gradient_valid,"smooth image-plane convergence");
  check(!result.smooth_source_used&&result.smooth_source_samples==0&&result.smooth_source_reused_samples==0,"no source-plane shortcut");
  if(u==.5){
   check(std::abs(result.magnification-radial.magnification)<=1e-4*std::abs(radial.magnification),"smooth flux agreement");
   for(int j=0;j<8;++j)check(std::abs(result.normalized.df0[j]-radial.normalized.df0[j])<=1e-4+2e-4*std::abs(radial.normalized.df0[j]),"smooth derivative agreement");
  }
 }
 auto wrong=p;wrong.X+=.001;
 check(tr::integrate<true>(wrong,.5,reference,&plan).reason=="support_plan_identity","reject stale support proof");
 std::cout<<"PASS: smooth image-plane policies, reference agreement and proof identity\n";
}

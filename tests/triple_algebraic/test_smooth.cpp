#include "lcbinint/magnification/algebraic_boundary/triple/smooth_source.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include <iostream>
namespace tr=lcbinint::algebraic_boundary::triple;
void check(bool x,const char*s){if(!x)throw std::runtime_error(s);}
int main(){
 for(double u:{0.,.5,1.})for(int level=1;level<=5;++level){
  auto w=tr::smooth_source::weights(level,u);int m=1<<level;
  for(int d=0;d<=std::min(m-2,6);++d){double sum=0;
   for(int k=1;k<m;++k){double z=(1+std::cos(std::acos(-1.)*k/m))*.5;sum+=w[k-1]*std::pow(z,d);}
   double exact=2*((1-u)/(d+2)+u/(d+3))/(1-u/3);
   check(std::abs(sum-exact)<3e-14,"source product-weight moment");
  }
 }
 tr::Geometry p{.5,.4,.03,.2,.08,1.,-.4,.6};tr::PreparedRegularSupport prepared;
 auto plan=prepared.plan(p,1e-4,1000000,false);check(plan.completed,"complete full source support");
 // Unchecked tangent arithmetic must not weaken primal point bounds.
 for(tr::Complex offset:{tr::Complex{},tr::Complex(.01,.005)}){
  auto source=tr::Complex(p.X,p.Y)+offset;
  auto checked=tr::smooth_source::point(source,1e-14,p,plan.component_images);
  auto fast=tr::smooth_source::point(source,1e-14,p,plan.component_images,true);
  check(checked.ok&&fast.ok,"finite checked and value-controlled point derivatives");
  check(checked.mu.value==fast.mu.value&&checked.mu.error==fast.mu.error,"unchanged primal point bounds");
  for(int j=0;j<7;++j)check(std::abs(checked.g[j].value-fast.g[j].value)<1e-12*(1+std::abs(checked.g[j].value)),"shared analytic tangent parity");
 }
 auto disk=tr::smooth_source::integrate(p,.5,0,1e-6,128,plan);
 check(disk.converged&&disk.reused_samples>0,"strict source convergence and nested reuse");
 tr::IntegrationOptions opt;opt.absolute_tolerance=0;opt.relative_tolerance=1e-7;
 auto radial=tr::integrate<true>(p,.5,opt,&plan);check(radial.converged&&radial.gradient_reliable,"independent strict radial reference");
 check(std::abs(radial.magnification-disk.mu)<=radial.error_estimate+disk.error,"source flux agreement");
 for(int j=0;j<8;++j)check(std::abs(radial.normalized.df0[j]-disk.gradient[j])<=radial.normalized.gradient_error[j]+disk.gradient_error[j]+1e-8,"source derivative agreement");
 auto vm=tr::smooth_source::integrate(p,.5,0,1e-4,128,plan,true);check(vm.converged,"value mesh convergence");
 for(int j=0;j<8;++j)check(std::isfinite(vm.gradient[j])&&std::isinf(vm.gradient_error[j]),"value mesh unchecked finite derivatives");
 auto wrong=p;wrong.X+=.001;check(!tr::smooth_source::integrate(wrong,.5,0,1e-4,128,plan).attempted,"reject stale source proof");
 std::cout<<"PASS: product moments, strict radial agreement, cached refinement, policy and proof identity\n";
}

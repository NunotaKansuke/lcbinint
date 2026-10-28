#include <time.h>
static double cpu_time(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return 1e9*t.tv_sec+t.tv_nsec;}
#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
#include "lcbinint/model/triple_lens_geometry.hpp"
#include <iostream>
#include <iomanip>
using namespace lcbinint;namespace tr=algebraic_boundary::triple;
int main(int argc,char**argv){int id,j;double t0,u0,tE,alpha,s,q,q2,sep,ang,rho,limb,tol,t;
 std::cout<<std::setprecision(17);
 while(std::cin>>id>>j>>t0>>u0>>tE>>alpha>>s>>q>>q2>>sep>>ang>>rho>>limb>>tol>>t){
 auto g=model::make_triple_lens_geometry_vbm(s,q,sep,ang,q2);std::array<tr::Complex,3> l;for(int k=0;k<3;++k)l[k]={g.lens_positions[k].x,g.lens_positions[k].y};double tau=(t-t0)/tE;tr::Complex source(tau*std::cos(alpha)-u0*std::sin(alpha),tau*std::sin(alpha)+u0*std::cos(alpha));
 const std::array<std::array<int,3>,3>orders{{{0,1,2},{1,0,2},{2,0,1}}};
 for(int k=0;k<3;++k){auto o=orders[k];std::array<tr::Complex,3>a{l[o[0]],l[o[1]],l[o[2]]};auto f=tr::primary_frame<true>(a,source,rho,g.masses[o[1]],g.masses[o[2]]);tr::PreparedRegularSupport prep;tr::IntegrationOptions opt;opt.absolute_tolerance=0;opt.relative_tolerance=tol;if(k)opt.event_box_budget=8192;
 const double start=cpu_time();auto plan=prep.plan(f.geometry,tol,opt.event_box_budget,false);double prepared=cpu_time()-start; const double begin=cpu_time();opt.maximum_refinements=argc>1?std::atoi(argv[1]):4;auto r=tr::integrate<false>(f.geometry,limb,opt,plan.completed?&plan:nullptr);double elapsed=cpu_time()-begin;
 std::cout<<"{\"case\":"<<id<<",\"epoch\":"<<j<<",\"frame\":"<<k<<",\"plan_seconds\":"<<prepared/1e9<<",\"pilot_seconds\":"<<elapsed/1e9<<",\"completed\":"<<r.converged<<",\"stop\":\""<<r.reason<<"\",\"nodes\":"<<r.value_evaluations<<",\"refinements\":"<<r.refinements<<",\"support_events\":"<<plan.events.size()<<",\"component_images\":"<<plan.component_images.size()<<",\"support_completed\":"<<plan.completed<<",\"error_ratio\":"<<(std::isfinite(r.error_estimate/(tol*std::abs(r.magnification)))?r.error_estimate/(tol*std::abs(r.magnification)):-1)<<"}\n"<<std::flush;
 }}}

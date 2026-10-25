#include "VBMicrolensingLibrary.h"
#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include <chrono>
using namespace lcbinint;
namespace tr=algebraic_boundary::triple;
struct Output {double mu,error,seconds;int samples,annuli,converged;};
struct Algebraic {std::array<Complex,3> lenses;std::array<double,3> masses;tr::PreparedRegularSupport prepared;};
extern "C" {
void* vbm_create(int method,double* geometry){auto* b=new VBMicrolensing();if(method>=0)b->SetMethod(static_cast<VBMicrolensing::Method>(method));b->a2=0;b->SetLDprofile(VBMicrolensing::LDlinear);b->SetLensGeometry(3,geometry);return b;}
void vbm_destroy(void* p){delete static_cast<VBMicrolensing*>(p);}
void vbm_point(void* p,double x,double y,double rho,double limb,double rtol,int route,Output* out){auto&b=*static_cast<VBMicrolensing*>(p);b.Tol=0;b.RelTol=rtol;b.a1=limb;b.NPS=0;b.nannuli=0;b.therr=NAN;auto start=std::chrono::steady_clock::now();double mu=route==0?b.MultiMag(x,y,rho):b.MultiMagDark(x,y,rho,0.);out->seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();out->mu=mu;out->error=b.therr;out->samples=b.NPS;out->annuli=b.nannuli;out->converged=-1;}
void* algebraic_create(double* g){auto*b=new Algebraic;for(int j=0;j<3;++j){b->lenses[j]={g[3*j],g[3*j+1]};b->masses[j]=g[3*j+2];}return b;}
void algebraic_destroy(void*p){delete static_cast<Algebraic*>(p);}
void algebraic_point(void*p,double x,double y,double rho,double limb,double rtol,int,Output*out){auto&b=*static_cast<Algebraic*>(p);auto g=tr::primary_frame(b.lenses,{x,y},rho,b.masses[1],b.masses[2]).geometry;tr::IntegrationOptions opt;opt.absolute_tolerance=0;opt.relative_tolerance=rtol;auto start=std::chrono::steady_clock::now();auto plan=b.prepared.plan(g,rtol);auto r=tr::integrate(g,limb,opt,plan.completed?&plan:nullptr);out->seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();out->mu=r.magnification;out->error=r.error_estimate;out->samples=r.value_evaluations;out->annuli=0;out->converged=r.converged&&r.support_complete;}
}
extern "C" void algebraic_stages(void*p,double x,double y,double rho,double limb,double rtol,Output*out,double*cost) {
 auto&b=*static_cast<Algebraic*>(p);auto g=tr::primary_frame(b.lenses,{x,y},rho,b.masses[1],b.masses[2]).geometry;
 tr::IntegrationOptions opt;opt.absolute_tolerance=0;opt.relative_tolerance=rtol;
#ifdef STAGE_SEED_COUNTERS
 auto native_before=b.prepared.native_seed_solves(),hit_before=b.prepared.continued_seed_hits();
#endif
 auto t0=std::chrono::steady_clock::now();auto plan=b.prepared.plan(g,rtol);auto t1=std::chrono::steady_clock::now();
 if(!plan.completed)plan=tr::radial_event_plan(g,opt.event_box_budget,rtol*.1);
 auto t2=std::chrono::steady_clock::now();auto r=tr::integrate(g,limb,opt,&plan);auto t3=std::chrono::steady_clock::now();
 cost[0]=std::chrono::duration<double>(t1-t0).count();cost[1]=std::chrono::duration<double>(t2-t1).count();cost[2]=std::chrono::duration<double>(t3-t2).count();
#ifdef STAGE_SEED_COUNTERS
 cost[3]=b.prepared.native_seed_solves()-native_before;cost[4]=b.prepared.continued_seed_hits()-hit_before;cost[5]=r.component_domains;
#else
 cost[3]=cost[4]=-1;cost[5]=1;
#endif
 out->seconds=std::chrono::duration<double>(t3-t0).count();out->mu=r.magnification;out->error=r.error_estimate;out->samples=r.value_evaluations;out->annuli=0;out->converged=r.converged&&r.support_complete;
}

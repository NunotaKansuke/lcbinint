#include "lcbinint/magnification/algebraic_boundary/triple/regular_support.hpp"
#include <chrono>
#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/stream_integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/transport_stream.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/patterson_stream.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/gradient_replay.hpp"
namespace tr=lcbinint::algebraic_boundary::triple;
#ifdef LCBININT_TRIPLE_FOLD_LEADING_CONTROL
extern "C" void fold_counts(double* out){for(int i=0;i<3;++i)out[i]=tr::fold_control_counts[i];}
extern "C" void reset_fold_counts(){tr::fold_control_counts.fill(0);}
#endif
#ifdef LCBININT_TRIPLE_PARTIAL_SECTOR
extern "C" void partial_counts(double* out){for(int i=0;i<4;++i)out[i]=tr::partial_sector_counts[i];}
extern "C" void reset_partial_counts(){tr::partial_sector_counts.fill(0);}
#endif
#ifdef LCBININT_TRIPLE_SECTOR_PROFILE
extern "C" void sector_counts(double* out){for(int i=0;i<2;++i)out[i]=tr::sector_plan_profile[i];}
extern "C" void reset_sector_counts(){tr::sector_plan_profile.fill(0);}
#endif
struct State {tr::PreparedRegularSupport prepared;std::string reason;};
extern "C" void* create(){return new State;}
extern "C" void destroy(void*p){delete static_cast<State*>(p);}
template<bool G> void run(State&s,const double*p,double u,double tol,int mode,double*out){
 tr::Geometry g{p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]};
 tr::IntegrationOptions o;o.absolute_tolerance=0;o.relative_tolerance=tol;o.maximum_refinements=tol<1e-6?1000:160;
 const auto t0=std::chrono::steady_clock::now();
 auto plan=s.prepared.plan(g,tol,o.event_box_budget,!G);const auto t1=std::chrono::steady_clock::now();
 auto r=[&] {if constexpr(G){if(mode==2){
#if defined(STREAM_PATTERSON)
return tr::integrate_patterson<G>(g,u,o,plan.completed?&plan:nullptr);
#elif defined(STREAM_TRANSPORT)
return tr::integrate_transport<G>(g,u,o,plan.completed?&plan:nullptr);
#elif defined(STREAM_COUPLED)
return tr::integrate_coupled<G>(g,u,o,plan.completed?&plan:nullptr);
#else
return tr::integrate_stream<G>(g,u,o,plan.completed?&plan:nullptr);
#endif
}}return tr::integrate<G>(g,u,o,plan.completed?&plan:nullptr);}();const auto t2=std::chrono::steady_clock::now();
 out[20]=std::chrono::duration<double,std::milli>(t1-t0).count();out[21]=std::chrono::duration<double,std::milli>(t2-t1).count();
 s.reason=r.reason;out[0]=r.magnification;out[1]=r.error_estimate;out[2]=r.converged;out[3]=r.support_complete;
 out[4]=r.gradient_valid;out[5]=r.value_evaluations;out[6]=r.parameter_derivative_evaluations;out[7]=r.quad_samples;
 out[8]=r.refinements;out[22]=r.h_splits;out[23]=r.moving_radial_panels;out[9]=r.global_radii;out[10]=r.continued_radii;
 if constexpr(G)for(int j=0;j<8;++j)out[12+j]=r.normalized.df0[j];
}
extern "C" void evaluate(void*p,const double*g,double u,double tol,int mode,double*out){if(mode)run<true>(*static_cast<State*>(p),g,u,tol,mode,out);else run<false>(*static_cast<State*>(p),g,u,tol,mode,out);}
extern "C" void replay_tuned(void*ptr,const double*p,double u,double tol,int radial,int angular,double*out){
 auto&s=*static_cast<State*>(ptr);tr::Geometry g{p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]};
 tr::IntegrationOptions o;o.absolute_tolerance=0;o.relative_tolerance=tol;
 const auto t0=std::chrono::steady_clock::now();
 auto plan=s.prepared.plan(g,tol,o.event_box_budget,false);const auto t1=std::chrono::steady_clock::now();tr::ValueIntegralTrace trace;
 auto r=tr::integrate<false>(g,u,o,plan.completed?&plan:nullptr,&trace);
 const auto t2=std::chrono::steady_clock::now();
 const auto d=tr::replay_shape_gradient(g,u,trace,r.support,radial,angular);const auto t3=std::chrono::steady_clock::now();
 out[22]=d.paired_arcs;out[23]=d.paired_declines;
 out[9]=d.reused_gradient_panels;out[10]=d.reused_gradient_radii;out[11]=d.geometry_checked_panels;
 out[6]=std::chrono::duration<double,std::milli>(t1-t0).count();out[7]=std::chrono::duration<double,std::milli>(t2-t1).count();out[8]=std::chrono::duration<double,std::milli>(t3-t2).count();
 out[0]=r.magnification;out[1]=r.error_estimate;out[2]=r.converged;out[3]=d.completed;out[4]=d.evaluations;out[5]=d.quad_samples;
 const double den=std::acos(-1.)*g.rho*g.rho*(1-u/3);
 for(int j=0;j<8;++j)out[12+j]=((1-u)*d.uniform[j]+u*d.limb[j])/den;
 out[14]-=2*((1-u)*d.uniform_flux+u*d.limb_flux)/den/g.rho;
}

extern "C" const char* reason(void*p){return static_cast<State*>(p)->reason.c_str();}

#ifdef LCBININT_TRIPLE_REPLAY_PROFILE
extern "C" void replay_profile(void*ptr,const double*p,double u,double tol,double*out){
 tr::shape_gradient_detail::complex_search_ns=0;tr::shape_gradient_detail::complex_search_calls=0;
 replay_tuned(ptr,p,u,tol,6,4,out);
 out[24]=tr::shape_gradient_detail::complex_search_ns/1e6;out[25]=tr::shape_gradient_detail::complex_search_calls;
}
#endif


#ifdef LCBININT_TRIPLE_COUPLED_PROFILE
extern "C" void reset_profile(){tr::coupled_profile::nanoseconds.fill(0);tr::coupled_profile::calls.fill(0);tr::coupled_profile::chart_nanoseconds.fill(0);tr::coupled_profile::chart_calls.fill(0);}
extern "C" void get_profile(double*out){for(int i=0;i<tr::coupled_profile::count;++i){out[i]=tr::coupled_profile::nanoseconds[i];out[6+i]=tr::coupled_profile::calls[i];}}
#endif

extern "C" void stream_trials(double*out){for(int i=0;i<12;++i)out[i]=tr::stream_detail::trials[i];}

extern "C" void transport_counts(double*out){for(int i=0;i<6;++i)out[i]=tr::fused_detail::transport_counts[i];}

#ifdef LCBININT_TRIPLE_DIAGNOSTIC_TUNING
extern "C" void evaluate_tuned(void*h,const double*p,double u,double tol,int radial,double angular,int critical,double*out){
 const auto previous=tr::patterson_tuning;tr::patterson_records.clear();tr::patterson_tuning={angular,radial,critical};evaluate(h,p,u,tol,2,out);tr::patterson_tuning=previous;
}
#endif

#ifdef LCBININT_TRIPLE_DIAGNOSTIC_TUNING
extern "C" int panel_count(){return int(tr::patterson_records.size());}
extern "C" void panel_record(int i,double*out){const auto& r=tr::patterson_records.at(i);out[0]=r.lo;out[1]=r.hi;out[2]=r.level;out[3]=r.radial_error;out[4]=r.inner_error;out[5]=r.cells.size();out[6]=r.points;out[7]=r.value.f0;out[8]=r.value.fh;for(int j=0;j<8;++j){out[9+j]=r.value.df0[j];out[17+j]=r.value.dfh[j];}}
extern "C" void panel_cell(int i,int j,double*out){const auto& c=tr::patterson_records.at(i).cells.at(j);for(int k=0;k<5;++k)out[k]=c[k];}
extern "C" void evaluate_selected(void*h,const double*p,double u,double tol,int radial,double lo,double hi,double*out){
 const auto previous=tr::patterson_tuning;tr::patterson_records.clear();tr::patterson_tuning={1,radial,8,lo,hi};evaluate(h,p,u,tol,2,out);tr::patterson_tuning=previous;
}
#endif

#include "lcbinint/magnification/algebraic_boundary/triple/frame.hpp"
extern "C" void evaluate_permuted(void*h,const double*p,double u,double tol,int mode,int primary,double*out){
 if(primary==0){evaluate(h,p,u,tol,mode,out);return;}
 const std::array<tr::Complex,3> original{tr::Complex{},tr::Complex(p[5],0),tr::Complex(p[6],p[7])};
 const std::array<double,3> masses{1-p[3]-p[4],p[3],p[4]};const std::array<int,3> order=primary==1?std::array<int,3>{1,0,2}:std::array<int,3>{2,0,1};
 const std::array<tr::Complex,3> lenses{original[order[0]],original[order[1]],original[order[2]]};
 if(!mode){const auto frame=tr::primary_frame<false>(lenses,{p[0],p[1]},p[2],masses[order[1]],masses[order[2]]);const auto&g=frame.geometry;const double q[8]={g.X,g.Y,g.rho,g.m1,g.m2,g.a,g.bx,g.by};evaluate(h,q,u,tol,mode,out);return;}
 const auto frame=tr::primary_frame<true>(lenses,{p[0],p[1]},p[2],masses[order[1]],masses[order[2]]);const auto&g=frame.geometry;const double q[8]={g.X,g.Y,g.rho,g.m1,g.m2,g.a,g.bx,g.by};evaluate(h,q,u,tol,mode,out);
 std::array<double,8> local{};for(int j=0;j<8;++j)local[j]=out[12+j];const auto physical=frame.pullback(local);
 std::array<tr::Complex,3> position{};for(int j=0;j<3;++j)position[order[j]]={physical[5+2*j],physical[6+2*j]};
 out[12]=physical[0];out[13]=physical[1];out[14]=physical[2];out[15]=out[16]=0;
 for(int j=0;j<2;++j){const int index=order[j+1];out[15]+=physical[3+j]*(index==0?-1:index==1?1:0);out[16]+=physical[3+j]*(index==0?-1:index==2?1:0);}
 out[17]=position[1].real();out[18]=position[2].real();out[19]=position[2].imag();
}

#ifdef LCBININT_TRIPLE_DIAGNOSTIC_TUNING
extern "C" void evaluate_charted(void*h,const double*p,double u,double tol,double width,int radial,double*out){const auto previous=tr::patterson_tuning;tr::patterson_records.clear();tr::patterson_tuning={1,radial,8,-INFINITY,INFINITY,width};evaluate(h,p,u,tol,2,out);tr::patterson_tuning=previous;}
#endif

extern "C" void support_workload(void*ptr,const double*p,double tol,int omit,double*out){auto&s=*static_cast<State*>(ptr);tr::Geometry g{p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]};const auto plan=s.prepared.plan(g,tol,200000,omit);out[0]=plan.completed;out[1]=plan.bounded_component_value;out[2]=plan.bounded_component_error;out[3]=plan.excluded_images.size();out[4]=plan.events.size();out[5]=plan.component_images.size();}

#ifdef LCBININT_TRIPLE_COUPLED_PROFILE
extern "C" void get_chart_profile(double*out){for(int i=0;i<tr::coupled_profile::chart_count;++i){out[i]=tr::coupled_profile::chart_nanoseconds[i];out[3+i]=tr::coupled_profile::chart_calls[i];}}
#endif

#ifdef LCBININT_TRIPLE_DIAGNOSTIC_TUNING
extern "C" void evaluate_quad_uniform(void*h,const double*p,double u,double tol,double width,int radial,double*out){const auto previous=tr::patterson_tuning;tr::patterson_records.clear();tr::patterson_tuning={1,radial,8,-INFINITY,INFINITY,width,true};evaluate(h,p,u,tol,2,out);tr::patterson_tuning=previous;}
#endif

#ifdef LCBININT_TRIPLE_DIAGNOSTIC_TUNING
extern "C" void evaluate_controlled(void*h,const double*p,double u,double tol,double width,int radial,double angular,int critical,double*out){const auto previous=tr::patterson_tuning;tr::patterson_records.clear();tr::patterson_tuning={angular,radial,critical,-INFINITY,INFINITY,width};evaluate(h,p,u,tol,2,out);tr::patterson_tuning=previous;}
#endif

#ifdef LCBININT_TRIPLE_ORDINARY_MOMENTS
extern "C" void ordinary_moment_counts(double*out){for(int k=0;k<8;++k)out[k]=tr::shape_gradient_detail::ordinary_moment_counts[k];}
extern "C" void reset_ordinary_moment_counts(){tr::shape_gradient_detail::ordinary_moment_counts={};}
#endif

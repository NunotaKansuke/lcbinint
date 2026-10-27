#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/algebraic_boundary/source_disk_gradient.hpp"
#include "lcbinint/magnification/algebraic_boundary/seeded_radial_frontier_value.hpp"
#include <limits>
namespace ab=lcbinint::algebraic_boundary;
using namespace lcbinint;
struct State {
 magnification::FiniteSourceMagnifier finite; magnification::PointSourceMagnifier point;
 State(double u,double tol):finite([&]{magnification::FiniteSourceSettings s;s.limb_darkening_c=u;s.finite_source_tol=0;s.finite_source_reltol=tol;return s;}()){}
};
thread_local std::string message;
extern "C" void* create(double u,double tol){return new State(u,tol);}
extern "C" void destroy(void*p){delete static_cast<State*>(p);}
extern "C" const char* reason(){return message.c_str();}
extern "C" void evaluate(void*ptr,const double*p,double u,double tol,int mode,double*out){
 std::fill(out,out+20,0);message.clear();auto&s=*static_cast<State*>(ptr);
 try {
  const magnification::AlgebraicBoundarySettings opt{0,tol};
  magnification::FiniteSourceResult value;
  if(mode==0){auto v=s.finite.algebraic_boundary_binary_mag(p[4],p[3],{p[0],p[1]},p[2],opt,&value,nullptr,true);out[0]=v.magnification;}
  else if(mode==1){auto g=s.finite.algebraic_boundary_binary_jacobian(p[4],p[3],{p[0],p[1]},p[2],opt,&value,nullptr,true,&s.point);out[0]=g.magnification;out[3]=g.success;out[4]=g.grad_reliable;for(int j=0;j<5;++j)out[10+j]=g.jacobian[j];}
  else {
   ab::LensParams lens{p[0],p[1],p[2],p[3],p[4],true};ab::FusedGradientOptions options{0,tol};ab::FusedGradientResult g;int images=0;
   const bool smooth=mode==2&&ab::source_disk_detail::evaluate(lens,u,options,s.finite,s.point,g,images);
   if(!smooth){auto seeds=ab::make_caustic_aware_binary_seeds(s.finite,s.point,p[4],p[3],{p[0],p[1]},p[2]);std::vector<std::array<double,2>> positions;for(auto z:seeds.radial_seeds)positions.push_back({z.re,z.im});g=ab::epoch_fused_gradient(lens,u,positions,options,{},&seeds.support_plan);out[8]=seeds.support_proven;}
   out[0]=g.mu;out[1]=g.value_error_estimate;out[2]=g.value_converged;out[3]=std::all_of(g.gradient_valid.begin(),g.gradient_valid.end(),[](bool b){return b;});out[4]=out[3]&&out[2];out[5]=7;out[6]=smooth;out[7]=g.node_evaluations;for(int j=0;j<5;++j)out[10+j]=g.grad_mu[j];message=g.stop;return;
  }
  out[1]=value.error_estimate;out[2]=value.value_converged||value.converged;out[5]=int(value.decision.method);out[6]=value.diagnostic_stop.find("tangent_chart=smooth_source_disk")!=std::string::npos;out[7]=value.decision.estimated_evaluations;out[8]=value.seed_support_proven;out[9]=value.integral_support_certified;message=value.diagnostic_stop+"; "+value.decision.reason;
 }catch(const std::exception&e){out[0]=NAN;message=e.what();}
}

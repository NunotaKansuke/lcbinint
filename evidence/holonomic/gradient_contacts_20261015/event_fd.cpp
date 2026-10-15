#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
#include "fused_k.hpp"
using namespace lcbinint::holonomic_fused;
struct E { long double r; double drho; bool map; };
int main(){
 const double s=.18626282170711353,q=1.0/5.1248562665389e-7,rho0=.46850782321964624;
 const double alpha=1.4366018640558513,tau=-rho0,u0=0,limb=.5,tol=1e-8;
 const double xs=tau*std::cos(alpha)-u0*std::sin(alpha),ys=tau*std::sin(alpha)+u0*std::cos(alpha);
 lcbinint::magnification::FiniteSourceSettings settings;settings.caustic_bins=1400;
 lcbinint::magnification::FiniteSourceMagnifier finite(settings);finite.binary_caustic_branches(s,q);
 lcbinint::magnification::PointSourceMagnifier point;
 auto run=[&](double rho){
  LensParams p{xs,ys,rho,q,s,true};
  const auto seeds=make_caustic_aware_binary_seeds(finite,point,s,q,{xs,ys},rho);
  AdaptiveConfig cfg;cfg.tol.mu_atol=tol;cfg.tol.mu_rtol=tol;cfg.with_jacobian=true;cfg.fused_value_mesh=true;cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;
  AdaptiveWorkspace workspace;const auto result=epoch_seeded_radial_frontier(p,limb,seeds.radial_seeds,cfg,workspace);
  std::vector<E> out;const auto pf=PrimaryFrame::from(p);
  for(const auto&e:result.topology.events)if(adaptive_detail::adaptive_physical_fold_event(e)&&e.radius<1e-5){
   const auto x=adaptive_detail::topology_event_estimate(e,pf,tol,true);auto left=x;left.radius-=1;
   const auto map=fused_gradient::make_material_map(left,x,pf,true,false);
   out.push_back({(long double)x.radius+x.radius_lo,map.db[2],map.valid});
  }
  std::sort(out.begin(),out.end(),[](const E&a,const E&b){return a.r<b.r;});
  std::printf("RUN rho=%.17g conv=%d stop=%s roots=%zu",rho,int(result.integral.value_converged),adaptive_stop_name(result.integral.stop),out.size());
  for(const auto&e:out)std::printf(" [%.20Lg %.12g %d]",e.r,e.drho,int(e.map));std::puts("");
  return out;
 };
 auto base=run(rho0);
 for(double f:{1e-4,1e-5,1e-6,1e-7}){
  const double h=rho0*f;auto plus=run(rho0+h),minus=run(rho0-h);
  if(base.size()!=2||plus.size()!=2||minus.size()!=2)continue;
  for(int j=0;j<2;++j){const double fd=double((plus[j].r-minus[j].r)/(2*h));std::printf("FD fraction=%.1e root=%d analytic=%.12g fd=%.12g rel=%.3g\n",f,j,base[j].drho,fd,std::fabs(fd-base[j].drho)/std::max(1e-30,std::fabs(fd)));}
 }
}

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"

int main(){
 using namespace lcbinint; using namespace lcbinint::holonomic;
 constexpr double s=.95, q=1e7, rho=1e-6, u0=9.846481110931334e-7, alpha=.5, tE=10.;
 magnification::FiniteSourceSettings settings; settings.caustic_bins=1400;
 magnification::FiniteSourceMagnifier finite(settings); magnification::PointSourceMagnifier point;
 finite.binary_caustic_branches(s,q);
 std::cout<<std::setprecision(17);
 for(auto [name,t]: {std::pair<const char*,double>{"neg",-1.7455113096375089e-6},{"center",0.},{"pos",1.7455113096375089e-6}}){
  const double tau=t/tE, xs=tau*std::cos(alpha)-u0*std::sin(alpha), ys=tau*std::sin(alpha)+u0*std::cos(alpha);
  const SourcePosition source{xs,ys};
  auto seed=make_caustic_aware_binary_seeds(finite,point,s,q,source,rho);
  LensParams p{xs,ys,rho,q,s,true};
  std::cout<<"CASE "<<name<<" xs="<<xs<<" ys="<<ys<<" seeds="<<seed.radial_seeds.size()<<" support="<<seed.support_proven<<"\n";
  for(double tol:{1e-3,1e-5,1e-7}) for(bool fold:{true,false}) {
   AdaptiveConfig cfg; cfg.tol.mu_atol=tol;cfg.tol.mu_rtol=tol;cfg.with_jacobian=true;
   cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;
   cfg.fold_maps=fold;cfg.initial_level=fold?3:8;cfg.collect_diagnostics=true;
   AdaptiveWorkspace w; auto r=epoch_seeded_radial_frontier(p,.5,seed.radial_seeds,cfg,w);
   const auto& a=r.integral;
   std::cout<<"RUN tol="<<tol<<" fold="<<fold<<" mu="<<a.mu<<" valerr="<<a.estimated_abs_error_mu<<" conv="<<a.value_converged<<" stop="<<adaptive_stop_name(a.stop)<<" nodes="<<a.stats.node_evaluations<<" panels="<<w.panels.size()<<" refine="<<a.stats.refinement_history.size()<<" reject=";
   for(auto x:a.stats.sample_reject_counts)std::cout<<x<<',';
   std::cout<<" cold=";for(auto x:a.stats.sample_cold_reject_counts)std::cout<<x<<',';
   std::cout<<" graderr=";for(auto x:a.estimated_abs_error_grad)std::cout<<x<<',';
   std::cout<<" gradq=";for(auto x:a.grad_quality)std::cout<<int(x)<<',';
   std::cout<<"\n";
   std::vector<const AdaptivePanel*> active;for(const auto& x:w.panels)if(x.active)active.push_back(&x);
   std::sort(active.begin(),active.end(),[](auto*a,auto*b){return a->error[0]+a->error[1]+a->error[2]+a->error[3]+a->error[4]+a->error[5] > b->error[0]+b->error[1]+b->error[2]+b->error[3]+b->error[4]+b->error[5];});
   for(size_t j=0;j<std::min<size_t>(4,active.size());++j){auto&x=*active[j];std::cout<<" P cell="<<x.cell<<':'<<x.cell_end<<" R="<<x.map.a<<':'<<x.map.b<<" lv="<<x.level<<" dep="<<x.depth<<" err=";for(auto e:x.error)std::cout<<e<<',';std::cout<<" radial=";for(auto e:x.radial)std::cout<<e<<',';std::cout<<" inner=";for(auto e:x.inner)std::cout<<e<<',';std::cout<<" event=";for(auto e:x.event)std::cout<<e<<',';std::cout<<"\n";}
   if(!a.stats.sample_diagnostics.empty()){std::cout<<" REJ sample=";for(size_t j=0;j<std::min<size_t>(5,a.stats.sample_diagnostics.size());++j){const auto&d=a.stats.sample_diagnostics[j];std::cout<<adaptive_sample_reject_reason_name(d.reason)<<"@"<<d.R<<"/c"<<d.cell<<"/l"<<d.level<<' ';}std::cout<<"\n";}
   if(!a.stats.event_diagnostics.empty()){std::cout<<" EVT ";for(const auto&d:a.stats.event_diagnostics){if(std::abs(d.input_radius-rho)<5e-8 || std::abs(d.selected_radius-rho)<5e-8)std::cout<<"in="<<d.input_radius<<" out="<<d.selected_radius<<" unc="<<d.uncertainty<<" tier="<<d.precision_tier<<" qf="<<event_decision_reason_name(d.qf_reason)<<"; ";}std::cout<<"\n";}
  }
 }
}

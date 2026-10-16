#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
#include "../../../src/lcbinint/magnification/holonomic/fused_gradient.hpp"
int main() {
 constexpr double s=.710053684596912,qv=7.886132346228951e-7,q=1./qv,rho=.1671007233467463,u0=1e-9,alpha=1.186247434824958;
 const double tau=-rho,xs=tau*cos(alpha)-u0*sin(alpha),ys=tau*sin(alpha)+u0*cos(alpha);
 std::cout<<std::setprecision(17);
 lcbinint::magnification::FiniteSourceSettings settings;settings.caustic_bins=1400;
 lcbinint::magnification::FiniteSourceMagnifier finite(settings);lcbinint::magnification::PointSourceMagnifier point;
 finite.binary_caustic_branches(s,q);
 auto seeds=lcbinint::holonomic_fused::make_caustic_aware_binary_seeds(finite,point,s,q,{xs,ys},rho);
 lcbinint::holonomic_fused::LensParams p{xs,ys,rho,q,s,true};
 for(double tol:{1e-3,1e-4,1e-5,3e-6}) {
  lcbinint::holonomic_fused::AdaptiveConfig cfg;cfg.tol.mu_atol=tol;cfg.tol.mu_rtol=tol;cfg.with_jacobian=true;cfg.fused_value_mesh=true;
  cfg.gradient_policy=lcbinint::holonomic_fused::GradientPolicy::ValueFirst;cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;
  lcbinint::holonomic_fused::AdaptiveWorkspace w;
  auto r=lcbinint::holonomic_fused::epoch_seeded_radial_frontier(p,0.5,seeds.radial_seeds,cfg,w);
  std::vector<std::array<double,2>> api_seeds;for(const auto&z:seeds.radial_seeds)api_seeds.push_back({z.re,z.im});
  lcbinint::holonomic::FusedGradientOptions gopts;gopts.mu_atol=tol;gopts.mu_rtol=tol;
  auto api=lcbinint::holonomic::epoch_fused_gradient({xs,ys,rho,q,s,true},0.5,api_seeds,gopts);
  std::cout<<"API "<<tol<<" NODES "<<api.node_evaluations<<" GRAD";for(double v:api.grad_mu)std::cout<<' '<<v;std::cout<<'\n';
  std::cout<<"TOL "<<tol<<" MU "<<r.integral.mu<<" NODES "<<r.integral.stats.node_evaluations<<" EVENTS "<<r.topology.events.size()<<" CELLS "<<r.topology.cells.size()<<" PANELS "<<w.panels.size()<<" GRAD";for(double v:r.integral.grad_mu)std::cout<<' '<<v;std::cout<<'\n';
  bool complete=false;
  auto pairs=lcbinint::holonomic_fused::geometry_experiment::detect(
      lcbinint::holonomic_fused::PrimaryFrame::from(p),r.topology,seeds.radial_seeds,
      &complete,(1<<cfg.initial_level)-1);
  std::cout<<"GEOMETRY complete "<<complete<<" pairs "<<pairs.size()<<'\n';
  for(const auto& pair:pairs)std::cout<<"PAIR "<<pair.left<<' '<<pair.right<<" scale "<<pair.scale<<" real "<<pair.real<<" existing "<<pair.existing<<'\n';
  std::vector<lcbinint::holonomic_fused::RadialEvent> ev=r.topology.events;
  std::sort(ev.begin(),ev.end(),[=](const auto&a,const auto&b){return std::abs(rho-a.radius)<std::abs(rho-b.radius);});
  int n=0;for(const auto&e:ev){if(n++>=30)continue;std::cout<<"EVENT "<<e.radius<<" GAP "<<rho-e.radius<<" RLO "<<e.radius_lo<<" PHYS "<<e.physically_real<<" KIND "<<e.kind<<" DETAIL "<<e.detail<<'\n';}
  for(const auto&c:r.topology.cells)if(rho-c.r_hi<1e-3)std::cout<<"CELL "<<c.r_lo<<' '<<c.r_hi<<" GAP "<<rho-c.r_hi<<" KIND "<<int(c.kind)<<" N "<<c.n_crossings<<'\n';
  for(const auto&x:w.panels)if(x.active){std::cout<<"PANEL "<<x.map.a<<' '<<x.map.b<<" DEPTH "<<x.depth<<" LEVEL "<<x.level<<" WEIGHTED "<<x.weighted<<" Q";for(double v:x.q)std::cout<<' '<<v;std::cout<<'\n';}
 }
}

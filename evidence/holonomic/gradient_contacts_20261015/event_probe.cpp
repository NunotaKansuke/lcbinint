#include <cmath>
#include <cstdio>
#include <map>
#include <cstdlib>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
#include "fused_k.hpp"
int main(int argc,char**argv){
 if(argc<9){std::fprintf(stderr,"usage s q rho alpha tau u0 limb tol [level]\n");return 2;}
 const double s=std::atof(argv[1]),q=std::atof(argv[2]),rho=std::atof(argv[3]);
 const double alpha=std::atof(argv[4]),tau=std::atof(argv[5]),u0=std::atof(argv[6]);
 const double limb=std::atof(argv[7]),tol=std::atof(argv[8]);const int level=argc>9?std::atoi(argv[9]):3;const bool strict=argc>10&&std::atoi(argv[10])!=0;
 const double xs=tau*std::cos(alpha)-u0*std::sin(alpha),ys=tau*std::sin(alpha)+u0*std::cos(alpha);
 lcbinint::magnification::FiniteSourceSettings settings;settings.caustic_bins=1400;
 lcbinint::magnification::FiniteSourceMagnifier finite(settings);finite.binary_caustic_branches(s,q);
 lcbinint::magnification::PointSourceMagnifier point;
 const auto seeds=lcbinint::holonomic_fused::make_caustic_aware_binary_seeds(finite,point,s,q,{xs,ys},rho);
 lcbinint::holonomic_fused::LensParams p{xs,ys,rho,q,s,true};
 lcbinint::holonomic_fused::AdaptiveConfig cfg;cfg.tol.mu_atol=tol;cfg.tol.mu_rtol=tol;cfg.with_jacobian=true;cfg.fused_value_mesh=true;cfg.gradient_policy=strict?lcbinint::holonomic_fused::GradientPolicy::Strict:lcbinint::holonomic_fused::GradientPolicy::ValueFirst;if(strict){cfg.tol.grad_atol.fill(tol);cfg.tol.grad_rtol.fill(tol);}cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;cfg.initial_level=level;
 lcbinint::holonomic_fused::AdaptiveWorkspace workspace;
 const auto result=lcbinint::holonomic_fused::epoch_seeded_radial_frontier(p,limb,seeds.radial_seeds,cfg,workspace);
 const auto&r=result.integral;std::printf("RESULT mu=%.17g drho=%.17g conv=%d stop=%s events=%zu spans=%zu nodes=%zu level=%d strict=%d\n",r.mu,r.grad_mu[2],int(r.value_converged),lcbinint::holonomic_fused::adaptive_stop_name(r.stop),result.topology.events.size(),r.stats.integration_spans,r.stats.node_evaluations,level,int(strict));std::map<int,std::array<double,4>> span_sums;for(const auto&panel:workspace.panels)if(panel.active){auto&v=span_sums[panel.evaluator_cell];v[0]+=panel.q[0];v[1]+=panel.q[3];v[2]=panel.map.a;v[3]=panel.map.b;}for(const auto&entry:span_sums)std::printf("SPAN idx=%d lo=%.17g hi=%.17g qmu=%.12g qgrad_rho=%.12g\n",entry.first,entry.second[2],entry.second[3],entry.second[0],entry.second[1]);for(const auto& panel:workspace.panels)if(panel.active&&panel.map.b-panel.map.a<1e-4)std::printf("PANEL cell=%d depth=%d level=%d lo=%.17g hi=%.17g xl=%.5g xr=%.5g qmu=%.8g qgrad_rho=%.8g errrho=%.3g innerrho=%.3g radialrho=%.3g\n",panel.cell,panel.depth,panel.level,panel.map.a,panel.map.b,panel.xl,panel.xr,panel.q[0],panel.q[3],panel.error[3],panel.inner[3],panel.radial[3]);
 const auto pf=lcbinint::holonomic_fused::PrimaryFrame::from(p);
 for(const auto&e:result.topology.events){if(!lcbinint::holonomic_fused::adaptive_detail::adaptive_physical_fold_event(e))continue;const auto x=lcbinint::holonomic_fused::adaptive_detail::topology_event_estimate(e,pf,tol,true);std::printf("EVENT R=%.17g kind=%s proj=%d tu=%d/%d top_pair=%.17g loc_t=%.17g/%d loc_u=%.17g/%d reciprocal=%d pair_valid=%d pair=%.17g dm=%.5g dv=%.5g uncertain=%.3g\n",e.radius,e.kind.c_str(),int(e.projective_fold_certified),int(e.fold_t_seed_valid),int(e.fold_u_seed_valid),e.fold_u_seed,x.t_seed,int(x.t_seed_valid),x.u_seed,int(x.u_seed_valid),int(x.seed_reciprocal),int(x.pair_motion_valid),x.pair_m,x.pair_dm,x.pair_dv,x.uncertainty);auto left=x;left.radius-=1.0;const auto map=lcbinint::holonomic_fused::fused_gradient::make_material_map(left,x,pf,true,false);std::printf("MAP R=%.17g valid=%d drho=%.17g\n",x.radius,int(map.valid),map.db[2]);}
}

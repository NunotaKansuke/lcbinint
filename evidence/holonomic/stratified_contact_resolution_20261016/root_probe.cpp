#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier_value.hpp"
int main(){using namespace lcbinint;using namespace lcbinint::holonomic;
 constexpr double s=.95,q=1e7,rho=1e-6,u0=9.846481110931334e-7,alpha=.5;
 magnification::FiniteSourceSettings settings;settings.caustic_bins=1400;magnification::FiniteSourceMagnifier finite(settings);magnification::PointSourceMagnifier point;finite.binary_caustic_branches(s,q);
 double xs=-u0*std::sin(alpha),ys=u0*std::cos(alpha);SourcePosition source{xs,ys};auto seed=make_caustic_aware_binary_seeds(finite,point,s,q,source,rho);LensParams p{xs,ys,rho,q,s,true};
 AdaptiveConfig cfg;cfg.tol.mu_atol=1e-3;cfg.tol.mu_rtol=1e-3;cfg.with_jacobian=true;cfg.gradient_policy=GradientPolicy::ValueFirst;cfg.value_first_gradient_node_budget=0;cfg.value_first_gradient_round_budget=0;cfg.fold_maps=false;cfg.initial_level=8;cfg.collect_diagnostics=true;AdaptiveWorkspace w;auto fr=epoch_seeded_radial_frontier(p,.5,seed.radial_seeds,cfg,w);auto pf=PrimaryFrame::from(p);
 std::cout<<std::setprecision(17)<<"fr stop="<<adaptive_stop_name(fr.integral.stop)<<"\n";
 for(size_t i=0;i<fr.topology.events.size();++i){auto&e=fr.topology.events[i];if(e.radius>1.9&&e.radius<1.95)std::cout<<"E "<<i<<" R="<<e.radius<<" real="<<e.physically_real<<" kind="<<e.kind<<" detail="<<e.detail<<"\n";}
 for(size_t i=0;i<fr.topology.cells.size();++i){auto&c=fr.topology.cells[i];if(c.r_lo>1.9&&c.r_lo<1.95)std::cout<<"C "<<i<<" "<<c.r_lo<<" "<<c.r_hi<<" n="<<c.n_crossings<<" k="<<int(c.kind)<<"\n";}
 double ev=1.9179771245669932;for(double dr:{1e-10,1e-8,1e-7,6.028148478e-7,1e-6,1e-5,3.2e-2}){
  double R=ev+dr;auto qf=boundary_quartic(R,pf);auto cert=certify_quartic(qf.p);auto roots=sturm_isolate_real_roots(cert.chart_coeffs,cert.root_count);std::vector<double>theta;
  for(double z:roots){double th=(cert.reciprocal?kPi:0.)+2*std::atan(z);th=std::fmod(th,kTwoPi);if(th<0)th+=kTwoPi;theta.push_back(th);}std::sort(theta.begin(),theta.end());
  auto st=arc_set_from_root_thetas(R,pf,theta);QuarticWarm qw;RootPairWarm rp;QuarticCoeffs pc;auto direct=arc_intervals(R,pf,&qw,&rp,&pc,&qf,cert.root_count);QuarticWarm q2;RootPairWarm r2;QuarticCoeffs p2;auto adapt=adaptive_detail::adaptive_arc_intervals(R,pf,&q2,&r2,&p2,cert.root_count,&qf);
  std::cout<<"R "<<R<<" dr "<<dr<<" cert="<<cert.certified<<" n="<<cert.root_count<<" tier="<<cert.precision_tier<<" recip="<<cert.reciprocal<<" sturmroots="<<roots.size()<<" sturmArc="<<int(st.kind)<<":"<<st.arcs.size()<<" direct="<<int(direct.kind)<<":"<<direct.arcs.size()<<" adaptive="<<int(adapt.kind)<<":"<<adapt.arcs.size()<<" theta=";for(double x:theta)std::cout<<x<<",";std::cout<<" gaps=";for(size_t i=1;i<theta.size();++i)std::cout<<theta[i]-theta[i-1]<<",";std::cout<<"\n";
 }
}

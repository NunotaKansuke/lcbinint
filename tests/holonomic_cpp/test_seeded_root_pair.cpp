#include <iostream>
#include "lcbinint/magnification/holonomic/seeded_boundary_events.hpp"
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic;
int main(){
 int checks=0,failures=0;
 auto check=[&](bool good){++checks;if(!good)++failures;};
 // P=(t-m)^2-v: the signed pair passes through an ordinary fold smoothly.
 const double m=.3;
 for(double v:{.04,1e-12,0.,-1e-12,-.04}){
  std::array<double,5> pc{m*m-v,-2*m,1,0,0};
  RootPair pair{m+.002,v+.003};
  auto correction=correct_root_pair(pair,pc,8,1e-14);
  check(correction.converged);
  check(std::fabs(pair.m-m)<1e-13&&std::fabs(pair.v-v)<1e-13);
  auto residual=eo_residuals(pair,pc);
  check(std::fabs(residual.E)+std::fabs(residual.O)<1e-13);
  check(std::fabs(tangency_determinant(m,pc)+2)<1e-14);
 }
 LensParams p;p.xs=.12;p.ys=.04;p.rho=.03;p.q=.001;p.a=1.1;
 auto pf=PrimaryFrame::from(p);
 for(double R:{.3,.9,1.10002,1.7})for(double u:{0.,1e-10,.3,-.5}){
  const auto reversed=adaptive_detail::reciprocal_local_fold_quantities<long double>(R,u,pf);
  const auto direct=local_fold_quantities<long double>(R,u,pf,true);
  check(std::isfinite(reversed.P)&&std::isfinite(reversed.Pt)&&std::isfinite(reversed.PR));
  check(reversed.P==direct.P&&reversed.Pt==direct.Pt&&reversed.Ptt==direct.Ptt&&reversed.Ptr==direct.Ptr);
 }
 // Projective contacts supplied without D14 metadata must reach the
 // existing adaptive event estimator, including the exact u=0 coordinate.
 p.ys=0;pf=PrimaryFrame::from(p);
 for(double R:re_detail::chart_p4_factor_roots(pf))if(R>0&&R<3){
  RadialEvent event;
  if(!seeded_boundary_detail::fold(R,0,true,pf,3,event)){check(false);continue;}
  auto estimate=adaptive_detail::topology_event_estimate(event,pf,1e-4);
  check(estimate.seed_reciprocal&&estimate.u_seed_valid&&!estimate.needs_qf);
  check(std::isfinite(estimate.uncertainty));
  check(std::fabs(estimate.radius-R)<1e-10);
 }
 std::cout<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}

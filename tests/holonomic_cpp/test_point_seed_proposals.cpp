#include <iostream>
#include "lcbinint/magnification/holonomic/seeded_point_epoch.hpp"
using namespace lcbinint::holonomic;
int main(){int checks=0,fail=0;for(double q:{.001,.1,1.,3.}){
 LensParams p;p.q=q;p.a=.9;p.xs=.13;p.ys=.04;p.rho=.02;auto pf=PrimaryFrame::from(p);
 for(auto z:{point_seed_detail::Z(-.5,.4),point_seed_detail::Z(.4,.7),point_seed_detail::Z(1.4,.2)}){
  auto s=point_seed_detail::critical_state(z,pf);double h=1e-6;
  auto xp=point_seed_detail::critical_state(z+h,pf),xm=point_seed_detail::critical_state(z-h,pf);
  auto yp=point_seed_detail::critical_state(z+point_seed_detail::Z(0,h),pf),ym=point_seed_detail::critical_state(z-point_seed_detail::Z(0,h),pf);
  double exact[]={s.hx,s.hy,s.gx,s.gy};double fd[]={(xp.h-xm.h)/(2*h),(yp.h-ym.h)/(2*h),(xp.g-xm.g)/(2*h),(yp.g-ym.g)/(2*h)};
  for(int i=0;i<4;++i){++checks;if(std::abs(exact[i]-fd[i])>1e-7*(1+std::abs(exact[i])))++fail;}
 }
 lcbinint::magnification::PointSourceMagnifier pm;auto a=point_seed_proposals(p,pm);
 p.barycentric=false;p.xs=pf.X;lcbinint::magnification::PointSourceMagnifier pm2;auto b=point_seed_proposals(p,pm2);
 ++checks;if(a.positions.size()!=b.positions.size())++fail;
 for(size_t i=0;i<std::min(a.positions.size(),b.positions.size());++i){++checks;if(std::hypot(a.positions[i].re-b.positions[i].re,a.positions[i].im-b.positions[i].im)>1e-10)++fail;}
 }

 // Analytic point-lens critical circle and witness acceptance.
 PrimaryFrame analytic{2.,1.,.001,0.,.01};
 for(double phase:{-.7,.2,1.3}){
  auto z=std::polar(1.,-.5*phase)*point_seed_detail::Z(1.00001,.00001);
  ++checks;if(!point_seed_detail::phase_correct(z,phase,analytic))++fail;
  ++checks;if(std::abs(1./(z*z)-std::polar(1.,phase))>1e-10)++fail;
  ++checks;if(!point_seed_detail::critical_phase_witness(z,analytic))++fail;
 }
 std::cout<<checks<<" checks, "<<fail<<" failures\n";return fail?1:0;}

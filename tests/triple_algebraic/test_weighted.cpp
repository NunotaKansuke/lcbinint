#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary;
using namespace lcbinint::algebraic_boundary::triple;
int main(){
 for(int lev=1;lev<=8;++lev){
  const Fejer2Rule reference(lev);
  const auto& fejer=fejer_rule(lev);const auto& weighted=radial_rule(lev,true);
  if(fejer.x!=reference.x || fejer.w!=reference.w || fejer.norm_w!=reference.norm_w || fejer.interp!=reference.interp ||
     weighted.x!=reference.x || weighted.w!=reference.norm_w || weighted.interp!=reference.interp)
   throw std::runtime_error("lazy Fejer or weighted rule changed");
  if(&fejer!=&fejer_rule(lev) || &weighted!=&radial_rule(lev,true))
   throw std::runtime_error("canonical quadrature rule not reused");
  const auto& rule=linear_radial_rule(lev);const int m=1<<lev;
  for(int i=1;i<m;++i){
   const long double theta=3.1415926535897932384626433832795029L*i/m;long double sum=0;
   for(int k=1;k<m;k+=2)sum+=4*sinl(k*theta)/(k*(4.L-k*k));
   if(rule.w[i-1]!=double(2*sinl(theta)*sum/m))throw std::runtime_error("lazy product weights changed");
  }
  if(&rule!=&linear_radial_rule(lev))throw std::runtime_error("product rule not reused");
 }
 Geometry p{.5,.4,.03,.2,.08,1,-.4,.6};
 auto plan=radial_event_plan(p,1000000);
 for(double u:{0.,.5}){
  IntegrationOptions coarse;coarse.absolute_tolerance=0;coarse.relative_tolerance=1e-3;
  IntegrationOptions fine;fine.absolute_tolerance=1e-9;fine.relative_tolerance=1e-8;
  auto a=integrate(p,u,coarse,&plan),b=integrate(p,u,fine,&plan);
  if(!a.converged || !b.converged || std::abs(a.magnification-b.magnification)>coarse.relative_tolerance*std::abs(b.magnification))
   throw std::runtime_error("isolated weighted integration mismatch");
  if(u==0 && (a.value_evaluations || a.quad_samples || a.parameter_derivative_evaluations))
   throw std::runtime_error("uniform integration evaluated limb or derivative samples");
 }
 std::cout<<"lazy product rules bitwise unchanged; isolated weighted value convergence\n";
}

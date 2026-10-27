#include "lcbinint/magnification/algebraic_boundary/triple/gradient_hermite.hpp"
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple::shape_gradient_detail;
int main() {
 const double R=1.2,l=-.4,h=.6,lr=.2,hr=-.1,lp=.3,hp=-.2,lrp=.12,hrp=-.07,K=.7,Kr=.13,Kp=-.23,Krp=.17;
 const double H=(h-l)/2,Hr=(hr-lr)/2,Hp=(hp-lp)/2,Hrp=(hrp-lrp)/2,root=std::sqrt(K);
 const double value=R*H*H*root,radial=(H*H+2*R*H*Hr)*root+R*H*H*Kr/(2*root);
 const double gradient=2*R*H*Hp*root+R*H*H*Kp/(2*root);
 const double cross=2*(H*Hp+R*Hr*Hp+R*H*Hrp)*root+R*H*Hp*Kr/root+(H*H+2*R*H*Hr)*Kp/(2*root)+R*H*H*(Krp/(2*root)-Kr*Kp/(4*root*root*root));
 int points=0;
 for(double v0:{0.,.07})for(double x:{-.999,-.9,-.1,0.,.3,.9,.999}) {
  const double t=(x+1)/2,theta=l+(h-l)*t,a=theta-l,b=h-theta;
  PolarMixedJet q;q.phi=a*b*K;q.r=(-lr*b+hr*a)*K+a*b*Kr;q.t=(b-a)*K;
  q.rr=-2*lr*hr*K+2*(-lr*b+hr*a)*Kr;q.rt=(lr+hr)*K+(b-a)*Kr;q.tt=-2*K;
  q.p[0]=(-lp*b+hp*a)*K+a*b*Kp;
  q.rp[0]=(-lrp*b-lr*hp+hrp*a-hr*lp)*K+(-lr*b+hr*a)*Kp+(-lp*b+hp*a)*Kr+a*b*Krp;
  q.tp[0]=(lp+hp)*K+(b-a)*Kp;
  std::array<double,8> P{},RP{},A{},C{},v{},s{};const double s0=v0==0?0:-.09;v[0]=v0;s[0]=s0;
  P[0]=Hp+v0*Hr;RP[0]=Hrp+s0*Hr;A[0]=(lp+v0*lr)*(1-t)+(hp+v0*hr)*t;C[0]=(lrp+s0*lr)*(1-t)+(hrp+s0*hr)*t;
  const double rr=(4*H*Hr+2*R*Hr*Hr)*root+(H*H+2*R*H*Hr)*Kr/root-R*H*H*Kr*Kr/(4*root*root*root);
  const double expected_gradient=gradient+v0*radial,expected_cross=cross+v0*rr+s0*radial;
  const auto result=mapped_limb_mixed_density(q,R,H,Hr,P,RP,lr*(1-t)+hr*t,A,C,v,s,(1-x)*(1+x));
  if(std::abs(result.flux-value)>1e-10 || std::abs(result.radial_flux-radial)>1e-10 || std::abs(result.gradient[0]-expected_gradient)>1e-10 || std::abs(result.radial_gradient[0]-expected_cross)>1e-9){std::cerr<<x<<" values "<<result.flux-value<<" "<<result.radial_flux-radial<<" "<<result.gradient[0]-expected_gradient<<" "<<result.radial_gradient[0]-expected_cross<<"\n";throw std::runtime_error("transported limb mixed derivative");}
  ++points;
 }
 std::cout<<"PASS "<<points<<" moving-boundary mixed jets against exact polynomial fold\n";
}

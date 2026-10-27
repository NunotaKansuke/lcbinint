#include "lcbinint/magnification/algebraic_boundary/triple/root_pair.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/angular_topology.hpp"
#include <iostream>
#include "lcbinint/magnification/algebraic_boundary/nested_fejer2.hpp"
using namespace lcbinint::algebraic_boundary::triple;
namespace rp= root_pair_detail;
int main(){
 // The pair Jacobian is regular at a double root, including degree six.
 const std::array<double,7> fold{0,0,2,.3,1,.1,.4};
 const auto f=rp::equations(fold,0,0);
 if(f.e!=0 || f.o!=0 || f.em*f.ov-f.ev*f.om!=-8)throw std::runtime_error("fold Jacobian");
 for(double v:{1e-4,1e-12,1e-20,1e-28}) {
  std::array<rp::Jet<8>,7> p{};
  p[0]=-v;p[1]=-.2*v;p[2]=1;p[3]=.2;p[4]=-.1*v;p[6]=.1;
  p[0].dp[0]=-1;p[1].dp[0]=-.2;p[4].dp[0]=-.1;
  rp::Pair<8> pair;
  if(!pair.initialize(p,-std::sqrt(v),std::sqrt(v)) || std::abs(pair.square.dp[0]-1)>1e-12 || std::abs(pair.mean.dp[0])>1e-12)throw std::runtime_error("vanishing separation tangent");
 }
 int checked=0,moving_checked=0,screened=0;

 double Geometry::*members[]={&Geometry::X,&Geometry::Y,&Geometry::rho,&Geometry::m1,&Geometry::m2,&Geometry::a,&Geometry::bx,&Geometry::by};
 for(Geometry p:{Geometry{.12,.08,.25,.08,.02,1.25,-.6,.4},Geometry{.1,.15,.15,.2,.08,1.,-.4,.6},Geometry{.1258549239677236,.06379371886791989,.3,.2,.08,1.,-.4,.6}})
 for(double R:{.3,.7,.85,.9,.95,1.,1.03,1.1,1.2,1.5}) {
  auto topology=angular_topology(R,p);if(!topology.completed || topology.roots.empty())continue;
  std::array<std::optional<rp::Coefficients<8>>,2> caches;
  const rp::RadialCache<8> radial(p);
  for(auto arc:topology.arcs)for(bool local_center:{false,true})for(bool moving:{false,true}) {
   std::array<double,8> velocity{};if(moving)for(int k=0;k<8;++k)velocity[k]=.02*(k+1);
   rp::Arc<8> a;if(!a.initialize(R,p,arc.lo,arc.hi,velocity,&caches[moving],local_center,moving?&radial:nullptr))continue;
   if(a.short_level()<6) {
    std::array<double,9> low{},high{};
    for(int ell:{a.short_level(),8}) {
     auto& sum=ell==8?high:low;const auto& rule=lcbinint::algebraic_boundary::fejer_rule(ell);
     for(std::size_t j=0;j<rule.x.size();++j){rp::Jet<8> value;if(!a.density(rule.x[j],value))throw std::runtime_error("screen quadrature");sum[0]+=rule.norm_w[j]*value.value;for(int k=0;k<8;++k)sum[k+1]+=rule.norm_w[j]*value.dp[k];}
    }
    for(int k=0;k<9;++k)if(std::abs(low[k]-high[k])>2e-5*(1+std::abs(high[k])))throw std::runtime_error("Cauchy screened quadrature");
    ++screened;
   }
   for(double x:{-.99,-.8,0.,.8,.99}) {
    rp::Jet<8> result;if(!a.density(x,result))throw std::runtime_error("density positivity");
    const double half=std::sqrt(a.pair.square.value),t=a.pair.mean.value+half*x;
    const double theta=2*std::atan(t)+a.center;
    const double expected=R*std::sqrt(evaluate(R,theta,p).phi)*2*half/(1+t*t)/std::sqrt(1-x*x);
    if(std::abs(result.value-expected)>1e-8*(1+std::abs(expected)))throw std::runtime_error("deflated physical value");
    const double tl=a.pair.mean.value-half,th=a.pair.mean.value+half;
    const double orient=a.center;
    const auto jl=evaluate<true>(R,2*std::atan(tl)+orient,p),jh=evaluate<true>(R,2*std::atan(th)+orient,p);
    const auto jq=evaluate<true>(R,theta,p);
    for(int k=0;k<8;++k) {
     const double dl=-(jl.dp[k]+jl.dR*velocity[k])/jl.dtheta*(1+tl*tl)/2;
     const double dh=-(jh.dp[k]+jh.dR*velocity[k])/jh.dtheta*(1+th*th)/2;
     const double dt=((1-x)*dl+(1+x)*dh)/2,ds=(dh-dl)/2;
     const double physical=expected*(velocity[k]/R+ds/half-2*t*dt/(1+t*t)
       +(jq.dp[k]+jq.dR*velocity[k]+jq.dtheta*2*dt/(1+t*t))/(2*jq.phi));
     if(std::abs(physical-result.dp[k])>2e-5*(1+std::abs(physical))){std::cerr<<"physical gradient R="<<R<<" x="<<x<<" p="<<k<<" "<<result.dp[k]<<" vs "<<physical<<"\n";throw std::runtime_error("physical density derivative");}
    }
    for(int k=0;k<8;++k)for(double h:{1e-6,3e-7}) {
     auto plus=p,minus=p;plus.*members[k]+=h;minus.*members[k]-=h;
     rp::Arc<0> ap,am;rp::Jet<0> vp,vm;
     if(!ap.initialize(R+h*velocity[k],plus,arc.lo,arc.hi,{},nullptr,local_center) || !am.initialize(R-h*velocity[k],minus,arc.lo,arc.hi,{},nullptr,local_center) || !ap.density(x,vp) || !am.density(x,vm)){std::cerr<<"FD correction R="<<R<<" width="<<arc.hi-arc.lo<<" k="<<k<<" h="<<h<<" moving="<<moving<<"\n";throw std::runtime_error("FD pair correction");}
     const double fd=(vp.value-vm.value)/(2*h);
     if(std::abs(fd-result.dp[k])>2e-5*(1+std::abs(fd))){std::cerr<<R<<" x="<<x<<" p="<<k<<" "<<result.dp[k]<<" vs "<<fd<<"\n";throw std::runtime_error("pair parameter FD");}
    }
    ++checked;if(moving)++moving_checked;
   }
  }
 }
 if(checked<50 || moving_checked<25)throw std::runtime_error("pair coverage");
 std::cout<<"PASS "<<screened<<" Cauchy screens; "<<checked<<" deflated densities, "<<moving_checked<<" moving; 8 parameters, 2 FD steps\n";
}

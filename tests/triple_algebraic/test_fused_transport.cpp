#include "lcbinint/magnification/algebraic_boundary/triple/fused_integrator.hpp"
#include <iostream>
#include <random>
using namespace lcbinint::algebraic_boundary::triple;
namespace fused=fused_detail;
using lcbinint::algebraic_boundary::fejer_rule;
int main() {
 for(int level:{1,2,3,4,5})for(int mode:{0,1,2}) {
  const auto& x=fejer_rule(level).x;const auto& q=fused::radial_rule(level,mode);long double moment=mode==0?2.L:mode==1?acosl(-1.L)/2:4.L/3;
  for(int power=0;power<2*int(x.size());++power) {
   if(power && power%2==0)moment*=mode==0?(long double)(power-1)/(power+1):mode==1?(long double)(power-1)/(power+2):(long double)(power-1)/(power+3);
   long double actual=0;for(std::size_t j=0;j<x.size();++j){actual+=q.value[j]*powl(x[j],power);if(power)actual+=q.derivative[j]*power*powl(x[j],power-1);}
   if(fabsl(actual-(power%2?0:moment))>5e-11L)throw std::runtime_error("fused radial polynomial moment");
  }
 }
 std::mt19937 gen(20261027);std::uniform_real_distribution<double> random(-1,1);int fields=0;
 for(int k=0;k<80;++k) {
  Geometry p{.1+.2*random(gen),.1+.2*random(gen),.1+.2*std::abs(random(gen)),.2,.08,1.,-.4,.6};const double R=.3+1.1*std::abs(random(gen)),theta=3*random(gen);
  fused::Ray<double> ray(R,p);const auto q=ray.sample(theta);const auto ref=evaluate<true>(R,theta,p);
  const auto close=[](double a,double b){if(std::abs(a-b)>2e-5*(1+std::abs(b)))throw std::runtime_error("fused field independent spatial FD");};
  close(q.phi,ref.phi);close(q.r,ref.dR);close(q.t,ref.dtheta);
  for(double h:{1e-5,3e-6}) {
   const auto rp=evaluate<true>(R+h,theta,p),rm=evaluate<true>(R-h,theta,p),tp=evaluate<true>(R,theta+h,p),tm=evaluate<true>(R,theta-h,p);
   close(q.rr,(rp.dR-rm.dR)/(2*h));close(q.rt,(rp.dtheta-rm.dtheta)/(2*h));close(q.tt,(tp.dtheta-tm.dtheta)/(2*h));
   for(int j=0;j<8;++j){close(q.p[j],ref.dp[j]);close(q.rp[j],(rp.dp[j]-rm.dp[j])/(2*h));close(q.tp[j],(tp.dp[j]-tm.dp[j])/(2*h));}
  }
  ++fields;
 }
 Geometry p{.1,.15,.15,.2,.08,1.,-.4,.6};int caches=0;
 for(double R:{.4,.7,.9,1.,1.1,1.3}) {
  const auto topology=angular_topology(R,p);if(!topology.completed||topology.arcs.empty())continue;
  std::array<double,8> v{},s{};for(int j=0;j<8;++j){v[j]=.01*(j+1);s[j]=.002*(j+1);}
  fused::Radius radius(R,p,v,s);if(!radius.initialize(topology))continue;
  fused::Result a,b,c;if(!radius.evaluate(2,false,a)||!radius.evaluate(3,false,b))throw std::runtime_error("fused cached field");
  const auto count=radius.evaluations;if(!radius.evaluate(2,false,c)||radius.evaluations!=count||a.f0!=c.f0||a.fh!=c.fh)throw std::runtime_error("fused nested sample reuse");
  ++caches;
 }
 if(caches<2)throw std::runtime_error("fused radius coverage");
 int pairs=0;
 for(int cached:{0,1})for(double R:{.4,.7,.9,1.,1.1,1.3}) {
  const auto topology=angular_topology(R,p);if(!topology.completed)continue;
  std::array<double,8> v{},slope{};for(int j=0;j<8;++j){v[j]=.01*(j+1);slope[j]=.002*(j+1);}
  root_pair_detail::PreparedFrames frames;auto* cache=cached?&frames:nullptr;
  for(const auto& arc:topology.arcs){fused::MixedPair base;if(!base.initialize(R,p,arc.lo,arc.hi,v,slope,cache))continue;
   const double h=1e-6;fused::MixedPair plus,minus;auto vp=v,vm=v;for(int j=0;j<8;++j){vp[j]+=h*slope[j];vm[j]-=h*slope[j];}
   if(!plus.initialize(R+h,p,arc.lo,arc.hi,vp,slope,cache)||!minus.initialize(R-h,p,arc.lo,arc.hi,vm,slope,cache))continue;
   for(double x:{-.7,0.,.7}){fused::Mixed q,a,b;if(!base.density(x,q)||!plus.density(x,a)||!minus.density(x,b))throw std::runtime_error("fused common pair density");
    const auto close=[](double a,double b){if(std::abs(a-b)>3e-5*(1+std::abs(b)))throw std::runtime_error("fused common pair mixed FD");};
    close(q.r,(a.value-b.value)/(2*h));for(int j=0;j<8;++j)close(q.rp[j],(a.p[j]-b.p[j])/(2*h));
    for(int j=0;j<8;++j){Geometry pp=p,pm=p;auto perturb=[&](Geometry& g,double d){switch(j){case 0:g.X+=d;break;case 1:g.Y+=d;break;case 2:g.rho+=d;break;case 3:g.m1+=d;break;case 4:g.m2+=d;break;case 5:g.a+=d;break;case 6:g.bx+=d;break;case 7:g.by+=d;break;}};perturb(pp,h);perturb(pm,-h);fused::MixedPair ap,bp;if(!ap.initialize(R+h*v[j],pp,arc.lo,arc.hi,v,slope)||!bp.initialize(R-h*v[j],pm,arc.lo,arc.hi,v,slope))throw std::runtime_error("fused pair parameter FD initialization");fused::Mixed aq,bq;if(!ap.density(x,aq)||!bp.density(x,bq))throw std::runtime_error("fused pair parameter FD density");close(q.p[j],(aq.value-bq.value)/(2*h));}
   }++pairs;
  }
 }
 if(pairs<4)throw std::runtime_error("fused common pair coverage");

 // Independent analytic point-lens central-disc integral. The LD source
 // circle maps to a known image annulus; compare its area-normalized rho
 // derivative, including the moving radial Jacobian, to an analytic source-
 // coordinate integral. This exercises the complete coupled accumulation.
 {
  const double rho=.2,S=std::sqrt(rho*rho+4),lo=(S-rho)*.5,hi=(S+rho)*.5;
  Geometry geometry{0,0,rho,0,0,100,-100,100};
  AngularTopology circle;circle.completed=true;circle.arcs.push_back({0,2*std::acos(-1.)});
  RadialPanel<true> panel(geometry,lo,hi,true,true,AngularOptions{},nullptr,0,1e-7,false,false,1);
  panel.set_certified_band(circle,lo,hi);std::array<double,8> base{},slope{};base[2]=(rho/S-1)*.5;slope[2]=1/(hi-lo);
  panel.set_radial_motion(lo,base,slope);
  if(!panel.refine(5,5))throw std::runtime_error("fused point-lens annulus");
  const auto result=normalize(panel.result(),geometry,1);
  long double integral=0,derivative=0;const long double pi=acosl(-1.L);const int n=256;
  for(int k=1;k<n;++k){const long double angle=pi*k/n,x=cosl(angle),y=rho*x,z=sqrtl(y*y+4),w=pi/n*sinl(angle)*sinl(angle);integral+=w*(y*y+2)/z;derivative+=w*x*y*(y*y+6)/(z*z*z);}
  const double mu=double(3*integral/(2*rho)),drho=double(3*derivative/(2*rho)-3*integral/(2*rho*rho));
  if(std::abs(result.f0-mu)>2e-6*mu || std::abs(result.df0[2]-drho)>2e-5*(1+std::abs(drho))){std::cerr<<result.f0<<" vs "<<mu<<" derivative "<<result.df0[2]<<" vs "<<drho<<"\n";throw std::runtime_error("fused analytic point-lens gradient");}
 }
 std::cout<<"PASS confluent polynomial moments, "<<fields<<" physical mixed jets, "<<caches<<" nested caches, "<<pairs<<" common pairs\n";
}

#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/gradient_replay.hpp"
#include <random>
#include <iostream>
using namespace lcbinint::algebraic_boundary::triple;
int main(){
 for(int level:{3,4,5})for(bool weighted:{false,true}){
  const auto& q=shape_gradient_detail::hermite_rule(level,weighted);long double moment=weighted?acosl(-1.L)/2:2;double worst=0;
  for(int power=0;power<2*int(q.x.size());++power){
   long double sum=0;for(int i=0;i<int(q.x.size());++i){const long double x=q.x[i];sum+=q.value[i]*powl(x,power);if(power)sum+=q.derivative[i]*power*powl(x,power-1);}
   if(power && power%2==0)moment*=weighted?(long double)(power-1)/(power+2):(long double)(power-1)/(power+1);
   const long double exact=power%2?0:moment;worst=std::max(worst,double(fabsl(sum-exact)));
   if(fabsl(sum-exact)>1e-11L)throw std::runtime_error("Hermite moment");
   for(const auto& w:q.witnesses) {
    long double prediction=0;
    for(int i=0;i<int(q.x.size());++i){const long double x=q.x[i];prediction+=w.value[i]*powl(x,power);if(power)prediction+=w.derivative[i]*power*powl(x,power-1);}
    if(fabsl(prediction-powl((long double)w.x,power))>1e-11L)throw std::runtime_error("Hermite interpolation witness");
   }
  }
  std::cout<<level<<" "<<weighted<<" "<<worst<<"\n";
 }
 std::mt19937 random(1783);std::uniform_real_distribution<double> u(-1,1);
 for(int i=0;i<200;++i){Geometry p{u(random),u(random),.1+.2*std::abs(u(random)),.2,.08,1.,-.4,.6};double R=.3+1.4*std::abs(u(random)),theta=3*u(random);const auto a=shape_gradient_detail::polar_mixed(R,theta,p);
  for(double h:{1e-5,3e-6}){
   const auto rp=evaluate<true>(R+h,theta,p),rm=evaluate<true>(R-h,theta,p),tp=evaluate<true>(R,theta+h,p),tm=evaluate<true>(R,theta-h,p);
   auto close=[](double a,double b){if(std::abs(a-b)>2e-5*(1+std::abs(b)))throw std::runtime_error("mixed jet finite difference");};
   close(a.rr,(rp.dR-rm.dR)/(2*h));close(a.rt,(tp.dR-tm.dR)/(2*h));close(a.tt,(tp.dtheta-tm.dtheta)/(2*h));
   for(int j=0;j<8;++j){close(a.rp[j],(rp.dp[j]-rm.dp[j])/(2*h));close(a.tp[j],(tp.dp[j]-tm.dp[j])/(2*h));}
  }
 }
 std::cout<<"PASS mixed spatial/parameter jets\n";
 int checked=0,full_checked=0;
 for(Geometry p:{Geometry{.1,.15,.3,.2,.08,1.,-.4,.6},Geometry{.12,.08,.25,.08,.02,1.25,-.6,.4}})
 for(double R:{.3,.5,.7,.9,1.,1.03,1.1,1.3,1.5}) {
  std::array<double,8> v{},s{};for(int k=0;k<8;++k){v[k]=.02*(k+1);s[k]=.005*(k+1);}
  const auto topo=angular_topology(R,p);if(!topo.completed || topo.arcs.empty())continue;
  // Isolate transport calculus from moving complex-chart quadrature errors.
  // Charted caustic cases remain covered by the full-integrator FD campaign.
  ShapeGradientResult probe;shape_gradient_detail::Ray ray(p,R);bool charted=false;
  for(const auto& arc:topo.arcs)if(shape_gradient_detail::complex_arc_geometry(ray,R,p,arc.lo,arc.hi,!topo.roots.empty(),probe)){charted=true;break;}
  if(charted)continue;
  ShapeGradientResult work;shape_gradient_detail::HermiteRadius a;
  if(!shape_gradient_detail::hermite_radius(R,p,topo,v,s,a,work,true,8))continue;
  for(double h:{3e-6,1e-6}) {
   const auto tp=angular_topology(R+h,p),tm=angular_topology(R-h,p);auto vp=v,vm=v;
   for(int k=0;k<8;++k){vp[k]+=h*s[k];vm[k]-=h*s[k];}
   shape_gradient_detail::HermiteRadius plus,minus;
   if(!shape_gradient_detail::hermite_radius(R+h,p,tp,vp,s,plus,work,true,8) || !shape_gradient_detail::hermite_radius(R-h,p,tm,vm,s,minus,work,true,8))throw std::runtime_error("transport FD sample");
   int where=0;auto close=[&](double a,double b){const int index=where++;if(std::abs(a-b)>2e-5*(1+std::abs(b))){std::cerr<<"m1="<<p.m1<<" R="<<R<<" h="<<h<<" check="<<index<<" "<<a<<" vs "<<b<<"\n";throw std::runtime_error("transport radial derivative");}};
   close(a.radial_flux,(plus.flux-minus.flux)/(2*h));close(a.radial_limb_flux,(plus.limb_flux-minus.limb_flux)/(2*h));
   for(int k=0;k<8;++k){close(a.radial_gradient[k],(plus.gradient[k]-minus.gradient[k])/(2*h));close(a.radial_limb_gradient[k],(plus.limb_gradient[k]-minus.limb_gradient[k])/(2*h));}
  }
  ++checked;if(topo.roots.empty())++full_checked;
 }
 if(checked<2)throw std::runtime_error("transport test coverage");
 std::cout<<"PASS transport at "<<checked<<" radii, full="<<full_checked<<"\n";

 // Pointwise mixed transport is checked against independently evaluated
 // geometry before integration; no quadrature or chart motion enters this FD.
 int density_checks=0;
 for(int i=0;i<100;++i) {
  Geometry p{.12,.08,.4,.08,.02,1.25,-.6,.4};const double R=1+.1*u(random),theta=3*u(random);
  const auto q=shape_gradient_detail::polar_mixed(R,theta,p);if(q.phi<.05)continue;
  std::array<double,8> v{},s{};for(int k=0;k<8;++k){v[k]=.02*(k+1);s[k]=.005*(k+1);}
  const double B=.07,Bt=-.03;const auto actual=shape_gradient_detail::limb_mixed_density(q,R,B,Bt,v,s);
  for(double h:{1e-5,3e-6}) {
   const auto jp=evaluate<true>(R+h,theta+h*B,p),jm=evaluate<true>(R-h,theta-h*B,p);
   const double rp=std::sqrt(jp.phi),rm=std::sqrt(jm.phi);
   auto close=[](double a,double b){if(std::abs(a-b)>2e-5*(1+std::abs(b)))throw std::runtime_error("pointwise mixed transport");};
   close(actual.radial_flux,((R+h)*rp*(1+h*Bt)-(R-h)*rm*(1-h*Bt))/(2*h));
   for(int k=0;k<8;++k) {
    const double vp=v[k]+h*s[k],vm=v[k]-h*s[k];
    const double gp=(vp*rp+(R+h)*(jp.dp[k]+jp.dR*vp)/(2*rp))*(1+h*Bt);
    const double gm=(vm*rm+(R-h)*(jm.dp[k]+jm.dR*vm)/(2*rm))*(1-h*Bt);
    close(actual.radial_gradient[k],(gp-gm)/(2*h));
   }
  }
  ++density_checks;
 }
 if(density_checks<10)throw std::runtime_error("pointwise transport coverage");
 // A seam crossing cyclically permutes image arcs. Equivalent angular
 // labels must preserve the physical interpolation/reuse decision.
 int cyclic_panels=0;
 for(double R:{.3,.5,.7,.85,.9,.95,1.,1.03,1.1,1.3,1.5}) {
  Geometry geometry{.1,.15,.15,.2,.08,1.,-.4,.6};
  RadialPanel<false> candidate(geometry,R,R+.001,false,false,AngularOptions{},nullptr,0,1e-4,true);
  if(!candidate.refine(3,3) || !shape_gradient_detail::hermite_boundary_resolved(candidate,geometry,false))continue;
  const auto& radii=candidate.retained_radii();bool multiple=false;
  for(std::size_t k=0;k<7;++k) {
   const auto& seed=radii[int(k+1)<<(8-3)];if(!seed)throw std::runtime_error("cyclic test retained radius");
   auto& arcs=seed->topology.arcs;
   if(arcs.size()>1){multiple=true;std::rotate(arcs.begin(),arcs.begin()+k%arcs.size(),arcs.end());}
   for(std::size_t j=0;j<arcs.size();++j){const double shift=2*std::acos(-1.)*(int((k+j)%3)-1);arcs[j].lo+=shift;arcs[j].hi+=shift;}
  }
  if(!multiple)continue;
  if(!shape_gradient_detail::hermite_boundary_resolved(candidate,geometry,false))throw std::runtime_error("cyclic image-arc label invariance");
  ++cyclic_panels;
 }
 if(cyclic_panels<2)throw std::runtime_error("cyclic arc coverage");
 std::cout<<"PASS cyclic arc invariance at "<<cyclic_panels<<" panels\n";

 // A proved full angular circle integrates exactly without radius samples.
 Geometry annulus{.12,.08,.25,0,0,1.25,-.6,.4};
 AngularTopology circle;circle.completed=true;circle.arcs.push_back({0,2*std::acos(-1.)});
 auto panel=std::make_unique<RadialPanel<false>>(annulus,.99,1.01,true,true,AngularOptions{},nullptr,0,1e-4,true);
 panel->set_certified_band(circle,.99,1.01);
 if(!panel->refine(3,3) || panel->radial_evaluations!=0 || panel->value_evaluations()!=0)throw std::runtime_error("exact annulus avoids samples");
 const double exact=std::acos(-1.)*(1.01-.99)*(1.01+.99);
 if(std::abs(panel->result().f0-exact)>panel->result().angular_error)throw std::runtime_error("exact annulus flux bound");
 ValueIntegralTrace trace;trace.panels.emplace_back(std::move(panel));
 const auto replay=replay_shape_gradient(annulus,0,trace,RadialEventPlan{});
 if(!replay.completed || replay.evaluations || replay.uniform_flux!=exact)throw std::runtime_error("exact annulus replay");
 for(double g:replay.uniform)if(g!=0)throw std::runtime_error("fixed annulus boundary derivative");

}

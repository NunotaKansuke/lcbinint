#include "lcbinint/magnification/algebraic_boundary/triple/integrator.hpp"
#include "lcbinint/magnification/algebraic_boundary/triple/gradient_replay.hpp"
#include <iostream>
#include "lcbinint/magnification/algebraic_boundary/triple/pair_moments.hpp"
using namespace lcbinint::algebraic_boundary::triple;
namespace rp=root_pair_detail;namespace sg=shape_gradient_detail;
int main(){int series_count=0,models=0,points=0,moved=0,declined=0;
 for(double width:{1e-3,1e-6,1e-10}) {
  rp::Arc<8> model;model.pair.mean=rp::Jet<8>(.1);model.pair.mean.dp[0]=1;
  model.pair.square=rp::Jet<8>(-width*width);model.pair.square.dp[1]=1;
  model.pair.quotient[0]=rp::Jet<8>(1);model.pair.quotient[0].dp[2]=1;
  model.rho=rp::Jet<8>(1);model.rho.dp[3]=1;model.inverse_rho=1;model.half_width=width;
  for(auto& pole:model.poles){pole[0]=rp::Jet<8>(1);pole[2]=rp::Jet<8>(.2);}
  rp::CofactorSeries<8> series;if(!rp::prepare_series(model,1e-6,series))throw std::runtime_error("synthetic series coverage");
  rp::Jet<8> analytic,numerical;const double ta=.1-.8*series.radius,tb=.1+.9*series.radius;
  if(!rp::integrate_series(model,series,ta,tb,analytic))throw std::runtime_error("synthetic series integral");
  const double ua=std::asinh((ta-.1)/width),ub=std::asinh((tb-.1)/width);const auto& rule=sg::gauss_rule(32);
  const int cells=int(std::ceil(ub-ua));
  for(int cell=0;cell<cells;++cell)for(std::size_t k=0;k<rule.x.size();++k) {
   const double u=ua+(ub-ua)*(cell+(rule.x[k]+1)*.5)/cells,delta=width*std::sinh(u),w=rule.w[k]*(ub-ua)/(2*cells);rp::Jet<8> point;
   if(!model.critical_density(delta,delta*delta+width*width,point))throw std::runtime_error("synthetic density");
   numerical.value+=w*point.value;for(int i=0;i<8;++i)numerical.dp[i]+=w*point.dp[i];
  }
  if(std::abs(analytic.value-numerical.value)>2e-6*std::abs(numerical.value))throw std::runtime_error("synthetic series value");
  for(int i=0;i<8;++i)if(std::abs(analytic.dp[i]-numerical.dp[i])>2e-5*(1+std::abs(numerical.dp[i])))throw std::runtime_error("synthetic series derivative");
  ++series_count;
 }
 for(Geometry p:{Geometry{.12,.08,.25,.08,.02,1.25,-.6,.4},Geometry{.123,.078,.255,.08,.02,1.25,-.6,.4},Geometry{.1,.15,.15,.2,.08,1.,-.4,.6},Geometry{.1258549239677236,.06379371886791989,.3,.2,.08,1.,-.4,.6}})
 for(double R:{.3,.5,.7,.85,.9,.95,1.,1.03,1.1,1.3,1.5}) {
  const auto topology=angular_topology(R,p);if(!topology.completed || topology.arcs.empty())continue;
  sg::Ray ray(p,R);const rp::RadialCache<8> radial(p);
  for(bool moving:{false,true}) {
   std::array<double,8> velocity{};if(moving)for(int k=0;k<8;++k)velocity[k]=.02*(k+1);
   std::optional<rp::Coefficients<8>> cache;ShapeGradientResult work;
   for(const auto& arc:topology.arcs) {
    auto geometry=sg::complex_arc_geometry(ray,R,p,arc.lo,arc.hi,!topology.roots.empty(),work,16,12,false);
    if(!geometry)continue;
    for(auto minimum:geometry->minima) {
     rp::Arc<8> base;if(!base.configure(R,p,minimum.center,velocity,&cache,&radial))throw std::runtime_error("critical basis");
     auto model=rp::complex_model(base,R,p,minimum.center,minimum.scale,minimum.phi,minimum.error,*cache,work);
     if(!model){++declined;continue;}++models;
     rp::CofactorSeries<8> series;
     if(rp::prepare_series(*model,1e-6,series)) {
      ++series_count;
      rp::Jet<8> analytic,numerical;
      const double ta=model->pair.mean.value-series.radius*.8,tb=model->pair.mean.value+series.radius*.9;
      if(!rp::integrate_series(*model,series,ta,tb,analytic))throw std::runtime_error("series integral");
      const double ua=std::asinh((ta-model->pair.mean.value)/model->half_width),ub=std::asinh((tb-model->pair.mean.value)/model->half_width);
      const auto& rule=sg::gauss_rule(32);const int cells=std::max(1,int(std::ceil(ub-ua)));
      for(int cell=0;cell<cells;++cell)for(std::size_t k=0;k<rule.x.size();++k) {
       const double u=ua+(ub-ua)*(cell+(rule.x[k]+1)*.5)/cells;
       const double delta=model->half_width*std::sinh(u),w=rule.w[k]*(ub-ua)/(2*cells);rp::Jet<8> point;
       if(!model->critical_density(delta,delta*delta-model->pair.square.value,point))throw std::runtime_error("series reference");
       numerical.value+=w*point.value;for(int i=0;i<8;++i)numerical.dp[i]+=w*point.dp[i];
      }
      if(std::abs(analytic.value-numerical.value)>2e-6*std::abs(numerical.value))throw std::runtime_error("series value mismatch");
      for(int i=0;i<8;++i)if(std::abs(analytic.dp[i]-numerical.dp[i])>2e-5*(1+std::abs(numerical.dp[i]))) {std::cerr<<analytic.dp[i]<<" vs "<<numerical.dp[i]<<" p="<<i<<"\n";throw std::runtime_error("series gradient mismatch");}
     }
     for(double shift:{0.,.7,-1.2}) {
      rp::Arc<8> active;
      if(!sg::move_pair<8>(*model,model->center+shift,R,p,velocity,&cache,&radial,active))throw std::runtime_error("pair Mobius transform");
      for(double u:{-.7,0.,.7}) {
       const double delta=model->half_width*std::sinh(u),t=model->pair.mean.value+delta;
       const __float128 theta=__float128(model->center)+2*atanq(__float128(t));
       const double target=double(tanq((theta-__float128(active.center))/2));
       const double d=target-active.pair.mean.value,factor=d*d-active.pair.square.value;
       rp::Jet<8> result;if(!active.critical_density(d,factor,result))throw std::runtime_error("critical density");
       const auto q=local_precision::evaluate<true>(__float128(R),theta,p);
       if(!(q.phi>0))throw std::runtime_error("critical physical support");
       const __float128 root=sqrtq(q.phi),jac=2*sqrtq(__float128(factor))/(1+__float128(target)*__float128(target));
       const double value=double(__float128(R)*root*jac);
       if(std::abs(result.value-value)>2e-6*std::abs(value)){
        std::cerr<<"value R="<<R<<" shift="<<shift<<" u="<<u<<" "<<result.value<<" vs "<<value<<"\n";throw std::runtime_error("critical physical value");}
       for(int k=0;k<8;++k){const double derivative=double((__float128(velocity[k])*root+__float128(R)*(q.dp[k]+q.dR*__float128(velocity[k]))/(2*root))*jac);
        if(std::abs(result.dp[k]-derivative)>2e-5*(1+std::abs(derivative))){std::cerr<<"gradient R="<<R<<" p="<<k<<" shift="<<shift<<" u="<<u<<" "<<result.dp[k]<<" vs "<<derivative<<"\n";throw std::runtime_error("critical physical parameter derivative");}}
       ++points;if(shift!=0)++moved;
      }
     }
    }
   }
  }
 }
 if(models<5 || points<45 || moved<30)throw std::runtime_error("critical pair coverage");
 std::cout<<"series="<<series_count<<" PASS "<<models<<" critical models, "<<points<<" physical point jets, "<<moved<<" moved-basis jets; declined="<<declined<<"\n";
}

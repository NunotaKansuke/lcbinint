#pragma once
#include "shape_transport.hpp"
#include "radial_coefficient_frame.hpp"
namespace lcbinint::algebraic_boundary::triple::stream_detail {
using Frame=root_pair_detail::PreparedFrame;
inline std::array<std::size_t,12> trials{};
inline bool radius(double R,const Geometry& p,const AngularTopology& topology,bool limb,const std::array<double,8>& motion,
 shape_gradient_detail::RadiusResult& result,shape_gradient_detail::RadiusAnchor& anchor,double& error,
 std::vector<std::shared_ptr<Frame>>& frames,ShapeGradientResult& work,double gate){
 ++trials[0];if(topology.roots.empty()||topology.arcs.size()>3)return false;
 shape_gradient_detail::RadiusResult next;shape_gradient_detail::RadiusAnchor next_anchor;next_anchor.radius=R;
 shape_gradient_detail::Ray ray(p,R);std::array<AngularArc,3> resolved{};std::size_t count=0;double total_error=0;
 for(const auto& arc:topology.arcs){
  const double center=(arc.lo+arc.hi)*.5;std::shared_ptr<Frame> frame;
  for(auto& f:frames)if(std::abs(center-f->basis)<.3){frame=f;break;}
  if(!frame){frame=std::make_shared<Frame>(p,center,R);frames.push_back(frame);}
  root_pair_detail::Arc<8> model;double lr,hr;if(!frame->prepare(R,motion,arc.lo,arc.hi,model,lr,hr)){++trials[1];return false;}
  const auto& pair=model.pair;const double m=pair.mean.value,v=pair.square.value,h=model.half_width;
  const double lo=frame->basis+2*std::atan(m-h),hi=frame->basis+2*std::atan(m+h),width=hi-lo;
  double roundoff=0;auto jl=ray.sample(lo,roundoff);++work.evaluations;
  if(!(jl.dtheta>0)||std::abs(jl.phi/jl.dtheta)>width*1e-8||roundoff>std::abs(jl.dtheta)*width*1e-7){++trials[2];return false;}
  auto jh=ray.sample(hi,roundoff);++work.evaluations;
  if(!(jh.dtheta<0)||std::abs(jh.phi/jh.dtheta)>width*1e-8||roundoff>std::abs(jh.dtheta)*width*1e-7){++trials[3];return false;}
  // Coefficient cancellation must not silently corrupt a parameter RHS.
  for(int j=0;j<8;++j){const double a=2*(pair.mean.dp[j]-pair.square.dp[j]/(2*h))/(1+(m-h)*(m-h)),b=2*(pair.mean.dp[j]+pair.square.dp[j]/(2*h))/(1+(m+h)*(m+h));
   const double pa=-(jl.dp[j]+jl.dR*motion[j])/jl.dtheta,pb=-(jh.dp[j]+jh.dR*motion[j])/jh.dtheta;
   if(std::abs(a-pa)>1e-6*(1+std::abs(pa))||std::abs(b-pb)>1e-6*(1+std::abs(pb))){++trials[4];return false;}
  }
  for(std::size_t i=0;i<count;++i)for(int turn:{-1,0,1}){const double shift=turn*2*std::acos(-1.);if(std::max(lo,resolved[i].lo+shift)<std::min(hi,resolved[i].hi+shift))return false;}
  if(!(ray.sample<false>((lo+hi)*.5,roundoff).phi>0))return false;
  resolved[count++]={lo,hi};next_anchor.arcs[next_anchor.count++]={lo,hi,lr,hr};
  const double denom=(1+m*m-v)*(1+m*m-v)+4*v;
  next.f0+=R*width;
  for(int j=0;j<8;++j)next.uniform[j]+=motion[j]*width+R*(2/h)*((1+m*m+v)*pair.square.dp[j]-4*m*v*pair.mean.dp[j])/denom;
  if(!limb)continue;
  // The disk proof excludes unresolved interior singularities; these retain
  // the existing local critical-pair kernel instead of oversampling blindly.
  if(!model.disk_resolved(2)){++trials[5];return false;}
  const int maximum=7;std::array<root_pair_detail::Jet<8>,128> cache{};std::array<bool,128> ready{};double previous=0;bool done=false;
  for(int lev=2;lev<=maximum;++lev){const auto& rule=angular_rule_extended(lev);root_pair_detail::Jet<8> sum;const int n=1<<lev;
   for(int k=1;k<n;++k){const int index=k<<(maximum-lev);if(!ready[index]){if(!model.density(rule.x[k-1],cache[index]))return false;ready[index]=true;++work.evaluations;}
    sum=sum+root_pair_detail::Jet<8>(rule.norm_w[k-1])*cache[index];}
   if(lev>=3&&std::abs(sum.value-previous)<=gate*std::abs(sum.value)){next.fh+=sum.value;for(int j=0;j<8;++j)next.limb[j]+=sum.dp[j];total_error+=std::abs(sum.value-previous);done=true;break;}previous=sum.value;
  }
  if(!done)return false;
 }
 ++trials[11];result=next;anchor=next_anchor;error=total_error;return true;
}
}

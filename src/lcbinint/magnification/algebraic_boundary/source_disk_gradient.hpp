#pragma once
#include "fused_gradient.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/binary_caustic_enclosure.hpp"
#include <complex>
#include <array>
#include <vector>
#include <cmath>
#include <limits>
namespace lcbinint::algebraic_boundary::source_disk_detail {
using C=std::complex<long double>;
struct RefinedCover {double separation=0,q=0;int intervals=0;std::vector<magnification::detail::CausticEnclosureDisk> disks;};
inline RefinedCover& refined_cover() {static thread_local RefinedCover value;return value;}
inline void clear_epoch_cache() {refined_cover()=RefinedCover{};}

struct Sample {long double mu=0;std::array<long double,5> g{};int images=0;bool ok=false;};
// Differentiate the real lens-map Jacobian at each physical point image.
// Coordinates and derivatives are mapped to LensParams at this boundary.
inline Sample sample(const LensParams&p,const magnification::PointSourceMagnifier&point){
 Sample out;const long double m=1.L/(1.L+p.q),n=1-m,a=p.a;
 const auto images=point.binary_images(p.a,p.q,{p.xs,p.ys});
 if(images.size()!=3&&images.size()!=5)return out;
 for(auto image:images){
  C z(image.position.x+(p.q<1?p.a:0),image.position.y),b=std::conj(z),d=b-a;
  if(std::abs(b)==0||std::abs(d)==0)return out;
  const C K=m/(b*b)+n/(d*d);const long double D=1-std::norm(K);
  if(D==0||!std::isfinite(D))return out;
  out.mu+=1/std::fabs(D);
  std::array<long double,4> gradient{};
  for(int j=0;j<4;++j){
   const C rhs=j==0?C(1,0):j==1?C(0,1):j==2?1.L/b-1.L/d:n/(d*d);
   const C motion= C(((1-K.real())*rhs.real()-K.imag()*rhs.imag())/D,
                    (-K.imag()*rhs.real()+(1+K.real())*rhs.imag())/D);
   C dk=-2.L*m*std::conj(motion)/(b*b*b)-2.L*n*std::conj(motion)/(d*d*d);
   if(j==2)dk+=1.L/(b*b)-1.L/(d*d);
   if(j==3)dk+=2.L*n/(d*d*d);
   gradient[j]=(D>0?1.L:-1.L)*2*std::real(std::conj(K)*dk)/(D*D);
  }
  out.g[0]+=gradient[0];out.g[1]+=gradient[1];
  out.g[3]+=m*m*((p.barycentric?a*gradient[0]:0)-gradient[2]);
  out.g[4]+=gradient[3]+(p.barycentric?n*gradient[0]:0);
 }
 out.images=int(images.size());out.ok=std::isfinite(out.mu);
 for(auto g:out.g)out.ok=out.ok&&std::isfinite(g);return out;
}
struct Rule {std::vector<double>x,w;};
inline Rule rule(int n){Rule r;r.x.resize(n);r.w.resize(n);const long double pi=acosl(-1.L);
 for(int i=0;i<(n+1)/2;++i){long double z=cosl(pi*(i+.75L)/(n+.5L)),dp=0;
  for(int k=0;k<32;++k){long double p=1,old=0;for(int j=1;j<=n;++j){auto v=((2*j-1)*z*p-(j-1)*old)/j;old=p;p=v;}dp=n*(z*p-old)/(z*z-1);auto step=p/dp;z-=step;if(fabsl(step)<4e-19L)break;}
  const long double w=1/((1-z*z)*dp*dp);r.x[i]=double((1-z)/2);r.x[n-i-1]=double((1+z)/2);r.w[i]=r.w[n-i-1]=double(w);
 }return r;}
inline const Rule& disk_rule(int n){static const Rule a=rule(2),b=rule(4),c=rule(8),d=rule(16);return n==2?a:n==4?b:n==8?c:d;}
// With z=sqrt(1-r^2), the normalized source-disk measure is
// 2*z*(1-limb+limb*z)/(1-limb/3) dz dtheta/(2*pi).
// It is independent of rho. Differentiation therefore integrates grad(M)
// directly and never subtracts 2*A/rho from an almost equal area derivative.
// Clearance proves that differentiation under this smooth integral is valid.
// This is a quadrature of the original finite-source model, not a point-source
// truncation. Caustic clearance gates smoothness; only primal differences select
// order. No derivative error is estimated or used for refinement.
inline bool evaluate(const LensParams&p,double limb,const FusedGradientOptions&opts,
 const magnification::FiniteSourceMagnifier&finite,const magnification::PointSourceMagnifier&point,
 FusedGradientResult&result,int&image_count,bool repair_invalid=false){
 if(!p.barycentric||!(limb>=0&&limb<=1)||
    !std::isfinite(opts.mu_atol)||opts.mu_atol<0||
    !std::isfinite(opts.mu_rtol)||!(opts.mu_rtol>0))return false;
 // In thin, highly magnified images the normalized area derivative loses
 // digits like (mu/rho)^2. Switch coordinates before half the binary64
 // significand is exposed to that amplification. This depends on geometry
 // and arithmetic precision, not on a measured derivative error.
 const auto center_value=point.binary_mag0_cached(p.a,p.q,{p.xs,p.ys});
 const double half_precision=std::sqrt(std::numeric_limits<double>::epsilon());
 const bool cancellation_prone=p.rho*p.rho<=half_precision*
     center_value.magnification*center_value.magnification;
 const bool near_axis=std::fabs(p.ys)<=p.rho/32;
 if(!repair_invalid && !near_axis && !cancellation_prone)return false;
 double clearance=finite.binary_caustic_clearance_lower_bound(p.a,p.q,{p.xs,p.ys},0.25*p.rho);
 if(repair_invalid && !(clearance>p.rho)) {
  auto& refined=refined_cover();
  if(refined.separation!=p.a||refined.q!=p.q){
   refined.disks.clear();refined.intervals=0;refined.separation=p.a;refined.q=p.q;
  }
  // A coarse phase cover can contain a source that is safely inside a broad
  // caustic region even when it is far from the caustic curve itself. Refine
  // only on this failed-gradient retry. Each disk cover encloses the complete
  // continuous caustic, so a positive lower bound still certifies that the
  // entire source disk is smooth before differentiating under the integral.
  for(int intervals:{512,2048,8192}) {
   if(refined.intervals<intervals) {
    refined.disks=magnification::detail::build_binary_caustic_enclosure(
        p.a,p.q,8,intervals);
    refined.intervals=intervals;
   }
   // A cached finer cover already contains the requested coarser cover's
   // information. In particular, do not rescan the 8192-disk cover three
   // times on every later failed epoch.
   if(refined.intervals>intervals)continue;
   clearance=std::max(clearance,
       magnification::detail::binary_caustic_enclosure_clearance(
           refined.disks,{p.xs,p.ys}));
   if(clearance>p.rho || refined.intervals>=8192)break;
  }
 }
 // Separate smooth disks by the dimensionless radius/clearance. This is a
 // coordinate-selection gate, not a gradient accuracy or convergence test.
 if(!(p.rho<(repair_invalid?1.0:1.0/32)*clearance))return false;
 const auto center=sample(p,point);if(!center.ok)return false;image_count=center.images;
 long double previous=0;std::size_t work=0;
 for(int n:{2,4,8,16}){
  const auto&r=disk_rule(n);const int angles=4*n;const long double pi=acosl(-1.L);
  long double value=0;std::array<long double,5> grad{};
  for(int i=0;i<n;++i){long double z=r.x[i],rr=sqrtl(1-z*z);
   const long double weight=2*z*r.w[i]*(1-limb+limb*z)/((1-limb/3.L)*angles);
   for(int j=0;j<angles;++j){const long double theta=2*pi*j/angles,dx=rr*cosl(theta),dy=rr*sinl(theta);
    LensParams shifted=p;shifted.xs+=double(p.rho*dx);shifted.ys+=double(p.rho*dy);auto v=sample(shifted,point);++work;
    if(!v.ok||v.images!=center.images)return false;
    value+=weight*v.mu;for(int k=0;k<5;++k)grad[k]+=weight*v.g[k];
    // The constant centre gradient has zero angular moment. Subtract it before
    // accumulation to avoid cancellation of opposite source-plane directions.
    grad[2]+=weight*((v.g[0]-center.g[0])*dx+(v.g[1]-center.g[1])*dy);
   }
  }
  if(n>2&&fabsl(value-previous)<=std::max(opts.mu_atol,opts.mu_rtol*double(fabsl(value)))){
   auto candidate=result;
   candidate.mu=double(value);for(int k=0;k<5;++k){candidate.grad_mu[k]=double(grad[k]);candidate.gradient_valid[k]=std::isfinite(candidate.grad_mu[k]);if(!candidate.gradient_valid[k])return false;}
   result=std::move(candidate);
   result.value_error_estimate=double(fabsl(value-previous));
   result.frontier_discovery_completed=false;
   result.value_converged=true;result.gradient_precision_checked=false;result.support_certified=false;result.stop="Converged";result.node_evaluations+=work;return true;
  }previous=value;
 }return false;
}
}

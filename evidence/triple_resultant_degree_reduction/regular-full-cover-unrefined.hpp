#pragma once
// Geometry-only critical-curve enclosures adapted from the binary caustic
// tree. Source queries and boundary-branch work never cache a flux value.
// A refused enclosure/count/branch proof leaves the full event planner in
// charge. See docs/design/triple-algebraic-integration-plan.md.
#include "lcbinint/magnification/algebraic_boundary/triple/geometry.hpp"
#include "lcbinint/math/polynomial_roots.hpp"
#include "interval.hpp"
#include <vector>
#include <optional>
#include <algorithm>
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::algebraic_boundary::triple::regular_support_detail {
using namespace lcbinint::algebraic_boundary::triple;
constexpr double dm=4*std::numeric_limits<double>::epsilon();
constexpr double eps=1024*std::numeric_limits<double>::epsilon();
constexpr double pi=3.14159265358979323846;
// In the ordinary range the squared norm is well-scaled and needs only
// a hardware square root. Keep libm's scaled hypot at extreme magnitudes.
// The few-ulp roundoff stays inside this planner's existing eps margins.
template<class T> inline auto fast_abs(T x){return std::abs(x);}
inline double fast_abs(Complex z) {
 const double n=std::norm(z);
 if(n>=32*std::numeric_limits<double>::min() && std::isfinite(n))return std::sqrt(n);
 return std::abs(z);
}

struct Disk {Complex z,w;double r=0,h=0;};
using Poly=std::vector<Complex>;
inline Poly mul(const Poly&a,const Poly&b){Poly c(a.size()+b.size()-1);for(size_t i=0;i<a.size();++i)for(size_t j=0;j<b.size();++j)c[i+j]+=a[i]*b[j];return c;}
inline Poly critical(const Geometry&p,double phi){auto l=p.lenses();auto m=p.masses();Poly all{1};for(auto z:l)all=mul(all,{z*z,-2.*z,1.});Poly c=all;for(auto&v:c)v*=-std::polar(1.,phi);for(int j=0;j<3;++j){Poly a{1};for(int k=0;k<3;++k)if(k!=j)a=mul(a,{l[k]*l[k],-2.*l[k],1.});for(size_t k=0;k<a.size();++k)c[k]+=m[j]*a[k];}return c;}
inline std::pair<Complex,Complex> kappa(Complex z,const Geometry&p){Complex k{},d{};auto l=p.lenses();auto m=p.masses();for(int j=0;j<3;++j){auto inv=1./(z-l[j]);k+=m[j]*inv*inv;d-=2.*m[j]*inv*inv*inv;}return {k,d};}
inline std::vector<Disk> cover(const Geometry&p,double phi,double half,std::vector<Complex>&roots){
 const auto l=p.lenses();const auto m=p.masses();const auto phase=std::polar(1.,phi);
 if(roots.size()!=6){auto coefficients=critical(p,phi);for(auto c:coefficients)if(!std::isfinite(c.real()) || !std::isfinite(c.imag()))return {};lcbinint::math::PolynomialRootSolver solver;roots=solver.solve(coefficients).roots;}
 if(roots.size()!=6)return {};
 for(auto&z:roots)for(int i=0;i<8;++i){auto[k,d]=kappa(z,p);if(fast_abs(d)==0)return {};auto step=(k-phase)/d;z-=step;if(fast_abs(step)<8e-16*(1+fast_abs(z)))break;}
 std::vector<Disk> out;
 for(int i=0;i<6;++i){auto z=roots[i];auto[k,d]=kappa(z,p);double deriv=fast_abs(d);if(!(deriv>0&&std::isfinite(deriv)))return {};
  double limit=INFINITY,condition=1,dc=0;for(int j=0;j<3;++j){double a=fast_abs(z-l[j]);limit=std::min(limit,.45*a);condition+=(m[j]+.01)/(a*a);dc+=2*(m[j]+.01)/(a*a*a);}
  for(int j=0;j<6;++j)if(i!=j)limit=std::min(limit,.45*fast_abs(z-roots[j]));
  const double beta=(fast_abs(k-phase)+2*std::sin(half/2)+eps*condition)/deriv;
  double r=std::max(1.3*beta,eps*(1+fast_abs(z)));bool ok=false;
  for(int attempt=0;attempt<48&&r<limit;++attempt,r*=1.3){double q=eps*(1+dc)/deriv;for(int j=0;j<3;++j){double lower=fast_abs(z-l[j])-r;q+=6*(m[j]+dm)*r/(lower*lower*lower*lower*deriv);}if(beta+q*r<r*(1-eps)&&q<1-eps){ok=true;break;}}
  if(!ok)return {};
  Complex w=z;double h=r,cond=1+fast_abs(z);for(int j=0;j<3;++j){double a=fast_abs(z-l[j]);w-=m[j]/std::conj(z-l[j]);h+=(m[j]+dm)*r/(a*(a-r));cond+=(m[j]+.01)/(a-r);}
  h=h*(1+eps)+eps*(cond+fast_abs(w));
  if(!std::isfinite(w.real()) || !std::isfinite(w.imag()) || !std::isfinite(h) || !(h>=0))return {};
  out.push_back({z,w,r,h});
 }
 return out;
}
struct Node {double phi,half;int depth,left=-1;bool ready=false;std::vector<Complex> roots;std::vector<Disk> disks,ends[2];};
class Tree {
 Geometry geometry_;std::vector<Node> nodes_;
 Complex witness_source_{};double witness_clearance_=0;int witness_count_=-1;std::size_t witness_hits_=0;
 std::vector<Disk> interior_witnesses_;
 std::optional<std::array<double,3>> refused_query_;
 std::size_t refusal_hits_=0;
 bool visit(int id,Complex source,double rho,int&budget,double&angle,double&uncertainty,double&clearance){
  if(!nodes_[id].ready){if(--budget<0)return false;auto&n=nodes_[id];n.disks=cover(geometry_,n.phi,n.half,n.roots);n.ready=true;}
  auto&n=nodes_[id];bool clear=n.disks.size()==6;
  if(clear)for(auto d:n.disks){double distance=fast_abs(d.w-source);if(distance-d.h<=rho+eps*(1+distance+rho)){clear=false;break;}}
  if(clear){
   for(auto disk:n.disks) {
    const double distance=fast_abs(disk.w-source);
    clearance=std::min(clearance,distance-disk.h-eps*(1+distance+disk.h+fast_abs(source)));
   }
   if(n.ends[0].empty())for(int side=0;side<2;++side){auto roots=n.roots;n.ends[side]=cover(geometry_,n.phi+(side?1:-1)*n.half,0,roots);if(n.ends[side].size()!=6){n.ends[0].clear();n.ends[1].clear();return false;}for(int j=0;j<6;++j)if(fast_abs(n.ends[side][j].z-n.disks[j].z)+n.ends[side][j].r>=n.disks[j].r){n.ends[0].clear();n.ends[1].clear();return false;}}
   for(int j=0;j<6;++j){auto a=n.ends[0][j],b=n.ends[1][j];auto u=a.w-source,v=b.w-source;const auto cross=std::conj(u)*v;angle+=std::atan2(cross.imag(),cross.real());uncertainty+=2*a.h/fast_abs(u)+2*b.h/fast_abs(v)+eps;}
   return true;
  }
  for(auto d:n.disks)if(fast_abs(d.w-source)<rho-eps*(1+rho+fast_abs(source))) {
   // A contained mapped critical point certifies refusal for neighbouring
   // source disks too. A broad phase disk may only suggest containment;
   // reconstruct the same critical roots at the phase centre before caching.
   std::vector<Disk> points;
   if(fast_abs(d.w-source)+d.h<rho-eps*(1+rho+fast_abs(source)))points={d};
   else {auto roots=n.roots;points=cover(geometry_,n.phi,0,roots);}
   for(auto point:points)if(fast_abs(point.w-source)+point.h<rho-eps*(1+rho+fast_abs(source))) {
    if(interior_witnesses_.size()==16)interior_witnesses_.erase(interior_witnesses_.begin());
    interior_witnesses_.push_back(point);break;
   }
   return false;
  }
  if(n.depth>=18)return false;
  if(n.left<0){if(nodes_.size()>8190)return false;auto phi=n.phi,half=n.half*.5;auto depth=n.depth+1;auto roots=n.roots;int left=nodes_.size();nodes_[id].left=left;nodes_.push_back({phi-half,half,depth,-1,false,roots});nodes_.push_back({phi+half,half,depth,-1,false,roots});}
  int left=nodes_[id].left;
  return visit(left,source,rho,budget,angle,uncertainty,clearance)&&visit(left+1,source,rho,budget,angle,uncertainty,clearance);
 }
public:
 explicit Tree(const Geometry&p):geometry_(p){for(int j=0;j<32;++j)nodes_.push_back({2*pi*(j+.5)/32,pi/32,0});}
 int count(Complex source,double rho,int budget=256){
  if(!(rho>0) || !std::isfinite(rho+source.real()+source.imag()))return -1;
  // The same witness-clearance reuse as BinaryCausticTree. This caches
  // a topology count in a caustic-free disk, never a source flux or plan.
  const double moved=fast_abs(source-witness_source_);
  const double available=witness_clearance_-moved-eps*(1+witness_clearance_+moved+fast_abs(source)+fast_abs(witness_source_));
  if(witness_count_>=4 && available>rho){++witness_hits_;return witness_count_;}
  for(auto point:interior_witnesses_)if(fast_abs(point.w-source)+point.h<rho-eps*(1+rho+fast_abs(source))) {
   ++refusal_hits_;return -1;
  }
  if(refused_query_ && *refused_query_==std::array<double,3>{source.real(),source.imag(),rho}) {++refusal_hits_;return -1;}
  double angle=0,error=0,clearance=INFINITY;
  for(int j=0;j<32;++j)if(!visit(j,source,rho,budget,angle,error,clearance)) {
   if(budget>=0)refused_query_=std::array<double,3>{source.real(),source.imag(),rho};return -1;
  }
  if(!std::isfinite(angle) || !std::isfinite(error) || !std::isfinite(clearance))return -1;
  const double winding=std::round(angle/(2*pi));
  if(winding<0 || winding>3 || fast_abs(angle/(2*pi)-winding)+error/(2*pi)>.1)return -1;
  const int count=4+2*int(winding);
  if(count<4 || count>10 || !(clearance>rho))return -1;
  witness_source_=source;witness_clearance_=clearance;witness_count_=count;return count;
 }
 std::size_t witness_hits()const{return witness_hits_;}
 std::size_t refusal_hits()const{return refusal_hits_;}
 size_t nodes()const{return nodes_.size();}
};
// Isolate a dominant pole algebraically instead of linearizing it. The
// inverse of a disk not containing zero is another disk, giving a direct
// self-map and contraction proof even when a centred Newton ball is wide.
inline std::optional<Disk> dominant_image_ball(Complex z,const Geometry&p) {
 using B=interval_detail::Interval<double>;using C=coefficient_detail::C<B>;
 auto inv=[](B a){return B(interval_detail::down(1/a.hi),interval_detail::up(1/a.lo));};
 auto magnitude=[](B a){return std::max(fast_abs(a.lo),fast_abs(a.hi));};
 auto norm_upper=[&](C a){B x(magnitude(a.re)),y(magnitude(a.im));return interval_detail::up(std::sqrt((x*x+y*y).hi));};
 const auto l=p.lenses();const std::array<B,3>m{B(1)-B(p.m1)-B(p.m2),B(p.m1),B(p.m2)};
 const C center{B(z.real()),B(z.imag())};std::array<C,3> inverse{};std::array<double,3> distance{};
 int pole=0;double strongest=-1;
 for(int j=0;j<3;++j){
  C d=center-C{B(l[j].real()),B(l[j].imag())};B n=d.re*d.re+d.im*d.im;
  if(!(n.lo>0 && m[j].lo>0))return std::nullopt;
  distance[j]=interval_detail::down(std::sqrt(n.lo));inverse[j]=d*C{inv(n)};
  const double strength=m[j].hi/n.lo;if(strength>strongest){strongest=strength;pole=j;}
 }
 C H=center-C{B(p.X),B(p.Y)};
 for(int j=0;j<3;++j)if(j!=pole)H=H-C{m[j]}*inverse[j];
 const Complex a(H.re.lo+(H.re.hi-H.re.lo)*.5,H.im.lo+(H.im.hi-H.im.lo)*.5);
 const double uncertainty=norm_upper(H-C{B(a.real()),B(a.imag())});
 const B ar(a.real()),ai(a.imag()),a2=ar*ar+ai*ai;
 if(!(a2.lo>0))return std::nullopt;
 double limit=.9*distance[pole];for(int j=0;j<3;++j)if(j!=pole)limit=std::min(limit,.45*distance[j]);
 double r=1.25*p.rho*distance[pole]*distance[pole]/m[pole].lo;
 r=std::max(r,eps*(1+fast_abs(z)));
 for(int attempt=0;attempt<48 && r<limit;++attempt,r*=1.25) {
  B e=B(p.rho)+B(r)+B(uncertainty),slope(1);
  for(int j=0;j<3;++j)if(j!=pole){
   const B lo=B(distance[j])-B(r);if(!(lo.lo>0))return std::nullopt;
   e=e+m[j]*B(r)*inv(B(distance[j])*lo);
   slope=slope+m[j]*inv(lo*lo);
  }
  const B E(e.hi),lower=B(interval_detail::down(std::sqrt(a2.lo)))-E;
  if(!(lower.lo>0))return std::nullopt;
  const B q=m[pole]*slope*inv(lower*lower);if(!(q.hi<1))return std::nullopt;
  const B denominator=a2-E*E;if(!(denominator.lo>0))return std::nullopt;
  const B factor=m[pole]*inv(denominator);
  const C image=C{B(l[pole].real()),B(l[pole].imag())}+C{ar,ai}*C{factor};
  const B image_radius=factor*E;
  const B self_map=B(norm_upper(image-center))+B(image_radius.hi);
  if(self_map.hi<r)return Disk{z,{},r,0};
 }
 return std::nullopt;
}
inline std::optional<Disk> image_ball(Complex z,const Geometry&p){
 auto l=p.lenses();auto m=p.masses();std::array<double,3> distances{};Complex g=z-Complex(p.X,p.Y),shear{};double shear_condition=0;double condition=1+fast_abs(z)+fast_abs(Complex(p.X,p.Y)),limit=INFINITY;
 for(int j=0;j<3;++j){distances[j]=fast_abs(z-l[j]);auto inv=inverse_conjugate(z-l[j]);g-=m[j]*inv;shear+=m[j]*inv*inv;condition+=(m[j]+.01)*fast_abs(inv);shear_condition+=(m[j]+.01)*std::norm(inv);limit=std::min(limit,.45*distances[j]);}
 double det=1-std::norm(shear),normA=(1+fast_abs(shear))/fast_abs(det);if(!std::isfinite(normA))return std::nullopt;
 double beta=fast_abs((g-shear*std::conj(g))/det)+normA*(p.rho+eps*condition);
 double r=1.25*beta;bool ok=false;
 for(int attempt=0;attempt<48&&r<limit;++attempt,r*=1.25){
  const double rounding=eps*normA*(1+shear_condition);
  double q=rounding,remainder=rounding*r;
  for(int j=0;j<3;++j){
   const double d=distances[j]-eps*(1+distances[j]),lower=d-r;
   if(!(lower>0)){q=INFINITY;break;}
   const double weight=normA*(m[j]+dm)/(d*d);
   // Exact rational remainder after cancelling the centre Jacobian:
   // 1/(d+dz) = 1/d - dz/d^2 + dz^2/(d^2*(d+dz)).
   // Self-mapping needs this quadratic remainder, not q*r.
   remainder+=weight*r*r/lower;
   q+=weight*r*(2*d+r)/(lower*lower);
  }
  if(beta+remainder<r*(1-eps)&&q<1-eps){ok=true;break;}
 }
 if(!ok)return dominant_image_ball(z,p);return Disk{z,{},r,0};
}
inline std::optional<std::pair<double,double>> component_magnification_bounds(const Disk&ball,const Geometry&p){
 auto l=p.lenses();auto m=p.masses();Complex k{};double delta=0,condition=1;
 for(int j=0;j<3;++j){double d=fast_abs(ball.z-l[j]),lo=d-ball.r;if(!(lo>0))return std::nullopt;auto inv=1./std::conj(ball.z-l[j]);k+=m[j]*inv*inv;delta+=2*(m[j]+dm)*ball.r/(lo*lo*lo);condition+=(m[j]+.01)/(lo*lo);}
 delta+=eps*condition;double lower=std::max(0.,fast_abs(k)-delta),upper=fast_abs(k)+delta;
 if(lower<=1 && upper>=1)return std::nullopt;
 double a=fast_abs(1-lower*lower),b=fast_abs(1-upper*upper);
 double detlo=std::min(a,b)-eps*(1+upper*upper),dethi=std::max(a,b)+eps*(1+upper*upper);
 if(!(detlo>0 && std::isfinite(dethi)))return std::nullopt;
 return std::make_pair(std::nextafter(1/dethi,0.),std::nextafter(1/detlo,INFINITY));
}

}

#include "lcbinint/magnification/algebraic_boundary/triple/radial_events.hpp"
namespace lcbinint::algebraic_boundary::triple::regular_support_detail {
using I=angular_detail::I<double>;template<class T>using C=coefficient_detail::C<T>;
inline Complex solve_image(Complex z,Complex target,const Geometry&p){auto l=p.lenses();auto m=p.masses();for(int i=0;i<20;++i){Complex f=z-target,shear{};for(int j=0;j<3;++j){auto v=inverse_conjugate(z-l[j]);f-=m[j]*v;shear+=m[j]*v*v;}double det=1-std::norm(shear);if(det==0)break;auto step=(f-shear*std::conj(f))/det;double length=fast_abs(step);if(!std::isfinite(length))break;if(length>.1){step*=.1/length;length=.1;}z-=step;if(length<2e-15*(1+fast_abs(z)))break;}return z;}
inline std::optional<Disk> branch_at(double phi,Complex anchor,const Geometry&p){auto target=Complex(p.X,p.Y)+std::polar(p.rho,phi);auto z=solve_image(anchor,target,p);Geometry local=p;local.X=target.real();local.Y=target.imag();local.rho=eps*(1+fast_abs(target)+p.rho);return image_ball(z,local);}
// A(z)=z-kappa(z)*conj(z)=w-sum(m_j*conj(l_j)/conj(z-l_j)^2).
// Multiplying d|z|^2/dphi by det(J) cancels its ill-conditioned inverse.
// The primary lens at l_0=0 cancels exactly from U and T below.
template<class T> std::optional<std::pair<T,T>> stationary_motion(C<T> z,T phase_re,T phase_im,const Geometry&p){
 auto l=p.lenses();std::array<T,3>m{T(1)-T(p.m1)-T(p.m2),T(p.m1),T(p.m2)};C<T>k{},U{},third{};
 for(int j=0;j<3;++j){C<T>d=coefficient_detail::conj(z-C<T>{T(l[j].real()),T(l[j].imag())});T norm=d.re*d.re+d.im*d.im;if(!physical_detail::positive_value(norm))return std::nullopt;C<T>v=coefficient_detail::conj(d)*C<T>{physical_detail::reciprocal_value(norm)};k=k+C<T>{m[j]}*v*v;if(j){C<T>weight{m[j]*T(l[j].real()),-m[j]*T(l[j].imag())};U=U+weight*v*v;third=third+weight*v*v*v;}}
 T det=T(1)-k.re*k.re-k.im*k.im,inverse;
 if constexpr(std::is_same_v<T,I>){if(!det.valid() || fast_abs(det.sign())!=1)return std::nullopt;inverse={interval_detail::down(1/det.hi),interval_detail::up(1/det.lo)};}else{if(det==0)return std::nullopt;inverse=1/det;}
 C<T>wp{-T(p.rho)*phase_im,T(p.rho)*phase_re},wpp{-T(p.rho)*phase_re,-T(p.rho)*phase_im};
 C<T>zp=(wp-k*coefficient_detail::conj(wp))*C<T>{inverse};C<T>offset=C<T>{T(p.X),T(p.Y)}-U;
 return std::make_pair((coefficient_detail::conj(offset)*wp).re,(coefficient_detail::conj(offset)*wpp+C<T>{T(2)}*coefficient_detail::conj(third*coefficient_detail::conj(zp))*wp).re);
}
inline std::optional<std::pair<I,I>> interval_arc_motion(const Disk&d,double phi,double half,const Geometry&p){const double delta=2*std::sin(half*.5)+eps;const auto e=std::polar(1.,phi);const C<I>z{{d.z.real()-d.r,d.z.real()+d.r},{d.z.imag()-d.r,d.z.imag()+d.r}};return stationary_motion(z,I(e.real()-delta,e.real()+delta),I(e.imag()-delta,e.imag()+delta),p);}
inline std::optional<std::pair<I,I>> arc_motion(const Disk& disk,double phi,double half,const Geometry&p){
 const auto z=disk.z;const double r=disk.r;auto l=p.lenses();auto m=p.masses();Complex k{},U{},third{};double dk=0,dU=0,dT=0,kc=0,uc=0,tc=0;
 for(int j=0;j<3;++j){double distance=fast_abs(z-l[j]),lower=distance-r;if(!(lower>0))return std::nullopt;auto v=inverse_conjugate(z-l[j]);k+=m[j]*v*v;dk+=2*(m[j]+dm)*r/(lower*lower*lower);kc+=(m[j]+.01)/(lower*lower);if(j){auto weight=m[j]*std::conj(l[j]);U+=weight*v*v;third+=weight*v*v*v;dU+=2*m[j]*fast_abs(l[j])*r/(lower*lower*lower);dT+=3*m[j]*fast_abs(l[j])*r/(lower*lower*lower*lower);uc+=m[j]*fast_abs(l[j])/(lower*lower);tc+=m[j]*fast_abs(l[j])/(lower*lower*lower);}}
 dk+=eps*(1+kc);dU+=eps*(1+uc);dT+=eps*(1+tc);
 const double det=1-std::norm(k),normA=(1+fast_abs(k))/fast_abs(det),q=normA*dk;
 if(!(q<.9))return interval_arc_motion(disk,phi,half,p);
 auto inv=[&](Complex v){return (v-k*std::conj(v))/det;};const Complex phase=std::polar(1.,phi),wp=Complex(0,p.rho)*phase,wpp=-p.rho*phase;
 const double de=2*std::sin(half*.5)+eps;const Complex v=inv(wp),offset=Complex(p.X,p.Y)-U;
 const double ev=normA*(p.rho*de+dk*fast_abs(v))/(1-q)+eps*(1+normA)*(1+fast_abs(v));
 const Complex W=third*std::conj(v);const double eW=dT*(fast_abs(v)+ev)+fast_abs(third)*ev;
 const double g=std::real(std::conj(offset)*wp),gg=std::real(std::conj(offset)*wpp+2.*std::conj(W)*wp);
 const double common=fast_abs(offset)*p.rho*de+dU*p.rho*(1+de);
 const double error=common+eps*(1+fast_abs(g));
 const double error2=common+2*p.rho*((fast_abs(W)+eW)*de+eW)+eps*(1+fast_abs(gg));
 return std::make_pair(I(interval_detail::down(g-error),interval_detail::up(g+error)),I(interval_detail::down(gg-error2),interval_detail::up(gg+error2)));
}
inline bool parent_contains(const Disk&parent,const Disk&child){return fast_abs(parent.z-child.z)+child.r<parent.r*(1-eps);}
inline std::optional<RadialEvent> event_from_arc(const Disk&parent,double phi,double half,const Geometry&p,double relative_tolerance=0,int* quad_events=nullptr){
 double angle=phi;Complex z=parent.z;
 for(int j=0;j<12;++j){auto image=branch_at(angle,z,p);if(!image || !parent_contains(parent,*image))return std::nullopt;z=image->z;auto e=std::polar(1.,angle);auto f=stationary_motion(C<double>{z.real(),z.imag()},e.real(),e.imag(),p);if(!f || f->second==0)return std::nullopt;double step=f->first/f->second;if(!std::isfinite(step))return std::nullopt;double next=angle-step;if(!(next>phi-half && next<phi+half))return std::nullopt;angle=next;if(fast_abs(step)<1e-13)break;}
 if(relative_tolerance>0) {
  auto point=branch_at(angle,z,p);auto whole=arc_motion(parent,phi,half,p);
  if(point && whole && fast_abs(whole->second.sign())==1) {
   auto value=arc_motion(*point,angle,0,p);
   double slope=std::min(fast_abs(whole->second.lo),fast_abs(whole->second.hi));
   double width=value?2*std::max(fast_abs(value->first.lo),fast_abs(value->first.hi))/slope:INFINITY;
   width=std::max(width,64*std::numeric_limits<double>::epsilon()*(1+fast_abs(angle)));
   for(int attempt=0;attempt<5 && fast_abs(angle-phi)+width<half;++attempt,width*=2) {
    auto left=branch_at(angle-width,z,p),right=branch_at(angle+width,z,p);
    if(!left || !right || !parent_contains(parent,*left) || !parent_contains(parent,*right))continue;
    auto a=arc_motion(*left,angle-width,0,p),b=arc_motion(*right,angle+width,0,p);
    if(!a || !b || a->first.sign()*b->first.sign()!=-1)continue;
    Geometry local=p;auto target=Complex(p.X,p.Y)+std::polar(p.rho,angle);local.X=target.real();local.Y=target.imag();local.rho=2*p.rho*std::sin(width*.5)+eps*(1+fast_abs(target)+p.rho);
    auto cover=image_ball(z,local);
    if(!cover || !parent_contains(*cover,*point) || !parent_contains(parent,*point))continue;
    double R=fast_abs(cover->z),lo=std::max(0.,R-cover->r-eps*(1+R)),hi=R+cover->r+eps*(1+R);
    if((hi-lo)*(hi+lo)/(p.rho*p.rho)>relative_tolerance/32)break;
    double theta=std::arg(cover->z),t=std::tan(theta*.5);bool chart=fast_abs(t)>1;
    double shift=chart?pi:0;while(theta-shift>pi)shift+=2*pi;while(theta-shift<-pi)shift-=2*pi;
    double h=std::asin(cover->r/R)+eps;
    double tl=std::tan((theta-shift-h)*.5)-eps,th=std::tan((theta-shift+h)*.5)+eps;
    t=std::tan((theta-shift)*.5);
    return RadialEvent{R,t,lo,hi,chart,tl,th};
   }
  }
 }
 double r=fast_abs(z),t=std::tan(std::arg(z)*.5);bool chart=fast_abs(t)>1;if(chart)t=-1/t;
 for(int j=0;j<12;++j){auto f=event_detail::physical(r,t,p,chart);if(!f)return std::nullopt;double det=f->r*f->tt-f->t*f->rt;if(det==0)return std::nullopt;double dr=(-f->v*f->tt+f->t*f->t)/det,dt=(-f->r*f->t+f->rt*f->v)/det;r+=dr;t+=dt;if(fast_abs(dr)+fast_abs(dt)<1e-14*(1+r+fast_abs(t)))break;}
 const double dr=64*std::numeric_limits<double>::epsilon()*(1+r),dt=64*std::numeric_limits<double>::epsilon()*(1+fast_abs(t));event_detail::Box box{r-dr,r+dr,t-dt,t+dt,chart};auto proof=event_detail::prove_physical(box,p);
 if(proof.kind!=2){if(quad_events)++*quad_events;proof=event_detail::prove_quad(box,p);if(proof.kind!=2)return std::nullopt;}
 // The existing two-dimensional proof encloses a physical stationary point.
 // Match it to the unique stationary branch in this source-phase interval.
 auto center=std::polar(r,(chart?pi:0)+2*std::atan(t));double radius=(proof.r.hi-proof.r.lo)+2*(r+dr)*(proof.t.hi-proof.t.lo)+eps*(1+r);
 if(fast_abs(center-parent.z)+radius>=parent.r)return std::nullopt;
 Complex w=center-Complex(p.X,p.Y);double error=radius,condition=1+fast_abs(center)+fast_abs(Complex(p.X,p.Y));auto l=p.lenses();auto m=p.masses();
 for(int j=0;j<3;++j){double d=fast_abs(center-l[j]);if(d<=radius)return std::nullopt;w-=m[j]/std::conj(center-l[j]);error+=m[j]*radius/(d*(d-radius));condition+=m[j]/(d-radius);}
 error+=eps*condition;if(error>=p.rho)return std::nullopt;
 double phase=phi+std::remainder(std::arg(w)-phi,2*pi),phase_error=2*error/p.rho+eps;
 if(fast_abs(phase-phi)+phase_error>=half)return std::nullopt;
 return RadialEvent{r,t,proof.r.lo,proof.r.hi,chart,proof.t.lo,proof.t.hi};
}
inline RadialEventPlan event_plan(const Geometry&p,const std::vector<Complex>&proposals,int max_nodes=256,double omission_budget=0,bool allow_omission=true,const Disk* component=nullptr){
 RadialEventPlan out;out.identity={p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by};out.rmin=INFINITY;out.rmax=0;out.expected_images=int(proposals.size());
 std::vector<Complex> retained;std::vector<Disk> verified_images;
 for(auto z:proposals) {
  auto ball=component?std::optional<Disk>(*component):image_ball(z,p);double bound=INFINITY,value=0;
  if(ball && allow_omission){auto range=component_magnification_bounds(*ball,p);if(range){value=(range->first+range->second)*.5;bound=std::nextafter(std::max(value-range->first,range->second-value),INFINITY);}}
  if(allow_omission && ball && bound<=omission_budget-out.bounded_component_error) {
   out.excluded_images.push_back({ball->z,ball->r});out.bounded_component_error=std::nextafter(out.bounded_component_error+bound,INFINITY);out.bounded_component_value+=value;out.bounded_component_error=std::nextafter(out.bounded_component_error+eps*fast_abs(value),INFINITY);
  } else {retained.push_back(z);if(ball){out.component_images.push_back({ball->z,ball->r});verified_images.push_back(*ball);}}
 }

 // Disjoint full-source IFT disks already cover all of their image
 // branches. Reuse them instead of proving eight new source-arc disks.
 // Overlap or an unproved disk retains the original initial-cover route.
 bool full_cover=!component && !retained.empty() && verified_images.size()==retained.size();
 if(full_cover)for(std::size_t i=0;i<verified_images.size();++i){
  const auto&ball=verified_images[i];
  for(std::size_t j=0;j<i;++j)if(!(fast_abs(ball.z-verified_images[j].z)>
      ball.r+verified_images[j].r+eps*(1+fast_abs(ball.z)+fast_abs(verified_images[j].z))))full_cover=false;
  for(auto omitted:out.excluded_images)if(!(fast_abs(ball.z-omitted.center)>
      ball.r+omitted.radius+eps*(1+fast_abs(ball.z)+fast_abs(omitted.center))))full_cover=false;
 }
 struct Arc{double phi,half;int depth;};struct Branch{Arc arc;Disk ball;};std::vector<Arc>pending;std::vector<Branch>branches;
 for(int j=0;j<8;++j) {
  Arc arc{.137+2*pi*(j+.5)/8,pi/8,0};
  if(component)branches.push_back({arc,*component});
  else if(full_cover)for(auto ball:verified_images)branches.push_back({arc,ball});
  else pending.push_back(arc);
 }
 if(component){out.rmin=fast_abs(component->z)-component->r;out.rmax=fast_abs(component->z)+component->r;}
 if(full_cover)for(auto ball:verified_images){out.rmin=std::min(out.rmin,fast_abs(ball.z)-ball.r);out.rmax=std::max(out.rmax,fast_abs(ball.z)+ball.r);}
 auto ball_for=[&](Arc a,Complex anchor)->std::optional<Disk>{Geometry local=p;auto target=Complex(p.X,p.Y)+std::polar(p.rho,a.phi);local.X=target.real();local.Y=target.imag();local.rho=2*p.rho*std::sin(a.half*.5)+eps*(1+fast_abs(target)+p.rho);return image_ball(solve_image(anchor,target,p),local);};
 while(!pending.empty()){
  if(++out.boxes>max_nodes){out.reason="regular_arc_budget";return out;}auto a=pending.back();pending.pop_back();
  std::vector<Disk>balls;bool valid=true;
  for(auto z:retained){auto ball=ball_for(a,z);if(!ball){valid=false;break;}
   if(component) {
    auto witness=branch_at(a.phi,ball->z,p);
    if(!witness || !parent_contains(*component,*witness) || !parent_contains(*ball,*witness)){valid=false;break;}
   }
   if(!out.excluded_images.empty()) {
    auto witness=branch_at(a.phi,ball->z,p);
    if(!witness || !parent_contains(*ball,*witness)){valid=false;break;}
    for(auto excluded:out.excluded_images)if(fast_abs(witness->z-excluded.center)-witness->r<=excluded.radius){valid=false;break;}
    if(!valid)break;
   }
   for(auto other:balls)if(fast_abs(other.z-ball->z)<=other.r+ball->r){valid=false;break;}if(!valid)break;balls.push_back(*ball);}
  if(!valid){if(a.depth>=18){out.reason="regular_initial_cover";return out;}pending.push_back({a.phi-a.half*.5,a.half*.5,a.depth+1});pending.push_back({a.phi+a.half*.5,a.half*.5,a.depth+1});continue;}
  for(auto b:balls){branches.push_back({a,b});out.rmin=std::min(out.rmin,fast_abs(b.z)-b.r);out.rmax=std::max(out.rmax,fast_abs(b.z)+b.r);}
 }
 while(!branches.empty()){
  if(++out.boxes>max_nodes){out.reason="regular_branch_budget";return out;}auto branch=branches.back();branches.pop_back();auto a=branch.arc;auto ball=branch.ball;bool resolved=false;
  auto f=arc_motion(ball,a.phi,a.half,p);
  if(f && fast_abs(f->first.sign())==1)continue;
  if(f && fast_abs(f->second.sign())==1){
   auto left=branch_at(a.phi-a.half,ball.z,p),right=branch_at(a.phi+a.half,ball.z,p);
   if(left && right && parent_contains(ball,*left) && parent_contains(ball,*right)){
    auto lo=arc_motion(*left,a.phi-a.half,0,p),hi=arc_motion(*right,a.phi+a.half,0,p);
    if(lo && hi && fast_abs(lo->first.sign())==1 && fast_abs(hi->first.sign())==1){
     if(lo->first.sign()==hi->first.sign())continue;
     auto event=event_from_arc(ball,a.phi,a.half,p,omission_budget*8,&out.quad_refinements);if(event){out.events.push_back(*event);resolved=true;}
    }
   }
  }
  if(resolved)continue;
  if(a.depth>=18){out.reason="regular_branch_unresolved";return out;}
  for(int side:{-1,1}) {
   Arc child{a.phi+side*a.half*.5,a.half*.5,a.depth+1};auto b=ball_for(child,ball.z);auto witness=branch_at(child.phi,ball.z,p);
   if(b && witness && parent_contains(ball,*witness) && parent_contains(*b,*witness))branches.push_back({child,*b});
   else branches.push_back({child,ball});
  }
 }
 std::sort(out.events.begin(),out.events.end(),[](auto a,auto b){return a.radius<b.radius;});out.rmin=std::max(0.,out.rmin-eps*(1+out.rmax));out.rmax+=eps*(1+out.rmax);out.completed=!out.events.empty()&&out.rmin>0;out.reason=out.completed?"regular_source_boundary":"regular_empty_events";return out;
}
}

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "projected_radial_events.hpp"
namespace lcbinint::algebraic_boundary::triple {
class PreparedRegularSupport {
    std::shared_ptr<point_polynomial_detail::LensContext> lens_context_;
    std::optional<std::array<double,5>> identity_;
    std::optional<regular_support_detail::Tree> tree_;
    magnification::PointSourceMagnifier point_;
    std::vector<Complex> image_seeds_;
    Complex seed_source_{};
    std::size_t native_solves_=0,continued_seeds_=0;
    std::optional<Geometry> refused_partial_;
    std::optional<Geometry> event_seed_geometry_;
    std::vector<projected_event_detail::Guess> event_seeds_;
    RadialEventPlan contact_plan(const Geometry&p,double tolerance,int budget) {
        RadialEventPlan out;out.reason="regular_caustic_or_budget";
        if(budget<64)return out;
        const auto* previous=event_seed_geometry_ && event_seed_geometry_->rho==p.rho &&
            std::hypot(p.X-event_seed_geometry_->X,p.Y-event_seed_geometry_->Y)<=p.rho*.75
            ?&event_seeds_:nullptr;
        auto projected=[&](auto make_components) {
            auto result=projected_radial_event_plan(p,budget,tolerance*.1,nullptr,previous,make_components);
            if(result.completed && result.resultant_nodes>0 && result.events.size()<=64) {
                event_seeds_.clear();
                for(auto event:result.events)
                    event_seeds_.push_back({event.radius,event.coordinate,event.reciprocal});
                event_seed_geometry_=p;
            }
            return result;
        };
        if(refused_partial_ && refused_partial_->rho==p.rho &&
            std::hypot(p.X-refused_partial_->X,p.Y-refused_partial_->Y)<=p.rho*.5)
            return projected(projected_event_detail::NoComponents{});
        auto make_components=[&]() {

            std::vector<Complex> proposals;
            if(!image_seeds_.empty()) {
                for(auto z:image_seeds_)proposals.push_back(regular_support_detail::solve_image(z,{p.X,p.Y},p));
            } else {
                model::TripleLensGeometry geometry;geometry.masses=p.masses();auto lenses=p.lenses();
                for(int j=0;j<3;++j)geometry.lens_positions[j]={lenses[j].real(),lenses[j].imag()};
                ++native_solves_;
                for(auto image:point_.triple_images(geometry,{p.X,p.Y}))proposals.push_back({image.position.x,image.position.y});
            }
            image_seeds_=proposals;seed_source_={p.X,p.Y};
            PartialComponents known;known.identity={p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by};
            bool any_ball=false;
            for(auto z:proposals) {
                auto ball=regular_support_detail::image_ball(z,p);if(!ball)continue;any_ball=true;
                bool disjoint=true;for(auto old:known.images)if(regular_support_detail::fast_abs(old.center-ball->z)<=old.radius+ball->r+regular_support_detail::eps*(1+regular_support_detail::fast_abs(old.center)+regular_support_detail::fast_abs(ball->z)))disjoint=false;
                if(!disjoint)continue;
                auto branch=regular_support_detail::event_plan(p,{z},64,tolerance/8,false,&*ball);
                if(!branch.completed || branch.events.size()!=2)continue;
                bool interior=true;
                for(auto e:branch.events)if(!event_detail::inside_component({e.radius_lo,e.radius_hi,e.coordinate_lo,e.coordinate_hi,e.reciprocal},{{ball->z,ball->r}}))interior=false;
                if(!interior)continue;
                known.images.push_back({ball->z,ball->r});known.events.insert(known.events.end(),branch.events.begin(),branch.events.end());
            }
            if(!any_ball)refused_partial_=p;
            return known;
        };
        return projected(make_components);
    }
public:
    std::size_t native_seed_solves() const{return native_solves_;}
    std::size_t continued_seed_hits() const{return continued_seeds_;}
    RadialEventPlan plan(const Geometry& p,double relative_tolerance,int event_budget=1000000) {
        auto out=plan_impl(p,relative_tolerance,event_budget);
        if(out.completed) {
            if(!lens_context_ || !lens_context_->matches(p))lens_context_=std::make_shared<point_polynomial_detail::LensContext>(p);
            lens_context_->note_query();out.lens_context=lens_context_;
        }
        return out;
    }
private:
    RadialEventPlan plan_impl(const Geometry& p,double relative_tolerance,int event_budget) {
        RadialEventPlan out;
        // This optimization is validated for loose relative value requests.
        // Strict references and gradients retain the full support planner.
        if(relative_tolerance<1e-4){out.reason="regular_tolerance_gate";return out;}
        if(event_budget<1){out.reason="regular_event_budget";return out;}
        p.validate();
        const auto masses=p.masses();const auto lenses=p.lenses();
        const double sum=p.m1+p.m2,bvirtual=sum-p.m1;
        const double error=(p.m1-(sum-bvirtual))+(p.m2-bvirtual);
        if(!(sum<1 || (sum==1 && error<0)) || !(masses[0]>0 && p.m1>0 && p.m2>0) ||
           p.a==0 || regular_support_detail::fast_abs(lenses[2])==0 || regular_support_detail::fast_abs(lenses[1]-lenses[2])==0) {
            out.reason="regular_geometry_gate";return out;
        }
        if(!angular_detail::enclosure::environment_ok()){out.reason="floating_environment";return out;}
        const std::array<double,5> identity{p.m1,p.m2,p.a,p.bx,p.by};
        if(!identity_ || *identity_!=identity){tree_.emplace(p);identity_=identity;image_seeds_.clear();refused_partial_.reset();event_seeds_.clear();event_seed_geometry_.reset();}
        const int expected=tree_->count({p.X,p.Y},p.rho);
        if(expected<0)return contact_plan(p,relative_tolerance,event_budget);
        std::vector<Complex> proposals;
        // Root positions are reusable proposals, not flux/support results.
        // The caustic tree independently supplies N. N disjoint point-source
        // IFT disks certify continued seeds before any boundary work starts.
        if(image_seeds_.size()==std::size_t(expected)) {
            Geometry local=p;local.rho=regular_support_detail::eps*(1+regular_support_detail::fast_abs(Complex(p.X,p.Y)));
            std::vector<regular_support_detail::Disk> disks;
            bool valid=true;
            for(auto z:image_seeds_) {
                if(seed_source_!=Complex(p.X,p.Y))z=regular_support_detail::solve_image(z,{p.X,p.Y},p);
                auto disk=regular_support_detail::image_ball(z,local);
                if(!disk){valid=false;break;}
                for(auto old:disks)if(regular_support_detail::fast_abs(old.z-disk->z)<=old.r+disk->r){valid=false;break;}
                if(!valid)break;
                disks.push_back(*disk);proposals.push_back(z);
            }
            if(valid)++continued_seeds_;else proposals.clear();
        }
        if(proposals.size()!=std::size_t(expected)) {
            model::TripleLensGeometry geometry;geometry.masses=masses;
            for(int j=0;j<3;++j)geometry.lens_positions[j]={lenses[j].real(),lenses[j].imag()};
            ++native_solves_;
            const auto images=point_.triple_images(geometry,{p.X,p.Y});
            if(images.size()!=std::size_t(expected)){out.reason="regular_image_count";return out;}
            for(auto image:images)proposals.push_back({image.position.x,image.position.y});
        }
        image_seeds_=proposals;seed_source_={p.X,p.Y};
        return regular_support_detail::event_plan(p,proposals,std::min(256,event_budget),relative_tolerance/8);
    }
};
}

#if defined(__GNUC__)
#pragma GCC pop_options
#endif

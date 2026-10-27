#pragma once
#include "point_polynomial.hpp"
#include "angular_topology.hpp"
#include "radial_events.hpp"
namespace lcbinint::algebraic_boundary::triple::sector_certificate_detail {
using B=coefficient_ball_detail::Ball<double>;
inline constexpr auto binomials=[](){std::array<std::array<int,15>,15> b{};for(int n=0;n<=14;++n){b[n][0]=b[n][n]=1;for(int k=1;k<n;++k)b[n][k]=b[n-1][k-1]+b[n-1][k];}return b;}();
inline int choose(int n,int k){return binomials[n][k];}
template<std::size_t N> bool negative_bernstein(const std::array<B,N>& coefficients,int depth=0){
 constexpr int degree=int(N)-1;
 if(coefficients.front().value-coefficients.front().error>0||coefficients.back().value-coefficients.back().error>0)return false;
 bool negative=true;for(const auto& c:coefficients)negative&=std::isfinite(c.value)&&std::isfinite(c.error)&&c.value+c.error+16*std::numeric_limits<double>::epsilon()*(std::abs(c.value)+c.error)<0;if(negative)return true;if(depth==5)return false;
 auto work=coefficients;std::array<B,N> left{},right{};left[0]=work[0];right[degree]=work[degree];for(int k=1;k<=degree;++k){for(int j=0;j<=degree-k;++j)work[j]=(work[j]+work[j+1])*B(.5);left[k]=work[0];right[degree-k]=work[degree-k];}if(work[0].value-work[0].error>0)return false;return negative_bernstein(left,depth+1)&&negative_bernstein(right,depth+1);
}
inline bool negative_ray(const point_polynomial_detail::Cache<double>& tape,double a,double b,double theta,bool positive=false){
 const double pi=std::acos(-1.);double t=std::tan(theta*.5);const bool reciprocal=std::abs(t)>1;if(reciprocal)t=-1/t;std::array<B,9> polynomial{};B power(1);
 for(int k=0;k<7;++k){const auto& c=tape.coefficients[reciprocal?6-k:k];const B weight=reciprocal&&k%2?-power:power;for(int j=0;j<=c.degree;++j)polynomial[j]=polynomial[j]+weight*c.c[j];power=power*B(t);}
 if(positive)for(auto& coefficient:polynomial)coefficient=-coefficient;
 const B origin(a),width=B(b)-B(a);std::array<B,9> ap{},wp{},translated{},bernstein{};ap[0]=wp[0]=B(1);for(int j=1;j<=8;++j){ap[j]=ap[j-1]*origin;wp[j]=wp[j-1]*width;}
 for(int j=0;j<=8;++j)for(int k=j;k<=8;++k)translated[j]=translated[j]+polynomial[k]*B(choose(k,j))*ap[k-j]*wp[j];
 for(int i=0;i<=8;++i)for(int j=0;j<=i;++j){const double ratio=double(choose(i,j))/choose(8,j);const B q(ratio,4*std::numeric_limits<double>::epsilon()*ratio);bernstein[i]=bernstein[i]+translated[j]*q;}
 return negative_bernstein(bernstein);
}
// Negative separating rays over the entire radial band prevent arcs from
// crossing sectors. The incumbent complete event/count proof then gives
// one continued positive arc per sector; no new coverage assumption is made.
inline std::optional<std::vector<double>> certify(const Geometry& p,const AngularTopology& topology,double a,double b,const point_polynomial_detail::Cache<double>& tape,bool positive=false){
 const int n=int(topology.arcs.size());if(n<2||n>3||topology.roots.size()!=std::size_t(2*n))return std::nullopt;const double pi=std::acos(-1.),period=2*pi;
 std::vector<std::pair<double,double>> arcs;for(const auto& arc:topology.arcs){double mid=.5*(arc.lo+arc.hi),half=.5*(arc.hi-arc.lo);mid-=period*std::floor(mid/period);arcs.push_back({mid,half});}std::sort(arcs.begin(),arcs.end());std::vector<double> cuts;
 if(positive){for(const auto& arc:arcs){if(!negative_ray(tape,a,b,arc.first,true))return std::nullopt;cuts.push_back(arc.first);}return cuts;}
 for(int i=0;i<n;++i){const double lo=arcs[i].first+arcs[i].second,hi=arcs[(i+1)%n].first-arcs[(i+1)%n].second+(i+1==n?period:0);if(!(hi-lo>1e-9))return std::nullopt;const double theta=.5*(lo+hi);if(!negative_ray(tape,a,b,theta))return std::nullopt;double cut=theta-period*std::floor(theta/period);cuts.push_back(cut);}
 std::sort(cuts.begin(),cuts.end());return cuts;
}
inline int sector(const std::vector<double>& cuts,double theta){const double period=2*std::acos(-1.);theta=cuts[0]+std::fmod(std::fmod(theta-cuts[0],period)+period,period);return int(std::upper_bound(cuts.begin(),cuts.end(),theta)-cuts.begin())-1;}
struct MovingRay {
 double radius=0,coordinate=0,slope=0,angle=0;bool reciprocal=false;
 Complex unit(double R)const{const double t=coordinate+slope*(R-radius),sign=reciprocal?-1.:1.;return sign*Complex(1-t*t,2*t);}
};
inline bool ray_order(Complex a,Complex b){const bool ah=a.imag()>0||(a.imag()==0&&a.real()>=0),bh=b.imag()>0||(b.imag()==0&&b.real()>=0);if(ah!=bh)return ah;return a.real()*b.imag()-a.imag()*b.real()>0;}
inline int sector(const std::vector<MovingRay>& rays,double R,Complex unit){const Complex first=std::conj(rays[0].unit(R)),point=first*unit;int result=0;for(std::size_t j=1;j<rays.size();++j){if(ray_order(point,first*rays[j].unit(R)))break;result=int(j);}return result;}
inline bool moving_ray_sign(const point_polynomial_detail::Cache<double>& tape,double a,double b,const MovingRay& ray,bool positive){
 const double ta=ray.coordinate+ray.slope*(a-ray.radius),tb=ray.coordinate+ray.slope*(b-ray.radius);if(std::max(std::abs(ta),std::abs(tb))>2)return false;
 const B origin(a),width=B(b)-B(a);std::array<B,9> ap{},wp{};ap[0]=wp[0]=B(1);for(int j=1;j<=8;++j){ap[j]=ap[j-1]*origin;wp[j]=wp[j-1]*width;}
 B u=B(ray.coordinate)+B(ray.slope)*(origin-B(ray.radius)),v=B(ray.slope)*width;const double eps=std::numeric_limits<double>::epsilon();u.error+=8*eps*(1+std::abs(ray.coordinate)+std::abs(ray.slope)*(std::abs(a)+std::abs(ray.radius)));v.error+=8*eps*(1+std::abs(v.value));
 std::array<std::array<B,7>,7> powers{};powers[0][0]=B(1);for(int k=1;k<=6;++k)for(int j=0;j<=k;++j){powers[k][j]=powers[k-1][j]*u;if(j)powers[k][j]=powers[k][j]+powers[k-1][j-1]*v;}
 std::array<B,15> polynomial{},bernstein{};
 for(int k=0;k<7;++k){const auto& p=tape.coefficients[ray.reciprocal?6-k:k];std::array<B,9> translated{};for(int j=0;j<=p.degree;++j)for(int n=j;n<=p.degree;++n)translated[j]=translated[j]+p.c[n]*B(choose(n,j))*ap[n-j]*wp[j];const double sign=(positive?-1.:1.)*(ray.reciprocal&&k%2?-1.:1.);for(int j=0;j<=p.degree;++j)for(int n=0;n<=k;++n)polynomial[j+n]=polynomial[j+n]+translated[j]*powers[k][n]*B(sign);}
 for(int i=0;i<=14;++i)for(int j=0;j<=i;++j){const double ratio=double(choose(i,j))/choose(14,j);bernstein[i]=bernstein[i]+polynomial[j]*B(ratio,4*eps*ratio);}return negative_bernstein(bernstein);
}
template<class Anchor> std::optional<std::vector<MovingRay>> certify_moving(const AngularTopology& topology,const Anchor& anchor,double a,double b,const point_polynomial_detail::Cache<double>& tape,bool positive){
 const int n=int(anchor.count);if(n<2||n>3||topology.roots.size()!=std::size_t(2*n))return std::nullopt;const double period=2*std::acos(-1.);struct Arc{double mid,half,dl,dh;};std::vector<Arc> arcs;for(int j=0;j<n;++j){const auto& q=anchor.arcs[j];double mid=.5*(q.lo+q.hi);mid-=period*std::floor(mid/period);arcs.push_back({mid,.5*(q.hi-q.lo),q.dlo,q.dhi});}std::sort(arcs.begin(),arcs.end(),[](const auto& x,const auto& y){return x.mid<y.mid;});std::vector<MovingRay> rays;
 for(int j=0;j<n;++j){const auto& x=arcs[j];const auto& y=arcs[(j+1)%n];double angle=positive?x.mid:.5*(x.mid+x.half+y.mid-y.half+(j+1==n?period:0)),radial=positive?.5*(x.dl+x.dh):.5*(x.dh+y.dl);angle-=period*std::floor(angle/period);double t=std::tan(angle*.5);const bool reciprocal=std::abs(t)>1;if(reciprocal)t=-1/t;MovingRay ray{.5*(a+b),t,.5*(1+t*t)*radial,angle,reciprocal};if(!moving_ray_sign(tape,a,b,ray,positive))return std::nullopt;rays.push_back(ray);}
 std::sort(rays.begin(),rays.end(),[](const auto& x,const auto& y){return x.angle<y.angle;});return rays;
}

// At a known simple fold F=F_t=0. For a ray anchored to the event box,
// |F(Re,t(Re))| <= .5 max|F_tt| delta_t^2. A one-sided interval bound on
// F_R+sF_t then certifies the sign up to the first representable interior
// sample, without subtracting an uncertain polynomial constant near zero.
inline bool fold_ray_collar(const Geometry& p,const MovingRay& ray,const RadialEvent& event,double outer,bool left,bool positive,double safe){
 using I=interval_detail::Interval<double>;if(event.reciprocal!=ray.reciprocal)return false;
 const double ra=left?event.radius_lo:outer,rb=left?outer:event.radius_hi;if(!(rb>ra))return false;
 const I t=I(ray.coordinate)+I(ray.slope)*(I(ra,rb)-I(ray.radius));
 const I coords(std::min(t.lo,event.coordinate_lo),std::max(t.hi,event.coordinate_hi));const auto f=physical_detail::physical(I(ra,rb),coords,p,ray.reciprocal);if(!f)return false;
 const I derivative=f->r+f->t*I(ray.slope);const int wanted=(positive?1:-1)*(left?1:-1);if(derivative.sign()!=wanted)return false;
 const double d=std::min(std::abs(derivative.lo),std::abs(derivative.hi));
 const I delta=I(ray.coordinate)+I(ray.slope)*(I(event.radius_lo,event.radius_hi)-I(ray.radius))-I(event.coordinate_lo,event.coordinate_hi);
 const double delta_max=std::max(std::abs(delta.lo),std::abs(delta.hi)),curvature=std::max(std::abs(f->tt.lo),std::abs(f->tt.hi));
 const double sample=std::nextafter(safe,outer);const I distance=left?I(sample)-I(event.radius_hi):I(event.radius_lo)-I(sample);if(!(distance.lo>0))return false;
 const I residue=I(.5)*I(curvature)*I(delta_max)*I(delta_max),margin=I(d)*distance;return residue.hi<margin.lo;
}
inline bool moving_ray_sign_with_folds(const Geometry& p,const point_polynomial_detail::Cache<double>& tape,double a,double b,double safe_a,double safe_b,const MovingRay& ray,bool positive,const std::vector<RadialEvent>& events){
 const double sign=positive?1.:-1.;for(double fraction:{.001,.125,.5,.875,.999}){const double R=a+(b-a)*fraction,t=ray.coordinate+ray.slope*(R-ray.radius);const auto value=physical_detail::Ray<double>(R,p).value(t,ray.reciprocal);if(!value||sign*(*value)<0)return false;}
 using I=interval_detail::Interval<double>;const auto intersects=[&](const RadialEvent& e){const I delta=I(ray.coordinate)+I(ray.slope)*(I(e.radius_lo,e.radius_hi)-I(ray.radius))-I(e.coordinate_lo,e.coordinate_hi);return delta.lo<=0&&delta.hi>=0;};
 bool anchored=false;for(const auto& e:events)if((e.radius==a||e.radius==b)&&e.reciprocal==ray.reciprocal&&intersects(e))anchored=true;
 if(!anchored&&moving_ray_sign(tape,a,b,ray,positive))return true;double lo=a,hi=b;bool trimmed=false;
 for(const auto& e:events){if((e.radius!=a&&e.radius!=b)||e.reciprocal!=ray.reciprocal||!intersects(e))continue;const bool left=e.radius==a;const double outer=left?a+(b-a)/16:b-(b-a)/16;
  if(fold_ray_collar(p,ray,e,outer,left,positive,left?safe_a:safe_b)){if(left)lo=outer;else hi=outer;trimmed=true;}}
 return trimmed&&moving_ray_sign(tape,lo,hi,ray,positive);
}
template<class Anchor> std::optional<std::vector<MovingRay>> certify_following_folds(const Geometry& p,const AngularTopology& topology,const Anchor& anchor,double a,double b,double safe_a,double safe_b,const point_polynomial_detail::Cache<double>& tape,bool positive,const std::vector<RadialEvent>& events){
 const int n=int(anchor.count);if(n<2||n>3||topology.roots.size()!=std::size_t(2*n))return std::nullopt;const double period=2*std::acos(-1.),mid=.5*(a+b);struct Arc{double mid,half,dl,dh;};std::vector<Arc> arcs;for(int j=0;j<n;++j){const auto& q=anchor.arcs[j];double center=.5*(q.lo+q.hi);center-=period*std::floor(center/period);arcs.push_back({center,.5*(q.hi-q.lo),q.dlo,q.dhi});}std::sort(arcs.begin(),arcs.end(),[](const auto& x,const auto& y){return x.mid<y.mid;});std::vector<MovingRay> result;
 for(int j=0;j<n;++j){const auto& x=arcs[j];const auto& y=arcs[(j+1)%n];const double component_lo=positive?x.mid-x.half:x.mid+x.half,component_hi=positive?x.mid+x.half:y.mid-y.half+(j+1==n?period:0),angle=.5*(component_lo+component_hi),radial=positive?.5*(x.dl+x.dh):.5*(x.dh+y.dl);
  std::vector<MovingRay> candidates;double t=std::tan(angle*.5);bool chart=std::abs(t)>1;if(chart)t=-1/t;candidates.push_back({mid,t,0,angle,chart});candidates.push_back({mid,t,.5*(1+t*t)*radial,angle,chart});
  for(const auto& e:events){if(e.radius!=a&&e.radius!=b)continue;const double base=e.reciprocal?std::acos(-1.):0;const double target=std::tan((angle-base)*.5),slope=(target-e.coordinate)/(mid-e.radius);if(!std::isfinite(slope))continue;candidates.push_back({e.radius,e.coordinate,slope,angle,e.reciprocal});
   for(const auto& other:events){if(other.radius==e.radius||(other.radius!=a&&other.radius!=b)||other.reciprocal!=e.reciprocal)continue;const double s=(other.coordinate-e.coordinate)/(other.radius-e.radius);candidates.push_back({e.radius,e.coordinate,s,angle,e.reciprocal});}}
  bool found=false;for(auto ray:candidates){auto u=ray.unit(mid);double actual=std::arg(u);actual=angle+std::remainder(actual-angle,period);if(!(actual>component_lo+1e-12&&actual<component_hi-1e-12))continue;
   if(!moving_ray_sign_with_folds(p,tape,a,b,safe_a,safe_b,ray,positive,events))continue;ray.angle=actual-period*std::floor(actual/period);result.push_back(ray);found=true;break;}
  if(!found)return std::nullopt;
 }
 std::sort(result.begin(),result.end(),[](const auto& x,const auto& y){return x.angle<y.angle;});return result;
}

}

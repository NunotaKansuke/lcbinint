#pragma once
#include "shape_transport.hpp"
#include "patterson_rule.hpp"
#include "patterson_embedded_rule.hpp"
#include "mixed_fold_rule.hpp"
#include "gauss_kronrod_rule.hpp"
#include "fused_integrator.hpp"
#ifdef LCBININT_TRIPLE_BOUNDARY_CONTROL
#include "boundary_control_atlas.hpp"
#endif
#include <map>
#include <boost/container/flat_map.hpp>
#include <tuple>
#ifdef LCBININT_TRIPLE_WINDING_ARCS
#include "ray_winding_labels.hpp"
#endif
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
#include "angular_sector_certificate.hpp"
#endif
#ifdef LCBININT_TRIPLE_COMPLEX_PRODUCT_RULE
#include "complex_fold_product_rule.hpp"
#endif
namespace lcbinint::algebraic_boundary::triple {
#ifdef LCBININT_TRIPLE_KRONROD_STREAM
inline constexpr int first_order_minimum=3;
#else
inline constexpr int first_order_minimum=3;
#endif
inline double first_order_chart_width(bool uniform){
#ifdef LCBININT_TRIPLE_WIDE_CHARTS
 return uniform?6.:2.;
#else
 // Exterior/complex fold coordinates can have nearly flat flux while
 // boundary transport stays sharp. Bound each uniform chart geometrically;
 // the adaptive controller continues to observe scalar flux only.
 return uniform?1.5:2.;
#endif
}
struct PattersonTuning {double angular_scale=1;int minimum_radial=0,critical_order=6;double selected_lo=-INFINITY,selected_hi=INFINITY,chart_width=0;bool quad_uniform=false;};
#ifdef LCBININT_TRIPLE_DIAGNOSTIC_TUNING
inline thread_local PattersonTuning patterson_tuning;
struct PattersonRecord {double lo,hi,radial_error,inner_error;int level;std::size_t points;RadiusIntegral<true> value;std::vector<std::array<double,5>> cells;};
inline thread_local std::vector<PattersonRecord> patterson_records;
inline PattersonTuning active_patterson_tuning(){return patterson_tuning;}
#else
inline PattersonTuning active_patterson_tuning(){return {};}
#endif
// A gradient-owned stream: no completed scalar integral or tangent replay.
// The proved band supplies the root count. Successful angular endpoints and
// complete value/tangent packets are retained as the radial rule grows.
template<bool Gradient,bool Fejer=false> class FirstOrderPanel {
 static_assert(Gradient,"stream panel is a gradient-specific implementation");
 // Quadrature family is independent of the first-order physical transport.
 // Both families retain the same complete value/tangent packets.
 static const auto& first_order_rule(int level){
  if constexpr(Fejer)return fejer_rule(level);
  else return patterson_rule(level);
 }

#ifdef LCBININT_TRIPLE_KRONROD_STREAM
 static constexpr bool high_degree_=true;
#else
 static constexpr bool high_degree_=false;
#endif
 bool use_kronrod()const{
#if defined(LCBININT_TRIPLE_KRONROD_LIMB_ONLY)
  return high_degree_&&!uniform_&&!weighted_;
#elif defined(LCBININT_TRIPLE_KRONROD_UNIFORM_ONLY)
  return high_degree_&&uniform_&&!weighted_;
#else
  return high_degree_&&!weighted_;
#endif
 }
 // This is a coordinate-regularity guard, independent of flux convergence.
 // Tightening the flux request must not force a second gradient controller.
 double geometry_resolution_budget()const{
  if(shared_&&shared_->requested_relative_tolerance>0)return relative_*std::max(1.,1e-3/shared_->requested_relative_tolerance);
  return relative_;
 }
 bool controlled_uniform()const{
#ifdef LCBININT_TRIPLE_BOUNDARY_CONTROL
  return uniform_&&context_->control&&context_->control->valid();
#else
  return false;
#endif
 }
 using Packet=shape_gradient_detail::RadiusResult;
 using Anchor=shape_gradient_detail::RadiusAnchor;
 struct Point {Packet value;Anchor anchor;AngularTopology topology;double angular_error=0;std::array<double,8> motion{};bool value_ready=false;std::unique_ptr<shape_gradient_detail::GeometryCores> cores;
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
 std::array<int,2> endpoint_labels{-1,-1};
#endif
#ifdef LCBININT_TRIPLE_WINDING_ARCS
 mutable bool winding_ready=false;mutable std::array<int,3> winding_phase{},lo_rank{},hi_rank{};
#endif
};
 struct Context {
  // Sorted contiguous keys, with stable pointees during insertions.
  boost::container::flat_map<double,std::unique_ptr<Point>> points;
  Context(){points.reserve(32);}
#ifdef LCBININT_TRIPLE_BOUNDARY_CONTROL
std::shared_ptr<const BoundaryControlAtlas> control;
#endif
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
 std::map<std::tuple<double,int,std::array<double,8>>,Point> arc_packets;
#endif
 std::map<std::array<double,2>,shape_gradient_detail::RadialComplexPair> radial_roots;std::shared_ptr<std::unique_ptr<root_pair_detail::RadialCache<8>>> pairs;ShapeGradientResult work;std::shared_ptr<const point_polynomial_detail::Cache<double>> stationary_coefficients;};
 std::shared_ptr<Context> context_=std::make_shared<Context>();bool owns_context_=true;
 Geometry p_;FoldRadialMap map_;AngularOptions options_;std::shared_ptr<SharedRadiusCache<Gradient>> shared_;
 double relative_,absolute_,uniform_weight_=1,limb_weight_=0,origin_=0,safe_lo_=0,safe_hi_=0;
 bool uniform_,weighted_,moving_=false,charts_checked_=false,mixed_=false,mixed_declined_=false,geometry_probe_=false,discovery_retry_=false;
 mutable bool trace_active_=true;
 std::array<double,8> base_{},slope_{};
 std::optional<AngularTopology> band_;std::vector<RadialEvent> events_;
 struct FoldHint {double radius,theta,square_slope,curvature;};std::vector<FoldHint> hints_;
 RadiusIntegral<true> accepted_;int level_=0,angular_level_=0;double radial_error_=INFINITY,inner_error_=0;
 // Hermite interpolation is only a root predictor. Every predicted endpoint
 // still passes physical Newton, orientation, membership and disjointness.
 struct Curve {int n=0,count=0;std::array<double,7> x{},w{},d{};std::array<std::array<std::array<double,4>,3>,7> data{};};
 struct Cell {int kind=0;FoldRadialMap coordinates;double center=0,scale=1;std::shared_ptr<Curve> curve;std::array<double,8> center_motion{},scale_motion{},lower_motion{},upper_motion{};bool material=false;
 RadiusIntegral<true> accepted{};double error=INFINITY,inner=0,detail=0;int level=0;
 };
 std::vector<Cell> cells_;
 std::vector<std::array<double,2>> previous_cells_;
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
 int arc_sector_=-1;bool lanes_checked_=false,gap_lanes_=false,whole_gap_background_=false,geometry_guard_=false;std::vector<double> sector_cuts_,discovery_radii_;std::vector<sector_certificate_detail::MovingRay> sector_rays_;
 std::vector<std::unique_ptr<FirstOrderPanel>> lanes_;
#ifdef LCBININT_TRIPLE_WINDING_ARCS
 std::shared_ptr<winding_label_detail::Plan> winding_;int reference_lo_=0,reference_hi_=0;std::array<std::pair<double,unsigned>,2> fold_masks_{};unsigned fold_masks_count_=0;
 static int cyclic(int i,int n){return (i%n+n)%n;}
 bool label_point(const Point& point)const{
  if(point.winding_ready)return true;if(!winding_||!point.cores)return false;const int n=int(point.anchor.count);if(n<1||n>3)return false;const auto indices=winding_->indices(point.anchor.radius,&context_->work.quad_samples);if(!indices)return false;point.winding_phase=*indices;
  const Complex origin=std::conj(winding_->unit);for(int side=0;side<2;++side){std::array<std::pair<Complex,int>,3> order;
   for(int i=0;i<n;++i){Complex unit=origin*(*point.cores)[i][side].w;const double eps=64*std::numeric_limits<double>::epsilon();if(unit.real()>0&&unit.imag()*unit.imag()<eps*eps*std::norm(unit)){const bool below=side==0?(*indices)[2]>0:(*indices)[2]<0;unit={1,below?-eps:eps};}order[i]={unit,i};}
   std::sort(order.begin(),order.begin()+n,[](const auto& a,const auto& b){return sector_certificate_detail::ray_order(a.first,b.first);});for(int rank=0;rank<n;++rank)(side?point.hi_rank:point.lo_rank)[order[rank].second]=rank;
  }point.winding_ready=true;return true;
 }
 int arc_label(const Point& point,int i)const{return cyclic(point.lo_rank[i]-point.winding_phase[0]+reference_lo_,int(point.anchor.count));}
 int gap_label(const Point& point,int i,bool before=false)const{return cyclic(point.hi_rank[i]-(before?1:0)-point.winding_phase[1]+reference_hi_,int(point.anchor.count));}
 bool endpoint_selected(const Point& point,int i,bool upper)const{if(!label_point(point))return false;return gap_label(point,i,!upper)==arc_sector_;}
 void winding_fold_masks(){
  fold_masks_count_=0;const double period=2*std::acos(-1.);
  for(const auto& event:events_){if(event.radius!=map_.a&&event.radius!=map_.b)continue;if(fold_masks_count_==2)break;bool overlap=false;for(const auto& other:events_)if(&other!=&event&&other.radius_lo<=event.radius_hi&&other.radius_hi>=event.radius_lo)overlap=true;if(overlap)continue;
   const bool left=event.radius==map_.a;const auto f=event_detail::physical(event.radius,event.coordinate,p_,event.reciprocal);if(!f||f->r==0)continue;const bool positive=(left?f->r:-f->r)>0;sector_certificate_detail::MovingRay ray{event.radius,event.coordinate,0,0,event.reciprocal};double probe=.5*(map_.a+map_.b);bool proved=false;
   for(int attempt=0;attempt<7;++attempt){if(sector_certificate_detail::fold_ray_collar(p_,ray,event,probe,left,positive,left?safe_lo_:safe_hi_)){proved=true;break;}probe=event.radius+(probe-event.radius)*.5;}if(!proved)continue;
   Point* point=nullptr;if(!geometry_point(probe,point)||!label_point(*point))continue;const int n=int(point->anchor.count);double theta=std::arg(ray.unit(probe));unsigned mask=0;int matches=0;
   if(positive){for(int i=0;i<n;++i){const auto& a=point->anchor.arcs[i];const double mid=.5*(a.lo+a.hi),angle=mid+std::remainder(theta-mid,period);if(angle>a.lo+1e-12&&angle<a.hi-1e-12){++matches;if(gap_lanes_){mask|=1u<<gap_label(*point,i);mask|=1u<<gap_label(*point,i,true);}else mask|=1u<<arc_label(*point,i);}}}
   else{for(int i=0;i<n;++i){int next=-1;for(int j=0;j<n;++j)if(point->hi_rank[j]==(point->hi_rank[i]+1)%n)next=j;if(next<0)continue;const double lo=point->anchor.arcs[i].hi;double width=point->anchor.arcs[next].lo-lo;while(width<=0)width+=period;const double mid=lo+.5*width,angle=mid+std::remainder(theta-mid,period);if(angle>lo+1e-12&&angle<lo+width-1e-12){++matches;if(gap_lanes_)mask|=1u<<gap_label(*point,i);else{mask|=1u<<arc_label(*point,i);mask|=1u<<arc_label(*point,next);}}}}
   if(matches==1)fold_masks_[fold_masks_count_++]={event.radius,mask};
  }
 }
#endif
 int sector_id(double R,Complex unit)const{return sector_rays_.empty()?sector_certificate_detail::sector(sector_cuts_,std::arg(unit)):sector_certificate_detail::sector(sector_rays_,R,unit);}
 bool relevant_fold(double R,int sector)const{
#ifdef LCBININT_TRIPLE_WINDING_ARCS
  if(winding_){for(unsigned j=0;j<fold_masks_count_;++j)if(fold_masks_[j].first==R)return bool(fold_masks_[j].second&(1u<<sector));return true;}
#endif
  if(sector_rays_.empty())return true;bool found=false;
  for(const auto& e:events_)if(e.radius==R){found=true;const double t=e.coordinate,sign=e.reciprocal?-1.:1.;const Complex unit=sign*Complex(1-t*t,2*t);
   for(const auto& cut:sector_rays_){const Complex u=cut.unit(R);const double cross=u.real()*unit.imag()-u.imag()*unit.real();if(std::abs(cross)<1e-9*std::sqrt(std::norm(u)*std::norm(unit)))return true;}
   if(sector_id(R,unit)==sector)return true;
  }return !found;
 }
 int selected_arc(const Point& point)const{
#ifdef LCBININT_TRIPLE_WINDING_ARCS
  if(winding_){if(!label_point(point))return -1;for(int i=0;i<int(point.anchor.count);++i)if(arc_label(point,i)==arc_sector_)return i;return -1;}
#endif
  int index=-1;for(std::size_t i=0;i<point.anchor.count;++i)if(sector_id(point.anchor.radius,point.cores?(*point.cores)[i][2].w:std::polar(1.,.5*(point.anchor.arcs[i].lo+point.anchor.arcs[i].hi)))==arc_sector_){if(index>=0)return -1;index=int(i);}return index;
 }
 std::unique_ptr<FirstOrderPanel> lane(double a,double b,bool left,bool right,int sector){
  const bool had_left=left,had_right=right;
#if defined(LCBININT_TRIPLE_FOLD_SECTORS) || defined(LCBININT_TRIPLE_WINDING_ARCS)
  left=left&&relevant_fold(a,sector);right=right&&relevant_fold(b,sector);
#endif
  auto out=std::make_unique<FirstOrderPanel>(p_,a,b,left,right,options_,shared_,absolute_,relative_,uniform_,false,limb_weight_,true);
  out->context_=context_;out->owns_context_=false;out->arc_sector_=sector;out->sector_cuts_=sector_cuts_;out->sector_rays_=sector_rays_;out->gap_lanes_=gap_lanes_;out->whole_gap_background_=whole_gap_background_;out->geometry_guard_=geometry_guard_||(had_left&&!left)||(had_right&&!right);
#ifdef LCBININT_TRIPLE_WINDING_ARCS
 out->winding_=winding_;out->reference_lo_=reference_lo_;out->reference_hi_=reference_hi_;out->fold_masks_=fold_masks_;out->fold_masks_count_=fold_masks_count_;
#endif
out->events_=events_;out->hints_=hints_;out->origin_=origin_;out->base_=base_;out->slope_=slope_;out->moving_=moving_;out->band_=band_;out->safe_lo_=std::max(a,safe_lo_);out->safe_hi_=std::min(b,safe_hi_);return out;
 }
 bool start_lanes(int requested,int angular){
  if(lanes_checked_||arc_sector_>=0)return false;lanes_checked_=true;
  if(!band_||!moving_||weighted_||band_->arcs.size()<2||band_->arcs.size()>3)return false;
#ifdef LCBININT_TRIPLE_WINDING_ARCS
  if(events_.empty())return false;
#endif
#ifdef LCBININT_TRIPLE_COMPLEMENT_ARCS
  if(!uniform_)return false;
#endif
  if(!context_->stationary_coefficients){if(shared_&&shared_->coefficients){if(!shared_->coefficients->point_signs)shared_->coefficients->point_signs=std::make_shared<point_polynomial_detail::Cache<double>>(p_);context_->stationary_coefficients=shared_->coefficients->point_signs;}else context_->stationary_coefficients=std::make_shared<point_polynomial_detail::Cache<double>>(p_);}
  Point* center=nullptr;if(!geometry_point(.5*(map_.a+map_.b),center))return false;
#ifdef LCBININT_TRIPLE_WINDING_ARCS
  if(shared_&&shared_->winding_labels)winding_=shared_->winding_labels;
  else{double lo=INFINITY,hi=0;for(const auto& e:events_){lo=std::min(lo,e.radius_lo);hi=std::max(hi,e.radius_hi);}winding_=std::make_shared<winding_label_detail::Plan>(p_,*context_->stationary_coefficients,lo,hi);if(shared_)shared_->winding_labels=winding_;}
  if(!winding_->covers(map_.a,map_.b)||!label_point(*center))return false;reference_lo_=center->winding_phase[0];reference_hi_=center->winding_phase[1];gap_lanes_=uniform_;whole_gap_background_=gap_lanes_;sector_cuts_.resize(center->anchor.count);for(std::size_t i=0;i<sector_cuts_.size();++i)sector_cuts_[i]=double(i);winding_fold_masks();
#elif defined(LCBININT_TRIPLE_FOLD_SECTORS)
  auto proof=sector_certificate_detail::certify_following_folds(p_,center->topology,center->anchor,map_.a,map_.b,safe_lo_,safe_hi_,*context_->stationary_coefficients,uniform_,events_);gap_lanes_=uniform_;whole_gap_background_=gap_lanes_;
  if(!proof&&uniform_){proof=sector_certificate_detail::certify_following_folds(p_,center->topology,center->anchor,map_.a,map_.b,safe_lo_,safe_hi_,*context_->stationary_coefficients,false,events_);gap_lanes_=false;}
  if(!proof)return false;sector_rays_=*proof;sector_cuts_.clear();for(const auto& ray:*proof)sector_cuts_.push_back(ray.angle);
#elif defined(LCBININT_TRIPLE_COMPLEMENT_ARCS)
  const auto proof=sector_certificate_detail::certify(p_,center->topology,map_.a,map_.b,*context_->stationary_coefficients,true);gap_lanes_=true;
#else
  const auto proof=sector_certificate_detail::certify(p_,center->topology,map_.a,map_.b,*context_->stationary_coefficients);
#endif
#if !defined(LCBININT_TRIPLE_FOLD_SECTORS) && !defined(LCBININT_TRIPLE_WINDING_ARCS)
  if(!proof)return false;sector_cuts_=*proof;
#endif
  for(int sector=0;sector<int(sector_cuts_.size());++sector)lanes_.push_back(lane(map_.a,map_.b,map_.left,map_.right,sector));return true;
 }
 void collect_lanes(){
  accepted_={};radial_error_=inner_error_=0;for(const auto& child:lanes_){const auto& r=child->result();accepted_.f0+=r.f0;accepted_.fh+=r.fh;accepted_.endpoint_error+=r.endpoint_error;for(int k=0;k<8;++k){accepted_.df0[k]+=r.df0[k];accepted_.dfh[k]+=r.dfh[k];}radial_error_+=child->radial_error();inner_error_+=child->inner_error();}
  if(gap_lanes_&&whole_gap_background_){const double pi=std::acos(-1.),background=pi*(map_.b-map_.a)*(map_.a+map_.b);accepted_.f0+=background;for(int j=0;j<8;++j){const double a=base_[j]+slope_[j]*(map_.a-origin_),b=base_[j]+slope_[j]*(map_.b-origin_);accepted_.df0[j]+=2*pi*(map_.b*b-map_.a*a);}radial_error_+=32*std::numeric_limits<double>::epsilon()*std::abs(background);}
  accepted_.angular_error=radial_error_+inner_error_;reason="stream_arc_lanes_completed";
 }
 bool uniform_gap(Point& value,const Point& geometry,double R,const std::array<double,8>& motion){
  const std::size_t n=geometry.anchor.count;const double period=2*std::acos(-1.);std::array<std::pair<double,std::size_t>,3> order;
  for(std::size_t i=0;i<n;++i){double mid=.5*(geometry.anchor.arcs[i].lo+geometry.anchor.arcs[i].hi);mid-=period*std::floor(mid/period);order[i]={mid,i};}std::sort(order.begin(),order.begin()+n);
  int selected=-1;std::size_t l=0,r=0;double width=0;
  for(std::size_t j=0;j<n;++j){const std::size_t li=order[j].second,ri=order[(j+1)%n].second;const auto& a=geometry.anchor.arcs[li];const auto& b=geometry.anchor.arcs[ri];const double lo=order[j].first+.5*(a.hi-a.lo),hi=order[(j+1)%n].first-.5*(b.hi-b.lo)+(j+1==n?period:0);if(!(hi>lo))return false;Complex middle;if(geometry.cores){middle=(*geometry.cores)[li][1].w+(*geometry.cores)[ri][0].w;if(hi-lo>std::acos(-1.))middle=-middle;}if(std::norm(middle)<1e-24)middle=std::polar(1.,.5*(lo+hi));bool chosen;
#ifdef LCBININT_TRIPLE_WINDING_ARCS
 if(winding_){if(!label_point(geometry))return false;chosen=gap_label(geometry,int(li))==arc_sector_;}else
#endif
 chosen=sector_id(R,middle)==arc_sector_;if(chosen){if(selected>=0)return false;selected=int(j);l=li;r=ri;width=hi-lo;}}
  if(selected<0||!geometry.cores)return false;value.endpoint_labels={int(2*l+1),int(2*r)};shape_gradient_detail::Ray ray(p_,R);shape_gradient_detail::Endpoint left,right;const auto& a=geometry.anchor.arcs[l];const auto& b=geometry.anchor.arcs[r];
  if(!shape_gradient_detail::endpoint_impl(ray,R,p_,a.hi,width,left,context_->work,nullptr,true,a.wide_hi?&a.precise_hi:nullptr,&(*geometry.cores)[l][1]))return false;
  if(!shape_gradient_detail::endpoint_impl(ray,R,p_,b.lo,width,right,context_->work,nullptr,true,b.wide_lo?&b.precise_lo:nullptr,&(*geometry.cores)[r][0]))return false;
  width+=(right.theta-b.lo)-(left.theta-a.hi);value.value.f0=-R*width;value.value.endpoint_error=R*(left.error+right.error);
  for(int j=0;j<8;++j)value.value.uniform[j]=-motion[j]*width-R*(right.motion[j]-left.motion[j]+(right.radial_motion-left.radial_motion)*motion[j]);return true;
 }
 bool lane_point(double R,const std::array<double,8>& motion,Point*& output,const AngularTopology* predicted){
  const auto key=std::make_tuple(R,arc_sector_,motion);const auto cached=context_->arc_packets.find(key);if(cached!=context_->arc_packets.end()){output=&cached->second;return true;}
  Point* geometry=nullptr;if(!geometry_point(R,geometry,predicted))return false;const int index=gap_lanes_?-1:selected_arc(*geometry);if(!gap_lanes_&&index<0){reason="stream_arc_sector_identity";return false;}
  Point value;value.topology=geometry->topology;value.anchor=geometry->anchor;if(index>=0)value.endpoint_labels={2*index,2*index+1};const double gate=std::max(1e-12,relative_*.1*active_patterson_tuning().angular_scale);
  if(gap_lanes_){if(!uniform_gap(value,*geometry,R,motion)){reason="stream_gap_transport";return false;}}
  else if(!shape_gradient_detail::radius(R,p_,value.topology,!uniform_,4,motion,value.value,context_->work,R,&value.anchor,context_->pairs.get(),gate,&value.angular_error,true,nullptr,active_patterson_tuning().critical_order,geometry->cores.get(),nullptr,index)){reason="stream_arc_transport";return false;}
  value.motion=motion;value.value_ready=true;output=&context_->arc_packets.emplace(key,std::move(value)).first->second;return true;
 }
#endif
 double canonical_radius(double R)const{
  if(R<=safe_lo_)R=std::nextafter(safe_lo_,map_.b);
  if(R>=safe_hi_)R=std::nextafter(safe_hi_,map_.a);
  return R;
 }
 std::array<double,2> mapped(const Cell& cell,double x)const{
  auto t=cell.coordinates(x);if(!cell.kind||cell.kind==4){t[0]=canonical_radius(t[0]);return t;}
  if(cell.kind==1)return {canonical_radius(cell.center+cell.scale*std::sinh(t[0])),cell.scale*std::cosh(t[0])*t[1]};
  const double sh=std::sinh(t[0]*.5),d=2*cell.scale*sh*sh;
  return {canonical_radius(cell.kind==2?cell.center+d:cell.center-d),cell.scale*std::sinh(t[0])*t[1]};
 }
 struct MappedTransport {std::array<double,2> node;std::array<std::array<double,8>,2> material{};};
 MappedTransport mapped_transport(const Cell& cell,double x)const{
  MappedTransport out;const auto u=cell.coordinates(x);out.node=u;double sh=0,ch=0,half=0;
  if(cell.kind==1){sh=std::sinh(u[0]);ch=std::cosh(u[0]);out.node={cell.center+cell.scale*sh,cell.scale*ch*u[1]};}
  else if(cell.kind==2||cell.kind==3){half=std::sinh(u[0]*.5);sh=std::sinh(u[0]);if(cell.material)ch=std::cosh(u[0]);const double d=2*cell.scale*half*half;out.node={cell.kind==2?cell.center+d:cell.center-d,cell.scale*sh*u[1]};}
  out.node[0]=canonical_radius(out.node[0]);
  for(int j=0;j<8;++j){out.material[0][j]=base_[j]+slope_[j]*(out.node[0]-origin_);out.material[1][j]=slope_[j];}
  if(!cell.kind||cell.kind==4||!cell.material)return out;
  const double span=cell.coordinates.b-cell.coordinates.a,fraction=(u[0]-cell.coordinates.a)/span,sign=cell.kind==2?1:-1;
  for(int j=0;j<8;++j){const double up=cell.lower_motion[j]+fraction*(cell.upper_motion[j]-cell.lower_motion[j]),stretch=(cell.upper_motion[j]-cell.lower_motion[j])/span;
   if(cell.kind==1){out.material[0][j]=cell.center_motion[j]+cell.scale_motion[j]*sh+cell.scale*ch*up;out.material[1][j]=cell.scale_motion[j]/cell.scale+sh/ch*up+stretch;}
   else{out.material[0][j]=cell.center_motion[j]+sign*(2*cell.scale_motion[j]*half*half+cell.scale*sh*up);out.material[1][j]=cell.scale_motion[j]/cell.scale+ch/sh*up+stretch;}
  }return out;
 }
 struct RuleView {const std::vector<double>& x;const std::vector<double>& w;const std::vector<double>& interp;};
 RuleView integration_rule(int level,bool weighted,bool limb=false)const{
  if(mixed_){const auto& q=mixed_fold_rule(level);return {q.x,q.w,q.interp};}
  if(weighted){const auto& q=limb?linear_radial_rule(level):radial_rule(level,true);return {q.x,q.w,q.interp};}
  if(use_kronrod()){const auto& q=gradient_kronrod_rule(level);return {q.x,q.w,q.interp};}
  const auto& q=first_order_rule(level);return {q.x,q.w,q.interp};
 }

 bool build_curve(Cell& cell,int level,bool discovery=false){
  const auto ordinary_rule=integration_rule(level,weighted_&&cell.kind==0);const auto& seed_rule=first_order_rule(2);const RuleView rule=discovery?RuleView{seed_rule.x,seed_rule.w,seed_rule.interp}:ordinary_rule;if(rule.x.size()>7)return false;
  auto curve=std::make_shared<Curve>();curve->n=int(rule.x.size());
  for(int k=0;k<curve->n;++k){const auto node=mapped(cell,rule.x[k]);auto it=context_->points.find(node[0]);if(it==context_->points.end()||it->second->anchor.radius!=node[0])return false;const auto& anchor=it->second->anchor;
   if(!anchor.count||anchor.count>3||(k&&int(anchor.count)!=curve->count))return false;curve->count=int(anchor.count);curve->x[k]=rule.x[k];
   std::array<bool,3> used{};
   for(int a=0;a<curve->count;++a){int selected=a;double shift=0;
    if(k){double best=INFINITY;for(int b=0;b<curve->count;++b)if(!used[b]){const auto& arc=anchor.arcs[b];const double mean=.5*(arc.lo+arc.hi),ref=curve->data[k-1][a][0],delta=std::remainder(mean-ref,2*std::acos(-1.));if(std::abs(delta)<best){best=std::abs(delta);selected=b;shift=ref+delta-mean;}}}
    used[selected]=true;const auto& arc=anchor.arcs[selected];const double jac=(cell.kind==3?-1:1)*node[1];
    curve->data[k][a]={.5*(arc.lo+arc.hi)+shift,.5*(arc.hi-arc.lo),.5*jac*(arc.dlo+arc.dhi),.5*jac*(arc.dhi-arc.dlo)};
   }
  }
  double largest=0;for(int j=0;j<curve->n;++j){double w=1,d=0;for(int k=0;k<curve->n;++k)if(k!=j){const double dx=curve->x[j]-curve->x[k];w/=dx;d+=1/dx;}curve->w[j]=w;curve->d[j]=d;largest=std::max(largest,std::abs(w));}for(int j=0;j<curve->n;++j)curve->w[j]/=largest;
  cell.curve=std::move(curve);return true;
 }
 bool predict(const Cell& cell,double x,AngularTopology& result)const{
  if(!cell.curve||!band_)return false;const auto& c=*cell.curve;std::array<double,7> l{};double sum=0;
  for(int k=0;k<c.n;++k){const double dx=x-c.x[k];if(std::abs(dx)<8*std::numeric_limits<double>::epsilon())return false;l[k]=c.w[k]/dx;sum+=l[k];}if(!std::isfinite(sum)||sum==0)return false;
  result=*band_;result.arcs.resize(c.count);
  for(int a=0;a<c.count;++a){double mean=0,half=0;for(int k=0;k<c.n;++k){const double dx=x-c.x[k],lk=l[k]/sum,h=(1-2*dx*c.d[k])*lk*lk,q=dx*lk*lk;mean+=h*c.data[k][a][0]+q*c.data[k][a][2];half+=h*c.data[k][a][1]+q*c.data[k][a][3];}if(!(half>0&&std::isfinite(mean)&&std::isfinite(half)))return false;result.arcs[a]={mean-half,mean+half};}return true;
 }
 // Test the retained seven positions/slopes against three integration-by-
 // parts identities. No additional endpoint solves or parameter jets occur.
 bool spatially_resolved(const Cell& cell)const{
  if(!cell.curve||cell.curve->n!=7)return false;
  const auto& c=*cell.curve;const auto& q=first_order_rule(3);
  struct Weights{std::array<std::array<double,7>,3>a{},b{};};
  static const auto weights=[](){
   Weights w;const auto& q=first_order_rule(3);
   std::array<double,7> bary{},plus{},minus{};double sp=0,sm=0;
   for(int j=0;j<7;++j){
    bary[j]=1;for(int k=0;k<7;++k)if(k!=j)bary[j]/=q.x[j]-q.x[k];
    plus[j]=bary[j]/(1-q.x[j]);minus[j]=bary[j]/(-1-q.x[j]);
    sp+=plus[j];sm+=minus[j];
   }
   for(int j=0;j<7;++j){
    const double x=q.x[j],powers[]={1,x,x*x};
    for(int k=0;k<3;++k){
     w.a[k][j]=q.w[j]*powers[k];
     w.b[k][j]=(k?k*q.w[j]*powers[k-1]:0)-plus[j]/sp+(k%2?-1:1)*minus[j]/sm;
    }
   }
   return w;
  }();
  for(int arc=0;arc<c.count;++arc){
   double width=0,motion=0;
   for(int j=0;j<7;++j){
    width+=q.w[j]*2*c.data[j][arc][1];
    motion+=q.w[j]*(std::abs(c.data[j][arc][2])+std::abs(c.data[j][arc][3]));
   }
   const double scale=std::max(width,motion);double mismatch=0;
   for(int k=0;k<3;++k)for(int side=0;side<2;++side){
    double residual=0;const double origin=c.data[3][arc][side];
    for(int j=0;j<7;++j)
     residual+=weights.a[k][j]*c.data[j][arc][side+2]+weights.b[k][j]*(c.data[j][arc][side]-origin);
    mismatch=std::max(mismatch,std::abs(residual));
   }
   if(mismatch>4*geometry_resolution_budget()*scale+
      128*std::numeric_limits<double>::epsilon()*(1+scale))return false;
  }
  return true;
 }
 bool proposal(double R,AngularTopology& topology,double& seed_radius,Anchor& anchor){
  seed_radius=R;anchor={};
  const auto exact=context_->points.find(R);if(exact!=context_->points.end()){topology=exact->second->topology;anchor=exact->second->anchor;return true;}
  auto it=context_->points.lower_bound(R);
  if(it!=context_->points.end() || it!=context_->points.begin()){
   auto seed=it;if(seed==context_->points.end() || (seed!=context_->points.begin()&&std::abs(std::prev(seed)->first-R)<std::abs(seed->first-R)))--seed;
   topology=seed->second->topology;seed_radius=seed->first;anchor=seed->second->anchor;
  }else if(band_){topology=*band_;seed_radius=map_.a+(map_.b-map_.a)*.5;}
  else {topology=angular_topology(R,p_,options_,shared_?shared_->coefficients.get():nullptr);++context_->work.global_radii;}
  if(!topology.completed){reason="stream_topology_missing";return false;}

  if(anchor.count&&seed_radius!=R){
   struct Root {std::size_t arc;bool high;double theta,radial;};std::vector<Root> roots;
   for(std::size_t i=0;i<anchor.count;++i){const auto& a=anchor.arcs[i];roots.push_back({i,false,a.lo,a.dlo});roots.push_back({i,true,a.hi,a.dhi});const double width=a.hi-a.lo,dl=a.dlo*(R-seed_radius),dh=a.dhi*(R-seed_radius);if(std::abs(dl)<width*.25)topology.arcs[i].lo+=dl;if(std::abs(dh)<width*.25)topology.arcs[i].hi+=dh;}
   auto ordered_hints=hints_;std::sort(ordered_hints.begin(),ordered_hints.end(),[&](auto a,auto b){return std::abs(R-a.radius)<std::abs(R-b.radius);});
   for(const auto& hint:ordered_hints){const double old=seed_radius-hint.radius,distance=R-hint.radius;if(!(old*distance>0&&std::abs(distance)<16*std::abs(old)&&hint.square_slope*old>0))continue;
    auto ranked=roots;for(auto& root:ranked)root.theta=hint.theta+std::remainder(root.theta-hint.theta,2*std::acos(-1.));std::sort(ranked.begin(),ranked.end(),[&](auto a,auto b){return std::abs(a.theta-hint.theta)<std::abs(b.theta-hint.theta);});if(ranked.size()<2)continue;
    const double reach=2*std::sqrt(hint.square_slope*old),far=std::max(std::abs(ranked[0].theta-hint.theta),std::abs(ranked[1].theta-hint.theta));if(far>reach||(ranked.size()>2&&std::abs(ranked[2].theta-hint.theta)<1.5*far))continue;
    auto a=ranked[0],b=ranked[1];if(a.theta>b.theta)std::swap(a,b);const double mean=(a.theta+b.theta)*.5+.5*(a.radial+b.radial)*(R-seed_radius),half=(b.theta-a.theta)*.5*std::sqrt(distance/old);
    for(auto proposal:std::array<std::pair<Root,double>,2>{{{a,mean-half},{b,mean+half}}}){auto& angle=proposal.first.high?topology.arcs[proposal.first.arc].hi:topology.arcs[proposal.first.arc].lo;angle+=std::remainder(proposal.second-angle,2*std::acos(-1.));}break;
   }
   // All guesses now belong to the requested radius. The physical Newton,
   // orientation and disjointness checks still decide local acceptance.
   seed_radius=R;anchor={};
  }

  return true;
 }
 bool point(double R,Point*& output,const AngularTopology* predicted=nullptr,const std::array<double,8>* transported_motion=nullptr){
  if(R<=safe_lo_)R=std::nextafter(safe_lo_,map_.b);if(R>=safe_hi_)R=std::nextafter(safe_hi_,map_.a);
  std::array<double,8> motion;for(int j=0;j<8;++j)motion[j]=transported_motion?(*transported_motion)[j]:base_[j]+slope_[j]*(R-origin_);
  if(controlled_uniform())motion.fill(0);
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
  if(arc_sector_>=0)return lane_point(R,motion,output,predicted);
#endif
  auto found=context_->points.find(R);if(found!=context_->points.end()&&found->second->value_ready&&found->second->motion==motion){output=found->second.get();return true;}
  AngularTopology topology;double seed_radius=R;Anchor anchor;if(predicted&&found==context_->points.end())topology=*predicted;else if(!proposal(R,topology,seed_radius,anchor))return false;

  Point value;const double gate=std::max(1e-12,relative_*(use_kronrod()?.03:.1)*active_patterson_tuning().angular_scale);
#ifdef LCBININT_TRIPLE_CERTIFIED_STATIONARY
  if(!uniform_&&!context_->stationary_coefficients){if(shared_&&shared_->coefficients){if(!shared_->coefficients->point_signs)shared_->coefficients->point_signs=std::make_shared<point_polynomial_detail::Cache<double>>(p_);context_->stationary_coefficients=shared_->coefficients->point_signs;}else context_->stationary_coefficients=std::make_shared<point_polynomial_detail::Cache<double>>(p_);}
#endif
  bool completed=shape_gradient_detail::radius(R,p_,topology,!uniform_,4,motion,value.value,context_->work,seed_radius,&anchor,context_->pairs.get(),gate,&value.angular_error,true,nullptr,active_patterson_tuning().critical_order,found!=context_->points.end()?found->second->cores.get():nullptr,context_->stationary_coefficients.get());
  if(!completed){
   LCBININT_COUPLED_SCOPE(topology);
   auto proposals=topology;proposals.roots.clear();for(const auto& arc:topology.arcs){for(double angle:{arc.lo,arc.hi}){while(angle<-std::acos(-1.)*.5)angle+=2*std::acos(-1.);while(angle>std::acos(-1.)*1.5)angle-=2*std::acos(-1.);proposals.roots.push_back({angle,angle});}}std::sort(proposals.roots.begin(),proposals.roots.end(),[](auto a,auto b){return a.lo<b.lo;});
   auto repaired=continued_topology(R,p_,proposals,options_,int(topology.roots.size()),shared_?shared_->coefficients.get():nullptr);
   if(!repaired.completed){repaired=angular_topology(R,p_,options_,shared_?shared_->coefficients.get():nullptr);++context_->work.global_radii;}else ++context_->work.continued_radii;
   value={};anchor={};if(!repaired.completed){reason="stream_local_repair";return false;}
   if(!shape_gradient_detail::radius(R,p_,repaired,!uniform_,4,motion,value.value,context_->work,R,&anchor,context_->pairs.get(),gate,&value.angular_error,true,nullptr,active_patterson_tuning().critical_order)){
    value={};anchor={};if(!shape_gradient_detail::radius(R,p_,repaired,!uniform_,4,motion,value.value,context_->work,R,&anchor,context_->pairs.get(),gate,&value.angular_error,true,nullptr,active_patterson_tuning().critical_order,nullptr,nullptr,-1,true)){reason="stream_local_repair";return false;}
   }
   topology=std::move(repaired);
  }
  // The same proved root count, with its continued endpoint coordinates.
  value.topology=std::move(topology);value.anchor=anchor;
  for(std::size_t i=0;i<anchor.count;++i)value.topology.arcs[i]={anchor.arcs[i].lo,anchor.arcs[i].hi};
#ifdef LCBININT_TRIPLE_BOUNDARY_CONTROL
  if(controlled_uniform()&&!value.topology.roots.empty())for(std::size_t i=0;i<anchor.count;++i){const auto& a=anchor.arcs[i];const auto l=context_->control->total_derivative(R,a.lo,a.dlo),h=context_->control->total_derivative(R,a.hi,a.dhi);for(int j=0;j<8;++j)value.value.uniform[j]+=h[j]-l[j];}
#endif
  value.value_ready=true;value.motion=motion;if(found!=context_->points.end()){*found->second=std::move(value);output=found->second.get();}else output=context_->points.emplace(R,std::make_unique<Point>(std::move(value))).first->second.get();return true;
 }
 bool geometry_point(double R,Point*& output,const AngularTopology* predicted=nullptr){
  if(R<=safe_lo_)R=std::nextafter(safe_lo_,map_.b);if(R>=safe_hi_)R=std::nextafter(safe_hi_,map_.a);auto found=context_->points.find(R);if(found!=context_->points.end()){output=found->second.get();return true;}
  AngularTopology topology;Anchor anchor;double seed_radius=R;if(predicted)topology=*predicted;else if(!proposal(R,topology,seed_radius,anchor))return false;
  Point value;value.cores=std::make_unique<shape_gradient_detail::GeometryCores>();const auto resolve=[&](AngularTopology& t,bool promote_unresolved,bool recover_roundoff=false){shape_gradient_detail::Ray ray(p_,R);value.anchor={};value.anchor.radius=R;
   if(t.arcs.size()>3)return false;
   for(std::size_t i=0;i<t.arcs.size();++i){const auto arc=t.arcs[i];shape_gradient_detail::Endpoint lo,hi;lo.theta=arc.lo;hi.theta=arc.hi;
    if(!t.roots.empty()){const double width=arc.hi-arc.lo;if(!shape_gradient_detail::spatial_endpoint(ray,R,p_,arc.lo,width,lo,context_->work,&(*value.cores)[i][0],shape_gradient_detail::endpoint_relative_budget(std::max(1e-12,relative_*.1)),promote_unresolved)||!shape_gradient_detail::spatial_endpoint(ray,R,p_,arc.hi,width,hi,context_->work,&(*value.cores)[i][1],shape_gradient_detail::endpoint_relative_budget(std::max(1e-12,relative_*.1)),promote_unresolved)||!(lo.slope>0&&hi.slope<0&&hi.theta>lo.theta))return false;
     for(std::size_t j=0;j<i;++j)for(int turn:{-1,0,1}){const auto& previous=value.anchor.arcs[j];const double shift=turn*2*std::acos(-1.);if(std::max(lo.theta,previous.lo+shift)<std::min(hi.theta,previous.hi+shift))return false;}
     double error=0;const double midpoint=(lo.theta+hi.theta)*.5;const double phi=ray.sample<false>(midpoint,error,nullptr,&(*value.cores)[i][2]).phi;
     if(!(phi>0)){if(!recover_roundoff || !(phi>=-error))return false;const auto q=ray.precise_value().evaluate(local_precision::Q(midpoint)).phi;++context_->work.quad_samples;if(!finiteq(q)||!(q>0))return false;}
    }
    value.anchor.arcs[value.anchor.count++]={lo.theta,hi.theta,lo.radial_motion,hi.radial_motion,lo.precise_theta,hi.precise_theta,lo.wide,hi.wide};t.arcs[i]={lo.theta,hi.theta};
   }value.anchor.geometry_verified=true;return true;
  };
  if(!resolve(topology,false)){LCBININT_COUPLED_SCOPE(topology);auto proposals=topology;proposals.roots.clear();for(const auto& arc:topology.arcs)for(double angle:{arc.lo,arc.hi}){while(angle<-std::acos(-1.)*.5)angle+=2*std::acos(-1.);while(angle>std::acos(-1.)*1.5)angle-=2*std::acos(-1.);proposals.roots.push_back({angle,angle});}std::sort(proposals.roots.begin(),proposals.roots.end(),[](auto a,auto b){return a.lo<b.lo;});auto repaired=continued_topology(R,p_,proposals,options_,int(topology.roots.size()),shared_?shared_->coefficients.get():nullptr);
   const bool continued=repaired.completed;
   bool resolved=continued&&resolve(repaired,false);
   if(continued)++context_->work.continued_radii;
   if(!resolved){
    repaired=angular_topology(R,p_,options_,shared_?shared_->coefficients.get():nullptr);++context_->work.global_radii;
    resolved=repaired.completed&&resolve(repaired,true);
   }
   if(!resolved){
    // Re-isolation itself can refuse a nearly singular binary64 endpoint.
    // The retained band still supplies the certified crossing count; promote
    // its endpoint candidates and keep every physical acceptance check.
    repaired=topology;
    if(!resolve(repaired,true) && !resolve(repaired,true,true)){reason="patterson_geometry_repair";return false;}
   }topology=std::move(repaired);
  }
  value.topology=std::move(topology);output=context_->points.emplace(R,std::make_unique<Point>(std::move(value))).first->second.get();return true;
 }

 void real_cells(){
  // A real neighbouring fold also limits the analytic disk of a nominally
  // isolated component. Keep the incumbent hyperbolic pair chart when that
  // disk is close; a two-event count alone does not certify a smooth weight.
  if(weighted_){const double width=map_.b-map_.a,match=64*std::numeric_limits<double>::epsilon()*(1+map_.b);for(const auto& e:events_)if((e.radius<map_.a-match&&map_.a-e.radius<4*width)||(e.radius>map_.b+match&&e.radius-map_.b<4*width)){weighted_=false;break;}}
  if(weighted_){cells_.push_back({0,{map_.a,map_.b,false,false},0,1});return;}
  if(!mixed_declined_&&moving_){const double width=map_.b-map_.a,match=64*std::numeric_limits<double>::epsilon()*(1+map_.b);const FoldHint *left=nullptr,*right=nullptr;bool isolated=true;
   for(const auto& e:events_)if((e.radius<map_.a-match&&map_.a-e.radius<4*width)||(e.radius>map_.b+match&&e.radius-map_.b<4*width))isolated=false;
   for(const auto& h:hints_){if(std::abs(h.radius-map_.a)<match)left=&h;if(std::abs(h.radius-map_.b)<match)right=&h;}
   if(isolated&&left&&right&&left->square_slope>0&&right->square_slope<0&&left->curvature*right->curvature>0&&std::abs(std::remainder(left->theta-right->theta,2*std::acos(-1.)))<.5){mixed_=true;cells_.push_back({4,{map_.a,map_.b,false,false},0,1});return;}
  }
  const double width=map_.b-map_.a,match=64*std::numeric_limits<double>::epsilon()*(1+map_.b);
  double ls=INFINITY,rs=INFINITY;const RadialEvent *left_neighbour=nullptr,*right_neighbour=nullptr;bool left=false,right=false;
  for(const auto& e:events_){if(e.radius<map_.a-match&&(map_.a-e.radius)*.5<ls){ls=(map_.a-e.radius)*.5;left_neighbour=&e;}if(e.radius>map_.b+match&&(e.radius-map_.b)*.5<rs){rs=(e.radius-map_.b)*.5;right_neighbour=&e;}left|=std::abs(e.radius-map_.a)<match;right|=std::abs(e.radius-map_.b)<match;}
  // Real neighbouring events form the two factors of the same exterior
  // fold chart. Do not discard a moderate gap by a fixed width ratio: its
  // scalar flux can look converged while material transport stays sharp.
  left&=std::isfinite(ls);right&=std::isfinite(rs);
#if defined(LCBININT_TRIPLE_MULTIRATE_ARCS) && (defined(LCBININT_TRIPLE_FOLD_SECTORS) || defined(LCBININT_TRIPLE_WINDING_ARCS))
  if(arc_sector_>=0){left&=map_.left;right&=map_.right;}
#endif
  const auto event_motion=[&](const RadialEvent* e)->std::optional<std::array<double,8>>{if(!e||!moving_)return std::nullopt;const double angle=(e->reciprocal?std::acos(-1.):0)+2*std::atan(e->coordinate);const auto q=fused_detail::Ray<double>(e->radius,p_).sample(angle);if(!(std::isfinite(q.r)&&std::abs(q.r)>1e-12&&q.roundoff<1e-7&&std::abs(q.phi)<1e-6&&std::abs(q.t)<1e-6))return std::nullopt;std::array<double,8> out{};for(int j=0;j<8;++j){out[j]=-q.p[j]/q.r;if(!std::isfinite(out[j]))return std::nullopt;}return out;};
  auto add=[&](double a,double b,bool at_left,double scale,bool fold){const double limit=2*std::asinh(std::sqrt((b-a)/(2*scale))),center=at_left?a:b,end=at_left?b:a,sign=at_left?1:-1;const auto neighbour=event_motion(at_left?left_neighbour:right_neighbour);
   const int n=std::max(1,int(std::ceil(limit/(active_patterson_tuning().chart_width>0?active_patterson_tuning().chart_width:(geometry_probe_?INFINITY:first_order_chart_width(uniform_))))));
   std::array<double,8> cm{},sm{},limit_motion{};if(neighbour)for(int j=0;j<8;++j){cm[j]=base_[j]+slope_[j]*(center-origin_);sm[j]=(at_left?1:-1)*.5*(cm[j]-(*neighbour)[j]);const double ve=base_[j]+slope_[j]*(end-origin_),sh=std::sinh(limit*.5);limit_motion[j]=(sign*(ve-cm[j])-sm[j]*2*sh*sh)/(scale*std::sinh(limit));}
   for(int i=0;i<n;++i){Cell c{at_left?2:3,{limit*i/n,limit*(i+1)/n,false,fold&&i+1==n},center,scale};c.material=bool(neighbour);c.center_motion=cm;c.scale_motion=sm;for(int j=0;j<8;++j){c.lower_motion[j]=limit_motion[j]*i/n;c.upper_motion[j]=limit_motion[j]*(i+1)/n;}cells_.push_back(std::move(c));}
  };
  if(left&&right){double mid=map_.a+width*.5;add(map_.a,mid,true,ls,false);add(mid,map_.b,false,rs,false);}
  else if(left)add(map_.a,map_.b,true,ls,map_.right);
  else if(right)add(map_.a,map_.b,false,rs,map_.left);
  else cells_.push_back({0,weighted_?FoldRadialMap{map_.a,map_.b,false,false}:map_,0,1});
 }
 bool boundary_resolved(){
  const auto& witnesses=first_order_rule(3);bool okay=true;
  for(auto& cell:cells_){if(!build_curve(cell,2))return false;
   for(std::size_t k=0;k<witnesses.x.size();k+=2){const double x=witnesses.x[k];const auto node=mapped(cell,x);AngularTopology guess;if(!predict(cell,x,guess))return false;Point* actual=nullptr;if(!geometry_point(node[0],actual))return false;if(actual->anchor.radius!=node[0]||actual->anchor.count!=guess.arcs.size())return false;
    std::array<bool,3> used{};for(const auto& arc:guess.arcs){int selected=-1;double best=INFINITY,shift=0;const double center=(arc.lo+arc.hi)*.5;for(std::size_t j=0;j<actual->anchor.count;++j)if(!used[j]){const auto& root=actual->anchor.arcs[j];const double s=2*std::acos(-1.)*std::round((center-(root.lo+root.hi)*.5)/(2*std::acos(-1.))),distance=std::abs((root.lo+root.hi)*.5+s-center);if(distance<best){best=distance;selected=int(j);shift=s;}}
     if(selected<0)return false;used[selected]=true;const auto& root=actual->anchor.arcs[selected];const double delta=std::abs(arc.lo-root.lo-shift)+std::abs(arc.hi-root.hi-shift);
     if(delta>.5*relative_*(root.hi-root.lo))okay=false;
    }
   }
  }return okay;
 }
 bool complex_cells(){
  if(charts_checked_)return false;LCBININT_COUPLED_SCOPE(chart);charts_checked_=true;
  std::vector<shape_gradient_detail::RadialComplexPair> pairs;
  for(const auto& entry:context_->points){if(entry.first<=map_.a||entry.first>=map_.b)continue;
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
   if(arc_sector_>=0&&std::find(discovery_radii_.begin(),discovery_radii_.end(),entry.first)==discovery_radii_.end())continue;
#endif
   for(std::size_t i=0;i<entry.second->anchor.count;++i)
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
   if(arc_sector_<0||gap_lanes_||selected_arc(*entry.second)==int(i))
#endif
   for(double theta:{entry.second->anchor.arcs[i].lo,entry.second->anchor.arcs[i].hi}){
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
#ifdef LCBININT_TRIPLE_WINDING_ARCS
    if(arc_sector_>=0&&gap_lanes_&&winding_){if(!endpoint_selected(*entry.second,int(i),theta==entry.second->anchor.arcs[i].hi))continue;}else
#endif
    if(arc_sector_>=0&&gap_lanes_&&sector_id(entry.first,entry.second->cores?(*entry.second->cores)[i][theta==entry.second->anchor.arcs[i].lo?0:1].w:std::polar(1.,theta))!=arc_sector_)continue;
#endif
    const std::array<double,2> seed{entry.first,theta};std::optional<shape_gradient_detail::RadialComplexPair> pair;
    const auto retained=context_->radial_roots.find(seed);const double width=map_.b-map_.a;
    if(retained!=context_->radial_roots.end()&&retained->second.center>map_.a&&retained->second.center<map_.b&&retained->second.scale>1e-10*width&&retained->second.scale<=.25*width)pair=retained->second;
    else {pair=shape_gradient_detail::radial_complex_pair(entry.first,theta,map_.a,map_.b,p_,true);if(pair)context_->radial_roots.insert_or_assign(seed,*pair);}
    if(!pair)continue;bool known=false;
    for(const auto& old:pairs)if(std::abs(old.center-pair->center)<1e-6*(map_.b-map_.a))known=true;if(!known)pairs.push_back(*pair);
   }
  }
  if(pairs.empty())return false;std::sort(pairs.begin(),pairs.end(),[](auto a,auto b){return a.center<b.center;});
  // A proposed partition must remain representable before replacing the
  // retained physical-radius cells. Coincident cuts otherwise create 0/0
  // material motion and poison a value-converged gradient.
  for(std::size_t i=0;i<pairs.size();++i){const auto p=pairs[i];const double a=i?(pairs[i-1].center+p.center)*.5:map_.a,b=i+1<pairs.size()?(p.center+pairs[i+1].center)*.5:map_.b;
   const double lo=std::asinh((a-p.center)/p.scale),hi=std::asinh((b-p.center)/p.scale);
   if(!(b>a) || !(hi>lo) || !std::isfinite(lo) || !std::isfinite(hi))return false;
  }
  cells_.clear();weighted_=false;mixed_=false;
  for(std::size_t i=0;i<pairs.size();++i){const auto p=pairs[i];const double a=i?(pairs[i-1].center+p.center)*.5:map_.a,b=i+1<pairs.size()?(p.center+pairs[i+1].center)*.5:map_.b;
   const double lo=std::asinh((a-p.center)/p.scale),hi=std::asinh((b-p.center)/p.scale);const int n=std::max(1,int(std::ceil((hi-lo)/(active_patterson_tuning().chart_width>0?active_patterson_tuning().chart_width:(geometry_probe_?INFINITY:first_order_chart_width(uniform_))))));
   std::array<double,8> lower{},upper{};const bool material=p.material&&(i==0||pairs[i-1].material)&&(i+1==pairs.size()||pairs[i+1].material);
   if(material)for(int j=0;j<8;++j){const double va=i?.5*(pairs[i-1].center_motion[j]+p.center_motion[j]):base_[j]+slope_[j]*(a-origin_),vb=i+1<pairs.size()?.5*(p.center_motion[j]+pairs[i+1].center_motion[j]):base_[j]+slope_[j]*(b-origin_);
    lower[j]=(va-p.center_motion[j]-p.scale_motion[j]*std::sinh(lo))/(p.scale*std::cosh(lo));upper[j]=(vb-p.center_motion[j]-p.scale_motion[j]*std::sinh(hi))/(p.scale*std::cosh(hi));}
   for(int k=0;k<n;++k){Cell c{1,{lo+(hi-lo)*k/n,lo+(hi-lo)*(k+1)/n,i==0&&k==0&&map_.left,i+1==pairs.size()&&k+1==n&&map_.right},p.center,p.scale};c.material=material;c.center_motion=p.center_motion;c.scale_motion=p.scale_motion;for(int j=0;j<8;++j){c.lower_motion[j]=lower[j]+(upper[j]-lower[j])*k/n;c.upper_motion[j]=lower[j]+(upper[j]-lower[j])*(k+1)/n;}cells_.push_back(std::move(c));}
  }return true;
 }
public:
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
 FirstOrderPanel(FirstOrderPanel&&)=default;FirstOrderPanel& operator=(FirstOrderPanel&&)=default;
#endif
 int split_level=5;
 std::string reason="stream_not_started";
 FirstOrderPanel(const Geometry&p,double a,double b,bool left,bool right,AngularOptions options,std::shared_ptr<SharedRadiusCache<Gradient>> shared,double absolute,double relative,bool uniform,bool weighted,double limb):p_(p),map_{a,b,left,right},options_(options),shared_(std::move(shared)),relative_(relative),absolute_(absolute),uniform_weight_(1-limb),limb_weight_(limb),safe_lo_(a),safe_hi_(b),uniform_(uniform),weighted_(weighted){
  if(shared_){if(!shared_->first_order_coefficients)shared_->first_order_coefficients=std::make_shared<std::unique_ptr<root_pair_detail::RadialCache<8>>>();context_->pairs=shared_->first_order_coefficients;}
  else context_->pairs=std::make_shared<std::unique_ptr<root_pair_detail::RadialCache<8>>>();
 }
 ~FirstOrderPanel(){
#ifdef LCBININT_TRIPLE_DIAGNOSTIC_TUNING
  if(trace_active_&&context_&&level_){PattersonRecord r{map_.a,map_.b,radial_error_,inner_error_,level_,context_->points.size(),accepted_,{}};for(const auto& c:cells_)r.cells.push_back({double(c.kind),c.coordinates.a,c.coordinates.b,c.center,c.scale});patterson_records.push_back(std::move(r));}
#endif
 }
 void set_events(const std::vector<RadialEvent>& e){events_=e;hints_.clear();const double match=64*std::numeric_limits<double>::epsilon()*(1+map_.b);for(const auto& event:e)if(std::abs(event.radius-map_.a)<match||std::abs(event.radius-map_.b)<match){const double theta=(event.reciprocal?std::acos(-1.):0)+2*std::atan(event.coordinate);shape_gradient_detail::Ray ray(p_,event.radius);double error=0,curvature=0;const auto q=ray.sample<false>(theta,error,&curvature);if(curvature!=0&&std::isfinite(curvature)&&std::isfinite(q.dR))hints_.push_back({event.radius,theta,-2*q.dR/curvature,curvature});}
#ifdef LCBININT_TRIPLE_BOUNDARY_CONTROL
  if(uniform_){if(shared_){if(!shared_->boundary_control)shared_->boundary_control=std::make_shared<BoundaryControlAtlas>(p_,events_);context_->control=shared_->boundary_control;}else context_->control=std::make_shared<BoundaryControlAtlas>(p_,events_);}
#endif
}
 void inherit_stream(const FirstOrderPanel& parent){context_=parent.context_;owns_context_=false;parent.trace_active_=false;events_=parent.events_;hints_=parent.hints_;mixed_declined_=parent.mixed_declined_;}
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
 void inherited_arc_identity(const FirstOrderPanel& parent){arc_sector_=parent.arc_sector_;sector_cuts_=parent.sector_cuts_;}
 bool branches_active()const{return !lanes_.empty();}
 bool refine_branches(int angular){
  std::size_t worst=0;double error=-1;for(std::size_t i=0;i<lanes_.size();++i){const double e=lanes_[i]->radial_error()+lanes_[i]->inner_error();if(e>error){error=e;worst=i;}}
  auto& child=lanes_[worst];if(child->level_>=child->split_level){const double middle=.5*(child->map_.a+child->map_.b);auto left=child->lane(child->map_.a,middle,child->map_.left,false,child->arc_sector_),right=child->lane(middle,child->map_.b,false,child->map_.right,child->arc_sector_);if(!left->refine(2,angular)||!right->refine(2,angular)){reason="stream_arc_split";return false;}child->trace_active_=false;child=std::move(left);lanes_.push_back(std::move(right));}
  else if(!child->refine(child->level_+1,angular)){reason=child->reason;return false;}
  collect_lanes();return true;
 }
#endif
 void set_radial_motion(double origin,const std::array<double,8>& base,const std::array<double,8>& slope){origin_=origin;base_=base;slope_=slope;moving_=true;}
 bool has_radial_motion()const{return moving_;}double motion_origin()const{return origin_;}const auto& motion_base()const{return base_;}const auto& motion_slope()const{return slope_;}
 void set_certified_band(const AngularTopology& t,double lo,double hi){band_=t;safe_lo_=lo;safe_hi_=hi;}
 const auto& band_topology()const{return band_;}double safe_lo()const{return safe_lo_;}double safe_hi()const{return safe_hi_;}
 const auto& result()const{return accepted_;}double radial_error()const{return radial_error_;}double inner_error()const{return inner_error_;}
 std::size_t geometry_promotions()const{return 0;}
 const ShapeGradientResult& work()const{static const ShapeGradientResult empty{};return owns_context_?context_->work:empty;}
 bool refine(int requested,int angular){
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
  start_lanes(requested,angular);if(!lanes_.empty()){for(auto& child:lanes_)if(!child->refine(requested,angular)){reason=child->reason;return false;}collect_lanes();level_=std::max(level_,requested);angular_level_=std::max(angular_level_,angular);return true;}
#endif
  if(mixed_&&requested>3){mixed_=false;mixed_declined_=true;cells_.clear();charts_checked_=false;previous_cells_.clear();level_=0;accepted_={};}

  if(uniform_&&band_&&band_->roots.empty()&&band_->arcs.size()==1&&band_->arcs[0].hi-band_->arcs[0].lo>=2*std::acos(-1.)-1e-12){
   const double pi=std::acos(-1.);accepted_={};accepted_.f0=pi*(map_.b-map_.a)*(map_.a+map_.b);
   if(!controlled_uniform())for(int j=0;j<8;++j){const double a=base_[j]+slope_[j]*(map_.a-origin_),b=base_[j]+slope_[j]*(map_.b-origin_);accepted_.df0[j]=2*pi*(map_.b*b-map_.a*a);}
   radial_error_=16*std::numeric_limits<double>::epsilon()*std::abs(accepted_.f0);inner_error_=0;accepted_.angular_error=radial_error_;level_=std::max(level_,requested);angular_level_=std::max(angular_level_,angular);reason="stream_full_annulus_exact";return true;
  }
  if(cells_.empty()){
   real_cells();const auto& probes=first_order_rule(2);
   for(const auto& cell:cells_)for(double x:probes.x){const double R=mapped(cell,x)[0];
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
    if(arc_sector_>=0)discovery_radii_.push_back(R);
#endif
    Point* geometry=nullptr;if(!geometry_point(R,geometry))return false;}
   if(geometry_probe_){
    if(boundary_resolved())charts_checked_=true;
    else{geometry_probe_=false;cells_.clear();real_cells();for(const auto& cell:cells_)for(double x:probes.x){Point* geometry=nullptr;if(!geometry_point(mapped(cell,x)[0],geometry))return false;}complex_cells();}
   }else complex_cells();
   const int geometry_level=2;for(auto& cell:cells_){const auto normal_probes=integration_rule(geometry_level,weighted_&&cell.kind==0);const auto& seed_rule=first_order_rule(2);const RuleView final_probes=use_kronrod()?RuleView{seed_rule.x,seed_rule.w,seed_rule.interp}:normal_probes;for(double x:final_probes.x){Point* geometry=nullptr;if(!geometry_point(mapped(cell,x)[0],geometry))return false;}build_curve(cell,geometry_level,use_kronrod());}
  }
  split_level=use_kronrod()?3:5;
  if(requested<=level_)return true;auto previous=accepted_;bool have=level_>0;
  for(int lev=std::max(1,level_+1);lev<=requested;++lev){bool boundary_ok=true;RadiusIntegral<true> next;double inner=0,detail=0,cell_error=0;std::vector<std::array<double,2>> next_cells;
   // Refine one unresolved cell; keep every other complete tangent packet.
   const bool local_step=level_>=3&&!mixed_&&!use_kronrod()&&cells_.size()>1;std::size_t selected_cell=0;
   if(local_step){double worst=-1;for(std::size_t i=0;i<cells_.size();++i){const double error=std::max(cells_[i].error,(QuadratureErrorPolicy::weighted_detail_floor/QuadratureErrorPolicy::difference_safety)*cells_[i].detail)+cells_[i].inner;if(error>worst){worst=error;selected_cell=i;}}split_level=std::min(8,3+2*int(cells_.size()));}
   for(auto& cell:cells_){const auto cell_index=next_cells.size();
    if(local_step&&cell_index!=selected_cell){const auto& v=cell.accepted;next.f0+=v.f0;next.fh+=v.fh;next.endpoint_error+=v.endpoint_error;for(int j=0;j<8;++j){next.df0[j]+=v.df0[j];next.dfh[j]+=v.dfh[j];}inner+=cell.inner;detail+=cell.detail;cell_error+=cell.error;next_cells.push_back({v.f0,v.fh});continue;}
    const int node_level=local_step?cell.level+1:lev;const auto start=next;const double inner_start=inner,detail_start=detail;
    bool cell_geometry_ok=true;const double start0=next.f0,starth=next.fh;const bool weighted=weighted_&&cell.kind==0;const auto rule=integration_rule(node_level,weighted);const auto limb_rule=integration_rule(node_level,weighted,true);
    std::vector<RadiusIntegral<false>> samples(rule.x.size());
#ifdef LCBININT_TRIPLE_COMPLEX_PRODUCT_RULE
    std::optional<complex_product_detail::Rule> product;std::vector<std::array<double,3>> arc_samples(rule.x.size());
    if(uniform_&&!weighted&&cell.kind>=1&&cell.kind<=3&&cell.material&&!use_kronrod()&&(node_level==2||node_level==3)){
     using J=complex_product_detail::J;const auto packet=[&](double x){const auto q=mapped_transport(cell,x);J r(q.node[0]);r.dp=q.material[0];return r;};J c(cell.center),scale(cell.scale);c.dp=cell.center_motion;scale.dp=cell.scale_motion;if(cell.kind==2)c=c-scale;if(cell.kind==3)c=c+scale;std::vector<J> nodes;for(double x:rule.x)nodes.push_back(packet(x));auto a=packet(-1),b=packet(1);if(a.value>b.value)std::swap(a,b);product=complex_product_detail::make(nodes,a,b,c,scale,cell.kind==1?1:-1);
    }
#endif
    for(std::size_t k=0;k<rule.x.size();++k){const double x=rule.x[k];
     const double shape=(1-x)*(1+x);const auto mapped_value=mapped_transport(cell,x);const auto& node=mapped_value.node;AngularTopology predicted;
     const bool witness=node_level>=3&&k%2==0&&!weighted&&!uniform_;
     const bool predicted_ok=predict(cell,x,predicted);
     const auto& material=mapped_value.material;Point* cached=nullptr;if(!point(node[0],cached,predicted_ok?&predicted:nullptr,&material[0]))return false;const auto& value=cached->value;
     if(witness){if(!predicted_ok||predicted.arcs.size()!=cached->anchor.count)cell_geometry_ok=false;else{std::array<bool,3> used{};const double period=2*std::acos(-1.);for(std::size_t a=0;a<predicted.arcs.size();++a){const auto& guess=predicted.arcs[a];int chosen=-1;double best=INFINITY,shift=0;for(std::size_t j=0;j<cached->anchor.count;++j)if(!used[j]){const auto& actual=cached->anchor.arcs[j];const double turns=period*std::round((guess.lo+guess.hi-actual.lo-actual.hi)/(2*period)),distance=std::abs(guess.lo+guess.hi-actual.lo-actual.hi-2*turns);if(distance<best){best=distance;chosen=int(j);shift=turns;}}if(chosen<0){cell_geometry_ok=false;continue;}used[chosen]=true;const auto& actual=cached->anchor.arcs[chosen];const double width=actual.hi-actual.lo;if(std::abs(guess.lo-actual.lo-shift)+std::abs(guess.hi-actual.hi-shift)>
 4*geometry_resolution_budget()*width
 )cell_geometry_ok=false;
     }}}
#if defined(LCBININT_TRIPLE_MULTIRATE_ARCS) && defined(LCBININT_TRIPLE_SECTOR_BOUNDARY_WITNESS)
     if(uniform_&&geometry_guard_&&node_level>=3&&k%2==0){if(!predicted_ok)boundary_ok=false;else{const double width=std::abs(value.f0/node[0]),period=2*std::acos(-1.);for(int endpoint:cached->endpoint_labels){if(endpoint<0){boundary_ok=false;continue;}const auto& arc=cached->anchor.arcs[endpoint/2];const double theta=endpoint%2?arc.hi:arc.lo;double delta=INFINITY;for(const auto& proposal:predicted.arcs)delta=std::min(delta,std::abs(std::remainder((endpoint%2?proposal.hi:proposal.lo)-theta,period)));const double tolerance=std::max(.5*relative_*width,64*std::numeric_limits<double>::epsilon()*(1+std::abs(theta)));if(delta>tolerance)boundary_ok=false;}}}
#endif
#ifdef LCBININT_TRIPLE_COMPLEX_PRODUCT_RULE
     if(product)for(std::size_t j=0;j<cached->anchor.count;++j)arc_samples[k][j]=node[0]*(cached->anchor.arcs[j].hi-cached->anchor.arcs[j].lo);
#endif
     const double wu=
#ifdef LCBININT_TRIPLE_COMPLEX_PRODUCT_RULE
      product?product->weight[k].value:
#endif
      rule.w[k]*node[1]*(weighted?1/std::sqrt(shape):1),wh=limb_rule.w[k]*node[1]*(weighted?1/shape:1);
     samples[k].f0=
#ifdef LCBININT_TRIPLE_COMPLEX_PRODUCT_RULE
      product?value.f0:
#endif
      node[1]*(weighted?1/std::sqrt(shape):1)*value.f0;samples[k].fh=node[1]*(weighted?1/shape:1)*value.fh;
     next.f0+=wu*value.f0;next.fh+=wh*value.fh;next.endpoint_error+=std::abs(wu)*value.endpoint_error;inner+=limb_weight_*std::abs(wh)*cached->angular_error;
#ifdef LCBININT_TRIPLE_DIAGNOSTIC_TUNING
     if(uniform_&&!controlled_uniform()&&active_patterson_tuning().quad_uniform){using Q=local_precision::Q;std::array<Q,8> uniform{};
      for(std::size_t arc=0;arc<cached->anchor.count;++arc){const auto& ends=cached->anchor.arcs[arc];Q l,h;if(!local_precision::endpoint(cached->anchor.radius,ends.lo,p_,l)||!local_precision::endpoint(cached->anchor.radius,ends.hi,p_,h))return false;const auto jl=local_precision::evaluate<true>(Q(cached->anchor.radius),l,p_),jh=local_precision::evaluate<true>(Q(cached->anchor.radius),h,p_);for(int j=0;j<8;++j)uniform[j]+=Q(material[0][j])*(h-l)+Q(cached->anchor.radius)*((jl.dp[j]+jl.dR*Q(material[0][j]))/jl.dtheta-(jh.dp[j]+jh.dR*Q(material[0][j]))/jh.dtheta)+Q(material[1][j])*Q(cached->anchor.radius)*(h-l);}
      for(int j=0;j<8;++j)next.df0[j]+=double(Q(wu)*uniform[j]);
     }else
#endif
     for(int j=0;j<8;++j){
#ifdef LCBININT_TRIPLE_COMPLEX_PRODUCT_RULE
      if(product)next.df0[j]+=wu*value.uniform[j]+product->weight[k].dp[j]*value.f0;else
#endif
      next.df0[j]+=wu*(value.uniform[j]+(controlled_uniform()?0:material[1][j])*value.f0);next.dfh[j]+=wh*(value.limb[j]+material[1][j]*value.fh);}
    }
    if(use_kronrod()&&node_level==2)build_curve(cell,2);else if(!use_kronrod()&&node_level==3)build_curve(cell,3);
    if(node_level==3&&!weighted&&!mixed_&&!use_kronrod()
    &&uniform_
    )cell_geometry_ok=spatially_resolved(cell);
    const std::array<double,2> cell_value{next.f0-start0,next.fh-starth};const auto index=next_cells.size();
    bool embedded_used=false;double embedded_detail=0,local_error=INFINITY;
    if(have&&index<previous_cells_.size()){
     local_error=uniform_weight_*std::abs(cell_value[0]-previous_cells_[index][0])+limb_weight_*std::abs(cell_value[1]-previous_cells_[index][1]);
     if(!weighted&&!mixed_&&!use_kronrod()&&node_level>=3&&node_level<=5){
      const auto observable=[&](const auto& a){return uniform_weight_*a.f0+limb_weight_*a.fh;};
      const auto embedded=[&](){
       if constexpr(Fejer)return panel_detail::embedded_estimate(node_level,samples,observable);
       else return embedded_patterson_detail::embedded_estimate(node_level,samples,observable);
      }();
      const double fine=uniform_weight_*cell_value[0]+limb_weight_*cell_value[1],old=uniform_weight_*previous_cells_[index][0]+limb_weight_*previous_cells_[index][1];
      const double low=std::abs(embedded.first-old),high=std::abs(fine-embedded.first);
      if(low>64*std::numeric_limits<double>::epsilon()*std::abs(fine)&&high<=low*.25){local_error=std::max(local_error,high);embedded_detail=embedded.second;embedded_used=true;}
     }
     if(!cell_geometry_ok){
      local_error=INFINITY;
      // A binary64 radial collision cannot be repaired by subdividing its
      // angular interpolant indefinitely. Bound the scalar area rather than
      // silently accepting interpolation or dropping an image component.
      const auto& lattice=first_order_rule(3);std::array<double,7> radius{};bool collision=false;
      for(int j=0;j<7;++j){radius[j]=mapped(cell,lattice.x[j])[0];for(int k=0;k<j;++k)collision=collision||radius[j]==radius[k];}
      if(collision){
       const double upper=std::nextafter(std::acos(-1.)*(map_.b-map_.a)*(map_.b+map_.a)* (1+8*std::numeric_limits<double>::epsilon()),INFINITY);
       const double fine=uniform_weight_*cell_value[0]+limb_weight_*cell_value[1];
       local_error=std::max(std::abs(fine),std::abs(upper-fine));
       // All retained physical packets and tangents stay in the sum. Only
       // the flux error estimator changes: any cell's true weighted area is
       // in [0, full annulus area], including a cell with unresolved shape.
       // This bound enters the ordinary requested value-error budget.
      }
     }
     cell_error+=local_error;
    }
    next_cells.push_back(cell_value);
    if(embedded_used)detail+=embedded_detail;else
    if(node_level>1&&!mixed_&&(!use_kronrod()||weighted||node_level>2)){
#ifdef LCBININT_TRIPLE_COMPLEX_PRODUCT_RULE
if(product){if(!product->interpolation.empty())for(std::size_t k=0;k<samples.size();k+=2)for(int arc=0;arc<3;++arc){double predicted=0;for(int j=0;j<3;++j)predicted+=product->interpolation[k][j]*arc_samples[2*j+1][arc];detail+=uniform_weight_*std::abs(product->weight[k].value)*std::abs(arc_samples[k][arc]-predicted);}}else
#endif
if(weighted)detail+=uniform_weight_*panel_detail::interpolation_detail(rule,samples,[](const auto& a){return a.f0;})+limb_weight_*panel_detail::interpolation_detail(limb_rule,samples,[](const auto& a){return a.fh;});
     else detail+=panel_detail::interpolation_detail(rule,samples,[&](const auto& a){return uniform_weight_*a.f0+limb_weight_*a.fh;});
    }
    cell.accepted=next;cell.accepted.f0-=start.f0;cell.accepted.fh-=start.fh;cell.accepted.endpoint_error-=start.endpoint_error;for(int j=0;j<8;++j){cell.accepted.df0[j]-=start.df0[j];cell.accepted.dfh[j]-=start.dfh[j];}cell.error=local_error;cell.inner=inner-inner_start;cell.detail=detail-detail_start;cell.level=node_level;
    if(node_level>=5)split_level=std::min(split_level,lev);

   }
   radial_error_=have&&previous_cells_.size()==next_cells.size()?cell_error:INFINITY;
   radial_error_=std::max(radial_error_,(QuadratureErrorPolicy::weighted_detail_floor/QuadratureErrorPolicy::difference_safety)*detail);
   // Never accept a one/three-node coincidence as a geometry certificate.
   const auto tuning=active_patterson_tuning();
   // Unweighted LD charts can have nearly flat flux while their
   // material angular transport still varies. Keep a fixed 31-node
   // Patterson floor, as a coordinate-resolution guard before the scalar
   // convergence test. No derivative error enters the controller.
   const int default_minimum=mixed_?3:(use_kronrod()?2:((Fejer||!uniform_)?5:3));const int minimum=weighted_&&!uniform_?4:(map_.a>=tuning.selected_lo&&map_.b<=tuning.selected_hi&&tuning.minimum_radial?tuning.minimum_radial:default_minimum);
   if(lev<minimum)radial_error_=INFINITY;
#if defined(LCBININT_TRIPLE_MULTIRATE_ARCS) && defined(LCBININT_TRIPLE_SECTOR_BOUNDARY_WITNESS)
   if(!boundary_ok){radial_error_=INFINITY;if(lev>=4)split_level=4;}
#endif

   inner_error_=inner;next.angular_error=radial_error_+inner;
   previous=next;previous_cells_=std::move(next_cells);have=true;
   bool rediscovered=false;
   if constexpr(Fejer)if(lev==3&&!discovery_retry_&&std::none_of(cells_.begin(),cells_.end(),[](const Cell& c){return c.kind==1;})){
    // An empty search from three discovery positions is not a certificate
    // that no nearby complex fold exists. Retry once on the new geometry;
    // retain all old physical packets when a chart is introduced.
    discovery_retry_=true;charts_checked_=false;rediscovered=complex_cells();
   }
   if(rediscovered||(lev==2&&complex_cells())){
    // The old nodes remain cached as continuation/discovery work. Rebuild
    // only the coarse rule on the new coordinates before forming a nested
    // difference; two different coordinate systems are not an embedded pair.
    have=false;previous_cells_.clear();--lev;continue;
   }
   level_=lev;
  }
  accepted_=previous;
#ifdef LCBININT_TRIPLE_MULTIRATE_ARCS
  if(arc_sector_>=0&&gap_lanes_&&!whole_gap_background_){const double width=sector_cuts_[(arc_sector_+1)%sector_cuts_.size()]-sector_cuts_[arc_sector_]+(arc_sector_+1==int(sector_cuts_.size())?2*std::acos(-1.):0),background=.5*width*(map_.b-map_.a)*(map_.a+map_.b);accepted_.f0+=background;for(int j=0;j<8;++j){const double a=base_[j]+slope_[j]*(map_.a-origin_),b=base_[j]+slope_[j]*(map_.b-origin_);accepted_.df0[j]+=width*(map_.b*b-map_.a*a);}radial_error_+=32*std::numeric_limits<double>::epsilon()*std::abs(background);accepted_.angular_error=radial_error_+inner_error_;}
#endif
  angular_level_=angular;reason="stream_completed";return true;
 }
};
template<bool Gradient> using PattersonPanel=FirstOrderPanel<Gradient,false>;
template<bool Gradient> using FejerPanel=FirstOrderPanel<Gradient,true>;
// Internal comparison/reference route. No mixed spatial/parameter jets and
// no derivative-error controller; production dispatch is unchanged until
// this route establishes a matched speed advantage.
template<bool Gradient=true> IntegrationResult<Gradient> integrate_fejer_first_order(const Geometry& p,double limb,IntegrationOptions options={},const RadialEventPlan* plan=nullptr){
 static_assert(Gradient,"dedicated derivative stream");
 return integrate_coupled<Gradient,FejerPanel>(p,limb,options,plan);
}
template<bool Gradient=true> IntegrationResult<Gradient> integrate_patterson(const Geometry& p,double limb,IntegrationOptions options={},const RadialEventPlan* plan=nullptr){
 static_assert(Gradient,"dedicated derivative stream");
 // Tight requests retain the higher-order mixed transport in the same
 // value-controlled frontier, rather than restarting a completed value run.
 if(options.relative_tolerance<1e-4)return integrate_coupled<Gradient>(p,limb,options,plan);
 return integrate_coupled<Gradient,PattersonPanel>(p,limb,options,plan);
}
}

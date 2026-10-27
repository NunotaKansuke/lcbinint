#pragma once
#include "shape_transport.hpp"

#include "fused_integrator.hpp"
#include <map>
namespace lcbinint::algebraic_boundary::triple {
// A gradient-owned stream: no completed scalar integral or tangent replay.
// The proved band supplies the root count. Successful angular endpoints and
// complete value/tangent packets are retained as the radial rule grows.
template<bool Gradient> class TransportPanel {
 static_assert(Gradient,"stream panel is a gradient-specific implementation");
 using Packet=shape_gradient_detail::RadiusResult;
 using Anchor=shape_gradient_detail::RadiusAnchor;
 struct Point {Packet value;Anchor anchor;AngularTopology topology;double angular_error=0,radial_angular_error=0;fused_detail::Result spatial;};
 struct Context {std::map<double,Point> points;std::unique_ptr<root_pair_detail::RadialCache<8>> pairs;ShapeGradientResult work;std::shared_ptr<root_pair_detail::PreparedFrames> frames=std::make_shared<root_pair_detail::PreparedFrames>();};
 std::shared_ptr<Context> context_=std::make_shared<Context>();bool owns_context_=true;
 Geometry p_;FoldRadialMap map_;AngularOptions options_;std::shared_ptr<SharedRadiusCache<Gradient>> shared_;
 double relative_,absolute_,uniform_weight_=1,limb_weight_=0,origin_=0,safe_lo_=0,safe_hi_=0;
 bool uniform_,weighted_,moving_=false,charts_checked_=false;
 std::array<double,8> base_{},slope_{},zero_errors_{};
 std::optional<AngularTopology> band_;std::vector<RadialEvent> events_;
 RadiusIntegral<true> accepted_;int level_=0,angular_level_=0;double radial_error_=INFINITY,inner_error_=0;
 struct Cell {int kind=0;FoldRadialMap coordinates;double center=0,scale=1;};
 std::vector<Cell> cells_;
 std::array<double,2> mapped(const Cell& cell,double x)const{
  const auto t=cell.coordinates(x);if(!cell.kind)return t;
  if(cell.kind==1)return {cell.center+cell.scale*std::sinh(t[0]),cell.scale*std::cosh(t[0])*t[1]};
  const double sh=std::sinh(t[0]*.5),d=2*cell.scale*sh*sh;
  return {cell.kind==2?cell.center+d:cell.center-d,cell.scale*std::sinh(t[0])*t[1]};
 }
 bool point(double R,Point*& output){
  if(R<=safe_lo_)R=std::nextafter(safe_lo_,map_.b);if(R>=safe_hi_)R=std::nextafter(safe_hi_,map_.a);
  auto found=context_->points.find(R);if(found!=context_->points.end()){output=&found->second;return true;}
  AngularTopology topology;double seed_radius=R;Anchor anchor;
  auto it=context_->points.lower_bound(R);
  if(it!=context_->points.end() || it!=context_->points.begin()){
   auto seed=it;if(seed==context_->points.end() || (seed!=context_->points.begin()&&std::abs(std::prev(seed)->first-R)<std::abs(seed->first-R)))--seed;
   topology=seed->second.topology;seed_radius=seed->first;anchor=seed->second.anchor;
  }else if(band_){topology=*band_;seed_radius=map_.a+(map_.b-map_.a)*.5;}
  else {topology=angular_topology(R,p_,options_,shared_?shared_->coefficients.get():nullptr);++context_->work.global_radii;}
  if(!topology.completed){reason="stream_topology_missing";return false;}
  if(anchor.count&&seed_radius!=R){for(std::size_t i=0;i<anchor.count;++i){const auto& a=anchor.arcs[i];const double width=a.hi-a.lo,dl=a.dlo*(R-seed_radius),dh=a.dhi*(R-seed_radius);if(std::abs(dl)<width*.25)topology.arcs[i].lo+=dl;if(std::abs(dh)<width*.25)topology.arcs[i].hi+=dh;}}

  std::array<double,8> motion;for(int j=0;j<8;++j)motion[j]=base_[j]+slope_[j]*(R-origin_);
  Point value;const double gate=std::max(1e-12,relative_*.1);
  const auto evaluate=[&](AngularTopology& t){
   fused_detail::Radius radius(R,p_,motion,slope_,context_->frames);if(!radius.initialize(t))return false;
   fused_detail::Result previous,current;bool accepted=false;
   for(int lev=1,minimum=uniform_?2:radius.minimum_level();lev<=10;++lev){
    if(!radius.evaluate(lev,uniform_,current))return false;
    if(lev>=minimum){value.angular_error=std::abs(current.fh-previous.fh);value.radial_angular_error=std::abs(current.rh-previous.rh);
     if(uniform_||value.angular_error<=gate*std::max(std::abs(current.f0),std::abs(current.fh))){accepted=true;break;}}
    previous=current;
   }
   context_->work.evaluations+=radius.evaluations;context_->work.quad_samples+=radius.promoted;if(!accepted)return false;
   value.spatial=current;value.value.f0=current.f0;value.value.fh=current.fh;value.value.uniform=current.p0;value.value.limb=current.ph;
   std::array<std::array<double,2>,3> radial{};radius.endpoint_geometry(t,&radial);anchor.radius=R;anchor.count=t.arcs.size();if(anchor.count>3)return false;
   for(std::size_t i=0;i<anchor.count;++i)anchor.arcs[i]={t.arcs[i].lo,t.arcs[i].hi,radial[i][0],radial[i][1]};return true;
  };
  bool completed=evaluate(topology);
  if(!completed){LCBININT_COUPLED_SCOPE(topology);auto repaired=continued_topology(R,p_,topology,options_,-1,shared_?shared_->coefficients.get():nullptr);
   if(!repaired.completed){repaired=angular_topology(R,p_,options_,shared_?shared_->coefficients.get():nullptr);++context_->work.global_radii;}else ++context_->work.continued_radii;
   value={};anchor={};if(!repaired.completed||!evaluate(repaired)){reason="transport_local_repair";return false;}topology=std::move(repaired);
  }
  // The same proved root count, with its continued endpoint coordinates.
  value.topology=std::move(topology);value.anchor=anchor;
  for(std::size_t i=0;i<anchor.count;++i)value.topology.arcs[i]={anchor.arcs[i].lo,anchor.arcs[i].hi};
  output=&context_->points.emplace(R,std::move(value)).first->second;return true;
 }
 void real_cells(){
  const double width=map_.b-map_.a,match=64*std::numeric_limits<double>::epsilon()*(1+map_.b);
  double ls=INFINITY,rs=INFINITY;bool left=false,right=false;
  for(const auto& e:events_){if(e.radius<map_.a-match)ls=std::min(ls,(map_.a-e.radius)*.5);if(e.radius>map_.b+match)rs=std::min(rs,(e.radius-map_.b)*.5);left|=std::abs(e.radius-map_.a)<match;right|=std::abs(e.radius-map_.b)<match;}
  left&=ls<width*.125;right&=rs<width*.125;
  auto add=[&](double a,double b,bool at_left,double scale,bool fold){const double limit=2*std::asinh(std::sqrt((b-a)/(2*scale)));const int n=std::max(1,int(std::ceil(limit)));for(int i=0;i<n;++i)cells_.push_back({at_left?2:3,{limit*i/n,limit*(i+1)/n,false,fold&&i+1==n},at_left?a:b,scale});};
  if(left&&right){double mid=map_.a+width*.5;add(map_.a,mid,true,ls,false);add(mid,map_.b,false,rs,false);}
  else if(left)add(map_.a,map_.b,true,ls,map_.right);
  else if(right)add(map_.a,map_.b,false,rs,map_.left);
  else cells_.push_back({0,weighted_?FoldRadialMap{map_.a,map_.b,false,false}:map_,0,1});
 }
 bool complex_cells(){
  if(charts_checked_)return false;LCBININT_COUPLED_SCOPE(chart);charts_checked_=true;
  std::vector<shape_gradient_detail::RadialComplexPair> pairs;
  for(const auto& entry:context_->points){if(entry.first<=map_.a||entry.first>=map_.b)continue;
   for(std::size_t i=0;i<entry.second.anchor.count;++i)for(double theta:{entry.second.anchor.arcs[i].lo,entry.second.anchor.arcs[i].hi}){
    auto pair=shape_gradient_detail::radial_complex_pair(entry.first,theta,map_.a,map_.b,p_);if(!pair)continue;bool known=false;
    for(const auto& old:pairs)if(std::abs(old.center-pair->center)<1e-6*(map_.b-map_.a))known=true;if(!known)pairs.push_back(*pair);
   }
  }
  if(pairs.empty())return false;std::sort(pairs.begin(),pairs.end(),[](auto a,auto b){return a.center<b.center;});cells_.clear();weighted_=false;
  for(std::size_t i=0;i<pairs.size();++i){const auto p=pairs[i];const double a=i?(pairs[i-1].center+p.center)*.5:map_.a,b=i+1<pairs.size()?(p.center+pairs[i+1].center)*.5:map_.b;
   const double lo=std::asinh((a-p.center)/p.scale),hi=std::asinh((b-p.center)/p.scale);const int n=std::max(1,int(std::ceil(hi-lo)));
   for(int k=0;k<n;++k)cells_.push_back({1,{lo+(hi-lo)*k/n,lo+(hi-lo)*(k+1)/n,i==0&&k==0&&map_.left,i+1==pairs.size()&&k+1==n&&map_.right},p.center,p.scale});
  }return true;
 }
public:
 static constexpr int split_level=6;
 std::string reason="stream_not_started";
 TransportPanel(const Geometry&p,double a,double b,bool left,bool right,AngularOptions options,std::shared_ptr<SharedRadiusCache<Gradient>> shared,double absolute,double relative,bool uniform,bool weighted,double limb,bool):p_(p),map_{a,b,left,right},options_(options),shared_(std::move(shared)),relative_(relative),absolute_(absolute),uniform_weight_(1-limb),limb_weight_(limb),safe_lo_(a),safe_hi_(b),uniform_(uniform),weighted_(weighted){zero_errors_.fill(INFINITY);}
 void set_events(const std::vector<RadialEvent>& e){events_=e;}
 void inherit_stream(const TransportPanel& parent){context_=parent.context_;owns_context_=false;events_=parent.events_;}
 void set_radial_motion(double origin,const std::array<double,8>& base,const std::array<double,8>& slope){origin_=origin;base_=base;slope_=slope;moving_=true;}
 bool has_radial_motion()const{return moving_;}double motion_origin()const{return origin_;}const auto& motion_base()const{return base_;}const auto& motion_slope()const{return slope_;}
 void set_certified_band(const AngularTopology& t,double lo,double hi){band_=t;safe_lo_=lo;safe_hi_=hi;}
 const auto& band_topology()const{return band_;}double safe_lo()const{return safe_lo_;}double safe_hi()const{return safe_hi_;}
 const auto& result()const{return accepted_;}double radial_error()const{return radial_error_;}double inner_error()const{return inner_error_;}
 const auto& gradient_radial_errors()const{return zero_errors_;}const auto& gradient_inner_errors()const{return zero_errors_;}
 std::size_t geometry_promotions()const{return 0;}
 const ShapeGradientResult& work()const{static const ShapeGradientResult empty{};return owns_context_?context_->work:empty;}
 bool refine(int requested,int angular){
  if(cells_.empty())real_cells();if(requested<=level_)return true;auto previous=accepted_;bool have=level_>0;
  for(int lev=std::max(1,level_+1);lev<=requested;++lev){RadiusIntegral<true> next;double inner=0,detail=0;
   for(const auto& cell:cells_){const bool weighted=weighted_&&cell.kind==0;const auto& rule=radial_rule(lev,weighted);const auto& limb_rule=weighted?linear_radial_rule(lev):rule;const auto& hu=fused_detail::radial_rule(lev,weighted?1:0);const auto& hl=fused_detail::radial_rule(lev,weighted?2:0);
    std::vector<RadiusIntegral<false>> samples(rule.x.size());
    for(std::size_t k=0;k<rule.x.size();++k){const double x=rule.x[k],shape=(1-x)*(1+x);const auto node=mapped(cell,x);Point* cached=nullptr;if(!point(node[0],cached))return false;const auto& value=cached->value;
     const auto& q=cached->spatial;
     const double scaleu=weighted?1/std::sqrt(shape):1,scalel=weighted?1/shape:1;
     const double t=cell.coordinates(x)[0],jt=cell.coordinates(x)[1],pi=std::acos(-1.);double jtp=0;
     if(cell.coordinates.left&&cell.coordinates.right)jtp=pi*pi*(cell.coordinates.b-cell.coordinates.a)/8*std::cos(pi*(1+x)*.5);
     else if(cell.coordinates.left)jtp=(cell.coordinates.b-cell.coordinates.a)*.5;else if(cell.coordinates.right)jtp=-(cell.coordinates.b-cell.coordinates.a)*.5;
     double jp=jtp;if(cell.kind==1)jp=cell.scale*(std::sinh(t)*jt*jt+std::cosh(t)*jtp);else if(cell.kind==2||cell.kind==3)jp=cell.scale*(std::cosh(t)*jt*jt+std::sinh(t)*jtp);
     const double au=jp*scaleu+(weighted?node[1]*scaleu*x/shape:0),al=jp*scalel+(weighted?2*x*node[1]*scalel/shape:0),bu=(cell.kind==3?-1:1)*node[1]*node[1]*scaleu,bl=(cell.kind==3?-1:1)*node[1]*node[1]*scalel;
     samples[k].f0=node[1]*scaleu*q.f0;samples[k].fh=node[1]*scalel*q.fh;
     const double wu=hu.value[k]*node[1]*scaleu+hu.derivative[k]*au,wr=hu.derivative[k]*bu,wh=hl.value[k]*node[1]*scalel+hl.derivative[k]*al,whr=hl.derivative[k]*bl;
     next.f0+=wu*q.f0+wr*q.r0;next.fh+=wh*q.fh+whr*q.rh;
     inner+=limb_weight_*(std::abs(wh)*cached->angular_error+std::abs(whr)*cached->radial_angular_error);
     for(int j=0;j<8;++j){next.df0[j]+=wu*(q.p0[j]+slope_[j]*q.f0)+wr*(q.rp0[j]+slope_[j]*q.r0);next.dfh[j]+=wh*(q.ph[j]+slope_[j]*q.fh)+whr*(q.rph[j]+slope_[j]*q.rh);}

    }
    if(lev>1){
     if(weighted)detail+=uniform_weight_*panel_detail::interpolation_detail(rule,samples,[](const auto& a){return a.f0;})+limb_weight_*panel_detail::interpolation_detail(limb_rule,samples,[](const auto& a){return a.fh;});
     else detail+=panel_detail::interpolation_detail(rule,samples,[&](const auto& a){return uniform_weight_*a.f0+limb_weight_*a.fh;});
    }
   }
   radial_error_=have?uniform_weight_*std::abs(next.f0-previous.f0)+limb_weight_*std::abs(next.fh-previous.fh):INFINITY;
   radial_error_=std::max(radial_error_,(QuadratureErrorPolicy::weighted_detail_floor/QuadratureErrorPolicy::difference_safety)*detail);
   // Never accept a one/three-node coincidence as a geometry certificate.
   if(lev<3)radial_error_=INFINITY;
   inner_error_=inner;next.angular_error=radial_error_+inner;next.gradient_error.fill(INFINITY);next.normalized_gradient_errors=true;
   previous=next;have=true;if(lev==2&&complex_cells()){}level_=lev;
  }
  accepted_=previous;angular_level_=angular;reason="stream_completed";return true;
 }
};
template<bool Gradient=true> IntegrationResult<Gradient> integrate_transport(const Geometry& p,double limb,IntegrationOptions options={},const RadialEventPlan* plan=nullptr){
 static_assert(Gradient,"dedicated derivative stream");options.gradient_value_mesh=true;
 return integrate_coupled<Gradient,TransportPanel>(p,limb,options,plan);
}
}

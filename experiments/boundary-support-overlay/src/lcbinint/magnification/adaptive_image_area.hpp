#pragma once
#include "adaptive_row_panels.hpp"
#include <queue>
#include "lcbinint/magnification/critical_phase_enclosure.hpp"
namespace lcbinint::magnification::detail {
// Area partition: each row band consists of an envelope minus its internal
// gaps. Terminal caps lie outside those bands. No Bennett weight is added.
// This constructs estimates for represented support; it is not a certificate.
class LensAdaptiveArea {
public:
 BoundaryTurnGeometry geometry;
 std::vector<CartesianRunBoundaryTraceEvent> coarse_rows;
 std::size_t map_evaluations=0,unresolved_topology=0,endpoint_caps=0,gap_caps=0,rotated_edges=0,small_images=0;
 double constant_area=0;
 std::vector<int> domain_kinds;
 std::vector<std::array<double,6>> root_failures;
 std::vector<std::array<double,5>> topology_failures;
 std::unique_ptr<AdaptiveRowPanels> panels;
 explicit LensAdaptiveArea(const BoundaryTurnGeometry& g,const std::vector<CartesianRunBoundaryTraceEvent>& rows,
   const std::vector<std::array<double,2>>& seeds={},double retained_spacing=0):geometry(g),coarse_rows(rows),retained_spacing_(retained_spacing) {
  if(rows.empty()&&retained_spacing>0&&!seeds.empty()){for(const auto& seed:seeds)add_small_image(seed[0],seed[1]);panels=std::make_unique<AdaptiveRowPanels>(std::move(domains));return;}
  if(rows.empty()||std::any_of(rows.begin(),rows.end(),[&](const auto& e){return e.fill_level!=0||!std::isfinite(e.cell_spacing)||e.cell_spacing<=0||e.cell_spacing!=rows.front().cell_spacing;})) {
   ++unresolved_topology;panels=std::make_unique<AdaptiveRowPanels>(std::vector<AdaptiveRowPanels::Domain>{});return;
  }
  const RowResidualGraph graph(rows);
  // Each connected bipartite group is integrated once per row band.
  for(const auto& [iy,indices]:graph.rows) {
   std::set<std::size_t> visited;
   for(auto start:indices) {
    if(visited.count(start)||graph.after[start].empty())continue;
    std::vector<std::size_t> low,high;std::queue<std::pair<std::size_t,bool>> queue;queue.push({start,false});
    std::set<std::pair<std::size_t,bool>> seen;
    while(!queue.empty()) {
     const auto [i,upper]=queue.front();queue.pop();if(!seen.insert({i,upper}).second)continue;
     if(upper) {high.push_back(i);for(auto j:graph.before[i])queue.push({j,false});}
     else {low.push_back(i);visited.insert(i);for(auto j:graph.after[i])queue.push({j,true});}
    }
    const auto ordered=[&](auto& nodes){std::sort(nodes.begin(),nodes.end(),[&](auto a,auto b){return rows[a].run.lo<rows[b].run.lo;});};
    ordered(low);ordered(high);if(low.empty()||high.empty()){++unresolved_topology;continue;}
    add_edge(edge(rows[low.front()],false),edge(rows[high.front()],false),-1);
    add_edge(edge(rows[low.back()],true),edge(rows[high.back()],true),1);
    std::vector<Span> a,b;
    for(std::size_t k=1;k<low.size();++k)a.push_back(gap(rows[low[k-1]],rows[low[k]]));
    for(std::size_t k=1;k<high.size();++k)b.push_back(gap(rows[high[k-1]],rows[high[k]]));
    std::vector<bool> used(b.size());
    for(const auto& x:a) {
     std::vector<std::size_t> matches;
     for(std::size_t j=0;j<b.size();++j)if(x.left<b[j].right&&b[j].left<x.right)matches.push_back(j);
     if(matches.empty())add_cap(x,-1,-1);
     else if(matches.size()==1&&!used[matches[0]]) {
      const auto j=matches[0];used[j]=true;
      add_edge({x.left,x.y,x.h},{b[j].left,b[j].y,b[j].h},1);
      add_edge({x.right,x.y,x.h},{b[j].right,b[j].y,b[j].h},-1);
     } else ++unresolved_topology;
    }
    for(std::size_t j=0;j<b.size();++j)if(!used[j])add_cap(b[j],1,-1);
   }
  }
  // Isolated physical tips. Their caps stop exactly at an existing row.
  for(std::size_t i=0;i<rows.size();++i) {
   if(graph.before[i].empty())add_cap(span(rows[i]),1,1);
   if(graph.after[i].empty())add_cap(span(rows[i]),-1,1);
  }
  for(const auto& seed:seeds)add_small_image(seed[0],seed[1]);
  panels=std::make_unique<AdaptiveRowPanels>(std::move(domains));
 }
 double area()const{return constant_area+panels->area();}
 bool certifies_interior_chord(double x0,double y0,double x1,double y1)const{return interior_chord(x0,y0,x1,y1);}
private:
 struct Point {double x,y,h;};
 struct Span {double left,right,y,h;};
 struct Turn {double x,y,coefficient;};
 std::vector<AdaptiveRowPanels::Domain> domains;double retained_spacing_=0;
 std::map<std::pair<double,double>,Point> polished;
 std::array<double,5> evaluate(double x,double y){++map_evaluations;return boundary_turn_values(geometry,x,y);}
 static Point edge(const CartesianRunBoundaryTraceEvent& e,bool right) {
  return {(right?e.run.hi+e.diagnostic.right_fraction:e.run.lo-e.diagnostic.left_fraction)*e.cell_spacing,e.run.iy*e.cell_spacing,e.cell_spacing};
 }
 static Span span(const CartesianRunBoundaryTraceEvent& e) {
  const auto a=edge(e,false),b=edge(e,true);return {a.x,b.x,a.y,a.h};
 }
 static Span gap(const CartesianRunBoundaryTraceEvent& a,const CartesianRunBoundaryTraceEvent& b) {
  const auto l=edge(a,true),r=edge(b,false);return {l.x,r.x,l.y,l.h};
 }
 double roundoff_indicator(double x,double y)const {
  using C=std::complex<double>;const C z{x,y},first=z-geometry.separation;
  const double scale=std::abs(z)+geometry.m1/std::abs(first)+geometry.m2/std::abs(z)+
      std::abs(C(geometry.shift+geometry.source_x,geometry.source_y));
  return 16*std::numeric_limits<double>::epsilon()*(geometry.rho2+std::sqrt(geometry.rho2)*scale);
 }
 bool polish(Point& p) {
  const auto key=std::make_pair(p.x,p.y);const auto found=polished.find(key);
  if(found!=polished.end()){p=found->second;return true;}
  const double original=p.x;
  for(int i=0;i<25;++i) {
   const auto v=evaluate(p.x,p.y);if(!std::isfinite(v[0])||!std::isfinite(v[1])||v[1]==0)return false;
   if(retained_spacing_>0&&std::abs(v[0])<=roundoff_indicator(p.x,p.y)){polished[key]=p;return true;}
   const double step=v[0]/v[1];p.x-=step;if(std::abs(p.x-original)>p.h)return false;
   if(std::abs(step)<=std::max(p.h*1e-12,64*std::numeric_limits<double>::epsilon()*std::abs(p.x))){polished[key]=p;return true;}
  }
  return false;
 }
 void add_edge(Point a,Point b,int sign) {
  if(!polish(a)||!polish(b)){topology_failures.push_back({1.,a.x,b.x,a.y,a.h});++unresolved_topology;return;}
  const double h=std::min(a.h,b.h);const auto va=evaluate(a.x,a.y),vb=evaluate(b.x,b.y);
  const bool graph_y=va[1]*vb[1]>0,graph_x=va[2]*vb[2]>0;
  if(!graph_y&&!graph_x){topology_failures.push_back({2.,a.x,b.x,a.y,a.h});++unresolved_topology;return;}
  const bool rotated=graph_x&&(!graph_y||std::abs(b.x-a.x)>h);
  const double p0=rotated?a.x:a.y,p1=rotated?b.x:b.y;
  if(p0==p1) {if(rotated)constant_area+=sign*(b.x*b.y-a.x*a.y);return;}
  const double orientation=p1>p0?1:-1,coefficient=rotated?-sign*orientation:sign*orientation;
  if(rotated){constant_area+=sign*(b.x*b.y-a.x*a.y);++rotated_edges;}
  auto positions=std::make_shared<std::map<double,double>>();
  (*positions)[p0]=rotated?a.y:a.x;(*positions)[p1]=rotated?b.y:b.x;
  auto values=std::make_shared<std::map<double,AdaptiveRowNode>>();
  const auto node=[this,a,b,h,rotated,p0,p1,coefficient,positions,values](double fixed) {
   const auto saved=values->find(fixed);if(saved!=values->end())return saved->second;
   auto high=positions->lower_bound(fixed);if(high==positions->end())--high;
   auto low=high;if(high!=positions->begin())--low;else if(std::next(high)!=positions->end())++high;
   double variable=low->first==high->first?low->second:low->second+(high->second-low->second)*(fixed-low->first)/(high->first-low->first);
   bool okay=false;
   for(int i=0;i<30;++i) {
    const auto v=rotated?evaluate(fixed,variable):evaluate(variable,fixed);
    const double derivative=rotated?v[2]:v[1];if(!std::isfinite(v[0])||!std::isfinite(derivative)||derivative==0)break;
    const double step=v[0]/derivative;variable-=step;
    if(rotated ? (variable<std::min(a.y,b.y)-3*h||variable>std::max(a.y,b.y)+3*h)
               : (variable<std::min(a.x,b.x)-3*h||variable>std::max(a.x,b.x)+3*h))break;
    if(std::abs(step)<=std::max(h*1e-12,64*std::numeric_limits<double>::epsilon()*std::abs(variable))){okay=true;break;}
   }
   const auto v=rotated?evaluate(fixed,variable):evaluate(variable,fixed);
   const double slope=rotated?-v[1]/v[2]:-v[2]/v[1];
   double uncertainty=0;
   if(!okay) {
    const double noise=rotated?roundoff_indicator(fixed,variable):roundoff_indicator(variable,fixed);
    if(std::isfinite(noise)&&std::abs(v[0])<=noise) {
     uncertainty=noise/std::abs(rotated?v[2]:v[1]);okay=std::isfinite(uncertainty);
    }
   }
   const AdaptiveRowNode result{fixed,coefficient*variable,coefficient*slope,okay&&std::isfinite(slope)&&std::abs(v[0])<geometry.rho2*1e-8,uncertainty};
   if(result.valid){(*positions)[fixed]=variable;(*values)[fixed]=result;}
   else root_failures.push_back({rotated?1.:0.,fixed,variable,v[0],rotated?v[2]:v[1],h});
   return result;
  };
  const double lo=std::min(p0,p1),hi=std::max(p0,p1);
  domains.push_back({node(lo),node(hi),node});domain_kinds.push_back(rotated?1:0);
 }
 bool turn(const Span& s,int direction,int sign,Turn& out) {
  double x=.5*(s.left+s.right),y=s.y-.5*direction*s.h;bool converged=false;
  const double xlo=s.left-s.h,xhi=s.right+s.h;
  for(int i=0;i<40;++i) {
   const auto v=evaluate(x,y);const double merit=v[0]*v[0]+s.h*s.h*v[1]*v[1];
   const double attainable=retained_spacing_>0?std::max(geometry.rho2*1e-11,roundoff_indicator(x,y)):geometry.rho2*1e-11;
   if(std::abs(v[0])<attainable&&std::abs(v[1])*s.h<(retained_spacing_>0?4:1)*attainable){converged=true;break;}
   const double det=v[1]*v[4]-v[2]*v[3];if(!std::isfinite(det)||det==0)return false;
   const double dx=(-v[0]*v[4]+v[2]*v[1])/det,dy=(v[0]*v[3]-v[1]*v[1])/det;
   bool accepted=false;
   for(double factor=1;factor>=1./256;factor*=.5) {
    const double nx=x+factor*dx,ny=y+factor*dy;
    if(nx<xlo||nx>xhi||std::abs(ny-s.y)>1.6*s.h)continue;
    const auto w=evaluate(nx,ny);if(w[0]*w[0]+s.h*s.h*w[1]*w[1]<merit){x=nx;y=ny;accepted=true;break;}
   }
   if(!accepted)return false;
  }
  if(!converged)return false;const auto v=evaluate(x,y);
  const double distance=direction*(s.y-y),c=-2*direction*v[2]/v[3];
  if(!(distance>0&&distance<1.6*s.h&&c>0&&sign*v[3]>0))return false;
  out={x,y,c};return true;
 }
 void add_cap(Span s,int direction,int sign) {
  Point a{s.left,s.y,s.h},b{s.right,s.y,s.h};
  if(!polish(a)||!polish(b)){topology_failures.push_back({1.,a.x,b.x,a.y,a.h});++unresolved_topology;return;}
  s.left=a.x;s.right=b.x;Turn endpoint;
  if(!turn(s,direction,sign,endpoint)){topology_failures.push_back({4.,s.left,s.right,s.y,s.h});++unresolved_topology;return;}
  cap(s,direction,sign,endpoint);
 }
 void cap(const Span& s,int direction,int sign,const Turn& endpoint) {
  const double end=std::sqrt(direction*(s.y-endpoint.y));
  auto roots_cache=std::make_shared<std::map<double,std::array<double,2>>>();
  (*roots_cache)[0]={endpoint.x,endpoint.x};(*roots_cache)[end]={s.left,s.right};
  auto values=std::make_shared<std::map<double,AdaptiveRowNode>>();
  const auto node=[this,s,direction,sign,endpoint,end,roots_cache,values](double u) {
   const auto saved=values->find(u);if(saved!=values->end())return saved->second;
   if(u==0)return AdaptiveRowNode{0,0,0,true};
   const double y=endpoint.y+direction*u*u;
      std::array<double,2> roots;double slope=0,uncertainty=0;bool okay=true;
   for(int side=0;side<2;++side) {
    auto high=roots_cache->lower_bound(u);if(high==roots_cache->end())--high;
    auto low=high;if(high!=roots_cache->begin())--low;else if(std::next(high)!=roots_cache->end())++high;
    double x=low->first==high->first?low->second[side]:low->second[side]+(high->second[side]-low->second[side])*(u-low->first)/(high->first-low->first);
    if(low->first==0&&high->first>0){const double leading=(side==0?-1:1)*std::sqrt(endpoint.coefficient);const double t=u/high->first;x=endpoint.x+leading*u+(high->second[side]-endpoint.x-leading*high->first)*t*t;}
    bool done=false;
    for(int i=0;i<30;++i) {
     const auto v=evaluate(x,y);if(!std::isfinite(v[0])||!std::isfinite(v[1])||v[1]==0)break;
     const double step=v[0]/v[1];x-=step;if(x<s.left-s.h||x>s.right+s.h)break;
     if(std::abs(step)<=std::max(s.h*1e-12,64*std::numeric_limits<double>::epsilon()*std::abs(x))){done=true;break;}
    }
    const auto v=evaluate(x,y);slope+=(side==0?1:-1)*v[2]/v[1];roots[side]=x;
    if(!done) {
     const double noise=roundoff_indicator(x,y);
     if(std::isfinite(noise)&&std::abs(v[0])<=noise&&std::isfinite(noise/v[1])){done=true;uncertainty+=2*u*noise/std::abs(v[1]);}
    }
    if(!done||!std::isfinite(slope)||std::abs(v[0])>geometry.rho2*1e-8) {okay=false;root_failures.push_back({sign>0?2.:3.,u,x,v[0],v[1],s.h});}
   }
   const double width=roots[1]-roots[0];
   const AdaptiveRowNode result{u,sign*2*u*width,sign*(2*width+4*direction*u*u*slope),okay&&width>0,uncertainty};
   if(result.valid){(*roots_cache)[u]=roots;(*values)[u]=result;}
   return result;
  };
  domains.push_back({node(0),node(end),node});domain_kinds.push_back(sign>0?2:3);
  if(sign>0)++endpoint_caps;else ++gap_caps;
 }
 // Certify a short interior chord: endpoint source distances plus a bound
 // on deviation from their source-plane chord. Used only for a nearby seed.
 bool interior_chord(double x0,double y0,double x1,double y1,unsigned depth=0)const {
  using namespace critical_enclosure_detail;
  if(!supported_arithmetic())return false;
  const auto norm=[](C z){const I v=math::square(z.x)+math::square(z.y);return I(math::down(std::sqrt(std::max(0.,v.lo))),math::up(std::sqrt(std::max(0.,v.hi))));};
  const C a{I(x0),I(y0)},b{I(x1),I(y1)},delta=subtract(b,a),mid=scale(add(a,b),I(.5));const I half=norm(delta)/I(2);
  const I d1=norm(subtract(mid,{I(geometry.separation),I(0)}))-half,d2=norm(mid)-half;
  if(d1.lo<=0||d2.lo<=0)return false;
  // Lens map uses conjugate reciprocal. Conjugate before reciprocal.
  const auto mapped=[&](C z){const C bar{z.x,-z.y};return subtract(subtract(subtract(z,scale(reciprocal(subtract(bar,{I(geometry.separation),I(0)})),I(geometry.m1))),scale(reciprocal(bar),I(geometry.m2))),{I(geometry.shift)+I(geometry.source_x),I(geometry.source_y)});};
  const I length2=math::square(delta.x)+math::square(delta.y);
  const I defect=length2*(I(geometry.m1)/(I(d1.lo)*I(d1.lo)*I(d1.lo))+I(geometry.m2)/(I(d2.lo)*I(d2.lo)*I(d2.lo)))/I(4);
  const double endpoints=std::max(norm(mapped(a)).hi,norm(mapped(b)).hi);
  if((I(endpoints)+defect).hi<math::down(std::sqrt(geometry.rho2)))return true;
  if(depth>=3)return false;const double mx=x0+(x1-x0)*.5,my=y0+(y1-y0)*.5;
  return interior_chord(x0,y0,mx,my,depth+1)&&interior_chord(mx,my,x1,y1,depth+1);
 }
 void add_small_image(double x,double y) {
  if(coarse_rows.empty()&&!(retained_spacing_>0)){++unresolved_topology;return;}
  const double h=coarse_rows.empty()?retained_spacing_:coarse_rows.front().cell_spacing;const auto ix=std::llround(x/h),iy=std::llround(y/h);
  for(const auto& e:coarse_rows)if(e.run.iy==iy&&e.run.lo<=ix&&ix<=e.run.hi)return;
  using C=std::complex<double>;const C z{x,y},k=geometry.m1/((z-geometry.separation)*(z-geometry.separation))+geometry.m2/(z*z);
  const double rho=std::sqrt(geometry.rho2),ex=rho/std::abs(1.+k),ey=rho*std::abs(1.+k)/std::abs(1-std::norm(k));
  if(!(std::isfinite(ex)&&std::isfinite(ey)&&((2*ex<h&&2*ey<h&&std::abs(k)>2)||(retained_spacing_>0&&std::max(ex,ey)<2*h)))){topology_failures.push_back({5.,x,y,ex,ey});++unresolved_topology;return;}
  bool nearby=false;
  for(const auto& e:coarse_rows)if(std::abs(e.run.iy*h-y)<3*ey&&e.run.lo*h<x+3*(ex+ey)&&e.run.hi*h>x-3*(ex+ey)){
   const double px=std::clamp<std::int64_t>(std::llround(x/h),e.run.lo,e.run.hi)*h;
   if(retained_spacing_>0&&interior_chord(x,y,px,e.run.iy*h))return;
   nearby=true;
  }
  if(nearby){topology_failures.push_back({6.,x,y,ex,ey});++unresolved_topology;return;}
  // The coarse spacing can exceed this image's size by an order of magnitude.
  // Start its turn solve on the image's own Jacobian scale, not half a coarse cell.
  Span s{x-ex,x+ex,y,retained_spacing_>0?2*std::max(ex,ey):std::min(h,2*std::max(ex,ey))};Point a{s.left,y,s.h},b{s.right,y,s.h};
  if(!polish(a)||!polish(b)){topology_failures.push_back({1.,a.x,b.x,a.y,a.h});++unresolved_topology;return;}s.left=a.x;s.right=b.x;
  Turn lower,upper;if(!turn(s,1,1,lower)||!turn(s,-1,1,upper)){topology_failures.push_back({8.,x,y,ex,ey});++unresolved_topology;return;}
  cap(s,1,1,lower);cap(s,-1,1,upper);++small_images;
 }
};
}

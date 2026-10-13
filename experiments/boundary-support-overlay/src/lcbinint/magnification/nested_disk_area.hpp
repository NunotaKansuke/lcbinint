#pragma once
#include "adaptive_image_area.hpp"
#include <set>
#include <deque>
#include <unordered_map>
namespace lcbinint::magnification::detail {
// Nested source disks are subsets of outer support. Reuse the represented
// outer lattice and centre images; completeness remains uncertified.
class NestedDiskArea {
public:
 struct Samples {std::vector<std::pair<std::int64_t,double>> cells;};
 BoundaryTurnGeometry geometry;double spacing=0;
 std::map<std::int64_t,Samples> rows;
 std::vector<std::array<double,2>> seeds;
 std::map<double,std::unique_ptr<LensAdaptiveArea>> states;
 std::size_t lattice_map_evaluations=0,crossing_map_evaluations=0,requests=0,unsupported=0;
 struct PairHash {std::size_t operator()(const std::pair<double,double>& p)const {const auto a=std::hash<double>{}(p.first),b=std::hash<double>{}(p.second);return a^(b+0x9e3779b97f4a7c15ULL+(a<<6)+(a>>2));}};
 std::unordered_map<std::pair<double,double>,double,PairHash> distance_cache;
 NestedDiskArea(const BoundaryTurnGeometry& g,const std::vector<CartesianRunBoundaryTraceEvent>& outer,const std::vector<std::array<double,2>>& centre):geometry(g),seeds(centre){
  distance_cache.reserve(16384);if(outer.empty()){++unsupported;return;}spacing=outer.front().cell_spacing;
  std::map<std::int64_t,std::set<std::int64_t>> cells;
  for(const auto& e:outer){if(e.fill_level||e.cell_spacing!=spacing||spacing<=0){++unsupported;return;}
   for(auto ix=e.run.lo-1;ix<=e.run.hi+1;++ix)cells[e.run.iy].insert(ix);
  }
  for(const auto& [iy,xs]:cells)for(auto ix:xs){
   const double distance=distance_at(ix*spacing,iy*spacing);rows[iy].cells.push_back({ix,distance});
  }
  states[std::sqrt(g.rho2)]=std::make_unique<LensAdaptiveArea>(g,outer,seeds);
 }
 std::array<double,2> area(double radius,double budget){
  ++requests;if(unsupported||!(radius>0)||radius>std::sqrt(geometry.rho2)*(1+1e-14))return {NAN,INFINITY};
  auto found=states.find(radius);
  if(found==states.end()) {
   std::vector<CartesianRunBoundaryTraceEvent> events;const double r2=radius*radius;
   const int level=std::max(0,static_cast<int>(std::ceil(std::log2(std::sqrt(geometry.rho2)/radius))));
   if(level>20)return {NAN,INFINITY};const double h=std::ldexp(spacing,-level);
   std::vector<CartesianLatticeSeed> offered;
   for(const auto& seed:seeds){const auto ix=std::llround(seed[0]/h),iy=std::llround(seed[1]/h);for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)offered.push_back({ix+dx,iy+dy});}
   for(const auto& [iy,row]:rows){bool previous=false;std::int64_t last=0;for(const auto& [ix,distance]:row.cells){const bool in=distance<r2;if(in&&(!previous||ix!=last+1))offered.push_back({ix*(1LL<<level),iy*(1LL<<level)});previous=in;last=ix;}}
   const auto classify=[&](std::int64_t ix,std::int64_t iy){return distance_at(ix*h,iy*h);};
   const auto boundary=[&](const CartesianRun& run,double,double,double,double){
    CartesianRunBoundaryTraceEvent e;e.run=run;e.cell_spacing=h;
    e.diagnostic.left_fraction=(run.lo*h-crossing(run.lo,run.lo-1,run.iy,h,r2))/h;
    e.diagnostic.right_fraction=(crossing(run.hi,run.hi+1,run.iy,h,r2)-run.hi*h)/h;events.push_back(e);return CartesianBoundaryContribution{};
   };
   CartesianRunFillLimits limits;limits.maximum_evaluations=200000;limits.maximum_runs=20000;
   const auto fill=fill_cartesian_runs<double>(std::move(offered),classify,[&](double v){return v<r2;},[](std::int64_t,std::int64_t,double){return 0.;},boundary,limits);
   if(!fill.ok())return {NAN,INFINITY};


   auto g=geometry;g.rho2=r2;found=states.emplace(radius,std::make_unique<LensAdaptiveArea>(g,events,seeds,h)).first;
  }
  auto& state=*found->second;const bool met=state.panels->refine(budget,40000);
  if(!met||state.unresolved_topology)return {NAN,INFINITY};return {state.area(),state.panels->error()};
 }
private:
 double crossing(std::int64_t inner,std::int64_t outer,std::int64_t iy,double h,double r2){
  double a=inner*h,b=outer*h;const double ga=distance_at(a,iy*h)-r2,gb=distance_at(b,iy*h)-r2;
  double x=a+(b-a)*(-ga)/(gb-ga);auto g=geometry;g.rho2=r2;
  for(int i=0;i<30;++i){const auto v=boundary_turn_values(g,x,iy*h);++crossing_map_evaluations;
   if(v[0]<0)a=x;else b=x;const double step=v[0]/v[1];
   if(std::isfinite(step)&&std::abs(step)<=std::max(h*1e-12,64*std::numeric_limits<double>::epsilon()*std::abs(x)))break;
   const double next=x-step;x=std::isfinite(next)&&next>std::min(a,b)&&next<std::max(a,b)?next:a+(b-a)*.5;
  }
  return x;
 }
 double distance_at(double x,double y){const auto key=std::make_pair(x,y);auto found=distance_cache.find(key);if(found!=distance_cache.end())return found->second;
  using C=std::complex<double>;const C z{x,y};const C f=z-geometry.m1/(std::conj(z)-geometry.separation)-geometry.m2/std::conj(z)-geometry.shift-C{geometry.source_x,geometry.source_y};
  const double value=std::norm(f);const double distance=std::isfinite(value)?value:INFINITY;distance_cache[key]=distance;++lattice_map_evaluations;return distance;
 }

};
}

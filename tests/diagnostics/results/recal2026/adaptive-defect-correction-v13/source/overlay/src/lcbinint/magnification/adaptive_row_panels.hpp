#pragma once
#include "global_row_residual.hpp"
#include <functional>
#include <memory>
#include <optional>
#include <queue>
namespace lcbinint::magnification::detail {
// Local Hermite model. Estimates are not certified Peano remainder bounds.
struct AdaptiveRowNode { double y=0,width=0,slope=0; bool valid=true; double uncertainty=0; };
struct AdaptiveRowLeaf {
 std::size_t a=0,m=0,b=0;unsigned depth=0;
 double area=0,error=INFINITY,order=4;bool valid=false;
};
class AdaptiveRowPanels {
public:
 using Evaluator=std::function<AdaptiveRowNode(double)>;
 struct Domain {AdaptiveRowNode a,b;Evaluator evaluate;std::optional<AdaptiveRowNode> midpoint;};
 explicit AdaptiveRowPanels(std::vector<Domain> domains):domains_(std::move(domains)) {
  for(std::size_t d=0;d<domains_.size();++d) {
   const auto& domain=domains_[d];const auto a=retain(domain.a),b=retain(domain.b);
   const auto m=retain(domain.midpoint?*domain.midpoint:domain.evaluate(.5*(domain.a.y+domain.b.y)));
   if(!domain.midpoint)++new_evaluations;
   append(d,model(a,m,b,0,4));
  }
 }
 // Each split adds only the two quarter nodes; endpoints and midpoint survive.
 bool refine(double tolerance,std::size_t max_evaluations) {
  if(!std::isfinite(tolerance)||tolerance<0||unresolved())return false;
  while(error()>tolerance) {
   while(!priority_.empty() &&
     (priority_.top().index>=leaves_.size() || priority_.top().revision!=revisions_[priority_.top().index]))priority_.pop();
   const std::size_t best=priority_.empty()?leaves_.size():priority_.top().index;
   if(best==leaves_.size()||new_evaluations+2>max_evaluations)return false;
   priority_.pop();
   const auto [domain,parent]=leaves_[best];const auto a=nodes_[parent.a],m=nodes_[parent.m],b=nodes_[parent.b];
   const auto left=retain(domains_[domain].evaluate(.5*(a.y+m.y)));
   const auto right=retain(domains_[domain].evaluate(.5*(m.y+b.y)));new_evaluations+=2;
   auto lo=model(parent.a,left,parent.m,parent.depth+1,parent.order);
   auto hi=model(parent.m,right,parent.b,parent.depth+1,parent.order);
   if(!lo.valid||!hi.valid) {
    error_sum_-=parent.error;leaves_[best].second.valid=false;leaves_[best].second.error=INFINITY;
    ++unresolved_;++revisions_[best];return false;
   }
   // Local shape consistency, using already requested quarter samples. This
   // does not measure convergence of successive whole-source grids.
   if(lo.error+hi.error>parent.error*.5)lo.order=hi.order=2;
   area_sum_+=static_cast<long double>(lo.area)-parent.area;
   error_sum_+=static_cast<long double>(lo.error)-parent.error;
   leaves_[best]={domain,lo};++revisions_[best];offer(best);
   append(domain,hi);++splits;
  }
  return std::isfinite(error());
 }
 double area()const {return static_cast<double>(area_sum_);}
 double error()const {return unresolved_?INFINITY:std::max(0.,static_cast<double>(error_sum_));}
 std::size_t unresolved()const {return unresolved_;}
 std::size_t leaves()const{return leaves_.size();}
 std::size_t retained_nodes()const{return nodes_.size();}
 std::vector<std::array<double,6>> invalid_details()const {
  std::vector<std::array<double,6>> out;
  for(const auto& [domain,p]:leaves_)if(!p.valid)out.push_back({static_cast<double>(domain),nodes_[p.a].y,nodes_[p.m].y,nodes_[p.b].y,nodes_[p.m].width,nodes_[p.m].slope});
  return out;
 }
 // Predict equal-width refinement from E_new ≈ E_old / factor^p, per leaf.
 std::size_t predicted_splits(double tolerance)const {
  if(!(tolerance>0)||!std::isfinite(error()))return 0;
  std::size_t count=0;const double share=tolerance/std::max<std::size_t>(1,leaves_.size());
  for(const auto& p:leaves_) {
   const double factor=std::pow(std::max(1.,p.second.error/share),1./p.second.order);
   count+=static_cast<std::size_t>(std::min(1048576.,std::ceil(factor)-1));
  }
  return count;
 }
 std::size_t new_evaluations=0,splits=0;
private:
 std::vector<Domain> domains_;std::vector<AdaptiveRowNode> nodes_;
 std::vector<std::pair<std::size_t,AdaptiveRowLeaf>> leaves_;
 struct Priority {double gain;std::size_t index,revision;
  bool operator<(const Priority& other)const {return gain==other.gain?index>other.index:gain<other.gain;}
 };
 std::priority_queue<Priority> priority_;std::vector<std::size_t> revisions_;
 long double area_sum_=0,error_sum_=0;std::size_t unresolved_=0;
 void offer(std::size_t index) {
  const auto& p=leaves_[index].second;
  if(p.valid&&p.depth<20&&p.error>0)priority_.push({p.error*(1-std::pow(2.,-p.order)),index,revisions_[index]});
 }
 void append(std::size_t domain,const AdaptiveRowLeaf& p) {
  leaves_.push_back({domain,p});revisions_.push_back(0);area_sum_+=p.area;
  if(p.valid)error_sum_+=p.error;else ++unresolved_;
  offer(leaves_.size()-1);
 }
 std::size_t retain(AdaptiveRowNode node){nodes_.push_back(node);return nodes_.size()-1;}
 AdaptiveRowLeaf model(std::size_t a,std::size_t m,std::size_t b,unsigned depth,double order)const {
  const auto& u=nodes_[a];const auto& v=nodes_[m];const auto& w=nodes_[b];const double h=w.y-u.y;
  const double area=h*(u.width+w.width)/2+h*h*(u.slope-w.slope)/12;
  const double middle=(u.width+w.width)/2+h*(u.slope-w.slope)/8;
  const double middle_slope=1.5*(w.width-u.width)/h-.25*(u.slope+w.slope);
  // Quintic Hermite model: the fourth derivative is linear. Its maximum
  // in the nonnegative Peano kernel gives the variation term below.
  // This bounds that model, not the true derivative remainder.
  const double value_error=std::abs(v.width-middle)*h*(8./15);
  const double variation_error=std::abs(v.slope-middle_slope)*h*h*(4./3);
  const double residual=value_error+
      variation_error+
      h*(.5*(u.uncertainty+w.uncertainty)+v.uncertainty);
  const bool valid=u.valid&&v.valid&&w.valid&&h>0&&std::isfinite(area)&&std::isfinite(residual);
  return {a,m,b,depth,area,valid?residual:INFINITY,order,valid};
 }
};
struct LensAdaptiveRows {
 BoundaryTurnGeometry geometry;
 std::vector<CartesianRunBoundaryTraceEvent> coarse_rows; // Includes excluded topology for incremental later repair.
 std::vector<AdaptiveRowPanels::Domain> domains;
 std::size_t map_evaluations=0,excluded_intervals=0;
 std::unique_ptr<AdaptiveRowPanels> panels;
 explicit LensAdaptiveRows(const BoundaryTurnGeometry& g,const std::vector<CartesianRunBoundaryTraceEvent>& events,bool reuse_two_rows=false):geometry(g),coarse_rows(events) {
  const RowResidualGraph graph(events);
  std::vector<std::optional<AdaptiveRowNode>> endpoints(events.size());
  std::set<std::size_t> consumed;
  for(std::size_t i=0;i<events.size();++i) {
   if(consumed.count(i)||graph.after[i].empty())continue;
   if(graph.after[i].size()!=1) {++excluded_intervals;continue;}
   auto j=graph.after[i][0];if(graph.before[j].size()!=1){++excluded_intervals;continue;}
   const auto a=events[i];const double h=a.cell_spacing;
   std::optional<std::size_t> middle;
   if(reuse_two_rows&&graph.after[j].size()==1) {
    const auto k=graph.after[j][0];
    const auto& c=events[k];const auto& b=events[j];
    if(graph.before[k].size()==1&&c.fill_level==0&&b.fill_level==0&&
       std::max(std::abs((c.run.lo-c.diagnostic.left_fraction)-(a.run.lo-a.diagnostic.left_fraction)),
                std::abs((c.run.hi+c.diagnostic.right_fraction)-(a.run.hi+a.diagnostic.right_fraction)))<=1) {
     middle=j;consumed.insert(j);j=k;
    }
   }
   const auto b=events[j];
   const double xa=(a.run.lo-a.diagnostic.left_fraction)*h,xb=(b.run.lo-b.diagnostic.left_fraction)*h;
   const double ua=(a.run.hi+a.diagnostic.right_fraction)*h,ub=(b.run.hi+b.diagnostic.right_fraction)*h;
   // Do not declare moving, unresolved topology to be a regular patch.
   if(std::max(std::abs(xa-xb),std::abs(ua-ub))>h ||a.fill_level||b.fill_level) {++excluded_intervals;continue;}
   const double ya=a.run.iy*h,yb=b.run.iy*h;
   const auto node=[this,xa,xb,ua,ub,ya,yb,h](double y,bool solve) {
    std::array<double,2> roots={xa+(xb-xa)*(y-ya)/(yb-ya),ua+(ub-ua)*(y-ya)/(yb-ya)};
    double slope=0;bool okay=true;
    for(int side=0;side<2;++side) {
     double& x=roots[side];bool done=!solve;
     if(solve)for(int n=0;n<25;++n) {
      ++map_evaluations;const auto v=boundary_turn_values(geometry,x,y);
      if(!std::isfinite(v[0])||!std::isfinite(v[1])||v[1]==0)break;
      const double step=v[0]/v[1];x-=step;
      if(x<std::min(xa,xb)-h||x>std::max(ua,ub)+h)break;
      if(std::abs(step)<h*1e-11) {done=true;break;}
     }
     ++map_evaluations;const auto v=boundary_turn_values(geometry,x,y);
     if(!done||!std::isfinite(v[1])||v[1]==0||std::abs(v[0])>geometry.rho2*1e-6)okay=false;
     slope+=(side==0?1:-1)*v[2]/v[1];
    }
    return AdaptiveRowNode{y,roots[1]-roots[0],slope,okay&&roots[1]>roots[0]};
   };
   if(!endpoints[i])endpoints[i]=node(ya,true);if(!endpoints[j])endpoints[j]=node(yb,true);
   std::optional<AdaptiveRowNode> retained_middle;
   if(middle) {
    const auto k=*middle;
    if(!endpoints[k])endpoints[k]=node(events[k].run.iy*h,true);
    retained_middle=endpoints[k];
   }
   domains.push_back({*endpoints[i],*endpoints[j],[node](double y){return node(y,true);},retained_middle});
  }
  panels=std::make_unique<AdaptiveRowPanels>(std::move(domains));
 }
};
}

#pragma once
#include "cartesian_run_fill.hpp"
#include "outer_endpoint_model.hpp"
#include <map>
#include <set>
namespace lcbinint::magnification::detail {
// Uncertified single-grid candidate. Does not establish component coverage,
// regularity, endpoint accuracy, or an integration error upper bound.
struct CartesianRowResidualCandidate {
 double endpoint=0,smooth=0,smooth_absolute=0,gap=0,max_fourth_defect=0;
 std::size_t panels=0,gap_panels=0,unresolved_rows=0;
 double total() const {return endpoint+smooth+gap;}
};
struct RowResidualGraph {
 std::map<std::int64_t,std::vector<std::size_t>> rows;
 std::vector<std::vector<std::size_t>> before,after;
 explicit RowResidualGraph(const std::vector<CartesianRunBoundaryTraceEvent>& events,bool geometric_gaps=false)
  :before(events.size()),after(events.size()) {
  for(std::size_t i=0;i<events.size();++i) rows[events[i].run.iy].push_back(i);
  for(const auto& [y,indices]:rows) {
   if(y==std::numeric_limits<std::int64_t>::max()) continue;
   const auto next=rows.find(y+1);if(next==rows.end())continue;
   for(auto i:indices)for(auto j:next->second) {
    const auto& a=events[i].run;const auto& b=events[j].run;
    const bool overlap=geometric_gaps
      ? (static_cast<long double>(a.lo)-events[i].diagnostic.left_fraction < static_cast<long double>(b.hi)+events[j].diagnostic.right_fraction &&
         static_cast<long double>(b.lo)-events[j].diagnostic.left_fraction < static_cast<long double>(a.hi)+events[i].diagnostic.right_fraction)
      : (static_cast<long double>(a.lo)<=static_cast<long double>(b.hi)+1 &&
         static_cast<long double>(b.lo)<=static_cast<long double>(a.hi)+1);
    if(overlap) {
      after[i].push_back(j);before[j].push_back(i);
    }
   }
  }
 }
};
template<class Consumer>
inline void visit_row_endpoint_panels(const std::vector<CartesianRunBoundaryTraceEvent>& events,
 const RowResidualGraph& graph,Consumer consumer) {
 for(std::size_t i=0;i<events.size();++i)for(int direction:{1,-1}) {
  const auto& links=direction==1?graph.after:graph.before;
  const auto& opposite=direction==1?graph.before:graph.after;
  if(!opposite[i].empty())continue;
  std::array<std::size_t,4> panel{i,0,0,0};bool eligible=true;
  for(int k=1;k<4;++k) {
   if(links[panel[k-1]].size()!=1) {eligible=false;break;}
   panel[k]=links[panel[k-1]][0];
   if(opposite[panel[k]].size()!=1) {eligible=false;break;}
  }
  if(eligible)consumer(direction,panel);
 }
}
inline OuterEndpointModel row_endpoint_model(const std::vector<CartesianRunBoundaryTraceEvent>& events,
 const std::array<std::size_t,4>& panel) {
 const double h=events[panel[0]].cell_spacing;
 if(!(h>0))return {};
 std::array<double,4> widths;
 for(int k=0;k<4;++k) {
  if(events[panel[k]].cell_spacing!=h||events[panel[k]].fill_level!=0)return {};
  // Internal trace areas are dimensionless lattice sums, unlike Python exports.
  widths[k]=events[panel[k]].midpoint_area*h;
 }
 return outer_endpoint_quadratic_model(widths,h,events[panel[0]].outer_weight,events[panel[1]].outer_weight);
}
inline CartesianRowResidualCandidate cartesian_row_residual_candidate(
 const std::vector<CartesianRunBoundaryTraceEvent>& events) {
 CartesianRowResidualCandidate result;const RowResidualGraph graph(events);
 std::set<std::size_t> used;
 visit_row_endpoint_panels(events,graph,[&](int,const auto& panel) {
  if(used.count(panel[0])||used.count(panel[1]))return;
  const auto model=row_endpoint_model(events,panel);if(!model.resolved)return;
  result.endpoint+=model.signed_area_residual;++result.panels;
  result.max_fourth_defect=std::max(result.max_fourth_defect,model.fourth_relative_defect);
  used.insert(panel[0]);used.insert(panel[1]);
 });
 for(std::size_t i=0;i<events.size();++i) {
  if(used.count(i))continue;
  if(graph.before[i].size()!=1||graph.after[i].size()!=1) {++result.unresolved_rows;continue;}
  const auto prev=graph.before[i][0],next=graph.after[i][0];
  if(graph.after[prev].size()!=1||graph.before[next].size()!=1) {++result.unresolved_rows;continue;}
  const auto area=[&](std::size_t j){return events[j].midpoint_area*events[j].cell_spacing*events[j].cell_spacing;};
  const double residual=(area(prev)-2*area(i)+area(next))/24;
  result.smooth+=residual;result.smooth_absolute+=std::abs(residual);
 }
 std::vector<CartesianRunBoundaryTraceEvent> gaps;
 for(const auto& [y,indices]:graph.rows) {
  auto ordered=indices;
  std::sort(ordered.begin(),ordered.end(),[&](auto i,auto j){return events[i].run.lo<events[j].run.lo;});
  for(std::size_t k=1;k<ordered.size();++k) {
   const auto& left=events[ordered[k-1]];const auto& right=events[ordered[k]];
   const double h=left.cell_spacing;
   const double width=(static_cast<double>(right.run.lo)-right.diagnostic.left_fraction-
     static_cast<double>(left.run.hi)-left.diagnostic.right_fraction)*h;
   if(!(std::isfinite(width)&&width>0)||left.fill_level!=0||right.fill_level!=0)continue;
   CartesianRunBoundaryTraceEvent gap;
   gap.run={y,left.run.hi,right.run.lo};gap.midpoint_area=width/h;gap.cell_spacing=h;
   gaps.push_back(gap);
  }
 }
 const RowResidualGraph gap_graph(gaps);
 visit_row_endpoint_panels(gaps,gap_graph,[&](int direction,const auto& panel) {
  const auto& first=gaps[panel[0]].run;
  if((direction>0&&first.iy==std::numeric_limits<std::int64_t>::min())||
     (direction<0&&first.iy==std::numeric_limits<std::int64_t>::max()))return;
  const auto parent=graph.rows.find(first.iy-direction);if(parent==graph.rows.end())return;
  bool enclosed=false;
  for(auto i:parent->second) if(events[i].run.lo<=first.lo&&events[i].run.hi>=first.hi)enclosed=true;
  if(!enclosed)return;
  const auto model=row_endpoint_model(gaps,panel);if(!model.resolved)return;
  result.gap-=model.signed_area_residual;++result.gap_panels;
 });
 return result;
}
}

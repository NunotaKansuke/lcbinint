#pragma once
#include "radial_stationary_atlas.hpp"
#include "cells.hpp"
namespace lcbinint::holonomic {
inline TopologyResult classify_cells_from_atlas(const PrimaryFrame& pf,const RadialAtlasResult& atlas){
 if(atlas.status!=AtlasStatus::Complete){TopologyResult t;t.status=Status::TOPOLOGY_UNCERTAIN;return t;}
 std::vector<RadialEvent> events;
 std::vector<const ContactAnchor*> sorted;for(const auto& c:atlas.contacts)sorted.push_back(&c);
 std::sort(sorted.begin(),sorted.end(),[](auto a,auto b){return a->R.lo<b->R.lo;});
 for(size_t i=0;i<sorted.size();++i){const auto& c=*sorted[i];
  if(i&&sorted[i-1]->R.hi>=c.R.lo){
   // Exact reflection is the only simultaneous-contact grouping here.
   const auto& prev=*sorted[i-1];ContactAnchor reflected=prev;
   reflected.uniqueness.s0=-prev.uniqueness.s1;reflected.uniqueness.s1=-prev.uniqueness.s0;
   reflected.s={-prev.s.hi,-prev.s.lo};reflected.s_center=-prev.s_center;
   bool symmetry=pf.Y==0&&atlas_detail::same_contact(pf,reflected,c);
   if(symmetry)continue;
   TopologyResult t;t.status=Status::TOPOLOGY_UNCERTAIN;return t;
  }
  double r=double(c.R_center);
  // CellPlan still uses binary64 endpoints. Distinct certified contacts
  // that round to one endpoint MUST NOT be merged by the common classifier.
  if(!events.empty() && !(r>events.back().radius)){
   TopologyResult t;t.status=Status::TOPOLOGY_UNCERTAIN;return t;
  }
  RadialEvent e{r,"physical_real",true,"stationary atlas unique contact"};
  e.radius_lo=double(d14_to_qf(c.R_center)-(__float128)r);
  e.radius_uncertainty=std::nextafter(double(fmaxq(atlas_detail::up(d14_to_qf(c.R_center)-c.R.lo),atlas_detail::up(c.R.hi-d14_to_qf(c.R_center)))),INFINITY);
  e.atlas_anchor=true;e.positive_certified=true;e.positive_root_id=int(c.id);e.precision_tier=2;e.certified_radius_lo=c.R.lo;e.certified_radius_hi=c.R.hi;
  __float128 seed=d14_to_qf(c.s_center);if(c.uniqueness.chart && seed!=0)seed=-1/seed;
  e.fold_t_seed=double(seed);e.fold_t_seed_valid=!c.uniqueness.chart||d14_to_qf(c.s_center)!=0;
  events.push_back(e);
 }
 auto add=[&](double r,const char* kind){if(r>0&&r<atlas.rmax)events.push_back({r,kind,false,"representation cut"});};
 add(pf.a,"R_eq_a");add(std::sqrt(pf.m0),"R_eq_sqrt_m0");
 for(double r:re_detail::chart_p4_factor_roots(pf))add(r,"chart_p4");
 if(pf.Y==0){double a=pf.a,x=pf.X;double q[3]={a-x,-(a-x)*a*a-a,a*a*a*pf.m0};int n=2;if(q[0]==0){q[0]=q[1];q[1]=q[2];n=1;}if(q[0]!=0)for(double v:positive_real_roots(aberth<double>(q,n,80),1e-8,1e-9))add(std::sqrt(v),"L_root");}
 return classify_event_cells(pf,std::move(events),atlas.rmax,{},true);
}
} // namespace

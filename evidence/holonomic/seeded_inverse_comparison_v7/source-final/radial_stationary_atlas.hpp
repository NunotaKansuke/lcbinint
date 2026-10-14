#pragma once
#include <variant>
#include <vector>
#include "radial_contact.hpp"
namespace lcbinint::holonomic {
enum class AtlasStatus { Complete,AtlasIncomplete,SingularContact,EventClusterUnresolved,ArithmeticUncertain,RepresentationLimited,BudgetExceeded };
inline const char* atlas_status_name(AtlasStatus s){switch(s){case AtlasStatus::Complete:return "Complete";case AtlasStatus::AtlasIncomplete:return "AtlasIncomplete";case AtlasStatus::SingularContact:return "SingularContact";case AtlasStatus::EventClusterUnresolved:return "EventClusterUnresolved";case AtlasStatus::ArithmeticUncertain:return "ArithmeticUncertain";case AtlasStatus::RepresentationLimited:return "RepresentationLimited";default:return "BudgetExceeded";}}
struct RadialAtlasConfig {int max_boxes=2048,max_depth=64,max_contacts=64,initial_tier=0;bool tubes=true,warm_proofs=true,warm_range_probe=true,contract_boxes=true;};
struct ContactAnchor {unsigned id=0,generation=0;atlas_detail::Box uniqueness;atlas_detail::IQ R,s;D14Real R_center,s_center;bool ordinary=false;};
struct AtlasLeaf {atlas_detail::Box box;int reason=0;double margin=0; /* current hull distance; 0 if not stored */};
struct RadialAtlasStats {int box_created=0,excluded=0,contracted=0,tubes=0,contacts=0,warm_reused=0,warm_invalidated=0,warm_range_accepted=0,event_seed_updates=0,dd_boxes=0,qf_boxes=0,legacy_calls=0;double coefficient_ms=0,proof_ms=0,refine_ms=0,total_ms=0;};
struct RadialAtlasResult {AtlasStatus status=AtlasStatus::AtlasIncomplete;double rmax=0;unsigned generation=0;std::vector<ContactAnchor> contacts;std::vector<AtlasLeaf> leaves;RadialAtlasStats stats;atlas_detail::Box unresolved;};
struct RadialEventAtlasCache {bool valid=false;PrimaryFrame frame{};RadialAtlasResult result;};
namespace atlas_detail {
using BD=Tensor<Interval<double>>;using BT=Tensor<Ball>;using BQ=Tensor<IQ>;using Data=std::variant<BD,BT,BQ>;
struct Node {Box box;Data data;bool old=false;};
inline Box contracted(const Box& b,IQ x,IQ y){Box c=b;IQ r=IQ(b.r0,b.r0)+(IQ(b.r1,b.r1)-IQ(b.r0,b.r0))*x;IQ s=IQ(b.s0,b.s0)+(IQ(b.s1,b.s1)-IQ(b.s0,b.s0))*y;c.r0=fmaxq(b.r0,r.lo);c.r1=fminq(b.r1,r.hi);c.s0=fmaxq(b.s0,s.lo);c.s1=fminq(b.s1,s.hi);return c;}
inline bool contains(const Box& a,const Box& b){return a.chart==b.chart&&a.r0<=b.r0&&a.r1>=b.r1&&a.s0<=b.s0&&a.s1>=b.s1;}
inline bool overlaps(const Box& a,const Box& b){return a.chart==b.chart&&a.r0<=b.r1&&a.r1>=b.r0&&a.s0<=b.s1&&a.s1>=b.s0;}
inline bool refine_contact(const PrimaryFrame& pf,Box box,ContactAnchor& c){
 c.uniqueness=box;
 Box candidate;
 if(contact_candidate(pf,box,candidate)){
  auto proof=contact_krawczyk(pf,candidate);
  if(proof.kind==3)box=contracted(candidate,proof.x,proof.y);
  else return false;
 }else return false;
 c.R={box.r0,box.r1};c.s={box.s0,box.s1};c.R_center=d14_from_qf(midpoint(c.R));c.s_center=d14_from_qf(midpoint(c.s));
 auto g=local_fold_quantities<IQ>(c.R,c.s,pf,box.chart);
 c.ordinary=(g.PR.sign()==1||g.PR.sign()==-1)&&(g.Ptt.sign()==1||g.Ptt.sign()==-1);
 return c.ordinary && box.r1-box.r0<scalbnq(fmaxq(1,fabsq(box.r0)),-48);
}
inline bool same_contact(const PrimaryFrame& pf,const ContactAnchor& a,const ContactAnchor& b){
 Box other=b.uniqueness;
 if(other.chart!=a.uniqueness.chart){IQ s(other.s0,other.s1);if(s.sign()!=1&&s.sign()!=-1)return false;auto inv=-inverse(s);other.s0=inv.lo;other.s1=inv.hi;other.chart=a.uniqueness.chart;}
 if(!overlaps(a.uniqueness,other))return false;
 Box both=a.uniqueness;both.r0=fminq(both.r0,other.r0);both.r1=fmaxq(both.r1,other.r1);both.s0=fminq(both.s0,other.s0);both.s1=fmaxq(both.s1,other.s1);
 return prove(boundary<IQ>(pf,both)).kind==3;
}
inline double outer_radius(const PrimaryFrame& pf){
 IQ x(pf.X),y(pf.Y),rho(pf.rho),a(pf.a);auto w2=x*x+y*y;IQ W(down(sqrtq(fmaxq(0,w2.lo))),up(sqrtq(w2.hi)));W=W+rho;
 Q guess=(pf.a+W.hi+sqrtq((pf.a-W.hi)*(pf.a-W.hi)+4))/2;double r=std::nextafter(double(guess),INFINITY);
 for(int k=0;k<32;++k){IQ R(r);auto gap=(R-W)*(R-a);if(r>pf.a&&(R-W).lo>0&&gap.lo>1)return r;r=std::nextafter(r,INFINITY);}return NAN;
}
} // detail
struct RadialAtlasWorkspace {std::vector<atlas_detail::Node> stack;};
inline RadialAtlasResult build_radial_event_atlas(const PrimaryFrame& pf,const RadialAtlasConfig& cfg,RadialAtlasWorkspace& ws,RadialEventAtlasCache* cache=nullptr){
 using namespace atlas_detail;RadialAtlasResult out;positive_detail::StageTimer total{&out.stats.total_ms};
 auto finish=[&](){out.stats.contacts=out.contacts.size();total.stop();return out;};
 if(!positive_detail::environment_ok()||!(pf.a>0&&pf.m0>0&&pf.m0<1&&pf.rho>0)){out.status=AtlasStatus::ArithmeticUncertain;return finish();}
 out.rmax=outer_radius(pf);if(!std::isfinite(out.rmax)){out.status=AtlasStatus::ArithmeticUncertain;return finish();}
 out.generation=cache&&cache->valid?cache->result.generation+1:1;ws.stack.clear();ws.stack.reserve(std::min(cfg.max_boxes,4096));
 auto make=[&](Box b,bool old=false){positive_detail::StageTimer timer{&out.stats.coefficient_ms};if(cfg.initial_tier==1){ws.stack.push_back(Node{b,boundary<Ball>(pf,b),old});++out.stats.dd_boxes;}else ws.stack.push_back(Node{b,boundary<Interval<double>>(pf,b),old});};
 bool warm=cache&&cache->valid&&cfg.warm_proofs;
 unsigned next_id=0;
 if(cache&&cache->valid)for(const auto& c:cache->result.contacts)next_id=std::max(next_id,c.id+1);
 if(warm){
  // Event seeds update against CURRENT G; never preserve old uniqueness.
  for(const auto& old:cache->result.contacts){ContactAnchor c=old;Box b=old.uniqueness;
   positive_detail::StageTimer timer{&out.stats.refine_ms};
   auto proof=prove(boundary<Ball>(pf,b),cfg.tubes);
   if(proof.kind==3 && refine_contact(pf,b,c)){c.generation=out.generation;out.contacts.push_back(c);++out.stats.event_seed_updates;}
  }
  // Rebuild each old leaf in its local coordinates. This is conservative
  // proof reuse, not unverified L1 topology reuse or an old sign lookup.
  for(const auto& leaf:cache->result.leaves){
   bool excluded=false;
   if(cfg.warm_range_probe && leaf.reason!=3){
    positive_detail::StageTimer timer{&out.stats.proof_ms};
    auto r=positive_detail::bounds(positive_detail::point<Interval<double>>(leaf.box.r0));
    auto h=positive_detail::bounds(positive_detail::point<Interval<double>>(leaf.box.r1));
    auto l=positive_detail::bounds(positive_detail::point<Interval<double>>(leaf.box.s0));
    auto u=positive_detail::bounds(positive_detail::point<Interval<double>>(leaf.box.s1));
    // Outward conversions, current physical polynomial, whole old box.
    auto g=local_fold_quantities(Interval<double>(double(r.lo),double(h.hi)),Interval<double>(double(l.lo),double(u.hi)),pf,bool(leaf.box.chart));
    auto v=leaf.reason==1?g.P:g.Pt;int sign=v.sign();
    if(sign==1||sign==-1){out.leaves.push_back({leaf.box,leaf.reason,double(fminq(fabsq(v.lo),fabsq(v.hi)))});++out.stats.warm_reused;++out.stats.warm_range_accepted;++out.stats.excluded;excluded=true;}
   }
   if(!excluded)make(leaf.box,true);
  }
  if(out.rmax>cache->result.rmax)for(int chart=0;chart<2;++chart)make(Box{cache->result.rmax,out.rmax,-1,1,chart,0});
 }else for(int chart=0;chart<2;++chart)make(Box{0,out.rmax,-1,1,chart,0});
 while(!ws.stack.empty()){
  Node node=std::move(ws.stack.back());ws.stack.pop_back();out.unresolved=node.box;if(++out.stats.box_created>cfg.max_boxes){out.status=AtlasStatus::BudgetExceeded;return finish();}
  bool covered=false;for(const auto& c:out.contacts)if(contains(c.uniqueness,node.box)){covered=true;break;}
  if(covered){out.leaves.push_back({node.box,3,0});continue;}
  positive_detail::StageTimer proof_timer{&out.stats.proof_ms};
  LocalProof proof=std::visit([&](const auto& p){return prove(p,cfg.tubes);},node.data);
  // Escalate only where coefficient uncertainty dominates the local hull.
  if(!proof.kind){bool noise=std::visit([](const auto& p){auto h=hull(p);Q width=0;for(int i=0;i<=p.nr;++i)for(int j=0;j<=p.ns;++j){auto b=bounds(p.c[i][j]);width=fmaxq(width,b.hi-b.lo);}return width>.02Q*(h.hi-h.lo);},node.data);
   if(noise&&node.data.index()==0){node.data=boundary<Ball>(pf,node.box);++out.stats.dd_boxes;proof=prove(std::get<BT>(node.data),cfg.tubes);}
   if(noise&&!proof.kind&&node.data.index()==1){node.data=boundary<IQ>(pf,node.box);++out.stats.qf_boxes;proof=prove(std::get<BQ>(node.data),cfg.tubes);}
  }
  Box unique_box=node.box;
  if(!proof.kind && node.box.depth>=6 && proof.x.hi-proof.x.lo<.5Q && proof.y.hi-proof.y.lo<.5Q){
   // A boundary root needs an enlarged neighborhood. The candidate covers
   // the ENTIRE leaf; no uncovered neighbor is thrown away.
   Box b=node.box;Q wr=(b.r1-b.r0)/8,ws_=(b.s1-b.s0)/8;b.r0-=wr;b.r1+=wr;b.s0-=ws_;b.s1+=ws_;
   auto big=prove(boundary<Ball>(pf,b),cfg.tubes);
   if(big.kind==3){proof=big;unique_box=b;++out.stats.dd_boxes;}
  }
  if(proof.kind==1||proof.kind==2){++out.stats.excluded;if(proof.tube)++out.stats.tubes;if(node.old)++out.stats.warm_reused;out.leaves.push_back({node.box,proof.kind,0});continue;}
  if(node.old)++out.stats.warm_invalidated;
  if(proof.kind==3){ContactAnchor contact;proof_timer.stop();positive_detail::StageTimer refine{&out.stats.refine_ms};
   if(refine_contact(pf,unique_box,contact)){
    if(contact.R.lo<=0||contact.R.hi>=out.rmax){out.status=AtlasStatus::RepresentationLimited;return finish();}
    bool duplicate=false;for(const auto& c:out.contacts)if(same_contact(pf,c,contact)){duplicate=true;break;}
    if(!duplicate){if(int(out.contacts.size())>=cfg.max_contacts){out.status=AtlasStatus::BudgetExceeded;return finish();}contact.id=next_id++;contact.generation=out.generation;out.contacts.push_back(contact);}
    if(proof.tube)++out.stats.tubes;out.leaves.push_back({node.box,3,0});continue;
   }
  }
  if(cfg.contract_boxes && proof.kind==0 && node.box.depth<cfg.max_depth &&
      (proof.x.hi-proof.x.lo<.5Q || proof.y.hi-proof.y.lo<.5Q)){
   // Krawczyk contains EVERY zero in the original box, independently of
   // existence. Leave slack outside its enclosure before excluding strips.
   IQ x{proof.x.lo/2,(proof.x.hi+1)/2},y{proof.y.lo/2,(proof.y.hi+1)/2};
   Box b=contracted(node.box,x,y);b.depth=node.box.depth+1;
   auto strip=[&](Box z){if(z.r1>z.r0&&z.s1>z.s0){out.leaves.push_back({z,2,0});++out.stats.excluded;}};
   Box z=node.box;z.r1=b.r0;strip(z);z=node.box;z.r0=b.r1;strip(z);
   z=node.box;z.r0=b.r0;z.r1=b.r1;z.s1=b.s0;strip(z);z.s0=b.s1;z.s1=node.box.s1;strip(z);
   // Reconstruct from exact frame in new coordinates, not a rounded cast.
   proof_timer.stop();{positive_detail::StageTimer timer{&out.stats.coefficient_ms};ws.stack.push_back({b,boundary<Ball>(pf,b),false});}
   ++out.stats.contracted;++out.stats.dd_boxes;continue;
  }
  if(node.box.depth>=cfg.max_depth){out.unresolved=node.box;out.status=AtlasStatus::ArithmeticUncertain;return finish();}
  // Alternate coordinates for balanced coverage; cheap range tests prune
  // branches before any fine angular mesh could be generated.
  int axis=node.box.depth%2;Box left=node.box,right=node.box;left.depth=right.depth=node.box.depth+1;
  if(axis){Q mid=(left.s0+left.s1)/2;if(mid==left.s0||mid==left.s1){out.status=AtlasStatus::RepresentationLimited;return finish();}left.s1=right.s0=mid;}
  else{Q mid=(left.r0+left.r1)/2;if(mid==left.r0||mid==left.r1){out.status=AtlasStatus::RepresentationLimited;return finish();}left.r1=right.r0=mid;}
  std::visit([&](const auto& p){auto parts=split_physical(p,node.box,axis);ws.stack.push_back({right,std::move(parts.second),false});ws.stack.push_back({left,std::move(parts.first),false});},node.data);
 }
 out.stats.contacts=out.contacts.size();out.status=AtlasStatus::Complete;
 out.stats.contacts=out.contacts.size();
 if(cache){cache->valid=true;cache->frame=pf;cache->result=out;}
 total.stop();
 return finish();
}
} // namespace

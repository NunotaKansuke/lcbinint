#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;
struct Row {int case_id=0,configuration_id=0,d_bin_index=0,epoch_index=0;
 std::string profile;double s=0,q=0,rho=0,x=0,y=0,time=0,u=0,X=0,reference=0;};
static bool read(const std::string& line,Row& r){if(line.empty()||line[0]=='#')return false;std::istringstream in(line);return bool(in>>r.case_id>>r.configuration_id>>r.profile>>r.d_bin_index>>r.epoch_index>>r.s>>r.q>>r.rho>>r.x>>r.y>>r.time>>r.u>>r.X>>r.reference);}
static bool same(const Row&a,const Row&b){return a.case_id==b.case_id&&a.configuration_id==b.configuration_id&&a.profile==b.profile&&a.d_bin_index==b.d_bin_index;}
static bool d14(const RadialEvent&e){return e.detail.rfind("D14 ",0)==0;}
static int kind_count(const std::vector<RadialEvent>&e,const char*k){int n=0;for(const auto&x:e)n+=x.kind==k;return n;}
static double radius_error(const std::vector<RadialEvent>&a,const std::vector<RadialEvent>&b,const std::string&kind){std::vector<double>x,y;for(const auto&e:a)if(d14(e)&&e.kind==kind)x.push_back(e.radius+e.radius_lo);for(const auto&e:b)if(d14(e)&&e.kind==kind)y.push_back(e.radius+e.radius_lo);if(x.size()!=y.size())return -1;std::sort(x.begin(),x.end());std::sort(y.begin(),y.end());double z=0;for(size_t i=0;i<x.size();++i)z=std::max(z,std::fabs(x[i]-y[i]));return z;}
static bool same_plan(const TopologyResult&a,const TopologyResult&b){if(a.status!=b.status||a.cells.size()!=b.cells.size())return false;for(size_t i=0;i<a.cells.size();++i)if(a.cells[i].kind!=b.cells[i].kind||a.cells[i].n_crossings!=b.cells[i].n_crossings)return false;return true;}
struct Continued {TopologyResult topology;int attempted=0,updated=0,failed=0;double ms=0,max_shift_gap_ratio=0;bool previous_plan_match=false,guard_01=false,guard_001=false;};

// Neumaier's Gershgorin-like inclusion disks.  For pairwise-distinct
// approximations z_i, let W_i=P(z_i)/(a_n prod_{j!=i}(z_i-z_j)) and form
// D(z_i-n W_i/2,n|W_i|/2).  Every connected component made from m disks
// contains exactly m roots.  Thus fourteen pairwise-disjoint disks certify
// one root per disk and completeness.  This point-qf implementation is a
// research screen; a production certificate needs outward-rounded bounds.
struct DiskResult {bool finite=false,disjoint=false,classifiable=false;int sweeps=0;double ms=0;
 int axis_ambiguous=0,zero_ambiguous=0;double max_radius=0,max_radius_sep_ratio=0;std::vector<Cplx<__float128>> roots;};

static DiskResult neumaier_candidate(const re_detail::D14StructQf& sc,
 const std::vector<__float128>& ascending,
 const std::vector<Cplx<__float128>>& initial,int sweeps,double rmax) {
 using qf=__float128;auto begin=Clock::now();DiskResult out;out.sweeps=sweeps;
 out.roots=initial;if(out.roots.size()!=14||ascending.size()!=15)return out;
 const qf leading=ascending[14];if(!finiteq(leading)||leading==0)return out;
 auto canonicalize=[&](){std::array<bool,14> done{};
  for(int i=0;i<14;++i){if(done[i])continue;
   const qf cut=(qf)1e-8*((qf)1+fabsq(out.roots[i].re));
   if(fabsq(out.roots[i].im)<=cut){out.roots[i].im=0;done[i]=true;continue;}
   int best=-1;qf distance=HUGE_VALQ;
   for(int j=i+1;j<14;++j)if(!done[j]){const qf d=hypotq(out.roots[i].re-out.roots[j].re,out.roots[i].im+out.roots[j].im);if(d<distance){distance=d;best=j;}}
   if(best<0)return false;const qf re=(out.roots[i].re+out.roots[best].re)/(qf)2;
   const qf im=(fabsq(out.roots[i].im)+fabsq(out.roots[best].im))/(qf)2;
   const qf sign=out.roots[i].im<0?(qf)-1:(qf)1;
   out.roots[i]={re,sign*im};out.roots[best]={re,-sign*im};done[i]=done[best]=true;
  }return true;
 };
 if(!canonicalize())return out;
 for(int sweep=0;sweep<sweeps;++sweep){
  auto old=out.roots;std::array<Cplx<qf>,14> correction{};
  for(int i=0;i<14;++i){Cplx<qf> p,dp;re_detail::d14_struct_eval(sc,old[i],p,dp);
   Cplx<qf> denominator{leading,0};
   for(int j=0;j<14;++j)if(i!=j)denominator=denominator*(old[i]-old[j]);
   correction[i]=p/denominator;
   if(!finiteq(correction[i].re)||!finiteq(correction[i].im))return out;
  }
  for(int i=0;i<14;++i)out.roots[i]=old[i]-correction[i];
  if(!canonicalize())return out;
 }
 std::array<Cplx<qf>,14> center{};std::array<qf,14> radius{};
 for(int i=0;i<14;++i){Cplx<qf> p,dp;re_detail::d14_struct_eval(sc,out.roots[i],p,dp);
  Cplx<qf> denominator{leading,0};
  for(int j=0;j<14;++j)if(i!=j)denominator=denominator*(out.roots[i]-out.roots[j]);
  const auto w=p/denominator;
  if(!finiteq(w.re)||!finiteq(w.im))return out;
  const auto r=w*Cplx<qf>{(qf)7,0};center[i]=out.roots[i]-r;
  if(out.roots[i].im==0)center[i].im=0; // exact real-arithmetic lane
  radius[i]=cabs(r);
  out.max_radius=std::max(out.max_radius,(double)radius[i]);
 }
 out.finite=true;out.disjoint=true;out.classifiable=true;
 for(int i=0;i<14;++i)for(int j=i+1;j<14;++j){
  const qf sep=cabs(center[i]-center[j]),sum=radius[i]+radius[j];
  if(!(sep>sum)){out.disjoint=false;break;}
  out.max_radius_sep_ratio=std::max(out.max_radius_sep_ratio,(double)(sum/sep));
 }
 // A one-root disk centered exactly on the real axis contains a real root:
 // conjugation preserves the disk, so a non-real root would bring its
 // conjugate and contradict uniqueness.  A disk separated from the real
 // axis contains a non-real root.  Every other disk is classification-
 // ambiguous and must fail closed.  Positive real roots must also be
 // separated from zero.
 for(int i=0;i<14;++i){
  const bool real=center[i].im==0;
  const bool nonreal=fabsq(center[i].im)>radius[i];
  const qf vmax=(qf)rmax*(qf)rmax;
  const bool relevant=center[i].re+radius[i]>(qf)0&&center[i].re-radius[i]<vmax;
  if(relevant&&!real&&!nonreal){out.classifiable=false;++out.axis_ambiguous;}
  if(relevant&&real&&fabsq(center[i].re)<=radius[i]){out.classifiable=false;++out.zero_ambiguous;}
 }
 out.roots.assign(center.begin(),center.end());
 out.ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();return out;
}

static TopologyResult roots_to_topology(const PrimaryFrame&pf,
 const TopologyResult&oracle,const std::vector<Cplx<__float128>>& roots) {
 using qf=__float128;std::vector<RadialEvent> events;
 for(const auto&e:oracle.events)if(!d14(e))events.push_back(e);
 std::vector<double> soft;
 for(const auto&z:roots){if(!(z.re>0))continue;const qf ai=fabsq(z.im),ar=fabsq(z.re);
  if(ai<=(qf)1e-8*((qf)1+ar)){const double R=(double)sqrtq(z.re);
   if(!(R>0&&R<oracle.r_max))continue;const auto probe=re_detail::probe_double_root(R,pf);
   events.push_back({R,probe.physically_real?"physical_real":"physical_complex",
    probe.physically_real,"D14 disk candidate real root"});
  }else if(std::sqrt((double)z.re)<oracle.r_max)soft.push_back((double)z.re);
 }
 std::sort(soft.begin(),soft.end());double last=-1;
 for(double v:soft){if(last>=0&&v-last<=1e-9)continue;last=v;
  events.push_back({std::sqrt(v),"physical_complex",false,"D14 disk candidate complex root"});}
 std::sort(events.begin(),events.end(),[](const auto&a,const auto&b){return a.radius<b.radius;});
 return classify_event_cells(pf,std::move(events),oracle.r_max);
}
static Continued continue_events(const PrimaryFrame&pf,const TopologyResult&previous,const TopologyResult&current){
 auto begin=Clock::now();Continued out;std::vector<RadialEvent> events;
 // Representation events have closed-form/current-parameter definitions and
 // are copied from the oracle only in this diagnostic. Their construction is
 // not part of the D14 cost under study.
 for(const auto&e:current.events)if(!d14(e))events.push_back(e);
 for(size_t old_index=0;old_index<previous.events.size();++old_index){const auto&old=previous.events[old_index];if(d14(old)){
  RadialEvent e=old;
  if(old.physically_real){++out.attempted;auto x=adaptive_detail::topology_event_estimate(old,pf,1e-4);
   if(x.needs_qf||!std::isfinite(x.radius)||!std::isfinite(x.uncertainty)){++out.failed;continue;}
   e.radius=x.radius;e.radius_lo=x.radius_lo;e.radius_uncertainty=x.uncertainty;e.precision_tier=x.precision_tier;++out.updated;
   double gap=std::numeric_limits<double>::infinity();for(size_t j=0;j<previous.events.size();++j)if(j!=old_index)gap=std::min(gap,std::fabs(old.radius-previous.events[j].radius));
   const double shift=std::fabs((e.radius+e.radius_lo)-(old.radius+old.radius_lo));
   out.max_shift_gap_ratio=std::max(out.max_shift_gap_ratio,shift/std::max(gap,1e-300));
  }
  if(e.radius>0&&e.radius<current.r_max)events.push_back(e);
 }}
 std::sort(events.begin(),events.end(),[](const auto&a,const auto&b){return a.radius<b.radius;});
 out.topology=classify_event_cells(pf,std::move(events),current.r_max);
 out.previous_plan_match=same_plan(out.topology,previous);
 out.guard_01=out.failed==0&&out.previous_plan_match&&out.max_shift_gap_ratio<0.1;
 out.guard_001=out.failed==0&&out.previous_plan_match&&out.max_shift_gap_ratio<0.01;
 out.ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();return out;
}
int main(int argc,char**argv){if(argc<3){std::cerr<<"usage: bench_d14_event_continuation INPUT OUT\n";return 2;}std::ifstream in(argv[1]);std::ofstream out(argv[2]);if(!in||!out)return 2;std::vector<Row>rows;std::string line;while(std::getline(in,line)){Row r;if(read(line,r))rows.push_back(r);}std::stable_sort(rows.begin(),rows.end(),[](const auto&a,const auto&b){if(a.case_id!=b.case_id)return a.case_id<b.case_id;if(a.configuration_id!=b.configuration_id)return a.configuration_id<b.configuration_id;if(a.profile!=b.profile)return a.profile<b.profile;if(a.d_bin_index!=b.d_bin_index)return a.d_bin_index<b.d_bin_index;return a.epoch_index<b.epoch_index;});
 out<<"case_id configuration_id profile d_bin_index epoch_index method_ms sweeps finite disk_disjoint disk_classifiable axis_ambiguous zero_ambiguous max_disk_radius max_disk_sep_ratio physical_count candidate_physical_count soft_count candidate_soft_count physical_radius_maxdiff soft_radius_maxdiff cells candidate_cells topology_status candidate_status cell_plan_match\n"<<std::setprecision(17);
 TopologyResult previous;std::vector<Cplx<__float128>> previous_roots;bool have=false;
 for(size_t i=0;i<rows.size();++i){if(i==0||!same(rows[i-1],rows[i])){have=false;previous_roots.clear();}
  const auto&r=rows[i];LensParams p{r.time,r.y,r.rho,1.0/r.q,r.s,true};auto pf=PrimaryFrame::from(p);
  std::vector<Cplx<__float128>> current_roots;auto current=classify_cells(pf,nullptr,&current_roots,true);
  if(have){auto sc=re_detail::d14_struct_build((__float128)pf.a,(__float128)pf.m0,(__float128)pf.X,(__float128)pf.Y,(__float128)pf.rho);
   auto asc=re_detail::d14_expanded_from_struct(sc);
   for(int sweeps:{3,4,6,8}){auto d=neumaier_candidate(sc,asc,previous_roots,sweeps,current.r_max);
    TopologyResult candidate;if(d.finite)candidate=roots_to_topology(pf,current,d.roots);
    const bool match=d.finite&&same_plan(candidate,current);
    out<<r.case_id<<' '<<r.configuration_id<<' '<<r.profile<<' '<<r.d_bin_index<<' '<<r.epoch_index<<' '<<d.ms<<' '<<sweeps<<' '<<d.finite<<' '<<d.disjoint<<' '<<d.classifiable<<' '<<d.axis_ambiguous<<' '<<d.zero_ambiguous<<' '<<d.max_radius<<' '<<d.max_radius_sep_ratio<<' '<<kind_count(current.events,"physical_real")<<' '<<(d.finite?kind_count(candidate.events,"physical_real"):-1)<<' '<<kind_count(current.events,"physical_complex")<<' '<<(d.finite?kind_count(candidate.events,"physical_complex"):-1)<<' '<<(d.finite?radius_error(candidate.events,current.events,"physical_real"):-1)<<' '<<(d.finite?radius_error(candidate.events,current.events,"physical_complex"):-1)<<' '<<current.cells.size()<<' '<<(d.finite?candidate.cells.size():0)<<' '<<int(current.status)<<' '<<(d.finite?int(candidate.status):-1)<<' '<<match<<'\n';
   }
  }
  previous=std::move(current);previous_roots=std::move(current_roots);have=true;
 }
 return 0;}

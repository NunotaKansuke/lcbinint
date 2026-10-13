#pragma once
// Research driver: reuse the integrator's symmetric boundary root pair.
// Local continuation is not a global completeness certificate.
#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/root_pair.hpp"
namespace lcbinint::holonomic {
struct SeededBoundaryConfig {bool width_control=true; int max_steps_per_path=32;};
struct SeededBoundaryStats {int quartics=0,steps=0,retries=0,paths=0,fold_solves=0,fold_failures=0;};
struct SeededBoundaryResult {std::vector<RadialEvent> events; std::vector<double> seed_radii; SeededBoundaryStats stats; // completed only means the local start queue fit the budget. It does not
 // certify all extrema, and individual paths have a separate step cap.
 bool completed=false,coverage_certified=false; const char* reason=""; double r_max=0;};
namespace seeded_boundary_detail {
inline std::array<double,5> coeff(double R,const PrimaryFrame&pf,bool reciprocal,bool derivative=false){
 if(!derivative&&std::fabs(R-pf.a)<1e-3*(1+R)){
  auto f=local_fold_quantities<long double>((long double)R,0.L,pf,reciprocal);
  return {(double)f.P,(double)f.Pt,(double)(f.Ptt/2),(double)(f.Psss/6),(double)(f.Pssss/24)};
 }
 auto q=derivative?boundary_quartic_dR(R,pf):boundary_quartic(R,pf);
 return (reciprocal?boundary_quartic_reciprocal(q):q).p;
}
inline bool fold(double seedR,double seedm,bool reciprocal,const PrimaryFrame&pf,double rmax,RadialEvent&e){
 long double R=seedR,m=seedm;
 for(int i=0;i<24;++i){
  auto f=local_fold_quantities<long double>(R,m,pf,reciprocal);
  long double det=f.PR*f.Ptt-f.Pt*f.Ptr;
  if(!std::isfinite(det)||det==0)return false;
  long double dr=(-f.P*f.Ptt+f.Pt*f.Pt)/det,dm=(-f.PR*f.Pt+f.Ptr*f.P)/det;
  R+=dr;m+=dm;
  if(!(R>0&&R<rmax)||!std::isfinite(m)||std::fabs(m)>1e5)return false;
  if(std::fabs(dr)+std::fabs(dm)<4e-17L*(1+R+std::fabs(m))){
   auto pc=coeff((double)R,pf,reciprocal);auto v=local_fold_quantities<long double>(R,m,pf,reciprocal);
   long double scale=0;for(int k=4;k>=0;--k)scale=scale*std::max(1.L,std::fabs(m))+std::fabs(pc[k]);
   if(std::fabs(v.P)+std::fabs(v.Pt)>2e-12L*scale)return false;
   e.radius=(double)R;e.kind="physical_real";e.detail="seed root-pair stationary refinement";e.physically_real=true;
   e.radius_lo=(double)(R-(long double)e.radius);
   if(reciprocal){e.fold_u_seed=(double)m;e.fold_u_seed_valid=true;}
   else {e.fold_t_seed=(double)m;e.fold_t_seed_valid=true;}
   return true;
  }
 }return false;
}
}
inline SeededBoundaryResult seeded_boundary_events(const PrimaryFrame&pf,const std::vector<Cplx<double>>&seeds,const SeededBoundaryConfig&cfg={}){
 using namespace seeded_boundary_detail;
 SeededBoundaryResult out;double W=std::hypot(pf.X,pf.Y)+pf.rho;
 out.r_max=.5*(pf.a+W+std::hypot(pf.a-W,2.))+1e-12;
 struct Sample {double R;RootPair pair;};
 struct Covered {bool reciprocal;int direction;double lo,hi;std::vector<Sample> samples;};
 std::vector<Covered> covered;
 std::vector<double> starts;
 for(auto z:seeds){double R=std::hypot(z.re,z.im);if(R>0&&R<out.r_max){bool seen=false;for(double r:starts)if(std::fabs(r-R)<1e-10*(1+R))seen=true;if(!seen)starts.push_back(R);}}
 out.seed_radii=starts;
 auto add=[&](const RadialEvent&e){for(auto x:out.events)if(std::fabs(x.radius-e.radius)<2e-10*(1+e.radius))return false;out.events.push_back(e);return true;};
 for(double R:re_detail::chart_p4_factor_roots(pf))if(R>0&&R<out.r_max){
  auto pc=coeff(R,pf,true);auto f=p_derivs(pc,0);double scale=0;for(double c:pc)scale+=std::fabs(c);
  if(std::fabs(f.P)+std::fabs(f.P1)<1e-12*scale){RadialEvent e;if(fold(R,0,true,pf,out.r_max,e))add(e);}
 }
 // Every adjacent pair can collide, including gaps outside an image arc.
 // Newly found folds supply local starts on both sides; accepted work is kept.
 for(std::size_t si=0;si<starts.size()&&si<256;++si){
  double start=starts[si];++out.stats.quartics;
  auto th=real_root_thetas(coeff(start,pf,false));
  // Boundary roots already available at this radius supply stationary-point
  // initial guesses without another lens-equation solve.
  for(double theta:th)for(bool rec:{false,true}){
   double m=rec?-1/std::tan(theta/2):std::tan(theta/2);if(!std::isfinite(m))continue;
   RadialEvent e;++out.stats.fold_solves;
   if(fold(start,m,rec,pf,out.r_max,e))if(add(e))for(int sign:{-1,1}){
    double next=e.radius+sign*std::max(1e-11*(1+e.radius),1e-5*std::fabs(e.radius-start));if(next>0&&next<out.r_max)starts.push_back(next);
   }
  }
  for(std::size_t j=0;j<th.size();++j){
   double a=th[j],b=th[(j+1)%th.size()];

   double t0=std::tan(a/2),t1=std::tan(b/2),u0=-1/t0,u1=-1/t1;
   bool preferred=std::max(std::fabs(u0),std::fabs(u1))<std::max(std::fabs(t0),std::fabs(t1));
   for(bool reciprocal:{preferred}) {
   if(!std::isfinite(reciprocal?u0+u1:t0+t1))continue;
   auto initial=root_pair_from_endpoints(reciprocal?u0:t0,reciprocal?u1:t1);
   for(int direction:{-1,1}){
    bool seen=false;
    for(const auto& path:covered){
     if(path.reciprocal!=reciprocal||path.direction!=direction||start<path.lo||start>path.hi)continue;
     const Sample* nearest=&path.samples.front();for(const auto& sample:path.samples)if(std::fabs(sample.R-start)<std::fabs(nearest->R-start))nearest=&sample;
     auto motion=root_pair_dR(nearest->pair,coeff(nearest->R,pf,reciprocal),coeff(nearest->R,pf,reciprocal,true));
     if(!motion.ok)continue;
     double h=start-nearest->R;RootPair probe{nearest->pair.m+h*motion.dm_dR,nearest->pair.v+h*motion.dv_dR};
     auto correction=correct_root_pair(probe,coeff(start,pf,reciprocal),8,2e-13);
     if(correction.converged&&std::fabs(probe.m-initial.m)<1e-5*std::sqrt(initial.v)+8e-13*(1+std::fabs(initial.m))&&
        std::fabs(probe.v-initial.v)<1e-5*initial.v+8e-26*(1+initial.m*initial.m)){seen=true;break;}
    }
    if(seen)continue;
    ++out.stats.paths;double R=start;RootPair pair=initial;
    std::vector<Sample> samples{{R,pair}};double next_step=.025*std::max(R,.01);
    for(int step=0;step<cfg.max_steps_per_path;++step){
     ++out.stats.steps;
     auto d=root_pair_dR(pair,coeff(R,pf,reciprocal),coeff(R,pf,reciprocal,true));
     if(!d.ok)break;
     double contact=-pair.v/d.dv_dR;
     double base=std::min(next_step,.025*std::max(R,.01));
     // Try the exact local stationary equations near a predicted zero of v.
     if(std::isfinite(contact)&&direction*contact>0&&std::fabs(contact)<4*base){
      RadialEvent e;++out.stats.fold_solves;
      if(fold(R+contact,pair.m+contact*d.dm_dR,reciprocal,pf,out.r_max,e)&&direction*(e.radius-R)>-1e-10&&std::fabs(e.radius-R-contact)<2*std::fabs(contact)){
       if(add(e))for(int sign:{-1,1}){double next=e.radius+sign*std::max(1e-11*(1+e.radius),1e-5*std::fabs(e.radius-start));if(next>0&&next<out.r_max)starts.push_back(next);}
       covered.push_back({reciprocal,direction,std::min(start,e.radius),std::max(start,e.radius),std::move(samples)});
       break;
      }++out.stats.fold_failures;
     }
     double h=direction*base;
     if(cfg.width_control&&std::isfinite(contact)&&direction*contact>0)h=direction*std::min(base,.5*std::fabs(contact));
     bool accepted=false;
     for(int retry=0;retry<28;++retry){
      double nextR=R+h;if(!(nextR>0&&nextR<out.r_max))break;
      RootPair next{pair.m+h*d.dm_dR,pair.v+h*d.dv_dR};
      auto correction=correct_root_pair(next,coeff(nextR,pf,reciprocal),8,2e-13);
      double m_error=std::fabs(next.m-pair.m-h*d.dm_dR);
      double v_error=std::fabs(next.v-pair.v-h*d.dv_dR);
      if(correction.converged&&std::isfinite(next.m+next.v)&&next.v>0&&m_error<.25*(std::sqrt(pair.v)+1e-14)&&v_error<.25*(pair.v+1e-28)){
       next_step=std::fabs(h)*1.5;pair=next;R=nextR;if(step%8==0)samples.push_back({R,pair});accepted=true;break;
      }h*=.5;++out.stats.retries;
     }
     if(!accepted||next_step<2e-14*(1+R))break;
    }
   }
   }
  }
 }
 std::sort(out.events.begin(),out.events.end(),[](auto&a,auto&b){return a.radius<b.radius;});
 out.completed=starts.size()<=256;out.reason=out.completed?"local discovery; completeness requires independent audit":"local start budget exceeded";return out;
}
inline TopologyResult seeded_boundary_cell_plan(const PrimaryFrame&pf,const SeededBoundaryResult&r){
 if(!r.completed){TopologyResult t;t.status=Status::TOPOLOGY_UNCERTAIN;return t;}
 auto events=r.events;
 auto add_representation=[&](double R,const char* kind){
  // The same contact may also be a chart root. Avoid a second binary64 cut
  // a few ULPs away; distinct nearby folds remain in the event list.
  for(const auto& e:events)if(e.physically_real&&std::fabs(e.radius-R)<=64*std::numeric_limits<double>::epsilon()*(1+R))return;
  RadialEvent e;e.radius=R;e.kind=kind;events.push_back(e);
 };
 for(double R:{pf.a,std::sqrt(pf.m0)})if(R>0&&R<r.r_max){add_representation(R,"representation");}
 for(double R:re_detail::chart_p4_factor_roots(pf))if(R>0&&R<r.r_max){add_representation(R,"chart_p4");}
 // Preserve physical cuts exactly; this requests strict partitioning,
 // not an atlas certificate.
 auto plan=classify_event_cells(pf,std::move(events),r.r_max,{},true);
 std::vector<CellPlan> split;
 for(auto cell:plan.cells){
  double seed=0,best=std::numeric_limits<double>::infinity();
  if(cell.kind==ArcKind::kArcs)for(double R:r.seed_radii){
   double width=cell.r_hi-cell.r_lo;
   if(R>cell.r_lo+.05*width&&R<cell.r_hi-.05*width&&std::fabs(R-cell.r_mid)<best){seed=R;best=std::fabs(R-cell.r_mid);}
  }
  // At most one existing seed cut per occupied cell; never all seed radii.
  if(seed>0){
   CellPlan left=cell,right=cell;left.r_hi=seed;right.r_lo=seed;
   left.r_mid=left.r_lo+.5*(left.r_hi-left.r_lo);right.r_mid=right.r_lo+.5*(right.r_hi-right.r_lo);
   split.push_back(left);split.push_back(right);
   RadialEvent e;e.radius=seed;e.kind="seed_soft";plan.events.push_back(e);
  }else split.push_back(cell);
 }
 for(std::size_t i=0;i<split.size();++i)split[i].index=(int)i;
 plan.cells=std::move(split);return plan;
}
}

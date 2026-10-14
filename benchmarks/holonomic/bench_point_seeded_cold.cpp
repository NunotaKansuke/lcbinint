#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "lcbinint/magnification/holonomic/seeded_point_epoch.hpp"
using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;
double us(Clock::time_point t){return std::chrono::duration<double,std::micro>(Clock::now()-t).count();}
int main(int argc,char**argv){
 if(argc!=3)return 2;std::ifstream in(argv[1]);std::ofstream out(argv[2]);if(!in||!out)return 2;
 out<<std::setprecision(17)<<"id u tol repetition seed_us topology_us integral_us total_us d14_topology_us d14_integral_us d14_total_us roots witnesses candidate_ok reference_ok relative_difference candidate_nodes reference_nodes candidate_spans reference_spans candidate_qf reference_qf discovery_us plan_us paths steps retries quartics folds critical_attempts\n";
 int id,rows=0;LensParams p;
 while(in>>id>>p.xs>>p.ys>>p.rho>>p.q>>p.a){
  auto pf=PrimaryFrame::from(p);
  for(double u:{0.,.5})for(double tolerance:{1e-3,1e-4})for(int rep=0;rep<3;++rep){
   AdaptiveConfig cfg;cfg.tol.mu_rtol=tolerance;cfg.gradient_policy=GradientPolicy::None;
   double seedtime,topotime,inttime,total,rtopo,rint,rtotal,discoverytime,plantime;SeededBoundaryStats discovery_stats;PointSeedProposals proposals;AdaptiveResult x,y;
   auto candidate=[&](){AdaptiveWorkspace ws;auto start=Clock::now();lcbinint::magnification::PointSourceMagnifier pm;
    proposals=point_seed_proposals(p,pm);seedtime=us(start);auto t=Clock::now();
    auto events=seeded_boundary_events(pf,proposals.positions);discoverytime=us(t);auto pt=Clock::now();auto plan=seeded_boundary_cell_plan(pf,events);plantime=us(pt);topotime=us(t);discovery_stats=events.stats;
    t=Clock::now();x=flux_adaptive_integrate(p,u,plan,cfg,ws);inttime=us(t);total=us(start);};
   auto reference=[&](){AdaptiveWorkspace ws;auto start=Clock::now();auto plan=classify_cells(pf,nullptr,nullptr,true);rtopo=us(start);
    auto t=Clock::now();y=flux_adaptive_integrate(p,u,plan,cfg,ws);rint=us(t);rtotal=us(start);};
   if(rep%2){reference();candidate();}else{candidate();reference();}
   double relative=std::fabs(x.mu-y.mu)/std::max(1e-30,std::fabs(y.mu));
   out<<id<<' '<<u<<' '<<tolerance<<' '<<rep<<' '<<seedtime<<' '<<topotime<<' '<<inttime<<' '<<total<<' '<<rtopo<<' '<<rint<<' '<<rtotal<<' '<<proposals.quintic_solves<<' '<<proposals.critical_witnesses<<' '<<x.value_converged<<' '<<y.value_converged<<' '<<relative<<' '<<x.stats.unique_nodes<<' '<<y.stats.unique_nodes<<' '<<x.stats.integration_spans<<' '<<y.stats.integration_spans<<' '<<x.stats.event_qf_refinements<<' '<<y.stats.event_qf_refinements<<' '<<discoverytime<<' '<<plantime<<' '<<discovery_stats.paths<<' '<<discovery_stats.steps<<' '<<discovery_stats.retries<<' '<<discovery_stats.quartics<<' '<<discovery_stats.fold_solves<<' '<<proposals.critical_attempts<<'\n';++rows;
  }
 }
 return rows?0:2;
}

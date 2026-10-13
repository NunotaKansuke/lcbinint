#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <limits>
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"
using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv) {
 if(argc<3)return 2;
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);
 if(!in||!out)return 2;
 int reps=argc>3?std::max(1,std::atoi(argv[3])):1;
 int cuts=argc>4?std::atoi(argv[4]):1;
 out<<std::setprecision(17)<<"id u tol candidate_converged d14_converged relative_difference candidate_mu d14_mu candidate_ms d14_ms planning_ms probes bands skipped seeds nodes d14_nodes stop local_repairs retained_nodes candidate_integration_ms d14_topology_ms d14_integration_ms discovery_reason preparation_ms center_ms caustic_ms seed_ms support candidate_no_caustic_ms\n";
 std::string line;int count=0,fail=0,unresolved=0;
 while(std::getline(in,line)) {
  std::istringstream is(line);int id,support,n;LensParams p;double seedms;
  if(!(is>>id>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>support>>seedms>>n))continue;
  std::vector<Cplx<double>> seeds(n);for(auto&z:seeds)if(!(is>>z.re>>z.im))return 2;
  if(!support)return 2;
  for(double u:{0.,.5})for(double tol:{1e-3,1e-4}) {
   AdaptiveConfig cfg;cfg.tol.mu_rtol=tol;cfg.gradient_policy=GradientPolicy::None;
   std::vector<double> a,b,ap,ai,bp,bi,prep,center_times,caustic_times,seed_times,no_caustic; bool proven=false;SeededFrontierResult x;AdaptiveResult y;
   for(int k=0;k<reps;++k) {
    AdaptiveWorkspace wa,wb;
    auto candidate=[&]{auto start=Clock::now();
      lcbinint::magnification::PointSourceMagnifier pm;
      lcbinint::magnification::FiniteSourceSettings settings; settings.caustic_bins=1400;
      lcbinint::magnification::FiniteSourceMagnifier fm(settings);
      auto stage=Clock::now(); std::vector<lcbinint::SourcePosition> center;
      for(const auto& image:pm.binary_images(p.a,p.q,{p.xs,p.ys}))center.push_back(image.position);
      auto elapsed=[](Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();};
      center_times.push_back(elapsed(stage)); stage=Clock::now();
      double distance=fm.binary_caustic_distance_for_source(p.a,p.q,{p.xs,p.ys});
      caustic_times.push_back(elapsed(stage)); stage=Clock::now();
      auto positions=fm.cached_binary_image_seeds(pm,p.a,p.q,{p.xs,p.ys},p.rho,std::numeric_limits<double>::infinity(),&center,&proven);
      std::vector<Cplx<double>> live_seeds; live_seeds.reserve(positions.size());
      for(auto z:positions)live_seeds.push_back({z.x+(p.q<1?p.a:0),z.y});
      seed_times.push_back(elapsed(stage)); prep.push_back(elapsed(start));
      x=epoch_seeded_radial_frontier(p,u,live_seeds,cfg,wa,cuts);
      a.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
      no_caustic.push_back(a.back()-caustic_times.back());
      ap.push_back(x.frontier.planning_ms);ai.push_back(x.frontier.integration_ms);};
    auto reference=[&]{auto start=Clock::now();y=epoch_adaptive(p,u,cfg,wb);
      b.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
      bp.push_back(y.stats.topology_ms);bi.push_back(b.back()-y.stats.topology_ms);};
    if(k&1){reference();candidate();}else{candidate();reference();}
   }
   std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
   auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
   double rel=std::fabs(x.integral.mu-y.mu)/std::max(1e-30,std::fabs(y.mu));
   ++count;if(!y.value_converged)++unresolved;
   else if(!proven || !x.integral.value_converged || !std::isfinite(rel) || rel>tol)++fail;
   out<<id<<' '<<u<<' '<<tol<<' '<<x.integral.value_converged<<' '<<y.value_converged<<' '<<rel<<' '<<x.integral.mu<<' '<<y.mu<<' '<<a[a.size()/2]<<' '<<b[b.size()/2]<<' '<<median(ap)<<' '<<x.frontier.radius_probes<<' '<<x.frontier.bands<<' '<<x.frontier.skipped_seeds<<' '<<x.frontier.seeds<<' '<<x.frontier.evaluated_nodes<<' '<<y.stats.unique_nodes<<' '<<int(x.integral.stop)<<' '<<x.frontier.local_repairs<<' '<<x.frontier.retained_nodes<<' '<<median(ai)<<' '<<median(bp)<<' '<<median(bi)<<' '<<x.frontier.reason<<' '<<median(prep)<<' '<<median(center_times)<<' '<<median(caustic_times)<<' '<<median(seed_times)<<' '<<proven<<' '<<median(no_caustic)<<'\n';
  }
 }
 std::cerr<<"rows="<<count<<" candidate_failed_on_resolved_reference="<<fail<<" reference_unresolved="<<unresolved<<'\n';
 return count==0?2:fail?1:0;
}

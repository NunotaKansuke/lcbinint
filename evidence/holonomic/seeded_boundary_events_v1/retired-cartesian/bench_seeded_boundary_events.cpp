#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "lcbinint/magnification/holonomic/seeded_boundary_events.hpp"
#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::micro>(Clock::now()-t).count();}
static double median(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char**argv){
 if(argc<3)return 2;int reps=argc>3?std::max(1,atoi(argv[3])):5;
 bool width=argc>4?atoi(argv[4])!=0:true;
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);std::string line;
 std::ofstream eventout(std::string(argv[2])+".events.tsv");eventout<<std::setprecision(17)<<"id lane radius kind physically_real\n";
 out<<"id support seeds seed_cold_ms trace_us d14_us completed loops steps retries quartics folds oracle_folds matched missing extra max_radius_error topology_disagreements value_relerr status reason\n"<<std::setprecision(17);
 int failed=0,rows=0;
 while(std::getline(in,line)){
  std::istringstream is(line);int id,support,n;LensParams p;double seedms;
  if(!(is>>id>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>support>>seedms>>n))continue;
  std::vector<Cplx<double>> seeds(n);for(auto&z:seeds)is>>z.re>>z.im;
  PrimaryFrame pf=PrimaryFrame::from(p);SeededBoundaryResult cand;TopologyResult oracle;
  SeededBoundaryConfig cfg;cfg.width_control=width;
  std::vector<double>ct,ot;
  for(int j=0;j<reps;++j){auto t=Clock::now();cand=seeded_boundary_events(pf,seeds,cfg);ct.push_back(elapsed(t));t=Clock::now();oracle=classify_cells(pf,nullptr,nullptr,true);ot.push_back(elapsed(t));}
  for(auto&e:oracle.events)eventout<<id<<" d14 "<<e.radius<<' '<<e.kind<<' '<<e.physically_real<<'\n';
  for(auto&e:cand.events)eventout<<id<<" trace "<<e.radius<<' '<<e.kind<<' '<<e.physically_real<<'\n';
  std::vector<double> expected;for(auto&e:oracle.events){
    bool physical=e.physically_real;
    // A projective double root at theta=pi is recorded as chart_p4 rather
    // than physically_real by the incumbent. Verify the stationary boundary
    // equation explicitly before counting it as an expected physical fold.
    if(e.kind=="chart_p4"){
      SeededBoundaryStats st;auto g=seeded_boundary_detail::geometry({-e.radius,0},pf,st);
      physical=g.valid && std::fabs(g.f)<1e-8*pf.rho*pf.rho && std::fabs(g.g.y)<1e-8*seeded_boundary_detail::norm(g.g);
    }
    if(physical){bool exists=false;for(double R:expected)if(std::fabs(R-e.radius)<2e-10*(1+R))exists=true;if(!exists)expected.push_back(e.radius);}
  }
  int matched=0,missing=0,extra=0;double maxerr=0;
  for(double R:expected){double best=1e100;for(auto&e:cand.events)best=std::min(best,std::fabs(R-e.radius));if(best<1e-8*(1+R)){++matched;maxerr=std::max(maxerr,best);}else ++missing;}
  for(auto&e:cand.events){double best=1e100;for(double R:expected)best=std::min(best,std::fabs(R-e.radius));if(best>1e-8*(1+e.radius))++extra;}
  int disagrees=0;double rel=std::numeric_limits<double>::quiet_NaN();
  TopologyResult plan;
  if(cand.completed){plan=seeded_boundary_cell_plan(pf,cand);
    // Independent D14 partitions expose events skipped by the candidate.
    for(auto&cell:plan.cells)for(auto&e:oracle.events)if(e.physically_real && e.radius>cell.r_lo+1e-10 && e.radius<cell.r_hi-1e-10)++disagrees;
    for(auto&cell:oracle.cells)for(double f:{.13,.5,.87}){
      double R=cell.r_lo+(cell.r_hi-cell.r_lo)*f;auto actual=quartic_topology(R,pf);
      bool found=false;for(auto&c:plan.cells)if(R>c.r_lo&&R<c.r_hi){found=true;if(!actual.certified||actual.n_crossings!=c.n_crossings||actual.kind!=c.kind)++disagrees;break;}if(!found)++disagrees;
    }
    // Same existing integrator and n_r on both plans; no integral is timed.
    auto a=flux_value_integrate(128,0,pf,plan),b=flux_value_integrate(128,0,pf,oracle);
    rel=std::fabs(a.F0-b.F0)/std::max(1e-30,std::fabs(b.F0));
  }
  bool ok=cand.completed&&missing==0&&extra==0&&disagrees==0;
  ++rows;if(!ok)++failed;
  out<<id<<' '<<support<<' '<<n<<' '<<seedms<<' '<<median(ct)<<' '<<median(ot)<<' '<<cand.completed<<' '<<cand.stats.loops<<' '<<cand.stats.steps<<' '<<cand.stats.retries<<' '<<cand.stats.quartics<<' '<<cand.events.size()<<' '<<expected.size()<<' '<<matched<<' '<<missing<<' '<<extra<<' '<<maxerr<<' '<<disagrees<<' '<<rel<<' '<<ok<<' '<<'"'<<cand.reason<<'"'<<'\n';
 }
 std::cerr<<"rows="<<rows<<" failed="<<failed<<"\n";
 return 0; // A research audit records failures; it never hides failed rows.
}

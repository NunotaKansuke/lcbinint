#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include "lcbinint/magnification/holonomic/certified_component_topology.hpp"

using namespace lcbinint;
using namespace lcbinint::magnification;
using namespace lcbinint::holonomic;
using Clock=std::chrono::steady_clock;

struct Row { int cid=0,cfgid=0,db=0,ep=0; std::string profile; double s=0,q=0,rho=0,x=0,y=0,time=0,u=0,X=0,ref=0; };
static bool parse(const std::string& line,Row& r){if(line.empty()||line[0]=='#')return false;std::istringstream in(line);return bool(in>>r.cid>>r.cfgid>>r.profile>>r.db>>r.ep>>r.s>>r.q>>r.rho>>r.x>>r.y>>r.time>>r.u>>r.X>>r.ref);}
static bool same(const Row&a,const Row&b){return a.cid==b.cid&&a.cfgid==b.cfgid&&a.profile==b.profile&&a.db==b.db;}
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static double med(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}

struct Built {TopologyResult topo;CertifiedComponentTopologyStats stats;double seed_ms=0,band_ms=0,total_ms=0;bool support=false;};
static Built build(const Row&r,FiniteSourceMagnifier& fm,const PointSourceMagnifier& pm){
    Built b;auto all=Clock::now();bool proven=false;auto t=Clock::now();
    const double hq=1.0/r.q;
    auto raw=fm.cached_binary_image_seeds(pm,r.s,hq,{r.time,r.y},r.rho,
        std::numeric_limits<double>::infinity(),nullptr,&proven);
    b.seed_ms=ms(t);b.support=proven;
    std::vector<Cplx<double>> seeds;seeds.reserve(raw.size());
    // The legacy mapper places the heavier lens at -s for q<1.  Holonomic
    // uses the reciprocal-q orientation (light lens at 0, heavy lens at +s).
    const double shift=0.0;
    for(const auto& z:raw)seeds.push_back({z.x+shift,z.y});
    LensParams p{r.time,r.y,r.rho,hq,r.s,true};
    t=Clock::now();b.topo=classify_certified_component_bands(
        PrimaryFrame::from(p),seeds,proven,&b.stats);b.band_ms=ms(t);
    if(std::getenv("HOLO_CERT_FOLD_MAPS")&&b.topo.status==Status::OK){const auto pf=PrimaryFrame::from(p);std::vector<double> edges;for(const auto&c:b.topo.cells)if(c.r_hi>0)edges.push_back(c.r_hi);std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());for(double R:edges){auto d=adaptive_detail::double_event_estimate(R,pf,1e-4);if(!d.t_seed_valid)continue;RadialEvent e;e.radius=R;e.kind="physical_real";e.physically_real=true;e.detail="certified support endpoint";e.radius_lo=d.radius_lo;e.radius_uncertainty=d.uncertainty;e.fold_t_seed=d.t_seed;e.fold_t_seed_valid=true;e.precision_tier=d.precision_tier;b.topo.events.push_back(e);}}
    b.total_ms=ms(all);return b;
}
static AdaptiveConfig config(double tol){AdaptiveConfig c;c.gradient_policy=GradientPolicy::None;c.with_jacobian=false;c.fold_maps=std::getenv("HOLO_CERT_FOLD_MAPS")!=nullptr;c.reuse_samples=true;c.tol.mu_atol=1e-16;c.tol.mu_rtol=tol;return c;}

int main(int argc,char**argv){
 if(argc<3){std::cerr<<"usage: bench INPUT OUTPUT [reps] [max_trajectories]\n";return 2;}int reps=argc>3?std::max(1,atoi(argv[3])):3;int maxtr=argc>4?atoi(argv[4]):-1;
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);std::vector<Row> rows;std::string line;while(std::getline(in,line)){Row r;if(parse(line,r))rows.push_back(r);}std::stable_sort(rows.begin(),rows.end(),[](auto&a,auto&b){if(a.cid!=b.cid)return a.cid<b.cid;if(a.cfgid!=b.cfgid)return a.cfgid<b.cfgid;if(a.profile!=b.profile)return a.profile<b.profile;if(a.db!=b.db)return a.db<b.db;return a.ep<b.ep;});
 out<<"case_id profile d_bin epoch target lane whole_ms topology_ms seed_ms band_ms mu reference relerr converged stop status support seeds bands probes nodes panels splits\n"<<std::setprecision(17);
 PointSourceMagnifier pm;FiniteSourceSettings fs;fs.caustic_bins=1400;std::size_t beg=0;int tid=0;volatile double sink=0;
 while(beg<rows.size()&&(maxtr<0||tid<maxtr)){size_t end=beg+1;while(end<rows.size()&&same(rows[beg],rows[end]))++end;for(double tol:{1e-3,1e-4}){
   FiniteSourceMagnifier warmfm(fs);AdaptiveWorkspace warmw;
   for(size_t i=beg;i<end;++i){const Row&r=rows[i];LensParams p{r.time,r.y,r.rho,1.0/r.q,r.s,true};auto cfg=config(tol);
     for(const char* lane:{"d14","alg-cold","alg-warm"}){std::vector<double> times;AdaptiveResult keep;Built bk;
       for(int k=0;k<reps;++k){AdaptiveWorkspace local;auto start=Clock::now();if(std::string(lane)=="d14")keep=epoch_adaptive(p,r.u,cfg,local);else {FiniteSourceMagnifier cold(fs);Built b=build(r,std::string(lane)=="alg-warm"?warmfm:cold,pm);keep=flux_adaptive_integrate(p,r.u,b.topo,cfg,std::string(lane)=="alg-warm"?warmw:local);keep.stats.topology_ms=b.total_ms;bk=b;}times.push_back(ms(start));sink+=keep.mu;}
       double rel=std::fabs(keep.mu-r.ref)/std::max(1.0,std::fabs(r.ref));out<<r.cid<<' '<<r.profile<<' '<<r.db<<' '<<r.ep<<' '<<tol<<' '<<lane<<' '<<med(times)<<' '<<keep.stats.topology_ms<<' '<<bk.seed_ms<<' '<<bk.band_ms<<' '<<keep.mu<<' '<<r.ref<<' '<<rel<<' '<<keep.value_converged<<' '<<adaptive_stop_name(keep.stop)<<' '<<to_string(keep.numerical_status)<<' '<<bk.support<<' '<<bk.stats.seed_count<<' '<<bk.stats.band_count<<' '<<bk.stats.radius_probes<<' '<<keep.stats.unique_nodes<<' '<<keep.stats.panels<<' '<<keep.stats.splits<<'\n';
     }
   }
 }++tid;beg=end;if(tid%25==0)std::cerr<<"trajectory "<<tid<<"\n";}
 std::cerr<<"done trajectories="<<tid<<" sink="<<sink<<"\n";
}

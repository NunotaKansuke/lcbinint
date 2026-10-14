#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
using namespace lcbinint::holonomic;using Clock=std::chrono::steady_clock;
struct Row{int case_id,configuration_id,d_bin,epoch;std::string profile;double s,q,rho,x,y,time,u,X,ref;};
int main(int argc,char**argv){if(argc!=3)return 2;std::ifstream in(argv[1]);std::ofstream out(argv[2]);std::vector<Row> rows;std::string line;
 while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;std::istringstream ss(line);Row r;if(ss>>r.case_id>>r.configuration_id>>r.profile>>r.d_bin>>r.epoch>>r.s>>r.q>>r.rho>>r.x>>r.y>>r.time>>r.u>>r.X>>r.ref){if(r.profile=="uniform"&&r.u==0.0)rows.push_back(r);}}
 std::stable_sort(rows.begin(),rows.end(),[](auto&a,auto&b){if(a.case_id!=b.case_id)return a.case_id<b.case_id;if(a.configuration_id!=b.configuration_id)return a.configuration_id<b.configuration_id;if(a.profile!=b.profile)return a.profile<b.profile;if(a.d_bin!=b.d_bin)return a.d_bin<b.d_bin;return a.epoch<b.epoch;});
 holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;holo_ode_transport_override()=0;
 out<<"case config profile bin epoch s q rho x y time u X ref tol cold_ms shadow_cold_ms warm_ms shadow_warm_ms radial_ms shadow_radial_ms cold_mu shadow_cold_mu warm_mu shadow_warm_mu radial_mu shadow_radial_mu cold_ok shadow_cold_ok warm_ok shadow_warm_ok radial_ok shadow_radial_ok cold_status shadow_cold_status warm_status shadow_warm_status radial_status shadow_radial_status cold_nodes shadow_cold_nodes warm_nodes shadow_warm_nodes radial_nodes shadow_radial_nodes cold_jet_attempts cold_jet_successes cold_shadow_panels cold_local_budget_passes cold_jet_ms cold_model_ms warm_jet_attempts warm_jet_successes warm_shadow_panels warm_local_budget_passes warm_jet_ms warm_model_ms radial_jet_attempts radial_jet_successes radial_shadow_panels radial_local_budget_passes radial_jet_ms radial_model_ms\n"<<std::setprecision(17);
 for(size_t begin=0;begin<rows.size();){size_t end=begin+1;while(end<rows.size()&&rows[end].case_id==rows[begin].case_id&&rows[end].configuration_id==rows[begin].configuration_id&&rows[end].profile==rows[begin].profile&&rows[end].d_bin==rows[begin].d_bin)++end;
 for(double tol:{1e-3,1e-4}){AdaptiveConfig a;a.gradient_policy=GradientPolicy::None;a.tol.mu_atol=1e-16;a.tol.mu_rtol=tol;a.radial_error_estimator=RadialErrorEstimator::HybridEmbedded;a.nested_difference_safety=2;
 AdaptiveConfig b=a;b.same_node_hermite_shadow=true;AdaptiveWorkspace wa,wb,wr0,wr1;PreparedEpochGeometry sa,sb;
 PreparedReuseConfig reuse;reuse.allow_topology_reuse=false;reuse.allow_warm_d14=true;reuse.l2_drift=1e18;
 for(size_t i=begin;i<end;++i){auto&r=rows[i];LensParams p{r.time,r.y,r.rho,1/r.q,r.s,true};AdaptiveResult ac,bc,aw,bw,ar,br;double t[6];
 auto run=[&](int lane){auto st=Clock::now();if(lane==0)ac=epoch_adaptive(p,r.u,a,wa);if(lane==1)bc=epoch_adaptive(p,r.u,b,wb);if(lane==2)aw=epoch_adaptive_prepared(p,r.u,a,wa,sa,reuse);if(lane==3)bw=epoch_adaptive_prepared(p,r.u,b,wb,sb,reuse);t[lane]=std::chrono::duration<double,std::milli>(Clock::now()-st).count();};
 if((i-begin)%2){run(1);run(0);}else{run(0);run(1);}if((i-begin)%2){run(3);run(2);}else{run(2);run(3);}
 // Isolate radial integration while keeping the same per-input D14/cell plan
 // outside the timed interval.  This is diagnostic; full cold/warm are primary.
 const auto radial_topology=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,
                                           /*retain_adaptive_metadata=*/true);
 auto run_radial=[&](int lane){auto st=Clock::now();if(lane==0)ar=flux_adaptive_integrate(p,r.u,radial_topology,a,wr0);else br=flux_adaptive_integrate(p,r.u,radial_topology,b,wr1);t[4+lane]=std::chrono::duration<double,std::milli>(Clock::now()-st).count();};
 if((i-begin)%2){run_radial(1);run_radial(0);}else{run_radial(0);run_radial(1);}
 out<<r.case_id<<' '<<r.configuration_id<<' '<<r.profile<<' '<<r.d_bin<<' '<<r.epoch<<' '<<r.s<<' '<<r.q<<' '<<r.rho<<' '<<r.x<<' '<<r.y<<' '<<r.time<<' '<<r.u<<' '<<r.X<<' '<<r.ref<<' '<<tol<<' '<<t[0]<<' '<<t[1]<<' '<<t[2]<<' '<<t[3]<<' '<<t[4]<<' '<<t[5]<<' '<<ac.mu<<' '<<bc.mu<<' '<<aw.mu<<' '<<bw.mu<<' '<<ar.mu<<' '<<br.mu<<' '<<ac.value_converged<<' '<<bc.value_converged<<' '<<aw.value_converged<<' '<<bw.value_converged<<' '<<ar.value_converged<<' '<<br.value_converged<<' '<<adaptive_stop_name(ac.stop)<<' '<<adaptive_stop_name(bc.stop)<<' '<<adaptive_stop_name(aw.stop)<<' '<<adaptive_stop_name(bw.stop)<<' '<<adaptive_stop_name(ar.stop)<<' '<<adaptive_stop_name(br.stop)<<' '<<ac.stats.unique_nodes<<' '<<bc.stats.unique_nodes<<' '<<aw.stats.unique_nodes<<' '<<bw.stats.unique_nodes<<' '<<ar.stats.unique_nodes<<' '<<br.stats.unique_nodes<<' '<<bc.stats.hermite_jet_attempts<<' '<<bc.stats.hermite_jet_successes<<' '<<bc.stats.hermite_shadow_panels<<' '<<bc.stats.hermite_shadow_local_budget_passes<<' '<<bc.stats.hermite_jet_ms<<' '<<bc.stats.hermite_shadow_ms<<' '<<bw.stats.hermite_jet_attempts<<' '<<bw.stats.hermite_jet_successes<<' '<<bw.stats.hermite_shadow_panels<<' '<<bw.stats.hermite_shadow_local_budget_passes<<' '<<bw.stats.hermite_jet_ms<<' '<<bw.stats.hermite_shadow_ms<<' '<<br.stats.hermite_jet_attempts<<' '<<br.stats.hermite_jet_successes<<' '<<br.stats.hermite_shadow_panels<<' '<<br.stats.hermite_shadow_local_budget_passes<<' '<<br.stats.hermite_jet_ms<<' '<<br.stats.hermite_shadow_ms<<'\n';
 }
 }begin=end;std::cerr<<"trajectories through row "<<end<<"/"<<rows.size()<<"\n";}
}

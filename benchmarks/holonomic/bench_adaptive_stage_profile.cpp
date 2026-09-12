#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 if(argc<2||argc>3)return 2;
 const bool warm=argc==3 && std::string(argv[2])=="warm";
 PreparedEpochGeometry state; AdaptiveWorkspace warm_workspace;
 std::string previous_key; int position=0;
 std::ifstream in(argv[1]);std::string line;
 std::cout<<std::setprecision(17)<<"case profile db epoch whole mu ok nodes topology setup physics estimator scheduler struct expand presearch real qf residual completeness classification soft probes arc endpoint k rescue trajectory_pos bracket_attempts bracket_successes native_ms native_calls native_pass native_violations native_root_pass native_root_violations\n";
 holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;holo_ode_transport_override()=0;
 while(std::getline(in,line)){
 int id,config,db,ep;std::string profile;double s,q,rho,x,y,t,u,X,ref;
 std::istringstream row(line);if(!(row>>id>>config>>profile>>db>>ep>>s>>q>>rho>>x>>y>>t>>u>>X>>ref))continue;
 const std::string key=std::to_string(id)+"/"+std::to_string(config)+"/"+profile+"/"+std::to_string(db);
 if(key!=previous_key){state=PreparedEpochGeometry{};warm_workspace.reset();position=0;previous_key=key;}else ++position;
 LensParams p{t,y,rho,1/q,s,true};AdaptiveConfig cfg;cfg.tol.mu_atol=1e-16;cfg.tol.mu_rtol=1e-3;AdaptiveWorkspace w;
 V2Profile v;V2ProfileScope scope(v);auto start=adaptive_detail::Clock::now();auto a=warm?epoch_adaptive_prepared(p,u,cfg,warm_workspace,state):epoch_adaptive(p,u,cfg,w);double whole=adaptive_detail::ms(start);
 std::cout<<id<<' '<<profile<<' '<<db<<' '<<ep<<' '<<whole<<' '<<a.mu<<' '<<a.value_converged<<' '<<a.stats.unique_nodes;
 for(double val:{a.stats.topology_ms,a.stats.setup_ms,a.stats.physical_ms,a.stats.estimator_ms,a.stats.scheduler_ms,v.d14_struct_build_ms,v.d14_expand_ms,v.d14_presearch_ms,v.d14_real_ms,v.d14_qf_ms,v.d14_residual_eval_ms,v.d14_completeness_check_ms,v.d14_event_classify_ms,v.d14_soft_event_ms,v.topology_probe_ms,v.arc_ms,v.endpoint_ms,v.k_ms,v.angular_rescue_ms})std::cout<<' '<<val;
 std::cout<<' '<<position<<' '<<v.local_bracket_attempts<<' '<<v.local_bracket_successes<<' '<<v.native_residual_ms<<' '<<v.native_residual_calls<<' '<<v.native_residual_pass<<' '<<v.native_residual_violations<<' '<<v.native_residual_root_pass<<' '<<v.native_residual_root_violations<<'\n';
 }
}

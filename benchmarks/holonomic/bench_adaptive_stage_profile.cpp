#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace lcbinint::holonomic;
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ifstream in(argv[1]);std::string line;
 std::cout<<std::setprecision(17)<<"case profile db epoch whole mu ok nodes topology setup physics estimator scheduler struct expand presearch real qf residual completeness classification soft probes arc endpoint k rescue\n";
 holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;holo_ode_transport_override()=0;
 while(std::getline(in,line)){
 int id,config,db,ep;std::string profile;double s,q,rho,x,y,t,u,X,ref;
 std::istringstream row(line);if(!(row>>id>>config>>profile>>db>>ep>>s>>q>>rho>>x>>y>>t>>u>>X>>ref))continue;
 LensParams p{t,y,rho,1/q,s,true};AdaptiveConfig cfg;cfg.tol.mu_atol=1e-16;cfg.tol.mu_rtol=1e-3;AdaptiveWorkspace w;
 V2Profile v;V2ProfileScope scope(v);auto start=adaptive_detail::Clock::now();auto a=epoch_adaptive(p,u,cfg,w);double whole=adaptive_detail::ms(start);
 std::cout<<id<<' '<<profile<<' '<<db<<' '<<ep<<' '<<whole<<' '<<a.mu<<' '<<a.value_converged<<' '<<a.stats.unique_nodes;
 for(double val:{a.stats.topology_ms,a.stats.setup_ms,a.stats.physical_ms,a.stats.estimator_ms,a.stats.scheduler_ms,v.d14_struct_build_ms,v.d14_expand_ms,v.d14_presearch_ms,v.d14_real_ms,v.d14_qf_ms,v.d14_residual_eval_ms,v.d14_completeness_check_ms,v.d14_event_classify_ms,v.d14_soft_event_ms,v.topology_probe_ms,v.arc_ms,v.endpoint_ms,v.k_ms,v.angular_rescue_ms})std::cout<<' '<<val;
 std::cout<<'\n';
 }
}

#define main trajectory_runner_main
#include "../../benchmarks/holonomic/bench_d14_positive_trajectory.cpp"
#undef main
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream in(argv[1]);InputRow row;std::string line;
 holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;holo_ode_transport_override()=0;
 std::cout<<std::setprecision(17);
 while(std::getline(in,line)){
  if(!read_row(line,row)||row.case_id!=0||row.d_bin_index!=0||row.configuration_id!=0||!((row.profile=="linear"&&(row.epoch_index==7||row.epoch_index==15||row.epoch_index==23))||(row.profile=="uniform"&&(row.epoch_index==7||row.epoch_index==15))))continue;
  auto p=LensParams{row.time,row.y,row.rho,1.0/row.q,row.s,true};
  for(double target:{1e-3,1e-4}) for(auto mode:{D14EventPolicy::AllComplexSoft,D14EventPolicy::NoProjectedComplexSoft,D14EventPolicy::PositiveReal}){
   D14EventPolicyScope scope(mode);auto topo=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,true);
   auto cfg=make_config(target);cfg.collect_diagnostics=true;AdaptiveWorkspace ws;
   auto r=flux_adaptive_integrate(p,row.u,topo,cfg,ws);
   std::cout<<"CASE "<<row.case_id<<' '<<row.profile<<' '<<row.epoch_index<<' '<<target<<" MODE "<<int(mode)<<" topology="<<to_string(topo.status)<<" stop="<<adaptive_stop_name(r.stop)<<" nodes="<<r.stats.unique_nodes<<'\n';
   for(const auto& e:topo.events)std::cout<<"EVENT "<<e.kind<<' '<<e.radius<<' '<<e.detail<<'\n';
   for(const auto& d:r.stats.sample_diagnostics) {
    std::cout<<"REJECT "<<d.cell<<' '<<d.panel<<' '<<d.level<<' '<<d.node_slot<<' '<<d.R<<' '<<adaptive_sample_reject_reason_name(d.reason)<<' '<<d.cold_retry<<' '<<d.cold_retry_succeeded<<'\n';
    for(const auto& cell:topo.cells)if(d.R>cell.r_lo&&d.R<cell.r_hi) {
     auto cold=adaptive_detail::mapped_radius(d.R,1,p,row.u,PrimaryFrame::from(p),cell,false,false,nullptr,&cfg);
     std::cout<<"SAME_R_COLD "<<adaptive_sample_reject_reason_name(cold.reject_reason)<<" reliable="<<cold.reliable<<'\n';
    }
   }
  }
 }
}

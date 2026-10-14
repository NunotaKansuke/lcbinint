#define main atlas_trajectory_main
#include "../../benchmarks/holonomic/bench_radial_atlas_trajectory.cpp"
#undef main
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream in(argv[1]);InputRow row;std::string line;
 holo_mv_transport_override()=1;holo_holonomic_transport_override()=1;holo_ode_transport_override()=0;
 std::cout<<std::setprecision(17);
 while(std::getline(in,line)){
  if(!read_row(line,row)||row.case_id!=0||row.d_bin_index!=0||row.configuration_id!=0||!((row.profile=="linear"&&(row.epoch_index==7||row.epoch_index==15||row.epoch_index==23))||(row.profile=="uniform"&&(row.epoch_index==7||row.epoch_index==15))))continue;
  auto p=LensParams{row.time,row.y,row.rho,1.0/row.q,row.s,true};auto pf=PrimaryFrame::from(p);
  RadialAtlasConfig ac;ac.max_boxes=20000;RadialAtlasWorkspace aws;auto atlas=build_radial_event_atlas(pf,ac,aws);auto at=classify_cells_from_atlas(pf,atlas);
  for(double target:{1e-3,1e-4}){
   D14EventPolicyScope policy(D14EventPolicy::NoProjectedComplexSoft);auto topo=classify_cells(pf,nullptr,nullptr,true);
   auto cfg=make_config(target);cfg.collect_diagnostics=true;AdaptiveWorkspace ws;auto old=flux_adaptive_integrate(p,row.u,topo,cfg,ws);
   std::vector<CellPlan> cells;if(!adaptive_detail::restore_physical_cuts(topo,pf,cells))return 3;
   AtlasSampleContext ctx{&atlas.contacts};AtlasSampleScope scope(&ctx);
   std::cout<<"CASE "<<row.profile<<' '<<row.epoch_index<<' '<<target<<" atlas="<<atlas_status_name(atlas.status)<<" contacts="<<atlas.contacts.size()<<" boxes="<<atlas.stats.box_created<<" old="<<adaptive_stop_name(old.stop)<<'\n';
   for(const auto& d:old.stats.sample_diagnostics){if(d.cold_retry_succeeded)continue;if(d.cell<0||size_t(d.cell)>=cells.size())return 4;const auto& cell=cells[d.cell];
    auto fresh=adaptive_detail::mapped_radius(d.R,1,p,row.u,pf,cell,false,false,nullptr,&cfg);
    std::cout<<"SAME_R_CELL "<<d.R<<' '<<d.cell<<' '<<cell.r_lo<<' '<<cell.r_hi<<" old="<<adaptive_sample_reject_reason_name(d.reason)<<" new="<<adaptive_sample_reject_reason_name(fresh.reject_reason)<<" reliable="<<fresh.reliable<<" value="<<fresh.value[0]<<'\n';
   }
   cfg.preserve_radial_offset=true;auto val=flux_adaptive_integrate(p,row.u,at,cfg,ws);
   std::cout<<"ATLAS_EPOCH "<<adaptive_stop_name(val.stop)<<" converged="<<val.value_converged<<" mu="<<val.mu<<" reference="<<row.reference<<" relative="<<std::abs(val.mu/row.reference-1)<<" error="<<val.estimated_abs_error_mu<<" pair="<<ctx.accepted<<'/'<<ctx.attempts<<'\n';
   for(const auto& c:atlas.contacts){char r[128],s[128];quadmath_snprintf(r,sizeof r,"%.34Qg",d14_to_qf(c.R_center));quadmath_snprintf(s,sizeof s,"%.34Qg",d14_to_qf(c.s_center));std::cout<<"ANCHOR "<<c.id<<' '<<c.uniqueness.chart<<' '<<r<<' '<<s<<" Rwidth="<<double(c.R.hi-c.R.lo)<<'\n';}
  }
 }
 const std::vector<std::pair<const char*,LensParams>> fixtures={
 {"c9",{1.1422819920252716,.09455346559252209,5.470597280025246e-05,992.0790840775868,.57986328980431667,true}},
 {"c92",{.016817436700993973,-1.7129651189458623e-05,3.1430126951461966e-05,218.24104917640977,3.8691964055932191,true}},
 {"axis",{.2,0,.03,1,1,true}}, {"near_axis",{.2,1e-10,.03,1,1,true}},
 {"wide",{.2,.1,.03,.1,4,true}}, {"close",{.2,.1,.03,.1,.2,true}},
 {"full",{.01,.02,3,1,1,true}}};
 for(const auto& fixture:fixtures){auto pf=PrimaryFrame::from(fixture.second);RadialAtlasConfig cfg;cfg.max_boxes=20000;RadialAtlasWorkspace ws;auto atlas=build_radial_event_atlas(pf,cfg,ws);auto topo=classify_cells_from_atlas(pf,atlas);
 std::cout<<"FIXTURE "<<fixture.first<<" atlas="<<atlas_status_name(atlas.status)<<" topology="<<to_string(topo.status)<<" contacts="<<atlas.contacts.size()<<" boxes="<<atlas.stats.box_created<<" ms="<<atlas.stats.total_ms<<" depth="<<atlas.unresolved.depth<<" dd="<<atlas.stats.dd_boxes<<" qf="<<atlas.stats.qf_boxes;
 if(atlas.status!=AtlasStatus::Complete){auto local=atlas_detail::prove_local<atlas_detail::IQ>(pf,atlas.unresolved);std::cout<<" fresh_local_qf_proof="<<local.kind<<" Rbox="<<double(atlas.unresolved.r0)<<','<<double(atlas.unresolved.r1)<<" sbox="<<double(atlas.unresolved.s0)<<','<<double(atlas.unresolved.s1)<<" chart="<<atlas.unresolved.chart;}
 std::cout<<'\n';}
}

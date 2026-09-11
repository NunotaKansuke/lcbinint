#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>
using namespace lcbinint::holonomic;
struct Case {LensParams p;double u;std::string name;};
int main(int argc,char**argv){
    if(argc<2)return 2;std::ifstream input(argv[1]);std::string line;std::vector<Case> cases;
    int repeats=argc>2?std::atoi(argv[2]):5;
    while(std::getline(input,line)){Case c;int b;double old;std::istringstream s(line);if(s>>c.p.xs>>c.p.ys>>c.p.rho>>c.p.q>>c.p.a>>b>>c.u>>old>>c.name){c.p.barycentric=b;cases.push_back(c);}}
    for(double u:{0.,.5})cases.push_back({{.00027748104958798373,.000043678813767388989,8.1133264861482082e-5,1/.00017213162835819022,1.6869965260211353,true},u,"paper_highA"});
    (void)fejer_rule(8);holo_holonomic_transport_override()=1;holo_mv_transport_override()=1;holo_ode_transport_override()=0;
    if(!cases.empty()){AdaptiveConfig c;c.with_jacobian=true;AdaptiveWorkspace w;
        (void)epoch_adaptive(cases.front().p,.5,c,w);(void)epoch_jacobian(cases.front().p,.5,64);}
    printf("name,u,jac,warm,rtol,cache,repeat,stop,mu,error,nodes,evaluations,reused,splits,whole_ms,topology_ms,setup_ms,setup_frame_ms,setup_cuts_ms,setup_event_ms,setup_panel_ms,event_double_checks,event_dd_checks,event_dd_accepts,event_qf_refinements,qf_family_constructions,event_topology_reuses,event_radius_reuses,event_direct_qf_failures,first_value_pass_nodes,first_gradient_error_pass_nodes,first_gradient_contract_nodes,gradient_error_pass_mask,gradient_contract_mask,physics_ms,estimator_ms,scheduler_ms,fixed64_ms,fixed64_mu,fixed64_status,inner,geometry,event,roundoff,grad0,grad1,grad2,grad3,grad4,graderr0,graderr1,graderr2,graderr3,graderr4\n");
    for(const auto& c:cases)for(bool jac:{false,true})for(bool warm:{false,true}) {
        AdaptiveWorkspace workspace;PreparedEpochGeometry prepared,fixed_prepared;
        for(double tol:{1e-3,1e-4,1e-5,1e-6,1e-8})for(bool cache:{true,false})for(int rep=0;rep<repeats;++rep){
            AdaptiveConfig cfg;cfg.reuse_samples=cache;cfg.with_jacobian=jac;cfg.tol.mu_rtol=tol;cfg.tol.grad_rtol.fill(tol);cfg.tol.grad_atol.fill(1e-6);
            auto p=c.p;
            if(warm){(void)epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared);
                if(jac)(void)epoch_jacobian_prepared(p,c.u,64,fixed_prepared,PreparedReuseConfig{});
                else (void)epoch_value_prepared(p,c.u,64,fixed_prepared,PreparedReuseConfig{});
                p.xs+=.01*p.rho;}
            AdaptiveResult r;double elapsed=0,fixed=0,fixedms=0;Status fixedstatus=Status::OK;
            auto adaptive_run=[&]{auto start=adaptive_detail::Clock::now();r=warm?epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared):epoch_adaptive(p,c.u,cfg,workspace);elapsed=adaptive_detail::ms(start);};
            auto fixed_run=[&]{auto start=adaptive_detail::Clock::now();
                if(jac){auto f=warm?epoch_jacobian_prepared(p,c.u,64,fixed_prepared,PreparedReuseConfig{}):epoch_jacobian(p,c.u,64);fixed=f.mu;fixedstatus=f.status;}
                else{auto f=warm?epoch_value_prepared(p,c.u,64,fixed_prepared,PreparedReuseConfig{}):epoch_value(p,c.u,64);fixed=f.mu;fixedstatus=f.status;}
                fixedms=adaptive_detail::ms(start);};
            if(rep%2){fixed_run();adaptive_run();}else{adaptive_run();fixed_run();}
            printf("%s,%.1f,%d,%d,%.1e,%d,%d,%s,%.17g,%.8g,%zu,%zu,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%u,%u,%.6f,%.6f,%.6f,%.6f,%.17g,%d,%.8g,%.8g,%.8g,%.8g",c.name.c_str(),c.u,jac,warm,tol,cache,rep,adaptive_stop_name(r.stop),r.mu,r.estimated_abs_error_mu,r.stats.unique_nodes,r.stats.node_evaluations,r.stats.reused_nodes,r.stats.splits,elapsed,r.stats.topology_ms,r.stats.setup_ms,r.stats.setup_frame_ms,r.stats.setup_cuts_ms,r.stats.setup_event_ms,r.stats.setup_panel_ms,r.stats.event_double_checks,r.stats.event_dd_checks,r.stats.event_dd_accepts,r.stats.event_qf_refinements,r.stats.qf_family_constructions,r.stats.event_topology_reuses,r.stats.event_radius_reuses,r.stats.event_direct_qf_failures,r.stats.first_value_pass_nodes,r.stats.first_gradient_error_pass_nodes,r.stats.first_gradient_contract_nodes,r.stats.gradient_error_pass_mask,r.stats.gradient_contract_mask,r.stats.physical_ms,r.stats.estimator_ms,r.stats.scheduler_ms,fixedms,fixed,int(fixedstatus),r.inner_error[0],r.geometry_error[0],r.event_error[0],r.roundoff_error[0]);
            for(double g:r.grad_mu)printf(",%.17g",g);for(double e:r.estimated_abs_error_grad)printf(",%.8g",e);puts("");fflush(stdout);
        }
    }
}

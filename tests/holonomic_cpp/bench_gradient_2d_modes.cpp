#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace lcbinint::holonomic;

struct Case {
    LensParams p;
    double u=0;
    std::string name;
};

static std::vector<Case> read_cases(const char* path, std::size_t limit) {
    std::ifstream in(path);
    std::vector<Case> out;
    std::string line;
    while(out.size()<limit && std::getline(in,line)) {
        if(line.empty() || line[0]=='#')continue;
        Case c;
        int bary=0;
        double reference=0;
        std::istringstream row(line);
        if(!(row>>c.p.xs>>c.p.ys>>c.p.rho>>c.p.q>>c.p.a>>bary>>
             c.u>>reference>>c.name))continue;
        c.p.barycentric=bary!=0;
        out.push_back(std::move(c));
    }
    return out;
}

static const char* mode_name(int mode) {
    switch(mode) {
    case 0:return "radial_only";
    case 1:return "dual_gc2";
    case 2:return "dual_nested15";
    case 3:return "dual_nested15_gradlocal";
    case 4:return "gradient_deferred16";
    }
    return "unknown";
}

static AdaptiveConfig make_config(GradientPolicy policy,int mode,bool generous) {
    AdaptiveConfig cfg;
    cfg.gradient_policy=policy;
    cfg.with_jacobian=true;
    cfg.tol.mu_rtol=1e-5;
    cfg.tol.grad_atol.fill(1e-6);
    cfg.tol.grad_rtol.fill(1e-4);
    cfg.initial_level=3;
    // The production adaptive lattice is deliberately capped at level 8;
    // the generous arm changes only the work budget and panel depth so the
    // experiment does not silently introduce a different radial rule.
    cfg.max_level=8;
    cfg.max_depth=generous?32:12;
    cfg.max_panels=generous?8192:512;
    cfg.max_node_evals=generous?262144:65536;
    cfg.value_first_gradient_node_budget=generous?262144:32768;
    cfg.value_first_gradient_round_budget=generous?256:64;
    cfg.max_bytes=generous?256*1024*1024:64*1024*1024;
    cfg.angular_adaptive=mode==1 || mode==2 || mode==3;
    cfg.angular_initial_order=16;
    cfg.angular_max_order=512;
    cfg.angular_nested=mode==2 || mode==3;
    cfg.angular_nested_initial_order=15;
    cfg.angular_nested_max_order=511;
    cfg.gradient_local_refinement=mode==3;
    cfg.gradient_adaptive=mode==4;
    if(mode==4) {
        cfg.gradient_adaptive_node_budget=8192;
        cfg.gradient_adaptive_round_budget=16;
        cfg.gradient_adaptive_dominance=1.25;
    }
    return cfg;
}

static void print_quality(const AdaptiveResult& result) {
    for(int j=0;j<5;++j) {
        if(j)std::putchar('|');
        std::printf("%s:%s",gradient_quality_name(result.grad_quality[j]),
                    gradient_reason_name(result.grad_reason[j]));
    }
}

int main(int argc,char** argv) {
    if(argc<2)return 2;
    const std::size_t limit=argc>2?std::strtoull(argv[2],nullptr,10):108;
    const bool warm=argc>3&&std::atoi(argv[3])!=0;
    const int repeats=argc>4?std::atoi(argv[4]):1;
    const bool generous=argc>5&&std::atoi(argv[5])!=0;
    // Optional mode selector: 0=all, 1=radial_only, 2=dual_gc2,
    // 3=dual_nested15_gradlocal, 4=dual_nested15_selective.  Keeping this selector in the research harness lets
    // the expensive generous-budget run focus on the actual dual path.
    const int only_mode=argc>6?std::atoi(argv[6]):0;
    const auto cases=read_cases(argv[1],limit);
    if(cases.empty())return 3;

    std::printf("name,u,policy,mode,warm,rep,stop,value_stop,gradient_stop,value_converged,"
                "mu,value_error,whole_ms,unique_nodes,evaluations,reused_nodes,splits,"
                "radial_error0,inner_error0,gradient_error_mask,gradient_contract_mask,"
                "angular_samples,angular_upgrades,angular_point_evaluations,"
                "angular_cached_point_reuses,order16,order24,order32,order64,order128,"
                "order256,order512,grad_quality_reason,grad_error0,grad_error1,"
                "grad_error2,grad_error3,grad_error4\n");

    const GradientPolicy policies[]={GradientPolicy::Strict,GradientPolicy::ValueFirst};
    (void)fejer_rule(8);
    for(const auto& c:cases) {
        for(const auto policy:policies) {
            const int mode_begin=only_mode>0?only_mode-1:0;
            const int mode_end=only_mode>0?only_mode:5;
            for(int mode=mode_begin;mode<mode_end;++mode) {
                const AdaptiveConfig cfg=make_config(policy,mode,generous);
                AdaptiveWorkspace workspace;
                PreparedEpochGeometry prepared;
                LensParams p=c.p;
                if(warm) {
                    (void)epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared);
                    p.xs+=.01*p.rho;
                    (void)epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared);
                } else {
                    AdaptiveWorkspace warmup;
                    (void)epoch_adaptive(p,c.u,cfg,warmup);
                }
                for(int rep=0;rep<repeats;++rep) {
                    const auto start=adaptive_detail::Clock::now();
                    const AdaptiveResult result=warm
                        ? epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared)
                        : epoch_adaptive(p,c.u,cfg,workspace);
                    const double elapsed=adaptive_detail::ms(start);
                    const auto& s=result.stats;
                    const auto& h=s.angular_order_histogram;
                    std::printf("%s,%.17g,%s,%s,%d,%d,%s,%s,%s,%d,%.17g,%.9g,%.9g,"
                                "%zu,%zu,%zu,%zu,%.9g,%.9g,%u,%u,%zu,%zu,%zu,%zu,"
                                "%zu,%zu,%zu,%zu,%zu,%zu,%zu,",
                        c.name.c_str(),c.u,gradient_policy_name(policy),mode_name(mode),
                        int(warm),rep,adaptive_stop_name(result.stop),
                        adaptive_stop_name(result.value_stop_reason),
                        adaptive_stop_name(result.gradient_stop_reason),
                        int(result.value_converged),result.mu,result.estimated_abs_error_mu,
                        elapsed,s.unique_nodes,s.node_evaluations,s.reused_nodes,s.splits,
                        result.radial_error[0],result.inner_error[0],
                        s.gradient_error_pass_mask,s.gradient_contract_mask,
                        s.angular_samples,s.angular_upgrades,s.angular_point_evaluations,
                        s.angular_cached_point_reuses,h[0],h[1],h[2],h[3],h[4],h[5],h[6]);
                    print_quality(result);
                    std::printf(",%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        result.estimated_abs_error_grad[0],result.estimated_abs_error_grad[1],
                        result.estimated_abs_error_grad[2],result.estimated_abs_error_grad[3],
                        result.estimated_abs_error_grad[4]);
                    std::fflush(stdout);
                }
            }
        }
    }
}

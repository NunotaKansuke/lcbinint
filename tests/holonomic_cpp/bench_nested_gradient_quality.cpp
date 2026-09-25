#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace lcbinint::holonomic;

struct Case { LensParams p; double u=0; std::string name; };

static std::vector<Case> read_cases(const char* path, std::size_t limit) {
    std::ifstream in(path);
    std::vector<Case> out;
    std::string line;
    while(out.size()<limit && std::getline(in,line)) {
        Case c; int bary=0; double reference=0;
        std::istringstream row(line);
        if(!(row>>c.p.xs>>c.p.ys>>c.p.rho>>c.p.q>>c.p.a>>bary>>
             c.u>>reference>>c.name)) continue;
        c.p.barycentric=bary!=0;
        out.push_back(std::move(c));
    }
    return out;
}

int main(int argc,char** argv) {
    if(argc<2)return 2;
    const auto cases=read_cases(argv[1],argc>2?std::strtoull(argv[2],nullptr,10):108);
    const bool warm=argc>3&&std::atoi(argv[3])!=0;
    const int repeats=argc>4?std::atoi(argv[4]):1;
    if(cases.empty())return 3;
    std::printf("name,u,policy,arm,warm,rep,stop,value_stop,gradient_stop,value_converged,"
                "mu,value_error,whole_ms,gradient_error_mask,gradient_contract_mask,"
                "nodes,evaluations,nested_upgrades,"
                "nested_point_evals,nested_cached_reuses,order7,order15,order31,"
                "order63,order127,order255,order511,grad0,grad1,grad2,grad3,grad4,"
                "reason0,reason1,reason2,reason3,reason4,grad_error0,grad_error1,"
                "grad_error2,grad_error3,grad_error4\n");
    const GradientPolicy policies[]={GradientPolicy::Strict,GradientPolicy::ValueFirst};
    const int arms[]={15,7};
    (void)fejer_rule(8);
    for(const auto& c:cases) for(const auto policy:policies) for(const int initial:arms) {
        AdaptiveConfig cfg;
        cfg.gradient_policy=policy;
        cfg.with_jacobian=true;
        cfg.tol.mu_rtol=1e-5;
        cfg.tol.grad_atol.fill(1e-6);
        cfg.tol.grad_rtol.fill(1e-4);
        cfg.value_first_gradient_node_budget=32768;
        cfg.value_first_gradient_round_budget=64;
        cfg.max_node_evals=65536;
        cfg.max_level=8;
        cfg.max_depth=12;
        cfg.angular_adaptive=true;
        cfg.angular_nested=true;
        cfg.angular_nested_initial_order=initial;
        cfg.angular_nested_max_order=511;
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
            const auto result=warm
                ? epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared)
                : epoch_adaptive(p,c.u,cfg,workspace);
            const double elapsed=adaptive_detail::ms(start);
            const auto& s=result.stats;
            const auto& h=s.nested_angular_order_histogram;
            std::printf("%s,%.17g,%s,nested%d,%d,%d,%s,%s,%s,%d,%.17g,%.9g,%.9g,"
                        "%u,%u,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
                        "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                c.name.c_str(),c.u,gradient_policy_name(policy),initial,int(warm),rep,
                adaptive_stop_name(result.stop),adaptive_stop_name(result.value_stop_reason),
                adaptive_stop_name(result.gradient_stop_reason),int(result.value_converged),
                result.mu,result.estimated_abs_error_mu,elapsed,
                s.gradient_error_pass_mask,s.gradient_contract_mask,
                s.unique_nodes,s.node_evaluations,s.nested_angular_upgrades,
                s.nested_angular_point_evaluations,s.nested_angular_cached_point_reuses,
                h[0],h[1],h[2],h[3],h[4],h[5],h[6],
                gradient_quality_name(result.grad_quality[0]),
                gradient_quality_name(result.grad_quality[1]),
                gradient_quality_name(result.grad_quality[2]),
                gradient_quality_name(result.grad_quality[3]),
                gradient_quality_name(result.grad_quality[4]),
                gradient_reason_name(result.grad_reason[0]),
                gradient_reason_name(result.grad_reason[1]),
                gradient_reason_name(result.grad_reason[2]),
                gradient_reason_name(result.grad_reason[3]),
                gradient_reason_name(result.grad_reason[4]),
                result.estimated_abs_error_grad[0],result.estimated_abs_error_grad[1],
                result.estimated_abs_error_grad[2],result.estimated_abs_error_grad[3],
                result.estimated_abs_error_grad[4]);
            std::fflush(stdout);
        }
    }
}

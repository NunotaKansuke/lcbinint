#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

#include <algorithm>
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
    std::vector<Case> cases;
    std::string line;
    while (cases.size()<limit && std::getline(in,line)) {
        Case c;
        int barycentric=0;
        double reference=0;
        std::istringstream row(line);
        if (!(row>>c.p.xs>>c.p.ys>>c.p.rho>>c.p.q>>c.p.a>>barycentric>>
              c.u>>reference>>c.name))continue;
        c.p.barycentric=barycentric!=0;
        cases.push_back(c);
    }
    return cases;
}

static const char* arm_name(int arm_index) {
    if(arm_index==0)return "incumbent";
    if(arm_index==1)return "angular_adaptive";
    return arm_index==2 ? "nested_angular_adaptive" :
                          "nested_angular_adaptive_7";
}

int main(int argc,char** argv) {
    if (argc<2) return 2;
    const std::size_t limit=argc>2?std::strtoull(argv[2],nullptr,10):110;
    const int repeats=argc>3?std::atoi(argv[3]):3;
    const bool warm=argc>4?std::atoi(argv[4])!=0:false;
    const bool nested_only=argc>5?std::atoi(argv[5])!=0:false;
    const auto cases=read_cases(argv[1],limit);
    if (cases.empty()) return 3;

    std::printf("name,u,policy,arm,warm,rep,stop,value_converged,mu,mu_error,"
                "whole_ms,topology_ms,setup_ms,physical_ms,estimator_ms,"
                "scheduler_ms,unique_nodes,node_evaluations,reused_nodes,splits,"
                "inner,radial,angular_samples,angular_upgrades,"
                "angular_geometry_reuses,angular_upgrade_failures,"
                "angular_rule_evaluations,"
                "angular_point_evaluations,angular_order16,angular_order24,"
                "angular_order32,angular_order64,angular_order128,"
                "angular_order256,angular_order512,angular_nested,"
                "angular_cached_point_reuses,nested_angular_samples,"
                "nested_angular_upgrades,nested_angular_point_evaluations,"
                "nested_angular_cached_point_reuses,nested_angular_weighted_sum_reuses,"
                "nested_angular_cache_failures,nested_order7,nested_order15,"
                "nested_order31,nested_order63,nested_order127,nested_order255,"
                "nested_order511\n");

    // Initialize static quadrature tables outside the measured rows.  The
    // arm comparison still measures the full epoch path, including topology
    // and adaptive setup; this only removes one-time C++ initialization noise.
    (void)fejer_rule(8);
    for (const auto& c:cases) {
        for (const auto policy:{GradientPolicy::None,GradientPolicy::ValueFirst}) {
            for (int arm_index=0;arm_index<4;++arm_index) {
                const bool angular=arm_index!=0;
                const bool nested=arm_index>=2;
                if(nested_only && !nested)continue;
                AdaptiveConfig cfg;
                cfg.gradient_policy=policy;
                cfg.with_jacobian=policy!=GradientPolicy::None;
                cfg.tol.mu_rtol=1e-5;
                cfg.tol.grad_atol.fill(1e-6);
                cfg.tol.grad_rtol.fill(1e-4);
                cfg.value_first_gradient_node_budget=32768;
                cfg.value_first_gradient_round_budget=64;
                cfg.max_node_evals=65536;
                cfg.max_level=8;
                cfg.max_depth=12;
                cfg.angular_adaptive=angular;
                cfg.angular_initial_order=16;
                cfg.angular_max_order=512;
                cfg.angular_nested=nested;
                cfg.angular_nested_initial_order=(arm_index==3)?7:15;
                cfg.angular_nested_max_order=511;

                AdaptiveWorkspace workspace;
                PreparedEpochGeometry prepared;
                LensParams p=c.p;
                if (warm) {
                    (void)epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared);
                    p.xs+=.01*p.rho;
                }
                // A single unrecorded call per arm/policy keeps lazy static
                // table construction out of the paired rows.  The caller
                // chooses warm=0/1 separately; no cross-epoch sample cache is
                // used by this diagnostic.
                AdaptiveWorkspace init_workspace;
                PreparedEpochGeometry init_prepared;
                if (warm)
                    (void)epoch_adaptive_prepared(p,c.u,cfg,init_workspace,init_prepared);
                else
                    (void)epoch_adaptive(p,c.u,cfg,init_workspace);
                for (int rep=0;rep<repeats;++rep) {
                    AdaptiveResult result;
                    double elapsed=0;
                    auto start=adaptive_detail::Clock::now();
                    if (warm)
                        result=epoch_adaptive_prepared(p,c.u,cfg,workspace,prepared);
                    else
                        result=epoch_adaptive(p,c.u,cfg,workspace);
                    elapsed=adaptive_detail::ms(start);
                    const auto& s=result.stats;
                    const auto& h=s.angular_order_histogram;
                std::printf("%s,%.17g,%s,%s,%d,%d,%s,%d,%.17g,%.9g,"
                                "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%zu,%zu,%zu,%zu,"
                                "%.9g,%.9g,%zu,%zu,%zu,%zu,%zu,%zu,"
                                "%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n",
                        c.name.c_str(),c.u,gradient_policy_name(policy),arm_name(arm_index),
                        int(warm),rep,adaptive_stop_name(result.stop),
                        int(result.value_converged),result.mu,
                        result.estimated_abs_error_mu,elapsed,s.topology_ms,s.setup_ms,
                        s.physical_ms,s.estimator_ms,s.scheduler_ms,s.unique_nodes,
                        s.node_evaluations,s.reused_nodes,s.splits,result.inner_error[0],
                        result.radial_error[0],s.angular_samples,s.angular_upgrades,
                        s.angular_geometry_reuses,s.angular_upgrade_failures,
                        s.angular_rule_evaluations,
                        s.angular_point_evaluations,h[0],h[1],h[2],h[3],h[4],h[5],h[6],
                        nested?1:0,s.angular_cached_point_reuses,
                        s.nested_angular_samples,s.nested_angular_upgrades,
                        s.nested_angular_point_evaluations,
                        s.nested_angular_cached_point_reuses,
                        s.nested_angular_weighted_sum_reuses,
                        s.nested_angular_cache_failures,
                        s.nested_angular_order_histogram[0],
                        s.nested_angular_order_histogram[1],
                        s.nested_angular_order_histogram[2],
                        s.nested_angular_order_histogram[3],
                        s.nested_angular_order_histogram[4],
                        s.nested_angular_order_histogram[5],
                        s.nested_angular_order_histogram[6]);
                    std::fflush(stdout);
                }
            }
        }
    }
    return 0;
}

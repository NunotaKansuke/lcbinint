// Phase 7 shared-geometry correctness gate.
//
// A single V2 topology/cell/root/F0 plan is passed to both LD evaluators.
// The test checks that the harness preserves the V2 K-rule result, that the
// equation-derived PF6/fold lane agrees with it on a clean case, and that the
// warm D14 plan preserves the same observables.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "lcbinint/magnification/holonomic/phase7_shared_harness.hpp"

using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::phase7_detail;

namespace {

int failures = 0;

void require(bool condition, const char* what) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

double relerr(double a, double b) {
    return std::fabs(a - b) /
           (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

void check_value_and_warm() {
    const LensParams p{0.9, -0.3, 0.05, 0.25, 0.7, false};
    constexpr double u = 0.5;
    constexpr int n_r = 64;

    const SharedEpochPlan cold = build_shared_plan(p, u, n_r, false);
    const SharedEpochEvaluation v2 = evaluate_v2_k_rule(cold, false);
    const SharedEpochEvaluation pf6 = evaluate_pf6_fold<20>(cold, false);
    require(cold.status == Status::OK, "cold shared plan is clean");
    require(v2.status == Status::OK && pf6.status == Status::OK,
            "value evaluators return OK");
    require(pf6.all_true && pf6.cost.failed == 0,
            "value PF6/fold lane has no failed node");
    require(pf6.cost.incumbent_angular_fallback == 0,
            "value PF6/fold lane has no angular/reseed fallback");
    require(v2.F0 == pf6.F0 && v2.dF0_internal == pf6.dF0_internal,
            "both evaluators reuse the shared F0");
    require(relerr(v2.mu, pf6.mu) < 5e-7,
            "value PF6/fold agrees with shared V2");

    const SharedEpochPlan warm = build_shared_plan(
        p, u, n_r, false, &cold.d14_roots);
    const SharedEpochEvaluation warm_v2 = evaluate_v2_k_rule(warm, false);
    const SharedEpochEvaluation warm_pf6 = evaluate_pf6_fold<20>(warm, false);
    require(warm.topology.from_warm_d14, "warm plan records D14 reuse");
    require(warm_v2.status == Status::OK && warm_pf6.status == Status::OK,
            "warm value evaluators return OK");
    require(warm_pf6.all_true && warm_pf6.cost.failed == 0,
            "warm value PF6/fold lane has no failed node");
    require(relerr(pf6.mu, warm_pf6.mu) < 5e-7,
            "warm PF6/fold agrees with cold PF6/fold");
    require(relerr(v2.mu, warm_v2.mu) < 5e-7,
            "warm V2 agrees with cold V2");
}

void check_jacobian() {
    const LensParams p{0.9, -0.3, 0.05, 0.25, 0.7, false};
    constexpr double u = 0.5;
    constexpr int n_r = 64;

    const SharedEpochPlan plan = build_shared_plan(p, u, n_r, true);
    const SharedEpochEvaluation v2 = evaluate_v2_k_rule(plan, true);
    const SharedEpochEvaluation pf6 = evaluate_pf6_fold<20>(plan, true);
    require(plan.status == Status::OK, "Jacobian shared plan is clean");
    require(v2.status == Status::OK && pf6.status == Status::OK,
            "Jacobian evaluators return OK");
    require(pf6.all_true && pf6.cost.failed == 0,
            "analytic PF6/fold lane has no failed node");
    require(pf6.cost.incumbent_angular_fallback == 0,
            "analytic PF6/fold lane has no angular/reseed fallback");
    require(relerr(v2.mu, pf6.mu) < 5e-7,
            "analytic PF6/fold value agrees with shared V2");
    for (int j = 0; j < 5; ++j)
        require(relerr(v2.grad_mu[j], pf6.grad_mu[j]) < 5e-3,
                "analytic PF6/fold gradient agrees with shared V2");

    // The existing V2 route is an independent harness-level parity check.
    // Force its K-rule on and keep the root-pair/ODE alternatives off.
    holo_holonomic_transport_override() = 1;
    holo_mv_transport_override() = 0;
    holo_ode_transport_override() = 0;
    const EpochJacobian incumbent = epoch_jacobian(p, u, n_r, false);
    holo_holonomic_transport_override() = 0;
    require(relerr(v2.mu, incumbent.mu) < 5e-7,
            "shared V2 K-rule agrees with incumbent V2 route");
    for (int j = 0; j < 5; ++j)
        require(relerr(v2.grad_mu[j], incumbent.grad_mu[j]) < 5e-3,
                "shared V2 gradient agrees with incumbent V2 route");
}

}  // namespace

int main() {
    holo_holonomic_transport_override() = 0;
    holo_mv_transport_override() = 0;
    holo_ode_transport_override() = 0;
    check_value_and_warm();
    check_jacobian();
    std::printf("gm_phase7_shared %s failures=%d\n",
                failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}

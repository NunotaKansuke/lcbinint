// ATPT holonomic solver -- full coupled-state ODE transport correctness +
// full-ODE audit (fast; ctest).
//
// Checks, over the bench-case TSV, with HOLO_ODE_TRANSPORT forced ON (V3) vs
// every transport OFF (V0):
//
//   1. value lane:  epoch_value V3 mu vs V0 mu   -- |dmu/mu| <= 5e-4
//   2. jac lane:     epoch_jacobian V3 mu vs V0 mu -- |dmu/mu| <= 5e-4
//   3. jac lane:     V3 grad_mu vs V0 grad_mu      -- worst rel-or-abs <= 5e-2
//                    (the rho-derivative catastrophic-cancellation gate,
//                     kOdeRhoCancelMax, keeps V3 == V0 on the egregious cases;
//                     the residual is the ODE-vs-direct disagreement on the
//                     mild-cancellation cases, where V3 is in fact closer to a
//                     high-n_r reference -- see evidence/holonomic).
//   4. status parity: V3 never downgrades a V0 OK status.
//   5. full-ODE audit: summed over every V3 jac-lane epoch,
//        angular_sweep_nodes == 0  AND  per_node_quartic_solves == 0
//      i.e. the ODE path really did replace the per-node quartic solve and the
//      per-node angular sqrt(phi) quadrature -- it is not a semi-holonomic path
//      in disguise.
//
// Run:  test_holonomic_ode <bench_cases.tsv>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/holonomic_ode_transport.hpp"
#include "lcbinint/magnification/holonomic/holonomic_transport.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

using namespace lcbinint::holonomic;

namespace {
constexpr double kMuTol = 5.0e-4;
constexpr double kGradTol = 5.0e-2;
}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }

    int n = 0, fails = 0;
    double worst_mu = 0.0, worst_grad = 0.0;
    std::string worst_mu_name, worst_grad_name;
    int status_downgrades = 0;

    OdeCounters audit;
    audit.reset();
    OdeCounters audit_val;
    audit_val.reset();

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        double xs, ys, rho, q, a, u, tj;
        int bary;
        std::string name;
        if (!(ls >> xs >> ys >> rho >> q >> a >> bary >> u >> tj >> name))
            continue;
        ++n;
        LensParams p{xs, ys, rho, q, a, (bool)bary};

        holo_mv_transport_override() = 0;
        holo_holonomic_transport_override() = 0;
        holo_ode_transport_override() = 0;
        EpochJacobian j0 = epoch_jacobian(p, u, 64, false);
        EpochValue v0 = epoch_value(p, u, 64, false);

        holo_mv_transport_override() = 1;
        holo_holonomic_transport_override() = 0;
        holo_ode_transport_override() = 1;
        ode_counters().reset();
        EpochJacobian j3 = epoch_jacobian(p, u, 64, false);
        {
            const OdeCounters& c = ode_counters();
            audit.angular_sweep_nodes += c.angular_sweep_nodes;
            audit.per_node_quartic_solves += c.per_node_quartic_solves;
            audit.cells_seen += c.cells_seen;
            audit.cells_ode += c.cells_ode;
            audit.cells_fallback += c.cells_fallback;
            audit.rho_cancel_gated += c.rho_cancel_gated;
            audit.jet_eligible_cells += c.jet_eligible_cells;
            audit.jet_cells += c.jet_cells;
            audit.r2_packet_cells += c.r2_packet_cells;
            audit.r2_packet_demote += c.r2_packet_demote;
            audit.r2_packet_seed_fail += c.r2_packet_seed_fail;
            audit.r2_jac_cells += c.r2_jac_cells;
            audit.r2_jac_demote += c.r2_jac_demote;
            audit.jet_rhs_evals += c.jet_rhs_evals;
            audit.jet_seed_krule_evals += c.jet_seed_krule_evals;
            audit.krule_gc2_evals += c.krule_gc2_evals;
        }
        ode_counters().reset();
        EpochValue v3 = epoch_value(p, u, 64, false);
        {
            const OdeCounters& c = ode_counters();
            audit_val.cells_seen += c.cells_seen;
            audit_val.cells_ode += c.cells_ode;
            audit_val.cells_fallback += c.cells_fallback;
            audit_val.jet_eligible_cells += c.jet_eligible_cells;
            audit_val.jet_cells += c.jet_cells;
            audit_val.r2_packet_cells += c.r2_packet_cells;
            audit_val.r2_packet_demote += c.r2_packet_demote;
            audit_val.r2_packet_seed_fail += c.r2_packet_seed_fail;
            audit_val.krule_gc2_evals += c.krule_gc2_evals;
        }

        double b = std::fabs(v0.mu) > 0 ? std::fabs(v0.mu) : 1.0;
        double dmv = std::fabs(v3.mu - v0.mu) / b;
        double dmj = std::fabs(j3.mu - j0.mu) / b;
        double dm = std::max(dmv, dmj);
        if (dm > worst_mu) { worst_mu = dm; worst_mu_name = name; }
        if (dm > kMuTol) {
            ++fails;
            std::fprintf(stderr, "  MU FAIL  %-12s |dmu/mu|=%.3e (val %.3e jac %.3e)\n",
                         name.c_str(), dm, dmv, dmj);
        }

        double gfloor = 1e-6 * (std::fabs(j0.mu) + 1.0), gw = 0.0;
        for (int k = 0; k < 5; ++k) {
            double base = std::max({std::fabs(j0.grad_mu[k]),
                                    std::fabs(j3.grad_mu[k]), gfloor});
            gw = std::max(gw, std::fabs(j0.grad_mu[k] - j3.grad_mu[k]) / base);
        }
        if (gw > worst_grad) { worst_grad = gw; worst_grad_name = name; }
        if (gw > kGradTol) {
            ++fails;
            std::fprintf(stderr, "  GRAD FAIL %-12s worst rel=%.3e\n",
                         name.c_str(), gw);
        }

        if (j0.status == Status::OK && j3.status != Status::OK) {
            ++status_downgrades;
            std::fprintf(stderr, "  STATUS DOWNGRADE %-12s\n", name.c_str());
        }
    }
    holo_mv_transport_override() = -1;
    holo_holonomic_transport_override() = -1;
    holo_ode_transport_override() = -1;

    std::printf("cases                 %d\n", n);
    std::printf("worst |dmu/mu|        %.3e  (%s)\n", worst_mu, worst_mu_name.c_str());
    std::printf("worst grad rel        %.3e  (%s)\n", worst_grad,
                worst_grad_name.c_str());
    std::printf("status downgrades     %d\n", status_downgrades);
    std::printf("ODE cells             %ld seen / %ld ode / %ld fb   epochs gated %ld\n",
                audit.cells_seen, audit.cells_ode, audit.cells_fallback,
                audit.rho_cancel_gated);
    std::printf("angular_sweep_nodes   %ld  (must be 0)\n", audit.angular_sweep_nodes);
    std::printf("per_node_quartic_solv %ld  (must be 0)\n",
                audit.per_node_quartic_solves);
    std::printf("regime-2 jet          %ld eligible / %ld active   "
                "seed K-rule %ld  Horner RHS %ld  on-march K-rule %ld\n",
                audit.jet_eligible_cells, audit.jet_cells,
                audit.jet_seed_krule_evals, audit.jet_rhs_evals,
                audit.krule_gc2_evals - audit.jet_seed_krule_evals);
    std::printf("regime-2 r2 packet    %ld cells   %ld demote / %ld seed-fail\n",
                audit.r2_packet_cells, audit.r2_packet_demote,
                audit.r2_packet_seed_fail);
    std::printf("JAC lane r2 packet    %ld cells   %ld demote\n",
                audit.r2_jac_cells, audit.r2_jac_demote);
    std::printf("VALUE lane: %ld seen / %ld ode / %ld fb   jet %ld/%ld   "
                "r2 %ld cells (%ld demote / %ld seed-fail)   on-march K-rule %ld\n",
                audit_val.cells_seen, audit_val.cells_ode, audit_val.cells_fallback,
                audit_val.jet_eligible_cells, audit_val.jet_cells,
                audit_val.r2_packet_cells, audit_val.r2_packet_demote,
                audit_val.r2_packet_seed_fail, audit_val.krule_gc2_evals);

    bool audit_ok = (audit.angular_sweep_nodes == 0 &&
                     audit.per_node_quartic_solves == 0);
    // the regime-2 fast path must actually engage on this stress set (a
    // silently-disabled jet/r2 would still pass parity but lose the speed win).
    if (audit.jet_cells + audit.r2_packet_cells + audit.r2_jac_cells == 0) {
        ++fails;
        std::fprintf(stderr, "  AUDIT FAIL: regime-2 jet + r2 packet never engaged\n");
    }
    if (!audit_ok) {
        ++fails;
        std::fprintf(stderr, "  AUDIT FAIL: ODE path used angular sweep / per-node "
                             "quartic solve\n");
    }
    if (status_downgrades > 0) fails += status_downgrades;

    std::printf("\n%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

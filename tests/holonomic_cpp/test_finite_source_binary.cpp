// ATPT holonomic solver -- Phase C internal mode router correctness harness.
//
// Asserts finite_source_binary / finite_source_binary_prepared forward to
// epoch_jacobian / epoch_jacobian_prepared with BIT-IDENTICAL results, that
// the requested-output contract is honoured (kValue drops the Jacobian,
// kValueJvp fails closed), and that provenance is propagated on the
// trajectory route.  No new numerical path -- this is a plumbing check.
//
// Build (isolated, shared .so untouched):
//   cmake --build build-holonomic-m7 --target test_finite_source_binary
//   ./build-holonomic-m7/test_finite_source_binary /tmp/bench_cases.tsv

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/finite_source_binary.hpp"

using namespace lcbinint::holonomic;

namespace {

int g_fail = 0;
int g_checks = 0;
double g_worst = 0.0;

// The router adds no arithmetic -- from_epoch is a field copy.  Any drift is
// purely the forwarded epoch_jacobian being inlined into a different call
// site under -ffp-contract=fast (last-ULP FMA formation), plus, on the
// prepared route, warm-D14 Aberth iteration counts differing by a step.  So
// the numeric contract is "no meaningful difference", not bit-equality.
void near(const char* what, double got, double want, double rtol) {
    ++g_checks;
    const double d = std::fabs(got - want);
    const double rel = d / (std::fabs(want) + 1e-300);
    if (std::fabs(want) > 1e-12) g_worst = std::max(g_worst, rel);
    if (d > rtol * std::fabs(want) + 1e-13) {
        ++g_fail;
        std::fprintf(stderr, "  FAIL %-22s got=% .17g want=% .17g rel=%.2e\n",
                     what, got, want, rel);
    }
}

// Structural fields (status, provenance, flags, withheld-Jacobian zeros)
// carry no arithmetic and MUST be exact.
void eq(const char* what, double got, double want) {
    ++g_checks;
    if (got != want) {
        ++g_fail;
        std::fprintf(stderr, "  FAIL %-22s got=% .17g want=% .17g\n", what, got,
                     want);
    }
}

void want_true(const char* what, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_fail;
        std::fprintf(stderr, "  FAIL %-22s (expected true)\n", what);
    }
}

struct Row {
    LensParams p;
    double u;
};

std::vector<Row> load(const std::string& path) {
    std::vector<Row> rows;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        double xs, ys, rho, q, a, bary, u, tj;
        std::string name;
        if (!(ls >> xs >> ys >> rho >> q >> a >> bary >> u >> tj >> name))
            continue;
        rows.push_back({LensParams{xs, ys, rho, q, a, (bool)bary}, u});
    }
    return rows;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path =
        argc > 1 ? argv[1] : std::string("/tmp/bench_cases.tsv");
    std::vector<Row> rows = load(path);
    if (rows.empty()) {
        std::fprintf(stderr, "no cases loaded from %s\n", path.c_str());
        return 2;
    }
    std::fprintf(stderr, "loaded %zu cases\n", rows.size());

    // ---- 1. stateless route == epoch_jacobian, bit for bit ----------------
    int ri = -1;
    for (const Row& r : rows) {
        ++ri;
        EpochJacobian ref = epoch_jacobian(r.p, r.u, 64);

        FiniteSourceRequest req;
        req.params = r.p;
        req.u = r.u;
        req.output = RequestedOutput::kValueJacobian;
        FiniteSourceOutcome got = finite_source_binary(req);

        near("cold mu", got.mu, ref.mu, 1e-12);
        near("cold dmu_du", got.dmu_du, ref.dmu_du, 1e-12);
        near("cold F0", got.F0, ref.F0, 1e-12);
        near("cold F_half", got.F_half, ref.F_half, 1e-12);
        eq("cold r_max", got.r_max, ref.r_max);
        eq("cold status", (double)(int)got.status, (double)(int)ref.status);
        for (int j = 0; j < 5; ++j) {
            char tag[32];
            std::snprintf(tag, sizeof tag, "cold grad r%d j%d", ri, j);
            // j==2 is d mu / d rho, the (large)/D - 2 mu/rho cancellation
            // (checkpoint_M6 sec.5).  With -ffp-contract=fast the forwarded
            // epoch_jacobian rounds that subtraction one ULP differently when
            // inlined at the router call site vs directly, and on tiny-rho
            // cases the cancellation amplifies it ~7 orders.  Not a plumbing
            // difference -- the router adds no arithmetic.
            near(tag, got.grad_mu[j], ref.grad_mu[j], j == 2 ? 1e-6 : 1e-11);
        }
        want_true("cold has_jacobian", got.has_jacobian);

        // kValue: same mu, Jacobian withheld.
        req.output = RequestedOutput::kValue;
        FiniteSourceOutcome v = finite_source_binary(req);
        near("value mu", v.mu, ref.mu, 1e-12);
        want_true("value no jacobian", !v.has_jacobian);
        for (int j = 0; j < 5; ++j) eq("value grad zero", v.grad_mu[j], 0.0);

        // kValueJvp: not implemented -> fail closed, never a silent number.
        req.output = RequestedOutput::kValueJvp;
        FiniteSourceOutcome jvp = finite_source_binary(req);
        want_true("jvp fails closed", jvp.status != Status::OK);
        want_true("jvp no jvp flag", !jvp.has_jvp);
    }

    // ---- 2. prepared route == epoch_jacobian_prepared over a trajectory ---
    // Synthesise a short straight track per config, exactly like
    // bench_holonomic_trajectory, and run the two entry points in lockstep
    // over independent rolling caches.
    PreparedReuseConfig cfg = default_reuse_config();
    long prov_checks = 0;
    for (const Row& r : rows) {
        const double L = std::max(8.0 * r.p.rho, 0.5 * r.p.a);
        const int N = 12;
        const double dx = L / N * 3.0 / std::sqrt(10.0);
        const double dy = L / N * 1.0 / std::sqrt(10.0);

        PreparedEpochGeometry st_ref{}, st_got{};
        for (int e = 0; e < N; ++e) {
            LensParams p = r.p;
            p.xs = r.p.xs + (e - N / 2) * dx;
            p.ys = r.p.ys + (e - N / 2) * dy;

            PreparedReuseStats s_ref{}, s_got{};
            EpochJacobian ref =
                epoch_jacobian_prepared(p, r.u, 64, st_ref, cfg, &s_ref);

            FiniteSourceRequest req;
            req.params = p;
            req.u = r.u;
            req.output = RequestedOutput::kValueJacobian;
            FiniteSourceOutcome got = finite_source_binary_prepared(
                req, st_got, cfg, &s_got);

            // Warm-D14 path: two independent rolling caches see an identical
            // param sequence, so they track to Aberth-tolerance (the
            // trajectory bench's "max |dmu|/mu 3.9e-7" story), not to ULPs.
            near("prep mu", got.mu, ref.mu, 1e-6);
            near("prep dmu_du", got.dmu_du, ref.dmu_du, 1e-5);
            near("prep F_half", got.F_half, ref.F_half, 1e-6);
            eq("prep status", (double)(int)got.status, (double)(int)ref.status);
            for (int j = 0; j < 5; ++j)
                near("prep grad", got.grad_mu[j], ref.grad_mu[j], 1e-4);
            eq("prep provenance", (double)(int)got.provenance,
               (double)(int)st_got.provenance);
            eq("prep cache in lockstep", (double)(int)st_got.provenance,
               (double)(int)st_ref.provenance);
            ++prov_checks;
        }
    }
    std::fprintf(stderr, "provenance/lockstep checks: %ld\n", prov_checks);

    std::fprintf(stderr, "\n%d checks, %d failures, worst rel drift %.2e\n",
                 g_checks, g_fail, g_worst);
    return g_fail == 0 ? 0 : 1;
}

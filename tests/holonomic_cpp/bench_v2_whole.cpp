// Uninstrumented whole-epoch V2 benchmark.
//
// bench_v2_profile is the production-epoch decomposition probe.  This
// companion keeps the same cold/warm geometry control flow but measures
// epoch_value(_prepared) directly for value-only and the finite-source router
// for value+5Jac.  The router itself is left unchanged.  The angular path
// with all transport overrides disabled is retained as an independent
// reference for parity.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/finite_source_binary.hpp"
#include "lcbinint/magnification/holonomic/holonomic_ode_transport.hpp"
#include "lcbinint/magnification/holonomic/holonomic_transport.hpp"
#include "lcbinint/magnification/holonomic/v2_profile.hpp"

using namespace lcbinint::holonomic;
using clock_type = std::chrono::steady_clock;

namespace {

struct Case {
    double xs, ys, rho, q, a;
    int bary;
    double u, t_jac;
    std::string name;
};

struct Lane {
    std::vector<double> wall;
    int status_ok = 0;
    int status_bad = 0;
    double max_mu_ref = 0.0;
    double max_grad_ref = 0.0;
    double checksum = 0.0;
};

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double x = p * 0.01 * (v.size() - 1);
    const size_t i = static_cast<size_t>(x);
    if (i + 1 >= v.size()) return v.back();
    return v[i] + (v[i + 1] - v[i]) * (x - i);
}

FiniteSourceRequest request(const Case& c, RequestedOutput output) {
    return FiniteSourceRequest{
        LensParams{c.xs, c.ys, c.rho, c.q, c.a, c.bary != 0}, c.u, output, 64,
        {}};
}

FiniteSourceOutcome execute_value_cold(const FiniteSourceRequest& req) {
    const EpochValue ev = epoch_value(req.params, req.u, req.n_r);
    FiniteSourceOutcome out;
    out.mu = ev.mu;
    out.F0 = ev.F0;
    out.F_half = ev.F_half;
    out.r_max = ev.r_max;
    out.status = ev.status;
    return out;
}

FiniteSourceOutcome execute_value_prepared(const FiniteSourceRequest& req,
                                           PreparedEpochGeometry& state,
                                           const PreparedReuseConfig& cfg) {
    const EpochValue ev = epoch_value_prepared(req.params, req.u, req.n_r,
                                               state, cfg);
    FiniteSourceOutcome out;
    out.mu = ev.mu;
    out.F0 = ev.F0;
    out.F_half = ev.F_half;
    out.r_max = ev.r_max;
    out.status = ev.status;
    out.provenance = state.provenance;
    return out;
}

void force_v2() {
    holo_mv_transport_override() = 1;
    holo_holonomic_transport_override() = 1;
    holo_ode_transport_override() = 0;
}

void force_angular_reference() {
    holo_mv_transport_override() = 0;
    holo_holonomic_transport_override() = 0;
    holo_ode_transport_override() = 0;
}

void compare(const FiniteSourceOutcome& got, const FiniteSourceOutcome& ref,
             bool jac, Lane& lane) {
    const double denom = std::max(std::fabs(ref.mu), 1.0);
    lane.max_mu_ref = std::max(lane.max_mu_ref,
                               std::fabs(got.mu - ref.mu) / denom);
    if (jac && got.has_jacobian && ref.has_jacobian) {
        for (int j = 0; j < 5; ++j) {
            const double d = std::fabs(got.grad_mu[j] - ref.grad_mu[j]) /
                             std::max({std::fabs(ref.grad_mu[j]),
                                       1e-6 * denom, 1e-300});
            lane.max_grad_ref = std::max(lane.max_grad_ref, d);
        }
    }
}

std::vector<Case> read_cases(const char* path) {
    std::ifstream in(path);
    std::vector<Case> out;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        Case c;
        if (ss >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
            c.t_jac >> c.name)
            out.push_back(c);
    }
    return out;
}

Case track_case(const Case& c, int k, int n) {
    const double length = std::max(8.0 * c.rho, 0.5 * c.a);
    const double s = (static_cast<double>(k) / (n - 1) - 0.5) * length;
    Case e = c;
    e.xs += s * 0.94868;
    e.ys += s * 0.31623;
    return e;
}

void add_cold(const std::vector<Case>& cases, int reps, Lane& value,
             Lane& jac) {
    for (const Case& c : cases) {
        force_angular_reference();
        const FiniteSourceOutcome ref_v = execute_value_cold(
            request(c, RequestedOutput::kValue));
        const FiniteSourceOutcome ref_j = finite_source_binary(
            request(c, RequestedOutput::kValueJacobian));

        double best_v = 1e300, best_j = 1e300;
        FiniteSourceOutcome out_v{}, out_j{};
        for (int r = 0; r < reps; ++r) {
            force_v2();
            auto t0 = clock_type::now();
            FiniteSourceOutcome v = execute_value_cold(
                request(c, RequestedOutput::kValue));
            auto t1 = clock_type::now();
            FiniteSourceOutcome j = finite_source_binary(
                request(c, RequestedOutput::kValueJacobian));
            auto t2 = clock_type::now();
            const double tv = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double tj = std::chrono::duration<double, std::milli>(t2 - t1).count();
            if (tv < best_v) { best_v = tv; out_v = v; }
            if (tj < best_j) { best_j = tj; out_j = j; }
        }
        value.wall.push_back(best_v);
        jac.wall.push_back(best_j);
        value.status_ok += out_v.status == Status::OK;
        value.status_bad += out_v.status != Status::OK;
        jac.status_ok += out_j.status == Status::OK;
        jac.status_bad += out_j.status != Status::OK;
        value.checksum += out_v.mu;
        jac.checksum += out_j.mu;
        compare(out_v, ref_v, false, value);
        compare(out_j, ref_j, true, jac);
    }
}

void add_warm(const std::vector<Case>& cases, int reps, int n, Lane& value,
             Lane& jac) {
    PreparedReuseConfig cfg = default_reuse_config();
    cfg.allow_topology_reuse = false;
    cfg.l2_drift = 1e18;
    for (const Case& c : cases) {
        std::vector<FiniteSourceOutcome> ref_v(n), ref_j(n);
        for (int k = 0; k < n; ++k) {
            const Case e = track_case(c, k, n);
            force_angular_reference();
            ref_v[k] = execute_value_cold(request(e, RequestedOutput::kValue));
            ref_j[k] = finite_source_binary(
                request(e, RequestedOutput::kValueJacobian));
        }

        std::vector<double> best_v(n, 1e300), best_j(n, 1e300);
        std::vector<FiniteSourceOutcome> out_v(n), out_j(n);
        for (int r = 0; r < reps; ++r) {
            PreparedEpochGeometry state_v, state_j;
            for (int k = 0; k < n; ++k) {
                const Case e = track_case(c, k, n);
                force_v2();
                auto t0 = clock_type::now();
                FiniteSourceOutcome v = execute_value_prepared(
                    request(e, RequestedOutput::kValue), state_v, cfg);
                auto t1 = clock_type::now();
                FiniteSourceOutcome j = finite_source_binary_prepared(
                    request(e, RequestedOutput::kValueJacobian), state_j, cfg);
                auto t2 = clock_type::now();
                const double tv = std::chrono::duration<double, std::milli>(t1 - t0).count();
                const double tj = std::chrono::duration<double, std::milli>(t2 - t1).count();
                if (tv < best_v[k]) { best_v[k] = tv; out_v[k] = v; }
                if (tj < best_j[k]) { best_j[k] = tj; out_j[k] = j; }
            }
        }
        for (int k = 1; k < n; ++k) {
            value.wall.push_back(best_v[k]);
            jac.wall.push_back(best_j[k]);
            value.status_ok += out_v[k].status == Status::OK;
            value.status_bad += out_v[k].status != Status::OK;
            jac.status_ok += out_j[k].status == Status::OK;
            jac.status_bad += out_j[k].status != Status::OK;
            value.checksum += out_v[k].mu;
            jac.checksum += out_j[k].mu;
            compare(out_v[k], ref_v[k], false, value);
            compare(out_j[k], ref_j[k], true, jac);
        }
    }
}

void report(const char* lane_name, const Lane& lane) {
    std::fprintf(stderr,
        "V2 %s cases=%zu p50=%.6f p90=%.6f p95=%.6f p99=%.6f max=%.6f "
        "status_ok=%d status_bad=%d max_mu_ref=%.3e max_grad_ref=%.3e\n",
        lane_name, lane.wall.size(), percentile(lane.wall, 50),
        percentile(lane.wall, 90), percentile(lane.wall, 95),
        percentile(lane.wall, 99), percentile(lane.wall, 100),
        lane.status_ok, lane.status_bad, lane.max_mu_ref,
        lane.max_grad_ref);
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "evidence/holonomic/gm_coverage_cases.tsv";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 8;
    const int n = argc > 3 ? std::atoi(argv[3]) : 8;
    const std::vector<Case> cases = read_cases(path);
    if (cases.empty()) {
        std::fprintf(stderr, "cannot read cases from %s\n", path);
        return 2;
    }
    std::fprintf(stderr, "loaded %zu cases reps=%d trajectory_epochs=%d n_r=64\n",
                 cases.size(), reps, n);
    Lane cold_v, cold_j, warm_v, warm_j;
    add_cold(cases, reps, cold_v, cold_j);
    add_warm(cases, reps, n, warm_v, warm_j);
    report("cold_value", cold_v);
    report("cold_value+5Jac", cold_j);
    report("warm_value_steady", warm_v);
    report("warm_value+5Jac_steady", warm_j);
    std::fprintf(stderr, "checksum=%.12f\n",
                 cold_v.checksum + cold_j.checksum + warm_v.checksum +
                     warm_j.checksum);
    return 0;
}

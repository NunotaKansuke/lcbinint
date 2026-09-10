// Production V2 path profile.
//
// This exercises the production epoch control flow.  Value-only calls enter
// epoch_value(_prepared) directly so their timing boundary remains genuinely
// value-only; Jacobian calls enter the finite-source router.  The profiler is
// installed only for the measured call.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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

struct Timed {
    double ms = 0.0;
    FiniteSourceOutcome out{};
    V2Profile profile{};
};

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double x = p * 0.01 * (v.size() - 1);
    const size_t i = static_cast<size_t>(x);
    if (i + 1 >= v.size()) return v.back();
    return v[i] + (v[i + 1] - v[i]) * (x - i);
}

void force_v2() {
    holo_mv_transport_override() = 1;
    holo_holonomic_transport_override() = 1;
    holo_ode_transport_override() = 0;
}

void restore_flags() {
    holo_mv_transport_override() = -1;
    holo_holonomic_transport_override() = -1;
    holo_ode_transport_override() = -1;
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

Timed run_cold(const FiniteSourceRequest& req, int reps) {
    Timed best;
    best.ms = 1e300;
    for (int r = 0; r < reps; ++r) {
        V2Profile p;
        V2ProfileScope scope(p);
        const auto t0 = clock_type::now();
        FiniteSourceOutcome out = req.output == RequestedOutput::kValue
                                      ? execute_value_cold(req)
                                      : finite_source_binary(req);
        const auto t1 = clock_type::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best.ms) {
            best.ms = ms;
            best.out = out;
            best.profile = p;
        }
    }
    return best;
}

struct LaneSummary {
    std::vector<double> wall;
    int status_ok = 0;
    int status_bad = 0;
    double max_mu_ref = 0.0;
    double max_grad_ref = 0.0;
    V2Profile sum{};
    std::array<int, 7> k_reason_cases{};
    std::array<std::vector<double>, 7> k_reason_rescue_ms{};
};

void record_k_reason_case(LaneSummary& lane, const V2Profile& p) {
    const V2Profile::u64 count[7] = {
        p.k_reject_nonfinite, p.k_reject_vfloor, p.k_reject_tmax,
        p.k_reject_s2, p.k_reject_b, p.k_reject_disagreement,
        p.k_reject_arc};
    for (int i = 0; i < 7; ++i) {
        if (count[i] == 0) continue;
        ++lane.k_reason_cases[i];
        lane.k_reason_rescue_ms[i].push_back(p.angular_rescue_ms);
    }
}

void add_profile(V2Profile& d, const V2Profile& s) {
#define ADD(name) d.name += s.name
    ADD(radial_event_calls); ADD(d14_solve_calls); ADD(d14_empty);
    ADD(d14_dd_calls); ADD(d14_qf_warm_calls); ADD(d14_qf_cold_calls);
    ADD(d14_completeness_fails); ADD(d14_struct_calls); ADD(d14_horner_calls);
    ADD(d14_root_clusters); ADD(d14_root_count_bad); ADD(d14_conjugacy_bad);
    ADD(d14_vieta_bad); ADD(d14_lifted_attempts); ADD(d14_lifted_success);
    ADD(d14_lifted_certificate_fail); ADD(d14_matrix_attempts);
    ADD(d14_matrix_success); ADD(d14_presearch_sweeps); ADD(d14_dd_sweeps);
    ADD(d14_warm_seeded);
    ADD(d14_lifted_fail_seed); ADD(d14_lifted_fail_newton);
    ADD(d14_lifted_fail_scalar); ADD(d14_lifted_fail_lift);
    ADD(d14_lifted_fail_conjugacy); ADD(d14_lifted_fail_vieta);
    ADD(d14_lifted_fail_reconstruct);
    ADD(d14_hybrid_attempts); ADD(d14_hybrid_success);
    ADD(d14_hybrid_certificate_fail); ADD(d14_hybrid_cheap_calls);
    ADD(d14_hybrid_structural_calls); ADD(d14_hybrid_unsafe_calls);
    ADD(d14_qf_warm_sweeps); ADD(d14_qf_cold_sweeps);
    ADD(quartic_cold_calls); ADD(quartic_warm_calls); ADD(quartic_warm_hits);
    ADD(quartic_cold_falls); ADD(classify_calls); ADD(classified_cells);
    ADD(topology_escalations); ADD(topology_uncertain); ADD(quartic_probe_calls);
    ADD(grid512_calls); ADD(grid3072_calls); ADD(grid4096_calls);
    ADD(grid_total_nodes); ADD(physical_real_events); ADD(physical_complex_events);
    ADD(d14_soft_events); ADD(chart_p4_events); ADD(representation_events);
    ADD(radial_eq_events); ADD(l_root_events); ADD(radius_terms_calls);
    ADD(radius_value_calls); ADD(radial_nodes); ADD(arc_interval_calls);
    ADD(arc_empty); ADD(arc_full); ADD(arc_degenerate); ADD(arc_sets);
    ADD(arc_count); ADD(endpoint_calls); ADD(endpoint_unreliable);
    ADD(f0_arcs); ADD(full_circle_calls); ADD(radius_unreliable);
    ADD(rootpair_calls); ADD(rootpair_warm_success); ADD(rootpair_cold_falls);
    ADD(rootpair_disc_mismatch); ADD(rootpair_predictor_reject);
    ADD(rootpair_newton_reject); ADD(rootpair_branch_reject);
    ADD(rootpair_certify_calls); ADD(rootpair_certify_falls);
    ADD(rootpair_tmax_reject); ADD(rootpair_vfloor_reject);
    ADD(rootpair_gap_reject); ADD(rootpair_residual_reject);
    ADD(rootpair_newton_iterations); ADD(k_attempts); ADD(k_success);
    ADD(k_reject); ADD(k_reject_nonfinite); ADD(k_reject_vfloor);
    ADD(k_reject_tmax); ADD(k_reject_s2); ADD(k_reject_b);
    ADD(k_reject_disagreement); ADD(k_mid_attempts); ADD(k_mid_success);
    ADD(k_mid_reject); ADD(k_reject_arc); ADD(k_arc_endpoint);
    ADD(k_arc_nonfinite); ADD(k_arc_order); ADD(k_arc_tmax);
    ADD(k_arc_vfloor); ADD(k_reciprocal_attempts);
    ADD(k_reciprocal_success); ADD(k_reciprocal_reject);
    ADD(angular_rescue_calls);
    ADD(angular_rescue_nodes); ADD(value_angular_nodes); ADD(jacobian_nodes);
#undef ADD
#define ADDT(name) d.name += s.name
    ADDT(radial_events_ms); ADDT(d14_coeff_ms); ADDT(d14_struct_build_ms);
    ADDT(d14_expand_ms); ADDT(d14_solve_ms); ADDT(d14_presearch_ms);
    ADDT(d14_dd_ms); ADDT(d14_qf_ms); ADDT(d14_validate_ms); ADDT(topology_ms);
    ADDT(topology_probe_ms); ADDT(topology_grid_ms); ADDT(arc_ms);
    ADDT(rootpair_ms); ADDT(endpoint_ms); ADDT(f0_ms); ADDT(k_ms); ADDT(k_mid_ms);
    ADDT(k_reciprocal_ms); ADDT(angular_rescue_ms); ADDT(value_angular_ms);
    ADDT(jacobian_ms);
    ADDT(d14_hybrid_ms);
#undef ADDT
    d.d14_lifted_max_reconstruct =
        std::max(d.d14_lifted_max_reconstruct, s.d14_lifted_max_reconstruct);
}

void print_profile(const LaneSummary& s) {
    const V2Profile& p = s.sum;
    auto ratio = [](double x, double y) { return y > 0.0 ? x / y : 0.0; };
    std::fprintf(stderr,
        "  calls radial_events=%llu classify=%llu cells=%llu nodes=%llu\n",
        (unsigned long long)p.radial_event_calls,
        (unsigned long long)p.classify_calls,
        (unsigned long long)p.classified_cells,
        (unsigned long long)p.radial_nodes);
    std::fprintf(stderr,
        "  D14 calls=%llu struct=%llu horner=%llu dd=%llu qf_warm=%llu qf_cold=%llu "
        "warm_seed=%llu sweeps(pre/dd/qfw/qfc)=%llu/%llu/%llu/%llu\n",
        (unsigned long long)p.d14_solve_calls,
        (unsigned long long)p.d14_struct_calls,
        (unsigned long long)p.d14_horner_calls,
        (unsigned long long)p.d14_dd_calls,
        (unsigned long long)p.d14_qf_warm_calls,
        (unsigned long long)p.d14_qf_cold_calls,
        (unsigned long long)p.d14_warm_seeded,
        (unsigned long long)p.d14_presearch_sweeps,
        (unsigned long long)p.d14_dd_sweeps,
        (unsigned long long)p.d14_qf_warm_sweeps,
        (unsigned long long)p.d14_qf_cold_sweeps);
    std::fprintf(stderr,
        "  D14 certificate root_count/conjugacy/vieta/completeness/close_pairs="
        "%llu/%llu/%llu/%llu/%llu\n",
        (unsigned long long)p.d14_root_count_bad,
        (unsigned long long)p.d14_conjugacy_bad,
        (unsigned long long)p.d14_vieta_bad,
        (unsigned long long)p.d14_completeness_fails,
        (unsigned long long)p.d14_root_clusters);
    std::fprintf(stderr,
        "  D14 lifted attempts/success/cert-fail=%llu/%llu/%llu\n",
        (unsigned long long)p.d14_lifted_attempts,
        (unsigned long long)p.d14_lifted_success,
        (unsigned long long)p.d14_lifted_certificate_fail);
    std::fprintf(stderr,
        "  D14 lifted fail seed/newton/scalar/lift/conj/vieta/recon="
        "%llu/%llu/%llu/%llu/%llu/%llu/%llu\n",
        (unsigned long long)p.d14_lifted_fail_seed,
        (unsigned long long)p.d14_lifted_fail_newton,
        (unsigned long long)p.d14_lifted_fail_scalar,
        (unsigned long long)p.d14_lifted_fail_lift,
        (unsigned long long)p.d14_lifted_fail_conjugacy,
        (unsigned long long)p.d14_lifted_fail_vieta,
        (unsigned long long)p.d14_lifted_fail_reconstruct);
    std::fprintf(stderr, "  D14 lifted max reconstruction error=%.3e\n",
                 p.d14_lifted_max_reconstruct);
    std::fprintf(stderr,
        "  D14 hybrid attempts/success/cert-fail=%llu/%llu/%llu "
        "cheap/struct/unsafe=%llu/%llu/%llu time_ms=%g\n",
        (unsigned long long)p.d14_hybrid_attempts,
        (unsigned long long)p.d14_hybrid_success,
        (unsigned long long)p.d14_hybrid_certificate_fail,
        (unsigned long long)p.d14_hybrid_cheap_calls,
        (unsigned long long)p.d14_hybrid_structural_calls,
        (unsigned long long)p.d14_hybrid_unsafe_calls,
        p.d14_hybrid_ms);
    std::fprintf(stderr,
        "  topology probes=%llu grid512/3072/4096=%llu/%llu/%llu "
        "escalations=%llu uncertain=%llu\n",
        (unsigned long long)p.quartic_probe_calls,
        (unsigned long long)p.grid512_calls,
        (unsigned long long)p.grid3072_calls,
        (unsigned long long)p.grid4096_calls,
        (unsigned long long)p.topology_escalations,
        (unsigned long long)p.topology_uncertain);
    std::fprintf(stderr,
        "  roots quartic warm/cold/hits=%llu/%llu/%llu "
        "pair calls/warm/cold=%llu/%llu/%llu pred/newton/branch=%llu/%llu/%llu "
        "iters=%llu guards(tmax/vfloor/gap/resid)=%llu/%llu/%llu/%llu\n",
        (unsigned long long)p.quartic_warm_calls,
        (unsigned long long)p.quartic_cold_calls,
        (unsigned long long)p.quartic_warm_hits,
        (unsigned long long)p.rootpair_calls,
        (unsigned long long)p.rootpair_warm_success,
        (unsigned long long)p.rootpair_cold_falls,
        (unsigned long long)p.rootpair_predictor_reject,
        (unsigned long long)p.rootpair_newton_reject,
        (unsigned long long)p.rootpair_branch_reject,
        (unsigned long long)p.rootpair_newton_iterations,
        (unsigned long long)p.rootpair_tmax_reject,
        (unsigned long long)p.rootpair_vfloor_reject,
        (unsigned long long)p.rootpair_gap_reject,
        (unsigned long long)p.rootpair_residual_reject);
    std::fprintf(stderr,
        "  radial terms/value calls=%llu/%llu jac_nodes=%llu "
        "f0_arcs=%llu endpoint=%llu full=%llu empty=%llu degenerate=%llu\n",
        (unsigned long long)p.radius_terms_calls,
        (unsigned long long)p.radius_value_calls,
        (unsigned long long)p.jacobian_nodes,
        (unsigned long long)p.f0_arcs,
        (unsigned long long)p.endpoint_calls,
        (unsigned long long)p.full_circle_calls,
        (unsigned long long)p.arc_empty,
        (unsigned long long)p.arc_degenerate);
    std::fprintf(stderr,
        "  K attempts/success/reject=%llu/%llu/%llu mid_attempt/success/reject=%llu/%llu/%llu "
        "reason nonfinite/vfloor/tmax/S2/B/disagree=%llu/%llu/%llu/%llu/%llu/%llu "
        "arc=%llu angular rescue calls/nodes=%llu/%llu\n",
        (unsigned long long)p.k_attempts,
        (unsigned long long)p.k_success,
        (unsigned long long)p.k_reject,
        (unsigned long long)p.k_mid_attempts,
        (unsigned long long)p.k_mid_success,
        (unsigned long long)p.k_mid_reject,
        (unsigned long long)p.k_reject_nonfinite,
        (unsigned long long)p.k_reject_vfloor,
        (unsigned long long)p.k_reject_tmax,
        (unsigned long long)p.k_reject_s2,
        (unsigned long long)p.k_reject_b,
        (unsigned long long)p.k_reject_disagreement,
        (unsigned long long)p.k_reject_arc,
        (unsigned long long)p.angular_rescue_calls,
        (unsigned long long)p.angular_rescue_nodes);
    std::fprintf(stderr,
        "  K arc rejects endpoint/nonfinite/order/tmax/vfloor=%llu/%llu/%llu/%llu/%llu\n",
        (unsigned long long)p.k_arc_endpoint,
        (unsigned long long)p.k_arc_nonfinite,
        (unsigned long long)p.k_arc_order,
        (unsigned long long)p.k_arc_tmax,
        (unsigned long long)p.k_arc_vfloor);
    std::fprintf(stderr,
        "  K reciprocal attempts/success/reject=%llu/%llu/%llu time_ms=%g\n",
        (unsigned long long)p.k_reciprocal_attempts,
        (unsigned long long)p.k_reciprocal_success,
        (unsigned long long)p.k_reciprocal_reject,
        p.k_reciprocal_ms);
    static const char* reason_name[7] = {
        "nonfinite", "vfloor", "tmax", "S2", "B", "disagreement", "arc"};
    std::fprintf(stderr,
        "  K reject affected cases nonfinite/vfloor/tmax/S2/B/disagree/arc=%d/%d/%d/%d/%d/%d/%d\n",
        s.k_reason_cases[0], s.k_reason_cases[1], s.k_reason_cases[2],
        s.k_reason_cases[3], s.k_reason_cases[4], s.k_reason_cases[5],
        s.k_reason_cases[6]);
    for (int i = 0; i < 7; ++i) {
        const auto& costs = s.k_reason_rescue_ms[i];
        if (costs.empty()) continue;
        std::fprintf(stderr,
            "  K rescue ms/case %-12s p50=%.6f p90=%.6f p99=%.6f max=%.6f\n",
            reason_name[i], percentile(costs, 50), percentile(costs, 90),
            percentile(costs, 99), percentile(costs, 100));
    }
    std::fprintf(stderr,
        "  time ms radial=%g d14(coeff/struct/expand/solve)=%g/%g/%g/%g "
        "d14 inner(pre/dd/qf/validate)=%g/%g/%g/%g topology=%g "
        "grid=%g arc=%g pair=%g endpoint=%g f0=%g K=%g Kmid=%g rescue=%g\n",
        p.radial_events_ms, p.d14_coeff_ms, p.d14_struct_build_ms,
        p.d14_expand_ms, p.d14_solve_ms, p.d14_presearch_ms, p.d14_dd_ms,
        p.d14_qf_ms, p.d14_validate_ms, p.topology_ms, p.topology_grid_ms,
        p.arc_ms, p.rootpair_ms, p.endpoint_ms, p.f0_ms, p.k_ms,
        p.k_mid_ms,
        p.angular_rescue_ms);
    std::fprintf(stderr,
        "  event physical_real/complex/soft/chart_p4/rep=%llu/%llu/%llu/%llu/%llu\n",
        (unsigned long long)p.physical_real_events,
        (unsigned long long)p.physical_complex_events,
        (unsigned long long)p.d14_soft_events,
        (unsigned long long)p.chart_p4_events,
        (unsigned long long)p.representation_events);
    std::fprintf(stderr,
        "  derived K reject rate=%.3f%% rescue_ms_per_call=%.6f "
        "D14 solve ms/call=%.6f\n",
        100.0 * ratio((double)p.k_reject, (double)p.k_attempts),
        ratio(p.angular_rescue_ms, (double)p.angular_rescue_calls),
        ratio(p.d14_solve_ms, (double)p.d14_solve_calls));
}

void print_wall(const char* lane, const LaneSummary& s) {
    std::fprintf(stderr,
        "PROFILE %s cases=%zu wall_ms p50=%.6f p90=%.6f p95=%.6f "
        "p99=%.6f max=%.6f status_ok=%d status_bad=%d "
        "max_mu_ref=%.3e max_grad_ref=%.3e\n",
        lane, s.wall.size(), percentile(s.wall, 50), percentile(s.wall, 90),
        percentile(s.wall, 95), percentile(s.wall, 99), percentile(s.wall, 100),
        s.status_ok, s.status_bad, s.max_mu_ref, s.max_grad_ref);
    print_profile(s);
}

void compare(const FiniteSourceOutcome& got, const FiniteSourceOutcome& ref,
             bool jac, LaneSummary& s) {
    const double denom = std::max(std::fabs(ref.mu), 1.0);
    s.max_mu_ref = std::max(s.max_mu_ref, std::fabs(got.mu - ref.mu) / denom);
    if (jac && got.has_jacobian && ref.has_jacobian) {
        for (int j = 0; j < 5; ++j) {
            const double d = std::fabs(got.grad_mu[j] - ref.grad_mu[j]) /
                             std::max({std::fabs(ref.grad_mu[j]), 1e-6 * denom, 1e-300});
            s.max_grad_ref = std::max(s.max_grad_ref, d);
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

// Same deterministic trajectory used by the existing prepared-geometry
// benchmark.  Epoch zero is the cold build and is excluded from warm wall
// statistics; every later epoch still runs the production L2 ladder.
Case track_case(const Case& c, int k, int n) {
    const double L = std::max(8.0 * c.rho, 0.5 * c.a);
    const double s = (static_cast<double>(k) / (n - 1) - 0.5) * L;
    Case e = c;
    e.xs += s * 0.94868;
    e.ys += s * 0.31623;
    return e;
}

void run_cold_lanes(const std::vector<Case>& cases, int reps,
                    LaneSummary& value, LaneSummary& jac) {
    for (const Case& c : cases) {
        force_v2();
        const FiniteSourceOutcome ref_v = [&] {
            holo_mv_transport_override() = 0;
            holo_holonomic_transport_override() = 0;
            FiniteSourceOutcome x = execute_value_cold(
                request(c, RequestedOutput::kValue));
            force_v2();
            return x;
        }();
        const FiniteSourceOutcome ref_j = [&] {
            holo_mv_transport_override() = 0;
            holo_holonomic_transport_override() = 0;
            FiniteSourceOutcome x = finite_source_binary(request(c, RequestedOutput::kValueJacobian));
            force_v2();
            return x;
        }();
        Timed v = run_cold(request(c, RequestedOutput::kValue), reps);
        Timed j = run_cold(request(c, RequestedOutput::kValueJacobian), reps);
        value.wall.push_back(v.ms);
        jac.wall.push_back(j.ms);
        value.status_ok += v.out.status == Status::OK;
        value.status_bad += v.out.status != Status::OK;
        jac.status_ok += j.out.status == Status::OK;
        jac.status_bad += j.out.status != Status::OK;
        compare(v.out, ref_v, false, value);
        compare(j.out, ref_j, true, jac);
        add_profile(value.sum, v.profile);
        add_profile(jac.sum, j.profile);
        record_k_reason_case(value, v.profile);
        record_k_reason_case(jac, j.profile);
    }
}

void run_warm_lanes(const std::vector<Case>& cases, int reps, int n,
                    LaneSummary& value, LaneSummary& jac) {
    PreparedReuseConfig cfg = default_reuse_config();
    cfg.l2_drift = 1e18;
    const char* reuse = std::getenv("HOLO_PROFILE_REUSE_TOPOLOGY");
    cfg.allow_topology_reuse = reuse && reuse[0] == '1';
    for (const Case& c : cases) {
        PreparedEpochGeometry state_v, state_j;
        V2Profile case_profile_v{}, case_profile_j{};
        for (int k = 0; k < n; ++k) {
            const Case e = track_case(c, k, n);
            force_v2();
            const FiniteSourceOutcome ref_v = [&] {
                holo_mv_transport_override() = 0;
                holo_holonomic_transport_override() = 0;
                FiniteSourceOutcome x = execute_value_cold(
                    request(e, RequestedOutput::kValue));
                force_v2();
                return x;
            }();
            const FiniteSourceOutcome ref_j = [&] {
                holo_mv_transport_override() = 0;
                holo_holonomic_transport_override() = 0;
                FiniteSourceOutcome x = finite_source_binary(request(e, RequestedOutput::kValueJacobian));
                force_v2();
                return x;
            }();
            Timed bv, bj;
            bv.ms = bj.ms = 1e300;
            for (int r = 0; r < reps; ++r) {
                PreparedEpochGeometry sv, sj;
                for (int kk = 0; kk <= k; ++kk) {
                    const Case ee = track_case(c, kk, n);
                    V2Profile pv;
                    V2ProfileScope scope_v(pv);
                    const auto ta = clock_type::now();
                    FiniteSourceOutcome ov = execute_value_prepared(
                        request(ee, RequestedOutput::kValue), sv, cfg);
                    const auto tb = clock_type::now();
                    const double tv = std::chrono::duration<double, std::milli>(tb - ta).count();
                    if (kk == k && tv < bv.ms) {
                        bv.ms = tv; bv.out = ov; bv.profile = pv;
                    }
                    V2Profile pj;
                    V2ProfileScope scope_j(pj);
                    const auto tc = clock_type::now();
                    FiniteSourceOutcome oj = finite_source_binary_prepared(
                        request(ee, RequestedOutput::kValueJacobian), sj, cfg);
                    const auto td = clock_type::now();
                    const double tj = std::chrono::duration<double, std::milli>(td - tc).count();
                    if (kk == k && tj < bj.ms) {
                        bj.ms = tj; bj.out = oj; bj.profile = pj;
                    }
                }
            }
            // Drop epoch zero: its cold build is measured by run_cold and is
            // deliberately not part of the steady-state warm distribution.
            if (k == 0) continue;
            value.wall.push_back(bv.ms);
            jac.wall.push_back(bj.ms);
            value.status_ok += bv.out.status == Status::OK;
            value.status_bad += bv.out.status != Status::OK;
            jac.status_ok += bj.out.status == Status::OK;
            jac.status_bad += bj.out.status != Status::OK;
            compare(bv.out, ref_v, false, value);
            compare(bj.out, ref_j, true, jac);
            add_profile(value.sum, bv.profile);
            add_profile(jac.sum, bj.profile);
            add_profile(case_profile_v, bv.profile);
            add_profile(case_profile_j, bj.profile);
        }
        record_k_reason_case(value, case_profile_v);
        record_k_reason_case(jac, case_profile_j);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "evidence/holonomic/gm_coverage_cases.tsv";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    const int n = argc > 3 ? std::atoi(argv[3]) : 8;
    const std::vector<Case> cases = read_cases(path);
    if (cases.empty()) {
        std::fprintf(stderr, "cannot read cases from %s\n", path);
        return 2;
    }
    std::fprintf(stderr, "loaded %zu cases reps=%d trajectory_epochs=%d n_r=64\n",
                 cases.size(), reps, n);
    force_v2();
    LaneSummary cold_v, cold_j, warm_v, warm_j;
    run_cold_lanes(cases, reps, cold_v, cold_j);
    run_warm_lanes(cases, reps, n, warm_v, warm_j);
    print_wall("cold_value", cold_v);
    print_wall("cold_value_jac", cold_j);
    print_wall("warm_value_steady", warm_v);
    print_wall("warm_value_jac_steady", warm_j);
    restore_flags();
    return 0;
}

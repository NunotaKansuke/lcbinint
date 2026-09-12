#pragma once

// Optional production-path instrumentation for the isolated V2 research
// harness.  The default path has no profiler object installed, so the hot
// solver only takes a null pointer branch.  All counters are deliberately
// descriptive: they are evidence for a whole-epoch decision, not a quality
// gate and never affect solver control flow.

#include <chrono>
#include <cstdint>

namespace lcbinint::holonomic {

struct V2Profile {
    using u64 = std::uint64_t;

    // D14 / radial-event oracle.
    u64 radial_event_calls = 0;
    u64 d14_solve_calls = 0;
    u64 d14_empty = 0;
    u64 d14_dd_calls = 0;
    u64 d14_qf_warm_calls = 0;
    u64 d14_qf_cold_calls = 0;
    u64 d14_completeness_fails = 0;
    u64 d14_struct_calls = 0;
    u64 d14_horner_calls = 0;
    u64 d14_warm_seeded = 0;
    u64 d14_root_clusters = 0;
    u64 d14_root_count_bad = 0;
    u64 d14_conjugacy_bad = 0;
    u64 d14_vieta_bad = 0;
    u64 d14_lifted_attempts = 0;
    u64 d14_lifted_success = 0;
    u64 d14_lifted_certificate_fail = 0;
    u64 d14_lifted_fail_seed = 0;
    u64 d14_lifted_fail_newton = 0;
    u64 d14_lifted_fail_scalar = 0;
    u64 d14_lifted_fail_lift = 0;
    u64 d14_lifted_fail_conjugacy = 0;
    u64 d14_lifted_fail_vieta = 0;
    u64 d14_lifted_fail_reconstruct = 0;
    u64 d14_matrix_attempts = 0;
    u64 d14_matrix_success = 0;
    u64 d14_hybrid_attempts = 0;
    u64 d14_hybrid_success = 0;
    u64 d14_hybrid_certificate_fail = 0;
    u64 d14_hybrid_cheap_calls = 0;
    u64 d14_hybrid_structural_calls = 0;
    u64 d14_hybrid_unsafe_calls = 0;
    u64 d14_real_calls = 0;
    u64 d14_real_mixed_pairs = 0;
    u64 d14_real_dangerous_pairs = 0;
    u64 d14_real_full_recompute_rows = 0;
    u64 d14_real_local_pair_calls = 0;
    u64 d14_real_local_pair_rows = 0;
    u64 d14_real_nonconverged = 0;
    u64 d14_direct_warm_attempts = 0;
    u64 d14_direct_warm_success = 0;
    u64 d14_direct_warm_reject = 0;
    u64 d14_direct_newton_converged = 0;
    u64 d14_direct_newton_nonfinite = 0;
    u64 d14_fold_seed_attempts = 0;
    u64 d14_fold_seed_success = 0;
    u64 d14_fold_seed_fallback = 0;

    u64 d14_presearch_sweeps = 0;
    u64 d14_presearch_active_calls = 0;
    u64 d14_presearch_active_sweeps = 0;
    u64 d14_presearch_active_skips = 0;
    u64 d14_presearch_active_fallbacks = 0;
    u64 d14_dd_sweeps = 0;
    u64 d14_qf_warm_sweeps = 0;
    u64 d14_qf_cold_sweeps = 0;
    u64 quartic_cold_calls = 0;
    u64 quartic_warm_calls = 0;
    u64 quartic_warm_hits = 0;
    u64 quartic_cold_falls = 0;

    // Topology and representation events.
    u64 classify_calls = 0;
    u64 classified_cells = 0;
    u64 topology_escalations = 0;
    u64 topology_uncertain = 0;
    u64 quartic_probe_calls = 0;
    u64 sturm_calls = 0;
    u64 sturm_double_accepts = 0;
    u64 sturm_dd_accepts = 0;
    u64 sturm_qf_accepts = 0;
    u64 sturm_ambiguous = 0;
    u64 sturm_root_count_mismatch = 0;
    u64 sturm_isolation_repairs = 0;
    u64 sturm_isolation_failures = 0;
    u64 grid512_calls = 0;
    u64 grid3072_calls = 0;
    u64 grid4096_calls = 0;
    u64 grid_total_nodes = 0;
    u64 physical_real_events = 0;
    u64 physical_complex_events = 0;
    u64 d14_soft_events = 0;
    u64 chart_p4_events = 0;
    u64 representation_events = 0;
    u64 radial_eq_events = 0;
    u64 l_root_events = 0;

    // Per-cell/per-node radial geometry.
    u64 radius_terms_calls = 0;
    u64 radius_value_calls = 0;
    u64 radial_nodes = 0;
    u64 arc_interval_calls = 0;
    u64 arc_empty = 0;
    u64 arc_full = 0;
    u64 arc_degenerate = 0;
    u64 arc_reciprocal_attempts = 0;
    u64 arc_reciprocal_success = 0;
    u64 arc_reciprocal_failures = 0;
    u64 arc_sets = 0;
    u64 arc_count = 0;
    u64 endpoint_calls = 0;
    u64 endpoint_unreliable = 0;
    u64 f0_arcs = 0;
    u64 full_circle_calls = 0;
    u64 radius_unreliable = 0;

    // Root-pair continuation and its fail-closed guards.
    u64 rootpair_calls = 0;
    u64 rootpair_warm_success = 0;
    u64 rootpair_cold_falls = 0;
    u64 rootpair_disc_mismatch = 0;
    u64 rootpair_predictor_reject = 0;
    u64 rootpair_newton_reject = 0;
    u64 rootpair_branch_reject = 0;
    u64 rootpair_certify_calls = 0;
    u64 rootpair_certify_falls = 0;
    u64 rootpair_tmax_reject = 0;
    u64 rootpair_vfloor_reject = 0;
    u64 rootpair_gap_reject = 0;
    u64 rootpair_residual_reject = 0;
    u64 rootpair_newton_iterations = 0;

    // Limb-darkening evaluator and fallback reasons.
    u64 k_attempts = 0;
    u64 k_success = 0;
    u64 k_reject = 0;
    u64 k_reject_nonfinite = 0;
    u64 k_reject_vfloor = 0;
    u64 k_reject_tmax = 0;
    u64 k_reject_s2 = 0;
    u64 k_reject_b = 0;
    u64 k_reject_disagreement = 0;
    u64 k_mid_attempts = 0;
    u64 k_mid_success = 0;
    u64 k_mid_reject = 0;
    u64 k_reject_arc = 0;
    u64 k_arc_endpoint = 0;
    u64 k_arc_nonfinite = 0;
    u64 k_arc_order = 0;
    u64 k_arc_tmax = 0;
    u64 k_arc_vfloor = 0;
    u64 k_reciprocal_attempts = 0;
    u64 k_reciprocal_success = 0;
    u64 k_reciprocal_reject = 0;
    u64 angular_rescue_calls = 0;
    u64 angular_rescue_nodes = 0;
    u64 value_angular_nodes = 0;
    u64 jacobian_nodes = 0;

    // Elapsed substage time in milliseconds.  Nested fields are intentional;
    // they expose where a stage is spent and are not expected to sum exactly.
    double radial_events_ms = 0.0;
    double d14_coeff_ms = 0.0;
    double d14_struct_build_ms = 0.0;
    double d14_expand_ms = 0.0;
    double d14_solve_ms = 0.0;
    double d14_presearch_ms = 0.0;
    double d14_dd_ms = 0.0;
    double d14_qf_ms = 0.0;
    double d14_validate_ms = 0.0;
    double topology_ms = 0.0;
    double topology_probe_ms = 0.0;
    double topology_grid_ms = 0.0;
    double arc_ms = 0.0;
    double rootpair_ms = 0.0;
    double endpoint_ms = 0.0;
    double f0_ms = 0.0;
    double k_ms = 0.0;
    double k_mid_ms = 0.0;
    double k_reciprocal_ms = 0.0;
    double angular_rescue_ms = 0.0;
    double value_angular_ms = 0.0;
    double jacobian_ms = 0.0;
    double d14_hybrid_ms = 0.0;
    double d14_real_ms = 0.0;
    double d14_warm_screen_ms = 0.0;
    double d14_fold_seed_ms = 0.0;
    double d14_lifted_max_reconstruct = 0.0;

    void reset() { *this = V2Profile{}; }
};

inline V2Profile*& v2_profile_slot() {
    static thread_local V2Profile* p = nullptr;
    return p;
}

inline V2Profile* v2_profile_current() { return v2_profile_slot(); }

struct V2ProfileScope {
    V2Profile* previous;
    explicit V2ProfileScope(V2Profile& p) : previous(v2_profile_slot()) {
        v2_profile_slot() = &p;
    }
    ~V2ProfileScope() { v2_profile_slot() = previous; }
};

using V2Clock = std::chrono::steady_clock;

inline void v2_profile_add_ms(double V2Profile::*field,
                              V2Clock::time_point begin,
                              V2Clock::time_point end) {
    if (V2Profile* p = v2_profile_current())
        p->*field += std::chrono::duration<double, std::milli>(end - begin).count();
}

struct V2ProfileTimer {
    V2Profile* p;
    double V2Profile::*field;
    V2Clock::time_point begin;
    V2ProfileTimer(double V2Profile::*f)
        : p(v2_profile_current()), field(f), begin(p ? V2Clock::now() : V2Clock::time_point{}) {}
    ~V2ProfileTimer() {
        if (p) p->*field += std::chrono::duration<double, std::milli>(V2Clock::now() - begin).count();
    }
};

}  // namespace lcbinint::holonomic

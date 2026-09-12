#pragma once

// Optional production-path instrumentation for the isolated V2 research
// harness.  The default path has no profiler object installed, so the hot
// solver only takes a null pointer branch.  All counters are deliberately
// descriptive: they are evidence for a whole-epoch decision, not a quality
// gate and never affect solver control flow.

#include <chrono>
#include <cstdint>
#include <vector>

namespace lcbinint::holonomic {

enum class D14RootUse : std::uint8_t {
    OtherWarmCompleteness = 0,
    PositiveRealCandidate = 1,
    ComplexSoftCut = 2
};

// Opt-in per-root evidence record.  Production profiling does not allocate
// these rows unless capture_d14_root_work is explicitly enabled by a research
// harness.
struct D14RootWorkRecord {
    int root_index = -1;
    int source_root_index = -1;
    int role = static_cast<int>(D14RootUse::OtherWarmCompleteness);
    int physical_real = -1;  // -1 not classified, 0 no, 1 yes
    int d14real_updates = 0;
    int cluster_id = -1;
    int cluster_size = 1;
    int freeze_count = 0;
    int reactivation_count = 0;
    int role_hint = static_cast<int>(D14RootUse::OtherWarmCompleteness);
    double v_re = 0.0;
    double v_im = 0.0;
    int double_seed_index = -1;
    double double_seed_v_re = 0.0;
    double double_seed_v_im = 0.0;
    bool double_seed_valid = false;
    int expanded_seed_source_index = -1;
    double expanded_seed_displacement = 0.0;
    bool expanded_seed_match_valid = false;
    double final_v_re = 0.0;
    double final_v_im = 0.0;
    double d14real_newton_correction = 0.0;
    double d14real_nearest_separation = 0.0;
    double d14real_relative_correction = 0.0;
    double d14real_position_error = 0.0;
    double qf_displacement = 0.0;
    double qf_newton_correction = 0.0;
    double qf_nearest_separation = 0.0;
    double qf_position_error = 0.0;
    double qf_expanded_relative_residual = 0.0;
    bool qf_escalated = false;
    bool qf_displacement_valid = false;
    bool qf_source_match_valid = false;
};

struct V2Profile {
    using u64 = std::uint64_t;

    // D14 / radial-event oracle.
    u64 radial_event_calls = 0;
    u64 d14_solve_calls = 0;
    u64 d14_empty = 0;
    u64 d14_dd_calls = 0;
    u64 d14_qf_warm_calls = 0;
    u64 d14_qf_cold_calls = 0;
    u64 d14_event_contract_attempts = 0;
    u64 d14_event_contract_accepts = 0;
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
    u64 d14_real_finite_calls = 0;
    u64 d14_real_mixed_pairs = 0;
    u64 d14_real_dangerous_pairs = 0;
    u64 d14_real_full_recompute_rows = 0;
    u64 d14_real_local_pair_calls = 0;
    u64 d14_real_local_pair_rows = 0;
    u64 d14_real_root_updates = 0;
    u64 d14_real_root_skips = 0;
    u64 d14_real_root_freezes = 0;
    u64 d14_real_root_reactivations = 0;
    u64 d14_real_cluster_wakeups = 0;
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
    u64 d14_noise_checks = 0, d14_noise_all = 0, d14_noise_valid = 0;
    int d14_noise_first_sweep = 0;
    double d14_noise_max_ratio = 0;
    u64 d14_presearch_active_sweeps = 0;
    u64 d14_presearch_active_skips = 0;
    u64 d14_presearch_active_fallbacks = 0;
    u64 d14_presearch_nonfinite_failures = 0;
    u64 d14_presearch_last_finite_handoffs = 0;
    int d14_presearch_failure_code = 0;
    int d14_presearch_failure_iteration = 0;
    int d14_presearch_failure_root = -1;
    u64 d14_dd_sweeps = 0;
    u64 d14_qf_warm_sweeps = 0;
    u64 d14_qf_cold_sweeps = 0;
    u64 native_residual_calls=0,native_residual_pass=0,native_residual_violations=0;
    u64 native_residual_root_pass=0,native_residual_root_violations=0;
    double native_residual_ms=0;
    u64 local_bracket_attempts = 0;
    u64 local_bracket_successes = 0;
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
    double d14_prepare_ms = 0.0, d14_conjugate_ms = 0.0;
    double d14_metadata_ms = 0.0, chart_event_ms = 0.0;
    double d14_diagnostic_ms = 0.0;
    double d14_presearch_ms = 0.0;
    double d14_dd_ms = 0.0;
    double d14_qf_ms = 0.0;
    double d14_qf_polish_ms = 0.0;
    double d14_validate_ms = 0.0;
    double d14_residual_eval_ms = 0.0;
    double d14_completeness_check_ms = 0.0;
    double d14_event_classify_ms = 0.0;
    double d14_soft_event_ms = 0.0;
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

    bool capture_d14_root_work = false;
    std::vector<D14RootWorkRecord> d14_root_work;

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

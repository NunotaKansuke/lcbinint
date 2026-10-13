#pragma once

// Phase 7 research harness: compare the incumbent V2 16-node K-rule with
// the equation-derived PF6/fold LD evaluator on one shared V2 geometry plan.
//
// The plan owns the exact classify_cells/D14 result, cell traversal, radial
// nodes, boundary-quartic continuation, endpoint polishing, and F0
// accumulation.  Evaluators consume that plan and may not rediscover arcs or
// endpoints.  This header is isolated: no production router includes it.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/gm_lauricella6.hpp"

namespace lcbinint::holonomic {
namespace phase7_detail {

using Clock = std::chrono::steady_clock;
using GmArcSeed = gm_l6_detail::ArcSeed;
using GmNodeResult = gm_l6_detail::NodeResult;

struct SharedArcNode {
    std::array<double, 2> arc{};  // the V2 angular endpoints
    std::array<double, 2> polished_arc{};  // the shared endpoint-polish result
    double t_lo = 0.0;
    double t_hi = 0.0;
    double m = 0.0;
    double v = 0.0;
    std::array<double, 5> dm{};
    std::array<double, 5> dv{};
    bool endpoint_ok = false;
    bool arc_clean = false;
    bool pair_ok = false;
};

struct SharedNode {
    double R = 0.0;
    double weight = 0.0;
    ArcKind kind = ArcKind::kEmpty;
    std::array<double, 5> pc{};
    std::array<std::array<double, 5>, 5> dpc{};
    std::vector<SharedArcNode> arcs;
};

struct SharedArcAnchor {
    std::array<double, 2> arc{};
    GmArcSeed seed{};
    bool seed_ok = false;
};

struct SharedCell {
    CellPlan cell{};
    double lo = 0.0;
    double hi = 0.0;
    double center = 0.0;
    double half = 0.0;
    std::vector<SharedArcAnchor> anchors;
    std::vector<SharedNode> nodes;
};

struct SharedPlanCost {
    double topology_ms = 0.0;
    double root_tracking_ms = 0.0;
    double f0_endpoint_ms = 0.0;
    double total_ms = 0.0;
};

struct SharedEpochPlan {
    LensParams params{};
    PrimaryFrame pf{};
    double u = 0.0;
    int n_r = 0;
    TopologyResult topology{};
    std::vector<Cplx<__float128>> d14_roots;
    std::vector<SharedCell> cells;
    double F0 = 0.0;
    std::array<double, 5> dF0_internal{};
    Status status = Status::OK;
    bool with_jacobian = false;
    SharedPlanCost cost{};
    int radial_nodes = 0;
    int arc_nodes = 0;
};

inline double arc_measure(const std::array<double, 2>& arc) {
    double hi = arc[1];
    if (hi <= arc[0]) hi += gm_l6_detail::kTwoPi;
    return hi - arc[0];
}

inline double arc_midpoint(const std::array<double, 2>& arc) {
    double hi = arc[1];
    if (hi <= arc[0]) hi += gm_l6_detail::kTwoPi;
    return 0.5 * (arc[0] + hi);
}

inline double arc_angle_distance(double a, double b) {
    return std::fabs(std::atan2(std::sin(a - b), std::cos(a - b)));
}

inline int nearest_arc(const std::vector<SharedArcNode>& arcs,
                       const std::array<double, 2>& target) {
    if (arcs.empty()) return -1;
    const double target_mid = arc_midpoint(target);
    int best = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < (int)arcs.size(); ++i) {
        const double d = arc_angle_distance(arc_midpoint(arcs[i].arc),
                                            target_mid);
        if (d < best_distance) {
            best_distance = d;
            best = i;
        }
    }
    return best;
}

inline int nearest_arc(const ArcSet& arcs, const std::array<double, 2>& target) {
    if (arcs.arcs.empty()) return -1;
    const double target_mid = arc_midpoint(target);
    int best = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < (int)arcs.arcs.size(); ++i) {
        const double d = arc_angle_distance(arc_midpoint(arcs.arcs[i]),
                                            target_mid);
        if (d < best_distance) {
            best_distance = d;
            best = i;
        }
    }
    return best;
}

inline void set_status(Status& status, Status candidate) {
    if (status == Status::OK) status = candidate;
}

inline SharedArcNode make_shared_arc_node(
    double R, const PrimaryFrame& pf, const std::array<double, 2>& arc,
    double tan_thresh, bool with_jacobian,
    std::array<double, 5>* f0_deriv, double& f0) {
    SharedArcNode out;
    out.arc = arc;
    const PolishResult pe = polish_endpoint(R, arc[0], pf);
    const PolishResult pl = polish_endpoint(R, arc[1], pf);
    double te = pe.theta;
    double tl = pl.theta;
    if (tl <= te) tl += gm_l6_detail::kTwoPi;
    out.polished_arc = {te, tl};
    f0 = R * (tl - te);
    out.endpoint_ok = pe.reliable && pl.reliable;
    out.arc_clean = out.endpoint_ok &&
                    std::fabs(pe.dphi_dtheta) >= tan_thresh &&
                    std::fabs(pl.dphi_dtheta) >= tan_thresh;

    std::array<double, 5> dte{}, dtl{};
    if (with_jacobian) {
        const PhiValDP ge = phi_val_dP(R, te, pf);
        const PhiValDP gl = phi_val_dP(R, tl, pf);
        for (int j = 0; j < 5; ++j) {
            dte[j] = pe.dphi_dtheta != 0.0
                ? -ge.dP[j] / pe.dphi_dtheta : 0.0;
            dtl[j] = pl.dphi_dtheta != 0.0
                ? -gl.dP[j] / pl.dphi_dtheta : 0.0;
            (*f0_deriv)[j] = R * (dtl[j] - dte[j]);
        }
    }
    if (!with_jacobian && f0_deriv) f0_deriv->fill(0.0);

    const ArcPairJac pair = arc_pair_jac(te, tl, dte, dtl);
    out.t_lo = std::tan(0.5 * te);
    out.t_hi = std::tan(0.5 * tl);
    out.m = pair.m;
    out.v = pair.v;
    out.dm = pair.dm;
    out.dv = pair.dv;
    out.pair_ok = pair.ok && out.arc_clean;
    return out;
}

// Build the sole Phase 7 geometry plan.  The radial pass is the same
// boundary-quartic continuation used by radius_terms/epoch_jacobian.  The
// extra center arc lookup is made here, once, so PF6 never performs its own
// endpoint or arc search.
inline SharedEpochPlan build_shared_plan(
    const LensParams& p, double u, int n_r, bool with_jacobian,
    const std::vector<Cplx<__float128>>* d14_warm = nullptr) {
    SharedEpochPlan out;
    out.params = p;
    out.pf = PrimaryFrame::from(p);
    out.u = u;
    out.n_r = n_r;
    out.with_jacobian = with_jacobian;
    const auto started = Clock::now();

    const auto topology_start = Clock::now();
    out.topology = classify_cells(out.pf, d14_warm, &out.d14_roots);
    out.topology.from_warm_d14 = d14_warm && !d14_warm->empty();
    out.cost.topology_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - topology_start).count();
    out.status = out.topology.status;

    if (n_r < 4 || !(p.rho > 0.0) || !(p.a > 0.0) || !(p.q >= 0.0)) {
        out.status = Status::BASIS_DEGENERATE;
        out.cost.total_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - started).count();
        return out;
    }

    const Cheb1Dyn rr(n_r);
    for (const auto& c : out.topology.cells) {
        SharedCell sc;
        sc.cell = c;
        const double width = c.r_hi - c.r_lo;
        if (!(width > 0.0)) {
            set_status(out.status, Status::EVENT_UNRESOLVED);
            out.cells.push_back(std::move(sc));
            continue;
        }
        const double inset = 1e-9 * width;
        sc.lo = c.r_lo + inset;
        sc.hi = c.r_hi - inset;
        sc.center = 0.5 * (sc.lo + sc.hi);
        sc.half = 0.5 * (sc.hi - sc.lo);
        if (!(sc.half > 0.0)) {
            set_status(out.status, Status::EVENT_UNRESOLVED);
            out.cells.push_back(std::move(sc));
            continue;
        }
        if (c.kind == ArcKind::kEmpty) {
            out.cells.push_back(std::move(sc));
            continue;
        }

        if (c.kind == ArcKind::kFull) {
            for (int k = 0; k < n_r; ++k) {
                SharedNode node;
                node.R = sc.center + sc.half * rr.x[k];
                node.weight = sc.half * rr.w[k];
                node.kind = ArcKind::kFull;
                node.pc = boundary_quartic(node.R, out.pf).p;
                if (with_jacobian)
                    node.dpc = boundary_quartic_dp(node.R, out.pf).dp;
                out.F0 += node.weight * node.R * gm_l6_detail::kTwoPi;
                sc.nodes.push_back(std::move(node));
            }
            set_status(out.status, Status::GRADIENT_UNRELIABLE);
            out.cells.push_back(std::move(sc));
            continue;
        }
        if (c.kind != ArcKind::kArcs || c.status != Status::OK) {
            set_status(out.status, Status::TOPOLOGY_UNCERTAIN);
            out.cells.push_back(std::move(sc));
            continue;
        }

        const auto center_start = Clock::now();
        const ArcSet center_arcs = arc_intervals(sc.center, out.pf);
        out.cost.root_tracking_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - center_start).count();
        if (center_arcs.kind != ArcKind::kArcs || center_arcs.arcs.empty()) {
            set_status(out.status, Status::ARC_TOPOLOGY_INVALID);
            out.cells.push_back(std::move(sc));
            continue;
        }
        for (const auto& arc : center_arcs.arcs) {
            SharedArcAnchor anchor;
            anchor.arc = arc;
            anchor.seed = gm_l6_detail::prepare_arc(out.pf, arc);
            anchor.seed_ok = anchor.seed.ok &&
                gm_l6_detail::certify_arc(sc.center, out.pf, arc, anchor.seed);
            if (!anchor.seed_ok) set_status(out.status, Status::BASIS_DEGENERATE);
            sc.anchors.push_back(std::move(anchor));
        }

        QuarticWarm qw;
        RootPairWarm rpw;
        rpw.certify = out.topology.from_warm_d14;
        const double tan_rel = kTanRel;
        for (int k = 0; k < n_r; ++k) {
            SharedNode node;
            node.R = sc.center + sc.half * rr.x[k];
            node.weight = sc.half * rr.w[k];
            ++out.radial_nodes;
            const auto root_start = Clock::now();
            const ArcSet as = arc_intervals(node.R, out.pf, &qw, &rpw);
            out.cost.root_tracking_ms +=
                std::chrono::duration<double, std::milli>(
                    Clock::now() - root_start).count();
            node.kind = as.kind;
            if (as.kind == ArcKind::kDegenerate) {
                set_status(out.status, Status::GRADIENT_UNRELIABLE);
                const ArcSet grid = grid_intervals(node.R, out.pf);
                node.kind = grid.kind;
                if (grid.kind == ArcKind::kArcs)
                    for (const auto& arc : grid.arcs) {
                        std::array<double, 5> df0{};
                        double f0 = 0.0;
                        node.arcs.push_back(make_shared_arc_node(
                            node.R, out.pf, arc,
                            tan_rel * out.pf.rho / std::max(node.R, 1e-9),
                            with_jacobian, &df0, f0));
                        out.F0 += node.weight * f0;
                        for (int j = 0; j < 5; ++j)
                            out.dF0_internal[j] += node.weight * df0[j];
                    }
            } else if (as.kind == ArcKind::kArcs) {
                node.pc = boundary_quartic(node.R, out.pf).p;
                if (with_jacobian)
                    node.dpc = boundary_quartic_dp(node.R, out.pf).dp;
                for (const auto& arc : as.arcs) {
                    const auto endpoint_start = Clock::now();
                    std::array<double, 5> df0{};
                    double f0 = 0.0;
                    SharedArcNode an = make_shared_arc_node(
                        node.R, out.pf, arc,
                        tan_rel * out.pf.rho / std::max(node.R, 1e-9),
                        with_jacobian, &df0, f0);
                    out.cost.f0_endpoint_ms +=
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - endpoint_start).count();
                    out.F0 += node.weight * f0;
                    for (int j = 0; j < 5; ++j)
                        out.dF0_internal[j] += node.weight * df0[j];
                    node.arcs.push_back(std::move(an));
                    ++out.arc_nodes;
                }
                if (node.arcs.empty()) set_status(out.status,
                                                   Status::ARC_TOPOLOGY_INVALID);
            } else if (as.kind == ArcKind::kFull) {
                node.pc = boundary_quartic(node.R, out.pf).p;
                if (with_jacobian)
                    node.dpc = boundary_quartic_dp(node.R, out.pf).dp;
                out.F0 += node.weight * node.R * gm_l6_detail::kTwoPi;
            } else if (as.kind != ArcKind::kEmpty) {
                set_status(out.status, Status::ARC_TOPOLOGY_INVALID);
            }
            if (node.kind == ArcKind::kArcs) {
                for (const auto& an : node.arcs) {
                    if (!an.endpoint_ok || !an.arc_clean)
                        set_status(out.status, Status::GRADIENT_UNRELIABLE);
                }
            }
            sc.nodes.push_back(std::move(node));
        }
        out.cells.push_back(std::move(sc));
    }
    out.cost.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    return out;
}

struct SharedEvalCost {
    double evaluation_ms = 0.0;
    double ld_ms = 0.0;
    double fold_build_ms = 0.0;
    double fold_evaluation_ms = 0.0;
    double pf6_chart_geometry_ms = 0.0;
    double physical_seed_ms = 0.0;
    double pf6_packet_ms = 0.0;
    double pf6_evaluation_ms = 0.0;
    double analytic_jacobian_ms = 0.0;
    int radial_nodes = 0;
    int arc_nodes = 0;
    int transported = 0;
    int failed = 0;
    int k_rule_success = 0;
    int k_rule_failed = 0;
    int incumbent_angular_fallback = 0;
    int fold_packet_constructions = 0;
    int fold_packet_accepted = 0;
    int fold_packet_rejected = 0;
    int fold_nodes = 0;
    int pf6_packet_constructions = 0;
    int pf6_packet_accepted = 0;
    int pf6_packet_rejected = 0;
    int pf6_blocks = 0;
};

struct SharedEpochEvaluation {
    double F0 = 0.0;
    double F_half = 0.0;
    double mu = 0.0;
    std::array<double, 5> dF0_internal{};
    std::array<double, 5> dF_half_internal{};
    std::array<double, 5> grad_mu{};
    Status status = Status::OK;
    bool all_true = false;
    SharedEvalCost cost{};
};

inline void finalize_evaluation(const SharedEpochPlan& plan,
                                SharedEpochEvaluation& out,
                                bool with_jacobian) {
    out.F0 = plan.F0;
    out.dF0_internal = plan.dF0_internal;
    const Status evaluation_status = out.status;
    out.status = plan.status;
    if (evaluation_status != Status::OK)
        set_status(out.status, evaluation_status);
    if (near_origin_source(plan.pf)) set_status(out.status,
                                                 Status::GRADIENT_UNRELIABLE);
    const double D = gm_l6_detail::kPi * plan.params.rho * plan.params.rho *
                     (1.0 - plan.u / 3.0);
    if (!(D > 0.0) || !std::isfinite(D)) {
        out.status = Status::BASIS_DEGENERATE;
        return;
    }
    out.mu = ((1.0 - plan.u) * out.F0 + plan.u * out.F_half) / D;
    if (with_jacobian) {
        const std::array<double, 5> d = internal_to_user_jac(
            out.dF0_internal, plan.params);
        const std::array<double, 5> dh = internal_to_user_jac(
            out.dF_half_internal, plan.params);
        for (int j = 0; j < 5; ++j)
            out.grad_mu[j] = ((1.0 - plan.u) * d[j] + plan.u * dh[j]) / D;
        out.grad_mu[2] -= 2.0 * out.mu / plan.params.rho;
    }
}

// Keep the incumbent angular rescue available when the V2 K-rule certificate
// rejects an arc.  It uses the already polished shared endpoints and never
// performs a second topology or endpoint search.
inline bool incumbent_angular_arc(
    double R, const PrimaryFrame& pf, const SharedArcNode& arc,
    bool with_jacobian, double& value, std::array<double, 5>& deriv) {
    const auto& rule = ang_rule();
    const double half = 0.5 * arc_measure(arc.arc);
    const double mid = arc_midpoint(arc.arc);
    double value_sum = 0.0;
    deriv.fill(0.0);
    for (int k = 0; k < 64; ++k) {
        const double th = mid + half * rule.x[k];
        if (!with_jacobian) {
            const double ph = phi_val(R, th, pf);
            if (!(ph > 0.0)) return false;
            value_sum += rule.w[k] * std::sqrt(ph);
        } else {
            const PhiValDP g = phi_val_dP(R, th, pf);
            if (!(g.phi > 0.0)) return false;
            const double sq = std::sqrt(g.phi);
            value_sum += rule.w[k] * sq;
            const double inv = rule.w[k] / (2.0 * sq);
            for (int j = 0; j < 5; ++j) deriv[j] += inv * g.dP[j];
        }
    }
    value = R * half * value_sum;
    for (int j = 0; j < 5; ++j) deriv[j] *= R * half;
    return std::isfinite(value);
}

inline void add_full_circle(const SharedNode& node, const PrimaryFrame& pf,
                            bool with_jacobian, double& value,
                            std::array<double, 5>& deriv) {
    const RadiusTerms rt = full_circle_terms(node.R, pf);
    value += node.weight * rt.fh;
    if (with_jacobian)
        for (int j = 0; j < 5; ++j) deriv[j] += node.weight * rt.dfh[j];
}

inline SharedEpochEvaluation evaluate_v2_k_rule(
    const SharedEpochPlan& plan, bool with_jacobian) {
    SharedEpochEvaluation out;
    const auto started = Clock::now();
    out.dF0_internal = plan.dF0_internal;
    out.cost.radial_nodes = 0;
    if (plan.u == 0.0) {
        finalize_evaluation(plan, out, with_jacobian);
        out.cost.ld_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - started).count();
        out.cost.evaluation_ms = out.cost.ld_ms;
        out.all_true = plan.status == Status::OK;
        return out;
    }
    for (const auto& cell : plan.cells) {
        for (const auto& node : cell.nodes) {
            ++out.cost.radial_nodes;
            if (node.kind == ArcKind::kEmpty) continue;
            if (node.kind == ArcKind::kFull) {
                add_full_circle(node, plan.pf, with_jacobian, out.F_half,
                                out.dF_half_internal);
                ++out.cost.incumbent_angular_fallback;
                continue;
            }
            if (node.kind != ArcKind::kArcs) {
                ++out.cost.failed;
                continue;
            }
            for (const auto& arc : node.arcs) {
                ++out.cost.arc_nodes;
                double value = 0.0;
                std::array<double, 5> deriv{};
                bool ok = false;
                if (arc.pair_ok) {
                    if (!with_jacobian) {
                        const auto v = v_times_K(arc.m, arc.v, node.R,
                                                 plan.pf, node.pc);
                        ok = v.ok;
                        if (ok) value = v.vK;
                    } else {
                        const auto v = v_times_K_jac(
                            arc.m, arc.v, arc.dm, arc.dv, node.R, plan.pf,
                            node.pc, node.dpc);
                        ok = v.ok;
                        if (ok) deriv = v.dvK;
                        if (ok) value = v.vK;
                    }
                }
                if (ok) {
                    ++out.cost.k_rule_success;
                } else {
                    ++out.cost.k_rule_failed;
                    if (!incumbent_angular_arc(node.R, plan.pf, arc,
                                               with_jacobian, value, deriv)) {
                        ++out.cost.failed;
                        continue;
                    }
                    ++out.cost.incumbent_angular_fallback;
                }
                out.F_half += node.weight * value;
                if (with_jacobian)
                    for (int j = 0; j < 5; ++j)
                        out.dF_half_internal[j] += node.weight * deriv[j];
                ++out.cost.transported;
            }
        }
    }
    finalize_evaluation(plan, out, with_jacobian);
    out.cost.ld_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    out.cost.evaluation_ms = out.cost.ld_ms;
    out.all_true = plan.status == Status::OK && out.cost.failed == 0 &&
                   out.cost.k_rule_failed == 0 &&
                   out.cost.incumbent_angular_fallback == 0;
    return out;
}

template <int Order>
inline GmNodeResult evaluate_shared_node(
    const gm_l6_detail::TaylorPacket<Order>& packet,
    const SharedArcNode& shared, double R, const PrimaryFrame& pf,
    bool chart2, bool with_jacobian,
    double* chart_geometry_ms = nullptr) {
    using namespace gm_l6_detail;
    const auto t0 = Clock::now();
    GmNodeResult out;
    const double h = R - packet.center;
    const ValueState state = eval_state(packet, h);
    const auto expected_xi = eval_xi(packet, h);
    const double l = eval_root(packet.t_lo, h);
    const double r = eval_root(packet.t_hi, h);
    const ArcSeed shared_seed = prepare_arc_chart(
        pf, shared.polished_arc, chart2);
    if (!shared_seed.ok || !(r > l)) return out;
    // The packet roots are the equation-derived continuation of the center
    // root pair supplied by the shared plan.  Check that continuation against
    // the already polished V2 pair; no endpoint/root search is performed.
    const double root_error = std::max(
        std::fabs(l - shared_seed.t_lo) /
            (1.0 + std::fabs(shared_seed.t_lo)),
        std::fabs(r - shared_seed.t_hi) /
            (1.0 + std::fabs(shared_seed.t_hi)));
    if (!std::isfinite(root_error) || root_error > 2e-4) return out;
    const Params<double> pv = params_for_chart<double>(pf, chart2);
    const Geometry<double> gv = geometry_scalar(R, l, r, pv);
    if (!gv.ok) return out;
    const Complex<double> value_c = gv.C * state[0];
    out.value = value_c.re;
    out.phase_error = std::fabs(value_c.im) /
                      (1.0 + std::fabs(value_c.re));
    if (!std::isfinite(out.value) || !std::isfinite(out.phase_error) ||
        out.phase_error > 2e-7)
        return out;
    out.value_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t0).count();
    if (chart_geometry_ms)
        *chart_geometry_ms += out.value_ms;
    if (!with_jacobian) {
        out.ok = true;
        return out;
    }

    const auto tj = Clock::now();
    const Dual5 ld = endpoint_dual(R, l, pf, chart2);
    const Dual5 rd = endpoint_dual(R, r, pf, chart2);
    if (!scalar_finite(ld) || !scalar_finite(rd)) {
        out.jacobian_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - tj).count();
        return out;
    }
    const Params<Dual5> pd = dual_params_for_chart(pf, chart2);
    Geometry<Dual5> gd = geometry_scalar(Dual5(R), ld, rd, pd);
    if (!gd.ok) {
        out.jacobian_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - tj).count();
        return out;
    }
    align_s_roots(gd, expected_xi);
    const Complex<double> F = state[0];
    for (int i = 0; i < 5; ++i) {
        const Complex<double> Mi = state[i + 1];
        const Complex<double> dF_dxi = cscale(
            (F - Mi) / Complex<double>{1.0 - expected_xi[i].re,
                                       -expected_xi[i].im},
            kBeta[i]);
        for (int j = 0; j < 5; ++j) {
            const Complex<double> dxi{gd.xi[i].re.deriv[j],
                                      gd.xi[i].im.deriv[j]};
            out.deriv[j] += (Complex<double>{gd.C.re.value, gd.C.im.value} *
                             dF_dxi * dxi).re;
        }
    }
    for (int j = 0; j < 5; ++j) {
        const Complex<double> dC{gd.C.re.deriv[j], gd.C.im.deriv[j]};
        out.deriv[j] += (dC * F).re;
    }
    out.jacobian_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - tj).count();
    if (chart_geometry_ms) *chart_geometry_ms += out.jacobian_ms;
    for (double x : out.deriv)
        if (!std::isfinite(x)) return GmNodeResult{};
    out.ok = true;
    return out;
}

template <int Order>
inline SharedEpochEvaluation evaluate_pf6_fold(
    const SharedEpochPlan& plan, bool with_jacobian) {
    using namespace gm_l6_detail;
    SharedEpochEvaluation out;
    const auto started = Clock::now();
    out.dF0_internal = plan.dF0_internal;
    out.cost.radial_nodes = 0;
    bool all = plan.status == Status::OK;

    if (plan.u == 0.0) {
        finalize_evaluation(plan, out, with_jacobian);
        out.cost.ld_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - started).count();
        out.cost.evaluation_ms = out.cost.ld_ms;
        out.all_true = plan.status == Status::OK;
        return out;
    }

    for (const auto& cell : plan.cells) {
        if (cell.cell.kind == ArcKind::kEmpty) continue;
        out.cost.radial_nodes += (int)cell.nodes.size();
        if (cell.cell.kind == ArcKind::kFull) {
            all = false;
            set_status(out.status, Status::GRADIENT_UNRELIABLE);
            for (const auto& node : cell.nodes) {
                add_full_circle(node, plan.pf, with_jacobian, out.F_half,
                                out.dF_half_internal);
                ++out.cost.incumbent_angular_fallback;
            }
            continue;
        }
        for (const auto& anchor : cell.anchors) {
            std::vector<int> node_indices;
            node_indices.reserve(cell.nodes.size());
            for (const auto& node : cell.nodes) {
                node_indices.push_back(nearest_arc(node.arcs, anchor.arc));
            }
            if (!anchor.seed_ok) {
                all = false;
                ++out.cost.failed;
                set_status(out.status, Status::BASIS_DEGENERATE);
                continue;
            }

            bool fold_have = false;
            FoldPacket<Order> fold_packet;
            if (gm_l6_detail::cell_touches_physical_fold(
                    cell.cell, plan.topology.events) &&
                gm_l6_detail::fold_packet_enabled()) {
                ++out.cost.fold_packet_constructions;
                const auto fold_start = Clock::now();
                const Params<double> params = params_for_chart<double>(
                    anchor.seed.chart_pf, false);
                FoldPacketBuildBreakdown timing;
                fold_have = build_fold_packet(
                    cell.center, cell.lo, cell.hi, anchor.seed.t_lo,
                    anchor.seed.t_hi, params, plan.pf, anchor.seed.chart2,
                    with_jacobian, fold_packet, &timing);
                out.cost.fold_build_ms += timing.total_ms;
                if (fold_have) ++out.cost.fold_packet_accepted;
                else ++out.cost.fold_packet_rejected;
                (void)fold_start;
            }

            std::vector<Block<Order>> blocks;
            PacketProfile profile;
            bool packet_have = false;
            Params<double> params{};
            if (!fold_have) {
                const auto seed_start = Clock::now();
                params = params_for_chart<double>(anchor.seed.chart_pf, false);
                const Geometry<double> geometry = geometry_scalar(
                    cell.center, anchor.seed.t_lo, anchor.seed.t_hi, params);
                SeedResult seed;
                if (geometry.ok)
                    seed = physical_seed(geometry, 512);
                out.cost.physical_seed_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - seed_start).count();
                if (!geometry.ok || !seed.ok) {
                    ++out.cost.failed;
                    all = false;
                    set_status(out.status, Status::BASIS_DEGENERATE);
                    continue;
                }
                double min_node = std::numeric_limits<double>::infinity();
                double max_node = -std::numeric_limits<double>::infinity();
                std::vector<double> node_radii;
                node_radii.reserve(cell.nodes.size());
                for (const auto& node : cell.nodes) {
                    min_node = std::min(min_node, node.R);
                    max_node = std::max(max_node, node.R);
                    node_radii.push_back(node.R);
                }
                TaylorPacket<Order> initial;
                const auto packet_start = Clock::now();
                const bool initial_ok = build_packet(
                    cell.center, min_node, max_node, anchor.seed.t_lo,
                    anchor.seed.t_hi, params, seed.state.z, initial, nullptr);
                out.cost.pf6_packet_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - packet_start).count();
                ++out.cost.pf6_packet_constructions;
                profile.timing_enabled = false;
                if (!initial_ok) {
                    ++out.cost.pf6_packet_rejected;
                    ++out.cost.failed;
                    all = false;
                    set_status(out.status, Status::CONNECTION_ILL_CONDITIONED);
                    continue;
                }
                const bool initial_accept = initial.tail < 2e-11;
                record_packet_build(&profile, nullptr, nullptr, true,
                                    initial_accept, !initial_accept, 0);
                if (initial_accept) {
                    blocks.push_back({cell.lo, cell.hi, std::move(initial)});
                    packet_have = true;
                } else {
                    int constructions = 1;
                    const auto cover_start = Clock::now();
                    const bool covered = cover_interval(
                        initial, min_node, max_node, params, 0, blocks,
                        constructions, &profile, nullptr, true, nullptr,
                        &node_radii);
                    out.cost.pf6_packet_ms +=
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - cover_start).count();
                    out.cost.pf6_packet_constructions += constructions - 1;
                    packet_have = covered;
                    if (!covered) {
                        ++out.cost.failed;
                        all = false;
                        set_status(out.status,
                                   Status::TRANSPORT_TOLERANCE_FAILED);
                    }
                }
                out.cost.pf6_packet_accepted += profile.accepted;
                out.cost.pf6_packet_rejected += profile.rejected;
                out.cost.pf6_blocks += (int)blocks.size();
            }

            for (size_t ni = 0; ni < cell.nodes.size(); ++ni) {
                const SharedNode& node = cell.nodes[ni];
                const int ai = node_indices[ni];
                ++out.cost.arc_nodes;
                if (ai < 0) {
                    ++out.cost.failed;
                    all = false;
                    set_status(out.status, Status::ARC_TOPOLOGY_INVALID);
                    continue;
                }
                const SharedArcNode& shared = node.arcs[ai];
                GmNodeResult result;
                if (fold_have) {
                    const auto eval_start = Clock::now();
                    result = evaluate_fold_node(fold_packet, node.R,
                                                with_jacobian);
                    const double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - eval_start).count();
                    out.cost.fold_evaluation_ms += ms;
                    ++out.cost.fold_nodes;
                } else if (packet_have) {
                    const TaylorPacket<Order>* packet = find_block(
                        blocks, node.R);
                    if (!packet) {
                        ++out.cost.failed;
                        all = false;
                        set_status(out.status,
                                   Status::TRANSPORT_TOLERANCE_FAILED);
                        continue;
                    }
                    const auto eval_start = Clock::now();
                    result = evaluate_shared_node(
                        *packet, shared, node.R, plan.pf, anchor.seed.chart2,
                        with_jacobian, &out.cost.pf6_chart_geometry_ms);
                    const double eval_ms =
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - eval_start).count();
                    out.cost.pf6_evaluation_ms += eval_ms;
                    out.cost.analytic_jacobian_ms += result.jacobian_ms;
                }
                if (!result.ok) {
                    ++out.cost.failed;
                    all = false;
                    set_status(out.status, with_jacobian
                        ? Status::GRADIENT_UNRELIABLE
                        : Status::TRANSPORT_TOLERANCE_FAILED);
                    continue;
                }
                out.F_half += node.weight * result.value;
                if (with_jacobian)
                    for (int j = 0; j < 5; ++j)
                        out.dF_half_internal[j] +=
                            node.weight * result.deriv[j];
                ++out.cost.transported;
            }
        }
    }
    finalize_evaluation(plan, out, with_jacobian);
    out.cost.ld_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    out.cost.evaluation_ms = out.cost.ld_ms;
    out.all_true = all && out.cost.failed == 0 &&
                   out.cost.incumbent_angular_fallback == 0;
    return out;
}

}  // namespace phase7_detail
}  // namespace lcbinint::holonomic

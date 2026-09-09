#pragma once

// Research-only whole-epoch value path for the true Gauss--Manin kernel.
//
// A topology cell is sampled at its physical period seed, then each arc's
// closed eta period is moved to the radial quadrature nodes by the
// equation-derived Taylor connection.  The observable is reconstructed from
// the period reduction covector h(R).  If a cell cannot be represented by a
// regular bounded t-chart or the Taylor quality gate fails, the routine uses
// a direct physical period re-seed at that node and records the fallback.
//
// This header is intentionally not included by epoch_jacobian.hpp.  It keeps
// the V2/V3 packet and production router unchanged while making the complete
// physical-seed -> F0/Fhalf -> mu path measurable.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/gm_physical_seed.hpp"
#include "lcbinint/magnification/holonomic/gm_taylor_transport.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

namespace lcbinint::holonomic {

struct GmEpochCost {
    double seed_ms = 0.0;
    double connection_ms = 0.0;
    double transport_ms = 0.0;
    double direct_ms = 0.0;
    double arc_ms = 0.0;
    double topology_ms = 0.0;
    int cells = 0;
    int arcs = 0;
    int seeds = 0;
    int transported = 0;
    int connection_attempts = 0;
    int connection_failed = 0;
    int transport_attempts = 0;
    int transport_failed = 0;
    int transport_rejected = 0;
    int direct_fallback = 0;
    int topology_fallback = 0;
    double transport_tail_max = 0.0;
};

struct GmEpochValue {
    double F0 = 0.0;
    double F_half = 0.0;
    double mu = 0.0;
    double r_max = 0.0;
    Status status = Status::OK;
    bool all_true_transport = false;
    GmEpochCost cost{};
};

namespace gm_transport_detail {

inline double arc_measure(const std::array<double, 2>& arc) {
    double hi = arc[1];
    if (hi <= arc[0]) hi += 2.0 * 3.14159265358979323846;
    return hi - arc[0];
}

inline double arc_midpoint(const std::array<double, 2>& arc) {
    double hi = arc[1];
    if (hi <= arc[0]) hi += 2.0 * 3.14159265358979323846;
    return 0.5 * (arc[0] + hi);
}

inline double angular_distance(double a, double b) {
    return std::fabs(std::atan2(std::sin(a - b), std::cos(a - b)));
}

inline int nearest_arc(const ArcSet& as, double midpoint) {
    int best = -1;
    double distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < (int)as.arcs.size(); ++i) {
        const double d = angular_distance(arc_midpoint(as.arcs[i]), midpoint);
        if (d < distance) { distance = d; best = i; }
    }
    return best;
}

inline bool full_circle_fhalf(double R, const PrimaryFrame& pf, int n,
                              double& fhalf) {
    if (!(R > 0.0) || !(pf.rho > 0.0)) return false;
    constexpr double pi = 3.14159265358979323846;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double theta = 2.0 * pi * (i + 0.5) / n;
        const double phi = phi_lens(R, theta, pf);
        if (!(phi > 0.0)) return false;
        sum += std::sqrt(phi);
    }
    fhalf = R * (2.0 * pi / n) * sum;
    return std::isfinite(fhalf);
}

template <int Order, class Scalar>
bool state_fhalf(double R, const PrimaryFrame& chart_pf,
                 const GmConnectionJet<Order, Scalar>& jet,
                 const std::array<double, 7>& seed,
                 double h, std::array<Scalar, 7>& state_out,
                 double& fhalf, double* tail_rel = nullptr) {
    std::array<Scalar, 7> state_in{};
    for (int k = 0; k < 7; ++k) state_in[k] = Scalar(seed[k]);
    if (!gm_taylor_transport(jet, state_in, h, state_out)) return false;
    const auto hc = gm_observable_h(R, chart_pf);
    auto observable = [&](const std::array<Scalar, 7>& state) {
        Scalar value = Scalar(0);
        for (int k = 0; k < 7; ++k)
            value = value + Scalar(hc[k]) * state[k];
        return value / Scalar(chart_pf.rho);
    };
    const Scalar fhalf_scalar = observable(state_out);
    fhalf = (double)fhalf_scalar;
    if (tail_rel) {
        *tail_rel = 0.0;
        if constexpr (Order >= 2) {
            std::array<Scalar, 7> lower{};
            if (!gm_taylor_transport_prefix<Order - 2>(
                    jet, state_in, h, lower))
                return false;
            const Scalar lower_fhalf = observable(lower);
            *tail_rel = gm_abs_value(fhalf_scalar - lower_fhalf) /
                        (1.0 + std::max(gm_abs_value(fhalf_scalar),
                                        gm_abs_value(lower_fhalf)));
        }
    }
    return gm_finite(fhalf_scalar) && std::isfinite(fhalf);
}

template <int Order, class Scalar>
GmEpochValue from_topology(const LensParams& p, double u, int n_r,
                           const PrimaryFrame& pf, const TopologyResult& topo,
                           bool allow_transport = true) {
    using Jet = GmConnectionJet<Order, Scalar>;
    constexpr double pi = 3.14159265358979323846;
    GmEpochValue out;
    out.r_max = topo.r_max;
    out.status = topo.status;
    const bool want_fhalf = u != 0.0;
    // A value-only epoch has no transported F_half lane.  Keep this flag
    // meaningful when the benchmark reports the fraction of value+5Jac
    // epochs that used true GM transport at every physical node.
    out.all_true_transport = allow_transport && want_fhalf &&
                             topo.status == Status::OK;
    Cheb1Dyn rr(n_r);

    for (const auto& cell : topo.cells) {
        ++out.cost.cells;
        const double width = cell.r_hi - cell.r_lo;
        if (!(width > 0.0)) continue;
        if (cell.kind == ArcKind::kEmpty) continue;

        const double inset = 1e-9 * width;
        const double lo = cell.r_lo + inset;
        const double hi = cell.r_hi - inset;
        const double Rc = 0.5 * (lo + hi);
        const double rh = 0.5 * (hi - lo);
        if (!(rh > 0.0)) continue;

        if (cell.kind == ArcKind::kFull) {
            if (want_fhalf) {
                out.all_true_transport = false;
                out.status = Status::LOCAL_REFERENCE_USED;
                ++out.cost.topology_fallback;
            }
            for (int n = 0; n < n_r; ++n) {
                const double R = Rc + rh * rr.x[n];
                const double wk = rh * rr.w[n];
                out.F0 += wk * R * 2.0 * pi;
                if (want_fhalf) {
                    double fh = 0.0;
                    if (full_circle_fhalf(R, pf, 1024, fh))
                        out.F_half += wk * fh;
                    else
                        out.status = Status::TRANSPORT_TOLERANCE_FAILED;
                }
            }
            continue;
        }

        const auto arc_start = std::chrono::steady_clock::now();
        const ArcSet center_arcs = arc_intervals(Rc, pf);
        const auto arc_end = std::chrono::steady_clock::now();
        out.cost.arc_ms += std::chrono::duration<double, std::milli>(
                               arc_end - arc_start).count();
        if (center_arcs.kind != ArcKind::kArcs || center_arcs.arcs.empty()) {
            out.all_true_transport = false;
            ++out.cost.topology_fallback;
            out.status = Status::ARC_TOPOLOGY_INVALID;
            continue;
        }

        // Uniform-source value-only mode has no F_half contribution.  Keep
        // this lane free of physical period seeds and GM coefficient work so
        // its timing is comparable to the incumbent value-only epoch path.
        if (!want_fhalf) {
            QuarticWarm node_qw;
            RootPairWarm node_rpw;
            node_rpw.certify = topo.from_warm_d14;
            for (int n = 0; n < n_r; ++n) {
                const double R = Rc + rh * rr.x[n];
                const double wk = rh * rr.w[n];
                const ArcSet node_arcs =
                    arc_intervals(R, pf, &node_qw, &node_rpw);
                if (node_arcs.kind != ArcKind::kArcs ||
                    node_arcs.arcs.empty()) {
                    out.status = Status::ARC_TOPOLOGY_INVALID;
                    continue;
                }
                double f0_node = 0.0;
                for (const auto& a : node_arcs.arcs) f0_node += arc_measure(a);
                out.F0 += wk * R * f0_node;
            }
            continue;
        }

        struct ArcWork {
            std::array<double, 2> arc{};
            GmPhysicalSeed seed{};
            PrimaryFrame chart_pf{};
            bool have_jet = false;
            const Jet* jet = nullptr;
        };
        std::vector<ArcWork> work;
        work.reserve(center_arcs.arcs.size());
        for (const auto& arc : center_arcs.arcs) {
            ArcWork aw;
            aw.arc = arc;
            const auto t0 = std::chrono::steady_clock::now();
            aw.seed = gm_physical_period_seed(Rc, pf, arc, 256);
            const auto t1 = std::chrono::steady_clock::now();
            out.cost.seed_ms += std::chrono::duration<double, std::milli>(
                                    t1 - t0).count();
            ++out.cost.seeds;
            if (!aw.seed.ok) {
                out.all_true_transport = false;
                out.status = Status::BASIS_DEGENERATE;
                ++out.cost.topology_fallback;
            }
            aw.chart_pf = aw.seed.chart_pf;
            work.push_back(aw);
            ++out.cost.arcs;
        }

        // Build one connection jet per chart and share it across the arcs in
        // that chart.  There are only the natural and reflected t charts.
        std::array<Jet, 2> chart_jets{};
        std::array<bool, 2> have_chart_jet{};
        for (auto& aw : work) {
            if (!allow_transport) continue;
            if (!aw.seed.ok) continue;
            const int chart = aw.seed.chart2 ? 1 : 0;
            if (have_chart_jet[chart]) {
                aw.jet = &chart_jets[chart];
                aw.have_jet = true;
                continue;
            }
            ++out.cost.connection_attempts;
            const auto t0 = std::chrono::steady_clock::now();
            chart_jets[chart] = gm_connection_jet<Order, Scalar>(Rc,
                GmParams<Scalar>{Scalar(aw.chart_pf.X), Scalar(aw.chart_pf.Y),
                                 Scalar(aw.chart_pf.rho), Scalar(aw.chart_pf.m0),
                                 Scalar(aw.chart_pf.a)});
            const auto t1 = std::chrono::steady_clock::now();
            out.cost.connection_ms += std::chrono::duration<double, std::milli>(
                                           t1 - t0).count();
            aw.have_jet = chart_jets[chart].ok;
            if (aw.have_jet) {
                have_chart_jet[chart] = true;
                aw.jet = &chart_jets[chart];
            }
            if (!aw.have_jet) {
                ++out.cost.connection_failed;
                out.all_true_transport = false;
                out.status = Status::CONNECTION_ILL_CONDITIONED;
                ++out.cost.topology_fallback;
            }
        }

        QuarticWarm node_qw;
        RootPairWarm node_rpw;
        node_rpw.certify = topo.from_warm_d14;
        for (int n = 0; n < n_r; ++n) {
            const double R = Rc + rh * rr.x[n];
            const double wk = rh * rr.w[n];
            const auto at0 = std::chrono::steady_clock::now();
            const ArcSet node_arcs =
                arc_intervals(R, pf, &node_qw, &node_rpw);
            const auto at1 = std::chrono::steady_clock::now();
            out.cost.arc_ms += std::chrono::duration<double, std::milli>(
                                   at1 - at0).count();
            if (node_arcs.kind != ArcKind::kArcs || node_arcs.arcs.empty()) {
                out.all_true_transport = false;
                out.status = Status::ARC_TOPOLOGY_INVALID;
                ++out.cost.topology_fallback;
                continue;
            }
            double f0_node = 0.0;
            for (const auto& a : node_arcs.arcs) f0_node += arc_measure(a);
            out.F0 += wk * R * f0_node;

            for (auto& aw : work) {
                const int ai = nearest_arc(node_arcs, arc_midpoint(aw.arc));
                double fh = 0.0;
                bool transported = false;
                if (ai >= 0 && allow_transport && aw.have_jet) {
                    ++out.cost.transport_attempts;
                    std::array<Scalar, 7> state{};
                    double tail_rel = 0.0;
                    const auto t0 = std::chrono::steady_clock::now();
                    transported = state_fhalf(R, aw.chart_pf, *aw.jet,
                                               aw.seed.eta_closed, R - Rc,
                                               state, fh, &tail_rel);
                    const auto t1 = std::chrono::steady_clock::now();
                    out.cost.transport_ms +=
                        std::chrono::duration<double, std::milli>(t1 - t0).count();
                    out.cost.transport_tail_max = std::max(
                        out.cost.transport_tail_max, tail_rel);
                    // Order-8 versus Order-6 is a cheap local truncation
                    // estimate.  A failed gate is re-seeded below so a long
                    // cell cannot silently turn a valid local connection
                    // into a whole-epoch value error.
                    if (transported && tail_rel > 1e-9) {
                        transported = false;
                        ++out.cost.transport_rejected;
                        out.all_true_transport = false;
                        if (out.status == Status::OK)
                            out.status = Status::TRANSPORT_TOLERANCE_FAILED;
                    }
                }
                if (!transported && ai >= 0 && allow_transport && aw.have_jet)
                    ++out.cost.transport_failed;
                if (transported) {
                    ++out.cost.transported;
                } else if (ai >= 0) {
                    const auto t0 = std::chrono::steady_clock::now();
                    const GmPhysicalSeed direct =
                        gm_physical_period_seed(R, pf, node_arcs.arcs[ai], 256);
                    const auto t1 = std::chrono::steady_clock::now();
                    out.cost.direct_ms +=
                        std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (direct.ok) {
                        fh = direct.fhalf_arc;
                        ++out.cost.direct_fallback;
                        out.all_true_transport = false;
                        if (out.status == Status::OK)
                            out.status = Status::LOCAL_REFERENCE_USED;
                    } else {
                        out.status = Status::TRANSPORT_TOLERANCE_FAILED;
                    }
                } else {
                    out.status = Status::ARC_TOPOLOGY_INVALID;
                }
                out.F_half += wk * fh;
            }
        }
    }

    const double denom = pi * p.rho * p.rho * (1.0 - u / 3.0);
    out.mu = ((1.0 - u) * out.F0 + u * out.F_half) / denom;
    return out;
}

}  // namespace gm_transport_detail

template <int Order, class Scalar>
GmEpochValue gm_epoch_value_from_topology(const LensParams& p, double u,
                                          int n_r, const TopologyResult& topo) {
    const PrimaryFrame pf = PrimaryFrame::from(p);
    return gm_transport_detail::from_topology<Order, Scalar>(p, u, n_r, pf,
                                                               topo);
}

template <int Order, class Scalar>
GmEpochValue gm_epoch_value_impl(const LensParams& p, double u, int n_r,
                                 bool allow_transport) {
    const PrimaryFrame pf = PrimaryFrame::from(p);
    const auto t0 = std::chrono::steady_clock::now();
    const TopologyResult topo = classify_cells(pf);
    const auto t1 = std::chrono::steady_clock::now();
    GmEpochValue out =
        gm_transport_detail::from_topology<Order, Scalar>(
            p, u, n_r, pf, topo, allow_transport);
    out.cost.topology_ms = std::chrono::duration<double, std::milli>(
                               t1 - t0).count();
    return out;
}

template <int Order, class Scalar>
GmEpochValue gm_epoch_value(const LensParams& p, double u, int n_r = 16) {
    return gm_epoch_value_impl<Order, Scalar>(p, u, n_r, true);
}

// Diagnostic reference lane: uses the same topology and physical seed
// construction but re-seeds every radial node.  It is useful for separating
// period-seed / arc bookkeeping error from Taylor transport error; it is not
// the proposed production path.
template <int Order, class Scalar>
GmEpochValue gm_epoch_value_direct_physical(const LensParams& p, double u,
                                            int n_r = 16) {
    return gm_epoch_value_impl<Order, Scalar>(p, u, n_r, false);
}

}  // namespace lcbinint::holonomic

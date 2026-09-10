// Correctness gates for the isolated physical FD6 -> FD5 six-state lane.
//
// Every physical comparison below is made against the independent angular
// lens-equation integral.  The retained K-rule/Chebyshev packet and the V2
// production router are not used as the GM seed or connection.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/gm_lauricella6.hpp"
#include "lcbinint/magnification/holonomic/gm_physical_transport.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

using namespace lcbinint::holonomic;
namespace l6 = lcbinint::holonomic::gm_l6_detail;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Case {
    const char* name;
    LensParams p;
};

const std::array<Case, 3> kCases = {{
    {"plan15", {0.2, 1.0 / 7.0, 0.125, 0.5, 1.2, false}},
    {"resonant", {0.05, 0.02, 0.05, 0.3, 0.9, false}},
    {"close", {0.4, -0.05, 0.09, 0.8, 0.55, false}},
}};

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

double arc_midpoint(const std::array<double, 2>& arc) {
    double hi = arc[1];
    if (hi <= arc[0]) hi += 2.0 * kPi;
    return 0.5 * (arc[0] + hi);
}

const std::array<double, 2>* nearest_arc(const ArcSet& arcs, double midpoint) {
    const std::array<double, 2>* best = nullptr;
    double distance = std::numeric_limits<double>::infinity();
    for (const auto& arc : arcs.arcs) {
        const double d = std::fabs(std::atan2(
            std::sin(arc_midpoint(arc) - midpoint),
            std::cos(arc_midpoint(arc) - midpoint)));
        if (d < distance) {
            distance = d;
            best = &arc;
        }
    }
    return best;
}

struct ArcSample {
    double R = 0.0;
    std::array<double, 2> arc{};
    l6::ArcSeed chart{};
    l6::Params<double> params{};
    l6::Geometry<double> geometry{};
    l6::SeedResult seed{};
    bool ok = false;
};

ArcSample first_arc_sample(const LensParams& p) {
    ArcSample out;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    const TopologyResult topo = classify_cells(pf);
    for (const auto& cell : topo.cells) {
        if (cell.kind != ArcKind::kArcs || !(cell.r_hi > cell.r_lo)) continue;
        out.R = 0.5 * (cell.r_lo + cell.r_hi);
        const ArcSet arcs = arc_intervals(out.R, pf);
        if (arcs.kind != ArcKind::kArcs || arcs.arcs.empty()) continue;
        out.arc = arcs.arcs.front();
        out.chart = l6::prepare_arc(pf, out.arc);
        if (!out.chart.ok || !l6::certify_arc(out.R, pf, out.arc, out.chart))
            continue;
        // chart_pf already contains the reflection selected by prepare_arc;
        // do not apply the reflection a second time here.
        out.params = l6::params_for_chart<double>(out.chart.chart_pf, false);
        out.geometry = l6::geometry_scalar(
            out.R, out.chart.t_lo, out.chart.t_hi, out.params);
        out.seed = l6::physical_seed(out.geometry, 512);
        out.ok = out.geometry.ok && out.seed.ok;
        if (out.ok) return out;
    }
    return out;
}

bool seed_at_radius(const LensParams& p, double R, double midpoint,
                    bool chart2, double& value) {
    const PrimaryFrame pf = PrimaryFrame::from(p);
    const ArcSet arcs = arc_intervals(R, pf);
    const auto* arc = nearest_arc(arcs, midpoint);
    if (arcs.kind != ArcKind::kArcs || !arc) return false;
    l6::ArcSeed chart = l6::prepare_arc_chart(pf, *arc, chart2);
    if (!chart.ok || !l6::certify_arc(R, pf, *arc, chart)) return false;
    const auto params = l6::params_for_chart<double>(chart.chart_pf, false);
    const auto geometry = l6::geometry_scalar(
        R, chart.t_lo, chart.t_hi, params);
    const auto seed = l6::physical_seed(geometry, 1024);
    if (!seed.ok) return false;
    value = seed.fhalf;
    return true;
}

void check_seed_and_local_transport(const Case& c) {
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    const ArcSample sample = first_arc_sample(c.p);
    require(sample.ok, "physical arc seed is constructible");
    if (!sample.ok) return;

    double angular = 0.0;
    require(gm_physical_arc_angular(sample.R, pf, sample.arc, 2048,
                                    angular),
            "independent angular reference is valid");
    require(relerr(sample.seed.fhalf, angular) < 2e-8,
            "FD6 physical seed agrees with independent angular reference");

    const auto topo = classify_cells(pf);
    const CellPlan* cell = nullptr;
    for (const auto& candidate : topo.cells) {
        if (candidate.kind == ArcKind::kArcs &&
            sample.R > candidate.r_lo && sample.R < candidate.r_hi) {
            cell = &candidate;
            break;
        }
    }
    require(cell != nullptr, "seed belongs to an arcs cell");
    if (!cell) return;

    const double inset = 0.1 * (cell->r_hi - cell->r_lo);
    l6::TaylorPacket<20> packet;
    const bool packet_ok = l6::build_packet(
        sample.R, cell->r_lo + inset, cell->r_hi - inset,
        sample.chart.t_lo, sample.chart.t_hi, sample.params,
        sample.seed.state.z, packet);
    require(packet_ok && packet.ok, "equation-derived Taylor packet builds");
    if (!packet_ok || !packet.ok) return;

    const auto node = l6::evaluate_node(
        packet, sample.R, pf, sample.chart.chart2, true);
    require(node.ok, "six-state dense output evaluates with analytic Jac");
    require(node.phase_error < 2e-7, "physical branch phase gate is small");
    if (!node.ok) return;
    require(relerr(node.value, angular) < 2e-8,
            "dense output at the seed agrees with angular reference");

    const double h = 1e-5 * std::max(1.0, cell->r_hi - cell->r_lo);
    double fp = 0.0, fm = 0.0;
    const bool okp = seed_at_radius(c.p, sample.R + h,
                                    arc_midpoint(sample.arc),
                                    sample.chart.chart2, fp);
    const bool okm = seed_at_radius(c.p, sample.R - h,
                                    arc_midpoint(sample.arc),
                                    sample.chart.chart2, fm);
    require(okp && okm, "nearby physical re-seeds are valid reference points");
    if (okp && okm) {
        const double fd = (fp - fm) / (2.0 * h);
        const auto plus = l6::evaluate_node(
            packet, sample.R + h, pf, sample.chart.chart2, false);
        const auto minus = l6::evaluate_node(
            packet, sample.R - h, pf, sample.chart.chart2, false);
        const double gd = (plus.value - minus.value) / (2.0 * h);
        require(plus.ok && minus.ok,
                "dense output remains valid at local reference points");
        require(relerr(gd, fd) < 2e-5,
                "Taylor transport derivative agrees with physical re-seed");
    }

    // The parameter derivative is the same six-state value plus algebraic
    // geometry chain rule.  Compare its internal-frame components with a
    // central difference of the independent angular integral.
    double max_jac_error = 0.0;
    const std::array<double, 5> base = {{pf.X, pf.Y, pf.rho, pf.m0, pf.a}};
    for (int j = 0; j < 5; ++j) {
        PrimaryFrame plus = pf, minus = pf;
        const double d = 2e-6 * std::max(1.0, std::fabs(base[j]));
        double* vp[5] = {&plus.X, &plus.Y, &plus.rho, &plus.m0, &plus.a};
        double* vm[5] = {&minus.X, &minus.Y, &minus.rho, &minus.m0, &minus.a};
        *vp[j] += d;
        *vm[j] -= d;
        const ArcSet ap = arc_intervals(sample.R, plus);
        const ArcSet am = arc_intervals(sample.R, minus);
        const auto* arp = nearest_arc(ap, arc_midpoint(sample.arc));
        const auto* arm = nearest_arc(am, arc_midpoint(sample.arc));
        double vpv = 0.0, vmv = 0.0;
        const bool valid = arp && arm &&
            gm_physical_arc_angular(sample.R, plus, *arp, 2048, vpv) &&
            gm_physical_arc_angular(sample.R, minus, *arm, 2048, vmv);
        require(valid, "independent angular parameter references are valid");
        if (!valid) continue;
        const double fd = (vpv - vmv) / (2.0 * d);
        max_jac_error = std::max(max_jac_error,
                                 relerr(node.deriv[j], fd));
    }
    require(max_jac_error < 3e-4,
            "six-state chain-rule Jacobian agrees with angular FD");

    std::printf("local %-9s seed_rel=%.3e packet_tail=%.3e jac_rel=%.3e "
                "chart=%d\n",
                c.name, relerr(sample.seed.fhalf, angular), packet.tail,
                max_jac_error, sample.chart.chart2 ? 2 : 1);
}

void check_fold_and_chart_authentication(const Case& c) {
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    const auto events = radial_events(pf, nullptr);
    const bool require_p4 = std::string(c.name) != "plan15";
    bool fold_ok = false;
    bool p4_seen = false;
    int p4_side_certified = 0;
    for (const auto& event : events) {
        if (event.kind == "physical_real") {
            const auto roots = real_root_thetas(
                boundary_quartic(event.radius, pf).p);
            for (double theta : roots) {
                for (bool chart2 : {false, true}) {
                    const double shift = chart2 ? kPi : 0.0;
                    const double t = std::tan(0.5 * l6::wrap_pi(theta - shift));
                    const auto params = l6::params_for_chart<double>(pf, chart2);
                    const auto germ = l6::fold_germ(event.radius, t, params);
                    if (!germ.ok) continue;
                    const bool state_ok =
                        std::fabs(germ.state[0].re - 1.0) < 1e-14 &&
                        std::fabs(germ.state[0].im) < 1e-14 &&
                        std::fabs(germ.state[1].re - 0.5) < 1e-14;
                    require(state_ok, "fold germ has the regular FD5 state");
                    require(std::isfinite(germ.K0) && germ.K0 > 0.0,
                            "fold germ flux coefficient is finite");
                    fold_ok = true;
                    break;
                }
                if (fold_ok) break;
            }
        }
        if (event.kind != "chart_p4") continue;
        p4_seen = true;
        const double eps = 1e-6 * std::max(1.0, event.radius);
        for (double R : {event.radius - eps, event.radius + eps}) {
            if (!(R > 0.0)) continue;
            const ArcSet arcs = arc_intervals(R, pf);
            if (arcs.kind != ArcKind::kArcs) continue;
            for (const auto& arc : arcs.arcs) {
                auto chart = l6::prepare_arc(pf, arc);
                if (chart.ok && l6::certify_arc(R, pf, arc, chart)) {
                    ++p4_side_certified;
                    break;
                }
            }
        }
    }
    require(fold_ok, "an exact physical fold germ is exercised");
    if (require_p4)
        require(p4_seen, "a chart_p4 event is present in the diagnostic case");
    if (p4_seen)
        require(p4_side_certified >= 2,
                "chart/branch certificate survives both chart_p4 sides");
}

void check_whole_epoch() {
    const LensParams p = kCases[0].p;
    holo_holonomic_transport_override() = 0;
    holo_mv_transport_override() = 0;
    const auto value = gm_lauricella6_epoch<20>(p, 0.5, 64, nullptr, false);
    const auto jac = gm_lauricella6_epoch<20>(p, 0.5, 64, nullptr, true);
    const auto v2 = epoch_jacobian(p, 0.5, 64, false);
    double grad_error = 0.0;
    for (int j = 0; j < 5; ++j)
        grad_error = std::max(grad_error, relerr(jac.grad_mu[j],
                                                   v2.grad_mu[j]));
    require(value.status == Status::OK && value.all_true_transport,
            "whole epoch value is true transport and OK");
    require(jac.status == Status::OK && jac.all_true_transport,
            "whole epoch analytic Jac is true transport and OK");
    require(value.cost.nodes == value.cost.transported &&
                value.cost.physical_fallback == 0,
            "whole epoch value has no physical re-seed fallback");
    require(jac.cost.nodes == jac.cost.transported &&
                jac.cost.jacobian_nodes == jac.cost.nodes &&
                jac.cost.physical_fallback == 0,
            "whole epoch Jac has no physical re-seed fallback");
    require(relerr(value.mu, v2.mu) < 2e-7,
            "whole epoch value matches V2 at the same radial precision");
    require(relerr(jac.mu, v2.mu) < 2e-7,
            "whole epoch Jac value matches V2");
    require(grad_error < 2e-6,
            "whole epoch analytic Jac matches V2 on the diagnostic case");
    std::printf("epoch value=%.12g v2=%.12g rel=%.3e jac_rel=%.3e "
                "grad_rel=%.3e nodes=%d/%d jacnodes=%d fallback=%d\n",
                value.mu, v2.mu, relerr(value.mu, v2.mu),
                relerr(jac.mu, v2.mu), grad_error, value.cost.nodes,
                value.cost.transported, jac.cost.jacobian_nodes,
                jac.cost.physical_fallback);
}

}  // namespace

int main() {
    for (const auto& c : kCases) {
        check_seed_and_local_transport(c);
        check_fold_and_chart_authentication(c);
    }
    check_whole_epoch();
    std::printf("gm_lauricella6 %s failures=%d\n",
                failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}

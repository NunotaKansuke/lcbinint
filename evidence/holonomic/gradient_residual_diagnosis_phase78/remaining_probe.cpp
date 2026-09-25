#include "tests/holonomic_cpp/adaptive_reference.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

using namespace lcbinint::holonomic;

namespace {

bool is_projective_fold(const RadialEvent& e) {
    return adaptive_detail::adaptive_physical_fold_event(e);
}

double direct_reference_projective(const LensParams& p, double u,
                                   const TopologyResult& input, int n,
                                   int subdivisions) {
    if (input.status != Status::OK || n < 8 || subdivisions < 1) return NAN;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    TopologyResult topo = input;
    adaptive_detail::certify_projective_p4_events(topo, pf);
    GL radial(n);
    Cheb1Dyn angular(2 * n);
    auto fold = [&](double R) {
        for (const auto& e : topo.events)
            if (e.radius == R && is_projective_fold(e)) return true;
        return false;
    };

    std::vector<double> cuts{0.0, topo.r_max};
    for (const auto& c : topo.cells) {
        cuts.push_back(c.r_lo);
        cuts.push_back(c.r_hi);
    }
    for (const auto& e : topo.events)
        if (std::isfinite(e.radius) && e.radius > 0.0 && e.radius < topo.r_max)
            cuts.push_back(e.radius);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    long double total = 0.0L;
    for (std::size_t ci = 1; ci < cuts.size(); ++ci) {
        const double a = cuts[ci - 1], b = cuts[ci];
        if (!(b > a)) continue;
        const auto middle = quartic_topology(a + 0.5 * (b - a), pf);
        if (!middle.certified) return NAN;
        if (middle.kind == ArcKind::kEmpty) continue;
        if (middle.kind != ArcKind::kArcs && middle.kind != ArcKind::kFull)
            return NAN;

        const FoldRadialMap map{a, b, fold(a), fold(b)};
        for (int s = 0; s < subdivisions; ++s) {
            const double xl = -1.0 + 2.0 * s / subdivisions;
            const double xr = -1.0 + 2.0 * (s + 1) / subdivisions;
            const double xc = 0.5 * (xl + xr), xh = 0.5 * (xr - xl);
            for (int k = 0; k < n; ++k) {
                const auto rr = map(xc + xh * radial.x[k]);
                const double R = rr[0], rw = xh * radial.w[k] * rr[1];
                double f0 = 0.0, fh = 0.0;
                if (middle.kind == ArcKind::kFull) {
                    f0 = kTwoPi * R;
                    if (u != 0.0) {
                        for (int j = 0; j < 8 * n; ++j) {
                            const double th = kTwoPi * (j + 0.5) / (8 * n);
                            const double ph = phi_val(R, th, pf);
                            if (!(ph > 0.0) || !std::isfinite(ph)) return NAN;
                            fh += kTwoPi * R * std::sqrt(ph) / (8 * n);
                        }
                    }
                } else {
                    const auto arcs = reference_arcs(R, pf);
                    if (arcs.kind != ArcKind::kArcs) return NAN;
                    for (const auto& arc : arcs.arcs) {
                        const auto lo = polish_endpoint(R, arc[0], pf);
                        const auto hi = polish_endpoint(R, arc[1], pf);
                        double tl = lo.theta, tr = hi.theta;
                        if (!std::isfinite(tl + tr)) return NAN;
                        if (tr <= tl) tr += kTwoPi;
                        const double h = 0.5 * (tr - tl), m = 0.5 * (tr + tl);
                        f0 += R * (tr - tl);
                        if (u != 0.0) {
                            for (int j = 0; j < 2 * n; ++j) {
                                const double th = m + h * angular.x[j];
                                const double ph = phi_val(R, th, pf);
                                if (!(ph > 0.0) || !std::isfinite(ph)) return NAN;
                                fh += R * h * angular.w[j] * std::sqrt(ph);
                            }
                        }
                    }
                }
                total += static_cast<long double>(rw) *
                         ((1.0 - u) * f0 + u * fh);
            }
        }
    }
    return static_cast<double>(total /
        (kPi * p.rho * p.rho * (1.0 - u / 3.0)));
}

TopologyResult cold_topology(const LensParams& p) {
    return classify_cells(PrimaryFrame::from(p), nullptr, nullptr, true);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: remaining_probe CASES_TSV OUTPUT_TSV\n";
        return 2;
    }
    LensParams base;
    bool found = false;
    std::ifstream in(argv[1]);
    std::string line;
    int row = 0, barycentric = 0;
    double input_u = 0.0, reference_mu = 0.0;
    std::string name;
    while (std::getline(in, line)) {
        ++row;
        std::istringstream ss(line);
        if (!(ss >> base.xs >> base.ys >> base.rho >> base.q >> base.a >>
              barycentric >> input_u >> reference_mu >> name)) continue;
        if (row == 17 && name == "caustic-cross") {
            found = true;
            break;
        }
    }
    if (!found) return 3;
    base.barycentric = barycentric != 0;
    constexpr double u = 0.5;
    const auto topo = cold_topology(base);
    if (topo.status != Status::OK) return 4;
    auto marked = topo;
    adaptive_detail::certify_projective_p4_events(
        marked, PrimaryFrame::from(base));
    int projective_count = 0;
    for (const auto& e : marked.events)
        projective_count += e.projective_fold_certified ? 1 : 0;

    std::ofstream out(argv[2], std::ios::trunc);
    if (!out) return 5;
    out << std::setprecision(17)
        << "kind\tarm\tu\tvalue_rtol\tnodes\tpanels\tmerges\tmu\t"
           "value_error\tgrad_x\tquality_x\treason_x\tgrad_error_x\t"
           "radial_x\tinner_x\tgeometry_x\tevent_x\troundoff_x\t"
           "projective_fold_count\n";
    const char* arm =
#if defined(HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH)
        "projective";
#else
        "baseline";
#endif
    for (double rtol : {1e-3, 1e-4}) {
        AdaptiveConfig cfg;
        cfg.gradient_policy = GradientPolicy::ValueFirst;
        cfg.with_jacobian = true;
        cfg.collect_diagnostics = true;
        cfg.tol.mu_rtol = rtol;
        AdaptiveWorkspace workspace;
        const auto r = flux_adaptive_integrate(base, u, topo, cfg, workspace);
        out << "adaptive\t" << arm << '\t' << u << '\t' << rtol << '\t'
            << r.stats.unique_nodes << '\t' << r.stats.panels << '\t'
            << r.stats.softcut_fold_merges << '\t' << r.mu << '\t'
            << r.estimated_abs_error_mu << '\t' << r.grad_mu[0] << '\t'
            << gradient_quality_name(r.grad_quality[0]) << '\t'
            << gradient_reason_name(r.grad_reason[0]) << '\t'
            << r.estimated_abs_error_grad[0] << '\t' << r.radial_error[1]
            << '\t' << r.inner_error[1] << '\t' << r.geometry_error[1]
            << '\t' << r.event_error[1] << '\t' << r.roundoff_error[1]
            << '\t' << projective_count << '\n';
    }

    out << "reference\tindependent_value_fd\t" << u
        << "\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t"
        << projective_count << '\n';
    for (const auto [n, subdivisions] :
         {std::pair<int,int>{64,1}, {128,2}, {256,4}}) {
        for (double hscale : {1e-3, 5e-4, 2.5e-4}) {
            const double h = base.rho * hscale;
            std::array<double, 4> values{};
            bool ok = true;
            for (int i = 0; i < 4; ++i) {
                const int offset = std::array<int,4>{{-2,-1,1,2}}[i];
                LensParams p = base;
                p.xs += offset * h;
                const auto perturbed_topology = cold_topology(p);
                values[i] = direct_reference_projective(
                    p, u, perturbed_topology, n, subdivisions);
                ok = ok && std::isfinite(values[i]);
            }
            double fd = NAN;
            if (ok) fd = (values[0] - 8.0 * values[1] +
                          8.0 * values[2] - values[3]) / (12.0 * h);
            out << "fd\tGL" << n << "x" << subdivisions << "\t" << u
                << "\t" << hscale << "\t-\t-\t-\t-\t-\t" << fd
                << "\t" << (ok ? "finite" : "failed")
                << "\t" << (ok ? "cold_topology_5point" : "reference_failure")
                << "\t-\t-\t-\t-\t-\t" << projective_count << '\n';
        }
    }
    return 0;
}

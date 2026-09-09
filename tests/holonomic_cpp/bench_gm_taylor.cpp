// Timing harness for the equation-derived Gauss--Manin kernel.
//
// The lanes are intentionally separated:
//   point       one numeric connection at the cell centre;
//   point_march a 16-substep RK4 reference using point connections;
//   taylor      one order-10 __float128 connection jet + Taylor recurrence;
//   taylor_jac one order-3 five-parameter dual jet + Taylor recurrence.
// The existing V3 Chebyshev/K-rule packet is not called here; its whole-epoch
// number is recorded by bench_holonomic_ode separately.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/gm_taylor_transport.hpp"
#include "lcbinint/magnification/holonomic/gm_residue_free.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"

using namespace lcbinint::holonomic;

namespace {

struct Case {
    LensParams p;
    std::string name;
    double R = 0.0;
    bool dual_ok = false;
};

struct Stats {
    std::vector<double> ms;
    double checksum = 0.0;
};

volatile double g_sink = 0.0;

inline __float128 qf_value(double x) { return (__float128)x; }
inline __float128 qf_value(DD x) { return qf_from_dd(x); }

double millis() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class Fn>
double best_of(int reps, Fn&& fn) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const double t0 = millis();
        g_sink += fn();
        best = std::min(best, millis() - t0);
    }
    return best;
}

void percentile(std::vector<double> x, double& med, double& p90, double& p99) {
    std::sort(x.begin(), x.end());
    auto at = [&](double q) {
        if (x.empty()) return 0.0;
        const size_t i = std::min(x.size() - 1,
                                  (size_t)std::floor(q * (x.size() - 1)));
        return x[i];
    };
    med = at(0.50);
    p90 = at(0.90);
    p99 = at(0.99);
}

bool rhs_point(const std::array<double, kGmEtaDim>& z, double R,
               const PrimaryFrame& pf, std::array<double, kGmEtaDim>& out) {
    const GmConnectionPoint c = gm_connection_at(R, pf);
    if (!c.ok) return false;
    out.fill(0.0);
    for (int i = 0; i < kGmEtaDim; ++i)
        for (int j = 0; j < kGmEtaDim; ++j) out[i] += c.C[i][j] * z[j];
    return true;
}

bool point_march(const Case& c, std::array<double, kGmEtaDim>& out) {
    out.fill(0.0);
    for (int i = 0; i < kGmEtaDim; ++i) out[i] = 0.25 + 0.17 * i;
    constexpr int nstep = 16;
    const double h = 0.01, dr = h / nstep;
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    for (int step = 0; step < nstep; ++step) {
        const double R = c.R + step * dr;
        std::array<double, kGmEtaDim> k1{}, k2{}, k3{}, k4{}, z{};
        if (!rhs_point(out, R, pf, k1)) return false;
        for (int i = 0; i < kGmEtaDim; ++i) z[i] = out[i] + 0.5 * dr * k1[i];
        if (!rhs_point(z, R + 0.5 * dr, pf, k2)) return false;
        for (int i = 0; i < kGmEtaDim; ++i) z[i] = out[i] + 0.5 * dr * k2[i];
        if (!rhs_point(z, R + 0.5 * dr, pf, k3)) return false;
        for (int i = 0; i < kGmEtaDim; ++i) z[i] = out[i] + dr * k3[i];
        if (!rhs_point(z, R + dr, pf, k4)) return false;
        for (int i = 0; i < kGmEtaDim; ++i)
            out[i] += dr * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0;
    }
    return true;
}

double run_point(const Case& c) {
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    const auto r = gm_connection_at(c.R, pf);
    return r.ok ? r.C[0][0] : 0.0;
}

template <int Order, class Scalar>
GmConnectionJet<Order, Scalar> make_jet(const Case& c) {
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    return gm_connection_jet<Order, Scalar>(c.R,
        GmParams<Scalar>{Scalar(pf.X), Scalar(pf.Y), Scalar(pf.rho),
                         Scalar(pf.m0), Scalar(pf.a)});
}

template <int Order, class Scalar>
double run_taylor_full(const Case& c) {
    const auto jet = make_jet<Order, Scalar>(c);
    if (!jet.ok) return 0.0;
    std::array<Scalar, kGmEtaDim> seed{}, out{};
    for (int i = 0; i < kGmEtaDim; ++i) seed[i] = Scalar(0.25 + 0.17 * i);
    if (!gm_taylor_transport(jet, seed, 0.01, out)) return 0.0;
    return (double)out[0];
}

template <int Order, class Scalar>
bool taylor_vector(const GmConnectionJet<Order, Scalar>& jet,
                   std::array<Scalar, kGmEtaDim>& out) {
    if (!jet.ok) return false;
    std::array<Scalar, kGmEtaDim> seed{};
    for (int i = 0; i < kGmEtaDim; ++i) seed[i] = Scalar(0.25 + 0.17 * i);
    return gm_taylor_transport(jet, seed, 0.01, out);
}

template <int Order, class Scalar>
double run_taylor_coeff(const Case& c) {
    const auto jet = make_jet<Order, Scalar>(c);
    return jet.ok ? (double)jet.C[0][0].c[0] : 0.0;
}

template <int Order, class Scalar>
double run_taylor_transport(const GmConnectionJet<Order, Scalar>& jet) {
    if (!jet.ok) return 0.0;
    std::array<Scalar, kGmEtaDim> seed{}, out{};
    for (int i = 0; i < kGmEtaDim; ++i) seed[i] = Scalar(0.25 + 0.17 * i);
    if (!gm_taylor_transport(jet, seed, 0.01, out)) return 0.0;
    return (double)out[0];
}

template <int Order, class Scalar>
double run_residue_free_projection(
    const GmConnectionJet<Order, Scalar>& eta) {
    const auto psi = gm_residue_free_jet(eta);
    return psi.ok ? (double)psi.C[0][0].c[0] : 0.0;
}

template <int Order, class Scalar>
double run_residue_free_transport(
    const GmPsiJet<Order, Scalar>& psi) {
    if (!psi.ok) return 0.0;
    std::array<Scalar, kGmPsiDim> seed{}, out{};
    for (int i = 0; i < kGmPsiDim; ++i) seed[i] = Scalar(0.25 + 0.17 * i);
    if (!gm_psi_taylor_transport(psi, seed, 0.01, out)) return 0.0;
    return (double)out[0];
}

double run_taylor_jac_transport(
    const GmConnectionJet<3, GmDual5>& jet) {
    if (!jet.ok) return 0.0;
    std::array<GmDual5, kGmEtaDim> seed{}, out{};
    for (int i = 0; i < kGmEtaDim; ++i) seed[i] = GmDual5(0.25 + 0.17 * i);
    if (!gm_taylor_transport_dual(jet, seed, 0.01, out)) return 0.0;
    return out[0].value + out[0].deriv[0];
}

double run_taylor_jac_full(const Case& c) {
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    const auto jet = gm_connection_jet_dual<3>(c.R, pf);
    if (!jet.ok) return 0.0;
    std::array<GmDual5, kGmEtaDim> seed{}, out{};
    for (int i = 0; i < kGmEtaDim; ++i) seed[i] = GmDual5(0.25 + 0.17 * i);
    if (!gm_taylor_transport_dual(jet, seed, 0.01, out)) return 0.0;
    return out[0].value + out[0].deriv[0];
}

double run_taylor_jac_coeff(const Case& c) {
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    const auto jet = gm_connection_jet_dual<3>(c.R, pf);
    return jet.ok ? jet.C[0][0].c[0].value : 0.0;
}

struct Accuracy {
    int accepted = 0;
    int transported = 0;
    double worst_rel = 0.0;
};

template <int Order, class Scalar>
void accumulate_accuracy(const Case& c,
                         const std::array<__float128, kGmEtaDim>& ref,
                         Accuracy& acc) {
    const auto jet = make_jet<Order, Scalar>(c);
    if (!jet.ok) return;
    ++acc.accepted;
    std::array<Scalar, kGmEtaDim> got{};
    if (!taylor_vector(jet, got)) return;
    ++acc.transported;
    for (int i = 0; i < kGmEtaDim; ++i) {
        const __float128 r = ref[i];
        const __float128 g = qf_value(got[i]);
        acc.worst_rel = std::max(acc.worst_rel,
                                 (double)(fabsq(g - r) /
                                          (1.0 + fabsq(r))));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    const int reps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 5;
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }

    std::vector<Case> cases;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        double xs, ys, rho, q, a, u, tj;
        int bary;
        std::string name;
        if (!(ls >> xs >> ys >> rho >> q >> a >> bary >> u >> tj >> name)) continue;
        Case c{LensParams{xs, ys, rho, q, a, (bool)bary}, name};
        const PrimaryFrame pf = PrimaryFrame::from(c.p);
        for (double R : {0.30, 0.55, 0.80, 1.05, 1.35, 1.80}) {
            const auto point = gm_connection_at(R, pf);
            if (!point.ok || point.matrix_pivot_rel <= 1e-8) continue;
            const auto jet = gm_connection_jet<10, __float128>(R,
                GmParams<__float128>{pf.X, pf.Y, pf.rho, pf.m0, pf.a});
            if (!jet.ok) continue;
            c.R = R;
            c.dual_ok = gm_connection_jet_dual<3>(R, pf).ok;
            cases.push_back(c);
            break;
        }
    }
    if (cases.empty()) {
        std::fprintf(stderr, "no well-conditioned GM cells\n");
        return 3;
    }

    Stats point, march, taylor, taylor_coeff, taylor_transport, tjac;
    Stats td3, td6, td8, td10, tdd3, tdd6, tdd8, tdd10;
    Stats dual_coeff, dual_transport, psi_projection, psi_transport;
    for (Stats* s : {&point, &march, &taylor, &taylor_coeff,
                     &taylor_transport, &tjac, &td3, &td6, &td8, &td10,
                     &tdd3, &tdd6, &tdd8, &tdd10, &dual_coeff,
                     &dual_transport, &psi_projection, &psi_transport})
        s->ms.reserve(cases.size());
    Accuracy a3, a6, a8, a10, ad3, ad6, ad8, ad10;
    int dual_cases = 0, psi_cases = 0;
    for (const Case& c : cases) {
        point.ms.push_back(best_of(reps, [&] { return run_point(c); }));
        march.ms.push_back(best_of(reps, [&] {
            std::array<double, kGmEtaDim> z{};
            return point_march(c, z) ? z[0] : 0.0;
        }));
        taylor.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<10, __float128>(c);
        }));
        taylor_coeff.ms.push_back(best_of(reps, [&] {
            return run_taylor_coeff<10, __float128>(c);
        }));
        const auto qf_jet = make_jet<10, __float128>(c);
        taylor_transport.ms.push_back(best_of(reps, [&] {
            return run_taylor_transport(qf_jet);
        }));
        std::array<__float128, kGmEtaDim> qf_out{};
        if (!taylor_vector(qf_jet, qf_out)) {
            std::fprintf(stderr, "qf transport failed for %s\n", c.name.c_str());
        } else {
            accumulate_accuracy<3, double>(c, qf_out, a3);
            accumulate_accuracy<6, double>(c, qf_out, a6);
            accumulate_accuracy<8, double>(c, qf_out, a8);
            accumulate_accuracy<10, double>(c, qf_out, a10);
            accumulate_accuracy<3, DD>(c, qf_out, ad3);
            accumulate_accuracy<6, DD>(c, qf_out, ad6);
            accumulate_accuracy<8, DD>(c, qf_out, ad8);
            accumulate_accuracy<10, DD>(c, qf_out, ad10);
        }
        td3.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<3, double>(c);
        }));
        td6.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<6, double>(c);
        }));
        td8.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<8, double>(c);
        }));
        td10.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<10, double>(c);
        }));
        tdd3.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<3, DD>(c);
        }));
        tdd6.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<6, DD>(c);
        }));
        tdd8.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<8, DD>(c);
        }));
        tdd10.ms.push_back(best_of(reps, [&] {
            return run_taylor_full<10, DD>(c);
        }));
        const auto eta_dd8 = make_jet<8, DD>(c);
        const auto psi_dd8 = gm_residue_free_jet(eta_dd8);
        if (psi_dd8.ok) {
            ++psi_cases;
            psi_projection.ms.push_back(best_of(reps, [&] {
                return run_residue_free_projection(eta_dd8);
            }));
            psi_transport.ms.push_back(best_of(reps, [&] {
                return run_residue_free_transport(psi_dd8);
            }));
        }
        if (c.dual_ok) {
            ++dual_cases;
            tjac.ms.push_back(best_of(reps, [&] {
                return run_taylor_jac_full(c);
            }));
            dual_coeff.ms.push_back(best_of(reps, [&] {
                return run_taylor_jac_coeff(c);
            }));
            const auto djet = gm_connection_jet_dual<3>(c.R,
                                                         PrimaryFrame::from(c.p));
            dual_transport.ms.push_back(best_of(reps, [&] {
                return run_taylor_jac_transport(djet);
            }));
        }
        point.checksum += run_point(c);
        std::array<double, kGmEtaDim> z{};
        if (point_march(c, z)) march.checksum += z[0];
        taylor.checksum += run_taylor_full<10, __float128>(c);
        if (c.dual_ok) tjac.checksum += run_taylor_jac_full(c);
    }

    double med, p90, p99;
    std::printf("GM equation-derived Taylor benchmark\n");
    std::printf("cases                 %zu  source=%s  reps=%d  step=0.01\n",
                cases.size(), path, reps);
    percentile(point.ms, med, p90, p99);
    std::printf("point value-only      median %.4f ms  p90 %.4f  p99 %.4f\n", med, p90, p99);
    percentile(march.ms, med, p90, p99);
    std::printf("16-step RK4 value     median %.4f ms  p90 %.4f  p99 %.4f\n", med, p90, p99);
    percentile(taylor.ms, med, p90, p99);
    std::printf("Taylor-10 value-only  median %.4f ms  p90 %.4f  p99 %.4f\n", med, p90, p99);
    const double taylor_med = med;
    percentile(taylor_coeff.ms, med, p90, p99);
    std::printf("  connection Taylor coefficients (__float128) median %.4f ms  p90 %.4f  p99 %.4f\n",
                med, p90, p99);
    percentile(taylor_transport.ms, med, p90, p99);
    std::printf("  transport recurrence only (__float128) median %.4f ms  p90 %.4f  p99 %.4f\n",
                med, p90, p99);
    auto report_lane = [&](const char* name, const Stats& s) {
        double a, b, d;
        percentile(s.ms, a, b, d);
        std::printf("%s median %.4f ms  p90 %.4f  p99 %.4f\n", name, a, b, d);
    };
    report_lane("Taylor-3 double       ", td3);
    report_lane("Taylor-6 double       ", td6);
    report_lane("Taylor-8 double       ", td8);
    report_lane("Taylor-10 double      ", td10);
    std::printf("double precision gate  Order-3 %d/%zu transported %d worst-vs-qf %.3e\n",
                a3.accepted, cases.size(), a3.transported, a3.worst_rel);
    std::printf("double precision gate  Order-6 %d/%zu transported %d worst-vs-qf %.3e\n",
                a6.accepted, cases.size(), a6.transported, a6.worst_rel);
    std::printf("double precision gate  Order-8 %d/%zu transported %d worst-vs-qf %.3e\n",
                a8.accepted, cases.size(), a8.transported, a8.worst_rel);
    std::printf("double precision gate  Order-10 %d/%zu transported %d worst-vs-qf %.3e\n",
                a10.accepted, cases.size(), a10.transported, a10.worst_rel);
    std::printf("DD precision gate      Order-3 %d/%zu transported %d worst-vs-qf %.3e\n",
                ad3.accepted, cases.size(), ad3.transported, ad3.worst_rel);
    std::printf("DD precision gate      Order-6 %d/%zu transported %d worst-vs-qf %.3e\n",
                ad6.accepted, cases.size(), ad6.transported, ad6.worst_rel);
    std::printf("DD precision gate      Order-8 %d/%zu transported %d worst-vs-qf %.3e\n",
                ad8.accepted, cases.size(), ad8.transported, ad8.worst_rel);
    std::printf("DD precision gate      Order-10 %d/%zu transported %d worst-vs-qf %.3e\n",
                ad10.accepted, cases.size(), ad10.transported, ad10.worst_rel);
    report_lane("Taylor-3 DD          ", tdd3);
    report_lane("Taylor-6 DD          ", tdd6);
    report_lane("Taylor-8 DD          ", tdd8);
    report_lane("Taylor-10 DD         ", tdd10);
    if (!psi_projection.ms.empty()) {
        report_lane("6D residue-free projection (DD O8)", psi_projection);
        report_lane("6D residue-free transport  (DD O8)", psi_transport);
        std::printf("6D residue-free closure-accepted %d/%zu; "
                    "timings exclude the 7D eta jet generation\n",
                    psi_cases, cases.size());
    }
    double point_med = 0.0, point_p90 = 0.0, point_p99 = 0.0;
    percentile(point.ms, point_med, point_p90, point_p99);
    if (!tjac.ms.empty()) {
        percentile(tjac.ms, med, p90, p99);
        std::printf("Taylor-3 + 5-dual     median %.4f ms  p90 %.4f  p99 %.4f  cases %d\n",
                    med, p90, p99, dual_cases);
        report_lane("  dual coefficients  ", dual_coeff);
        report_lane("  dual transport     ", dual_transport);
    }
    std::printf("Taylor/point          %.3fx  (median)\n",
                taylor.ms.empty() ? 0.0 : taylor_med / std::max(1e-300, point_med));
    std::printf("checksum              %.12g %.12g %.12g %.12g\n",
                point.checksum, march.checksum, taylor.checksum, tjac.checksum);
    return 0;
}

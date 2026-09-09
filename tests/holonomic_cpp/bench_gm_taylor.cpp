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

double run_taylor(const Case& c) {
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    const auto jet = gm_connection_jet<10, __float128>(c.R,
        GmParams<__float128>{pf.X, pf.Y, pf.rho, pf.m0, pf.a});
    if (!jet.ok) return 0.0;
    std::array<__float128, kGmEtaDim> seed{}, out{};
    for (int i = 0; i < kGmEtaDim; ++i) seed[i] = 0.25 + 0.17 * i;
    if (!gm_taylor_transport(jet, seed, 0.01, out)) return 0.0;
    return (double)out[0];
}

double run_taylor_jac(const Case& c) {
    const PrimaryFrame pf = PrimaryFrame::from(c.p);
    const auto jet = gm_connection_jet_dual<3>(c.R, pf);
    if (!jet.ok) return 0.0;
    std::array<GmDual5, kGmEtaDim> seed{}, out{};
    for (int i = 0; i < kGmEtaDim; ++i) seed[i] = GmDual5(0.25 + 0.17 * i);
    if (!gm_taylor_transport_dual(jet, seed, 0.01, out)) return 0.0;
    return out[0].value + out[0].deriv[0];
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

    Stats point, march, taylor, tjac;
    point.ms.reserve(cases.size());
    march.ms.reserve(cases.size());
    taylor.ms.reserve(cases.size());
    tjac.ms.reserve(cases.size());
    int dual_cases = 0;
    for (const Case& c : cases) {
        point.ms.push_back(best_of(reps, [&] { return run_point(c); }));
        march.ms.push_back(best_of(reps, [&] {
            std::array<double, kGmEtaDim> z{};
            return point_march(c, z) ? z[0] : 0.0;
        }));
        taylor.ms.push_back(best_of(reps, [&] { return run_taylor(c); }));
        if (c.dual_ok) {
            ++dual_cases;
            tjac.ms.push_back(best_of(reps, [&] { return run_taylor_jac(c); }));
        }
        point.checksum += run_point(c);
        std::array<double, kGmEtaDim> z{};
        if (point_march(c, z)) march.checksum += z[0];
        taylor.checksum += run_taylor(c);
        if (c.dual_ok) tjac.checksum += run_taylor_jac(c);
    }

    double med, p90, p99;
    std::printf("GM equation-derived Taylor benchmark\n");
    std::printf("cases                 %zu/%s  reps=%d  step=0.01\n",
                cases.size(), path, reps);
    percentile(point.ms, med, p90, p99);
    std::printf("point value-only      median %.4f ms  p90 %.4f  p99 %.4f\n", med, p90, p99);
    percentile(march.ms, med, p90, p99);
    std::printf("16-step RK4 value     median %.4f ms  p90 %.4f  p99 %.4f\n", med, p90, p99);
    percentile(taylor.ms, med, p90, p99);
    std::printf("Taylor-10 value-only  median %.4f ms  p90 %.4f  p99 %.4f\n", med, p90, p99);
    const double taylor_med = med;
    double point_med = 0.0, point_p90 = 0.0, point_p99 = 0.0;
    percentile(point.ms, point_med, point_p90, point_p99);
    if (!tjac.ms.empty()) {
        percentile(tjac.ms, med, p90, p99);
        std::printf("Taylor-3 + 5-dual     median %.4f ms  p90 %.4f  p99 %.4f  cases %d\n",
                    med, p90, p99, dual_cases);
    }
    std::printf("Taylor/point          %.3fx  (median)\n",
                taylor.ms.empty() ? 0.0 : taylor_med / std::max(1e-300, point_med));
    std::printf("checksum              %.12g %.12g %.12g %.12g\n",
                point.checksum, march.checksum, taylor.checksum, tjac.checksum);
    return 0;
}

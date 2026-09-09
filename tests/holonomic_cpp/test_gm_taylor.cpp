// Equation-derived Gauss--Manin connection and Taylor transport checks.
// This harness deliberately exercises gm_connection.hpp directly; the
// existing K-rule / Chebyshev packet path is not included in these checks.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "lcbinint/magnification/holonomic/gm_taylor_transport.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"

using namespace lcbinint::holonomic;

namespace {

struct Geom {
    const char* name;
    LensParams p;
};

const std::vector<Geom> kGeoms = {
    {"memo-s15", {0.2, 1.0 / 7.0, 0.125, 0.5, 1.2, true}},
    {"resonant", {0.05, 0.02, 0.05, 0.3, 0.9, true}},
    {"close", {0.4, -0.05, 0.09, 0.8, 0.55, true}},
    {"planet-wide", {1.4, 0.10, 0.03, 1e-3, 2.5, true}},
    {"caustic-xing", {0.10, 0.03, 0.02, 0.2, 1.15, true}},
    {"near-full", {0.02, 0.01, 0.03, 0.5, 0.6, true}},
    {"plan15-raw", {0.2, 1.0 / 7.0, 0.125, 0.5, 1.2, false}},
};

int g_fail = 0;

void require(bool condition, const char* what) {
    if (!condition) {
        ++g_fail;
        std::fprintf(stderr, "  FAIL %s\n", what);
    }
}

PrimaryFrame bump(PrimaryFrame p, int j, double h) {
    switch (j) {
        case 0: p.X += h; break;
        case 1: p.Y += h; break;
        case 2: p.rho += h; break;
        case 3: p.m0 += h; break;
        case 4: p.a += h; break;
    }
    return p;
}

bool rhs(const std::array<double, kGmEtaDim>& z, double R,
         const PrimaryFrame& pf, std::array<double, kGmEtaDim>& out) {
    const GmConnectionPoint c = gm_connection_at(R, pf);
    if (!c.ok) return false;
    out.fill(0.0);
    for (int i = 0; i < kGmEtaDim; ++i)
        for (int j = 0; j < kGmEtaDim; ++j) out[i] += c.C[i][j] * z[j];
    return true;
}

bool rk4(const std::array<double, kGmEtaDim>& z0, double R0, double h,
         int nstep, const PrimaryFrame& pf,
         std::array<double, kGmEtaDim>& out) {
    out = z0;
    const double dr = h / nstep;
    for (int step = 0; step < nstep; ++step) {
        const double R = R0 + step * dr;
        std::array<double, kGmEtaDim> k1{}, k2{}, k3{}, k4{}, z{};
        if (!rhs(out, R, pf, k1)) return false;
        for (int i = 0; i < kGmEtaDim; ++i) z[i] = out[i] + 0.5 * dr * k1[i];
        if (!rhs(z, R + 0.5 * dr, pf, k2)) return false;
        for (int i = 0; i < kGmEtaDim; ++i) z[i] = out[i] + 0.5 * dr * k2[i];
        if (!rhs(z, R + 0.5 * dr, pf, k3)) return false;
        for (int i = 0; i < kGmEtaDim; ++i) z[i] = out[i] + dr * k3[i];
        if (!rhs(z, R + dr, pf, k4)) return false;
        for (int i = 0; i < kGmEtaDim; ++i)
            out[i] += dr * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    return true;
}

void test_point_and_taylor() {
    int used = 0;
    int accepted = 0;
    int dual_checked = 0;
    int double_checked = 0;
    int rejected = 0;
    int dual_rejected = 0;
    double worst_identity = 0.0;
    double worst_transport = 0.0;
    double worst_dual = 0.0;
    double worst_double = 0.0;
    double worst_point_jet = 0.0;

    for (const Geom& g : kGeoms) {
        const PrimaryFrame pf = PrimaryFrame::from(g.p);
        double R = 0.0;
        GmConnectionPoint point;
        for (double candidate : {0.30, 0.55, 0.80, 1.05, 1.35, 1.80}) {
            point = gm_connection_at(candidate, pf);
            if (point.ok && point.matrix_pivot_rel > 1e-12) {
                R = candidate;
                break;
            }
        }
        if (!point.ok || R == 0.0) {
            std::fprintf(stderr, "  SKIP %s (no benign square-free radius)\n", g.name);
            continue;
        }
        ++used;
        require(point.degree_ok && point.squarefree && point.identity_residual < 1e-6,
                "numeric GM point identity / square-free gate");
        worst_identity = std::max(worst_identity, point.identity_residual);

        const auto jet = gm_connection_jet<10, __float128>(R, GmParams<__float128>{
            pf.X, pf.Y, pf.rho, pf.m0, pf.a});
        double c0diff = 0.0;
        for (int i = 0; i < kGmEtaDim; ++i)
            for (int j = 0; j < kGmEtaDim; ++j)
                c0diff = std::max(c0diff, std::fabs((double)jet.C[i][j].c[0] - point.C[i][j]) /
                                            (1.0 + std::fabs(point.C[i][j])));
        worst_point_jet = std::max(worst_point_jet, c0diff);
        if (!jet.ok) {
            ++rejected;
            continue;
        }
        ++accepted;
        require(jet.identity_residual <= 1e-12,
                "order-10 GM Taylor identity");
        worst_identity = std::max(worst_identity, jet.identity_residual);

        std::array<double, kGmEtaDim> seed{};
        for (int i = 0; i < kGmEtaDim; ++i) seed[i] = 0.25 + 0.17 * i;
        std::array<__float128, kGmEtaDim> seed_ld{}, tay{};
        for (int i = 0; i < kGmEtaDim; ++i) seed_ld[i] = seed[i];
        std::array<double, kGmEtaDim> ref{};
        const double h = 0.01;
        const bool tok = gm_taylor_transport(jet, seed_ld, h, tay);
        const bool rok = rk4(seed, R, h, 256, pf, ref);
        require(tok && rok, "GM Taylor and reference RK4 completed");
        if (tok && rok) {
            double geom_transport = 0.0;
            for (int i = 0; i < kGmEtaDim; ++i) {
                const double scale = 1.0 + std::fabs(ref[i]);
                geom_transport = std::max(geom_transport,
                                          (double)(fabsq(tay[i] - ref[i]) / scale));
            }
            worst_transport = std::max(worst_transport, geom_transport);
            std::fprintf(stderr, "  GM transport %-12s R=%.3f err=%.3e id=%.3e pivot=%.3e\n",
                         g.name, R, geom_transport, jet.identity_residual,
                         jet.matrix_pivot_rel);

            const auto d3 = gm_connection_jet<3, double>(R,
                GmParams<double>{pf.X, pf.Y, pf.rho, pf.m0, pf.a});
            std::array<double, kGmEtaDim> d3out{};
            if (d3.ok && gm_taylor_transport(d3, seed, h, d3out)) {
                ++double_checked;
                for (int i = 0; i < kGmEtaDim; ++i)
                    worst_double = std::max(worst_double,
                        std::fabs(d3out[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
            }
        }

        const auto djet = gm_connection_jet_dual<3>(R, pf);
        if (djet.ok) {
            ++dual_checked;
            for (int par = 0; par < 5; ++par) {
                const double base = par == 0 ? pf.X : par == 1 ? pf.Y :
                                    par == 2 ? pf.rho : par == 3 ? pf.m0 : pf.a;
                const double eps = 2e-6 * std::max(1.0, std::fabs(base));
                const GmConnectionPoint plus = gm_connection_at(R, bump(pf, par, eps));
                const GmConnectionPoint minus = gm_connection_at(R, bump(pf, par, -eps));
                if (!plus.ok || !minus.ok) continue;
                double par_worst = 0.0;
                for (int i = 0; i < kGmEtaDim; ++i)
                    for (int j = 0; j < kGmEtaDim; ++j) {
                        const double fd = (plus.C[i][j] - minus.C[i][j]) / (2.0 * eps);
                        const double got = djet.C[i][j].c[0].deriv[par];
                        const double scale = 1.0 + std::max(std::fabs(fd), std::fabs(got));
                        par_worst = std::max(par_worst, std::fabs(fd - got) / scale);
                    }
                worst_dual = std::max(worst_dual, par_worst);
            }
        } else ++dual_rejected;
    }

    std::fprintf(stderr, "  benign geometries       %d/%zu\n", used, kGeoms.size());
    std::fprintf(stderr, "  Taylor accepted         %d/%d\n", accepted, used);
    std::fprintf(stderr, "  Taylor rejected         %d\n", rejected);
    std::fprintf(stderr, "  dual checked/rejected   %d/%d\n", dual_checked, dual_rejected);
    std::fprintf(stderr, "  point/Taylor C(0) diff  %.3e\n", worst_point_jet);
    std::fprintf(stderr, "  worst identity residual %.3e\n", worst_identity);
    std::fprintf(stderr, "  worst Taylor vs RK4     %.3e\n", worst_transport);
    std::fprintf(stderr, "  double Order-3 cells    %d  worst vs RK4 %.3e\n",
                 double_checked, worst_double);
    if (worst_double >= 2e-4)
        std::fprintf(stderr, "  double Order-3 decision  REJECT (precision gate)\n");
    std::fprintf(stderr, "  worst dual vs FD        %.3e\n", worst_dual);
    require(used >= 4, "enough benign geometries exercised");
    require(accepted >= 4, "enough well-conditioned GM Taylor cells accepted");
    require(worst_point_jet < 1e-6, "point and Taylor-centre connections agree");
    require(worst_transport < 2e-8, "order-10 Taylor agrees with RK4 on short cell");
    require(double_checked >= 2, "enough double Order-3 GM cells checked");
    require(dual_checked >= 2, "enough well-conditioned GM dual cells checked");
    require(worst_dual < 2e-4, "GM dual derivatives agree with central differences");
}

void test_fail_closed() {
    // At R=1 this choice makes T proportional to (1+i t)^2 and, with
    // rho=0, Q contains repeated factors.  The leading coefficient remains
    // present, so this specifically exercises the Q_t inverse gate.
    const GmParams<double> repeated{0.0, 0.0, 0.0, 0.5, 1.0};
    const auto r = gm_connection_jet<0, double>(1.0, repeated);
    require(r.degree_ok && !r.squarefree && !r.ok,
            "repeated-root GM family fails closed");

    // All terms vanish at this point, so deg_t Q drops below eight.
    const GmParams<double> degree_drop{0.0, 0.0, 0.0, 0.0, 0.0};
    const auto d = gm_connection_jet<0, double>(1.0, degree_drop);
    require(!d.degree_ok && !d.ok, "degree-drop GM family fails closed");
}

}  // namespace

int main() {
    test_point_and_taylor();
    test_fail_closed();
    std::fprintf(stderr, "\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}

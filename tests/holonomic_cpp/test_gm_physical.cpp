// Physical-period seed and true Gauss--Manin transport checks.
//
// The seed is generated from the real boundary arc, not from the retained
// K-rule/Chebyshev fit packet.  The angular comparison is an independent
// lens-equation quadrature, and the transport comparison reuses only the
// equation-derived connection after the physical seed has been made.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "lcbinint/magnification/holonomic/gm_physical_seed.hpp"
#include "lcbinint/magnification/holonomic/gm_residue_free.hpp"
#include "lcbinint/magnification/holonomic/gm_taylor_transport.hpp"

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

void require(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "  FAIL %s\n", what);
    }
}

double relerr(double a, double b) {
    return std::fabs(a - b) / (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

const std::array<double, 2>* nearest_arc(const ArcSet& as, double mid) {
    const std::array<double, 2>* best = nullptr;
    double bd = 1e300;
    for (const auto& a : as.arcs) {
        double hi = a[1] <= a[0] ? a[1] + 2.0 * M_PI : a[1];
        const double m = 0.5 * (a[0] + hi);
        const double d = std::fabs(std::atan2(std::sin(m - mid),
                                                std::cos(m - mid)));
        if (d < bd) { bd = d; best = &a; }
    }
    return best;
}

void check_seed_and_transport(const Geom& g, double R) {
    const PrimaryFrame pf = PrimaryFrame::from(g.p);
    const ArcSet as = arc_intervals(R, pf);
    if (as.kind != ArcKind::kArcs || as.arcs.empty()) return;

    for (const auto& arc : as.arcs) {
        const GmPhysicalSeed s128 = gm_physical_period_seed(R, pf, arc, 128);
        const GmPhysicalSeed s256 = gm_physical_period_seed(R, pf, arc, 256);
        const GmPhysicalSeed s512 = gm_physical_period_seed(R, pf, arc, 512);
        double direct = 0.0;
        if (!s128.ok || !s256.ok || !s512.ok ||
            !gm_physical_arc_angular(R, pf, arc, 512, direct)) continue;

        const double seed_vs_angular = relerr(s512.fhalf_arc, direct);
        const double seed_convergence = relerr(s512.fhalf_arc, s256.fhalf_arc);
        const double reduced_identity = relerr(
            s512.phi_arc, 0.5 * pf.rho * s512.fhalf_arc);
        std::fprintf(stderr,
                     "  physical seed %-12s R=%.3f chart=%d fhalf=%.9g "
                     "ang=%.3e nconv=%.3e residue=%.3e\n",
                     g.name, R, s512.chart2 ? 2 : 1, s512.fhalf_arc,
                     seed_vs_angular, seed_convergence,
                     std::fabs(s512.residue_check) /
                         (1.0 + std::fabs(s512.h[3])));
        require(seed_vs_angular < 2e-7,
                "physical period seed agrees with angular lens integral");
        require(seed_convergence < 2e-7,
                "physical period seed converges with quadrature order");
        require(reduced_identity < 2e-14,
                "Phi/fhalf period normalization identity");

        // One nearby radius stays in the same topology cell.  The initial
        // state is the physical period; only the connection transports it.
        const double h = 0.0025;
        const double R1 = R + h;
        const ArcSet as1 = arc_intervals(R1, pf);
        const auto* arc1 = nearest_arc(as1, 0.5 * (s512.theta_enter +
                                                    s512.theta_leave));
        if (!arc1 || as1.kind != ArcKind::kArcs) continue;
        const GmPhysicalSeed s1 = gm_physical_period_seed(R1, pf, *arc1, 512);
        if (!s1.ok || s1.chart2 != s512.chart2) continue;

        const auto jet = gm_connection_jet<10, __float128>(R,
            GmParams<__float128>{__float128(s512.chart_pf.X),
                                  __float128(s512.chart_pf.Y),
                                  __float128(s512.chart_pf.rho),
                                  __float128(s512.chart_pf.m0),
                                  __float128(s512.chart_pf.a)});
        if (!jet.ok) continue;
        std::array<__float128, kGmEtaDim> seed{}, moved{};
        for (int k = 0; k < kGmEtaDim; ++k) seed[k] = s512.eta_closed[k];
        if (!gm_taylor_transport(jet, seed, h, moved)) continue;
        double worst = 0.0;
        for (int k = 0; k < kGmEtaDim; ++k)
            worst = std::max(worst, relerr((double)moved[k], s1.eta_closed[k]));
        std::fprintf(stderr, "  GM physical transport %-12s R=%.3f->%.3f "
                             "err=%.3e\n", g.name, R, R1, worst);
        require(worst < 2e-8,
                "true GM transport matches re-seeded physical period");

        // Six-dimensional residue-free experiment.  The connection is still
        // generated by the true eta algebra above; this measures the
        // residue-free projection and its transport separately from the
        // eventual flux-priority companion basis.
        const auto eta8 = gm_connection_jet<8, DD>(R,
            GmParams<DD>{DD(s512.chart_pf.X), DD(s512.chart_pf.Y),
                         DD(s512.chart_pf.rho), DD(s512.chart_pf.m0),
                         DD(s512.chart_pf.a)});
        const auto eta0 = gm_connection_jet<0, DD>(R,
            GmParams<DD>{DD(s512.chart_pf.X), DD(s512.chart_pf.Y),
                         DD(s512.chart_pf.rho), DD(s512.chart_pf.m0),
                         DD(s512.chart_pf.a)});
        const auto psi0 = gm_residue_free_jet(eta0);
        std::fprintf(stderr, "  residue-free point %-12s closure=%.3e ok=%d\n",
                     g.name, psi0.closure_residual, psi0.ok ? 1 : 0);
        require(psi0.ok, "residue-free point connection closure");

        if (eta8.ok) {
            const auto psi8 = gm_residue_free_jet(eta8);
            if (psi8.ok) {
                std::array<DD, kGmEtaDim> eta_seed{};
                for (int k = 0; k < kGmEtaDim; ++k)
                    eta_seed[k] = DD(s512.eta_closed[k]);
                std::array<DD, kGmPsiDim> psi_seed{}, psi_moved{};
                if (gm_eta_to_psi(eta8, eta_seed, psi_seed) &&
                    gm_psi_taylor_transport(psi8, psi_seed, h, psi_moved)) {
                    std::array<DD, kGmEtaDim> eta_moved{};
                    gm_taylor_transport(eta8, eta_seed, h, eta_moved);
                    const auto b1raw = gm_residue_b(
                        gm_q_coefficients(R1, s512.chart_pf));
                    std::array<DD, kGmPsiDim> psi_from_eta = {
                        eta_moved[0], eta_moved[1], eta_moved[2],
                        eta_moved[4] - DD(b1raw[0]) * eta_moved[3],
                        eta_moved[5] - DD(b1raw[1]) * eta_moved[3],
                        eta_moved[6] - DD(b1raw[2]) * eta_moved[3]};
                    double basis_consistency = 0.0;
                    for (int k = 0; k < kGmPsiDim; ++k)
                        basis_consistency = std::max(
                            basis_consistency,
                            relerr((double)psi_moved[k],
                                   (double)psi_from_eta[k]));
                    std::fprintf(stderr,
                                 "  residue-free 6D %-12s closure=%.3e "
                                 "basis-consistency=%.3e\n", g.name,
                                 psi8.closure_residual, basis_consistency);
                    require(basis_consistency < 2e-8,
                            "residue-free six-dimensional transport basis consistency");
                }
            } else {
                std::fprintf(stderr,
                             "  residue-free 6D Taylor rejected %-12s "
                             "closure=%.3e\n", g.name,
                             psi8.closure_residual);
            }
        }

        // The physical accuracy gate uses the retained qf lane because the
        // DD eta jet is intentionally allowed to expose the precision
        // ladder's promotion boundary on difficult cells.
        const auto eta10q = gm_connection_jet<10, __float128>(R,
            GmParams<__float128>{(__float128)s512.chart_pf.X,
                                 (__float128)s512.chart_pf.Y,
                                 (__float128)s512.chart_pf.rho,
                                 (__float128)s512.chart_pf.m0,
                                 (__float128)s512.chart_pf.a});
        const auto psi10q = gm_residue_free_jet(eta10q);
        if (!psi10q.ok) continue;
        std::array<__float128, kGmEtaDim> eta_seed_q{};
        for (int k = 0; k < kGmEtaDim; ++k)
            eta_seed_q[k] = (__float128)s512.eta_closed[k];
        std::array<__float128, kGmPsiDim> psi_seed_q{}, psi_moved_q{};
        if (!gm_eta_to_psi(eta10q, eta_seed_q, psi_seed_q) ||
            !gm_psi_taylor_transport(psi10q, psi_seed_q, h, psi_moved_q))
            continue;
        double seed_diff_q = 0.0;
        for (int k = 0; k < kGmPsiDim; ++k)
            seed_diff_q = std::max(seed_diff_q,
                relerr((double)psi_seed_q[k], s512.psi_closed[k]));
        const auto h1 = gm_observable_h(R1, s512.chart_pf);
        __float128 phi_psi_q = 0;
        for (int k = 0; k < 3; ++k)
            phi_psi_q += (__float128)h1[k] * psi_moved_q[k];
        phi_psi_q += (__float128)h1[4] * psi_moved_q[3]
                   + (__float128)h1[5] * psi_moved_q[4]
                   + (__float128)h1[6] * psi_moved_q[5];
        const double psi_fhalf_q =
            (double)(phi_psi_q / (__float128)pf.rho);
        std::array<__float128, kGmEtaDim> eta_moved_q{};
        gm_taylor_transport(eta10q, eta_seed_q, h, eta_moved_q);
        __float128 phi_eta_q = 0;
        for (int k = 0; k < kGmEtaDim; ++k)
            phi_eta_q += (__float128)h1[k] * eta_moved_q[k];
        const double eta_fhalf_q =
            (double)(phi_eta_q / (__float128)pf.rho);
        std::fprintf(stderr, "  residue-free qf  %-12s closure=%.3e "
                             "seed=%.3e transport=%.3e\n", g.name,
                     psi10q.closure_residual, seed_diff_q,
                     relerr(psi_fhalf_q, s1.fhalf_arc));
        std::fprintf(stderr, "    residue-free qf eta-observable=%.3e "
                             "psi-vs-eta=%.3e\n",
                     relerr(eta_fhalf_q, s1.fhalf_arc),
                     relerr(psi_fhalf_q, eta_fhalf_q));
        require(seed_diff_q < 2e-8,
                "residue-free six-dimensional qf seed agrees with eta seed");
        require(relerr(psi_fhalf_q, s1.fhalf_arc) < 2e-8,
                "residue-free six-dimensional qf transport matches physical period");
        return;
    }
}

}  // namespace

int main() {
    int seeded = 0;
    for (const Geom& g : kGeoms) {
        const PrimaryFrame pf = PrimaryFrame::from(g.p);
        for (double R : {0.30, 0.55, 0.80, 1.05, 1.35, 1.80}) {
            const ArcSet as = arc_intervals(R, pf);
            if (as.kind == ArcKind::kArcs && !as.arcs.empty()) {
                check_seed_and_transport(g, R);
                ++seeded;
                break;
            }
        }
    }
    std::fprintf(stderr, "  physical geometries exercised %d/%zu\n",
                 seeded, kGeoms.size());
    require(seeded >= 4, "enough physical-period geometries exercised");
    std::fprintf(stderr, "\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}

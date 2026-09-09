// ATPT holonomic solver -- holonomic_transport.hpp correctness.
//
//   A. v_times_K value  vs  the incumbent 64-point GC-1 angular sqrt(phi)
//      sweep, per inside arc, over the 6 rescue-spike geometries x their
//      cell plan x a few radial nodes.  Identity: R * int_arc sqrt(phi)
//      dtheta = (2/rho) Phi_arc = v K.
//   B. v_times_K_jac analytic d/dP_j[v K] (primary-frame params X,Y,rho,m0,a)
//      vs a central finite difference: perturb the frame, re-cold-solve the
//      arc, match by nearest midpoint, recompute v_times_K.  Tests the whole
//      chain incl. arc_pair_jac's endpoint IFT sensitivity.
//   C. epoch chain parity: epoch_jacobian(u=0.5) with the transport forced ON
//      (V2) vs OFF (V0) -- mu, grad_mu (user params xs,ys,rho,q,a), and
//      status must all match.  The transport replaces the 64-pt angular
//      sqrt(phi) sweep for the engaged arcs with the deflated x-chart K-rule;
//      Part A/B already pin the per-arc value + Jacobian to ~1e-12 / ~5e-6,
//      so the whole-epoch chain (incl. internal_to_user_jac and the -2 mu/rho
//      term) must be a no-op to working precision.  (An absolute grad_mu-vs-FD
//      check is the incumbent solver's own; near rho->0 with tiny q it is
//      dominated by the 2 mu/rho cancellation and is not a transport concern.)
//
// Build (isolated):
//   cmake --build build-holonomic-m7 --target test_holonomic_transport
//   ./build-holonomic-m7/test_holonomic_transport

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/holonomic_transport.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/phi.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

using namespace lcbinint::holonomic;

namespace {

int g_fail = 0;

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
    // raw (non-barycentric) frame: X = xs (no + m1 a shift) puts arc endpoints
    // near theta = pi in cells 3-5 -- the wide-arc / large-|t| regime that the
    // 8-vs-16-node K-rule convergence gate must fail closed on.
    {"plan15-raw", {0.2, 1.0 / 7.0, 0.125, 0.5, 1.2, false}},
};

// sorted real t-roots of the ascending quartic (same filter thetas_from_complex
// uses), then consecutive-pair inside arcs (phi > 0 at the t-midpoint).
std::vector<std::array<double, 2>> inside_pairs(const std::array<double, 5>& pc,
                                               double R,
                                               const PrimaryFrame& pf) {
    double c[5];
    int deg = quartic_descending(pc, c);
    std::vector<std::array<double, 2>> out;
    if (deg <= 0) return out;
    auto z = aberth<double>(c, deg, 60);
    std::vector<double> tr;
    for (auto& zz : z)
        if (std::fabs(zz.im) <= 1e-8 * (1.0 + std::fabs(zz.re)))
            tr.push_back(zz.re);
    std::sort(tr.begin(), tr.end());
    for (size_t i = 0; i + 1 < tr.size(); ++i) {
        const double tm = 0.5 * (tr[i] + tr[i + 1]);
        const double th = std::fmod(2.0 * std::atan(tm), kTwoPi);
        if (phi_lens(R, th < 0 ? th + kTwoPi : th, pf) > 0.0)
            out.push_back({tr[i], tr[i + 1]});
    }
    return out;
}

// incumbent: R * int_arc sqrt(phi) dtheta on the 64-pt GC-1 rule (the exact
// quantity radius_terms accumulates into rt.fh for one arc).
double sweep_arc(double R, const PrimaryFrame& pf, double t_lo, double t_hi) {
    const double te = std::fmod(2.0 * std::atan(t_lo), kTwoPi);
    const double tl0 = std::fmod(2.0 * std::atan(t_hi), kTwoPi);
    double a = te < 0 ? te + kTwoPi : te;
    double b = tl0 < 0 ? tl0 + kTwoPi : tl0;
    if (b <= a) b += kTwoPi;
    const double half = 0.5 * (b - a), mid = 0.5 * (a + b);
    const auto& AR = ang_rule();
    double acc = 0.0;
    for (int k = 0; k < 64; ++k) {
        const double thn = mid + half * AR.x[k];
        const double ph = phi_lens(R, thn, pf);
        if (ph <= 0.0) continue;
        acc += AR.w[k] * std::sqrt(ph);
    }
    return R * half * acc;
}

// (m,v) + endpoint-IFT sensitivities for the arc bounded by t_lo,t_hi, exactly
// as radius_terms would form them (phi_val_dtheta polish + phi_val_dP IFT).
bool arc_pair_from_frame(double R, const PrimaryFrame& pf, double t_lo,
                         double t_hi, ArcPairJac& ap) {
    auto polish = [&](double t0, double& theta, double& dth) {
        double th = std::fmod(2.0 * std::atan(t0), kTwoPi);
        if (th < 0) th += kTwoPi;
        for (int i = 0; i < 8; ++i) {
            PhiValDtheta g = phi_val_dtheta(R, th, pf);
            if (std::fabs(g.dphi_dtheta) < 1e-14) return false;
            th -= g.phi / g.dphi_dtheta;
        }
        PhiValDtheta g = phi_val_dtheta(R, th, pf);
        theta = th;
        dth = g.dphi_dtheta;
        return std::fabs(dth) > 1e-14;
    };
    double te, tl, td, tdl;
    if (!polish(t_lo, te, td) || !polish(t_hi, tl, tdl)) return false;
    if (tl <= te) tl += kTwoPi;
    PhiValDP ge = phi_val_dP(R, te, pf);
    PhiValDP gl = phi_val_dP(R, tl, pf);
    std::array<double, 5> dte{}, dtl{};
    for (int j = 0; j < 5; ++j) {
        dte[j] = -ge.dP[j] / td;
        dtl[j] = -gl.dP[j] / tdl;
    }
    ap = arc_pair_jac(te, tl, dte, dtl);
    return ap.ok;
}

// v_times_K for the arc nearest (in m) to m_ref, after perturbing the frame.
bool vk_perturbed(double R, const PrimaryFrame& pf, double m_ref, double& vk) {
    QuarticCoeffs pc = boundary_quartic(R, pf);
    auto prs = inside_pairs(pc.p, R, pf);
    if (prs.empty()) return false;
    double best = 1e300;
    std::array<double, 2> pick{};
    for (auto& pr : prs) {
        double m = 0.5 * (pr[0] + pr[1]);
        if (std::fabs(m - m_ref) < best) {
            best = std::fabs(m - m_ref);
            pick = pr;
        }
    }
    ArcPairJac ap;
    if (!arc_pair_from_frame(R, pf, pick[0], pick[1], ap)) return false;
    VKValue r = v_times_K(ap.m, ap.v, R, pf, boundary_quartic(R, pf).p);
    if (!r.ok) return false;
    vk = r.vK;
    return true;
}

PrimaryFrame bump(PrimaryFrame pf, int j, double h) {
    switch (j) {
        case 0: pf.X += h; break;
        case 1: pf.Y += h; break;
        case 2: pf.rho += h; break;
        case 3: pf.m0 += h; break;
        case 4: pf.a += h; break;
    }
    return pf;
}

void part_A_value() {
    std::fprintf(stderr, "== A. v_times_K  vs  64-pt angular sweep ==\n");
    double worst = 0.0;
    long engaged = 0, arcs = 0;
    for (const auto& g : kGeoms) {
        PrimaryFrame pf = PrimaryFrame::from(g.p);
        TopologyResult topo = classify_cells(pf);
        double gworst = 0.0;
        long geng = 0, garc = 0;
        for (const auto& c : topo.cells) {
            if (c.kind != ArcKind::kArcs) continue;
            const double lo = c.r_lo, hi = c.r_hi;
            if (hi - lo < 3e-4) continue;
            for (double frac : {0.15, 0.35, 0.5, 0.65, 0.85}) {
                const double R = lo + frac * (hi - lo);
                QuarticCoeffs pc = boundary_quartic(R, pf);
                for (auto& pr : inside_pairs(pc.p, R, pf)) {
                    ++garc;
                    ArcPairJac ap;
                    if (!arc_pair_from_frame(R, pf, pr[0], pr[1], ap)) continue;
                    VKValue r = v_times_K(ap.m, ap.v, R, pf, pc.p);
                    if (!r.ok) continue;
                    ++geng;
                    const double ref = sweep_arc(R, pf, pr[0], pr[1]);
                    const double rel =
                        std::fabs(r.vK - ref) / (std::fabs(ref) + 1e-300);
                    gworst = std::max(gworst, rel);
                }
            }
        }
        worst = std::max(worst, gworst);
        engaged += geng;
        arcs += garc;
        std::fprintf(stderr,
                     "  %-13s arcs=%3ld engaged=%3ld (%3.0f%%)  worst rel=%.2e\n",
                     g.name, garc, geng, garc ? 100.0 * geng / garc : 0.0,
                     gworst);
    }
    std::fprintf(stderr, "  --> worst rel err %.2e over %ld/%ld arcs engaged\n",
                 worst, engaged, arcs);
    if (!(worst < 5e-6)) {
        ++g_fail;
        std::fprintf(stderr, "  FAIL: value rel err %.2e >= 5e-6\n", worst);
    }
}

void part_B_jacobian() {
    std::fprintf(stderr,
                 "\n== B. analytic d/dP_j[v K]  vs  central finite diff ==\n");
    double worst = 0.0;
    long checked = 0;
    for (const auto& g : kGeoms) {
        PrimaryFrame pf = PrimaryFrame::from(g.p);
        TopologyResult topo = classify_cells(pf);
        double gworst = 0.0;
        for (const auto& c : topo.cells) {
            if (c.kind != ArcKind::kArcs) continue;
            if (c.r_hi - c.r_lo < 3e-4) continue;
            for (double frac : {0.3, 0.5, 0.7}) {
                const double R = c.r_lo + frac * (c.r_hi - c.r_lo);
                QuarticCoeffs pc = boundary_quartic(R, pf);
                QuarticParamJac dpc = boundary_quartic_dp(R, pf);
                for (auto& pr : inside_pairs(pc.p, R, pf)) {
                    ArcPairJac ap;
                    if (!arc_pair_from_frame(R, pf, pr[0], pr[1], ap)) continue;
                    VKJacobian an = v_times_K_jac(ap.m, ap.v, ap.dm, ap.dv, R,
                                                  pf, pc.p, dpc.dp);
                    if (!an.ok) continue;
                    // skip thin arcs: the x-chart node derivative ~ x/(2 sqrt v)
                    // and the incumbent sweep are both unreliable there
                    // (radius_terms already flags rt.reliable = false).
                    if (ap.v < 1e-6) continue;
                    for (int j = 0; j < 5; ++j) {
                        const double base =
                            (j == 2) ? pf.rho
                                     : std::fabs(j == 4 ? pf.a : 1.0);
                        const double h = 1e-6 * std::max(base, 1.0);
                        double vp, vm;
                        if (!vk_perturbed(R, bump(pf, j, h), ap.m, vp)) continue;
                        if (!vk_perturbed(R, bump(pf, j, -h), ap.m, vm)) continue;
                        const double fd = (vp - vm) / (2.0 * h);
                        const double rel = std::fabs(an.dvK[j] - fd) /
                                           (std::fabs(fd) + 1e-12);
                        gworst = std::max(gworst, rel);
                        ++checked;
                    }
                }
            }
        }
        worst = std::max(worst, gworst);
        std::fprintf(stderr, "  %-13s worst rel=%.2e\n", g.name, gworst);
    }
    std::fprintf(stderr, "  --> worst rel err %.2e over %ld component checks\n",
                 worst, checked);
    if (!(worst < 2e-4)) {
        ++g_fail;
        std::fprintf(stderr, "  FAIL: Jacobian rel err %.2e >= 2e-4\n", worst);
    }
}

void part_C_epoch() {
    std::fprintf(stderr,
                 "\n== C. epoch_jacobian(u=0.5) V2(holo ON) vs V0(OFF) parity ==\n");
    const double u = 0.5;
    // engaged >= 1e-12 relative delta on mu or grad => transport actually ran
    const double kMuTol = 1e-9;
    const double kGradTol = 1e-6;
    double worst_mu = 0.0, worst_grad = 0.0;
    int status_changes = 0;
    for (const auto& g : kGeoms) {
        holo_mv_transport_override() = 0;
        holo_holonomic_transport_override() = 0;
        auto v0 = epoch_jacobian(g.p, u, 64, /*fast=*/false);
        holo_mv_transport_override() = 1;
        holo_holonomic_transport_override() = 1;
        auto v2 = epoch_jacobian(g.p, u, 64, /*fast=*/false);

        double dmu = std::fabs(v0.mu - v2.mu) /
                     (std::fabs(v0.mu) > 0.0 ? std::fabs(v0.mu) : 1.0);
        double dgrad = 0.0;
        int jw = -1;
        for (int j = 0; j < 5; ++j) {
            double base = std::fabs(v0.grad_mu[j]);
            double d = std::fabs(v0.grad_mu[j] - v2.grad_mu[j]) /
                       (base > 1e-10 ? base : 1.0);
            if (d > dgrad) { dgrad = d; jw = j; }
        }
        bool schg = (v0.status != v2.status);
        status_changes += schg;
        worst_mu = std::max(worst_mu, dmu);
        worst_grad = std::max(worst_grad, dgrad);
        std::fprintf(stderr,
                     "  %-13s mu=%.10g  dmu=%.2e  dgrad=%.2e (j=%d)%s\n",
                     g.name, v0.mu, dmu, dgrad, jw,
                     schg ? "  STATUS CHANGE" : "");
    }
    holo_mv_transport_override() = -1;
    holo_holonomic_transport_override() = -1;
    std::fprintf(stderr,
                 "  --> worst dmu %.2e (tol %.0e)  worst dgrad %.2e (tol %.0e)"
                 "  status changes %d\n",
                 worst_mu, kMuTol, worst_grad, kGradTol, status_changes);
    if (!(worst_mu < kMuTol) || !(worst_grad < kGradTol) ||
        status_changes != 0) {
        ++g_fail;
        std::fprintf(stderr, "  FAIL: V2 not at parity with V0\n");
    }
}

}  // namespace

int main() {
    part_A_value();
    part_B_jacobian();
    part_C_epoch();
    std::fprintf(stderr, "\n%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED",
                 g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

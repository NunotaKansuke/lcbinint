#pragma once

// ATPT holonomic solver (M7) -- one radius: value + derivative integrands.
// Ports python/lcbinint/holonomic_ref/jacobian.py:
//   polish_endpoint, _real_root_thetas, arc_intervals, _grid_intervals,
//   _full_circle_terms, radius_terms
// and topology.arcs_at (grid fallback / cell kind).
//
//   f0    = R * sum_arcs (theta_leave - theta_enter)
//   fh    = R * sum_arcs int_arc sqrt(phi) dtheta        (GC-1(64) angular)
//   df0   = R * sum_arcs ( d theta_leave/dP - d theta_enter/dP )   (IFT-theta)
//   dfh   = R * sum_arcs int_arc  dphi/dP / (2 sqrt phi) dtheta
//
// `reliable` is false on a near-tangency, a degenerate quartic, or a
// full-circle radius -- the caller then marks the whole Jacobian
// GRADIENT_UNRELIABLE (fail closed).

#include <array>
#include <cmath>
#include <vector>

#include "lcbinint/magnification/holonomic/angular_rule.hpp"
#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/phi.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"

namespace lcbinint::holonomic {

constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kTanRel = 1e-4;       // jacobian._TAN_REL
constexpr double kRootImRel = 1e-8;    // jacobian._ROOT_IM_REL
constexpr double kP4DegenRel = 1e-11;  // jacobian._P4_DEGEN_REL

struct RadiusTerms {
    double f0 = 0.0;
    double fh = 0.0;
    std::array<double, 5> df0{};
    std::array<double, 5> dfh{};
    bool reliable = true;
};

// ---- Newton on phi(R, theta; P) = 0 -----------------------------------
struct PolishResult {
    double theta;
    double dphi_dtheta;
    bool reliable;
};
inline PolishResult polish_endpoint(double R, double theta,
                                    const PrimaryFrame& pf, int iters = 6) {
    double th = theta, dth = 1.0;
    for (int i = 0; i < iters; ++i) {
        PhiValDtheta g = phi_val_dtheta(R, th, pf);
        dth = g.dphi_dtheta;
        if (std::fabs(dth) < 1e-13) return {th, dth, false};
        double step = g.phi / dth;
        th -= step;
        if (std::fabs(step) < 1e-15) break;
    }
    PhiValDtheta g = phi_val_dtheta(R, th, pf);
    return {th, g.dphi_dtheta, std::fabs(g.dphi_dtheta) >= 1e-13};
}

// ---- sorted theta of the real roots of P(t), t = tan(theta/2) --------
// descending, leading-zero-deflated quartic coeffs -> working degree.
inline int quartic_descending(const std::array<double, 5>& pc, double c[5]) {
    c[0] = pc[4]; c[1] = pc[3]; c[2] = pc[2]; c[3] = pc[1]; c[4] = pc[0];
    int deg = 4;
    while (deg > 0 && c[0] == 0.0) {
        for (int i = 0; i < deg; ++i) c[i] = c[i + 1];
        --deg;
    }
    return deg;
}

// complex roots -> sorted, near-double-merged theta list.  Real filter
// matches np.roots + |Im| <= _ROOT_IM_REL (1 + |Re|); merge at 1e-11 in t.
inline std::vector<double> thetas_from_complex(
    const std::vector<Cplx<double>>& z) {
    std::vector<double> out;
    for (const auto& r : z)
        if (std::fabs(r.im) <= kRootImRel * (1.0 + std::fabs(r.re))) {
            double t = std::fmod(2.0 * std::atan(r.re), kTwoPi);
            if (t < 0.0) t += kTwoPi;
            out.push_back(t);
        }
    std::sort(out.begin(), out.end());
    std::vector<double> merged;
    for (double x : out)
        if (merged.empty() || x - merged.back() > 1e-11) merged.push_back(x);
    return merged;
}

inline std::vector<double> real_root_thetas(const std::array<double, 5>& pc) {
    double c[5];
    int deg = quartic_descending(pc, c);
    if (deg <= 0) return {};
    return thetas_from_complex(aberth<double>(c, deg, 40));
}

// Warm-start state for the per-radial-node quartic solve inside one cell.
// The Chebyshev nodes are monotone in R inside a cell and the boundary
// quartic's roots move smoothly with R away from a tangency, so node k
// seeds node k-1's converged roots into a short Aberth pass instead of a
// cold circle start.
//
// The warm result is adopted only when the final Aberth correction is
// <= kWarmStepTol (roots then good well past what the downstream
// 6-iteration polish_endpoint Newton needs -- the quartic roots are arc
// seeds, not final endpoints) AND the real-root count is unchanged from
// the seed.  Any failure -> exact cold 40-iteration solve, identical to
// real_root_thetas().  So an adopted set is never a degraded guess and the
// arc topology can never silently change.  After kColdStreak consecutive
// warm misses the warm path is disabled for the rest of the cell (a moving
// tangency -- stop paying warm+cold).  Reset (valid=false) at each cell
// boundary and at a chart (p4~0) radius.
constexpr int kWarmIters = 20;
constexpr double kWarmStepTol = 1e-7;
constexpr int kColdStreak = 3;

struct QuarticWarm {
    Cplx<double> z[4];
    int deg = 0;
    int n_real = -1;
    bool valid = false;    // z[] holds a usable previous-node root set
    int cold_streak = 0;   // consecutive warm misses -> disable at kColdStreak
    long warm_hits = 0, cold_falls = 0;
};

inline std::vector<double> real_root_thetas_warm(const std::array<double, 5>& pc,
                                                 QuarticWarm& w) {
    double c[5];
    int deg = quartic_descending(pc, c);
    if (deg <= 0) { w.valid = false; return {}; }

    if (w.valid && w.deg == deg && w.cold_streak < kColdStreak) {
        double step = 1.0;
        auto z = aberth<double>(c, deg, kWarmIters, w.z, 0.0, &step);
        if (step <= kWarmStepTol) {
            auto th = thetas_from_complex(z);
            if (w.n_real < 0 || (int)th.size() == w.n_real) {
                for (int i = 0; i < deg; ++i) w.z[i] = z[i];
                w.n_real = (int)th.size();
                w.cold_streak = 0;
                ++w.warm_hits;
                return th;
            }
        }
        ++w.cold_streak;
    }

    // cold path -- identical roots to real_root_thetas()
    auto z = aberth<double>(c, deg, 40);
    auto th = thetas_from_complex(z);
    for (int i = 0; i < deg && i < 4; ++i) w.z[i] = z[i];
    w.deg = deg;
    w.n_real = (int)th.size();
    w.valid = (deg <= 4);
    ++w.cold_falls;
    return th;
}

enum class ArcKind { kArcs, kFull, kEmpty, kDegenerate };
struct ArcSet {
    ArcKind kind;
    std::vector<std::array<double, 2>> arcs;  // (theta_enter, theta_leave)
};

// ---- topology.arcs_at : grid sign-scan + bisection refine ------------
struct GridArcs {
    ArcKind kind;  // kArcs / kFull / kEmpty
    int n_crossings;
    std::vector<std::array<double, 2>> arcs;
};
inline GridArcs arcs_at(double R, const PrimaryFrame& pf, int n_grid = 3072) {
    std::vector<double> val(n_grid);
    bool all_pos = true, all_neg = true;
    for (int i = 0; i < n_grid; ++i) {
        double th = kTwoPi * i / n_grid;
        double v = phi_lens(R, th, pf);
        val[i] = v;
        if (v >= 0.0)
            all_neg = false;
        else
            all_pos = false;
    }
    if (all_pos) return {ArcKind::kFull, 0, {}};
    if (all_neg) return {ArcKind::kEmpty, 0, {}};

    const double step = kTwoPi / n_grid;
    std::vector<double> roots;
    std::vector<char> rising;
    for (int i = 0; i < n_grid; ++i) {
        int j = (i + 1) % n_grid;
        bool pi = val[i] >= 0.0, pj = val[j] >= 0.0;
        if (pi == pj) continue;
        double t0 = kTwoPi * i / n_grid, t1 = t0 + step;
        double f0 = val[i], f1 = (j == 0) ? phi_lens(R, t1, pf) : val[j];
        for (int it = 0; it < 80; ++it) {
            double tm = 0.5 * (t0 + t1);
            double fm = phi_lens(R, tm, pf);
            if ((fm >= 0.0) == (f0 >= 0.0)) {
                t0 = tm;
                f0 = fm;
            } else {
                t1 = tm;
                f1 = fm;
            }
        }
        double r = std::fmod(0.5 * (t0 + t1), kTwoPi);
        if (r < 0.0) r += kTwoPi;
        roots.push_back(r);
        rising.push_back(!pi);  // neg -> pos == entering
        (void)f1;
    }
    // sort by root, keep rising flag aligned
    std::vector<int> order(roots.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return roots[a] < roots[b]; });
    std::vector<double> rs;
    std::vector<char> ri;
    for (int k : order) {
        rs.push_back(roots[k]);
        ri.push_back(rising[k]);
    }
    int n = (int)rs.size();
    if (n % 2 != 0) return {ArcKind::kArcs, n, {}};
    int start = ri.empty() || ri[0] ? 0 : -1;
    std::vector<std::array<double, 2>> arcs;
    for (int k = 0; k < n; k += 2) {
        double te = rs[((start + k) % n + n) % n];
        double tl = rs[((start + k + 1) % n + n) % n];
        arcs.push_back({te, tl});
    }
    std::sort(arcs.begin(), arcs.end(),
              [](const std::array<double, 2>& a, const std::array<double, 2>& b) {
                  return a[0] < b[0];
              });
    return {ArcKind::kArcs, n, arcs};
}

// ---- arc_intervals : from the boundary quartic ----------------------
// `w` (optional): per-cell warm-start state for the quartic root solve.
inline ArcSet arc_intervals(double R, const PrimaryFrame& pf,
                            QuarticWarm* w = nullptr) {
    QuarticCoeffs q = boundary_quartic(R, pf);
    double amax = 0.0;
    for (double c : q.p) amax = std::max(amax, std::fabs(c));
    if (amax == 0.0 || std::fabs(q.p[4]) < kP4DegenRel * amax) {
        if (w) w->valid = false;  // chart radius -> break the warm chain
        return {ArcKind::kDegenerate, {}};
    }
    auto th = w ? real_root_thetas_warm(q.p, *w) : real_root_thetas(q.p);
    if (th.empty())
        return {q.p[4] > 0.0 ? ArcKind::kFull : ArcKind::kEmpty, {}};
    if (th.size() % 2 != 0) return {ArcKind::kDegenerate, {}};
    ArcSet out{ArcKind::kArcs, {}};
    int n = (int)th.size();
    for (int i = 0; i < n; ++i) {
        double lo = th[i];
        double hi = th[(i + 1) % n];
        if (hi <= lo) hi += kTwoPi;
        if (phi_lens(R, 0.5 * (lo + hi), pf) > 0.0) out.arcs.push_back({lo, hi});
    }
    return out;
}

// Cheap cell-topology probe: (kind, crossing count) straight from the
// boundary quartic's real roots -- no 3072-point grid.  Falls back to the
// grid only at a p4~0 (degenerate) probe radius.  Used by classify_cells.
inline GridArcs quartic_topology(double R, const PrimaryFrame& pf) {
    QuarticCoeffs q = boundary_quartic(R, pf);
    double amax = 0.0;
    for (double c : q.p) amax = std::max(amax, std::fabs(c));
    if (amax == 0.0 || std::fabs(q.p[4]) < kP4DegenRel * amax)
        return arcs_at(R, pf, 3072);
    auto th = real_root_thetas(q.p);
    if (th.empty())
        return {q.p[4] > 0.0 ? ArcKind::kFull : ArcKind::kEmpty, 0, {}};
    return {ArcKind::kArcs, (int)th.size(), {}};
}

inline ArcSet grid_intervals(double R, const PrimaryFrame& pf) {
    GridArcs g = arcs_at(R, pf, 4096);
    if (g.kind == ArcKind::kFull) return {ArcKind::kFull, {}};
    if (g.kind == ArcKind::kEmpty) return {ArcKind::kEmpty, {}};
    ArcSet out{ArcKind::kArcs, {}};
    for (auto& arc : g.arcs) {
        double tl = arc[1];
        if (tl <= arc[0]) tl += kTwoPi;
        out.arcs.push_back({arc[0], tl});
    }
    return out;
}

// ---- full phi>0 circle : 256-point periodic rule --------------------
inline RadiusTerms full_circle_terms(double R, const PrimaryFrame& pf) {
    constexpr int M = 256;
    const double dth = kTwoPi / M;
    RadiusTerms rt;
    rt.reliable = false;
    double fh = 0.0;
    std::array<double, 5> dfh{};
    for (int i = 0; i < M; ++i) {
        double t = kTwoPi * i / M;
        PhiValDP g = phi_val_dP(R, t, pf);
        if (g.phi <= 0.0) continue;
        double sq = std::sqrt(g.phi);
        fh += sq;
        for (int j = 0; j < 5; ++j) dfh[j] += g.dP[j] / (2.0 * sq);
    }
    rt.f0 = R * kTwoPi;
    rt.fh = R * dth * fh;
    for (int j = 0; j < 5; ++j) rt.dfh[j] = R * dth * dfh[j];
    return rt;
}

// ---- radius_terms --------------------------------------------------
inline RadiusTerms radius_terms(double R, const PrimaryFrame& pf,
                                double tan_rel = kTanRel,
                                QuarticWarm* w = nullptr) {
    RadiusTerms rt;
    ArcSet as = arc_intervals(R, pf, w);
    if (as.kind == ArcKind::kEmpty) return rt;
    if (as.kind == ArcKind::kFull) return full_circle_terms(R, pf);
    if (as.kind == ArcKind::kDegenerate) {
        ArcSet g = grid_intervals(R, pf);
        rt.reliable = false;
        if (g.kind == ArcKind::kEmpty) return rt;
        if (g.kind == ArcKind::kFull) {
            RadiusTerms f = full_circle_terms(R, pf);
            f.reliable = false;
            return f;
        }
        as = g;
    }

    const double tan_thresh = tan_rel * pf.rho / std::max(R, 1e-9);
    const auto& AR = ang_rule();

    for (const auto& arc : as.arcs) {
        PolishResult pe = polish_endpoint(R, arc[0], pf);
        PolishResult pl = polish_endpoint(R, arc[1], pf);
        double te = pe.theta, tl = pl.theta;
        double td = pe.dphi_dtheta, tdl = pl.dphi_dtheta;
        if (tl <= te) tl += kTwoPi;
        rt.reliable = rt.reliable && pe.reliable && pl.reliable;
        if (std::fabs(td) < tan_thresh || std::fabs(tdl) < tan_thresh)
            rt.reliable = false;

        rt.f0 += R * (tl - te);
        PhiValDP ge = phi_val_dP(R, te, pf);
        PhiValDP gl = phi_val_dP(R, tl, pf);
        for (int j = 0; j < 5; ++j) {
            double dte = (td != 0.0) ? -ge.dP[j] / td : 0.0;
            double dtl = (tdl != 0.0) ? -gl.dP[j] / tdl : 0.0;
            rt.df0[j] += R * (dtl - dte);
        }

        double half = 0.5 * (tl - te);
        double mid = 0.5 * (te + tl);
        double acc_val = 0.0;
        std::array<double, 5> acc_der{};
        for (int k = 0; k < 64; ++k) {
            double thn = mid + half * AR.x[k];
            PhiValDP g = phi_val_dP(R, thn, pf);
            if (g.phi <= 0.0) continue;
            double sq = std::sqrt(g.phi);
            double wk = AR.w[k];
            acc_val += wk * sq;
            double inv = wk / (2.0 * sq);
            for (int j = 0; j < 5; ++j) acc_der[j] += inv * g.dP[j];
        }
        rt.fh += R * half * acc_val;
        for (int j = 0; j < 5; ++j) rt.dfh[j] += R * half * acc_der[j];
    }
    return rt;
}

}  // namespace lcbinint::holonomic

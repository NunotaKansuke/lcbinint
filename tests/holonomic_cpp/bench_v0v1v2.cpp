// ATPT holonomic solver -- V0 / V1 / V2 three-variant micro-benchmark.
//
// Checkpoint revision 2, section 13.  ISOLATED bench: no kernel unification,
// no wiring, no API change, no cache.  One executable, three variants that
// share EVERYTHING except the two things under test:
//
//   V0  today's fused epoch_jacobian: value + F_half + 5-Jacobian + dmu/du,
//       fixed n_r = 64 Gauss-Chebyshev radial rule per cell.  (the baseline)
//   V1  V0's kernel exactly -- same classify_cells, same cells, same fixed
//       GC64 nodes, same boundary-quartic / QuarticWarm / polish_endpoint
//       path -- but UNIFORM VALUE-ONLY: accumulate F0 only.  Skip F_half,
//       dF0, dF_half, the sqrt(phi) angular loop, the internal->user chain
//       rule, every derivative accumulator.
//   V2  V1's value-only kernel, radial rule ONLY swapped to the algebraic
//       adaptive one-panel-first Gauss-Kronrod (GK15 sine-mapped, priority-
//       queue subdivision, rel_tol 1e-6 -- algebraic_boundary.cpp defaults).
//
//   V0 - V1  =  marginal cost of the LD half-integral + full Jacobian state.
//   V1 - V2  =  marginal cost of radial SCHEDULING (fixed GC64 vs adaptive GK)
//               -- but see the caveat printed in the report: adaptive GK also
//               changes the node count and the node distribution, which can
//               weaken the QuarticWarm warm-start.  The per-variant node /
//               quartic / warm-hit metrics are what expose that.
//
// The multipole / hexadecapole shortcut does not exist on the holonomic path
// at all (classify_cells always solves the full topology), so "shortcut OFF"
// holds for all three variants by construction.
//
// Build: configured by tests/holonomic_cpp/CMakeLists.txt (isolated project).
// Run:   taskset -c 0-7 ./build-holonomic-m7/bench_v0v1v2 [cases.tsv] [reps]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

// ------------------------------------------------------------------ counters
struct Counters {
    long nodes = 0;        // radius evaluations (quartic-solve sites + degen)
    long warm_hits = 0;    // QuarticWarm warm adoptions
    long cold_falls = 0;   // QuarticWarm cold solves
    bool reliable = true;  // any unreliable radius -> false (status parity)
    void add(const QuarticWarm& w) { warm_hits += w.warm_hits; cold_falls += w.cold_falls; }
};

// ------------------------------------------------- V0: fused value+Jacobian
// Mirror of flux_jacobian (epoch_jacobian.hpp) with counter hooks; calls the
// real radius_terms full-derivative path so timing tracks epoch_jacobian.
struct V0Out { double mu_uniform, F0; Status status; };
static V0Out run_v0(const LensParams& p, Counters& ct) {
    const ScopedFlushDenormals _g;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    TopologyResult topo = classify_cells(pf);
    Status st = (topo.status == Status::OK) ? Status::OK : Status::GRADIENT_UNRELIABLE;

    Cheb1Dyn rr(64);
    double F0 = 0.0, Fh = 0.0;
    std::array<double, 5> dF0i{}, dFhi{};
    for (const auto& c : topo.cells) {
        double w = c.r_hi - c.r_lo;
        if (w <= 0.0 || c.kind == ArcKind::kEmpty) continue;
        if (c.kind == ArcKind::kFull) st = Status::GRADIENT_UNRELIABLE;
        const double ins = 1e-9 * w;
        const double lo = c.r_lo + ins, hi = c.r_hi - ins;
        const double rmid = 0.5 * (lo + hi), rhalf = 0.5 * (hi - lo);
        QuarticWarm qw;
        for (int k = 0; k < 64; ++k) {
            double R = rmid + rhalf * rr.x[k];
            RadiusTerms rt = radius_terms(R, pf, kTanRel, &qw);
            ++ct.nodes;
            if (!rt.reliable) { st = Status::GRADIENT_UNRELIABLE; ct.reliable = false; }
            double Wk = rhalf * rr.w[k];
            F0 += Wk * rt.f0;
            Fh += Wk * rt.fh;
            for (int j = 0; j < 5; ++j) { dF0i[j] += Wk * rt.df0[j]; dFhi[j] += Wk * rt.dfh[j]; }
        }
        ct.add(qw);
    }
    // internal->user chain rule (result unused here, but timed like the real path)
    volatile double sink = 0.0;
    auto du0 = internal_to_user_jac(dF0i, p);
    auto duh = internal_to_user_jac(dFhi, p);
    for (int j = 0; j < 5; ++j) sink += du0[j] + duh[j];
    (void)sink; (void)Fh;
    if (near_origin_source(pf)) st = Status::GRADIENT_UNRELIABLE;
    return {F0 / (kPi * p.rho * p.rho), F0, st};
}

// ---------------------------------------- value-only radius (shared V1 + V2)
// Exactly V1's per-radius computation: same arc_intervals (same QuarticWarm
// chain), same polish_endpoint on both ends, same near-tangency gate.  F0
// contribution only; no phi_val_dP, no F_half, no derivatives.
struct VO { double f0; bool reliable; };
static VO radius_f0_valueonly(double R, const PrimaryFrame& pf, QuarticWarm* w) {
    VO rt{0.0, true};
    ArcSet as = arc_intervals(R, pf, w);
    if (as.kind == ArcKind::kEmpty) return rt;
    if (as.kind == ArcKind::kFull) { rt.f0 = R * kTwoPi; rt.reliable = false; return rt; }
    if (as.kind == ArcKind::kDegenerate) {
        ArcSet g = grid_intervals(R, pf);
        rt.reliable = false;
        if (g.kind == ArcKind::kEmpty) return rt;
        if (g.kind == ArcKind::kFull) { rt.f0 = R * kTwoPi; return rt; }
        as = g;
    }
    const double tan_thresh = kTanRel * pf.rho / std::max(R, 1e-9);
    for (const auto& arc : as.arcs) {
        PolishResult pe = polish_endpoint(R, arc[0], pf);
        PolishResult pl = polish_endpoint(R, arc[1], pf);
        double te = pe.theta, tl = pl.theta;
        if (tl <= te) tl += kTwoPi;
        rt.reliable = rt.reliable && pe.reliable && pl.reliable;
        if (std::fabs(pe.dphi_dtheta) < tan_thresh ||
            std::fabs(pl.dphi_dtheta) < tan_thresh)
            rt.reliable = false;
        rt.f0 += R * (tl - te);
    }
    return rt;
}

// ------------------------------------------ V1: fixed GC64, uniform value-only
struct V1Out { double mu_uniform, F0; Status status; };
static V1Out run_v1(const LensParams& p, Counters& ct) {
    const ScopedFlushDenormals _g;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    TopologyResult topo = classify_cells(pf);
    Status st = (topo.status == Status::OK) ? Status::OK : Status::GRADIENT_UNRELIABLE;

    Cheb1Dyn rr(64);
    double F0 = 0.0;
    for (const auto& c : topo.cells) {
        double w = c.r_hi - c.r_lo;
        if (w <= 0.0 || c.kind == ArcKind::kEmpty) continue;
        if (c.kind == ArcKind::kFull) st = Status::GRADIENT_UNRELIABLE;
        const double ins = 1e-9 * w;
        const double lo = c.r_lo + ins, hi = c.r_hi - ins;
        const double rmid = 0.5 * (lo + hi), rhalf = 0.5 * (hi - lo);
        QuarticWarm qw;
        for (int k = 0; k < 64; ++k) {
            double R = rmid + rhalf * rr.x[k];
            VO rt = radius_f0_valueonly(R, pf, &qw);
            ++ct.nodes;
            if (!rt.reliable) { st = Status::GRADIENT_UNRELIABLE; ct.reliable = false; }
            F0 += rhalf * rr.w[k] * rt.f0;
        }
        ct.add(qw);
    }
    if (near_origin_source(pf)) st = Status::GRADIENT_UNRELIABLE;
    return {F0 / (kPi * p.rho * p.rho), F0, st};
}

// -------------------------------- V2: adaptive one-panel-first Gauss-Kronrod
// GK15 nodes + Kronrod / Gauss weights, verbatim from
// algebraic_boundary.cpp::gauss_kronrod15().  Sine-mapped (nodes clustered at
// the band endpoints), priority-queue subdivision on the |Kronrod - Gauss|
// panel error -- verbatim structure of integrate_outer_adaptive +
// experimental_algebraic_boundary_integral (one-panel estimate first,
// tolerance_per_band = requested_flux_error / n_bands).
namespace gk15 {
constexpr int N = 15;
constexpr double node[N] = {
    -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
    -0.7415311855993945, -0.5860872354676911, -0.4058451513773972,
    -0.2077849550078985,  0.0,                 0.2077849550078985,
     0.4058451513773972,  0.5860872354676911,  0.7415311855993945,
     0.8648644233597691,  0.9491079123427585,  0.9914553711208126};
constexpr double kw[N] = {
    0.02293532201052922, 0.06309209262997855, 0.1047900103222502,
    0.1406532597155259,  0.1690047266392679,  0.1903505780647854,
    0.2044329400752989,  0.2094821410847278,  0.2044329400752989,
    0.1903505780647854,  0.1690047266392679,  0.1406532597155259,
    0.1047900103222502,  0.06309209262997855, 0.02293532201052922};
constexpr double gw[N] = {
    0.0, 0.1294849661688697, 0.0, 0.2797053914892766, 0.0,
    0.3818300505051189, 0.0, 0.4179591836734694, 0.0, 0.3818300505051189,
    0.0, 0.2797053914892766, 0.0, 0.1294849661688697, 0.0};
struct Mapped { double coord[N], jac[N]; };
inline const Mapped& mapped() {
    static const Mapped m = [] {
        Mapped r;
        for (int i = 0; i < N; ++i) {
            double ph = 0.5 * kPi * node[i];
            r.coord[i] = std::sin(ph);
            r.jac[i] = 0.5 * kPi * std::cos(ph);
        }
        return r;
    }();
    return m;
}
}  // namespace gk15

constexpr double kV2RelTol = 1.0e-6;   // AlgebraicBoundarySettings::relative_tolerance
constexpr double kV2AbsTol = 1.0e-8;   // AlgebraicBoundarySettings::absolute_tolerance
constexpr int kV2MaxDepth = 26;        // AlgebraicBoundarySettings::maximum_radial_depth

struct GKPanel {
    double lo = 0.0, hi = 0.0;
    int depth = 0;
    double value = 0.0;       // Kronrod estimate of int f0 dR over [lo,hi]
    double radial_error = 0.0;  // |Kronrod - Gauss|
};

// One GK15 panel.  Nodes are ascending in R after the sine map, so the
// per-band QuarticWarm chain still sees a monotone march within a panel.
static GKPanel gk_panel(double lo, double hi, const PrimaryFrame& pf,
                        QuarticWarm* w, Counters& ct, bool* reliable) {
    const auto& mp = gk15::mapped();
    const double mid = 0.5 * (lo + hi), halfw = 0.5 * (hi - lo);
    double K = 0.0, G = 0.0;
    for (int i = 0; i < gk15::N; ++i) {
        double R = mid + halfw * mp.coord[i];
        double tw = halfw * mp.jac[i];
        VO rt = radius_f0_valueonly(R, pf, w);
        ++ct.nodes;
        if (!rt.reliable) { *reliable = false; ct.reliable = false; }
        K += tw * gk15::kw[i] * rt.f0;
        G += tw * gk15::gw[i] * rt.f0;
    }
    return {lo, hi, 0, K, std::fabs(K - G)};
}

struct V2Out { double mu_uniform, F0; Status status; };
static V2Out run_v2(const LensParams& p, Counters& ct) {
    const ScopedFlushDenormals _g;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    TopologyResult topo = classify_cells(pf);
    Status st = (topo.status == Status::OK) ? Status::OK : Status::GRADIENT_UNRELIABLE;
    bool reliable = true;

    struct Band { double lo, hi; };
    std::vector<Band> bands;
    for (const auto& c : topo.cells) {
        double w = c.r_hi - c.r_lo;
        if (w <= 0.0 || c.kind == ArcKind::kEmpty) continue;
        if (c.kind == ArcKind::kFull) st = Status::GRADIENT_UNRELIABLE;
        const double ins = 1e-9 * w;
        bands.push_back({c.r_lo + ins, c.r_hi - ins});
    }
    const double norm = kPi * p.rho * p.rho;

    // --- one-panel estimate per band first (tol = inf) --------------------
    double prelim_total = 0.0, prelim_err = 0.0;
    std::vector<GKPanel> firsts;
    firsts.reserve(bands.size());
    for (const auto& b : bands) {
        QuarticWarm qw;
        GKPanel fp = gk_panel(b.lo, b.hi, pf, &qw, ct, &reliable);
        ct.add(qw);
        firsts.push_back(fp);
        prelim_total += fp.value;
        prelim_err += fp.radial_error;
    }
    const double prelim_mag = prelim_total / norm;
    const double req_mag_err = std::max(kV2AbsTol, 0.0) +
                               std::max(kV2RelTol, 0.0) * std::max(std::fabs(prelim_mag), 1.0);
    const double req_flux_err = req_mag_err * norm;

    double F0;
    if (prelim_err <= req_flux_err) {
        F0 = prelim_total;
    } else {
        const double tol_per_band =
            bands.empty() ? req_flux_err : req_flux_err / (double)bands.size();
        double total = 0.0;
        for (size_t bi = 0; bi < bands.size(); ++bi) {
            QuarticWarm qw;
            bool rel_b = true;
            // priority queue on radial_error (max first), verbatim structure
            auto cmp = [](const GKPanel& a, const GKPanel& b) {
                return a.radial_error < b.radial_error;
            };
            std::priority_queue<GKPanel, std::vector<GKPanel>, decltype(cmp)> pq(cmp);
            GKPanel first = gk_panel(bands[bi].lo, bands[bi].hi, pf, &qw, ct, &rel_b);
            ct.add(qw);
            double band_total = first.value;
            double band_err = first.radial_error;
            if (first.depth < kV2MaxDepth) pq.push(first);
            int guard = 0;
            while (band_err > tol_per_band && !pq.empty() && guard++ < 4096) {
                GKPanel parent = pq.top();
                pq.pop();
                double m = 0.5 * (parent.lo + parent.hi);
                QuarticWarm lw, rw;
                GKPanel L = gk_panel(parent.lo, m, pf, &lw, ct, &rel_b);
                GKPanel R = gk_panel(m, parent.hi, pf, &rw, ct, &rel_b);
                ct.add(lw); ct.add(rw);
                L.depth = R.depth = parent.depth + 1;
                band_total += L.value + R.value - parent.value;
                band_err = std::max(0.0, band_err - parent.radial_error +
                                             L.radial_error + R.radial_error);
                if (L.depth < kV2MaxDepth) pq.push(L);
                if (R.depth < kV2MaxDepth) pq.push(R);
            }
            if (band_err > tol_per_band) reliable = false;  // budget not met
            reliable = reliable && rel_b;
            total += band_total;
        }
        F0 = total;
    }
    if (!reliable && st == Status::OK) st = Status::GRADIENT_UNRELIABLE;
    if (near_origin_source(pf)) st = Status::GRADIENT_UNRELIABLE;
    return {F0 / norm, F0, st};
}

// ------------------------------------------------------------------- harness
struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u;
    double t_jac_incumbent;
    std::string name;
};

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1 - frac) + v[lo + 1] * frac;
}
static double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
}
static double frac_pos(const std::vector<double>& v) {
    long n = 0;
    for (double x : v) if (x > 0.0) ++n;
    return v.empty() ? 0.0 : (double)n / v.size();
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 200;

    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
    std::vector<BC> cases;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        BC c;
        if (!(ls >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
              c.t_jac_incumbent >> c.name))
            continue;
        cases.push_back(c);
    }
    std::fprintf(stderr, "loaded %zu bench cases, reps=%d\n", cases.size(), reps);
    std::fprintf(stderr,
        "multipole/hexadecapole shortcut: N/A on the holonomic path "
        "(classify_cells always solves the full topology) -> OFF for all 3.\n\n");

    std::vector<double> t0, t1, t2;          // per-case best ms
    std::vector<double> d01, d12;            // per-case time deltas
    std::vector<double> nodes0, nodes1, nodes2;
    std::vector<double> qs0, qs1, qs2;       // quartic-solve sites (warm+cold)
    long wh0 = 0, cf0 = 0, wh1 = 0, cf1 = 0, wh2 = 0, cf2 = 0;
    std::vector<double> mupar1, mupar2;      // |dmu/mu| vs V0 uniform
    std::string worst2_name; double worst2 = -1.0; Status worst2_v0st = Status::OK;
    int ok0 = 0, ok1 = 0, ok2 = 0;
    int mism_status_1 = 0, mism_status_2 = 0;
    double checksum = 0.0;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};

        Counters ic0, ic1, ic2;
        V0Out r0 = run_v0(p, ic0);
        V1Out r1 = run_v1(p, ic1);
        V2Out r2 = run_v2(p, ic2);
        checksum += r0.mu_uniform + r1.mu_uniform + r2.mu_uniform;

        if (r0.status == Status::OK) ++ok0;
        if (r1.status == Status::OK) ++ok1;
        if (r2.status == Status::OK) ++ok2;
        if (r1.status != r0.status) ++mism_status_1;
        if (r2.status != r0.status) ++mism_status_2;

        nodes0.push_back(ic0.nodes); nodes1.push_back(ic1.nodes); nodes2.push_back(ic2.nodes);
        qs0.push_back(ic0.warm_hits + ic0.cold_falls);
        qs1.push_back(ic1.warm_hits + ic1.cold_falls);
        qs2.push_back(ic2.warm_hits + ic2.cold_falls);
        wh0 += ic0.warm_hits; cf0 += ic0.cold_falls;
        wh1 += ic1.warm_hits; cf1 += ic1.cold_falls;
        wh2 += ic2.warm_hits; cf2 += ic2.cold_falls;

        double base = std::fabs(r0.mu_uniform) > 0.0 ? std::fabs(r0.mu_uniform) : 1.0;
        mupar1.push_back(std::fabs(r1.mu_uniform - r0.mu_uniform) / base);
        double mp2 = std::fabs(r2.mu_uniform - r0.mu_uniform) / base;
        mupar2.push_back(mp2);
        if (mp2 > worst2) { worst2 = mp2; worst2_name = c.name; worst2_v0st = r0.status; }

        double b0 = 1e30, b1 = 1e30, b2 = 1e30;
        for (int r = 0; r < reps; ++r) {
            Counters s;
            auto ta = clk::now(); auto v0 = run_v0(p, s); auto tb = clk::now();
            auto v1 = run_v1(p, s); auto tc = clk::now();
            auto v2 = run_v2(p, s); auto td = clk::now();
            b0 = std::min(b0, std::chrono::duration<double, std::milli>(tb - ta).count());
            b1 = std::min(b1, std::chrono::duration<double, std::milli>(tc - tb).count());
            b2 = std::min(b2, std::chrono::duration<double, std::milli>(td - tc).count());
            checksum += v0.F0 + v1.F0 + v2.F0;
        }
        t0.push_back(b0); t1.push_back(b1); t2.push_back(b2);
        d01.push_back(b0 - b1);
        d12.push_back(b1 - b2);
    }

    auto row = [](const char* tag, std::vector<double>& t) {
        std::fprintf(stderr,
            "  %-4s  median %8.4f  p90 %8.4f  p95 %8.4f  p99 %8.4f  max %8.4f  (ms)\n",
            tag, pct(t, 50), pct(t, 90), pct(t, 95), pct(t, 99),
            *std::max_element(t.begin(), t.end()));
    };
    std::fprintf(stderr, "== wall time, best-of-%d per case, 108 full-solve cases ==\n", reps);
    row("V0", t0); row("V1", t1); row("V2", t2);

    std::fprintf(stderr, "\n== derived deltas (per-case, then percentiles) ==\n");
    std::fprintf(stderr,
        "  V0-V1 (LD + Jacobian-state marginal cost):\n"
        "     median %+8.4f  p90 %+8.4f  p95 %+8.4f  p99 %+8.4f  ms   (%.0f%% of cases > 0)\n",
        pct(d01, 50), pct(d01, 90), pct(d01, 95), pct(d01, 99), 100.0 * frac_pos(d01));
    std::fprintf(stderr,
        "  V1-V2 (radial scheduling marginal cost; node-count / warm-start effects folded in):\n"
        "     median %+8.4f  p90 %+8.4f  p95 %+8.4f  p99 %+8.4f  ms   (%.0f%% of cases > 0)\n",
        pct(d12, 50), pct(d12, 90), pct(d12, 95), pct(d12, 99), 100.0 * frac_pos(d12));

    std::fprintf(stderr, "\n== per-variant kernel work (mean over 108 cases) ==\n");
    std::fprintf(stderr, "  %-4s  radial nodes %8.1f   quartic-solve sites %8.1f   warm-hit rate %6.3f\n",
                 "V0", mean(nodes0), mean(qs0), wh0 + cf0 ? (double)wh0 / (wh0 + cf0) : 0.0);
    std::fprintf(stderr, "  %-4s  radial nodes %8.1f   quartic-solve sites %8.1f   warm-hit rate %6.3f\n",
                 "V1", mean(nodes1), mean(qs1), wh1 + cf1 ? (double)wh1 / (wh1 + cf1) : 0.0);
    std::fprintf(stderr, "  %-4s  radial nodes %8.1f   quartic-solve sites %8.1f   warm-hit rate %6.3f\n",
                 "V2", mean(nodes2), mean(qs2), wh2 + cf2 ? (double)wh2 / (wh2 + cf2) : 0.0);

    std::fprintf(stderr, "\n== mu parity (uniform mu = F0 / (pi rho^2), vs V0's F0) ==\n");
    std::fprintf(stderr, "  V1 vs V0:  max |dmu/mu| %.3e   median %.3e\n", pct(mupar1, 100), pct(mupar1, 50));
    std::fprintf(stderr, "  V2 vs V0:  max |dmu/mu| %.3e   median %.3e\n", pct(mupar2, 100), pct(mupar2, 50));
    std::fprintf(stderr, "  V2 worst-parity case: %s  (V0 status %s)  -- one-panel-first gate accepted early\n",
                 worst2_name.c_str(), worst2_v0st == Status::OK ? "OK" : "non-OK");

    std::fprintf(stderr, "\n== status ==\n");
    std::fprintf(stderr, "  OK count:  V0 %d/%zu   V1 %d/%zu   V2 %d/%zu\n",
                 ok0, cases.size(), ok1, cases.size(), ok2, cases.size());
    std::fprintf(stderr, "  status-change vs V0:  V1 %d   V2 %d\n", mism_status_1, mism_status_2);
    std::fprintf(stderr, "\n  checksum %.6f\n", checksum);
    std::fprintf(stderr,
        "\nNOTE: the router-inclusive production benchmark (shortcut ON) is a\n"
        "      SEPARATE table, deferred to phase 4 (needs finite_source_binary\n"
        "      wiring).  Do not compare these numbers to it line-to-line.\n");
    return 0;
}

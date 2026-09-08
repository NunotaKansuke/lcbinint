// ATPT holonomic solver -- planner cost-split micro-benchmark (intermediate
// experiment E1, unified-engine implementation-order decision).
//
// The V0/V1/V2 bench (checkpoint sec.13.6) concluded that on the holonomic
// path classify_cells -- always-on D14 __float128 band discovery plus the
// per-cell grid probes -- is the dominant fixed cost, untouched by the
// value-only skip or the adaptive radial rule.  Before committing the
// implementation order to "fast planner first" (checkpoint sec.10 phase 3)
// this bench measures, per case, how classify_cells' wall time splits:
//
//   (A) radial_events / D14      -- __float128 degree-14 Aberth.  The fast
//                                   planner (find_radial_bands port) REPLACES
//                                   this with a ~0.03 ms seed-anchored march.
//   (B) grid probes              -- arcs_at(mid, 512) per cell + any
//                                   arcs_at(*, 3072) escalations.  The sec.5.2
//                                   confidence screen DROPS the 512 grid
//                                   cross-check (keeps only root-based checks),
//                                   so a fast-planner path saves most of this.
//   (C) quartic_topology probes  -- 3 per cell, straight from the boundary
//                                   quartic's real roots.  The sec.5.2 screen
//                                   KEEPS these (check (1): 3 in-band probes
//                                   agree on kind + crossing count).
//   (D) radial integration pass  -- V1 value-only (F0 from arc-interval
//                                   lengths, fixed GC64).  Planner-independent;
//                                   the "rest" the planner cannot touch.
//
// Ceiling on the fast-planner saving  ~=  (A) + most of (B).
// Floor that stays no matter what     >=  (C) + (D).
//
// ISOLATED bench.  No kernel change, no wiring.  Build via
// tests/holonomic_cpp/CMakeLists.txt.
//   taskset -c 0-7 ./build-holonomic-m7/bench_planner_split [cases.tsv] [reps]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

// ------------------------------------------------------------ split counters
struct Split {
    double t_d14 = 0.0;      // (A) radial_events
    double t_grid = 0.0;     // (B) arcs_at(512) + 3072 escalations
    double t_qtop = 0.0;     // (C) quartic_topology x3 per cell
    double t_merge = 0.0;    // event merge + bounds (tiny; kept honest)
    double t_radial = 0.0;   // (D) V1 value-only radial pass
    int n_cells = 0;
    int n_escalations = 0;   // cells that hit the 3072 grid
};

// classify_cells (cells.hpp) re-implemented with per-stage timing.  Behaviour
// identical -- same events, same merge, same 3-fraction probe, same escalation
// rule, same TOPOLOGY_UNCERTAIN gate.
static TopologyResult classify_cells_timed(const PrimaryFrame& pf, Split& sp) {
    auto t0 = clk::now();
    double r_max = 0.0;
    auto events = radial_events(pf, &r_max);
    auto t1 = clk::now();
    sp.t_d14 += std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<double> radii;
    for (const auto& e : events) radii.push_back(e.radius);
    std::sort(radii.begin(), radii.end());
    std::vector<double> merged;
    for (double r : radii) {
        double tol = std::max(1e-9, 1e-7 * r);
        if (merged.empty() || r - merged.back() > tol) merged.push_back(r);
    }
    std::vector<double> bounds;
    bounds.push_back(0.0);
    for (double r : merged)
        if (r > 0.0 && r < r_max) bounds.push_back(r);
    bounds.push_back(r_max);
    auto t2 = clk::now();
    sp.t_merge += std::chrono::duration<double, std::milli>(t2 - t1).count();

    TopologyResult out;
    out.r_max = r_max;
    Status worst = Status::OK;
    const double fr[3] = {0.18, 0.50, 0.82};

    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        double lo = bounds[i], hi = bounds[i + 1];
        if (hi - lo < 1e-11) continue;
        double Rp[3];
        for (int k = 0; k < 3; ++k) Rp[k] = lo + fr[k] * (hi - lo);

        auto q0 = clk::now();
        GridArcs probe[3];
        for (int k = 0; k < 3; ++k) probe[k] = quartic_topology(Rp[k], pf);
        auto q1 = clk::now();
        sp.t_qtop += std::chrono::duration<double, std::milli>(q1 - q0).count();

        bool uniform = probe[1].kind == probe[0].kind &&
                       probe[1].n_crossings == probe[0].n_crossings &&
                       probe[1].kind == probe[2].kind &&
                       probe[1].n_crossings == probe[2].n_crossings;

        auto g0 = clk::now();
        GridArcs mid_grid = arcs_at(Rp[1], pf, 512);
        bool mid_agrees = probe[1].kind == mid_grid.kind &&
                          probe[1].n_crossings == mid_grid.n_crossings;
        if (!uniform || !mid_agrees) {
            for (int k = 0; k < 3; ++k) probe[k] = arcs_at(Rp[k], pf, 3072);
            uniform = probe[1].kind == probe[0].kind &&
                      probe[1].n_crossings == probe[0].n_crossings &&
                      probe[1].kind == probe[2].kind &&
                      probe[1].n_crossings == probe[2].n_crossings;
            ++sp.n_escalations;
        }
        auto g1 = clk::now();
        sp.t_grid += std::chrono::duration<double, std::milli>(g1 - g0).count();

        const GridArcs& mid = probe[1];
        Status cs = Status::OK;
        if (!uniform || (mid.n_crossings % 2 != 0)) {
            cs = Status::TOPOLOGY_UNCERTAIN;
            worst = Status::TOPOLOGY_UNCERTAIN;
        }
        out.cells.push_back(CellPlan{(int)out.cells.size(), lo, hi,
                                     0.5 * (lo + hi), mid.kind,
                                     mid.n_crossings, cs});
        ++sp.n_cells;
    }
    out.status = worst;
    return out;
}

// value-only per-radius (identical to bench_v0v1v2.cpp radius_f0_valueonly)
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

struct EpochOut { double mu_uniform; Status status; };
static EpochOut run_split(const LensParams& p, Split& sp) {
    const ScopedFlushDenormals _g;
    const PrimaryFrame pf = PrimaryFrame::from(p);
    TopologyResult topo = classify_cells_timed(pf, sp);
    Status st = (topo.status == Status::OK) ? Status::OK : Status::GRADIENT_UNRELIABLE;

    auto r0 = clk::now();
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
            if (!rt.reliable) st = Status::GRADIENT_UNRELIABLE;
            F0 += rhalf * rr.w[k] * rt.f0;
        }
    }
    auto r1 = clk::now();
    sp.t_radial += std::chrono::duration<double, std::milli>(r1 - r0).count();
    if (near_origin_source(pf)) st = Status::GRADIENT_UNRELIABLE;
    return {F0 / (kPi * p.rho * p.rho), st};
}

// ------------------------------------------------------------------- harness
struct BC { double xs, ys, rho, q, a; int bary; double u, t_jac_incumbent; std::string name; };

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
    std::fprintf(stderr, "loaded %zu bench cases, reps=%d\n\n", cases.size(), reps);

    std::vector<double> d14, grid, qtop, merge, radial, total, cls;
    std::vector<double> saveable, floorv;   // (A)+(B)   vs   (C)+(D)
    std::vector<double> cells, escs;
    double checksum = 0.0;

    for (const auto& c : cases) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        Split best;
        double best_total = 1e30;
        for (int r = 0; r < reps; ++r) {
            Split sp;
            auto ta = clk::now();
            EpochOut e = run_split(p, sp);
            auto tb = clk::now();
            double tt = std::chrono::duration<double, std::milli>(tb - ta).count();
            checksum += e.mu_uniform;
            if (tt < best_total) { best_total = tt; best = sp; }
        }
        d14.push_back(best.t_d14);
        grid.push_back(best.t_grid);
        qtop.push_back(best.t_qtop);
        merge.push_back(best.t_merge);
        radial.push_back(best.t_radial);
        cls.push_back(best.t_d14 + best.t_grid + best.t_qtop + best.t_merge);
        total.push_back(best_total);
        saveable.push_back(best.t_d14 + best.t_grid);
        floorv.push_back(best.t_qtop + best.t_radial);
        cells.push_back(best.n_cells);
        escs.push_back(best.n_escalations);
    }

    auto row = [](const char* tag, std::vector<double>& t, double share_of) {
        std::fprintf(stderr,
            "  %-26s median %7.4f  p90 %7.4f  p99 %7.4f  mean %7.4f  (%.0f%% of epoch)\n",
            tag, pct(t, 50), pct(t, 90), pct(t, 99), mean(t),
            share_of > 0 ? 100.0 * mean(t) / share_of : 0.0);
    };
    double etot = mean(total);
    std::fprintf(stderr, "== per-stage wall time, best-of-%d per case, %zu cases ==\n",
                 reps, cases.size());
    std::fprintf(stderr, "   (best-of picks the fastest FULL epoch; stage times are that epoch's)\n\n");
    row("(A) radial_events / D14", d14, etot);
    row("(B) grid probes 512+3072", grid, etot);
    row("(C) quartic_topology x3/cell", qtop, etot);
    row("    event merge / bounds", merge, etot);
    row("(D) radial pass (V1 value)", radial, etot);
    std::fprintf(stderr, "  %-26s median %7.4f  p90 %7.4f  p99 %7.4f  mean %7.4f\n",
                 "   classify_cells total", pct(cls, 50), pct(cls, 90), pct(cls, 99), mean(cls));
    std::fprintf(stderr, "  %-26s median %7.4f  p90 %7.4f  p99 %7.4f  mean %7.4f\n",
                 "   full epoch", pct(total, 50), pct(total, 90), pct(total, 99), mean(total));

    std::fprintf(stderr, "\n== fast-planner ceiling vs floor ==\n");
    std::fprintf(stderr,
        "  saveable  (A)+(B)  : median %7.4f  p90 %7.4f  p99 %7.4f  mean %7.4f  (%.0f%% of epoch)\n",
        pct(saveable, 50), pct(saveable, 90), pct(saveable, 99), mean(saveable),
        100.0 * mean(saveable) / etot);
    std::fprintf(stderr,
        "  floor     (C)+(D)  : median %7.4f  p90 %7.4f  p99 %7.4f  mean %7.4f  (%.0f%% of epoch)\n",
        pct(floorv, 50), pct(floorv, 90), pct(floorv, 99), mean(floorv),
        100.0 * mean(floorv) / etot);
    std::fprintf(stderr,
        "  NOTE: the fast march itself costs ~0.03 ms (algebraic-side measured) and the\n"
        "        sec.5.2 screen still pays (C); realistic planner-path saving ~= (A)+(B) - 0.03ms - screen.\n");

    std::fprintf(stderr, "\n== structure ==\n");
    std::fprintf(stderr, "  cells/epoch      mean %.1f\n", mean(cells));
    std::fprintf(stderr, "  3072-escalations mean %.2f/epoch   (cases with >=1: %d/%zu)\n",
                 mean(escs),
                 (int)std::count_if(escs.begin(), escs.end(), [](double x){ return x >= 1; }),
                 cases.size());
    std::fprintf(stderr, "\n  checksum %.6f\n", checksum);
    return 0;
}

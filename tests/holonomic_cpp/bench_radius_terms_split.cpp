// ATPT holonomic -- per-node cost decomposition inside radius_terms.
//
// The Gauss-Manin / period-transport proposal replaces the per-radial-node
// angular sqrt(phi) quadrature (F_half + dF_half) with a 6x6 ODE.  This
// bench measures how big that term actually is, post B1/B2, relative to the
// other per-epoch terms:
//   * classify_cells (D14 + topology)
//   * per-node boundary_quartic coefficient build
//   * per-node quartic root solve
//   * per-node arc_intervals (build + solve + midpoint sign)
//   * per-node polish_endpoint Newton (2 per arc)
//   * per-node 64-pt angular sqrt(phi) + 5-wide derivative sweep
//
// Build target: bench_radius_terms_split
// Run: taskset -c 0-7 ./build-holonomic-m7/bench_radius_terms_split [cases.tsv] [reps]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/fp_env.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u;
    double t_jac;
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

template <class F>
static double best_ms(int reps, F&& f) {
    double b = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        f();
        auto t1 = clk::now();
        b = std::min(b, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return b;
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 120;
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }

    std::vector<BC> cases;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        BC c;
        if (!(ls >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
              c.t_jac >> c.name)) continue;
        cases.push_back(c);
    }
    std::fprintf(stderr, "loaded %zu cases, reps=%d\n", cases.size(), reps);

    std::vector<double> v_epoch, v_classify, v_rt, v_rt_warm, v_qbuild,
        v_qsolve, v_qsolve_warm, v_arcs, v_polish, v_angular;
    const auto& AR = ang_rule();
    const int NR = 64;
    volatile double sink = 0.0;
    ScopedFlushDenormals _ftz;

    for (const auto& c : cases) {
        LensParams lp{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        PrimaryFrame pf = PrimaryFrame::from(lp);

        v_epoch.push_back(best_ms(reps, [&] {
            auto ej = epoch_jacobian(lp, c.u, NR);
            sink += ej.mu;
        }));

        TopologyResult topo = classify_cells(pf, nullptr, nullptr);
        v_classify.push_back(best_ms(reps, [&] {
            auto t = classify_cells(pf, nullptr, nullptr);
            sink += t.r_max;
        }));

        // node list over arc-bearing cells (GC abscissae in r as well).
        // `cellOf` groups nodes by cell so the warm-start chain restarts
        // at each cell's first node, exactly like flux_jacobian_integrate.
        std::vector<double> Rn;
        std::vector<int> cellOf;
        int cidx = 0;
        for (const auto& cell : topo.cells) {
            double w = cell.r_hi - cell.r_lo;
            if (w <= 0.0 || cell.kind == ArcKind::kEmpty) continue;
            double ins = 1e-9 * w;
            double lo = cell.r_lo + ins, hi = cell.r_hi - ins;
            double rmid = 0.5 * (lo + hi), rhalf = 0.5 * (hi - lo);
            for (int k = 0; k < NR; ++k) {
                Rn.push_back(rmid + rhalf * AR.x[k]);
                cellOf.push_back(cidx);
            }
            ++cidx;
        }
        if (Rn.empty()) {
            v_rt.push_back(0); v_rt_warm.push_back(0);
            v_qbuild.push_back(0); v_qsolve.push_back(0);
            v_qsolve_warm.push_back(0);
            v_arcs.push_back(0); v_polish.push_back(0); v_angular.push_back(0);
            continue;
        }

        v_rt.push_back(best_ms(reps, [&] {
            double acc = 0;
            for (double R : Rn) {
                RadiusTerms rt = radius_terms(R, pf, kTanRel, nullptr);
                acc += rt.f0 + rt.fh + rt.dfh[0];
            }
            sink += acc;
        }));

        // production per-node loop: QuarticWarm threaded, restarts per cell
        v_rt_warm.push_back(best_ms(reps, [&] {
            double acc = 0;
            QuarticWarm qw;
            int cur = cellOf.empty() ? -1 : cellOf[0];
            for (size_t i = 0; i < Rn.size(); ++i) {
                if (cellOf[i] != cur) { qw = QuarticWarm{}; cur = cellOf[i]; }
                RadiusTerms rt = radius_terms(Rn[i], pf, kTanRel, &qw);
                acc += rt.f0 + rt.fh + rt.dfh[0];
            }
            sink += acc;
        }));

        // warm quartic solve (build is ~2e-4 ms, not subtracted)
        v_qsolve_warm.push_back(best_ms(reps, [&] {
            double acc = 0;
            QuarticWarm qw;
            int cur = cellOf.empty() ? -1 : cellOf[0];
            for (size_t i = 0; i < Rn.size(); ++i) {
                if (cellOf[i] != cur) { qw = QuarticWarm{}; cur = cellOf[i]; }
                QuarticCoeffs q = boundary_quartic(Rn[i], pf);
                acc += real_root_thetas_warm(q.p, qw).size();
            }
            sink += acc;
        }));

        double t_qbuild = best_ms(reps, [&] {
            double acc = 0;
            for (double R : Rn) { QuarticCoeffs q = boundary_quartic(R, pf);
                acc += q.p[0] + q.p[4]; }
            sink += acc;
        });
        v_qbuild.push_back(t_qbuild);

        double t_qsolve = best_ms(reps, [&] {
            double acc = 0;
            for (double R : Rn) { QuarticCoeffs q = boundary_quartic(R, pf);
                acc += real_root_thetas(q.p).size(); }
            sink += acc;
        });
        v_qsolve.push_back(t_qsolve - t_qbuild);

        double t_arcs = best_ms(reps, [&] {
            double acc = 0;
            for (double R : Rn) { ArcSet as = arc_intervals(R, pf, nullptr);
                acc += (double)as.arcs.size() + (int)as.kind; }
            sink += acc;
        });
        v_arcs.push_back(t_arcs - t_qsolve);  // net of quartic build+solve

        double t_polish = best_ms(reps, [&] {
            double acc = 0;
            for (double R : Rn) {
                ArcSet as = arc_intervals(R, pf, nullptr);
                if (as.kind != ArcKind::kArcs) continue;
                for (auto& arc : as.arcs) {
                    PolishResult pe = polish_endpoint(R, arc[0], pf);
                    PolishResult pl = polish_endpoint(R, arc[1], pf);
                    acc += pe.theta + pl.theta;
                }
            }
            sink += acc;
        });
        v_polish.push_back(t_polish - t_arcs);

        double t_angular = best_ms(reps, [&] {
            double acc = 0;
            for (double R : Rn) {
                ArcSet as = arc_intervals(R, pf, nullptr);
                if (as.kind != ArcKind::kArcs) continue;
                for (auto& arc : as.arcs) {
                    double te = arc[0], tl = arc[1];
                    if (tl <= te) tl += kTwoPi;
                    double half = 0.5 * (tl - te), mid = 0.5 * (te + tl);
                    double av = 0; std::array<double, 5> ad{};
                    for (int k = 0; k < 64; ++k) {
                        double thn = mid + half * AR.x[k];
                        PhiValDP g = phi_val_dP(R, thn, pf);
                        if (g.phi <= 0.0) continue;
                        double sq = std::sqrt(g.phi);
                        double wk = AR.w[k];
                        av += wk * sq;
                        double inv = wk / (2.0 * sq);
                        for (int j = 0; j < 5; ++j) ad[j] += inv * g.dP[j];
                    }
                    acc += av + ad[0] + ad[4];
                }
            }
            sink += acc;
        });
        v_angular.push_back(t_angular - t_arcs);
    }

    auto row = [](const char* name, std::vector<double>& v) {
        std::fprintf(stderr, "  %-28s median %8.4f  p90 %8.4f  max %8.4f  ms\n",
                     name, pct(v, 50), pct(v, 90), pct(v, 100));
    };
    std::fprintf(stderr, "\n== per-epoch cost decomposition (best-of-%d) ==\n", reps);
    row("epoch_jacobian (full)", v_epoch);
    row("classify_cells (D14+topo)", v_classify);
    row("radius_terms cold (nodes)", v_rt);
    row("radius_terms WARM (nodes)", v_rt_warm);
    std::fprintf(stderr, "  --- per-node components, each NET of the stages it re-runs: ---\n");
    row("quartic coeff build", v_qbuild);
    row("quartic root solve COLD", v_qsolve);
    row("quartic root solve WARM", v_qsolve_warm);
    row("arc midpoint-sign filter", v_arcs);
    row("polish_endpoint x2 / arc", v_polish);
    row("64pt angular sqrtphi+dP", v_angular);

    std::vector<double> sh_ang, sh_rt, sh_cls, sh_qb, sh_qs;
    for (size_t i = 0; i < v_epoch.size(); ++i) {
        if (v_epoch[i] <= 0) continue;
        sh_ang.push_back(100.0 * v_angular[i] / v_epoch[i]);
        sh_rt.push_back(100.0 * v_rt[i] / v_epoch[i]);
        sh_cls.push_back(100.0 * v_classify[i] / v_epoch[i]);
        sh_qb.push_back(100.0 * v_qbuild[i] / v_epoch[i]);
        sh_qs.push_back(100.0 * v_qsolve[i] / v_epoch[i]);
    }
    std::fprintf(stderr, "\n  share of epoch  classify_cells   median %.1f%%  p90 %.1f%%\n", pct(sh_cls, 50), pct(sh_cls, 90));
    std::fprintf(stderr, "  share of epoch  radius_terms     median %.1f%%  p90 %.1f%%\n", pct(sh_rt, 50), pct(sh_rt, 90));
    std::fprintf(stderr, "  share of epoch  quartic build    median %.1f%%  p90 %.1f%%\n", pct(sh_qb, 50), pct(sh_qb, 90));
    std::fprintf(stderr, "  share of epoch  quartic solve    median %.1f%%  p90 %.1f%%\n", pct(sh_qs, 50), pct(sh_qs, 90));
    std::fprintf(stderr, "  share of epoch  ANGULAR sweep    median %.1f%%  p90 %.1f%%\n", pct(sh_ang, 50), pct(sh_ang, 90));
    std::fprintf(stderr, "\n  sink %.6f\n", (double)sink);
    return 0;
}

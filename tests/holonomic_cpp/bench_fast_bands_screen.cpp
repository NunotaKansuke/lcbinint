// ATPT holonomic solver -- fast-planner + local confidence screen vs D14.
//
// Phase A step 3 (unified-engine implementation order, checkpoint sec.0.7 /
// sec.5.2).  bench_fast_bands measured the seed-anchored planner's discovery
// parity; this bench adds the sec.5.2 confidence screen (fast_bands_screen.hpp)
// on top and measures how often the screen sends an epoch to the D14 oracle.
//
// For each of the 108 three-way bench cases:
//   seeds  = binary_point_images(pf)
//   fb     = fast_bands(pf, seeds)
//   scr    = fast_bands_screen(pf, fb, seeds)
//   D14 oracle (classify_cells) is invoked  <=>  seeds/fb unreliable OR
//   scr.pass == false.
//
// Reports, over the 108 cases:
//   * screen PASS / FAIL / planner-bail counts;
//   * combined D14-invocation rate (the sec.0.7 gate deliverable);
//   * per-check failure breakdown (failed_mask);
//   * fold-ratio (worst_fold) distribution -- for tuning kFoldLo/kFoldHi;
//   * screen probe-call cost (arc_intervals / quartic_topology evaluations);
//   * wall time: seeds + fast_bands + screen   vs   classify_cells, and the
//     BLENDED cost of the screened path (screen always + D14 only when the
//     screen does not pass).
//
// classify_cells stays the oracle; this bench does not change any wiring.
//
//   taskset -c 0-7 ./build-holonomic-m7/bench_fast_bands_screen [cases.tsv] [reps]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/fast_bands.hpp"
#include "lcbinint/magnification/holonomic/fast_bands_screen.hpp"
#include "lcbinint/magnification/holonomic/fp_env.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/point_images.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u, t_jac;
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
              c.t_jac >> c.name))
            continue;
        cases.push_back(c);
    }
    std::fprintf(stderr, "loaded %zu bench cases, reps=%d\n\n", cases.size(),
                 reps);

    int n_bail = 0, n_pass = 0, n_fail = 0;
    int fm_topo = 0, fm_fold = 0, fm_seed = 0, fm_compl = 0;
    int n_check4_ran = 0;
    std::vector<std::pair<std::string, std::string>> fails;
    std::vector<double> worst_fold, probe_calls;
    std::vector<double> t_screen, t_cls, t_blended;
    double checksum = 0.0;

    for (const auto& c : cases) {
        LensParams lp{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        ScopedFlushDenormals _ftz;
        PrimaryFrame pf = PrimaryFrame::from(lp);

        // --- correctness (single evaluation) ---
        PointImages seeds = binary_point_images(pf);
        FastBands fb = fast_bands(pf, seeds);
        const bool planner_ok = seeds.reliable && fb.reliable;

        bool to_oracle;
        if (!planner_ok) {
            ++n_bail;
            to_oracle = true;
        } else {
            FastBandsScreen scr = fast_bands_screen(pf, fb, seeds);
            worst_fold.push_back(scr.worst_fold);
            probe_calls.push_back(scr.probe_calls);
            if (scr.down_significant >= 0) ++n_check4_ran;
            if (scr.pass) {
                ++n_pass;
                to_oracle = false;
            } else {
                ++n_fail;
                to_oracle = true;
                if (scr.failed_mask & 1u) ++fm_topo;
                if (scr.failed_mask & 2u) ++fm_fold;
                if (scr.failed_mask & 4u) ++fm_seed;
                if (scr.failed_mask & 8u) ++fm_compl;
                fails.push_back({c.name + (c.u > 0 ? " u+" : " u0"), scr.reason});
            }
        }
        checksum += to_oracle ? 1.0 : 0.0;

        // --- timing ---
        double bs = 1e30, bc = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            PointImages s2 = binary_point_images(pf);
            FastBands f2 = fast_bands(pf, s2);
            bool ok2 = s2.reliable && f2.reliable;
            bool oracle2 = !ok2;
            if (ok2) {
                FastBandsScreen sc2 = fast_bands_screen(pf, f2, s2);
                oracle2 = !sc2.pass;
                checksum += sc2.probe_calls;
            }
            auto t1 = clk::now();
            TopologyResult tr2 = classify_cells(pf);
            auto t2 = clk::now();
            checksum += f2.bands.size() + tr2.cells.size() + (oracle2 ? 1 : 0);
            double ds = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double dc = std::chrono::duration<double, std::milli>(t2 - t1).count();
            bs = std::min(bs, ds);
            bc = std::min(bc, dc);
        }
        t_screen.push_back(bs);
        t_cls.push_back(bc);
        t_blended.push_back(bs + (to_oracle ? bc : 0.0));
    }

    const int N = (int)cases.size();
    const int oracle = n_bail + n_fail;
    (void)mean;
    std::fprintf(stderr, "== screen outcome (%d epochs) ==\n", N);
    std::fprintf(stderr, "  planner bail (seeds/fb unreliable) : %d  (%.1f%%)\n",
                 n_bail, 100.0 * n_bail / N);
    std::fprintf(stderr, "  screen PASS                        : %d  (%.1f%%)\n",
                 n_pass, 100.0 * n_pass / N);
    std::fprintf(stderr, "  screen FAIL                        : %d  (%.1f%%)\n",
                 n_fail, 100.0 * n_fail / N);
    std::fprintf(stderr, "  --> D14 oracle invocation rate     : %d / %d  "
                 "(%.1f%%)\n", oracle, N, 100.0 * oracle / N);
    std::fprintf(stderr, "  check 4 (complement scan) ran       : %d\n",
                 n_check4_ran);
    std::fprintf(stderr,
                 "  screen FAIL by check : 1 partition=%d  2 fold=%d  "
                 "3 seed-cov=%d  4 complement=%d\n",
                 fm_topo, fm_fold, fm_seed, fm_compl);
    if (!worst_fold.empty())
        std::fprintf(stderr,
                     "  worst fold ratio     : min %.3f  p10 %.3f  median %.3f  "
                     "p90 %.3f  max %.3f  [pass window %.2f..%.2f]\n",
                     pct(worst_fold, 0), pct(worst_fold, 10), pct(worst_fold, 50),
                     pct(worst_fold, 90), pct(worst_fold, 100),
                     fast_bands_screen_detail::kFoldLo,
                     fast_bands_screen_detail::kFoldHi);
    if (!probe_calls.empty())
        std::fprintf(stderr,
                     "  screen probe calls   : median %.0f  p90 %.0f  max %.0f\n",
                     pct(probe_calls, 50), pct(probe_calls, 90),
                     pct(probe_calls, 100));

    if (!fails.empty()) {
        std::fprintf(stderr, "\n  -- screen FAIL detail (%zu) --\n", fails.size());
        for (auto& f : fails)
            std::fprintf(stderr, "     %-18s %s\n", f.first.c_str(),
                         f.second.c_str());
    }

    std::fprintf(stderr, "\n== wall time, best-of-%d per epoch, %d epochs ==\n",
                 reps, N);
    std::fprintf(stderr,
                 "  seeds + fast_bands + screen : median %7.4f  p90 %7.4f  "
                 "p95 %7.4f  p99 %7.4f\n",
                 pct(t_screen, 50), pct(t_screen, 90), pct(t_screen, 95),
                 pct(t_screen, 99));
    std::fprintf(stderr,
                 "  classify_cells (D14 path)   : median %7.4f  p90 %7.4f  "
                 "p95 %7.4f  p99 %7.4f\n",
                 pct(t_cls, 50), pct(t_cls, 90), pct(t_cls, 95), pct(t_cls, 99));
    std::fprintf(stderr,
                 "  BLENDED screened path       : median %7.4f  p90 %7.4f  "
                 "p95 %7.4f  p99 %7.4f\n",
                 pct(t_blended, 50), pct(t_blended, 90), pct(t_blended, 95),
                 pct(t_blended, 99));
    std::fprintf(stderr,
                 "  speedup screen-only / blended vs D14 (median): %.2fx / %.2fx\n",
                 pct(t_cls, 50) / std::max(pct(t_screen, 50), 1e-12),
                 pct(t_cls, 50) / std::max(pct(t_blended, 50), 1e-12));

    std::fprintf(stderr, "\nchecksum %.3f\n", checksum);
    return 0;
}

// ATPT holonomic solver -- fast seed-anchored band planner vs classify_cells.
//
// Phase A step 2 (unified-engine implementation order, checkpoint sec.0.7).
// For each of the 108 three-way bench cases:
//
//   reference : classify_cells(pf)  -- D14 events + per-cell probes.  Its
//               image bands = maximal runs of consecutive non-kEmpty cells.
//   candidate : binary_point_images(pf) -> fast_bands(pf, seeds)  -- the
//               seed-anchored step-doubling march + boolean bisection.
//
// Reports, over the 108 cases:
//   * fast_bands.reliable rate (and the bail reasons);
//   * band-count parity vs the classify_cells band set;
//   * worst band-edge disagreement (only where counts match);
//   * wall time: point_images + fast_bands   vs   classify_cells.
//
// This is a DISCOVERY-PARITY bench, not a wiring change.  classify_cells stays
// the oracle; nothing in the solver calls fast_bands yet.
//
//   taskset -c 0-7 ./build-holonomic-m7/bench_fast_bands [cases.tsv] [reps]

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
#include "lcbinint/magnification/holonomic/fp_env.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/point_images.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u, t_jac;
    std::string name;
};

// classify_cells image bands: merge consecutive non-kEmpty cells.
static std::vector<FastBand> reference_bands(const TopologyResult& tr) {
    std::vector<FastBand> out;
    bool open = false;
    for (const auto& c : tr.cells) {
        const bool img = (c.kind != ArcKind::kEmpty);
        if (img && !open) {
            out.push_back({c.r_lo, c.r_hi});
            open = true;
        } else if (img && open) {
            out.back().r_hi = c.r_hi;
        } else {
            open = false;
        }
    }
    return out;
}

// Widest arc (radians) the boundary circle carries at a set of interior radii
// of the band.  classify_cells' probe cross-checks the boundary quartic
// against a 3072-point grid (step 2.04e-3 rad) and escalates on disagreement;
// a band whose arc never exceeds a few grid steps is INVISIBLE to that path
// (grid + escalation both miss it) and classify_cells silently drops it.
// fast_bands (boundary quartic only) still resolves it.  Such "razor" bands
// carry a magnification contribution ~ (angw * width / (2 pi r_max)) and are
// below the M7 reference tolerance -- a legitimate strict refinement, not a
// parity failure.
static constexpr double kRazorAng = 3.0e-3;  // ~1.5 * 3072-grid step

static double band_max_arc(const FastBand& b, const PrimaryFrame& pf) {
    double w = 0.0;
    for (double f : {0.25, 0.5, 0.75}) {
        double R = b.r_lo + f * (b.r_hi - b.r_lo);
        ArcSet as = arc_intervals(R, pf);
        if (as.kind == ArcKind::kFull) return 6.283185307179586;
        for (auto& a : as.arcs) {
            double d = a[1] - a[0];
            if (d < 0.0) d += 6.283185307179586;
            w = std::max(w, d);
        }
    }
    return w;
}
static bool is_significant(const FastBand& b, const PrimaryFrame& pf) {
    return band_max_arc(b, pf) >= kRazorAng;
}

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

    int n_reliable = 0, n_uncertain_ref = 0;
    int n_sig_count_match = 0;   // # significant fast bands == # significant ref
    int n_sig_covered = 0;      // every significant ref band overlapped by a fast
    int n_sig_edge_ok = 0;      // + edges agree to <5e-3 of band width
    int n_full_parity = 0;      // exact band-set match incl. razor
    int total_razor_extra = 0;  // fast razor bands with no significant ref match
    std::vector<std::pair<std::string, std::string>> bails;
    std::vector<std::pair<std::string, std::string>> mismatches;
    std::vector<double> worst_edge_rel;
    std::vector<double> t_fast, t_cls, hi_calls;
    double checksum = 0.0;

    for (const auto& c : cases) {
        LensParams lp{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        ScopedFlushDenormals _ftz;
        PrimaryFrame pf = PrimaryFrame::from(lp);

        // --- correctness (single evaluation) ---
        PointImages seeds = binary_point_images(pf);
        FastBands fb = fast_bands(pf, seeds);
        TopologyResult tr = classify_cells(pf);
        std::vector<FastBand> ref = reference_bands(tr);

        const bool ref_uncertain = (tr.status == Status::TOPOLOGY_UNCERTAIN);
        if (ref_uncertain) ++n_uncertain_ref;

        if (fb.reliable) {
            ++n_reliable;
            checksum += fb.bands.size();

            // significant (grid-visible) sub-sets of each band list
            std::vector<FastBand> ref_sig, fast_sig;
            for (auto& b : ref)
                if (is_significant(b, pf)) ref_sig.push_back(b);
            for (auto& b : fb.bands)
                if (is_significant(b, pf)) fast_sig.push_back(b);

            const bool sig_count = (fast_sig.size() == ref_sig.size());
            if (sig_count) ++n_sig_count_match;

            // coverage + edge agreement, significant ref bands only
            bool covered = true, edges = true;
            double worst = 0.0;
            for (auto& rb : ref_sig) {
                const FastBand* best = nullptr;
                double best_ov = 0.0;
                for (auto& xb : fast_sig) {
                    double ov = std::min(rb.r_hi, xb.r_hi) -
                                std::max(rb.r_lo, xb.r_lo);
                    if (ov > best_ov) { best_ov = ov; best = &xb; }
                }
                if (!best || best_ov <= 0.0) { covered = false; continue; }
                double scale = std::max(rb.r_hi - rb.r_lo, 1e-9 * (1.0 + rb.r_hi));
                double e = std::max(std::fabs(best->r_lo - rb.r_lo),
                                    std::fabs(best->r_hi - rb.r_hi)) /
                           scale;
                worst = std::max(worst, e);
                if (e >= 5e-3) edges = false;
            }
            if (!ref_sig.empty()) worst_edge_rel.push_back(worst);
            if (covered) ++n_sig_covered;
            if (covered && edges && sig_count) ++n_sig_edge_ok;
            if (fb.bands.size() == ref.size()) ++n_full_parity;

            // fast bands that match no significant ref band = razor extras
            int razor_extra = 0;
            for (auto& xb : fb.bands) {
                bool matches_sig = false;
                for (auto& rb : ref_sig) {
                    double ov = std::min(rb.r_hi, xb.r_hi) -
                                std::max(rb.r_lo, xb.r_lo);
                    if (ov > 0.0) { matches_sig = true; break; }
                }
                if (!matches_sig) ++razor_extra;
            }
            total_razor_extra += razor_extra;

            if (!(covered && edges && sig_count)) {
                char buf[160];
                std::snprintf(buf, sizeof buf,
                              "SIG MISMATCH  sig fast=%zu ref=%zu  covered=%d "
                              "edges=%d  worst=%.2e%s",
                              fast_sig.size(), ref_sig.size(), covered, edges,
                              worst, ref_uncertain ? " [ref UNCERTAIN]" : "");
                mismatches.push_back({c.name, buf});
            } else if (razor_extra > 0) {
                char buf[160];
                std::snprintf(buf, sizeof buf,
                              "ok + %d razor extra (sig parity clean)",
                              razor_extra);
                mismatches.push_back({c.name, buf});
            }
        } else {
            bails.push_back({c.name, fb.reason});
        }

        // --- timing ---
        double bf = 1e30, bc = 1e30;
        double last_calls = 0;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            PointImages s2 = binary_point_images(pf);
            FastBands f2 = fast_bands(pf, s2);
            auto t1 = clk::now();
            TopologyResult tr2 = classify_cells(pf);
            auto t2 = clk::now();
            checksum += f2.bands.size() + tr2.cells.size();
            double df = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double dc = std::chrono::duration<double, std::milli>(t2 - t1).count();
            if (df < bf) { bf = df; last_calls = f2.has_image_calls; }
            bc = std::min(bc, dc);
        }
        t_fast.push_back(bf);
        t_cls.push_back(bc);
        hi_calls.push_back(last_calls);
    }

    const int N = (int)cases.size();
    std::fprintf(stderr, "== discovery parity (%d cases) ==\n", N);
    std::fprintf(stderr, "  fast_bands reliable          : %d / %d  "
                 "(rest fail closed -> D14)\n", n_reliable, N);
    std::fprintf(stderr, "  significant-band count match : %d / %d\n",
                 n_sig_count_match, n_reliable);
    std::fprintf(stderr, "  significant ref bands covered: %d / %d\n",
                 n_sig_covered, n_reliable);
    std::fprintf(stderr, "  + edges agree (<5e-3 width)  : %d / %d\n",
                 n_sig_edge_ok, n_reliable);
    std::fprintf(stderr, "  exact band-set parity        : %d / %d\n",
                 n_full_parity, n_reliable);
    std::fprintf(stderr, "  razor extras (fast>ref, sub-grid arc): %d total\n",
                 total_razor_extra);
    std::fprintf(stderr, "  classify_cells UNCERTAIN     : %d / %d\n",
                 n_uncertain_ref, N);
    if (!worst_edge_rel.empty())
        std::fprintf(stderr,
                     "  sig band-edge rel error      : median %.2e  p90 %.2e  "
                     "max %.2e\n",
                     pct(worst_edge_rel, 50), pct(worst_edge_rel, 90),
                     pct(worst_edge_rel, 100));

    if (!bails.empty()) {
        std::fprintf(stderr, "\n  -- fast_bands bailed (%zu) --\n", bails.size());
        for (auto& b : bails)
            std::fprintf(stderr, "     %-16s %s\n", b.first.c_str(),
                         b.second.c_str());
    }
    if (!mismatches.empty()) {
        std::fprintf(stderr, "\n  -- notes (%zu) --\n", mismatches.size());
        for (auto& m : mismatches)
            std::fprintf(stderr, "     %-16s %s\n", m.first.c_str(),
                         m.second.c_str());
    }

    std::fprintf(stderr, "\n== wall time, best-of-%d per case, %d cases ==\n",
                 reps, N);
    std::fprintf(stderr,
                 "  point_images + fast_bands : median %7.4f  p90 %7.4f  "
                 "p99 %7.4f  mean %7.4f\n",
                 pct(t_fast, 50), pct(t_fast, 90), pct(t_fast, 99),
                 mean(t_fast));
    std::fprintf(stderr,
                 "  classify_cells (D14 path) : median %7.4f  p90 %7.4f  "
                 "p99 %7.4f  mean %7.4f\n",
                 pct(t_cls, 50), pct(t_cls, 90), pct(t_cls, 99), mean(t_cls));
    std::fprintf(stderr,
                 "  speedup (mean / median)   : %.2fx / %.2fx\n",
                 mean(t_cls) / std::max(mean(t_fast), 1e-12),
                 pct(t_cls, 50) / std::max(pct(t_fast, 50), 1e-12));
    std::fprintf(stderr,
                 "  has_image calls / epoch   : median %.0f  p90 %.0f  max %.0f\n",
                 pct(hi_calls, 50), pct(hi_calls, 90), pct(hi_calls, 100));

    std::fprintf(stderr, "\nchecksum %.3f\n", checksum);
    return 0;
}

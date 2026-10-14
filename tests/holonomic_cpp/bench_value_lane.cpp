// ATPT holonomic solver -- Phase D value lane micro-benchmark.
//
// Checkpoint sec.33.  Two variants through the SAME production router
// (finite_source_binary), one executable:
//
//   VF  RequestedOutput::kValueJacobian -- today's fused value+F_half+
//       5-Jacobian+dmu/du pass, then discard the Jacobian.  This is what
//       the production kValue path pays TODAY (checkpoint sec.32.3).
//   VD  RequestedOutput::kValue         -- the Phase D value lane:
//       epoch_value / epoch_value_prepared.  F0 from the identical arc
//       discovery; F_half only when u != 0 (same 64-node sweep); no
//       derivative state, no chain rule, no dP at the angular nodes.
//
// Reports, per {u=0 uniform, u=case} and per {cold, L2-prepared}:
//   - wall-clock median / p90 / p95 / p99 / max, best-of-reps per case
//   - VF/VD speedup at each percentile
//   - mu parity |mu_VD - mu_VF| / |mu_VF|  (target: bit-identical / ~1e-15)
//   - status-change count VD vs VF  (must be 0)
//
// Build: tests/holonomic_cpp/CMakeLists.txt (isolated project).
// Run:   taskset -c 0-7 ./build-holonomic-m7/bench_value_lane [cases.tsv] [reps]

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

#include "lcbinint/magnification/holonomic/finite_source_binary.hpp"

using namespace lcbinint::holonomic;
using clk = std::chrono::steady_clock;

namespace {

struct BC {
    double xs, ys, rho, q, a;
    int bary;
    double u;
    double tj;
    std::string name;
};

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1 - frac) + v[lo + 1] * frac;
}

// One short straight track per case (same synthesis as
// test_finite_source_binary / bench_holonomic_trajectory) so the prepared
// route actually exercises L2 warm-D14 reuse.
std::vector<LensParams> track(const BC& c, int N) {
    const double L = std::max(8.0 * c.rho, 0.5 * c.a);
    const double dx = L / N * 3.0 / std::sqrt(10.0);
    const double dy = L / N * 1.0 / std::sqrt(10.0);
    std::vector<LensParams> out;
    out.reserve(N);
    for (int e = 0; e < N; ++e) {
        LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        p.xs = c.xs + (e - N / 2) * dx;
        p.ys = c.ys + (e - N / 2) * dy;
        out.push_back(p);
    }
    return out;
}

struct Acc {
    std::vector<double> tvf, tvd;   // per-epoch best ms
    std::vector<double> mupar;      // |dmu|/|mu| VD vs VF
    int status_changes = 0;
    int epochs = 0;
    double checksum = 0.0;
};

void report(const char* tag, Acc& a) {
    std::fprintf(stderr, "\n-- %s  (%d epochs) --\n", tag, a.epochs);
    auto row = [](const char* t, std::vector<double>& v) {
        std::fprintf(stderr,
            "   %-14s median %8.4f  p90 %8.4f  p95 %8.4f  p99 %8.4f  max %8.4f ms\n",
            t, pct(v, 50), pct(v, 90), pct(v, 95), pct(v, 99),
            *std::max_element(v.begin(), v.end()));
    };
    row("VF fused", a.tvf);
    row("VD value lane", a.tvd);
    std::fprintf(stderr,
        "   speedup VF/VD  median %.3fx  p90 %.3fx  p95 %.3fx  p99 %.3fx\n",
        pct(a.tvf, 50) / pct(a.tvd, 50), pct(a.tvf, 90) / pct(a.tvd, 90),
        pct(a.tvf, 95) / pct(a.tvd, 95), pct(a.tvf, 99) / pct(a.tvd, 99));
    std::fprintf(stderr,
        "   mu parity  |dmu|/|mu|  median %.3e  p99 %.3e  max %.3e\n",
        pct(a.mupar, 50), pct(a.mupar, 99), pct(a.mupar, 100));
    std::fprintf(stderr, "   status-change VD vs VF (must be 0): %d\n",
                 a.status_changes);
}

FiniteSourceRequest mkreq(const LensParams& p, double u, RequestedOutput o) {
    FiniteSourceRequest r;
    r.params = p;
    r.u = u;
    r.output = o;
    r.n_r = 64;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 40;
    const int N = 16;  // epochs per synthesised track

    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 2;
    }
    std::vector<BC> cases;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        BC c;
        if (!(ls >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
              c.tj >> c.name))
            continue;
        cases.push_back(c);
    }
    std::fprintf(stderr,
                 "loaded %zu cases, reps=%d, N=%d epochs/track, n_r=64\n",
                 cases.size(), reps, N);
    std::fprintf(stderr,
                 "HOLO_HOLONOMIC_TRANSPORT=%s  HOLO_MV_TRANSPORT_LEGACY=%s\n",
                 std::getenv("HOLO_HOLONOMIC_TRANSPORT") ? "1" : "0",
                 std::getenv("HOLO_MV_TRANSPORT_LEGACY") ? "1" : "0");

    Acc cold_u0, cold_uc, prep_u0, prep_uc;

    auto do_cold = [&](Acc& acc, bool uniform) {
        for (const auto& c : cases) {
            LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
            const double u = uniform ? 0.0 : c.u;
            FiniteSourceRequest rf = mkreq(p, u, RequestedOutput::kValueJacobian);
            FiniteSourceRequest rd = mkreq(p, u, RequestedOutput::kValue);

            FiniteSourceOutcome of = finite_source_binary(rf);
            FiniteSourceOutcome od = finite_source_binary(rd);
            double base = std::fabs(of.mu) > 0.0 ? std::fabs(of.mu) : 1.0;
            acc.mupar.push_back(std::fabs(od.mu - of.mu) / base);
            if ((od.status == Status::OK) != (of.status == Status::OK))
                ++acc.status_changes;
            acc.checksum += of.mu + od.mu;
            ++acc.epochs;

            double bf = 1e30, bd = 1e30;
            for (int r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                auto x = finite_source_binary(rf);
                auto t1 = clk::now();
                auto y = finite_source_binary(rd);
                auto t2 = clk::now();
                bf = std::min(bf, std::chrono::duration<double, std::milli>(t1 - t0).count());
                bd = std::min(bd, std::chrono::duration<double, std::milli>(t2 - t1).count());
                acc.checksum += x.mu + y.mu;
            }
            acc.tvf.push_back(bf);
            acc.tvd.push_back(bd);
        }
    };

    auto do_prep = [&](Acc& acc, bool uniform) {
        PreparedReuseConfig cfg = default_reuse_config();
        for (const auto& c : cases) {
            const double u = uniform ? 0.0 : c.u;
            std::vector<LensParams> tr = track(c, N);

            // parity + status pass (fresh caches)
            {
                PreparedEpochGeometry sf{}, sd{};
                for (const auto& p : tr) {
                    FiniteSourceRequest rf =
                        mkreq(p, u, RequestedOutput::kValueJacobian);
                    FiniteSourceRequest rd = mkreq(p, u, RequestedOutput::kValue);
                    FiniteSourceOutcome of =
                        finite_source_binary_prepared(rf, sf, cfg);
                    FiniteSourceOutcome od =
                        finite_source_binary_prepared(rd, sd, cfg);
                    double base = std::fabs(of.mu) > 0.0 ? std::fabs(of.mu) : 1.0;
                    acc.mupar.push_back(std::fabs(od.mu - of.mu) / base);
                    if ((od.status == Status::OK) != (of.status == Status::OK))
                        ++acc.status_changes;
                    acc.checksum += of.mu + od.mu;
                    ++acc.epochs;
                }
            }
            // timing pass: best-of-reps per epoch, each rep rolls a fresh
            // cache through the whole track so epoch e always sees the same
            // reuse state.
            std::vector<double> bf(tr.size(), 1e30), bd(tr.size(), 1e30);
            for (int r = 0; r < reps; ++r) {
                PreparedEpochGeometry sf{}, sd{};
                for (size_t e = 0; e < tr.size(); ++e) {
                    FiniteSourceRequest rf =
                        mkreq(tr[e], u, RequestedOutput::kValueJacobian);
                    FiniteSourceRequest rd =
                        mkreq(tr[e], u, RequestedOutput::kValue);
                    auto t0 = clk::now();
                    auto x = finite_source_binary_prepared(rf, sf, cfg);
                    auto t1 = clk::now();
                    auto y = finite_source_binary_prepared(rd, sd, cfg);
                    auto t2 = clk::now();
                    bf[e] = std::min(bf[e], std::chrono::duration<double, std::milli>(t1 - t0).count());
                    bd[e] = std::min(bd[e], std::chrono::duration<double, std::milli>(t2 - t1).count());
                    acc.checksum += x.mu + y.mu;
                }
            }
            for (size_t e = 0; e < tr.size(); ++e) {
                acc.tvf.push_back(bf[e]);
                acc.tvd.push_back(bd[e]);
            }
        }
    };

    do_cold(cold_u0, true);
    do_cold(cold_uc, false);
    do_prep(prep_u0, true);
    do_prep(prep_uc, false);

    report("COLD  u = 0 (uniform)", cold_u0);
    report("COLD  u = case", cold_uc);
    report("PREPARED (L2 warm-D14)  u = 0 (uniform)", prep_u0);
    report("PREPARED (L2 warm-D14)  u = case", prep_uc);

    std::fprintf(stderr, "\nchecksum %.6f\n",
                 cold_u0.checksum + cold_uc.checksum + prep_u0.checksum +
                     prep_uc.checksum);
    return 0;
}

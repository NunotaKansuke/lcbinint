// 3-way finite-source magnification + 5-Jacobian benchmark.
//
//   (A) algebraic-boundary   -- experimental_algebraic_boundary_binary_{mag,jacobian}
//                               (analytic forward-mode Jacobian, ForwardJet<5>)
//   (H) holonomic M7         -- lcbinint::holonomic::epoch_jacobian  (fused value+Jac)
//   (I) incumbent            -- binary_ray_shooting value + central-FD x11 Jacobian
//                               (numbers read from evidence/holonomic/baseline_M0.json,
//                                re-exported to the extended TSV)
//
// Same 108 (config, u) points, same CoM frame mapping (Mapping B), same linear-LD
// convention (limb_darkening_c = u, d = 0), same fail-closed policy.
//
// Build (from the holonomic worktree):
//   AB=/rogue1_8/nunota/lcbinint/.claude/worktrees/algebraic-bench-cc5e55d
//   g++ -std=c++17 -O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno \
//       -I src -I $AB/src -I $AB/include \
//       tests/holonomic_cpp/bench_three_way.cpp \
//       -Wl,--start-group $AB/build-bench/liblcbinint_lightcurve.a \
//                         $AB/build-bench/liblcbinint_magnification.a -Wl,--end-group \
//       -L/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lm -lquadmath \
//       -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib \
//       -o build-holonomic-m7/bench_three_way
//
// Run: taskset -c 0-7 ./build-holonomic-m7/bench_three_way /tmp/bench_cases_ext.tsv [reps]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"

namespace hol = lcbinint::holonomic;
namespace mg = lcbinint::magnification;
using clk = std::chrono::steady_clock;

struct Case {
    double xs, ys, rho, q, a;
    int bary;
    double u;
    std::string name;
    double m0_mu, m0_tvalue, m0_tjac;
    std::string m0_status;
    std::array<double, 5> m0_grad;
};

static double pct(std::vector<double> v, double p) {
    v.erase(std::remove_if(v.begin(), v.end(),
            [](double x){ return !std::isfinite(x); }), v.end());
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1 - frac) + v[lo + 1] * frac;
}
static double med(std::vector<double> v) { return pct(v, 50); }

struct Stats {
    double p50 = 0, p90 = 0, p95 = 0, p99 = 0, mx = 0;
    size_t n = 0;
    void fill(std::vector<double> v) {
        v.erase(std::remove_if(v.begin(), v.end(),
                [](double x){ return !std::isfinite(x); }), v.end());
        n = v.size();
        p50 = pct(v, 50); p90 = pct(v, 90); p95 = pct(v, 95);
        p99 = pct(v, 99);
        std::sort(v.begin(), v.end());
        mx = v.empty() ? 0 : v.back();
    }
};
static void print_stats(const char* tag, const Stats& s) {
    std::printf("    %-28s  n=%3zu  p50 %8.3f  p90 %8.3f  p95 %8.3f  p99 %8.3f  max %8.3f  (ms)\n",
                tag, s.n, s.p50, s.p90, s.p95, s.p99, s.mx);
}

// central Richardson FD of a scalar functor f(params) over 5 params.
// Relative step with a per-parameter floor; the q and separation steps are
// additionally capped to a fraction of the value so a small planetary q is
// never stepped across zero or the q<->1/q branch.
template <class F>
static std::array<double, 5> richardson5(F f, std::array<double, 5> p) {
    const std::array<double, 5> rel{3e-5, 3e-5, 3e-5, 3e-5, 3e-5};
    const std::array<double, 5> flo{2e-6, 2e-6, 5e-7, 1e-7, 2e-6};
    std::array<double, 5> out{};
    for (int j = 0; j < 5; ++j) {
        double h = std::max(rel[j] * std::fabs(p[j]), flo[j]);
        if (j == 3) h = std::min(h, 0.25 * std::fabs(p[j]));  // q
        if (j == 4) h = std::min(h, 0.05 * std::fabs(p[j]));  // separation
        auto step = [&](double hh) {
            auto pp = p; pp[j] += hh; double fp = f(pp);
            pp = p; pp[j] -= hh; double fm = f(pp);
            return (fp - fm) / (2 * hh);
        };
        double d1 = step(h);
        double d2 = step(h * 0.5);
        out[j] = (4 * d2 - d1) / 3;
    }
    return out;
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases_ext.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 100;
    const bool do_fd = std::getenv("SKIP_FD") == nullptr;
    const double Q_PLANETARY = 1.5e-3;  // q <= this => planetary low-q regime

    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
    std::vector<Case> cs;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        Case c;
        if (!(ls >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
              c.name >> c.m0_mu >> c.m0_tvalue >> c.m0_tjac >> c.m0_status >>
              c.m0_grad[0] >> c.m0_grad[1] >> c.m0_grad[2] >> c.m0_grad[3] >>
              c.m0_grad[4]))
            continue;
        cs.push_back(c);
    }
    std::fprintf(stderr, "loaded %zu cases, reps=%d\n", cs.size(), reps);

    // ---- per-backend, per-case accumulators -------------------------------
    struct Row {
        std::string name; double q, u; bool planetary;
        // latency (ms)
        double a_val = 0, a_jac = 0, h_fused = 0;
        // values
        double a_mu = 0, h_mu = 0;
        bool a_ok = false, a_grad_ok = false, a_hang = false;
        std::string a_reason;
        hol::Status h_status = hol::Status::OK;
        // jac self-consistency (worst rel err vs own Richardson FD)
        double a_jac_fd = 0, h_jac_fd = 0;
        // cross-check holonomic vs algebraic on [dx,dy,drho]
        double cross_jac = 0;
    };
    std::vector<Row> rows;

    for (const auto& c : cs) {
        Row R;
        R.name = c.name; R.q = c.q; R.u = c.u;
        R.planetary = (c.q <= Q_PLANETARY && c.q > 0.0);

        const double m1frac = c.q / (1.0 + c.q);
        const lcbinint::SourcePosition src{c.xs - m1frac * c.a, c.ys};

        // ---------- algebraic-boundary --------------------------------------
        mg::FiniteSourceSettings st;
        st.limb_darkening_c = c.u;
        st.limb_darkening_d = 0.0;
        mg::FiniteSourceMagnifier magn(st);

        // rand002: the local algebraic-boundary value path does not terminate
        // (unbounded allocation).  Verified in isolation; skip so the harness
        // completes and record it as a hard (non-fail-closed) failure.
        const bool alg_hang = (c.name == "rand002");
        R.a_hang = alg_hang;
        mg::AlgebraicBoundaryResult av0{};
        lcbinint::magnification::autodiff::AlgebraicBoundaryJacobian aj0{};
        if (!alg_hang) {
            av0 = magn.experimental_algebraic_boundary_binary_mag(c.a, c.q, src, c.rho);
            aj0 = magn.experimental_algebraic_boundary_binary_jacobian(c.a, c.q, src, c.rho);
        }
        R.a_mu = av0.magnification;
        R.a_ok = av0.success;
        R.a_reason = av0.unsafe_reason.empty() ? aj0.unsafe_reason : av0.unsafe_reason;
        R.a_grad_ok = aj0.success && aj0.grad_reliable;

        double best = 1e30;
        for (int r = 0; r < reps && !alg_hang; ++r) {
            auto t0 = clk::now();
            auto v = magn.experimental_algebraic_boundary_binary_mag(c.a, c.q, src, c.rho);
            auto t1 = clk::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (v.magnification == -12345.0) std::printf("x");
        }
        R.a_val = alg_hang ? std::nan("") : best;

        best = 1e30;
        for (int r = 0; r < reps && !alg_hang; ++r) {
            auto t0 = clk::now();
            auto j = magn.experimental_algebraic_boundary_binary_jacobian(c.a, c.q, src, c.rho);
            auto t1 = clk::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (j.magnification == -12345.0) std::printf("x");
        }
        R.a_jac = alg_hang ? std::nan("") : best;
        if (alg_hang) R.a_reason = "HANG_nonterminating";

        // ---------- holonomic M7 ------------------------------------------
        hol::LensParams p{c.xs, c.ys, c.rho, c.q, c.a, (bool)c.bary};
        auto ej0 = hol::epoch_jacobian(p, c.u, 64);
        R.h_mu = ej0.mu;
        R.h_status = ej0.status;

        best = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now();
            auto ej = hol::epoch_jacobian(p, c.u, 64);
            auto t1 = clk::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (ej.mu == -12345.0) std::printf("x");
        }
        R.h_fused = best;

        // ---------- Jacobian self-consistency (own Richardson FD) ---------
        if (do_fd && R.a_grad_ok && c.q > 3e-3) {
            // algebraic params: [source_x, source_y, rho, q, separation]
            std::array<double, 5> ap{src.x, src.y, c.rho, c.q, c.a};
            auto af = [&](std::array<double, 5> pp) {
                lcbinint::SourcePosition s2{pp[0], pp[1]};
                return magn.experimental_algebraic_boundary_binary_mag(pp[4], pp[3], s2, pp[2]).magnification;
            };
            auto fd = richardson5(af, ap);
            double w = 0;
            for (int j = 0; j < 5; ++j) {
                double den = std::max(1e-9, std::fabs(fd[j]));
                w = std::max(w, std::fabs(aj0.jacobian[j] - fd[j]) / den);
            }
            R.a_jac_fd = w;
        } else {
            R.a_jac_fd = std::nan("");
        }
        if (do_fd && R.h_status == hol::Status::OK && c.q > 3e-3) {
            std::array<double, 5> hp{c.xs, c.ys, c.rho, c.q, c.a};
            auto hf = [&](std::array<double, 5> pp) {
                hol::LensParams p2{pp[0], pp[1], pp[2], pp[3], pp[4], (bool)c.bary};
                return hol::epoch_jacobian(p2, c.u, 64).mu;
            };
            auto fd = richardson5(hf, hp);
            double w = 0;
            for (int j = 0; j < 5; ++j) {
                double den = std::max(1e-9, std::fabs(fd[j]));
                w = std::max(w, std::fabs(ej0.grad_mu[j] - fd[j]) / den);
            }
            R.h_jac_fd = w;
        } else {
            R.h_jac_fd = std::nan("");
        }

        // ---------- cross-check holonomic vs algebraic on [dx, dy, drho] ---
        if (R.a_grad_ok && R.h_status == hol::Status::OK) {
            double w = 0;
            for (int j = 0; j < 3; ++j) {
                double den = std::max(1e-6, std::fabs(ej0.grad_mu[j]));
                w = std::max(w, std::fabs(ej0.grad_mu[j] - aj0.jacobian[j]) / den);
            }
            R.cross_jac = w;
        } else {
            R.cross_jac = std::nan("");
        }

        rows.push_back(R);
        std::fprintf(stderr, "  %-16s u=%.1f  A[val %6.3f jac %6.3f ok=%d gok=%d]  H[%6.3f st=%s]  mu A/H/M0 %.6g/%.6g/%.6g\n",
                     c.name.c_str(), c.u, R.a_val, R.a_jac, (int)R.a_ok, (int)R.a_grad_ok,
                     R.h_fused, hol::to_string(R.h_status), R.a_mu, R.h_mu, c.m0_mu);
    }

    // ===================================================================
    // Reporting
    // ===================================================================
    auto slice = [&](std::function<bool(const Row&)> pred) {
        std::vector<double> av, aj, hf;
        for (auto& r : rows) if (pred(r)) {
            av.push_back(r.a_val); aj.push_back(r.a_jac); hf.push_back(r.h_fused);
        }
        Stats sav, saj, shf;
        sav.fill(av); saj.fill(aj); shf.fill(hf);
        std::printf("  n=%zu\n", av.size());
        print_stats("algebraic  value-only", sav);
        print_stats("algebraic  value+5-Jac", saj);
        print_stats("holonomic  fused value+5-Jac", shf);
    };

    std::printf("\n================ LATENCY ================\n");
    std::printf("[ALL 108]\n");            slice([](const Row&){ return true; });
    std::printf("[ordinary binary  (q > %.1e)]\n", Q_PLANETARY);
    slice([&](const Row& r){ return !r.planetary; });
    std::printf("[planetary low-q  (0 < q <= %.1e)]\n", Q_PLANETARY);
    slice([&](const Row& r){ return r.planetary; });
    std::printf("[uniform source (u = 0)]\n");
    slice([](const Row& r){ return r.u == 0.0; });
    std::printf("[linear limb darkening (u > 0)]\n");
    slice([](const Row& r){ return r.u > 0.0; });

    // incumbent (from M0 json) -- same slicing
    auto slice_inc = [&](std::function<bool(const Row&, const Case&)> pred) {
        std::vector<double> tv, tj;
        for (size_t i = 0; i < rows.size(); ++i) if (pred(rows[i], cs[i])) {
            tv.push_back(cs[i].m0_tvalue * 1e3);
            tj.push_back(cs[i].m0_tjac * 1e3);
        }
        Stats stv, stj; stv.fill(tv); stj.fill(tj);
        print_stats("incumbent  value-only", stv);
        print_stats("incumbent  value+FD-Jac(x11)", stj);
    };
    std::printf("[incumbent  ALL 108]\n");     slice_inc([](const Row&, const Case&){ return true; });
    std::printf("[incumbent  ordinary binary]\n");
    slice_inc([&](const Row& r, const Case&){ return !r.planetary; });
    std::printf("[incumbent  planetary low-q]\n");
    slice_inc([&](const Row& r, const Case&){ return r.planetary; });
    std::printf("[incumbent  uniform u=0]\n");
    slice_inc([&](const Row& r, const Case&){ return r.u == 0.0; });
    std::printf("[incumbent  linear LD u>0]\n");
    slice_inc([&](const Row& r, const Case&){ return r.u > 0.0; });

    // ===================================================================
    // accuracy + failure policy
    // ===================================================================
    std::printf("\n================ VALUE ACCURACY (rel. to binary_ray_shooting M0 mu) ================\n");
    auto acc = [&](std::function<bool(const Row&, const Case&)> pred) {
        std::vector<double> da, dh, dah;
        for (size_t i = 0; i < rows.size(); ++i) {
            const Row& r = rows[i]; const Case& c = cs[i];
            if (!pred(r, c)) continue;
            if (c.m0_status == "OK" && std::isfinite(c.m0_mu) && c.m0_mu > 0) {
                if (r.a_ok)  da.push_back(std::fabs(r.a_mu - c.m0_mu) / c.m0_mu);
                if (r.h_status == hol::Status::OK)
                    dh.push_back(std::fabs(r.h_mu - c.m0_mu) / c.m0_mu);
            }
            if (r.a_ok && r.h_status == hol::Status::OK && r.h_mu > 0)
                dah.push_back(std::fabs(r.a_mu - r.h_mu) / r.h_mu);
        }
        std::printf("    algebraic vs M0   n=%3zu  median %.2e  p90 %.2e  max %.2e\n",
                    da.size(), med(da), pct(da, 90), da.empty()?0:*std::max_element(da.begin(),da.end()));
        std::printf("    holonomic vs M0   n=%3zu  median %.2e  p90 %.2e  max %.2e\n",
                    dh.size(), med(dh), pct(dh, 90), dh.empty()?0:*std::max_element(dh.begin(),dh.end()));
        std::printf("    algebraic vs holo n=%3zu  median %.2e  p90 %.2e  max %.2e\n",
                    dah.size(), med(dah), pct(dah, 90), dah.empty()?0:*std::max_element(dah.begin(),dah.end()));
    };
    std::printf("[ALL]\n");               acc([](const Row&, const Case&){ return true; });
    std::printf("[ordinary binary]\n");   acc([&](const Row& r, const Case&){ return !r.planetary; });
    std::printf("[planetary low-q]\n");    acc([&](const Row& r, const Case&){ return r.planetary; });
    std::printf("[uniform u=0]\n");        acc([](const Row& r, const Case&){ return r.u == 0.0; });
    std::printf("[linear LD u>0]\n");      acc([](const Row& r, const Case&){ return r.u > 0.0; });

    std::printf("\n================ JACOBIAN QUALITY ================\n");
    {
        std::vector<double> ajf, hjf, cj;
        for (auto& r : rows) {
            if (std::isfinite(r.a_jac_fd)) ajf.push_back(r.a_jac_fd);
            if (std::isfinite(r.h_jac_fd)) hjf.push_back(r.h_jac_fd);
            if (std::isfinite(r.cross_jac)) cj.push_back(r.cross_jac);
        }
        std::printf("    algebraic analytic vs own Richardson-FD   n=%3zu  median %.2e  p90 %.2e  max %.2e\n",
                    ajf.size(), med(ajf), pct(ajf,90), ajf.empty()?0:*std::max_element(ajf.begin(),ajf.end()));
        std::printf("    holonomic analytic vs own Richardson-FD   n=%3zu  median %.2e  p90 %.2e  max %.2e\n",
                    hjf.size(), med(hjf), pct(hjf,90), hjf.empty()?0:*std::max_element(hjf.begin(),hjf.end()));
        std::printf("    holo vs algebraic  d/d[x,y,rho] (shared)  n=%3zu  median %.2e  p90 %.2e  max %.2e\n",
                    cj.size(), med(cj), pct(cj,90), cj.empty()?0:*std::max_element(cj.begin(),cj.end()));
    }

    std::printf("\n================ FAILURE / FAIL-CLOSED POLICY ================\n");
    {
        int a_val_fail = 0, a_grad_fail = 0, a_silent = 0;
        int h_nonok = 0, h_silent = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            const Row& r = rows[i]; const Case& c = cs[i];
            bool ref_ok = (c.m0_status == "OK" && c.m0_mu > 0 && std::isfinite(c.m0_mu));
            // algebraic
            if (!r.a_ok) ++a_val_fail;
            if (!r.a_grad_ok) ++a_grad_fail;
            // silent failure := backend says OK but value is >5% off a good reference
            if (r.a_ok && ref_ok && std::fabs(r.a_mu - c.m0_mu) / c.m0_mu > 0.05) {
                ++a_silent;
                std::printf("    [SILENT? algebraic] %-14s u=%.1f  mu=%.6g  M0=%.6g\n",
                            c.name.c_str(), c.u, r.a_mu, c.m0_mu);
            }
            // holonomic
            if (r.h_status != hol::Status::OK) ++h_nonok;
            if (r.h_status == hol::Status::OK && ref_ok &&
                std::fabs(r.h_mu - c.m0_mu) / c.m0_mu > 0.05) {
                ++h_silent;
                std::printf("    [SILENT? holonomic] %-14s u=%.1f  mu=%.6g  M0=%.6g\n",
                            c.name.c_str(), c.u, r.h_mu, c.m0_mu);
            }
        }
        std::printf("    algebraic : value-fail %d / 108   grad-unreliable(fail-closed) %d / 108   silent %d\n",
                    a_val_fail, a_grad_fail, a_silent);
        std::printf("    holonomic : status != OK (fail-closed) %d / 108   silent %d\n", h_nonok, h_silent);
        std::printf("    incumbent : (from M0 json) non-OK status counted separately\n");
        int inc_nonok = 0;
        for (auto& c : cs) if (c.m0_status != "OK") ++inc_nonok;
        std::printf("    incumbent : status != OK %d / 108\n", inc_nonok);
    }

    // machine-readable dump for the checkpoint
    std::printf("\n#CSV name,u,q,planetary,a_val_ms,a_jac_ms,h_fused_ms,a_mu,h_mu,m0_mu,a_ok,a_grad_ok,h_status,inc_tval_ms,inc_tjac_ms\n");
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& r = rows[i]; const Case& c = cs[i];
        std::printf("#CSV %s,%.1f,%.6g,%d,%.4f,%.4f,%.4f,%.8g,%.8g,%.8g,%d,%d,%s,%.4f,%.4f\n",
                    c.name.c_str(), c.u, c.q, (int)r.planetary,
                    r.a_val, r.a_jac, r.h_fused, r.a_mu, r.h_mu, c.m0_mu,
                    (int)r.a_ok, (int)r.a_grad_ok, hol::to_string(r.h_status),
                    c.m0_tvalue * 1e3, c.m0_tjac * 1e3);
    }
    return 0;
}

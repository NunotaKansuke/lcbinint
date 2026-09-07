// Algorithm benchmark (class A): every backend forced to run the deep
// finite-source solve.  All point-source / hexadecapole / multipole shortcuts
// disabled.  This is the ONLY benchmark that may be called an "algebraic vs
// holonomic pure speed comparison".  Production routing numbers live in
// bench_three_way (class B) and must not be mixed with these.
//
// Backends:
//   H   holonomic M7/M8   -- epoch_jacobian (fused value + 5-Jac), shortcut-free
//   A1  algebraic current -- ALG_FORCE_FULLSOLVE, find_radial_bands (march)
//   A2  algebraic + D14   -- ALG_FORCE_FULLSOLVE + ALG_D14, find_radial_bands_d14
//   I   inverse-ray       -- experimental_raster_polar_binary_mag (deep polar)
//                            value only; central-FD x11 for the Jacobian
//
// Build (from the holonomic worktree):
//   AB=/rogue1_8/nunota/lcbinint/.claude/worktrees/algebraic-bench-cc5e55d
//   g++ -std=c++17 -O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno \
//     -I $AB/src -I $AB/include -I src tests/holonomic_cpp/bench_fullsolve.cpp \
//     -Wl,--start-group $AB/build-bench-d14/liblcbinint_lightcurve.a \
//                       $AB/build-bench-d14/liblcbinint_magnification.a -Wl,--end-group \
//     -L/home/nunota/.miniconda3/envs/myenv/lib -lgsl -lgslcblas -lm -lquadmath \
//     -Wl,-rpath,/home/nunota/.miniconda3/envs/myenv/lib -o build-holonomic-m7/bench_fullsolve
//
// Run: taskset -c 0-7 ./build-holonomic-m7/bench_fullsolve /tmp/bench_cases_ext.tsv [reps]

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
#include "lcbinint/magnification/point_source_magnifier.hpp"

namespace hol = lcbinint::holonomic;
namespace mg = lcbinint::magnification;
using clk = std::chrono::steady_clock;

struct Case {
    double xs, ys, rho, q, a; int bary; double u; std::string name;
    double m0_mu, m0_tvalue, m0_tjac; std::string m0_status;
    std::array<double, 5> m0_grad;
};

static double pct(std::vector<double> v, double p) {
    v.erase(std::remove_if(v.begin(), v.end(),
            [](double x){ return !std::isfinite(x); }), v.end());
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx; double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1 - frac) + v[lo + 1] * frac;
}
static double med(std::vector<double> v) { return pct(v, 50); }
static double meanv(std::vector<double> v) {
    v.erase(std::remove_if(v.begin(), v.end(),
            [](double x){ return !std::isfinite(x); }), v.end());
    if (v.empty()) return std::nan("");
    double s = 0; for (double x : v) s += x; return s / v.size();
}

struct Stats {
    double p50=0,p90=0,p95=0,p99=0,mx=0; size_t n=0;
    void fill(std::vector<double> v) {
        v.erase(std::remove_if(v.begin(), v.end(),
                [](double x){ return !std::isfinite(x); }), v.end());
        n = v.size();
        p50=pct(v,50); p90=pct(v,90); p95=pct(v,95); p99=pct(v,99);
        std::sort(v.begin(), v.end()); mx = v.empty()?0:v.back();
    }
};
static void ps(const char* tag, const Stats& s) {
    std::printf("    %-30s n=%3zu  p50 %8.3f  p90 %8.3f  p95 %8.3f  p99 %8.3f  max %8.3f\n",
                tag, s.n, s.p50, s.p90, s.p95, s.p99, s.mx);
}

template <class F>
static std::array<double,5> richardson5(F f, std::array<double,5> p) {
    const std::array<double,5> rel{3e-5,3e-5,3e-5,3e-5,3e-5};
    const std::array<double,5> flo{2e-6,2e-6,5e-7,1e-7,2e-6};
    std::array<double,5> out{};
    for (int j = 0; j < 5; ++j) {
        double h = std::max(rel[j]*std::fabs(p[j]), flo[j]);
        if (j==3) h = std::min(h, 0.25*std::fabs(p[j]));
        if (j==4) h = std::min(h, 0.05*std::fabs(p[j]));
        auto step=[&](double hh){ auto pp=p; pp[j]+=hh; double fp=f(pp);
            pp=p; pp[j]-=hh; double fm=f(pp); return (fp-fm)/(2*hh); };
        double d1=step(h), d2=step(h*0.5); out[j]=(4*d2-d1)/3;
    }
    return out;
}

static mg::AlgebraicBoundaryLens native_lens(double sep, double q_in) {
    const double s = std::abs(sep);
    const double q = std::abs(q_in) < 1.0 ? std::abs(q_in) : 1.0/std::abs(q_in);
    const double m1 = 1.0/(1.0+q), m2 = q*m1;
    const double a = std::abs(q_in) < 1.0 ? -s : s;
    return {a, m1, m2};
}

struct AlgRun {
    double val_ms=0, jac_ms=0, band_ms=0;
    double mu=0; bool ok=false, grad_ok=false;
    std::string reason;
    // diagnostics (from the value path)
    int root_solves=0, lens_evals=0, bands=0, subdiv=0;
    int quartic=0, quartic_fb=0, mv=0, mv_fb=0, cold=0;
    int tang_solve=0, tang_fb=0, d14_enum=0, d14_fb=0, d14_events=0;
    std::array<double,5> jac{};
};

static AlgRun run_algebraic(const Case& c, const mg::FiniteSourceMagnifier& magn,
                            const lcbinint::SourcePosition& src, int reps,
                            const std::vector<lcbinint::SourcePosition>& seeds) {
    AlgRun r;
    auto tw0 = clk::now();
    auto v0 = magn.experimental_algebraic_boundary_binary_mag(c.a, c.q, src, c.rho);
    double warm_ms = std::chrono::duration<double,std::milli>(clk::now()-tw0).count();
    auto j0 = magn.experimental_algebraic_boundary_binary_jacobian(c.a, c.q, src, c.rho);
    r.mu = v0.magnification; r.ok = v0.success;
    r.grad_ok = j0.success && j0.grad_reliable;
    r.reason = v0.unsafe_reason.empty() ? j0.unsafe_reason : v0.unsafe_reason;
    r.jac = j0.jacobian;
    const auto& d = v0.diagnostics;
    r.root_solves=d.root_solve_count; r.lens_evals=d.lens_evaluation_count;
    r.bands=d.radial_band_count; r.subdiv=d.radial_subdivision_count;
    r.quartic=d.real_quartic_solve_count; r.quartic_fb=d.real_quartic_fallback_count;
    r.mv=d.mv_transport_count; r.mv_fb=d.mv_transport_fallback_count;
    r.cold=d.cold_root_solve_count;
    r.tang_solve=d.tangency_solve_count; r.tang_fb=d.tangency_fallback_count;
    r.d14_enum=d.d14_band_enum_count; r.d14_fb=d.d14_fallback_count;
    r.d14_events=d.d14_event_count;

    // Pathology guard: if the warm-up deep solve already blew past 40 ms the
    // case is a forced-full-solve blowup (the production router would never
    // send it here).  Record the single measurement + counters, skip the loop.
    const bool pathological = !(warm_ms < 40.0);
    if (pathological) r.reason = r.reason.empty() ? "ALG_FULLSOLVE_BLOWUP" : r.reason;
    const int use_reps = pathological ? 1 : reps;
    double best = 1e30;
    for (int i = 0; i < use_reps; ++i) {
        auto t0 = clk::now();
        auto v = magn.experimental_algebraic_boundary_binary_mag(c.a, c.q, src, c.rho);
        auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double,std::milli>(t1-t0).count());
        if (v.magnification == -12345.0) std::printf("x");
    }
    r.val_ms = best;
    best = 1e30;
    for (int i = 0; i < use_reps; ++i) {
        auto t0 = clk::now();
        auto j = magn.experimental_algebraic_boundary_binary_jacobian(c.a, c.q, src, c.rho);
        auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double,std::milli>(t1-t0).count());
        if (j.magnification == -12345.0) std::printf("x");
    }
    r.jac_ms = best;

    // band-discovery phase, timed in isolation via the public export
    // (point-image seeds -- proxy for the internal augmented seed set).
    const auto lens = native_lens(c.a, c.q);
    mg::AlgebraicBoundarySettings abs_set;
    best = 1e30;
    for (int i = 0; i < use_reps; ++i) {
        auto t0 = clk::now();
        auto b = mg::experimental_algebraic_boundary_radial_bands(lens, src, c.rho, seeds, abs_set);
        auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double,std::milli>(t1-t0).count());
        if (!b.success && b.bands.size() == 999) std::printf("x");
    }
    r.band_ms = best;
    return r;
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/bench_cases_ext.tsv";
    int reps = argc > 2 ? std::atoi(argv[2]) : 60;
    const bool do_fd = std::getenv("SKIP_FD") == nullptr;
    const double QP = 1.5e-3;

    setenv("ALG_FORCE_FULLSOLVE", "1", 1);

    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
    std::vector<Case> cs; std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line); Case c;
        if (!(ls >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
              c.name >> c.m0_mu >> c.m0_tvalue >> c.m0_tjac >> c.m0_status >>
              c.m0_grad[0] >> c.m0_grad[1] >> c.m0_grad[2] >> c.m0_grad[3] >> c.m0_grad[4]))
            continue;
        cs.push_back(c);
    }
    std::fprintf(stderr, "loaded %zu cases, reps=%d (ALG_FORCE_FULLSOLVE=1)\n", cs.size(), reps);

    struct Row {
        std::string name; double q,u; bool planetary; bool caustic; bool smallrho; bool wide;
        double h_ms=0, h_mu=0; hol::Status h_st=hol::Status::OK;
        double i_ms=0, i_mu=0; bool i_ok=false;
        AlgRun a1, a2;
        double a1_fd=0, a2_fd=0, h_fd=0;
    };
    std::vector<Row> rows;

    for (const auto& c : cs) {
        Row R; R.name=c.name; R.q=c.q; R.u=c.u;
        R.planetary = (c.q > 0.0 && c.q <= QP);
        R.smallrho = (c.rho <= 0.02);
        R.wide = (c.a >= 2.0);
        R.caustic = (c.name.find("caustic")!=std::string::npos ||
                     c.name.find("resonant")!=std::string::npos ||
                     c.name.find("fold")!=std::string::npos ||
                     c.name.find("cusp")!=std::string::npos ||
                     c.name.find("very-close")!=std::string::npos);

        const double m1f = c.q/(1.0+c.q);
        const lcbinint::SourcePosition src{c.xs - m1f*c.a, c.ys};

        mg::FiniteSourceSettings st; st.limb_darkening_c=c.u; st.limb_darkening_d=0.0;
        mg::FiniteSourceMagnifier magn(st);
        mg::PointSourceMagnifier point;
        std::vector<lcbinint::SourcePosition> seeds;
        for (const auto& im : point.binary_images(c.a, c.q, src)) seeds.push_back(im.position);

        const bool alg_hang = (c.name == "rand002");

        // A1: current
        if (!alg_hang) {
            setenv("ALG_D14", "0", 1);
            R.a1 = run_algebraic(c, magn, src, reps, seeds);
            // A2: + D14
            setenv("ALG_D14", "1", 1);
            R.a2 = run_algebraic(c, magn, src, reps, seeds);
            setenv("ALG_D14", "0", 1);
        } else {
            R.a1.reason = R.a2.reason = "HANG_nonterminating";
            R.a1.val_ms = R.a1.jac_ms = R.a1.band_ms = std::nan("");
            R.a2.val_ms = R.a2.jac_ms = R.a2.band_ms = std::nan("");
        }

        // H
        hol::LensParams p{c.xs,c.ys,c.rho,c.q,c.a,(bool)c.bary};
        auto ej0 = hol::epoch_jacobian(p, c.u, 64);
        R.h_mu = ej0.mu; R.h_st = ej0.status;
        double best = 1e30;
        for (int i=0;i<reps;++i){ auto t0=clk::now(); auto e=hol::epoch_jacobian(p,c.u,64);
            auto t1=clk::now(); best=std::min(best,std::chrono::duration<double,std::milli>(t1-t0).count());
            if(e.mu==-12345.0) std::printf("x"); }
        R.h_ms = best;

        // I -- inverse-ray forced deep polar
        double iv = magn.experimental_raster_polar_binary_mag(c.a, c.q, src, c.rho);
        R.i_mu = iv; R.i_ok = std::isfinite(iv);
        best = 1e30;
        for (int i=0;i<reps;++i){ auto t0=clk::now();
            volatile double x = magn.experimental_raster_polar_binary_mag(c.a,c.q,src,c.rho); (void)x;
            auto t1=clk::now(); best=std::min(best,std::chrono::duration<double,std::milli>(t1-t0).count()); }
        R.i_ms = best;

        // Jacobian self-consistency vs own Richardson FD (ordinary binary only)
        const bool a1_sane = R.a1.ok && std::isfinite(c.m0_mu) && c.m0_mu>0 &&
            std::fabs(R.a1.mu-c.m0_mu)/c.m0_mu < 0.01 && R.a1.reason!="ALG_FULLSOLVE_BLOWUP";
        if (do_fd && c.q > 3e-3) {
            if (R.a1.grad_ok && a1_sane) {
                std::array<double,5> ap{src.x,src.y,c.rho,c.q,c.a};
                auto af=[&](std::array<double,5> pp){ lcbinint::SourcePosition s2{pp[0],pp[1]};
                    return magn.experimental_algebraic_boundary_binary_mag(pp[4],pp[3],s2,pp[2]).magnification; };
                auto fd=richardson5(af,ap); double w=0;
                for(int j=0;j<5;++j){ double den=std::max(1e-9,std::fabs(fd[j]));
                    w=std::max(w,std::fabs(R.a1.jac[j]-fd[j])/den); }
                R.a1_fd=w;
            } else R.a1_fd=std::nan("");
            R.a2_fd = R.a2.grad_ok ? 0.0 : std::nan("");  // A2 jac == A1 jac by construction (plan unchanged)
            if (R.a2.grad_ok && R.a1.grad_ok) {
                double w=0; for(int j=0;j<5;++j){ double den=std::max(1e-9,std::fabs(R.a1.jac[j]));
                    w=std::max(w,std::fabs(R.a2.jac[j]-R.a1.jac[j])/den); }
                R.a2_fd=w;
            }
            if (R.h_st==hol::Status::OK) {
                std::array<double,5> hp{c.xs,c.ys,c.rho,c.q,c.a};
                auto hf=[&](std::array<double,5> pp){ hol::LensParams p2{pp[0],pp[1],pp[2],pp[3],pp[4],(bool)c.bary};
                    return hol::epoch_jacobian(p2,c.u,64).mu; };
                auto fd=richardson5(hf,hp); double w=0;
                for(int j=0;j<5;++j){ double den=std::max(1e-9,std::fabs(fd[j]));
                    w=std::max(w,std::fabs(ej0.grad_mu[j]-fd[j])/den); }
                R.h_fd=w;
            } else R.h_fd=std::nan("");
        } else { R.a1_fd=R.a2_fd=R.h_fd=std::nan(""); }

        rows.push_back(R);
        std::fprintf(stderr,
            "  %-16s u=%.1f  A1[v %6.3f j %6.3f b %5.3f rs=%d q=%d mvfb=%d]  "
            "A2[v %6.3f j %6.3f b %5.3f rs=%d ev=%d d14=%d fb=%d]  H[%6.3f %s]  I[%6.3f]  mu A1/A2/H/M0 %.6g/%.6g/%.6g/%.6g\n",
            c.name.c_str(), c.u, R.a1.val_ms, R.a1.jac_ms, R.a1.band_ms, R.a1.root_solves, R.a1.quartic, R.a1.mv_fb,
            R.a2.val_ms, R.a2.jac_ms, R.a2.band_ms, R.a2.root_solves, R.a2.d14_events, R.a2.d14_enum, R.a2.d14_fb,
            R.h_ms, hol::to_string(R.h_st), R.i_ms, R.a1.mu, R.a2.mu, R.h_mu, c.m0_mu);
    }

    // ================= REPORT =================
    auto lat = [&](std::function<bool(const Row&)> pr) {
        std::vector<double> a1v,a1j,a2v,a2j,hf,iv,a1b,a2b;
        for (auto& r : rows) if (pr(r)) {
            a1v.push_back(r.a1.val_ms); a1j.push_back(r.a1.jac_ms);
            a2v.push_back(r.a2.val_ms); a2j.push_back(r.a2.jac_ms);
            a1b.push_back(r.a1.band_ms); a2b.push_back(r.a2.band_ms);
            hf.push_back(r.h_ms); iv.push_back(r.i_ms);
        }
        Stats s; s.fill(a1v); ps("A1 algebraic       value-only", s);
        s.fill(a1j); ps("A1 algebraic       value+5-Jac", s);
        s.fill(a2v); ps("A2 algebraic+D14   value-only", s);
        s.fill(a2j); ps("A2 algebraic+D14   value+5-Jac", s);
        s.fill(hf);  ps("H  holonomic       fused v+5-Jac", s);
        s.fill(iv);  ps("I  inverse-ray     value-only", s);
        std::printf("      band-discovery phase:  A1 p50 %.4f  A2 p50 %.4f  ms  (median over slice)\n",
                    med(a1b), med(a2b));
    };
    std::printf("\n================ CLASS A  (forced full solve)  LATENCY [ms] ================\n");
    std::printf("[ALL]\n");               lat([](const Row&){return true;});
    std::printf("[ordinary binary q>%.0e]\n", QP); lat([&](const Row& r){return !r.planetary;});
    std::printf("[planetary low-q]\n");    lat([&](const Row& r){return r.planetary;});
    std::printf("[caustic-ish]\n");        lat([&](const Row& r){return r.caustic;});
    std::printf("[small rho <=0.02]\n");   lat([&](const Row& r){return r.smallrho;});
    std::printf("[wide a>=2]\n");          lat([&](const Row& r){return r.wide;});
    std::printf("[uniform u=0]\n");        lat([](const Row& r){return r.u==0.0;});
    std::printf("[linear LD u>0]\n");      lat([](const Row& r){return r.u>0.0;});

    std::printf("\n================ COUNTERS  (per-epoch mean, value path) ================\n");
    auto cnt = [&](std::function<bool(const Row&)> pr) {
        std::vector<double> rs1,rs2,le1,le2,q1,q2,mv1,mv2,mvfb1,mvfb2,cold1,cold2,b1,b2,ev,d14ok,d14fb;
        for (auto& r : rows) if (pr(r) && r.name!="rand002") {
            rs1.push_back(r.a1.root_solves); rs2.push_back(r.a2.root_solves);
            le1.push_back(r.a1.lens_evals);  le2.push_back(r.a2.lens_evals);
            q1.push_back(r.a1.quartic);       q2.push_back(r.a2.quartic);
            mv1.push_back(r.a1.mv);           mv2.push_back(r.a2.mv);
            mvfb1.push_back(r.a1.mv_fb);      mvfb2.push_back(r.a2.mv_fb);
            cold1.push_back(r.a1.cold);       cold2.push_back(r.a2.cold);
            b1.push_back(r.a1.bands);         b2.push_back(r.a2.bands);
            ev.push_back(r.a2.d14_events);    d14ok.push_back(r.a2.d14_enum); d14fb.push_back(r.a2.d14_fb);
        }
        std::printf("    root_solve_count   A1 %8.1f   A2 %8.1f   (%.1f%%)\n", meanv(rs1), meanv(rs2),
                    100.0*(meanv(rs2)-meanv(rs1))/std::max(1.0,meanv(rs1)));
        std::printf("    lens_eval_count    A1 %8.1f   A2 %8.1f   (%.1f%%)\n", meanv(le1), meanv(le2),
                    100.0*(meanv(le2)-meanv(le1))/std::max(1.0,meanv(le1)));
        std::printf("    real_quartic_solve A1 %8.1f   A2 %8.1f   (%.1f%%)\n", meanv(q1), meanv(q2),
                    100.0*(meanv(q2)-meanv(q1))/std::max(1.0,meanv(q1)));
        std::printf("    mv_transport       A1 %8.1f   A2 %8.1f\n", meanv(mv1), meanv(mv2));
        std::printf("    mv_transport_fb    A1 %8.1f   A2 %8.1f   (cold re-solve)\n", meanv(mvfb1), meanv(mvfb2));
        std::printf("    cold_root_solve    A1 %8.1f   A2 %8.1f\n", meanv(cold1), meanv(cold2));
        std::printf("    radial_band_count  A1 %8.2f   A2 %8.2f\n", meanv(b1), meanv(b2));
        std::printf("    D14: physical_real events %.2f   used %.0f%%   fell back %.0f%%\n",
                    meanv(ev), 100.0*meanv(d14ok), 100.0*meanv(d14fb));
    };
    std::printf("[ALL]\n");             cnt([](const Row&){return true;});
    std::printf("[ordinary binary]\n"); cnt([&](const Row& r){return !r.planetary;});
    std::printf("[caustic-ish]\n");     cnt([&](const Row& r){return r.caustic;});

    std::printf("\n================ ACCURACY (rel. to binary_ray_shooting M0) ================\n");
    auto acc = [&](std::function<bool(const Row&,const Case&)> pr) {
        std::vector<double> d1,d2,dh,di,d12;
        for (size_t i=0;i<rows.size();++i){ const Row& r=rows[i]; const Case& c=cs[i];
            if(!pr(r,c)) continue;
            bool ref = (c.m0_status=="OK" && std::isfinite(c.m0_mu) && c.m0_mu>0);
            if(ref){
                if(r.a1.ok) d1.push_back(std::fabs(r.a1.mu-c.m0_mu)/c.m0_mu);
                if(r.a2.ok) d2.push_back(std::fabs(r.a2.mu-c.m0_mu)/c.m0_mu);
                if(r.h_st==hol::Status::OK) dh.push_back(std::fabs(r.h_mu-c.m0_mu)/c.m0_mu);
                if(r.i_ok) di.push_back(std::fabs(r.i_mu-c.m0_mu)/c.m0_mu);
            }
            if(r.a1.ok && r.a2.ok && r.a1.mu!=0.0)
                d12.push_back(std::fabs(r.a2.mu-r.a1.mu)/std::fabs(r.a1.mu));
        }
        auto pr1=[](const char* t, std::vector<double>& v){
            std::printf("    %-22s n=%3zu  median %.2e  p90 %.2e  max %.2e\n", t, v.size(),
                med(v), pct(v,90), v.empty()?0:*std::max_element(v.begin(),v.end())); };
        pr1("A1 vs M0", d1); pr1("A2(+D14) vs M0", d2); pr1("holonomic vs M0", dh);
        pr1("inverse-ray vs M0", di); pr1("A2 vs A1 (D14 shift)", d12);
    };
    std::printf("[ALL]\n");             acc([](const Row&,const Case&){return true;});
    std::printf("[ordinary binary]\n"); acc([&](const Row& r,const Case&){return !r.planetary;});
    std::printf("[caustic-ish]\n");     acc([&](const Row& r,const Case&){return r.caustic;});
    std::printf("[small rho]\n");       acc([&](const Row& r,const Case&){return r.smallrho;});

    std::printf("\n================ JACOBIAN QUALITY / DELIVERY ================\n");
    {
        std::vector<double> a1f,hf2; int a1g=0,a2g=0,hg=0,n=0,nhang=0;
        for (auto& r : rows) {
            if (r.name=="rand002"){ ++nhang; continue; }
            ++n;
            if (std::isfinite(r.a1_fd)) a1f.push_back(r.a1_fd);
            if (std::isfinite(r.h_fd)) hf2.push_back(r.h_fd);
            if (r.a1.grad_ok) ++a1g;
            if (r.a2.grad_ok) ++a2g;
            if (r.h_st==hol::Status::OK) ++hg;
        }
        std::printf("    analytic vs own Richardson-FD:  A1 median %.2e  p90 %.2e  |  holo median %.2e  p90 %.2e\n",
                    med(a1f), pct(a1f,90), med(hf2), pct(hf2,90));
        std::printf("    Jacobian delivery (reliable/total, excl. hang):  A1 %d/%d   A2 %d/%d   holo %d/%d   (%d hang)\n",
                    a1g,n, a2g,n, hg,n, nhang);
    }

    std::printf("\n================ FAILURE / FAIL-CLOSED ================\n");
    {
        int a1vf=0,a1gf=0,a1sil=0, a2vf=0,a2gf=0,a2sil=0, hnok=0,hsil=0, inok=0,isil=0, hang=0;
        for (size_t i=0;i<rows.size();++i){ const Row& r=rows[i]; const Case& c=cs[i];
            bool ref=(c.m0_status=="OK"&&c.m0_mu>0&&std::isfinite(c.m0_mu));
            if(r.name=="rand002"){ ++hang; continue; }
            if(!r.a1.ok)++a1vf; if(!r.a1.grad_ok)++a1gf;
            if(!r.a2.ok)++a2vf; if(!r.a2.grad_ok)++a2gf;
            if(r.h_st!=hol::Status::OK)++hnok;
            if(!r.i_ok)++inok;
            if(ref){
                if(r.a1.ok && std::fabs(r.a1.mu-c.m0_mu)/c.m0_mu>0.05){++a1sil;
                    std::printf("    [SILENT A1] %-14s u=%.1f mu=%.6g M0=%.6g\n",c.name.c_str(),c.u,r.a1.mu,c.m0_mu);}
                if(r.a2.ok && std::fabs(r.a2.mu-c.m0_mu)/c.m0_mu>0.05){++a2sil;
                    std::printf("    [SILENT A2] %-14s u=%.1f mu=%.6g M0=%.6g\n",c.name.c_str(),c.u,r.a2.mu,c.m0_mu);}
                if(r.h_st==hol::Status::OK && std::fabs(r.h_mu-c.m0_mu)/c.m0_mu>0.05){++hsil;
                    std::printf("    [SILENT H ] %-14s u=%.1f mu=%.6g M0=%.6g\n",c.name.c_str(),c.u,r.h_mu,c.m0_mu);}
                if(r.i_ok && std::fabs(r.i_mu-c.m0_mu)/c.m0_mu>0.05){++isil;
                    std::printf("    [SILENT I ] %-14s u=%.1f mu=%.6g M0=%.6g\n",c.name.c_str(),c.u,r.i_mu,c.m0_mu);}
            }
        }
        int N = (int)rows.size()-1;  // excl hang
        std::printf("    A1 : value-fail %d/%d  grad-unreliable %d/%d  silent %d\n", a1vf,N,a1gf,N,a1sil);
        std::printf("    A2 : value-fail %d/%d  grad-unreliable %d/%d  silent %d\n", a2vf,N,a2gf,N,a2sil);
        std::printf("    H  : status!=OK %d/%d  silent %d\n", hnok,N,hsil);
        std::printf("    I  : value-fail %d/%d  silent %d  (no analytic Jacobian)\n", inok,N,isil);
        std::printf("    hard non-termination (not fail-closed): %d  [%s]\n", hang, "rand002");
    }

    std::printf("\n#CSV name,u,q,planetary,caustic,smallrho,wide,"
                "a1_val,a1_jac,a1_band,a2_val,a2_jac,a2_band,h_ms,i_ms,"
                "a1_mu,a2_mu,h_mu,i_mu,m0_mu,a1_ok,a1_gok,a2_ok,a2_gok,h_st,"
                "a1_rs,a2_rs,a1_quartic,a2_quartic,a1_mvfb,a2_mvfb,a2_d14ev,a2_d14enum,a2_d14fb\n");
    for (size_t i=0;i<rows.size();++i){ const Row& r=rows[i]; const Case& c=cs[i];
        std::printf("#CSV %s,%.1f,%.6g,%d,%d,%d,%d,"
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                    "%.8g,%.8g,%.8g,%.8g,%.8g,%d,%d,%d,%d,%s,"
                    "%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    c.name.c_str(),c.u,c.q,(int)r.planetary,(int)r.caustic,(int)r.smallrho,(int)r.wide,
                    r.a1.val_ms,r.a1.jac_ms,r.a1.band_ms,r.a2.val_ms,r.a2.jac_ms,r.a2.band_ms,r.h_ms,r.i_ms,
                    r.a1.mu,r.a2.mu,r.h_mu,r.i_mu,c.m0_mu,
                    (int)r.a1.ok,(int)r.a1.grad_ok,(int)r.a2.ok,(int)r.a2.grad_ok,hol::to_string(r.h_st),
                    r.a1.root_solves,r.a2.root_solves,r.a1.quartic,r.a2.quartic,
                    r.a1.mv_fb,r.a2.mv_fb,r.a2.d14_events,r.a2.d14_enum,r.a2.d14_fb);
    }
    return 0;
}

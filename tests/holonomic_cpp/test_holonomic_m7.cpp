// ATPT holonomic solver (M7) -- C++ correctness harness.
//
// Reproduces, in C++, the quantities dumped by dump_reference.py from the
// Python holonomic_ref oracle and asserts agreement to the tolerances in
// docs/holonomic/checkpoint_M7.md sec. 3.  No JSON dependency: the fixture
// evidence/holonomic/m7_reference.tsv is a flat whitespace-delimited table.
//
// Build (isolated, shared .so untouched):
//   g++ -std=c++17 -O2 -I src tests/holonomic_cpp/test_holonomic_m7.cpp \
//       -o build-holonomic-m7/test_holonomic_m7 -lquadmath
//
// Covers: boundary_quartic{,_dp}, phi_grad, radial_events, radius_terms,
// epoch_jacobian (value + 5-component Jacobian + dmu/du).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/phi.hpp"
#include "lcbinint/magnification/holonomic/radial_events.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

using lcbinint::holonomic::LensParams;
using lcbinint::holonomic::PrimaryFrame;

namespace {

int g_fail = 0;
int g_checks = 0;

bool close(double got, double want, double rtol, double atol) {
    ++g_checks;
    return std::fabs(got - want) <= atol + rtol * std::fabs(want);
}

void check(const char* what, const std::string& tag, double got, double want,
           double rtol, double atol) {
    if (!close(got, want, rtol, atol)) {
        ++g_fail;
        std::fprintf(stderr,
                     "  FAIL %-10s %-46s got=% .17g want=% .17g |d|=%.3e\n",
                     what, tag.c_str(), got, want, std::fabs(got - want));
    }
}

struct Case {
    std::string name;
    PrimaryFrame pf{};
    double r_max = 0.0;
};

struct Ev {
    std::string name, kind;
    double radius;
    int physically_real;
};

struct Sample {
    std::string name;
    double R, theta, phi, dphi_dtheta;
    double dP[5];
    double bq[5];
    double dbq[5][5];
};

struct RT {
    std::string name, kind;
    double R, f0, fh, df0[5], dfh[5];
    int reliable;
};

struct Epoch {
    std::string name, status;
    double u, mu, grad[5], dmu_du, F0, F_half;
};

// user (xs,ys,rho,q,a) for each fixture case -- all barycentric=false.
struct UCase { const char* name; double xs, ys, rho, q, a; };
const UCase kUCases[] = {
    {"benign", 0.35, 0.22, 0.05, 0.4, 1.15},
    {"close", 0.4, -0.05, 0.09, 0.8, 0.55},
    {"resonant", 0.05, 0.02, 0.05, 0.3, 0.9},
    {"plan15", 1.0 / 5.0, 1.0 / 7.0, 1.0 / 8.0, 0.5, 1.2},
    {"wide-planet", 1.4, 0.10, 0.03, 1e-3, 2.5},
    {"cusp", 0.9, -0.3, 0.05, 0.25, 0.7},
    {"tiny-rho", 0.1, 0.02, 5e-3, 0.3, 1.0},
};
LensParams lp_of(const std::string& name) {
    for (const auto& u : kUCases)
        if (name == u.name)
            return LensParams{u.xs, u.ys, u.rho, u.q, u.a, false};
    std::fprintf(stderr, "no user case %s\n", name.c_str());
    std::exit(3);
}

std::string fixture_path() {
    const char* cands[] = {
        "evidence/holonomic/m7_reference.tsv",
        "../evidence/holonomic/m7_reference.tsv",
        "../../evidence/holonomic/m7_reference.tsv",
    };
    for (const char* c : cands) {
        std::ifstream f(c);
        if (f.good()) return c;
    }
    return cands[0];
}

}  // namespace

int main() {
    const std::string path = fixture_path();
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open fixture: %s\n", path.c_str());
        return 2;
    }
    std::fprintf(stderr, "fixture: %s\n", path.c_str());

    std::vector<Case> cases;
    std::vector<Sample> samples;
    std::vector<Ev> evs;
    std::vector<RT> rts;
    std::vector<Epoch> epochs;

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kind;
        ls >> kind;
        if (kind == "PF") {
            Case c;
            double a, m0, X, Y, rho, rmax;
            ls >> c.name >> a >> m0 >> X >> Y >> rho >> rmax;
            c.pf = PrimaryFrame{a, m0, X, Y, rho};
            c.r_max = rmax;
            cases.push_back(c);
        } else if (kind == "EV") {
            Ev e;
            ls >> e.name >> e.radius >> e.kind >> e.physically_real;
            evs.push_back(e);
        } else if (kind == "SAMPLE") {
            Sample s;
            ls >> s.name >> s.R >> s.theta >> s.phi >> s.dphi_dtheta;
            for (double& v : s.dP) ls >> v;
            for (double& v : s.bq) ls >> v;
            for (auto& row : s.dbq)
                for (double& v : row) ls >> v;
            samples.push_back(s);
        } else if (kind == "RT") {
            RT r;
            ls >> r.name >> r.R >> r.f0 >> r.fh;
            for (double& v : r.df0) ls >> v;
            for (double& v : r.dfh) ls >> v;
            ls >> r.reliable >> r.kind;
            rts.push_back(r);
        } else if (kind == "EPOCH") {
            Epoch e;
            ls >> e.name >> e.u >> e.mu;
            for (double& v : e.grad) ls >> v;
            ls >> e.dmu_du >> e.F0 >> e.F_half >> e.status;
            epochs.push_back(e);
        }
    }

    std::fprintf(stderr,
                 "loaded %zu cases, %zu samples, %zu evs, %zu rts, %zu epochs\n",
                 cases.size(), samples.size(), evs.size(), rts.size(),
                 epochs.size());

    auto pf_of = [&](const std::string& name) -> const PrimaryFrame& {
        for (const auto& c : cases)
            if (c.name == name) return c.pf;
        std::fprintf(stderr, "no PF for case %s\n", name.c_str());
        std::exit(3);
    };

    for (const auto& s : samples) {
        const PrimaryFrame& pf = pf_of(s.name);
        std::string tag = s.name + " R=" + std::to_string(s.R) +
                          " th=" + std::to_string(s.theta);

        auto q = lcbinint::holonomic::boundary_quartic(s.R, pf);
        double bqscale = 0.0;
        for (int j = 0; j < 5; ++j) bqscale = std::max(bqscale, std::fabs(s.bq[j]));
        for (int i = 0; i < 5; ++i)
            check("bq", tag + " p" + std::to_string(i), q.p[i], s.bq[i], 1e-12,
                  1e-12 * bqscale);

        auto dq = lcbinint::holonomic::boundary_quartic_dp(s.R, pf);
        for (int j = 0; j < 5; ++j) {
            double scale = 0.0;
            for (int i = 0; i < 5; ++i)
                scale = std::max(scale, std::fabs(s.dbq[j][i]));
            for (int i = 0; i < 5; ++i)
                check("bq_dp",
                      tag + " d" + std::to_string(j) + "/p" + std::to_string(i),
                      dq.dp[j][i], s.dbq[j][i], 1e-12, 1e-12 * scale);
        }

        auto g = lcbinint::holonomic::phi_grad(s.R, s.theta, pf);
        check("phi", tag, g.phi, s.phi, 1e-11, 1e-13);
        check("dphi_dth", tag, g.dphi_dtheta, s.dphi_dtheta, 1e-11, 1e-13);
        for (int j = 0; j < 5; ++j)
            check("dphi_dP", tag + " j" + std::to_string(j), g.dP[j], s.dP[j],
                  1e-11, 1e-13);
    }

    // ---- r_max + radial_events : radii within 1e-7, kinds identical ----
    for (const auto& c : cases) {
        double rmax = 0.0;
        auto got = lcbinint::holonomic::radial_events(c.pf, &rmax);
        check("r_max", c.name, rmax, c.r_max, 1e-9, 1e-12);

        std::vector<Ev> want;
        for (const auto& e : evs)
            if (e.name == c.name) want.push_back(e);

        if (got.size() != want.size()) {
            ++g_fail;
            ++g_checks;
            std::fprintf(stderr,
                         "  FAIL events    %-46s got %zu events, want %zu\n",
                         c.name.c_str(), got.size(), want.size());
        }
        size_t n = std::min(got.size(), want.size());
        for (size_t i = 0; i < n; ++i) {
            std::string tag = c.name + " ev[" + std::to_string(i) + "]";
            check("ev.radius", tag, got[i].radius, want[i].radius, 0.0, 1e-7);
            ++g_checks;
            if (got[i].kind != want[i].kind ||
                (int)got[i].physically_real != want[i].physically_real) {
                ++g_fail;
                std::fprintf(stderr,
                             "  FAIL ev.kind   %-46s got %s/%d want %s/%d\n",
                             tag.c_str(), got[i].kind.c_str(),
                             (int)got[i].physically_real, want[i].kind.c_str(),
                             want[i].physically_real);
            }
        }
    }

    // ---- radius_terms : f0/fh/df0/dfh to 1e-8 rel, reliable identical ----
    for (const auto& r : rts) {
        const PrimaryFrame& pf = pf_of(r.name);
        auto rt = lcbinint::holonomic::radius_terms(r.R, pf);
        std::string tag = r.name + " R=" + std::to_string(r.R);
        check("rt.f0", tag, rt.f0, r.f0, 1e-8, 1e-11);
        check("rt.fh", tag, rt.fh, r.fh, 1e-8, 1e-11);
        double s0 = 0.0, sh = 0.0;
        for (int j = 0; j < 5; ++j) {
            s0 = std::max(s0, std::fabs(r.df0[j]));
            sh = std::max(sh, std::fabs(r.dfh[j]));
        }
        for (int j = 0; j < 5; ++j) {
            check("rt.df0", tag + " j" + std::to_string(j), rt.df0[j], r.df0[j],
                  1e-7, 1e-9 * s0 + 1e-11);
            check("rt.dfh", tag + " j" + std::to_string(j), rt.dfh[j], r.dfh[j],
                  1e-7, 1e-9 * sh + 1e-11);
        }
        ++g_checks;
        if ((int)rt.reliable != r.reliable) {
            ++g_fail;
            std::fprintf(stderr, "  FAIL rt.reliable %-42s got %d want %d\n",
                         tag.c_str(), (int)rt.reliable, r.reliable);
        }
    }

    // ---- epoch_jacobian : mu 1e-8, grad 1e-7, dmu_du/F 1e-8, status equal ----
    for (const auto& e : epochs) {
        LensParams p = lp_of(e.name);
        auto ej = lcbinint::holonomic::epoch_jacobian(p, e.u, 64);
        std::string tag = e.name + " u=" + std::to_string(e.u);
        check("ej.mu", tag, ej.mu, e.mu, 1e-8, 1e-10);
        check("ej.F0", tag, ej.F0, e.F0, 1e-8, 1e-12);
        check("ej.F_half", tag, ej.F_half, e.F_half, 1e-8, 1e-12);
        check("ej.dmu_du", tag, ej.dmu_du, e.dmu_du, 1e-7, 1e-10);
        double gs = 0.0;
        for (int j = 0; j < 5; ++j) gs = std::max(gs, std::fabs(e.grad[j]));
        for (int j = 0; j < 5; ++j)
            check("ej.grad", tag + " j" + std::to_string(j), ej.grad_mu[j],
                  e.grad[j], 1e-7, 1e-8 * gs);
        ++g_checks;
        if (e.status != lcbinint::holonomic::to_string(ej.status)) {
            ++g_fail;
            std::fprintf(stderr, "  FAIL ej.status %-44s got %s want %s\n",
                         tag.c_str(), lcbinint::holonomic::to_string(ej.status),
                         e.status.c_str());
        }
    }

    std::fprintf(stderr, "\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

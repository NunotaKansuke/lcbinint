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
// Currently covers: boundary_quartic, boundary_quartic_dp, phi_grad.
// Later milestones add radial_events / radius_terms / epoch_jacobian checks.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/phi.hpp"
#include "lcbinint/magnification/holonomic/radial_events.hpp"

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
        }
    }

    std::fprintf(stderr, "loaded %zu cases, %zu samples\n", cases.size(),
                 samples.size());

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

    std::fprintf(stderr, "\n%d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

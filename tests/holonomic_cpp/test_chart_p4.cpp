// Parity harness for the parameter-independent chart_p4 cubic factorization.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/radial_events.hpp"

using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;

namespace {

std::vector<double> generic_roots(const PrimaryFrame& pf) {
    const PolyFamilyR fam = p_coeffs_in_R((__float128)pf.a, (__float128)pf.m0,
                                          (__float128)pf.X, (__float128)pf.Y,
                                          (__float128)pf.rho);
    int deg = fam.p[4].deg;
    while (deg > 0 && fabsq(fam.p[4].c[deg]) == 0) --deg;
    if (deg <= 0) return {};
    std::vector<double> desc(deg + 1);
    for (int i = 0; i <= deg; ++i) desc[i] = (double)fam.p[4].c[deg - i];
    return positive_real_roots(aberth<double>(desc.data(), deg, 200), 1e-8, 1e-9);
}

double p4_value(double R, const PrimaryFrame& pf) {
    return boundary_quartic(R, pf).p[4];
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/bench_cases.tsv";
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    int n = 0, fails = 0, applicable = 0;
    double worst_root = 0.0, worst_res = 0.0;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        double xs, ys, rho, q, a, u, tj;
        int bary;
        std::string name;
        if (!(ls >> xs >> ys >> rho >> q >> a >> bary >> u >> tj >> name)) continue;
        ++n;
        const LensParams lp{xs, ys, rho, q, a, (bool)bary};
        const PrimaryFrame pf = PrimaryFrame::from(lp);
        const auto fact = chart_p4_factor_roots(pf);
        const auto gen = generic_roots(pf);
        const double scale = 1.0 + std::fabs(pf.a) + std::fabs(pf.m0) +
                             std::fabs(pf.X) + std::fabs(pf.Y) + pf.rho;
        if (pf.rho >= std::fabs(pf.Y)) ++applicable;
        if (fact.size() != gen.size()) {
            ++fails;
            std::fprintf(stderr, "  COUNT FAIL %-16s factor=%zu generic=%zu\n",
                         name.c_str(), fact.size(), gen.size());
            continue;
        }
        for (size_t i = 0; i < fact.size(); ++i) {
            const double rd = std::fabs(fact[i] - gen[i]) /
                              (1.0 + std::fabs(gen[i]));
            worst_root = std::max(worst_root, rd);
            worst_res = std::max(worst_res, std::fabs(p4_value(fact[i], pf)) / scale);
            if (rd > 2e-7 || std::fabs(p4_value(fact[i], pf)) > 2e-6 * scale) {
                ++fails;
                std::fprintf(stderr, "  ROOT FAIL %-16s i=%zu factor=%.16g generic=%.16g\n",
                             name.c_str(), i, fact[i], gen[i]);
            }
        }
    }
    std::printf("cases                 %d\n", n);
    std::printf("|Y|<=rho cases        %d\n", applicable);
    std::printf("worst root rel diff    %.3e\n", worst_root);
    std::printf("worst p4 residual      %.3e\n", worst_res);
    std::printf("\n%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

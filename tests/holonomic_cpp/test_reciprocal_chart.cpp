// Exact algebra/parity checks for the reciprocal boundary chart u=-1/t.
// The production V2 router remains on the t chart; this test proves the
// representation primitive before any condition-driven switch is considered.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/poly_roots.hpp"

using namespace lcbinint::holonomic;

namespace {

double poly(const QuarticCoeffs& p, double x) {
    double y = 0.0;
    for (int i = 4; i >= 0; --i) y = y * x + p.p[i];
    return y;
}

double scale(const QuarticCoeffs& p, double x) {
    double s = 0.0, xp = 1.0;
    for (int i = 0; i <= 4; ++i) {
        s += std::fabs(p.p[i]) * std::fabs(xp);
        xp *= x;
    }
    return 1.0 + s;
}

std::vector<double> roots(const QuarticCoeffs& p) {
    double c[5] = {p.p[4], p.p[3], p.p[2], p.p[1], p.p[0]};
    return real_roots(aberth<double>(c, 4, 160), 1e-8, 1e-10);
}

double set_diff(std::vector<double> a, std::vector<double> b) {
    if (a.size() != b.size()) return 1e9;
    double out = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        out = std::max(out, std::fabs(a[i] - b[i]) /
                                (1.0 + std::fabs(b[i])));
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "evidence/holonomic/gm_coverage_cases.tsv";
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }
    int cases = 0, failures = 0, root_checks = 0;
    double worst_identity = 0.0, worst_derivative = 0.0, worst_root = 0.0;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        double xs, ys, rho, q, a, u, tj;
        int bary;
        std::string name;
        if (!(ss >> xs >> ys >> rho >> q >> a >> bary >> u >> tj >> name)) continue;
        ++cases;
        const PrimaryFrame pf = PrimaryFrame::from(
            LensParams{xs, ys, rho, q, a, bary != 0});
        // Include both regular and chart-near radii.  The latter is the
        // region where t can become large while u remains bounded.
        const double radii[] = {std::max(1e-7, 0.2 * rho),
                                std::max(1e-7, a),
                                std::max(1e-7, 0.5 * (a + rho)),
                                std::max(1e-7, 1.5 * (a + rho))};
        for (double R : radii) {
            const QuarticCoeffs p = boundary_quartic(R, pf);
            const QuarticCoeffs r = boundary_quartic_reciprocal(p);
            const QuarticCoeffs back = boundary_quartic_reciprocal(r);
            for (int i = 0; i < 5; ++i) {
                const double e = std::fabs(back.p[i] - p.p[i]) /
                                 (1.0 + std::fabs(p.p[i]));
                worst_identity = std::max(worst_identity, e);
                if (e != 0.0) ++failures;
            }
            const QuarticParamJac dp = boundary_quartic_dp(R, pf);
            const QuarticParamJac dr = boundary_quartic_reciprocal_dp(dp);
            for (double uu : {-3.0, -0.75, 0.2, 1.5, 4.0}) {
                const double lhs = std::pow(uu, 4) * poly(p, -1.0 / uu);
                const double rhs = poly(r, uu);
                const double e = std::fabs(lhs - rhs) /
                                 (scale(r, uu) + std::fabs(lhs));
                worst_identity = std::max(worst_identity, e);
                if (e > 2e-13) ++failures;
                for (int j = 0; j < 5; ++j) {
                    QuarticCoeffs d{};
                    d.p = dp.dp[j];
                    QuarticCoeffs rd{};
                    rd.p = dr.dp[j];
                    const double dl = std::pow(uu, 4) * poly(d, -1.0 / uu);
                    const double rr = poly(rd, uu);
                    const double de = std::fabs(dl - rr) /
                                      (scale(rd, uu) + std::fabs(dl));
                    worst_derivative = std::max(worst_derivative, de);
                    if (de > 2e-13) ++failures;
                }
            }
            const auto t_roots = roots(p);
            const auto u_roots = roots(r);
            std::vector<double> mapped;
            for (double t : t_roots)
                if (std::fabs(t) > 1e-8) mapped.push_back(-1.0 / t);
            std::sort(mapped.begin(), mapped.end());
            const double rd = set_diff(mapped, u_roots);
            if (!mapped.empty() || !u_roots.empty()) {
                ++root_checks;
                worst_root = std::max(worst_root, rd);
                if (rd > 2e-6) ++failures;
            }
        }
    }
    std::printf("cases                  %d\n", cases);
    std::printf("root-set checks        %d\n", root_checks);
    std::printf("worst coefficient id   %.3e\n", worst_identity);
    std::printf("worst derivative id    %.3e\n", worst_derivative);
    std::printf("worst mapped root diff  %.3e\n", worst_root);
    std::printf("\n%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

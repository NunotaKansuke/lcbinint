// Research benchmark for the exact 2x2 D14 matrix polynomial and its
// first-companion pencil.  It is deliberately separate from radial_events:
// no matrix/QZ result can reach the production router here.

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

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/d14_lifted.hpp"
#include "lcbinint/magnification/holonomic/d14_matrix.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"
#include "lcbinint/magnification/holonomic/radial_events.hpp"

using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
using qf = __float128;
using clk = std::chrono::steady_clock;

namespace {

struct Case {
    double xs, ys, rho, q, a;
    int bary;
    double u, t_jac;
    std::string name;
};

double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double x = p * 0.01 * (v.size() - 1);
    const size_t i = static_cast<size_t>(x);
    if (i + 1 >= v.size()) return v.back();
    return v[i] + (v[i + 1] - v[i]) * (x - i);
}

std::vector<Case> read_cases(const char* path) {
    std::ifstream in(path);
    std::vector<Case> out;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        Case c;
        if (ss >> c.xs >> c.ys >> c.rho >> c.q >> c.a >> c.bary >> c.u >>
            c.t_jac >> c.name)
            out.push_back(c);
    }
    return out;
}

std::vector<double> positive_real(const std::vector<Cplx<qf>>& z) {
    return positive_real_roots(z, 1e-8, 1e-9);
}

std::vector<double> complex_re(const std::vector<Cplx<qf>>& z) {
    std::vector<double> x;
    for (const auto& r : z) {
        const double re = static_cast<double>(r.re);
        const double im = static_cast<double>(r.im);
        if (re > 0.0 && std::fabs(im) >= 1e-8 * (1.0 + std::fabs(re)))
            x.push_back(re);
    }
    std::sort(x.begin(), x.end());
    std::vector<double> out;
    for (double v : x)
        if (out.empty() || v - out.back() > 1e-9) out.push_back(v);
    return out;
}

struct MatrixPencil {
    std::array<double, 16 * 16> a{};
    std::array<double, 16 * 16> b{};
};

MatrixPencil first_companion_pencil(const D14MatrixQf& m) {
    constexpr int block = 2;
    constexpr int grade = 8;
    constexpr int n = block * grade;
    MatrixPencil out;
    auto at = [n](std::array<double, 16 * 16>& x, int r, int c) -> double& {
        return x[static_cast<size_t>(r) * n + c];
    };
    // The pencil is A - lambda B.  Its finite determinant is the matrix
    // polynomial determinant up to a nonzero constant.  The grade-8 leading
    // block has rank one, so a regular generic instance has 14 finite and 2
    // infinite generalized eigenvalues.
    for (int j = 0; j < grade; ++j) {
        const int k = grade - 1 - j;
        for (int r = 0; r < block; ++r)
            for (int c = 0; c < block; ++c) {
                at(out.a, r, j * block + c) = -static_cast<double>(m[k][r][c]);
                if (j == 0)
                    at(out.b, r, j * block + c) =
                        static_cast<double>(m[grade][r][c]);
            }
    }
    for (int j = 1; j < grade; ++j) {
        for (int r = 0; r < block; ++r) {
            at(out.a, j * block + r, (j - 1) * block + r) = 1.0;
            at(out.b, j * block + r, j * block + r) = 1.0;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "evidence/holonomic/gm_coverage_cases.tsv";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const auto cases = read_cases(path);
    if (cases.empty()) {
        std::fprintf(stderr, "cannot read cases from %s\n", path);
        return 2;
    }

    std::vector<double> build_ms, pencil_ms, scalar_seed_ms;
    int identity_fail = 0, total = 0, degree14_bad = 0;
    double worst_identity = 0.0;
    double checksum = 0.0;
    for (const auto& c : cases) {
        const LensParams p{c.xs, c.ys, c.rho, c.q, c.a, c.bary != 0};
        const PrimaryFrame pf = PrimaryFrame::from(p);
        const auto s = d14_struct_build(static_cast<qf>(pf.a), static_cast<qf>(pf.m0),
                                        static_cast<qf>(pf.X), static_cast<qf>(pf.Y),
                                        static_cast<qf>(pf.rho));
        const auto block_d14 = d14_expanded_from_struct(s);
        if (block_d14.size() != 15) continue;
        ++total;

        const auto t0 = clk::now();
        const auto matrix = d14_matrix_build(s);
        const auto t1 = clk::now();
        build_ms.push_back(ms(t0, t1));
        const auto tp = clk::now();
        const auto pencil = first_companion_pencil(matrix);
        const auto tq = clk::now();
        pencil_ms.push_back(ms(tp, tq));
        const auto det = d14_matrix_determinant(matrix);
        qf scale = 0;
        for (qf x : block_d14) scale = std::max(scale, fabsq(x));
        double identity = 0.0;
        for (int k = 0; k < 15; ++k)
            identity = std::max(identity, static_cast<double>(
                fabsq(det[k] - qf(3) * block_d14[k] / qf(4096)) /
                (qf(1) + scale)));
        worst_identity = std::max(worst_identity, identity);
        if (identity > 1e-28) ++identity_fail;
        if (det[14] == 0.0) ++degree14_bad;

        std::array<double, 15> md{}, sd{};
        for (int i = 0; i < 15; ++i) {
            md[i] = static_cast<double>(det[14 - i]);
            sd[i] = static_cast<double>(block_d14[14 - i]);
        }
        auto balance = [](std::array<double, 15> x) {
            double ratio = std::fabs(x[14]) / std::max(1e-300, std::fabs(x[0]));
            double scale = std::pow(ratio, 1.0 / 14.0);
            if (!(scale > 0.0) || !std::isfinite(scale)) scale = 1.0;
            double sp = 1.0;
            for (int i = 14; i >= 0; --i) {
                x[i] *= sp;
                sp *= scale;
            }
            return x;
        };
        const auto mb = balance(md), sb = balance(sd);
        volatile double pencil_checksum = pencil.a[0] + pencil.b[0];
        checksum += pencil_checksum;
        double st = 1e300;
        for (int r = 0; r < reps; ++r) {
            auto a = clk::now();
            auto roots = aberth<double>(mb.data(), 14, 200);
            auto b = clk::now();
            st = std::min(st, ms(a, b));
            checksum += roots[0].re;
        }
        scalar_seed_ms.push_back(st);
    }
    auto report = [](const char* name, const std::vector<double>& x) {
        std::fprintf(stderr, "  %-14s p50 %.6f p90 %.6f p99 %.6f max %.6f ms\n",
                     name, pct(x, 50), pct(x, 90), pct(x, 99), pct(x, 100));
    };
    std::fprintf(stderr, "D14 matrix/pencil research cases=%d reps=%d\n", total, reps);
    report("matrix build", build_ms);
    report("pencil build", pencil_ms);
    report("scalar seed", scalar_seed_ms);
    std::fprintf(stderr, "identity worst %.3e failures %d/%d\n",
                 worst_identity, identity_fail, total);
    std::fprintf(stderr, "degree-14 determinant failures %d/%d\n",
                 degree14_bad, total);
    std::fprintf(stderr,
                 "QZ solve: not run; this isolated image only measures exact identity, "
                 "pencil assembly, and scalar seed cost\n");
    std::fprintf(stderr, "checksum %.17g\n", checksum);
    return identity_fail == 0 ? 0 : 1;
}

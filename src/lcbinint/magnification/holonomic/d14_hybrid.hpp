#pragma once

// Research-only root-wise Horner/structured D14 hybrid.
//
// The Aberth interaction is retained for every root.  A balanced double
// Horner correction is used only when a forward-error estimate covers the
// coefficient conversion, Horner operations, reciprocal interaction sum, and
// B = D' - D S formation.  Otherwise that root takes the DD structural
// evaluator.  The returned roots still require the independent qf/global
// certificate in radial_events.hpp.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/d14_structure.hpp"

namespace lcbinint::holonomic {
namespace re_detail {

using qf = __float128;

struct D14HybridResult {
    bool converged = false;
    std::vector<Cplx<qf>> roots;
    int iterations = 0;
    int cheap_calls = 0;
    int structural_calls = 0;
    int unsafe_calls = 0;
};

struct D14HybridCorrection {
    bool safe = false;
    Cplx<double> step{};
    double error = 0.0;
};

inline double hybrid_cabs(Cplx<double> z) {
    return std::hypot(z.re, z.im);
}

inline double hybrid_gamma(int n) {
    const double eps = std::numeric_limits<double>::epsilon();
    const double x = (8.0 * n) * eps;
    return x < 0.5 ? x / (1.0 - x) : 1.0;
}

inline D14HybridCorrection d14_hybrid_correction(
    const std::array<double, 15>& coeffs,
    const std::array<double, 15>& coeff_error,
    const Cplx<double>& w,
    const std::vector<Cplx<double>>& all_w,
    int index,
    double scale) {
    constexpr int deg = 14;
    const double eps = std::numeric_limits<double>::epsilon();
    Cplx<double> p = poly_eval_c(coeffs.data(), deg, w);
    Cplx<double> dp = polyder_eval_c(coeffs.data(), deg, w);
    Cplx<double> sum(0.0, 0.0);
    double sum_error = 0.0;
    for (int j = 0; j < deg; ++j) {
        if (j == index) continue;
        const Cplx<double> d = w - all_w[j];
        const double dn = hybrid_cabs(d);
        if (!(dn > 0.0) || !std::isfinite(dn))
            return {};
        sum = sum + Cplx<double>(1.0, 0.0) / d;
        // The reciprocal and two additions are bounded by this absolute
        // first-order term.  A zero/near-zero separation is therefore never
        // sent through the cheap path.
        sum_error += 32.0 * eps / dn;
    }
    const Cplx<double> ps = p * sum;
    const Cplx<double> b = dp - ps;
    const double bnorm = hybrid_cabs(b);
    const double pnorm = hybrid_cabs(p);
    const double dpnorm = hybrid_cabs(dp);

    // Horner forward error: coefficient conversion error plus a conservative
    // complex multiply/add gamma bound at each of the fixed 14 levels.
    double power = 1.0;
    double value_terms = 0.0;
    double deriv_terms = 0.0;
    for (int i = deg; i >= 0; --i) {
        value_terms += std::fabs(coeffs[i]) * power;
        if (i < deg) deriv_terms +=
            std::fabs(coeffs[i]) * static_cast<double>(deg - i) * power;
        power *= hybrid_cabs(w);
    }
    double coeff_value_error = 0.0;
    double coeff_deriv_error = 0.0;
    power = 1.0;
    for (int i = deg; i >= 0; --i) {
        coeff_value_error += coeff_error[i] * power;
        if (i < deg) coeff_deriv_error +=
            coeff_error[i] * static_cast<double>(deg - i) * power;
        power *= hybrid_cabs(w);
    }
    const double ep = coeff_value_error + hybrid_gamma(deg) * value_terms;
    const double edp = coeff_deriv_error + hybrid_gamma(deg) * deriv_terms;
    const double eb = edp + ep * hybrid_cabs(sum) + pnorm * sum_error +
                      8.0 * eps * (dpnorm + hybrid_cabs(ps));
    if (!(bnorm > eb) || !std::isfinite(eb)) return {};
    const Cplx<double> step = p / b;
    const double step_error = (ep + hybrid_cabs(step) * eb) / (bnorm - eb);
    // The target is the incumbent DD-to-qf residual gate in balanced root
    // coordinates.  The forward bound, rather than an epsilon multiplier,
    // decides whether a root is eligible for Horner.
    const double required = 0.1e-13 * std::max(1.0, scale + hybrid_cabs(w));
    if (!std::isfinite(step_error) || step_error > required) return {};
    return {true, step, step_error};
}

inline D14HybridResult d14_hybrid_solve(
    const D14StructQf& sqf, const std::vector<qf>& desc,
    const std::vector<Cplx<double>>& balanced_seed, double sscale) {
    constexpr int deg = 14;
    D14HybridResult out;
    if (desc.size() != deg + 1 || balanced_seed.size() != deg ||
        !(sscale > 0.0) || !std::isfinite(sscale)) return out;

    std::array<double, 15> coeffs{}, coeff_error{};
    qf sp = 1;
    for (int i = deg; i >= 0; --i) {
        const qf exact = desc[i] * sp;
        coeffs[i] = static_cast<double>(exact);
        coeff_error[i] = static_cast<double>(fabsq(exact - (qf)coeffs[i])) +
                         0.5 * std::fabs(std::nextafter(coeffs[i],
                             coeffs[i] >= 0.0 ? INFINITY : -INFINITY) - coeffs[i]);
        sp *= (qf)sscale;
    }
    const D14StructC<DD> sdd = d14_struct_cast<DD>(sqf);
    std::vector<Cplx<DD>> z(deg);
    for (int i = 0; i < deg; ++i)
        z[i] = Cplx<DD>(DD(balanced_seed[i].re * sscale),
                        DD(balanced_seed[i].im * sscale));

    for (int it = 0; it < 30; ++it) {
        out.iterations = it + 1;
        double max_step = 0.0;
        for (int i = 0; i < deg; ++i) {
            std::vector<Cplx<double>> w(deg);
            for (int j = 0; j < deg; ++j)
                w[j] = Cplx<double>(static_cast<double>(z[j].re) / sscale,
                                    static_cast<double>(z[j].im) / sscale);
            const D14HybridCorrection cheap = d14_hybrid_correction(
                coeffs, coeff_error, w[i], w, i,
                std::max(1.0, std::fabs(sscale)));
            Cplx<DD> step;
            if (cheap.safe && std::isfinite(cheap.step.re) &&
                std::isfinite(cheap.step.im)) {
                step = Cplx<DD>(DD(cheap.step.re * sscale),
                                DD(cheap.step.im * sscale));
                ++out.cheap_calls;
            } else {
                ++out.unsafe_calls;
                Cplx<DD> p, dp, sum(DD(0.0), DD(0.0));
                d14_struct_eval(sdd, z[i], p, dp);
                for (int j = 0; j < deg; ++j) {
                    if (j == i) continue;
                    sum = sum + Cplx<DD>(DD(1.0), DD(0.0)) /
                                  (z[i] - z[j]);
                }
                step = p / (dp - p * sum);
                ++out.structural_calls;
            }
            if (!std::isfinite(static_cast<double>(step.re)) ||
                !std::isfinite(static_cast<double>(step.im))) return out;
            z[i] = z[i] - step;
            max_step = std::max(max_step, static_cast<double>(cabs(step)));
        }
        if (max_step < 1e-25) {
            out.converged = true;
            break;
        }
    }
    out.roots.reserve(deg);
    for (const auto& r : z)
        out.roots.emplace_back(qf_from_dd(r.re), qf_from_dd(r.im));
    return out;
}

}  // namespace re_detail
}  // namespace lcbinint::holonomic

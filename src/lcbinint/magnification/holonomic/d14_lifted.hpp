#pragma once

// Research-only D14 polish candidate.
//
// The scalar eliminant is retained as the authority.  This file only tests
// whether its exact cubic-discriminant lift can replace the expensive
// Aberth polish locally:
//
//   f(v,lambda) = v lambda^3 + C(v) lambda^2 + G(v) lambda - 4 Z(v)
//
// with f = f_lambda = 0 solved by a 2x2 Newton correction.  A candidate is
// accepted only after the scalar D14 residual, conjugacy, Vieta sums, and
// coefficient reconstruction all pass.  Local convergence alone is never a
// success condition.

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/d14_structure.hpp"

namespace lcbinint::holonomic {
namespace re_detail {

using qf = __float128;

template <class T>
inline Cplx<T> lifted_real(T x) { return Cplx<T>(x, T(0)); }

template <class T>
inline Cplx<T> lifted_scale(const Cplx<T>& x, T a) {
    return Cplx<T>(x.re * a, x.im * a);
}

template <class T>
inline void lifted_cgz(const D14StructC<T>& s, const Cplx<T>& v,
                       Cplx<T>& c, Cplx<T>& cp, Cplx<T>& g,
                       Cplx<T>& gp, Cplx<T>& z, Cplx<T>& zp) {
    horner_vd(s.c3, 3, v, c, cp);
    horner_vd(s.g4, 4, v, g, gp);
    horner_vd(s.z3, 3, v, z, zp);
}

template <class T>
inline void lifted_eval(const D14StructC<T>& s, const Cplx<T>& v,
                        const Cplx<T>& lambda, Cplx<T>& f,
                        Cplx<T>& fl, Cplx<T>& fv, Cplx<T>& flv,
                        Cplx<T>& fll) {
    Cplx<T> c, cp, g, gp, z, zp;
    lifted_cgz(s, v, c, cp, g, gp, z, zp);
    const Cplx<T> l2 = lambda * lambda;
    const Cplx<T> l3 = l2 * lambda;
    f = v * l3 + c * l2 + g * lambda - lifted_real(T(4)) * z;
    fl = lifted_real(T(3)) * (v * l2) + lifted_real(T(2)) * (c * lambda) + g;
    fv = l3 + cp * l2 + gp * lambda - lifted_real(T(4)) * zp;
    flv = lifted_real(T(3)) * l2 + lifted_real(T(2)) * cp * lambda + gp;
    fll = lifted_real(T(6)) * (v * lambda) + lifted_real(T(2)) * c;
}

inline std::complex<double> lifted_lambda_seed(const D14StructQf& s,
                                               std::complex<double> v) {
    std::array<double, 4> c3{}, g4{};
    std::array<double, 5> g4_full{};
    for (int i = 0; i < 4; ++i) c3[i] = (double)s.c3[i];
    for (int i = 0; i < 5; ++i) g4_full[i] = (double)s.g4[i];
    auto hp = [](const auto& c, int n, std::complex<double> x) {
        std::complex<double> y(c[n], 0.0);
        for (int i = n - 1; i >= 0; --i) y = y * x + std::complex<double>(c[i], 0.0);
        return y;
    };
    const std::complex<double> C = hp(c3, 3, v);
    const std::complex<double> G = hp(g4_full, 4, v);
    const std::complex<double> aa = 3.0 * v;
    const std::complex<double> bb = 2.0 * C;
    const std::complex<double> cc = G;
    if (std::abs(aa) == 0.0) return std::complex<double>(0.0, 0.0);
    const std::complex<double> disc = bb * bb - 4.0 * aa * cc;
    const std::complex<double> sd = std::sqrt(disc);
    const std::complex<double> l0 = (-bb + sd) / (2.0 * aa);
    const std::complex<double> l1 = (-bb - sd) / (2.0 * aa);
    auto fres = [&](std::complex<double> l) {
        std::array<double, 4> z3{};
        for (int i = 0; i < 4; ++i) z3[i] = (double)s.z3[i];
        const std::complex<double> Z = hp(z3, 3, v);
        return std::abs(v * l * l * l + C * l * l + G * l - 4.0 * Z);
    };
    return fres(l0) <= fres(l1) ? l0 : l1;
}

struct D14LiftedResult {
    bool ok = false;
    std::vector<Cplx<__float128>> roots;
    std::vector<Cplx<__float128>> lambdas;
    int max_iterations = 0;
    int failed_root = -1;
    int failure_code = 0;  // 1: seed, 2: Newton, 3: non-finite
    double worst_lifted_residual = 0.0;
};

// Global certificate for a scalar root set.  The optional structural
// evaluator is used in the production candidate even when the candidate was
// polished with expanded Horner coefficients; this keeps the validation path
// independent of the cheap evaluator.
inline bool d14_scalar_certificate(
    const D14StructQf* structural, const std::vector<qf>& desc,
    const std::vector<Cplx<qf>>& roots, qf* worst_residual = nullptr,
    int* failure_reason = nullptr) {
    if (failure_reason) *failure_reason = 0;
    auto fail = [&](int code) {
        if (failure_reason) *failure_reason = code;
        return false;
    };
    const int deg = static_cast<int>(desc.size()) - 1;
    if (deg < 1 || roots.size() != static_cast<size_t>(deg) || desc[0] == 0)
        return fail(1);
    qf scale = 0;
    for (qf c : desc) scale = std::max(scale, fabsq(c));
    if (!(scale > 0)) return fail(1);
    qf worst = 0;
    for (const auto& r : roots) {
        if (!finiteq(r.re) || !finiteq(r.im)) return fail(2);
        Cplx<qf> value, deriv;
        if (structural)
            d14_struct_eval(*structural, r, value, deriv);
        else
            value = poly_eval_c(desc.data(), deg, r);
        worst = std::max(worst, cabs(value) / (scale + (qf)1e-300));
    }
    if (worst_residual) *worst_residual = worst;
    if (!(worst <= (qf)1e-12)) return fail(3);

    for (size_t i = 0; i < roots.size(); ++i) {
        if (fabsq(roots[i].im) <= (qf)1e-10 * (1 + fabsq(roots[i].re))) continue;
        bool found = false;
        for (size_t j = 0; j < roots.size(); ++j) {
            if (i == j) continue;
            qf e = fabsq(roots[i].re - roots[j].re) +
                   fabsq(roots[i].im + roots[j].im);
            if (e <= (qf)1e-7 * (1 + fabsq(roots[i].re) + fabsq(roots[i].im))) {
                found = true;
                break;
            }
        }
        if (!found) return fail(4);
    }

    for (int power = 1; power <= 3; ++power) {
        Cplx<qf> sum(0, 0);
        for (const auto& r : roots) {
            Cplx<qf> x(1, 0);
            for (int k = 0; k < power; ++k) x = x * r;
            sum = sum + x;
        }
        const qf c1 = desc[1] / desc[0];
        const qf c2 = desc[2] / desc[0];
        const qf c3 = desc[3] / desc[0];
        const qf want_real = power == 1 ? -c1
                               : power == 2 ? c1 * c1 - 2 * c2
                                            : -c1 * c1 * c1 + 3 * c1 * c2 - 3 * c3;
        if (!(cabs(sum - Cplx<qf>(want_real, 0)) <=
              (qf)2e-8 * (1 + fabsq(want_real)))) return fail(5);
    }

    std::array<Cplx<qf>, 15> rec{};
    rec[0] = Cplx<qf>(1, 0);
    int n = 0;
    for (const auto& root : roots) {
        std::array<Cplx<qf>, 15> next{};
        next[0] = rec[0];
        for (int k = 1; k <= n + 1; ++k)
            next[k] = (k <= n ? rec[k] : Cplx<qf>(0, 0)) - root * rec[k - 1];
        rec = next;
        ++n;
    }
    qf rec_err = 0;
    for (int k = 0; k <= deg; ++k) {
        const qf want = desc[k] / desc[0];
        rec_err = std::max(rec_err,
                           cabs(rec[k] - Cplx<qf>(want, 0)) /
                               (1 + fabsq(want)));
    }
    if (rec_err > (qf)2e-7) return fail(6);
    return true;
}

inline bool d14_lifted_finite(const Cplx<DD>& z) {
    return std::isfinite((double)z.re) && std::isfinite((double)z.im);
}

inline D14LiftedResult d14_lifted_solve(
    const D14StructQf& sqf, const std::vector<Cplx<double>>& balanced_seed,
    double scale) {
    D14LiftedResult out;
    if (balanced_seed.size() != 14 || !(scale > 0.0) || !std::isfinite(scale)) {
        out.failure_code = 1;
        return out;
    }
    const D14StructC<DD> s = d14_struct_cast<DD>(sqf);
    out.roots.reserve(14);
    out.lambdas.reserve(14);
    for (const auto& seed : balanced_seed) {
        const Cplx<DD> v0(DD(seed.re * scale), DD(seed.im * scale));
        Cplx<DD> v = v0;
        const std::complex<double> ls = lifted_lambda_seed(
            sqf, std::complex<double>(seed.re * scale, seed.im * scale));
        Cplx<DD> lambda(DD(ls.real()), DD(ls.imag()));
        bool converged = false;
        int used = 0;
        for (int it = 0; it < 24; ++it) {
            used = it + 1;
            Cplx<DD> f, fl, fv, flv, fll;
            lifted_eval(s, v, lambda, f, fl, fv, flv, fll);
            const double fn = (double)cabs(f) + (double)cabs(fl);
            const double sc = 1.0 + (double)cabs(v) + (double)cabs(lambda);
            // DD is only used to locate the lifted root.  The final qf
            // certificate below owns the production accuracy requirement;
            // requiring a DD residual below its useful scale would turn a
            // viable candidate into an unconditional fallback.
            if (fn <= 1e-18 * sc) {
                converged = true;
                break;
            }
            const Cplx<DD> det = fv * fll - fl * flv;
            const double dn = (double)cabs(det);
            const double ds = 1.0 + (double)cabs(fv) + (double)cabs(fll) +
                              (double)cabs(fl) + (double)cabs(flv);
            if (!(dn > 1e-30 * ds) || !d14_lifted_finite(det)) break;
            const Cplx<DD> dv = (fl * fl - f * fll) / det;
            const Cplx<DD> dl = (f * flv - fv * fl) / det;
            if (!d14_lifted_finite(dv) || !d14_lifted_finite(dl)) break;
            v = v + dv;
            lambda = lambda + dl;
            if ((double)cabs(dv) + (double)cabs(dl) <= 1e-18 * sc) {
                converged = true;
                break;
            }
        }
        out.max_iterations = std::max(out.max_iterations, used);
        if (!converged || !d14_lifted_finite(v) || !d14_lifted_finite(lambda)) {
            out.failed_root = static_cast<int>(out.roots.size());
            out.failure_code = converged ? 3 : 2;
            out.ok = false;
            return out;
        }
        Cplx<DD> f, fl, fv, flv, fll;
        lifted_eval(s, v, lambda, f, fl, fv, flv, fll);
        out.worst_lifted_residual = std::max(
            out.worst_lifted_residual,
            (double)cabs(f) + (double)cabs(fl));
        out.roots.emplace_back(qf_from_dd(v.re), qf_from_dd(v.im));
        out.lambdas.emplace_back(qf_from_dd(lambda.re), qf_from_dd(lambda.im));
    }
    out.ok = out.roots.size() == 14;
    return out;
}

inline bool d14_lifted_certificate(const D14StructQf& sqf,
                                   const std::vector<qf>& desc,
                                   const D14LiftedResult& candidate,
                                   qf* worst_scalar_residual = nullptr,
                                   int* failure_reason = nullptr,
                                   qf* reconstruction_error = nullptr) {
    if (failure_reason) *failure_reason = 0;
    if (reconstruction_error) *reconstruction_error = 0;
    auto fail = [&](int code) {
        if (failure_reason) *failure_reason = code;
        return false;
    };
    if (!candidate.ok || candidate.roots.size() != 14 || desc.size() != 15)
        return fail(1);
    qf scale = 0;
    for (qf c : desc) scale = std::max(scale, fabsq(c));
    if (!(scale > 0)) return fail(1);

    qf worst = 0;
    for (size_t i = 0; i < candidate.roots.size(); ++i) {
        const auto& r = candidate.roots[i];
        if (!finiteq(r.re) || !finiteq(r.im)) return fail(2);
        Cplx<qf> D, Dp;
        d14_struct_eval(sqf, r, D, Dp);
        worst = std::max(worst, cabs(D) / (scale + (qf)1e-300));
        Cplx<qf> f, fl, fv, flv, fll;
        lifted_eval(sqf, r, candidate.lambdas[i], f, fl, fv, flv, fll);
        if (!(cabs(f) + cabs(fl) <= (qf)1e-12 *
                                  ((qf)1 + cabs(r) + cabs(candidate.lambdas[i]))))
            return fail(3);
    }
    if (worst_scalar_residual) *worst_scalar_residual = worst;
    if (!(worst <= (qf)1e-12)) return fail(4);

    // Conjugacy is a global real-coefficient certificate, not a local Newton
    // test.  Real roots are allowed to stand alone; every non-real root must
    // have a matching conjugate among the fourteen outputs.
    for (size_t i = 0; i < candidate.roots.size(); ++i) {
        const auto& a = candidate.roots[i];
        if (fabsq(a.im) <= (qf)1e-10 * ((qf)1 + fabsq(a.re))) continue;
        bool found = false;
        for (size_t j = 0; j < candidate.roots.size(); ++j) {
            if (i == j) continue;
            const auto& b = candidate.roots[j];
            qf e = fabsq(a.re - b.re) + fabsq(a.im + b.im);
            if (e <= (qf)1e-7 * ((qf)1 + fabsq(a.re) + fabsq(a.im))) {
                found = true;
                break;
            }
        }
        if (!found) return fail(5);
    }

    // Vieta first through third power sums.
    for (int power = 1; power <= 3; ++power) {
        Cplx<qf> sum(0, 0);
        for (const auto& r : candidate.roots) {
            Cplx<qf> x(1, 0);
            for (int k = 0; k < power; ++k) x = x * r;
            sum = sum + x;
        }
        const qf c1 = desc[1] / desc[0];
        const qf c2 = desc[2] / desc[0];
        const qf c3 = desc[3] / desc[0];
        qf want_real = 0;
        if (power == 1) want_real = -c1;
        else if (power == 2) want_real = c1 * c1 - 2 * c2;
        else want_real = -c1 * c1 * c1 + 3 * c1 * c2 - 3 * c3;
        Cplx<qf> want(want_real, 0);
        qf err = cabs(sum - want);
        if (!(err <= (qf)2e-8 * ((qf)1 + cabs(want)))) return fail(6);
    }

    // Reconstruct every coefficient from the candidate roots.  This catches
    // a duplicated local basin and a missing root even when each residual is
    // small.
    std::array<Cplx<qf>, 15> rec{};
    rec[0] = Cplx<qf>(1, 0);
    int n = 0;
    for (const auto& root : candidate.roots) {
        std::array<Cplx<qf>, 15> next{};
        next[0] = rec[0];
        for (int k = 1; k <= n + 1; ++k)
            next[k] = (k <= n ? rec[k] : Cplx<qf>(0, 0)) - root * rec[k - 1];
        rec = next;
        ++n;
    }
    qf rec_err = 0;
    for (int k = 0; k <= 14; ++k) {
        // `rec` is monic; normalize the descending input by its leading
        // coefficient before comparing.  Comparing against raw coefficients
        // would report an O(1) false failure whenever D14 is not monic.
        Cplx<qf> want(desc[k] / desc[0], 0);
        rec_err = std::max(rec_err,
                           cabs(rec[k] - want) /
                               ((qf)1 + cabs(want)));
    }
    if (reconstruction_error) *reconstruction_error = rec_err;
    if (!(rec_err <= (qf)2e-7)) return fail(7);
    return true;
}

}  // namespace re_detail
}  // namespace lcbinint::holonomic

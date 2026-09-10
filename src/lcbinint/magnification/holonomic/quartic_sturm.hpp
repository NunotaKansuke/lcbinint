#pragma once

// Certified real-root count for the boundary quartic.
//
// The radial D14 events are the certificate that a cell contains no
// discriminant crossing.  Inside such an open cell the angular topology is
// therefore determined by one boundary quartic at one interior radius.  A
// fixed angular grid cannot certify this: a positive arc may be narrower than
// the grid spacing.  This header supplies the small, degree-4 certificate
// used by classify_cells.
//
// The first tier uses a normalized double Sturm chain.  A sign margin screen
// rejects cancellation-prone chains; the same chain is then rebuilt in DD
// and, if needed, __float128.  The result is accepted only when all pivots
// have a stable sign, the polynomial is square-free at the probe, and the
// count is an even value in {0,2,4}.  An unresolved chain is reported as
// uncertified.  No angular-grid result is used as a certificate or fallback.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/dd_real.hpp"

namespace lcbinint::holonomic {

namespace quartic_sturm_detail {

template <class R>
inline R abs_value(R x) {
    return x < R(0) ? -x : x;
}

template <class R>
inline int sign_value(R x) {
    if (x > R(0)) return 1;
    if (x < R(0)) return -1;
    return 0;
}

template <class R>
inline bool zero_value(R x) {
    return sign_value(x) == 0;
}

template <class R>
inline double as_double_abs(R x) {
    return static_cast<double>(abs_value(x));
}

template <class R>
struct Poly {
    std::array<R, 5> c{};  // ascending order
    int deg = -1;
};

template <class R>
inline Poly<R> trim(Poly<R> p) {
    while (p.deg >= 0 && zero_value(p.c[p.deg])) --p.deg;
    return p;
}

template <class R>
inline Poly<R> derivative(const Poly<R>& p) {
    Poly<R> d;
    if (p.deg <= 0) return d;
    d.deg = p.deg - 1;
    for (int k = 1; k <= p.deg; ++k) d.c[k - 1] = p.c[k] * R(k);
    return trim(d);
}

template <class R>
inline Poly<R> neg_remainder(Poly<R> a, const Poly<R>& b) {
    Poly<R> out = a;
    if (b.deg <= 0 || a.deg < b.deg) {
        for (int k = 0; k <= out.deg; ++k) out.c[k] = -out.c[k];
        return trim(out);
    }
    const R inv_lead = R(1) / b.c[b.deg];
    for (int k = a.deg; k >= b.deg; --k) {
        const R factor = out.c[k] * inv_lead;
        for (int j = 0; j <= b.deg; ++j)
            out.c[k - b.deg + j] =
                out.c[k - b.deg + j] - factor * b.c[j];
        out.c[k] = R(0);
    }
    out.deg = std::min(out.deg, b.deg - 1);
    out = trim(out);
    for (int k = 0; k <= out.deg; ++k) out.c[k] = -out.c[k];
    return out;
}

template <class R>
struct Chain {
    std::array<Poly<R>, 6> p{};
    int n = 0;
};

template <class R>
inline Chain<R> make_chain(const std::array<R, 5>& coeffs) {
    Chain<R> out;
    out.p[0].c = coeffs;
    out.p[0].deg = 4;
    out.p[0] = trim(out.p[0]);
    if (out.p[0].deg < 0) return out;
    out.n = 1;
    if (out.p[0].deg == 0) return out;

    out.p[1] = derivative(out.p[0]);
    out.n = 2;
    while (out.n < static_cast<int>(out.p.size()) && out.p[out.n - 1].deg >= 0) {
        Poly<R> r = neg_remainder(out.p[out.n - 2], out.p[out.n - 1]);
        if (r.deg < 0) break;
        out.p[out.n++] = r;
        if (r.deg == 0) break;
    }
    return out;
}

template <class R>
inline R eval(const Poly<R>& p, R x) {
    if (p.deg < 0) return R(0);
    R y = p.c[p.deg];
    for (int k = p.deg - 1; k >= 0; --k) y = y * x + p.c[k];
    return y;
}

template <class R>
inline int variation_at(const Chain<R>& chain, R x) {
    int previous = 0;
    int variations = 0;
    for (int i = 0; i < chain.n; ++i) {
        const int s = sign_value(eval(chain.p[i], x));
        if (s == 0) continue;
        if (previous != 0 && s != previous) ++variations;
        previous = s;
    }
    return variations;
}

template <class R>
inline int variation_at_infinity(const Chain<R>& chain, int side) {
    int previous = 0;
    int variations = 0;
    for (int i = 0; i < chain.n; ++i) {
        const Poly<R>& p = chain.p[i];
        if (p.deg < 0) continue;
        int s = sign_value(p.c[p.deg]);
        if (side < 0 && (p.deg & 1)) s = -s;
        if (s == 0) continue;
        if (previous != 0 && s != previous) ++variations;
        previous = s;
    }
    return variations;
}

template <class R>
inline double min_leading_margin(const Chain<R>& chain) {
    double margin = 1.0;
    bool saw = false;
    for (int i = 0; i < chain.n; ++i) {
        const Poly<R>& p = chain.p[i];
        if (p.deg < 0) continue;
        double scale = 0.0;
        for (int k = 0; k <= p.deg; ++k)
            scale = std::max(scale, as_double_abs(p.c[k]));
        const double lead = as_double_abs(p.c[p.deg]);
        if (!(scale > 0.0) || !std::isfinite(scale) || !std::isfinite(lead))
            return 0.0;
        margin = std::min(margin, lead / scale);
        saw = true;
    }
    return saw ? margin : 0.0;
}

template <class R>
struct Count {
    Chain<R> chain{};
    int degree = -1;
    int count = -1;
    bool square_free = false;
    double margin = 0.0;
};

template <class R>
inline Count<R> count_chain(const std::array<R, 5>& coeffs) {
    Count<R> out;
    out.chain = make_chain(coeffs);
    out.degree = out.chain.p[0].deg;
    out.margin = min_leading_margin(out.chain);
    if (out.degree <= 0) {
        out.count = 0;
        out.square_free = out.degree == 0;
        return out;
    }
    const int vm = variation_at_infinity(out.chain, -1);
    const int vp = variation_at_infinity(out.chain, +1);
    out.count = vm - vp;
    out.square_free = out.chain.n > 0 &&
                      out.chain.p[out.chain.n - 1].deg == 0;
    return out;
}

inline double max_abs(const std::array<double, 5>& p) {
    double s = 0.0;
    for (double x : p) s = std::max(s, std::fabs(x));
    return s;
}

template <class R>
inline std::array<R, 5> normalized(const std::array<double, 5>& p) {
    const double scale = max_abs(p);
    std::array<R, 5> out{};
    if (!(scale > 0.0) || !std::isfinite(scale)) return out;
    for (int k = 0; k < 5; ++k) out[k] = R(p[k]) / R(scale);
    return out;
}

inline bool valid_count(int degree, int count) {
    return degree == 4 && count >= 0 && count <= 4 && !(count & 1);
}

inline double required_margin(int tier) {
    if (tier == 0)
        return 4096.0 * std::numeric_limits<double>::epsilon();
    if (tier == 1)
        return std::ldexp(1.0, -80);
    return std::ldexp(1.0, -100);
}

template <class R>
inline bool stable(const Count<R>& result, int tier) {
    return result.square_free && valid_count(result.degree, result.count) &&
           result.margin > required_margin(tier);
}

inline double cauchy_bound(const std::array<double, 5>& p, bool reciprocal) {
    const double lead = std::fabs(reciprocal ? p[0] : p[4]);
    if (!(lead > 0.0) || !std::isfinite(lead))
        return std::numeric_limits<double>::infinity();
    double ratio = 0.0;
    if (!reciprocal) {
        for (int k = 0; k < 4; ++k)
            ratio = std::max(ratio, std::fabs(p[k]) / lead);
    } else {
        // The reciprocal polynomial is
        // q(u) = p4 - p3*u + p2*u^2 - p1*u^3 + p0*u^4.
        // Its non-leading coefficients are p4,p3,p2,p1.  Include p4 and
        // exclude p0 itself; the latter is the reciprocal leading term.
        for (int k = 0; k < 4; ++k)
            ratio = std::max(ratio, std::fabs(p[4 - k]) / lead);
    }
    return 2.0 * (1.0 + ratio) + 1.0;
}

template <class R>
struct Interval {
    R lo, hi;
    int count;
    int depth;
};

template <class R>
inline std::vector<double> isolate_roots(const std::array<double, 5>& p,
                                         int expected) {
    const auto coeffs = normalized<R>(p);
    const Count<R> counted = count_chain(coeffs);
    if (!stable(counted, 2) || counted.count != expected) return {};

    R max_ratio = R(0);
    const R lead = abs_value(R(p[4]));
    if (!(lead > R(0))) return {};
    for (int k = 0; k < 4; ++k) {
        const R ratio = abs_value(R(p[k])) / lead;
        if (ratio > max_ratio) max_ratio = ratio;
    }
    const R bound = R(2) * (R(1) + max_ratio) + R(1);
    const R lo = -bound, hi = bound;
    const int vlo = variation_at(counted.chain, lo);
    const int vhi = variation_at(counted.chain, hi);
    if (vlo - vhi != expected) return {};

    std::vector<Interval<R>> pending;
    pending.push_back({lo, hi, expected, 0});
    std::vector<R> roots;
    roots.reserve(static_cast<size_t>(expected));
    while (!pending.empty()) {
        const Interval<R> current = pending.back();
        pending.pop_back();
        if (current.count <= 0) continue;
        if (current.count == 1) {
            R a = current.lo, b = current.hi;
            for (int it = 0; it < 180; ++it) {
                const R m = (a + b) / R(2);
                const int left = variation_at(counted.chain, a) -
                                 variation_at(counted.chain, m);
                if (left > 0)
                    b = m;
                else
                    a = m;
            }
            roots.push_back((a + b) / R(2));
            continue;
        }
        if (current.depth >= 80) return {};
        const R m = (current.lo + current.hi) / R(2);
        const int left = variation_at(counted.chain, current.lo) -
                         variation_at(counted.chain, m);
        const int right = current.count - left;
        if (left < 0 || right < 0 || left + right != current.count)
            return {};
        pending.push_back({m, current.hi, right, current.depth + 1});
        pending.push_back({current.lo, m, left, current.depth + 1});
    }
    if (static_cast<int>(roots.size()) != expected) return {};
    std::vector<double> out;
    out.reserve(roots.size());
    for (R root : roots) out.push_back(static_cast<double>(root));
    std::sort(out.begin(), out.end());
    for (size_t i = 1; i < out.size(); ++i) {
        if (!(out[i] > out[i - 1]) ||
            out[i] - out[i - 1] <= 1e-13 * (1.0 + std::fabs(out[i])))
            return {};
    }
    return out;
}

}  // namespace quartic_sturm_detail

struct QuarticSturmCertificate {
    int root_count = -1;
    int precision_tier = -1;  // 0=double, 1=DD, 2=__float128
    bool certified = false;
    bool reciprocal = false;
    bool chart_singular = false;  // exact p4=0: root at t=infinity
    double sign_margin = 0.0;
    std::array<double, 5> chart_coeffs{};
};

inline QuarticSturmCertificate certify_quartic(
    const std::array<double, 5>& p) {
    using namespace quartic_sturm_detail;
    QuarticSturmCertificate out;
    const double scale = max_abs(p);
    if (!(scale > 0.0) || !std::isfinite(scale)) return out;
    for (double x : p)
        if (!std::isfinite(x)) return out;

    const bool direct = p[4] != 0.0;
    const bool reciprocal = p[0] != 0.0;
    if (!direct && !reciprocal) return out;
    out.chart_singular = !direct;

    const double direct_bound = direct ? cauchy_bound(p, false)
                                       : std::numeric_limits<double>::infinity();
    const double reciprocal_bound = reciprocal
                                        ? cauchy_bound(p, true)
                                        : std::numeric_limits<double>::infinity();
    out.reciprocal = reciprocal && (!direct || reciprocal_bound < direct_bound);
    if (out.reciprocal)
        out.chart_coeffs = {p[4], -p[3], p[2], -p[1], p[0]};
    else
        out.chart_coeffs = p;

    const auto d = count_chain(normalized<double>(out.chart_coeffs));
    out.root_count = d.count;
    out.sign_margin = d.margin;
    // A zero t^4 coefficient is a chart event, not automatically a failure:
    // when p0 != 0 the reciprocal polynomial is a full degree-4 projective
    // chart and its u=0 root represents theta=pi.  Only the selected chart's
    // square-free/sign certificate matters here.  A multiple u=0 root still
    // fails the square-free test and is handled fail-closed.
    if (stable(d, 0)) {
        out.precision_tier = 0;
        out.certified = true;
        return out;
    }

    const auto dd = count_chain(normalized<DD>(out.chart_coeffs));
    out.root_count = dd.count;
    out.sign_margin = dd.margin;
    if (stable(dd, 1)) {
        out.precision_tier = 1;
        out.certified = true;
        return out;
    }

    const auto qf = count_chain(normalized<__float128>(out.chart_coeffs));
    out.root_count = qf.count;
    out.sign_margin = qf.margin;
    if (stable(qf, 2)) {
        out.precision_tier = 2;
        out.certified = true;
    }
    return out;
}

// Research/diagnostic fallback after the existing complex quartic root set
// disagrees with the certified count.  The caller supplies the chart selected
// by certify_quartic(); this function returns real roots in that chart.
inline std::vector<double> sturm_isolate_real_roots(
    const std::array<double, 5>& chart_coeffs, int expected) {
    return quartic_sturm_detail::isolate_roots<__float128>(chart_coeffs,
                                                            expected);
}

}  // namespace lcbinint::holonomic

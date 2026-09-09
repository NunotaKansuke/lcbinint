#pragma once

// Equation-derived Gauss--Manin connection for the boundary period family.
//
// Q(t;R) = P(t;R) (1+t^2) ((R-a)^2 + (R+a)^2 t^2),  deg_t Q = 8.
// For eta_k = t^k dt/sqrt(Q), k=0..6, the coefficients C_k are generated
// from the polynomial identity
//
//   -1/2 t^k Q_R = C_k Q + S'_k Q - 1/2 S_k Q_t,
//
// with S_k = t^k Q_R (Q_t^{-1} mod Q) mod Q.  This file deliberately does
// not sample or fit an integrand.  Its Taylor coefficients are obtained by
// doing the same algebra over truncated power series in h=R-Rc.
//
// The implementation is header-only and isolated with the other holonomic
// headers.  It is intended first as an independently testable research
// kernel; production period/flux routing remains unchanged until the
// transport and endpoint gates pass the same V2 accuracy checks.

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <quadmath.h>

#include "lcbinint/magnification/holonomic/dd_real.hpp"
#include "lcbinint/magnification/holonomic/lens_frame.hpp"

namespace lcbinint::holonomic {

constexpr int kGmEtaDim = 7;
constexpr int kGmQDegree = 8;
// The largest product used by the fixed degree-8 reduction is degree 15.
// Keeping a 32-degree scratch polynomial made every returned object carry
// 17 never-touched series coefficients.  The connection code below is fixed
// to Q degree 8, so capacity 16 is sufficient for all intermediate products.
constexpr int kGmPolyCapacity = 16;

// A small forward-mode parameter dual used by the derivative-inclusive
// kernel.  Parameter order is (X, Y, rho, m0, a), matching PrimaryFrame.
struct GmDual5 {
    double value = 0.0;
    std::array<double, 5> deriv{};

    GmDual5() = default;
    GmDual5(double v) : value(v) {}

    static GmDual5 variable(int index, double v) {
        GmDual5 out(v);
        if (index >= 0 && index < 5) out.deriv[index] = 1.0;
        return out;
    }
};

inline GmDual5 operator+(GmDual5 a, GmDual5 b) {
    GmDual5 r;
    r.value = a.value + b.value;
    for (int i = 0; i < 5; ++i) r.deriv[i] = a.deriv[i] + b.deriv[i];
    return r;
}
inline GmDual5 operator-(GmDual5 a, GmDual5 b) {
    GmDual5 r;
    r.value = a.value - b.value;
    for (int i = 0; i < 5; ++i) r.deriv[i] = a.deriv[i] - b.deriv[i];
    return r;
}
inline GmDual5 operator-(GmDual5 a) {
    GmDual5 r;
    r.value = -a.value;
    for (int i = 0; i < 5; ++i) r.deriv[i] = -a.deriv[i];
    return r;
}
inline GmDual5 operator*(GmDual5 a, GmDual5 b) {
    GmDual5 r;
    r.value = a.value * b.value;
    for (int i = 0; i < 5; ++i)
        r.deriv[i] = a.deriv[i] * b.value + a.value * b.deriv[i];
    return r;
}
inline GmDual5 operator/(GmDual5 a, GmDual5 b) {
    GmDual5 r;
    const double inv = 1.0 / b.value;
    r.value = a.value * inv;
    const double inv2 = inv * inv;
    for (int i = 0; i < 5; ++i)
        r.deriv[i] = (a.deriv[i] * b.value - a.value * b.deriv[i]) * inv2;
    return r;
}

inline double gm_abs_value(double x) { return std::fabs(x); }
inline double gm_abs_value(long double x) { return (double)std::fabs(x); }
inline double gm_abs_value(__float128 x) { return (double)fabsq(x); }
inline double gm_abs_value(DD x) { return std::fabs(x.hi + x.lo); }
inline double gm_abs_value(const GmDual5& x) { return std::fabs(x.value); }
inline bool gm_finite(double x) { return std::isfinite(x); }
inline bool gm_finite(long double x) { return std::isfinite(x); }
inline bool gm_finite(__float128 x) { return finiteq(x); }
inline bool gm_finite(DD x) {
    return std::isfinite(x.hi) && std::isfinite(x.lo);
}
inline bool gm_finite(const GmDual5& x) {
    if (!std::isfinite(x.value)) return false;
    for (double d : x.deriv)
        if (!std::isfinite(d)) return false;
    return true;
}

inline double gm_quality_limit(double) { return 1e-6; }
inline double gm_quality_limit(long double) { return 1e-10; }
inline double gm_quality_limit(__float128) { return 1e-12; }
inline double gm_quality_limit(DD) { return 1e-12; }
inline double gm_quality_limit(const GmDual5&) { return 1e-6; }

// Truncated scalar series in h=R-Rc, with coefficients in Scalar.
template <int Order, class Scalar = double>
struct GmSeries {
    std::array<Scalar, Order + 1> c{};

    GmSeries() = default;
    GmSeries(const Scalar& x) { c[0] = x; }

    static GmSeries constant(const Scalar& x) { return GmSeries(x); }
};

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> operator+(const GmSeries<Order, Scalar>& a,
                                         const GmSeries<Order, Scalar>& b) {
    GmSeries<Order, Scalar> r;
    for (int i = 0; i <= Order; ++i) r.c[i] = a.c[i] + b.c[i];
    return r;
}
template <int Order, class Scalar>
inline GmSeries<Order, Scalar> operator-(const GmSeries<Order, Scalar>& a,
                                         const GmSeries<Order, Scalar>& b) {
    GmSeries<Order, Scalar> r;
    for (int i = 0; i <= Order; ++i) r.c[i] = a.c[i] - b.c[i];
    return r;
}
template <int Order, class Scalar>
inline GmSeries<Order, Scalar> operator-(const GmSeries<Order, Scalar>& a) {
    GmSeries<Order, Scalar> r;
    for (int i = 0; i <= Order; ++i) r.c[i] = -a.c[i];
    return r;
}
template <int Order, class Scalar>
inline void gm_series_mul_accumulate(GmSeries<Order, Scalar>& dst,
                                     const GmSeries<Order, Scalar>& a,
                                     const GmSeries<Order, Scalar>& b) {
    for (int i = 0; i <= Order; ++i)
        for (int j = 0; j + i <= Order; ++j)
            dst.c[i + j] = dst.c[i + j] + a.c[i] * b.c[j];
}

template <int Order, class Scalar>
inline void gm_series_mul_subtract(GmSeries<Order, Scalar>& dst,
                                   const GmSeries<Order, Scalar>& a,
                                   const GmSeries<Order, Scalar>& b) {
    for (int i = 0; i <= Order; ++i)
        for (int j = 0; j + i <= Order; ++j)
            dst.c[i + j] = dst.c[i + j] - a.c[i] * b.c[j];
}

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> operator*(const GmSeries<Order, Scalar>& a,
                                         const GmSeries<Order, Scalar>& b) {
    GmSeries<Order, Scalar> r;
    gm_series_mul_accumulate(r, a, b);
    return r;
}

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> gm_series_scale_scalar(
    const GmSeries<Order, Scalar>& a, const Scalar& x) {
    GmSeries<Order, Scalar> r;
    for (int i = 0; i <= Order; ++i) r.c[i] = a.c[i] * x;
    return r;
}

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> gm_series_inv(const GmSeries<Order, Scalar>& a) {
    GmSeries<Order, Scalar> r;
    r.c[0] = Scalar(1) / a.c[0];
    for (int n = 1; n <= Order; ++n) {
        Scalar sum = Scalar(0);
        for (int j = 1; j <= n; ++j) sum = sum + a.c[j] * r.c[n - j];
        r.c[n] = -r.c[0] * sum;
    }
    return r;
}

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> operator/(const GmSeries<Order, Scalar>& a,
                                         const GmSeries<Order, Scalar>& b) {
    return a * gm_series_inv(b);
}

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> gm_series_derivative(
    const GmSeries<Order, Scalar>& a) {
    GmSeries<Order, Scalar> r;
    for (int i = 1; i <= Order; ++i) r.c[i - 1] = a.c[i] * Scalar(i);
    return r;
}

template <int Order, class Scalar>
inline bool gm_series_finite(const GmSeries<Order, Scalar>& a) {
    for (const Scalar& x : a.c)
        if (!gm_finite(x)) return false;
    return true;
}

template <int Order, class Scalar>
inline double gm_series_max_abs(const GmSeries<Order, Scalar>& a) {
    double r = 0.0;
    for (const Scalar& x : a.c) r = std::max(r, gm_abs_value(x));
    return r;
}

template <class T>
struct GmComplex {
    T re{};
    T im{};
    GmComplex() = default;
    GmComplex(T r, T i) : re(r), im(i) {}
};

template <class T>
inline GmComplex<T> operator+(const GmComplex<T>& a, const GmComplex<T>& b) {
    return {a.re + b.re, a.im + b.im};
}
template <class T>
inline GmComplex<T> operator-(const GmComplex<T>& a, const GmComplex<T>& b) {
    return {a.re - b.re, a.im - b.im};
}
template <class T>
inline GmComplex<T> operator-(const GmComplex<T>& a) {
    return {-a.re, -a.im};
}
template <class T>
inline GmComplex<T> operator*(const GmComplex<T>& a, const GmComplex<T>& b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <class T>
inline GmComplex<T> gm_complex_scale(const GmComplex<T>& a, const T& x) {
    return {a.re * x, a.im * x};
}
template <class T>
inline T gm_complex_norm(const GmComplex<T>& a) {
    return a.re * a.re + a.im * a.im;
}
template <class T>
inline T gm_complex_re_cross(const GmComplex<T>& a, const GmComplex<T>& b) {
    return a.re * b.re + a.im * b.im;
}

// Fixed-capacity ascending polynomial.  Degrees are kept explicitly; the
// callers below know the formal degrees (so no coefficient-dependent trim is
// needed while working over a dual or a series ring).
template <class T, int Capacity = kGmPolyCapacity>
struct GmPoly {
    std::array<T, Capacity + 1> c{};
    int deg = -1;
};

template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_add(const GmPoly<T, Capacity>& a,
                                       const GmPoly<T, Capacity>& b) {
    GmPoly<T, Capacity> r;
    r.deg = std::max(a.deg, b.deg);
    for (int i = 0; i <= r.deg; ++i) r.c[i] = a.c[i] + b.c[i];
    return r;
}
template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_sub(const GmPoly<T, Capacity>& a,
                                       const GmPoly<T, Capacity>& b) {
    GmPoly<T, Capacity> r;
    r.deg = std::max(a.deg, b.deg);
    for (int i = 0; i <= r.deg; ++i) r.c[i] = a.c[i] - b.c[i];
    return r;
}
template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_mul(const GmPoly<T, Capacity>& a,
                                       const GmPoly<T, Capacity>& b) {
    GmPoly<T, Capacity> r;
    if (a.deg < 0 || b.deg < 0) return r;
    r.deg = std::min(Capacity, a.deg + b.deg);
    for (int i = 0; i <= a.deg; ++i)
        for (int j = 0; j <= b.deg && i + j <= Capacity; ++j)
            r.c[i + j] = r.c[i + j] + a.c[i] * b.c[j];
    return r;
}

// The GM kernel uses a polynomial over a truncated series ring.  The generic
// version above computes one temporary series product and then adds that
// temporary series to the result for every polynomial coefficient pair.  This
// overload fuses those two operations and avoids the repeated full-series
// addition and its temporary object.
template <int Order, class Scalar, int Capacity>
inline GmPoly<GmSeries<Order, Scalar>, Capacity> gm_poly_mul(
    const GmPoly<GmSeries<Order, Scalar>, Capacity>& a,
    const GmPoly<GmSeries<Order, Scalar>, Capacity>& b) {
    using Series = GmSeries<Order, Scalar>;
    GmPoly<Series, Capacity> r;
    if (a.deg < 0 || b.deg < 0) return r;
    r.deg = std::min(Capacity, a.deg + b.deg);
    for (int i = 0; i <= a.deg; ++i)
        for (int j = 0; j <= b.deg && i + j <= Capacity; ++j) {
            if constexpr (std::is_same_v<Scalar, GmDual5>) {
                // Keep the original operation grouping for the derivative
                // lane.  Its finite-difference parity is more sensitive to
                // reassociation than the value-only double/qf lanes.
                const Series term = a.c[i] * b.c[j];
                r.c[i + j] = r.c[i + j] + term;
            } else {
                gm_series_mul_accumulate(r.c[i + j], a.c[i], b.c[j]);
            }
        }
    return r;
}
template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_scale(const GmPoly<T, Capacity>& a,
                                         const T& x) {
    GmPoly<T, Capacity> r;
    r.deg = a.deg;
    for (int i = 0; i <= a.deg; ++i) r.c[i] = a.c[i] * x;
    return r;
}

template <int Order, class Scalar, int Capacity>
inline GmPoly<GmSeries<Order, Scalar>, Capacity> gm_poly_scale_scalar(
    const GmPoly<GmSeries<Order, Scalar>, Capacity>& a, const Scalar& x) {
    using Series = GmSeries<Order, Scalar>;
    GmPoly<Series, Capacity> r;
    r.deg = a.deg;
    for (int i = 0; i <= a.deg; ++i)
        r.c[i] = gm_series_scale_scalar(a.c[i], x);
    return r;
}

template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_shift(const GmPoly<T, Capacity>& a,
                                         int shift) {
    GmPoly<T, Capacity> r;
    if (a.deg < 0 || shift > Capacity) return r;
    r.deg = std::min(Capacity, a.deg + shift);
    for (int i = 0; i <= a.deg && i + shift <= Capacity; ++i)
        r.c[i + shift] = a.c[i];
    return r;
}
template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_derivative(const GmPoly<T, Capacity>& a) {
    GmPoly<T, Capacity> r;
    r.deg = std::max(-1, a.deg - 1);
    for (int i = 1; i <= a.deg; ++i) r.c[i - 1] = a.c[i] * T(i);
    return r;
}

template <int Order, class Scalar, int Capacity>
inline GmPoly<GmSeries<Order, Scalar>, Capacity> gm_poly_derivative(
    const GmPoly<GmSeries<Order, Scalar>, Capacity>& a) {
    using Series = GmSeries<Order, Scalar>;
    GmPoly<Series, Capacity> r;
    r.deg = std::max(-1, a.deg - 1);
    for (int i = 1; i <= a.deg; ++i) {
        if constexpr (std::is_same_v<Scalar, GmDual5>)
            r.c[i - 1] = a.c[i] * Series(Scalar(i));
        else
            r.c[i - 1] = gm_series_scale_scalar(a.c[i], Scalar(i));
    }
    return r;
}

template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_mod(const GmPoly<T, Capacity>& p,
                                       const GmPoly<T, Capacity>& q) {
    GmPoly<T, Capacity> r = p;
    if (q.deg < 0 || r.deg < q.deg) return r;
    for (int d = r.deg; d >= q.deg; --d) {
        const T factor = r.c[d] / q.c[q.deg];
        const int shift = d - q.deg;
        for (int j = 0; j <= q.deg; ++j)
            r.c[shift + j] = r.c[shift + j] - factor * q.c[j];
        r.c[d] = T(0);
    }
    r.deg = q.deg - 1;
    return r;
}

// Fixed-Q reduction avoids constructing a truncated-series reciprocal for
// every long-division step.  In the GM kernel q is always the degree-8 Q and
// inv_lead is computed once per connection jet.
template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_mod_with_lead_inv(
    const GmPoly<T, Capacity>& p, const GmPoly<T, Capacity>& q,
    const T& inv_lead) {
    GmPoly<T, Capacity> r = p;
    if (q.deg < 0 || r.deg < q.deg) return r;
    for (int d = r.deg; d >= q.deg; --d) {
        const T factor = r.c[d] * inv_lead;
        const int shift = d - q.deg;
        for (int j = 0; j <= q.deg; ++j)
            r.c[shift + j] = r.c[shift + j] - factor * q.c[j];
        r.c[d] = T(0);
    }
    r.deg = q.deg - 1;
    return r;
}

template <int Order, class Scalar, int Capacity>
inline GmPoly<GmSeries<Order, Scalar>, Capacity> gm_poly_mod_with_lead_inv(
    const GmPoly<GmSeries<Order, Scalar>, Capacity>& p,
    const GmPoly<GmSeries<Order, Scalar>, Capacity>& q,
    const GmSeries<Order, Scalar>& inv_lead) {
    using Series = GmSeries<Order, Scalar>;
    GmPoly<Series, Capacity> r = p;
    if (q.deg < 0 || r.deg < q.deg) return r;
    for (int d = r.deg; d >= q.deg; --d) {
        const Series factor = r.c[d] * inv_lead;
        const int shift = d - q.deg;
        for (int j = 0; j <= q.deg; ++j) {
            if constexpr (std::is_same_v<Scalar, GmDual5>)
                r.c[shift + j] = r.c[shift + j] - factor * q.c[j];
            else
                gm_series_mul_subtract(r.c[shift + j], factor, q.c[j]);
        }
        r.c[d] = Series{};
    }
    r.deg = q.deg - 1;
    return r;
}

template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_div_exact(const GmPoly<T, Capacity>& p,
                                             const GmPoly<T, Capacity>& q) {
    GmPoly<T, Capacity> r = p;
    GmPoly<T, Capacity> out;
    if (q.deg < 0 || p.deg < q.deg) return out;
    out.deg = p.deg - q.deg;
    for (int d = p.deg; d >= q.deg; --d) {
        const T factor = r.c[d] / q.c[q.deg];
        out.c[d - q.deg] = factor;
        const int shift = d - q.deg;
        for (int j = 0; j <= q.deg; ++j)
            r.c[shift + j] = r.c[shift + j] - factor * q.c[j];
        r.c[d] = T(0);
    }
    return out;
}

// Exact division with a reusable inverse of q's leading series.  The
// residual is measured after the same elimination that constructs the
// quotient, so it supplies the GM identity gate without rebuilding three
// polynomial products for every eta row.
template <class T, int Capacity>
inline GmPoly<T, Capacity> gm_poly_div_exact_with_lead_inv(
    const GmPoly<T, Capacity>& p, const GmPoly<T, Capacity>& q,
    const T& inv_lead, double* remainder_max = nullptr) {
    GmPoly<T, Capacity> r = p;
    GmPoly<T, Capacity> out;
    if (q.deg < 0 || p.deg < q.deg) {
        if (remainder_max) *remainder_max = 0.0;
        return out;
    }
    out.deg = p.deg - q.deg;
    for (int d = p.deg; d >= q.deg; --d) {
        const T factor = r.c[d] * inv_lead;
        out.c[d - q.deg] = factor;
        const int shift = d - q.deg;
        for (int j = 0; j <= q.deg; ++j)
            r.c[shift + j] = r.c[shift + j] - factor * q.c[j];
        r.c[d] = T(0);
    }
    if (remainder_max) {
        double worst = 0.0;
        for (int i = 0; i < q.deg; ++i)
            worst = std::max(worst, gm_series_max_abs(r.c[i]));
        *remainder_max = worst;
    }
    return out;
}

template <int Order, class Scalar, int Capacity>
inline GmPoly<GmSeries<Order, Scalar>, Capacity>
gm_poly_div_exact_with_lead_inv(
    const GmPoly<GmSeries<Order, Scalar>, Capacity>& p,
    const GmPoly<GmSeries<Order, Scalar>, Capacity>& q,
    const GmSeries<Order, Scalar>& inv_lead, double* remainder_max = nullptr) {
    using Series = GmSeries<Order, Scalar>;
    GmPoly<Series, Capacity> r = p;
    GmPoly<Series, Capacity> out;
    if (q.deg < 0 || p.deg < q.deg) {
        if (remainder_max) *remainder_max = 0.0;
        return out;
    }
    out.deg = p.deg - q.deg;
    for (int d = p.deg; d >= q.deg; --d) {
        const Series factor = r.c[d] * inv_lead;
        out.c[d - q.deg] = factor;
        const int shift = d - q.deg;
        for (int j = 0; j <= q.deg; ++j) {
            if constexpr (std::is_same_v<Scalar, GmDual5>)
                r.c[shift + j] = r.c[shift + j] - factor * q.c[j];
            else
                gm_series_mul_subtract(r.c[shift + j], factor, q.c[j]);
        }
        r.c[d] = Series{};
    }
    if (remainder_max) {
        double worst = 0.0;
        for (int i = 0; i < q.deg; ++i)
            worst = std::max(worst, gm_series_max_abs(r.c[i]));
        *remainder_max = worst;
    }
    return out;
}

template <class Scalar>
struct GmParams {
    Scalar X{};
    Scalar Y{};
    Scalar rho{};
    Scalar m0{};
    Scalar a{};
};

template <int Order, class Scalar>
struct GmConnectionJet {
    using Series = GmSeries<Order, Scalar>;
    using Poly = GmPoly<Series, kGmPolyCapacity>;

    bool ok = false;
    bool degree_ok = false;
    bool finite = false;
    bool squarefree = false;
    bool quality_ok = false;
    double identity_residual = std::numeric_limits<double>::infinity();
    double matrix_pivot_rel = 0.0;
    double chart_scale = 1.0;
    Poly q;
    Poly qR;
    std::array<std::array<Series, kGmEtaDim>, kGmEtaDim> C{};
    std::array<Poly, kGmEtaDim> S{};
};

template <class Scalar>
struct GmLU8 {
    std::array<std::array<Scalar, kGmQDegree>, kGmQDegree> lu{};
    std::array<int, kGmQDegree> row_pivot{};
    double pivot_rel = 0.0;
    bool ok = false;

    std::array<Scalar, kGmQDegree> solve(
        const std::array<Scalar, kGmQDegree>& rhs) const {
        std::array<Scalar, kGmQDegree> b{};
        for (int i = 0; i < kGmQDegree; ++i) b[i] = rhs[row_pivot[i]];
        for (int i = 0; i < kGmQDegree; ++i)
            for (int j = 0; j < i; ++j) b[i] = b[i] - lu[i][j] * b[j];
        for (int i = kGmQDegree - 1; i >= 0; --i) {
            for (int j = i + 1; j < kGmQDegree; ++j)
                b[i] = b[i] - lu[i][j] * b[j];
            b[i] = b[i] / lu[i][i];
        }
        return b;
    }
};

template <class Scalar>
inline GmLU8<Scalar> gm_lu_factor(
    const std::array<std::array<Scalar, kGmQDegree>, kGmQDegree>& input) {
    GmLU8<Scalar> out;
    out.lu = input;
    for (int i = 0; i < kGmQDegree; ++i) out.row_pivot[i] = i;

    double scale = 0.0;
    for (const auto& row : input)
        for (const Scalar& x : row) scale = std::max(scale, gm_abs_value(x));
    if (!(scale > 0.0) || !std::isfinite(scale)) return out;

    double min_pivot = std::numeric_limits<double>::infinity();
    for (int k = 0; k < kGmQDegree; ++k) {
        int piv = k;
        double best = gm_abs_value(out.lu[k][k]);
        for (int i = k + 1; i < kGmQDegree; ++i) {
            const double candidate = gm_abs_value(out.lu[i][k]);
            if (candidate > best) { best = candidate; piv = i; }
        }
        if (!(best > 0.0) || !std::isfinite(best)) return out;
        if (piv != k) {
            std::swap(out.lu[piv], out.lu[k]);
            std::swap(out.row_pivot[piv], out.row_pivot[k]);
        }
        min_pivot = std::min(min_pivot, best);
        for (int i = k + 1; i < kGmQDegree; ++i) {
            out.lu[i][k] = out.lu[i][k] / out.lu[k][k];
            for (int j = k + 1; j < kGmQDegree; ++j)
                out.lu[i][j] = out.lu[i][j] - out.lu[i][k] * out.lu[k][j];
        }
    }
    out.pivot_rel = min_pivot / scale;
    out.ok = out.pivot_rel > 1e-30 && std::isfinite(out.pivot_rel);
    return out;
}

// Construct the degree-8 Q and its exact h derivative over a series ring.
template <int Order, class Scalar>
inline GmPoly<GmSeries<Order, Scalar>, kGmPolyCapacity> gm_q_series_raw(
    double Rc, const GmParams<Scalar>& p,
    GmPoly<GmSeries<Order, Scalar>, kGmPolyCapacity>& qR_out) {
    using Series = GmSeries<Order, Scalar>;
    using Poly = GmPoly<Series, kGmPolyCapacity>;
    using C = GmComplex<Series>;

    const Series one(Scalar(1));
    Series R{Scalar(Rc)};
    if constexpr (Order >= 1) R.c[1] = Scalar(1);
    const Series a(p.a), m0(p.m0), X(p.X), Y(p.Y), rho(p.rho);
    const Series R2 = R * R;
    const C zeta(X, Y);

    const C n0(-zeta.re * R2, -zeta.im * R2);
    const C n1 = gm_complex_scale(zeta, a * R)
               + C(R * (R2 - one), Series(Scalar(0)));
    const Series n2s = a * (m0 - R2);
    const C n2(n2s, Series(Scalar(0)));
    const C cT0 = n0 + n1 + n2;
    const C dT = n2 - n0;
    C cT1;
    Series a1, a2, a3;
    if constexpr (std::is_same_v<Scalar, GmDual5>) {
        // Preserve the established dual arithmetic grouping so its
        // finite-difference parity remains comparable with the legacy jet.
        cT1 = C(-Series(Scalar(2)) * dT.im,
                Series(Scalar(2)) * dT.re);
    } else {
        cT1 = C(gm_series_scale_scalar(dT.im, Scalar(-2)),
                gm_series_scale_scalar(dT.re, Scalar(2)));
    }
    const C cT2 = n1 - n0 - n2;

    const Series rho2 = rho * rho;
    const Series k = rho2 * R2;
    const Series bm = (R - a) * (R - a);
    const Series bp = (R + a) * (R + a);
    const Series a0 = gm_complex_norm(cT0);
    if constexpr (std::is_same_v<Scalar, GmDual5>) {
        a1 = Series(Scalar(2)) * gm_complex_re_cross(cT0, cT1);
        a2 = gm_complex_norm(cT1)
           + Series(Scalar(2)) * gm_complex_re_cross(cT0, cT2);
        a3 = Series(Scalar(2)) * gm_complex_re_cross(cT1, cT2);
    } else {
        a1 = gm_series_scale_scalar(
            gm_complex_re_cross(cT0, cT1), Scalar(2));
        a2 = gm_complex_norm(cT1)
           + gm_series_scale_scalar(
               gm_complex_re_cross(cT0, cT2), Scalar(2));
        a3 = gm_series_scale_scalar(
            gm_complex_re_cross(cT1, cT2), Scalar(2));
    }
    const Series a4 = gm_complex_norm(cT2);

    Poly P;
    P.deg = 4;
    P.c[0] = k * bm - a0;
    P.c[1] = -a1;
    P.c[2] = k * (bm + bp) - a2;
    P.c[3] = -a3;
    P.c[4] = k * bp - a4;

    if constexpr (std::is_same_v<Scalar, GmDual5>) {
        Poly A;
        A.deg = 2;
        A.c[0] = one;
        A.c[2] = one;
        Poly B;
        B.deg = 2;
        B.c[0] = bm;
        B.c[2] = bp;
        const Poly Q = gm_poly_mul(gm_poly_mul(P, A), B);
        qR_out.deg = Q.deg;
        for (int i = 0; i <= Q.deg; ++i)
            qR_out.c[i] = gm_series_derivative(Q.c[i]);
        return Q;
    }

    // A(t)B(t) is even.  Expand its three nonzero coefficients directly;
    // the generic polynomial products used here previously accounted for a
    // large fraction of the __float128 profile despite the fixed degrees.
    const Series d0 = bm;
    const Series d2 = bm + bp;
    const Series d4 = bp;
    Poly Q;
    Q.deg = 8;
    Q.c[0] = P.c[0] * d0;
    Q.c[1] = P.c[1] * d0;
    Q.c[2] = P.c[2] * d0 + P.c[0] * d2;
    Q.c[3] = P.c[3] * d0 + P.c[1] * d2;
    Q.c[4] = P.c[4] * d0 + P.c[2] * d2 + P.c[0] * d4;
    Q.c[5] = P.c[3] * d2 + P.c[1] * d4;
    Q.c[6] = P.c[4] * d2 + P.c[2] * d4;
    Q.c[7] = P.c[3] * d4;
    Q.c[8] = P.c[4] * d4;
    qR_out.deg = Q.deg;
    for (int i = 0; i <= Q.deg; ++i)
        qR_out.c[i] = gm_series_derivative(Q.c[i]);
    return Q;
}

template <int Order, class Scalar>
inline GmConnectionJet<Order, Scalar> gm_connection_jet(
    double Rc, const GmParams<Scalar>& params) {
    using Result = GmConnectionJet<Order, Scalar>;
    using Series = GmSeries<Order, Scalar>;
    using Poly = GmPoly<Series, kGmPolyCapacity>;
    Result out;
    // Q_R is the h derivative.  One extra coefficient is required even for
    // a point connection (Order=0); differentiating an Order=0 series would
    // otherwise silently return zero and produce a zero connection matrix.
    using FullSeries = GmSeries<Order + 1, Scalar>;
    using FullPoly = GmPoly<FullSeries, kGmPolyCapacity>;
    FullPoly qfull, qrfull;
    qfull = gm_q_series_raw<Order + 1, Scalar>(Rc, params, qrfull);
    out.q.deg = qfull.deg;
    out.qR.deg = qrfull.deg;
    for (int i = 0; i <= qfull.deg; ++i) {
        for (int n = 0; n <= Order; ++n) {
            out.q.c[i].c[n] = qfull.c[i].c[n];
            out.qR.c[i].c[n] = qrfull.c[i].c[n];
        }
    }

    const double raw_q8 = gm_abs_value(out.q.c[kGmQDegree].c[0]);
    out.degree_ok = out.q.deg == kGmQDegree && raw_q8 > 0.0 &&
                    gm_series_finite(out.q.c[kGmQDegree]);
    if (!out.degree_ok) return out;

    // Work in a fixed balanced t-chart for the whole Taylor cell.  The
    // scale is computed from the centre only, hence it introduces no hidden
    // d/dR term.  If t=c*tau, Q(c*tau)=sum q_i c^i tau^i and the connection
    // transforms back as C_t[k,j] = C_tau[k,j] c^(k-j).  This is the same
    // local chart balance used by the Python numeric oracle, but applied to
    // the equation-derived series rather than to fitted samples.
    const double raw_q0 = gm_abs_value(out.q.c[0].c[0]);
    double chart_scale = 1.0;
    if (raw_q0 > 0.0 && raw_q8 > 0.0 && std::isfinite(raw_q0) &&
        std::isfinite(raw_q8)) {
        chart_scale = std::pow(raw_q0 / raw_q8, 0.125);
        if (!(chart_scale > 0.0) || !std::isfinite(chart_scale)) chart_scale = 1.0;
    }
    out.chart_scale = chart_scale;
    double chart_norm = 0.0;
    double cp = 1.0;
    for (int i = 0; i <= out.q.deg; ++i) {
        chart_norm = std::max(chart_norm,
                              gm_abs_value(out.q.c[i].c[0]) * std::fabs(cp));
        cp *= chart_scale;
    }
    if (!(chart_norm > 0.0) || !std::isfinite(chart_norm)) chart_norm = 1.0;
    // Apply only constant chart factors.  In particular, the normalization
    // is not a series inverse and carries no artificial h dependence.
    cp = 1.0;
    if constexpr (std::is_same_v<Scalar, GmDual5>) {
        const Series inv_norm{Scalar(1.0 / chart_norm)};
        for (int i = 0; i <= out.q.deg; ++i) {
            const Series factor{Scalar(cp)};
            out.q.c[i] = out.q.c[i] * factor * inv_norm;
            out.qR.c[i] = out.qR.c[i] * factor * inv_norm;
            cp *= chart_scale;
        }
    } else {
        for (int i = 0; i <= out.q.deg; ++i) {
            const Scalar factor = Scalar(cp / chart_norm);
            out.q.c[i] = gm_series_scale_scalar(out.q.c[i], factor);
            out.qR.c[i] = gm_series_scale_scalar(out.qR.c[i], factor);
            cp *= chart_scale;
        }
    }

    const Poly Qt = gm_poly_derivative(out.q);
    const Series inv_q8 = gm_series_inv(out.q.c[kGmQDegree]);
    std::array<std::array<Series, kGmQDegree>, kGmQDegree> M{};
    if constexpr (std::is_same_v<Scalar, GmDual5>) {
        for (int j = 0; j < kGmQDegree; ++j) {
            Poly basis;
            basis.deg = j;
            basis.c[j] = Series(Scalar(1));
            const Poly col = gm_poly_mod(gm_poly_mul(Qt, basis), out.q);
            for (int i = 0; i < kGmQDegree; ++i) M[i][j] = col.c[i];
        }
    } else {
        for (int j = 0; j < kGmQDegree; ++j) {
            // The j-th column is t^j Q_t mod Q.  Multiplication by the
            // monomial is just a shift; do not materialise a one-hot
            // polynomial and feed it through the generic product kernel.
            const Poly col = gm_poly_mod_with_lead_inv(
                gm_poly_shift(Qt, j), out.q, inv_q8);
            for (int i = 0; i < kGmQDegree; ++i) M[i][j] = col.c[i];
        }
    }

    std::array<std::array<Scalar, kGmQDegree>, kGmQDegree> M0{};
    for (int i = 0; i < kGmQDegree; ++i)
        for (int j = 0; j < kGmQDegree; ++j) M0[i][j] = M[i][j].c[0];
    const GmLU8<Scalar> lu = gm_lu_factor(M0);
    out.matrix_pivot_rel = lu.pivot_rel;
    out.squarefree = lu.ok;
    if (!out.squarefree) return out;

    // M(h) U(h) = e_0.  The LU factors of M_0 are reused at every Taylor
    // order; no new polynomial factorization or sample fit is introduced.
    std::array<Series, kGmQDegree> U{};
    for (int n = 0; n <= Order; ++n) {
        std::array<Scalar, kGmQDegree> rhs{};
        if (n == 0) rhs[0] = Scalar(1);
        else {
            for (int i = 0; i < kGmQDegree; ++i) {
                Scalar sum = Scalar(0);
                for (int j = 1; j <= n; ++j)
                    for (int col = 0; col < kGmQDegree; ++col)
                        sum = sum + M[i][col].c[j] * U[col].c[n - j];
                rhs[i] = -sum;
            }
        }
        const auto un = lu.solve(rhs);
        for (int col = 0; col < kGmQDegree; ++col) U[col].c[n] = un[col];
    }

    Poly Up;
    Up.deg = kGmQDegree - 1;
    for (int j = 0; j < kGmQDegree; ++j) Up.c[j] = U[j];
    double residual_scale = 1.0;
    double residual = 0.0;
    if constexpr (std::is_same_v<Scalar, GmDual5>) {
        // Keep the established dual path's operation grouping.  This lane
        // is used for parameter-Jacobian parity, where reassociation can
        // move a central-difference comparison by several ulps.
        // The original construction also retained degree 21 intermediates
        // in t^k Q_R U.  Use a wide local scratch polynomial for this
        // compatibility lane while keeping the value-only result buffers at
        // the fixed degree-16 capacity.
        using WidePoly = GmPoly<Series, 32>;
        auto widen = [](const Poly& p) {
            WidePoly r;
            r.deg = p.deg;
            for (int i = 0; i <= p.deg; ++i) r.c[i] = p.c[i];
            return r;
        };
        const WidePoly wide_q = widen(out.q);
        const WidePoly wide_qR = widen(out.qR);
        const WidePoly wide_Qt = widen(Qt);
        WidePoly wide_Up;
        wide_Up.deg = Up.deg;
        for (int i = 0; i <= Up.deg; ++i) wide_Up.c[i] = Up.c[i];
        const WidePoly twoQ = gm_poly_scale(wide_q, Series(Scalar(2)));
        const WidePoly Qt_again = wide_Qt;
        for (int k = 0; k < kGmEtaDim; ++k) {
            WidePoly tk;
            tk.deg = k;
            tk.c[k] = Series(Scalar(1));
            const WidePoly tkQR = gm_poly_mul(tk, wide_qR);
            const WidePoly Sk = gm_poly_mod(gm_poly_mul(tkQR, wide_Up), wide_q);
            const WidePoly num = gm_poly_sub(gm_poly_mul(Sk, Qt_again), tkQR);
            const WidePoly Tk = gm_poly_div_exact(num, twoQ);
            const WidePoly Ck = gm_poly_sub(Tk, gm_poly_derivative(Sk));
            out.S[k].deg = Sk.deg;
            for (int i = 0; i <= Sk.deg; ++i) out.S[k].c[i] = Sk.c[i];
            for (int j = 0; j < kGmEtaDim; ++j) {
                const double back = std::pow(chart_scale, double(k - j));
                out.C[k][j] = Ck.c[j] * Series(Scalar(back));
            }

            const WidePoly lhs = gm_poly_scale(tkQR, Series(Scalar(-0.5)));
            const WidePoly dSk = gm_poly_derivative(Sk);
            const WidePoly rhs = gm_poly_sub(
                gm_poly_add(gm_poly_mul(Ck, wide_q),
                            gm_poly_mul(dSk, wide_q)),
                gm_poly_scale(gm_poly_mul(Sk, Qt_again),
                              Series(Scalar(0.5))));
            const WidePoly diff = gm_poly_sub(lhs, rhs);
            for (int i = 0; i <= diff.deg; ++i) {
                residual_scale = std::max(residual_scale,
                                          gm_series_max_abs(out.q.c[i]));
                residual = std::max(residual, gm_series_max_abs(diff.c[i]));
            }
        }
    } else {
        for (int i = 0; i <= out.q.deg; ++i)
            residual_scale = std::max(residual_scale,
                                      gm_series_max_abs(out.q.c[i]));
        const Poly twoQ = gm_poly_scale_scalar(out.q, Scalar(2));
        const Series inv_two_q8 = gm_series_scale_scalar(inv_q8, Scalar(0.5));
        Poly Sk = gm_poly_mod_with_lead_inv(
            gm_poly_mul(out.qR, Up), out.q, inv_q8);

        for (int k = 0; k < kGmEtaDim; ++k) {
            // S_{k+1} = t S_k mod Q, so one product/modulo chain replaces
            // the repeated t^k Q_R * U construction for all seven eta rows.
            const Poly tkQR = gm_poly_shift(out.qR, k);
            const Poly num = gm_poly_sub(gm_poly_mul(Sk, Qt), tkQR);
            double div_residual = 0.0;
            const Poly Tk = gm_poly_div_exact_with_lead_inv(
                num, twoQ, inv_two_q8, &div_residual);
            const Poly Ck = gm_poly_sub(Tk, gm_poly_derivative(Sk));
            out.S[k] = Sk;
            for (int j = 0; j < kGmEtaDim; ++j) {
                const double back = std::pow(chart_scale, double(k - j));
                out.C[k][j] = gm_series_scale_scalar(Ck.c[j], Scalar(back));
            }

            // Since Ck = Tk - S'_k and 2 Q Tk = S_k Q_t - t^k Q_R by
            // construction, the long-division remainder is the residual of
            // the full differential-form identity.  Avoid rebuilding its
            // three degree-14 products solely to measure the same quantity.
            residual = std::max(residual, div_residual);
            if (k + 1 < kGmEtaDim)
                Sk = gm_poly_mod_with_lead_inv(gm_poly_shift(Sk, 1), out.q,
                                               inv_q8);
        }
    }

    out.identity_residual = residual / residual_scale;
    out.finite = true;
    for (int i = 0; i <= out.q.deg; ++i)
        out.finite = out.finite && gm_series_finite(out.q.c[i]) &&
                     gm_series_finite(out.qR.c[i]);
    for (const auto& row : out.C)
        for (const Series& x : row) out.finite = out.finite && gm_series_finite(x);
    out.quality_ok = out.finite && std::isfinite(out.identity_residual) &&
                     out.identity_residual <= gm_quality_limit(Scalar(0));
    out.ok = out.quality_ok;
    return out;
}

struct GmConnectionPoint {
    bool ok = false;
    bool degree_ok = false;
    bool squarefree = false;
    double identity_residual = std::numeric_limits<double>::infinity();
    double matrix_pivot_rel = 0.0;
    std::array<std::array<double, kGmEtaDim>, kGmEtaDim> C{};
};

inline GmConnectionPoint gm_connection_at(double R, const PrimaryFrame& pf) {
    const GmParams<double> p{pf.X, pf.Y, pf.rho, pf.m0, pf.a};
    const auto jet = gm_connection_jet<0, double>(R, p);
    GmConnectionPoint out;
    out.ok = jet.ok;
    out.degree_ok = jet.degree_ok;
    out.squarefree = jet.squarefree;
    out.identity_residual = jet.identity_residual;
    out.matrix_pivot_rel = jet.matrix_pivot_rel;
    for (int i = 0; i < kGmEtaDim; ++i)
        for (int j = 0; j < kGmEtaDim; ++j) out.C[i][j] = jet.C[i][j].c[0];
    return out;
}

template <int Order>
inline GmConnectionJet<Order, GmDual5> gm_connection_jet_dual(
    double R, const PrimaryFrame& pf) {
    const GmParams<GmDual5> p{
        GmDual5::variable(0, pf.X), GmDual5::variable(1, pf.Y),
        GmDual5::variable(2, pf.rho), GmDual5::variable(3, pf.m0),
        GmDual5::variable(4, pf.a)};
    return gm_connection_jet<Order, GmDual5>(R, p);
}

}  // namespace lcbinint::holonomic

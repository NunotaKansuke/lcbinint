#pragma once

// Research-only physical Lauricella FD6 -> FD5 transport.
//
// The boundary period is first written with the six quadratic factors
// S(t), 1+t^2, and B(t;R).  Since the six Euler exponents sum to gamma=3,
// the factor rooted at t=+i can be removed by the Euler Mobius transform.
// The resulting state is
//
//   Z = (F, M_1, ..., M_5),
//   M_i = F + (xi_i-1)/beta_i * dF/dxi_i,
//
// and is transported by the logarithmic Lauricella Pfaffian system.  This
// header deliberately does not call the 7D eta connection, a 15x15 Hermite
// reduction, a scalar Picard--Fuchs expansion, or a sample-fit packet.
//
// It is an isolated diagnostic kernel.  The V2/V3 production router remains
// unchanged until the whole-epoch gates in the accompanying harness pass.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/gm_connection.hpp"
#include "lcbinint/magnification/holonomic/angular_rule.hpp"
#include "lcbinint/magnification/holonomic/phi.hpp"
#include "lcbinint/magnification/holonomic/status.hpp"

namespace lcbinint::holonomic {

namespace gm_l6_detail {

constexpr int kStateDim = 6;
constexpr int kVariableDim = 5;
constexpr std::array<double, kVariableDim> kBeta =
    {{-0.5, -0.5, 1.5, 0.5, 0.5}};
constexpr double kAlpha = 1.5;
constexpr double kGamma = 3.0;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

template <class T>
struct Complex {
    T re{};
    T im{};

    Complex() = default;
    Complex(const T& r, const T& i) : re(r), im(i) {}
};

template <class T>
inline Complex<T> operator+(const Complex<T>& a, const Complex<T>& b) {
    return {a.re + b.re, a.im + b.im};
}
template <class T>
inline Complex<T> operator-(const Complex<T>& a, const Complex<T>& b) {
    return {a.re - b.re, a.im - b.im};
}
template <class T>
inline Complex<T> operator-(const Complex<T>& a) {
    return {-a.re, -a.im};
}
template <class T>
inline Complex<T> operator*(const Complex<T>& a, const Complex<T>& b) {
    return {a.re * b.re - a.im * b.im,
            a.re * b.im + a.im * b.re};
}
template <class T>
inline Complex<T> operator/(const Complex<T>& a, const Complex<T>& b) {
    const T d = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / d,
            (a.im * b.re - a.re * b.im) / d};
}
template <class T>
inline Complex<T> cscale(const Complex<T>& a, const T& x) {
    return {a.re * x, a.im * x};
}
template <class T>
inline Complex<T> cconj(const Complex<T>& a) {
    return {a.re, -a.im};
}

template <class T>
inline T scalar_value(const T& x) {
    return x;
}
inline double scalar_value(const double& x) { return x; }
inline double scalar_value(const long double& x) { return (double)x; }

// The parameter dual is intentionally separate from GmDual5.  It is used
// only for the algebraic geometry chain rule after the six-state value has
// already been transported.
struct Dual5 {
    double value = 0.0;
    std::array<double, 5> deriv{};

    Dual5() = default;
    Dual5(double x) : value(x) {}

    static Dual5 variable(int j, double x) {
        Dual5 r(x);
        if (j >= 0 && j < 5) r.deriv[j] = 1.0;
        return r;
    }
};

inline Dual5 operator+(Dual5 a, const Dual5& b) {
    a.value += b.value;
    for (int j = 0; j < 5; ++j) a.deriv[j] += b.deriv[j];
    return a;
}
inline Dual5& operator+=(Dual5& a, const Dual5& b) {
    a.value += b.value;
    for (int j = 0; j < 5; ++j) a.deriv[j] += b.deriv[j];
    return a;
}
inline Dual5 operator-(Dual5 a, const Dual5& b) {
    a.value -= b.value;
    for (int j = 0; j < 5; ++j) a.deriv[j] -= b.deriv[j];
    return a;
}
inline Dual5 operator-(Dual5 a) {
    a.value = -a.value;
    for (double& x : a.deriv) x = -x;
    return a;
}
inline Dual5 operator*(const Dual5& a, const Dual5& b) {
    Dual5 r(a.value * b.value);
    for (int j = 0; j < 5; ++j)
        r.deriv[j] = a.deriv[j] * b.value + a.value * b.deriv[j];
    return r;
}
inline Dual5 operator/(const Dual5& a, const Dual5& b) {
    const double inv = 1.0 / b.value;
    Dual5 r(a.value * inv);
    const double inv2 = inv * inv;
    for (int j = 0; j < 5; ++j)
        r.deriv[j] = (a.deriv[j] * b.value - a.value * b.deriv[j]) * inv2;
    return r;
}
inline double scalar_value(const Dual5& x) { return x.value; }

template <class T>
inline bool scalar_finite(const T& x) {
    using std::isfinite;
    return isfinite(x);
}
inline bool scalar_finite(const Dual5& x) {
    if (!std::isfinite(x.value)) return false;
    for (double d : x.deriv)
        if (!std::isfinite(d)) return false;
    return true;
}

inline double scalar_sqrt(double x) { return std::sqrt(x); }
inline long double scalar_sqrt(long double x) { return std::sqrt(x); }
inline Dual5 scalar_sqrt(const Dual5& x) {
    if (!(x.value > 0.0)) return Dual5(0.0);
    const double s = std::sqrt(x.value);
    Dual5 r(s);
    for (int j = 0; j < 5; ++j) r.deriv[j] = x.deriv[j] / (2.0 * s);
    return r;
}

template <class T>
inline Complex<T> complex_sqrt_scalar(const Complex<T>& x);

template <>
inline Complex<double> complex_sqrt_scalar(const Complex<double>& x) {
    const std::complex<double> z(x.re, x.im);
    const auto s = std::sqrt(z);
    return {s.real(), s.imag()};
}
template <>
inline Complex<long double> complex_sqrt_scalar(
    const Complex<long double>& x) {
    const std::complex<long double> z(x.re, x.im);
    const auto s = std::sqrt(z);
    return {s.real(), s.imag()};
}

inline Complex<Dual5> complex_sqrt_scalar(const Complex<Dual5>& x) {
    const std::complex<double> z(x.re.value, x.im.value);
    const auto s = std::sqrt(z);
    Complex<Dual5> r{Dual5(s.real()), Dual5(s.imag())};
    const std::complex<double> den = 2.0 * s;
    if (std::abs(den) == 0.0) return r;
    for (int j = 0; j < 5; ++j) {
        const std::complex<double> dx(x.re.deriv[j], x.im.deriv[j]);
        const std::complex<double> dy = dx / den;
        r.re.deriv[j] = dy.real();
        r.im.deriv[j] = dy.imag();
    }
    return r;
}

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> series_sqrt(
    const GmSeries<Order, Scalar>& x) {
    GmSeries<Order, Scalar> r;
    r.c[0] = scalar_sqrt(x.c[0]);
    if (!scalar_finite(r.c[0]) || scalar_value(r.c[0]) == 0.0) return r;
    for (int n = 1; n <= Order; ++n) {
        Scalar sum = Scalar(0);
        for (int k = 1; k < n; ++k) sum = sum + r.c[k] * r.c[n - k];
        r.c[n] = (x.c[n] - sum) / (Scalar(2) * r.c[0]);
    }
    return r;
}

template <int Order, class Scalar>
inline Complex<GmSeries<Order, Scalar>> complex_sqrt_series_clean(
    const Complex<GmSeries<Order, Scalar>>& x) {
    using Series = GmSeries<Order, Scalar>;
    using C = Complex<Series>;
    using CS = Complex<Scalar>;
    const CS y0 = complex_sqrt_scalar(CS{x.re.c[0], x.im.c[0]});
    C r;
    r.re.c[0] = y0.re;
    r.im.c[0] = y0.im;
    for (int n = 1; n <= Order; ++n) {
        CS known{};
        for (int k = 1; k < n; ++k) {
            known.re = known.re + r.re.c[k] * r.re.c[n - k]
                       - r.im.c[k] * r.im.c[n - k];
            known.im = known.im + r.re.c[k] * r.im.c[n - k]
                       + r.im.c[k] * r.re.c[n - k];
        }
        const CS rhs{x.re.c[n] - known.re, x.im.c[n] - known.im};
        const CS den{Scalar(2) * y0.re, Scalar(2) * y0.im};
        const CS q = rhs / den;
        r.re.c[n] = q.re;
        r.im.c[n] = q.im;
    }
    return r;
}

// Coefficient recurrence for division in the double series lane.  The
// generic GmSeries division forms an inverse series and then multiplies a
// second full series.  Geometry has several fixed complex quotients per
// packet, so the direct recurrence removes both the duplicate convolution
// and the temporary inverse.
template <int Order>
inline GmSeries<Order, double> series_div_coeff(
    const GmSeries<Order, double>& a, const GmSeries<Order, double>& b) {
    GmSeries<Order, double> q;
    q.c[0] = a.c[0] / b.c[0];
    for (int n = 1; n <= Order; ++n) {
        double rhs = a.c[n];
        for (int k = 1; k <= n; ++k) rhs -= b.c[k] * q.c[n - k];
        q.c[n] = rhs / b.c[0];
    }
    return q;
}

template <int Order>
inline Complex<GmSeries<Order, double>> complex_series_div_coeff(
    const Complex<GmSeries<Order, double>>& a,
    const Complex<GmSeries<Order, double>>& b) {
    Complex<GmSeries<Order, double>> q;
    const Complex<double> b0{b.re.c[0], b.im.c[0]};
    std::array<Complex<double>, Order + 1> qc{};
    for (int n = 0; n <= Order; ++n) {
        Complex<double> rhs{a.re.c[n], a.im.c[n]};
        for (int k = 1; k <= n; ++k) {
            const Complex<double> bk{b.re.c[k], b.im.c[k]};
            rhs = rhs - bk * qc[n - k];
        }
        qc[n] = rhs / b0;
        q.re.c[n] = qc[n].re;
        q.im.c[n] = qc[n].im;
    }
    return q;
}

template <class T>
inline Complex<T> complex_pow(const Complex<T>& x, double exponent);

template <>
inline Complex<double> complex_pow(const Complex<double>& x, double exponent) {
    const std::complex<double> z(x.re, x.im);
    const auto r = std::pow(z, exponent);
    return {r.real(), r.imag()};
}
template <>
inline Complex<long double> complex_pow(const Complex<long double>& x,
                                        double exponent) {
    const std::complex<long double> z(x.re, x.im);
    const auto r = std::pow(z, (long double)exponent);
    return {r.real(), r.imag()};
}
inline Complex<Dual5> complex_pow(const Complex<Dual5>& x, double exponent) {
    const std::complex<double> z(x.re.value, x.im.value);
    const auto p = std::pow(z, exponent);
    Complex<Dual5> r{Dual5(p.real()), Dual5(p.imag())};
    if (std::abs(z) == 0.0) return r;
    const std::complex<double> d = exponent * p / z;
    for (int j = 0; j < 5; ++j) {
        const std::complex<double> dx(x.re.deriv[j], x.im.deriv[j]);
        const auto dy = d * dx;
        r.re.deriv[j] = dy.real();
        r.im.deriv[j] = dy.imag();
    }
    return r;
}

template <int Order, class Scalar>
inline bool series_finite(const GmSeries<Order, Scalar>& x) {
    for (const Scalar& c : x.c)
        if (!scalar_finite(c)) return false;
    return true;
}
template <int Order, class Scalar>
inline bool complex_series_finite(
    const Complex<GmSeries<Order, Scalar>>& x) {
    return series_finite(x.re) && series_finite(x.im);
}

template <class S>
struct Params {
    S a, m0, X, Y, rho;
};

template <class S>
inline std::array<S, 5> boundary_P(const S& R, const Params<S>& p) {
    using C = Complex<S>;
    const S R2 = R * R;
    const C zeta{p.X, p.Y};
    const C n0{-p.X * R2, -p.Y * R2};
    const C n1 = cscale(zeta, p.a * R) + C{R * (R2 - S(1)), S(0)};
    const C n2{p.a * (p.m0 - R2), S(0)};
    const C c0 = n0 + n1 + n2;
    const C d = n2 - n0;
    const C c1{-S(2) * d.im, S(2) * d.re};
    const C c2 = n1 - n0 - n2;
    const S k = p.rho * p.rho * R2;
    const S bm = (R - p.a) * (R - p.a);
    const S bp = (R + p.a) * (R + p.a);
    return {k * bm - (c0.re * c0.re + c0.im * c0.im),
            -S(2) * (c0.re * c1.re + c0.im * c1.im),
            k * (bm + bp) - (c1.re * c1.re + c1.im * c1.im)
                - S(2) * (c0.re * c2.re + c0.im * c2.im),
            -S(2) * (c1.re * c2.re + c1.im * c2.im),
            k * bp - (c2.re * c2.re + c2.im * c2.im)};
}

template <int Order>
inline GmSeries<Order, double> l6_scale_series(
    const GmSeries<Order, double>& a, double x) {
    return gm_series_scale_scalar(a, x);
}

// Multiplication by R=R0+h in the packet jet.  The radius series is always
// affine, so a general O(Order^2) convolution is unnecessary here.
template <int Order>
inline GmSeries<Order, double> l6_radius_mul(
    const GmSeries<Order, double>& a, double R0) {
    GmSeries<Order, double> out;
    out.c[0] = R0 * a.c[0];
    for (int n = 1; n <= Order; ++n)
        out.c[n] = R0 * a.c[n] + a.c[n - 1];
    return out;
}

template <int Order>
inline GmSeries<Order, double> l6_linear_square(double x0) {
    GmSeries<Order, double> x(x0);
    if constexpr (Order >= 1) x.c[1] = 1.0;
    return l6_radius_mul(x, x0);
}

template <int DA, int DB, int Order, class Scalar>
inline GmSeries<Order, Scalar> l6_mul_low(
    const GmSeries<Order, Scalar>& a, const GmSeries<Order, Scalar>& b) {
    GmSeries<Order, Scalar> out;
    const int degree = std::min(Order, DA + DB);
    for (int n = 0; n <= degree; ++n) {
        const int first = std::max(0, n - DB);
        const int last = std::min(DA, n);
        for (int k = first; k <= last; ++k)
            out.c[n] += a.c[k] * b.c[n - k];
    }
    return out;
}

template <int DB, int Order, class Scalar>
inline GmSeries<Order, Scalar> l6_mul_low_right(
    const GmSeries<Order, Scalar>& full,
    const GmSeries<Order, Scalar>& low) {
    GmSeries<Order, Scalar> out;
    for (int n = 0; n <= Order; ++n) {
        const int first = std::max(0, n - DB);
        for (int k = first; k <= n; ++k)
            out.c[n] += full.c[k] * low.c[n - k];
    }
    return out;
}

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> l6_mul_affine(
    const GmSeries<Order, Scalar>& full, const Scalar& b0,
    const Scalar& b1) {
    GmSeries<Order, Scalar> out;
    out.c[0] = full.c[0] * b0;
    for (int n = 1; n <= Order; ++n)
        out.c[n] = full.c[n] * b0 + full.c[n - 1] * b1;
    return out;
}

template <int Order>
inline std::array<GmSeries<Order, double>, 5> boundary_P_radius_series(
    double R0, const Params<GmSeries<Order, double>>& p) {
    using S = GmSeries<Order, double>;
    using C = Complex<S>;
    S R(R0);
    if constexpr (Order >= 1) R.c[1] = 1.0;
    const S R2 = l6_radius_mul(R, R0);
    const double a = p.a.c[0];
    const double m0 = p.m0.c[0];
    const double X = p.X.c[0];
    const double Y = p.Y.c[0];
    const double rho = p.rho.c[0];
    const C zeta{S(X), S(Y)};
    const C n0{l6_scale_series(R2, -X), l6_scale_series(R2, -Y)};
    const S aR = l6_scale_series(R, a);
    const C n1{l6_scale_series(aR, X) +
                   l6_radius_mul(R2 - S(1), R0),
               l6_scale_series(aR, Y)};
    const C n2{l6_scale_series(S(m0) - R2, a), S(0)};
    const C c0 = n0 + n1 + n2;
    const C d = n2 - n0;
    const C c1{l6_scale_series(-d.im, 2.0),
               l6_scale_series(d.re, 2.0)};
    const C c2 = n1 - n0 - n2;
    const S rho2 = S(rho * rho);
    const S k = l6_scale_series(R2, rho2.c[0]);
    const S bm = l6_linear_square<Order>(R0 - a);
    const S bp = l6_linear_square<Order>(R0 + a);
    const S c0norm = l6_mul_low<3, 3>(c0.re, c0.re) +
                     l6_mul_low<3, 3>(c0.im, c0.im);
    const S c1norm = l6_mul_low<2, 2>(c1.re, c1.re) +
                     l6_mul_low<2, 2>(c1.im, c1.im);
    const S c2norm = l6_mul_low<3, 3>(c2.re, c2.re) +
                     l6_mul_low<3, 3>(c2.im, c2.im);
    const S c0c1 = l6_mul_low<3, 2>(c0.re, c1.re) +
                  l6_mul_low<3, 2>(c0.im, c1.im);
    const S c0c2 = l6_mul_low<3, 3>(c0.re, c2.re) +
                  l6_mul_low<3, 3>(c0.im, c2.im);
    const S c1c2 = l6_mul_low<2, 3>(c1.re, c2.re) +
                  l6_mul_low<2, 3>(c1.im, c2.im);
    return {l6_mul_low<2, 2>(k, bm) - c0norm,
            l6_scale_series(-c0c1, 2.0),
            l6_mul_low<2, 2>(k, bm + bp) - c1norm -
                l6_scale_series(c0c2, 2.0),
            l6_scale_series(-c1c2, 2.0),
            l6_mul_low<2, 2>(k, bp) - c2norm};
}

// Same fixed-degree boundary polynomial for the Dual5 lane.  The old generic
// path called boundary_P with full series products and divisions even though
// R is affine and every parameter is constant in the packet variable.  Keep
// the value expression and the low-degree products identical to the double
// implementation, while letting each coefficient carry its five analytic
// derivatives.
template <int Order, class Scalar>
inline std::array<GmSeries<Order, Scalar>, 5>
boundary_P_radius_series_dual(
    double R0, const Params<GmSeries<Order, Scalar>>& p) {
    using S = GmSeries<Order, Scalar>;
    using C = Complex<S>;
    S R(R0);
    if constexpr (Order >= 1) R.c[1] = Scalar(1);
    const S R2 = l6_mul_affine(R, Scalar(R0), Scalar(1));
    const Scalar a = p.a.c[0];
    const Scalar m0 = p.m0.c[0];
    const Scalar X = p.X.c[0];
    const Scalar Y = p.Y.c[0];
    const Scalar rho = p.rho.c[0];
    const C n0{gm_series_scale_scalar(R2, -X),
               gm_series_scale_scalar(R2, -Y)};
    const S aR = gm_series_scale_scalar(R, a);
    const C n1{
        gm_series_scale_scalar(aR, X) +
            l6_mul_affine(R2 - S(Scalar(1)), Scalar(R0), Scalar(1)),
        gm_series_scale_scalar(aR, Y)};
    const C n2{gm_series_scale_scalar(S(m0) - R2, a), S(Scalar(0))};
    const C c0 = n0 + n1 + n2;
    const C d = n2 - n0;
    const C c1{gm_series_scale_scalar(-d.im, Scalar(2)),
               gm_series_scale_scalar(d.re, Scalar(2))};
    const C c2 = n1 - n0 - n2;
    const S rho2 = S(rho * rho);
    const S k = gm_series_scale_scalar(R2, rho2.c[0]);
    S xm(R0);
    S xp(R0);
    xm.c[0] = Scalar(R0) - a;
    xp.c[0] = Scalar(R0) + a;
    if constexpr (Order >= 1) {
        xm.c[1] = Scalar(1);
        xp.c[1] = Scalar(1);
    }
    const S bm = l6_mul_affine(xm, Scalar(R0) - a, Scalar(1));
    const S bp = l6_mul_affine(xp, Scalar(R0) + a, Scalar(1));
    const S c0norm = l6_mul_low<3, 3>(c0.re, c0.re) +
                     l6_mul_low<3, 3>(c0.im, c0.im);
    const S c1norm = l6_mul_low<2, 2>(c1.re, c1.re) +
                     l6_mul_low<2, 2>(c1.im, c1.im);
    const S c2norm = l6_mul_low<3, 3>(c2.re, c2.re) +
                     l6_mul_low<3, 3>(c2.im, c2.im);
    const S c0c1 = l6_mul_low<3, 2>(c0.re, c1.re) +
                  l6_mul_low<3, 2>(c0.im, c1.im);
    const S c0c2 = l6_mul_low<3, 3>(c0.re, c2.re) +
                  l6_mul_low<3, 3>(c0.im, c2.im);
    const S c1c2 = l6_mul_low<2, 3>(c1.re, c2.re) +
                  l6_mul_low<2, 3>(c1.im, c2.im);
    return {l6_mul_low<2, 2>(k, bm) - c0norm,
            gm_series_scale_scalar(-c0c1, Scalar(2)),
            l6_mul_low<2, 2>(k, bm + bp) - c1norm -
                gm_series_scale_scalar(c0c2, Scalar(2)),
            gm_series_scale_scalar(-c1c2, Scalar(2)),
            l6_mul_low<2, 2>(k, bp) - c2norm};
}

template <class S>
inline Params<S> params_for_chart(const PrimaryFrame& pf, bool chart2) {
    const double sign = chart2 ? -1.0 : 1.0;
    return {S(sign * pf.a), S(pf.m0), S(sign * pf.X), S(sign * pf.Y),
            S(pf.rho)};
}

inline Dual5 dual_variable_signed(int j, double x, double sign) {
    Dual5 r = Dual5::variable(j, x);
    for (double& d : r.deriv) d *= sign;
    return r;
}

inline Params<Dual5> dual_params_for_chart(const PrimaryFrame& pf,
                                            bool chart2) {
    const double sign = chart2 ? -1.0 : 1.0;
    return {dual_variable_signed(4, sign * pf.a, sign),
            Dual5::variable(3, pf.m0),
            dual_variable_signed(0, sign * pf.X, sign),
            dual_variable_signed(1, sign * pf.Y, sign),
            Dual5::variable(2, pf.rho)};
}

template <class S>
struct Geometry {
    bool ok = false;
    S v{};
    S S_left{};
    Complex<S> q{};
    Complex<S> C6{};
    Complex<S> C{};
    std::array<Complex<S>, 6> z{};
    std::array<Complex<S>, 5> xi{};
};

template <class S>
inline double cabs_value(const Complex<S>& x) {
    const double re = scalar_value(x.re);
    const double im = scalar_value(x.im);
    return std::hypot(re, im);
}

template <class S>
inline bool cfinite(const Complex<S>& x) {
    return scalar_finite(x.re) && scalar_finite(x.im);
}

template <class S>
inline Geometry<S> geometry_scalar(const S& R, const S& l, const S& r,
                                  const Params<S>& p) {
    Geometry<S> out;
    const auto P = boundary_P(R, p);
    const S delta = r - l;
    const S half = delta / S(2);
    out.v = half * half;
    const S beta2 = -(l + r);
    const S gamma = l * r;
    const S d1 = P[3] - beta2 * P[4];
    const S d0 = P[2] - beta2 * d1 - gamma * P[4];
    out.S_left = -(d0 + l * (d1 + l * P[4]));
    const S Aleft = S(1) + l * l;
    const S bm = (R - p.a) * (R - p.a);
    const S bp = (R + p.a) * (R + p.a);
    const S Bleft = bm + bp * l * l;
    if (!scalar_finite(out.S_left) || !scalar_finite(Aleft) ||
        !scalar_finite(Bleft) || !(scalar_value(out.S_left) > 0.0) ||
        !(scalar_value(Aleft) > 0.0) || !(scalar_value(Bleft) > 0.0) ||
        !(scalar_value(p.rho) > 0.0))
        return out;

    const S S1left = -(d1 + S(2) * P[4] * l);
    const S L = delta * S1left / out.S_left;
    const S M = delta * delta * (-P[4]) / out.S_left;
    const Complex<S> discr{L * L - S(4) * M, S(0)};
    const Complex<S> sq = complex_sqrt_scalar(discr);
    const Complex<S> zS1{(-L + sq.re) / S(2), sq.im / S(2)};
    const Complex<S> zS2{(-L - sq.re) / S(2), -sq.im / S(2)};
    const Complex<S> dz{delta, S(0)};
    const Complex<S> zi{ -l, S(1)};
    const Complex<S> zmi{-l, S(-1)};
    const Complex<S> zstar = dz / zi;
    const Complex<S> zaminus = dz / zmi;
    const Complex<S> bnum{delta * (R + p.a), S(0)};
    const Complex<S> zbplus{-l * (R + p.a), R - p.a};
    const Complex<S> zbminus{-l * (R + p.a), -(R - p.a)};
    out.z = {zS1, zS2, zstar, zaminus, bnum / zbplus, bnum / zbminus};
    out.q = Complex<S>{S(1) - zstar.re, -zstar.im};
    const Complex<S> one_minus_zstar{S(1) - zstar.re, -zstar.im};
    for (int j = 0; j < 5; ++j)
        out.xi[j] = (out.z[j < 2 ? j : j + 1] - zstar) /
                    one_minus_zstar;

    const S sqrtS = scalar_sqrt(out.S_left);
    const S sqrtA = scalar_sqrt(Aleft);
    const S sqrtB = scalar_sqrt(Bleft);
    out.C6 = {S(kPi) * out.v / p.rho * sqrtS / (Aleft * sqrtA * sqrtB),
              S(0)};
    const Complex<S> qpow = complex_pow(out.q, -1.5);
    out.C = out.C6 * qpow;
    out.ok = cfinite(out.q) && cfinite(out.C) && cabs_value(out.q) > 1e-14;
    for (const auto& x : out.xi)
        out.ok = out.ok && cfinite(x) && cabs_value(x) < 1e14;
    return out;
}

template <int Order, class Scalar>
struct SeriesGeometry {
    bool ok = false;
    GmSeries<Order, Scalar> v{};
    GmSeries<Order, Scalar> S_left{};
    Complex<GmSeries<Order, Scalar>> q{};
    std::array<Complex<GmSeries<Order, Scalar>>, 6> z{};
    std::array<Complex<GmSeries<Order, Scalar>>, 5> xi{};
};

template <int Order, class Scalar>
inline SeriesGeometry<Order, Scalar> geometry_series(
    const GmSeries<Order, Scalar>& R, const GmSeries<Order, Scalar>& l,
    const GmSeries<Order, Scalar>& r,
    const Params<GmSeries<Order, Scalar>>& p,
    const std::array<GmSeries<Order, Scalar>, 5>& P) {
    using S = GmSeries<Order, Scalar>;
    using C = Complex<S>;
    SeriesGeometry<Order, Scalar> out;
    const S delta = r - l;
    const S half = gm_series_scale_scalar(delta, Scalar(0.5));
    out.v = half * half;
    const S beta2 = -(l + r);
    const S gamma = l * r;
    const S d1 = P[3] - beta2 * P[4];
    const S d0 = P[2] - beta2 * d1 - gamma * P[4];
    out.S_left = -(d0 + l * (d1 + l * P[4]));
    const S Aleft = S(1) + l * l;
    const S bm = (R - p.a) * (R - p.a);
    const S bp = (R + p.a) * (R + p.a);
    const S Bleft = bm + bp * l * l;
    if (!(out.S_left.c[0] > 0.0) || !(Aleft.c[0] > 0.0) ||
        !(Bleft.c[0] > 0.0) || !(p.rho.c[0] > 0.0)) return out;
    const S S1left = -(d1 +
                       l6_mul_low_right<6>(l,
                                           gm_series_scale_scalar(P[4],
                                                                  Scalar(2))));
    const S L = [&] {
        if constexpr (std::is_same_v<Scalar, double>)
            return series_div_coeff(delta * S1left, out.S_left);
        else
            return delta * S1left / out.S_left;
    }();
    const S M = [&] {
        if constexpr (std::is_same_v<Scalar, double>)
            return series_div_coeff(
                l6_mul_low_right<6>(delta * delta, -P[4]), out.S_left);
        else
            return delta * delta * (-P[4]) / out.S_left;
    }();
    const C discr{L * L - gm_series_scale_scalar(M, Scalar(4)), S(0)};
    const C sq = complex_sqrt_series_clean(discr);
    const C zS1{gm_series_scale_scalar(-L + sq.re, Scalar(0.5)),
                gm_series_scale_scalar(sq.im, Scalar(0.5))};
    const C zS2{gm_series_scale_scalar(-L - sq.re, Scalar(0.5)),
                gm_series_scale_scalar(-sq.im, Scalar(0.5))};
    const C dz{delta, S(0)};
    const C zstar = [&] {
        const C den{-l, S(1)};
        if constexpr (std::is_same_v<Scalar, double>)
            return complex_series_div_coeff(dz, den);
        else
            return dz / den;
    }();
    const C zaminus = [&] {
        const C den{-l, S(-1)};
        if constexpr (std::is_same_v<Scalar, double>)
            return complex_series_div_coeff(dz, den);
        else
            return dz / den;
    }();
    const C bnum{l6_mul_affine(delta, R.c[0] + p.a.c[0], 1.0), S(0)};
    const C zbp = [&] {
        const C den{-l * (R + p.a), R - p.a};
        if constexpr (std::is_same_v<Scalar, double>)
            return complex_series_div_coeff(bnum, den);
        else
            return bnum / den;
    }();
    const C zbm = [&] {
        const C den{-l * (R + p.a), -(R - p.a)};
        if constexpr (std::is_same_v<Scalar, double>)
            return complex_series_div_coeff(bnum, den);
        else
            return bnum / den;
    }();
    out.z = {zS1, zS2, zstar, zaminus, zbp, zbm};
    out.q = {S(1) - zstar.re, -zstar.im};
    const C den{S(1) - zstar.re, -zstar.im};
    for (int j = 0; j < 5; ++j) {
        const C numerator = out.z[j < 2 ? j : j + 1] - zstar;
        if constexpr (std::is_same_v<Scalar, double>)
            out.xi[j] = complex_series_div_coeff(numerator, den);
        else
            out.xi[j] = numerator / den;
    }
    out.ok = series_finite(out.v) && series_finite(out.S_left) &&
             complex_series_finite(out.q);
    for (const auto& x : out.xi) out.ok = out.ok && complex_series_finite(x);
    return out;
}

// The physical-fold lane works with the symmetric root-pair variables
// t_\pm=m +/- sqrt(v).  The two equations below are the even and odd parts of
// P(m+x) at x=sqrt(v).  Their Jacobian with respect to (m,v) remains nonzero
// at an ordinary double root, so m(R) and v(R) have ordinary Taylor series
// even though sqrt(v) itself does not.
template <int Order, class Scalar>
inline std::array<GmSeries<Order, Scalar>, 2> fold_pair_equations(
    const std::array<GmSeries<Order, Scalar>, 5>& p,
    const GmSeries<Order, Scalar>& m, const GmSeries<Order, Scalar>& v) {
    using S = GmSeries<Order, Scalar>;
    const S m2 = m * m;
    const S m3 = m2 * m;
    const S m4 = m2 * m2;
    const S mv = m * v;
    const S m2v = m2 * v;
    const S v2 = v * v;
    const S e = p[0] + p[1] * m + p[2] * (m2 + v) +
                p[3] * (m3 + gm_series_scale_scalar(mv, Scalar(3))) +
                p[4] * (m4 + gm_series_scale_scalar(m2v, Scalar(6)) + v2);
    const S o = p[1] + gm_series_scale_scalar(p[2] * m, Scalar(2)) +
                p[3] * (gm_series_scale_scalar(m2, Scalar(3)) + v) +
                p[4] * (gm_series_scale_scalar(m3, Scalar(4)) +
                         gm_series_scale_scalar(mv, Scalar(4)));
    return {e, o};
}

inline void fold_pair_equations_value(const std::array<double, 5>& p,
                                      double m, double v, double& e,
                                      double& o) {
    const double m2 = m * m;
    const double m3 = m2 * m;
    const double m4 = m2 * m2;
    e = p[0] + p[1] * m + p[2] * (m2 + v) +
        p[3] * (m3 + 3.0 * m * v) +
        p[4] * (m4 + 6.0 * m2 * v + v * v);
    o = p[1] + 2.0 * p[2] * m + p[3] * (3.0 * m2 + v) +
        p[4] * (4.0 * m3 + 4.0 * m * v);
}

inline bool fold_pair_refine(const std::array<double, 5>& p, double& m,
                             double& v) {
    if (!(v > 0.0) || !std::isfinite(m) || !std::isfinite(v)) return false;
    for (int iter = 0; iter < 4; ++iter) {
        double e = 0.0, o = 0.0;
        fold_pair_equations_value(p, m, v, e, o);
        const double m2 = m * m;
        const double em = p[1] + 2.0 * p[2] * m +
                          p[3] * (3.0 * m2 + 3.0 * v) +
                          p[4] * (4.0 * m2 * m + 12.0 * m * v);
        const double ev = p[2] + 3.0 * p[3] * m +
                          p[4] * (6.0 * m2 + 2.0 * v);
        const double om = 2.0 * p[2] + 6.0 * p[3] * m +
                          p[4] * (12.0 * m2 + 4.0 * v);
        const double ov = p[3] + 4.0 * p[4] * m;
        const double det = em * ov - ev * om;
        if (!std::isfinite(det) || std::fabs(det) < 1e-30) return false;
        const double dm = (-e * ov + ev * o) / det;
        const double dv = (om * e - em * o) / det;
        if (!std::isfinite(dm) || !std::isfinite(dv)) return false;
        m += dm;
        v += dv;
        if (!(v > 0.0) || !std::isfinite(m) || !std::isfinite(v))
            return false;
        if (std::max(std::fabs(dm), std::fabs(dv)) <=
            2e-14 * (1.0 + std::max(std::fabs(m), std::fabs(v))))
            break;
    }
    const double s = std::sqrt(v);
    const double tl = m - s, th = m + s;
    double et = 0.0, ot = 0.0;
    fold_pair_equations_value(p, m, v, et, ot);
    double scale = 1.0;
    for (int k = 0; k < 5; ++k)
        scale += std::fabs(p[k]) * std::pow(1.0 + std::max(std::fabs(tl),
                                                            std::fabs(th)), k);
    return std::isfinite(et) && std::isfinite(ot) &&
           std::max(std::fabs(et), std::fabs(ot)) / scale < 2e-11;
}

template <int Order, class Scalar>
inline std::array<GmSeries<Order, Scalar>, 5> fold_boundary_series(
    double center, const Params<Scalar>& p) {
    using S = GmSeries<Order, Scalar>;
    Params<S> ps{S(p.a), S(p.m0), S(p.X), S(p.Y), S(p.rho)};
    if constexpr (std::is_same_v<Scalar, double>) {
        return boundary_P_radius_series<Order>(center, ps);
    } else {
        return boundary_P_radius_series_dual<Order>(center, ps);
    }
}

template <int Order, class Scalar>
inline bool fold_root_pair_series(
    const std::array<GmSeries<Order, Scalar>, 5>& p, double m0, double v0,
    GmSeries<Order, Scalar>& m, GmSeries<Order, Scalar>& v) {
    using S = GmSeries<Order, Scalar>;
    m = S(Scalar(m0));
    v = S(Scalar(v0));
    const auto jacobian_value = [&] {
        const double m2 = m0 * m0;
        const double em = scalar_value(p[1].c[0]) +
                          2.0 * scalar_value(p[2].c[0]) * m0 +
                          scalar_value(p[3].c[0]) * (3.0 * m2 + 3.0 * v0) +
                          scalar_value(p[4].c[0]) *
                              (4.0 * m2 * m0 + 12.0 * m0 * v0);
        const double ev = scalar_value(p[2].c[0]) +
                          3.0 * scalar_value(p[3].c[0]) * m0 +
                          scalar_value(p[4].c[0]) * (6.0 * m2 + 2.0 * v0);
        const double om = 2.0 * scalar_value(p[2].c[0]) +
                          6.0 * scalar_value(p[3].c[0]) * m0 +
                          scalar_value(p[4].c[0]) * (12.0 * m2 + 4.0 * v0);
        const double ov = scalar_value(p[3].c[0]) +
                          4.0 * scalar_value(p[4].c[0]) * m0;
        return std::array<double, 5>{{em, ev, om, ov,
                                      em * ov - ev * om}};
    };
    const auto j0 = jacobian_value();
    if (!std::isfinite(j0[4]) || std::fabs(j0[4]) < 1e-30) return false;

    if constexpr (std::is_same_v<Scalar, Dual5>) {
        // At h=0 the parameter derivatives are the IFT solution of the same
        // two equations.  The R Taylor coefficients below then propagate the
        // derivatives without finite differences.
        S m_fixed{Dual5(m0)}, v_fixed{Dual5(v0)};
        const auto e0 = fold_pair_equations(p, m_fixed, v_fixed);
        for (int q = 0; q < 5; ++q) {
            const double ep = e0[0].c[0].deriv[q];
            const double op = e0[1].c[0].deriv[q];
            m.c[0].deriv[q] = (-ep * j0[3] + j0[1] * op) / j0[4];
            v.c[0].deriv[q] = (j0[2] * ep - j0[0] * op) / j0[4];
        }
    }

    // For Dual5 the coefficient Jacobian itself carries parameter
    // derivatives through p[0] and the solved c[0] of (m,v).  Keeping only
    // its value would make the order-n root sensitivities correct at the
    // packet center but wrong as soon as h != 0.  Rebuild it in Scalar after
    // the c[0] IFT solve so the recurrence differentiates J^{-1} as well.
    const Scalar mm = m.c[0];
    const Scalar vv = v.c[0];
    const Scalar m2 = mm * mm;
    const Scalar em = p[1].c[0] + Scalar(2) * p[2].c[0] * mm +
                      p[3].c[0] * (Scalar(3) * m2 + Scalar(3) * vv) +
                      p[4].c[0] * (Scalar(4) * m2 * mm +
                                    Scalar(12) * mm * vv);
    const Scalar ev = p[2].c[0] + Scalar(3) * p[3].c[0] * mm +
                      p[4].c[0] * (Scalar(6) * m2 + Scalar(2) * vv);
    const Scalar om = Scalar(2) * p[2].c[0] + Scalar(6) * p[3].c[0] * mm +
                      p[4].c[0] * (Scalar(12) * m2 + Scalar(4) * vv);
    const Scalar ov = p[3].c[0] + Scalar(4) * p[4].c[0] * mm;
    const Scalar det = em * ov - ev * om;
    if (!scalar_finite(det) || std::fabs(scalar_value(det)) < 1e-30)
        return false;

    for (int n = 1; n <= Order; ++n) {
        const auto e = fold_pair_equations(p, m, v);
        const Scalar en = e[0].c[n];
        const Scalar on = e[1].c[n];
        m.c[n] = (en * (-ov) + on * ev) / det;
        v.c[n] = (en * om - on * em) / det;
        if (!scalar_finite(m.c[n]) || !scalar_finite(v.c[n])) return false;
    }
    return series_finite(m) && series_finite(v);
}

template <int Order, class Scalar>
inline GmSeries<Order, Scalar> fold_series_power0(
    const GmSeries<Order, Scalar>& x, double exponent) {
    if (exponent == 0.5) return series_sqrt(x);
    const auto inv = gm_series_inv(x);
    if (exponent == -1.0) return inv;
    const auto root = series_sqrt(x);
    const auto inv_root = gm_series_inv(root);
    if (exponent == -0.5) return inv_root;
    if (exponent == -1.5) return inv * inv_root;
    return GmSeries<Order, Scalar>(Scalar(0));
}

template <int Degree, int Order, class Scalar>
using FoldTJet = std::array<GmSeries<Order, Scalar>, Degree + 1>;

template <int Degree, int Order, class Scalar>
inline FoldTJet<Degree, Order, Scalar> fold_tjet_mul(
    const FoldTJet<Degree, Order, Scalar>& a,
    const FoldTJet<Degree, Order, Scalar>& b) {
    FoldTJet<Degree, Order, Scalar> out{};
    for (int i = 0; i <= Degree; ++i)
        for (int j = 0; j + i <= Degree; ++j)
            out[i + j] = out[i + j] + a[i] * b[j];
    return out;
}

template <int Degree, int Order, class Scalar>
inline FoldTJet<Degree, Order, Scalar> fold_tjet_binomial(
    const FoldTJet<Degree, Order, Scalar>& q, double exponent) {
    using S = GmSeries<Order, Scalar>;
    const S inv_q0 = gm_series_inv(q[0]);
    FoldTJet<Degree, Order, Scalar> u{};
    for (int k = 0; k <= Degree; ++k) u[k] = q[k] * inv_q0;
    u[0] = u[0] - S(Scalar(1));

    FoldTJet<Degree, Order, Scalar> term{};
    FoldTJet<Degree, Order, Scalar> sum{};
    term[0] = S(Scalar(1));
    double binomial = 1.0;
    for (int power = 0; power <= Degree; ++power) {
        if (power != 0) {
            term = fold_tjet_mul<Degree, Order, Scalar>(term, u);
            binomial *= (exponent - (power - 1.0)) / power;
        }
        for (int k = 0; k <= Degree; ++k)
            sum[k] = sum[k] + gm_series_scale_scalar(
                                  term[k], Scalar(binomial));
    }
    const S base = fold_series_power0(q[0], exponent);
    for (int k = 0; k <= Degree; ++k) sum[k] = base * sum[k];
    return sum;
}

template <int Degree, int Order, class Scalar>
inline FoldTJet<Degree, Order, Scalar> fold_smooth_taylor(
    const GmSeries<Order, Scalar>& R,
    const GmSeries<Order, Scalar>& m,
    const GmSeries<Order, Scalar>& v,
    const Params<GmSeries<Order, Scalar>>& params,
    const std::array<GmSeries<Order, Scalar>, 5>& p) {
    using S = GmSeries<Order, Scalar>;
    const S p4 = p[4];
    const S d1 = p[3] + gm_series_scale_scalar(m * p4, Scalar(2));
    const S d0 = p[2] + gm_series_scale_scalar(m * d1, Scalar(2)) -
                 (m * m - v) * p4;

    FoldTJet<Degree, Order, Scalar> s2{}, aa{}, bb{};
    s2[0] = -(d0 + m * (d1 + m * p4));
    s2[1] = -(d1 + gm_series_scale_scalar(m * p4, Scalar(2)));
    s2[2] = -p4;
    aa[0] = S(Scalar(1)) + m * m;
    aa[1] = gm_series_scale_scalar(m, Scalar(2));
    aa[2] = S(Scalar(1));
    const S rma = R - params.a;
    const S rpa = R + params.a;
    const S rma2 = rma * rma;
    const S rpa2 = rpa * rpa;
    bb[0] = rma2 + rpa2 * m * m;
    bb[1] = gm_series_scale_scalar(rpa2 * m, Scalar(2));
    bb[2] = rpa2;

    const auto hs = fold_tjet_binomial<Degree, Order, Scalar>(s2, 0.5);
    const auto ha = fold_tjet_binomial<Degree, Order, Scalar>(aa, -1.5);
    const auto hb = fold_tjet_binomial<Degree, Order, Scalar>(bb, -0.5);
    return fold_tjet_mul<Degree, Order, Scalar>(
        fold_tjet_mul<Degree, Order, Scalar>(hs, ha), hb);
}

inline double fold_even_moment(int n) {
    // mu_(2n) = pi Catalan(n) / (2 * 4^n).  Phase 6 currently keeps
    // n=0..4; spelling the constants out avoids any runtime combinatorics.
    static constexpr double moments[] = {
        kPi / 2.0, kPi / 8.0, kPi / 16.0, 5.0 * kPi / 128.0,
        7.0 * kPi / 256.0};
    return (n >= 0 && n < 5) ? moments[n] : 0.0;
}

template <int Order, class Scalar>
struct FoldFluxSeries {
    GmSeries<Order, Scalar> full{};
    GmSeries<Order, Scalar> low{};
};

template <int Order, class Scalar>
inline FoldFluxSeries<Order, Scalar> fold_flux_series(
    const GmSeries<Order, Scalar>& R,
    const GmSeries<Order, Scalar>& m,
    const GmSeries<Order, Scalar>& v,
    const Params<GmSeries<Order, Scalar>>& params,
    const std::array<GmSeries<Order, Scalar>, 5>& p) {
    using S = GmSeries<Order, Scalar>;
    constexpr int kMomentTerms = 4;
    const auto h = fold_smooth_taylor<2 * kMomentTerms>(R, m, v, params, p);
    S v_power(Scalar(1));
    S sum_full(Scalar(0));
    S sum_low(Scalar(0));
    for (int n = 0; n <= kMomentTerms; ++n) {
        const S term = h[2 * n] * v_power;
        sum_full = sum_full + gm_series_scale_scalar(
            term, Scalar(fold_even_moment(n)));
        if (n < kMomentTerms)
            sum_low = sum_low + gm_series_scale_scalar(
                term, Scalar(fold_even_moment(n)));
        v_power = v_power * v;
    }
    const S inv_rho = gm_series_inv(params.rho);
    const S prefactor = gm_series_scale_scalar(v, Scalar(2)) * inv_rho;
    return {prefactor * sum_full, prefactor * sum_low};
}

struct ArcSeed {
    bool ok = false;
    bool chart2 = false;
    bool branch_certified = false;
    PrimaryFrame chart_pf{};
    double t_lo = 0.0;
    double t_hi = 0.0;
    double theta_lo = 0.0;
    double theta_hi = 0.0;
    double chart_measure = 0.0;
};

inline double wrap_pi(double x) {
    x = std::fmod(x + kPi, kTwoPi);
    if (x < 0.0) x += kTwoPi;
    return x - kPi;
}

inline bool singular_inside(double singular, double enter, double measure) {
    double d = std::fmod(singular - enter, kTwoPi);
    if (d < 0.0) d += kTwoPi;
    return d > 1e-13 && d < measure - 1e-13;
}

inline ArcSeed prepare_arc_chart(const PrimaryFrame& pf,
                                 const std::array<double, 2>& arc,
                                 bool chart2) {
    double enter = arc[0], leave = arc[1];
    if (leave <= enter) leave += kTwoPi;
    const double measure = leave - enter;
    if (!(measure > 0.0) || !(measure < kTwoPi)) return {};
    const double shift = chart2 ? kPi : 0.0;
    // t = tan((theta-shift)/2) has its point at infinity at
    // theta = shift + pi.  With the pi-shifted frame this is theta=0,
    // not theta=pi.  Rejecting an arc that crosses this point is part of
    // the chart/branch certificate; otherwise sorting the two endpoint
    // tangents silently turns an arc through infinity into its complement.
    const double singular = chart2 ? 0.0 : kPi;
    if (singular_inside(singular, enter, measure)) return {};
    const double tl = std::tan(0.5 * wrap_pi(enter - shift));
    const double tr = std::tan(0.5 * wrap_pi(leave - shift));
    if (!std::isfinite(tl) || !std::isfinite(tr) || tl == tr) return {};
    ArcSeed cand;
    cand.ok = true;
    cand.chart2 = chart2;
    cand.chart_pf = chart2
        ? PrimaryFrame{-pf.a, pf.m0, -pf.X, -pf.Y, pf.rho}
        : pf;
    cand.t_lo = std::min(tl, tr);
    cand.t_hi = std::max(tl, tr);
    cand.theta_lo = enter;
    cand.theta_hi = leave;
    cand.chart_measure = 2.0 *
        (std::atan(cand.t_hi) - std::atan(cand.t_lo));
    if (!(cand.chart_measure > 0.0) ||
        std::fabs(cand.chart_measure - measure) >
            2e-12 * (1.0 + measure))
        return {};
    return cand;
}

inline ArcSeed prepare_arc(const PrimaryFrame& pf,
                           const std::array<double, 2>& arc) {
    ArcSeed best;
    double best_score = std::numeric_limits<double>::infinity();
    for (bool chart2 : {false, true}) {
        const ArcSeed cand = prepare_arc_chart(pf, arc, chart2);
        if (!cand.ok) continue;
        const double score = std::max(std::fabs(cand.t_lo),
                                      std::fabs(cand.t_hi));
        if (score < best_score) {
            best = cand;
            best_score = score;
        }
    }
    return best;
}

inline bool certify_arc(double R, const PrimaryFrame& pf,
                        const std::array<double, 2>& arc, ArcSeed& seed) {
    if (!seed.ok) return false;
    const double tc = 0.5 * (seed.t_lo + seed.t_hi);
    const double tq0 = 0.5 * (seed.t_lo + tc);
    const double tq1 = 0.5 * (tc + seed.t_hi);
    const double ph0 = phi_lens(R, 2.0 * std::atan(tq0), seed.chart_pf);
    const double phc = phi_lens(R, 2.0 * std::atan(tc), seed.chart_pf);
    const double ph1 = phi_lens(R, 2.0 * std::atan(tq1), seed.chart_pf);
    if (!(ph0 > 0.0) || !(phc > 0.0) || !(ph1 > 0.0)) return false;
    double enter = arc[0], leave = arc[1];
    if (leave <= enter) leave += kTwoPi;
    const double measure = leave - enter;
    if (!(measure > 0.0) || !(measure < kTwoPi) ||
        std::fabs(seed.chart_measure - measure) >
            2e-12 * (1.0 + measure))
        return false;
    seed.branch_certified = true;
    return true;
}

inline std::complex<double> factor_half(std::complex<double> x) {
    return std::sqrt(x);
}

inline std::complex<double> fd6_factor_product(
    const std::array<Complex<double>, 6>& z, double u) {
    std::complex<double> h(1.0, 0.0);
    for (int j = 0; j < 6; ++j) {
        const std::complex<double> f(1.0 - z[j].re * u, -z[j].im * u);
        if (j < 2) {
            h *= factor_half(f);
        } else if (j < 4) {
            h /= f * factor_half(f);
        } else {
            h /= factor_half(f);
        }
    }
    return h;
}

struct State {
    std::array<Complex<double>, kStateDim> z{};
};

// Regular physical branch at an exact root-pair fold.  At v=0 all finite
// cross-ratios vanish, so the normalized FD5 branch has F=1 and M_i=1/2.
// The observable is J=v*K and the finite germ coefficient is evaluated
// directly from the deflated quadratic, without dividing by v.
struct FoldGerm {
    bool ok = false;
    std::array<Complex<double>, kStateDim> state{};
    double K0 = 0.0;
};

inline FoldGerm fold_germ(double R, double t_fold,
                          const Params<double>& p) {
    FoldGerm out;
    const auto P = boundary_P(R, p);
    const double P2 = 2.0 * P[2] + 6.0 * P[3] * t_fold +
                      12.0 * P[4] * t_fold * t_fold;
    const double S = -0.5 * P2;
    const double A = 1.0 + t_fold * t_fold;
    const double B = (R - p.a) * (R - p.a) +
                     (R + p.a) * (R + p.a) * t_fold * t_fold;
    if (!(S > 0.0) || !(A > 0.0) || !(B > 0.0) || !(p.rho > 0.0) ||
        !std::isfinite(S) || !std::isfinite(B))
        return out;
    out.state[0] = {1.0, 0.0};
    for (int i = 1; i < kStateDim; ++i) out.state[i] = {0.5, 0.0};
    out.K0 = kPi * std::sqrt(S) /
             (p.rho * A * std::sqrt(A) * std::sqrt(B));
    out.ok = std::isfinite(out.K0) && out.K0 > 0.0;
    return out;
}

struct SeedResult {
    bool ok = false;
    State state{};
    double fhalf = 0.0;
    double f6 = 0.0;
    double phase_error = 0.0;
};

inline SeedResult physical_seed(const Geometry<double>& g, int n = 512) {
    SeedResult out;
    if (!g.ok || n < 16) return out;
    if (g.v == 0.0) {
        out.state.z[0] = {1.0, 0.0};
        for (int i = 1; i < kStateDim; ++i) out.state.z[i] = {0.5, 0.0};
        out.ok = true;
        return out;
    }
    std::array<std::complex<double>, 5> xi{};
    for (int j = 0; j < 5; ++j) xi[j] = {g.xi[j].re, g.xi[j].im};
    std::complex<double> F6(0.0, 0.0);
    std::array<std::complex<double>, 5> G6{};
    for (int k = 1; k <= n; ++k) {
        const double th = kPi * k / (n + 1.0);
        const double x = std::cos(th);
        const double u = 0.5 * (1.0 + x);
        const double s2 = std::sin(th) * std::sin(th);
        const auto h = fd6_factor_product(g.z, u);
        F6 += (2.0 / (n + 1.0)) * s2 * h;
        for (int j = 0; j < 5; ++j)
            G6[j] += (2.0 / (n + 1.0)) * s2 *
                     (u / std::complex<double>(1.0 - g.z[j < 2 ? j : j + 1].re * u,
                                                -g.z[j < 2 ? j : j + 1].im * u)) * h;
    }
    const std::complex<double> q(g.q.re, g.q.im);
    const std::complex<double> qpow = std::pow(q, 1.5);
    const std::complex<double> F = qpow * F6;
    out.state.z[0] = {F.real(), F.imag()};
    for (int j = 0; j < 5; ++j) {
        const std::complex<double> G = qpow * q * G6[j];
        const std::complex<double> M = F + (xi[j] - 1.0) * G;
        out.state.z[j + 1] = {M.real(), M.imag()};
    }
    const std::complex<double> C(g.C.re, g.C.im);
    const std::complex<double> C6(g.C6.re, g.C6.im);
    const std::complex<double> value = C * F;
    const std::complex<double> value6 = C6 * F6;
    out.f6 = value6.real();
    out.fhalf = value.real();
    out.phase_error = std::fabs(value.imag()) /
                      (1.0 + std::fabs(value.real()));
    out.ok = std::isfinite(out.fhalf) && std::isfinite(out.f6) &&
             out.phase_error < 1e-8;
    return out;
}

// One-rung precision promotion for the seed only.  The transport and its
// algebraic geometry remain double; this path is entered only after the
// double seed/geometry gate has failed, so ordinary cells do not pay for it.
inline SeedResult physical_seed_long_double(const Geometry<long double>& g,
                                            int n = 512) {
    SeedResult out;
    if (!g.ok || n < 16) return out;
    if (g.v == 0.0L) {
        out.state.z[0] = {1.0, 0.0};
        for (int i = 1; i < kStateDim; ++i) out.state.z[i] = {0.5, 0.0};
        out.ok = true;
        return out;
    }
    using CL = std::complex<long double>;
    std::array<CL, 5> xi{};
    std::array<CL, 6> z{};
    for (int j = 0; j < 5; ++j) xi[j] = {g.xi[j].re, g.xi[j].im};
    for (int j = 0; j < 6; ++j) z[j] = {g.z[j].re, g.z[j].im};
    CL F6(0.0L, 0.0L);
    std::array<CL, 5> G6{};
    for (int k = 1; k <= n; ++k) {
        const long double th = (long double)kPi * k / (n + 1.0L);
        const long double u = 0.5L * (1.0L + std::cos(th));
        const long double s2 = std::sin(th) * std::sin(th);
        CL h(1.0L, 0.0L);
        for (int j = 0; j < 6; ++j) {
            const CL f = 1.0L - z[j] * u;
            if (j < 2) h *= std::sqrt(f);
            else if (j < 4) h /= f * std::sqrt(f);
            else h /= std::sqrt(f);
        }
        const long double w = 2.0L / (n + 1.0L) * s2;
        F6 += w * h;
        for (int j = 0; j < 5; ++j)
            G6[j] += w * (u / (1.0L - z[j < 2 ? j : j + 1] * u)) * h;
    }
    const CL q(g.q.re, g.q.im);
    const CL qpow = std::pow(q, 1.5L);
    const CL F = qpow * F6;
    out.state.z[0] = {(double)F.real(), (double)F.imag()};
    for (int j = 0; j < 5; ++j) {
        const CL G = qpow * q * G6[j];
        const CL M = F + (xi[j] - 1.0L) * G;
        out.state.z[j + 1] = {(double)M.real(), (double)M.imag()};
    }
    const CL C(g.C.re, g.C.im);
    const CL C6(g.C6.re, g.C6.im);
    const CL value = C * F;
    const CL value6 = C6 * F6;
    out.fhalf = (double)value.real();
    out.f6 = (double)value6.real();
    out.phase_error = (double)(std::fabs(value.imag()) /
                                (1.0L + std::fabs(value.real())));
    out.ok = std::isfinite(out.fhalf) && std::isfinite(out.f6) &&
             std::isfinite(out.phase_error) && out.phase_error < 1e-10;
    return out;
}

template <int Order>
inline GmSeries<Order, double> eval_series(
    const GmSeries<Order, double>& a, double h) {
    GmSeries<Order, double> r;
    r.c[Order] = a.c[Order];
    for (int k = Order - 1; k >= 0; --k)
        r = r * GmSeries<Order, double>(h) +
            GmSeries<Order, double>(a.c[k]);
    return r;
}

template <int Order>
inline double eval_series_value(const GmSeries<Order, double>& a, double h,
                                int degree = Order) {
    double r = a.c[degree];
    for (int k = degree - 1; k >= 0; --k) r = r * h + a.c[k];
    return r;
}

struct FoldPacketBuildBreakdown {
    double pair_series_ms = 0.0;
    double smooth_series_ms = 0.0;
    double quality_ms = 0.0;
    double total_ms = 0.0;
};

template <int Order>
struct FoldPacket {
    bool ok = false;
    bool has_jacobian = false;
    double center = 0.0;
    double lo = 0.0;
    double hi = 0.0;
    GmSeries<Order, double> coeff{};
    GmSeries<Order, double> low_coeff{};
    std::array<GmSeries<Order, double>, 5> deriv{};
    std::array<GmSeries<Order, double>, 5> low_deriv{};
    double taylor_tail = std::numeric_limits<double>::infinity();
    double moment_tail = std::numeric_limits<double>::infinity();
    double pair_residual = std::numeric_limits<double>::infinity();
    double min_v = -std::numeric_limits<double>::infinity();
    int failure_reason = 0;
};

template <int Order>
inline double fold_series_value(const GmSeries<Order, double>& a, double h,
                                int degree = Order) {
    return eval_series_value(a, h, degree);
}

template <int Order>
inline double fold_pair_residual_at(
    const std::array<GmSeries<Order, double>, 5>& p,
    const GmSeries<Order, double>& m, const GmSeries<Order, double>& v,
    double h) {
    std::array<double, 5> pv{};
    for (int k = 0; k < 5; ++k) pv[k] = eval_series_value(p[k], h);
    const double mv = eval_series_value(m, h);
    const double vv = eval_series_value(v, h);
    double e = 0.0, o = 0.0;
    fold_pair_equations_value(pv, mv, vv, e, o);
    const double scale = 1.0 + std::fabs(e) + std::fabs(o) +
                         std::fabs(pv[0]) + std::fabs(pv[1]) *
                             (1.0 + std::fabs(mv));
    return std::max(std::fabs(e), std::fabs(o)) / scale;
}

template <int Order>
inline double fold_relative_series_tail(
    const GmSeries<Order, double>& full,
    const GmSeries<Order, double>& low, double h) {
    const double a = eval_series_value(full, h);
    const double b = eval_series_value(low, h);
    return std::fabs(a - b) / (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

template <int Order>
inline double fold_order_tail(const GmSeries<Order, double>& full, double h) {
    if constexpr (Order < 2) {
        return 0.0;
    } else {
        const double a = eval_series_value(full, h, Order);
        const double b = eval_series_value(full, h, Order - 2);
        return std::fabs(a - b) / (1.0 + std::max(std::fabs(a), std::fabs(b)));
    }
}

template <int Order>
inline bool fold_packet_quality(
    FoldPacket<Order>& packet,
    const std::array<GmSeries<Order, double>, 5>& p,
    const GmSeries<Order, double>& m, const GmSeries<Order, double>& v) {
    if constexpr (Order < 2) {
        packet.taylor_tail = 0.0;
        packet.moment_tail = 0.0;
        packet.pair_residual = 0.0;
        packet.min_v = eval_series_value(v, 0.0);
        return packet.ok;
    } else {
        const double span = 0.98 *
            std::min(packet.center - packet.lo, packet.hi - packet.center);
        if (!(span > 0.0)) return false;
        packet.taylor_tail = 0.0;
        packet.moment_tail = 0.0;
        packet.pair_residual = 0.0;
        packet.min_v = std::numeric_limits<double>::infinity();
        for (double h : {span, -span}) {
            packet.taylor_tail = std::max(
                packet.taylor_tail,
                fold_order_tail(packet.coeff, h));
            packet.moment_tail = std::max(
                packet.moment_tail,
                fold_relative_series_tail(packet.coeff, packet.low_coeff, h));
            if (packet.has_jacobian) {
                for (int j = 0; j < 5; ++j) {
                    packet.taylor_tail = std::max(
                        packet.taylor_tail,
                        fold_order_tail(packet.deriv[j], h));
                    packet.moment_tail = std::max(
                        packet.moment_tail,
                        fold_relative_series_tail(packet.deriv[j],
                                                  packet.low_deriv[j], h));
                }
            }
            packet.pair_residual = std::max(
                packet.pair_residual, fold_pair_residual_at(p, m, v, h));
            packet.min_v = std::min(packet.min_v, eval_series_value(v, h));
        }
        packet.min_v = std::min(packet.min_v, eval_series_value(v, 0.0));
        return packet.ok && std::isfinite(packet.taylor_tail) &&
               std::isfinite(packet.moment_tail) &&
               std::isfinite(packet.pair_residual) &&
               std::isfinite(packet.min_v) && packet.min_v > 0.0 &&
               packet.taylor_tail < 2e-11 &&
               packet.moment_tail < 2e-11 &&
               packet.pair_residual < 2e-10;
    }
}

template <int Order>
inline bool build_fold_packet(
    double center, double lo, double hi, double tlo, double thi,
    const Params<double>& params, const PrimaryFrame& pf, bool chart2,
    bool with_jacobian, FoldPacket<Order>& out,
    FoldPacketBuildBreakdown* timing = nullptr) {
    using Clock = std::chrono::steady_clock;
    using S = GmSeries<Order, double>;
    const auto started = timing ? Clock::now() : Clock::time_point{};
    auto finish = [&](bool ok) {
        if (timing)
            timing->total_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - started).count();
        return ok;
    };
    out = FoldPacket<Order>{};
    out.center = center;
    out.lo = lo;
    out.hi = hi;
    double m0 = 0.5 * (tlo + thi);
    double s0 = 0.5 * (thi - tlo);
    double v0 = s0 * s0;
    if (!(thi > tlo) || !(s0 > 0.0) || !std::isfinite(m0) ||
        !std::isfinite(v0)) {
        out.failure_reason = 1;
        return finish(false);
    }

    const auto pair_start = timing ? Clock::now() : Clock::time_point{};
    const auto pc = boundary_P(center, params);
    if (!fold_pair_refine(pc, m0, v0)) {
        out.failure_reason = 2;
        return finish(false);
    }
    const auto p = fold_boundary_series<Order>(center, params);
    S R{center};
    if constexpr (Order >= 1) R.c[1] = 1.0;
    const Params<S> ps{S(params.a), S(params.m0), S(params.X),
                       S(params.Y), S(params.rho)};
    S m, v;
    if (!fold_root_pair_series(p, m0, v0, m, v)) {
        out.failure_reason = 3;
        return finish(false);
    }
    if (timing) {
        timing->pair_series_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - pair_start).count();
    }

    const auto smooth_start = timing ? Clock::now() : Clock::time_point{};
    if (!with_jacobian) {
        const auto flux = fold_flux_series(R, m, v, ps, p);
        out.coeff = flux.full;
        out.low_coeff = flux.low;
        out.has_jacobian = false;
    } else {
        const Params<Dual5> dp = dual_params_for_chart(pf, chart2);
        const auto pd = fold_boundary_series<Order>(center, dp);
        using DS = GmSeries<Order, Dual5>;
        DS Rd{Dual5(center)};
        if constexpr (Order >= 1) Rd.c[1] = Dual5(1.0);
        const Params<DS> dps{DS(dp.a), DS(dp.m0), DS(dp.X), DS(dp.Y),
                             DS(dp.rho)};
        DS md, vd;
        if (!fold_root_pair_series(pd, m0, v0, md, vd)) {
            out.failure_reason = 4;
            return finish(false);
        }
        const auto flux = fold_flux_series(Rd, md, vd, dps, pd);
        out.has_jacobian = true;
        for (int n = 0; n <= Order; ++n) {
            out.coeff.c[n] = flux.full.c[n].value;
            out.low_coeff.c[n] = flux.low.c[n].value;
            for (int j = 0; j < 5; ++j) {
                out.deriv[j].c[n] = flux.full.c[n].deriv[j];
                out.low_deriv[j].c[n] = flux.low.c[n].deriv[j];
            }
        }
    }
    if (timing) {
        timing->smooth_series_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - smooth_start).count();
    }
    out.ok = series_finite(out.coeff) && series_finite(out.low_coeff);
    if (out.has_jacobian) {
        for (int j = 0; j < 5; ++j)
            out.ok = out.ok && series_finite(out.deriv[j]) &&
                     series_finite(out.low_deriv[j]);
    }
    if (!out.ok) {
        out.failure_reason = 5;
        return finish(false);
    }
    const auto quality_start = timing ? Clock::now() : Clock::time_point{};
    const bool quality_ok = fold_packet_quality(out, p, m, v);
    if (timing) {
        timing->quality_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - quality_start).count();
    }
    if (!quality_ok) out.failure_reason = 6;
    return finish(quality_ok);
}

template <int Order>
inline GmSeries<Order, double> eval_poly_series(
    const std::array<GmSeries<Order, double>, 5>& p,
    const GmSeries<Order, double>& t) {
    GmSeries<Order, double> r = p[4];
    for (int k = 3; k >= 0; --k) r = r * t + p[k];
    return r;
}

template <int Order>
inline double poly_derivative_at(const std::array<GmSeries<Order, double>, 5>& p,
                                 double t) {
    return p[1].c[0] + t * (2.0 * p[2].c[0] +
                            t * (3.0 * p[3].c[0] + 4.0 * t * p[4].c[0]));
}

template <int Order>
inline bool root_series(const std::array<GmSeries<Order, double>, 5>& p,
                        double t0, GmSeries<Order, double>& out,
                        int p_degree = Order) {
    out = GmSeries<Order, double>(t0);
    const double pt = poly_derivative_at(p, t0);
    if (!std::isfinite(pt) || std::fabs(pt) < 1e-15) return false;

    // Coefficient-only implicit composition.  The old path evaluated a
    // complete Horner polynomial in series arithmetic for every n, which
    // repeatedly recomputed all coefficients below n.  `powers[k][n]` is the
    // coefficient of t(h)^k at the current order; its linear dependence on
    // the newly solved t_n is restored after the residual is measured.
    std::array<std::array<double, Order + 1>, 5> powers{};
    powers[0][0] = 1.0;
    for (int k = 1; k <= 4; ++k)
        powers[k][0] = powers[k - 1][0] * t0;
    for (int n = 1; n <= Order; ++n) {
        // First form all t^k coefficients with t_n set to zero.  Lower
        // powers at order n are already available because k increases here.
        for (int k = 1; k <= 4; ++k) {
            double v = 0.0;
            for (int j = 0; j < n; ++j)
                v += out.c[j] * powers[k - 1][n - j];
            powers[k][n] = v;
        }
        double residual = 0.0;
        const int degree = std::min(n, p_degree);
        for (int k = 0; k <= 4; ++k)
            for (int j = 0; j <= degree; ++j)
                residual += p[k].c[j] * powers[k][n - j];
        out.c[n] = -residual / pt;
        if (!std::isfinite(out.c[n])) return false;
        double t0_power = 1.0;
        for (int k = 1; k <= 4; ++k) {
            powers[k][n] += k * t0_power * out.c[n];
            t0_power *= t0;
        }
    }
    return true;
}

template <int Order>
inline std::array<Complex<double>, Order> log_derivative_series(
    const Complex<GmSeries<Order, double>>& g) {
    std::array<Complex<double>, Order> ell{};
    const Complex<double> g0{g.re.c[0], g.im.c[0]};
    const double g0_norm = g0.re * g0.re + g0.im * g0.im;
    const Complex<double> inv_g0{g0.re / g0_norm, -g0.im / g0_norm};
    for (int n = 0; n < Order; ++n) {
        Complex<double> rhs{(n + 1) * g.re.c[n + 1],
                            (n + 1) * g.im.c[n + 1]};
        for (int k = 1; k <= n; ++k) {
            const Complex<double> gk{g.re.c[k], g.im.c[k]};
            rhs = rhs - gk * ell[n - k];
        }
        ell[n] = rhs * inv_g0;
    }
    return ell;
}

using ValueState = std::array<Complex<double>, kStateDim>;

template <int Order>
struct TaylorPacket {
    bool ok = false;
    double center = 0.0;
    double lo = 0.0;
    double hi = 0.0;
    GmSeries<Order, double> t_lo{};
    GmSeries<Order, double> t_hi{};
    SeriesGeometry<Order, double> geometry{};
    std::array<Complex<double>, Order + 1> coeff[kStateDim]{};
    double tail = std::numeric_limits<double>::infinity();
    double strict_tail = std::numeric_limits<double>::infinity();
    double projected_tail = std::numeric_limits<double>::infinity();
    double pfaffian_residual = std::numeric_limits<double>::infinity();
};

template <int Order>
inline ValueState eval_state(const TaylorPacket<Order>& packet, double h,
                             int degree = Order) {
    ValueState out{};
    for (int i = 0; i < kStateDim; ++i) {
        Complex<double> v = packet.coeff[i][degree];
        for (int n = degree - 1; n >= 0; --n)
            v = v * Complex<double>{h, 0.0} + packet.coeff[i][n];
        out[i] = v;
    }
    return out;
}

template <int Order>
inline std::array<Complex<double>, 5> eval_xi(
    const TaylorPacket<Order>& packet, double h, int degree = Order) {
    std::array<Complex<double>, 5> out{};
    for (int i = 0; i < 5; ++i) {
        out[i].re = eval_series_value(packet.geometry.xi[i].re, h, degree);
        out[i].im = eval_series_value(packet.geometry.xi[i].im, h, degree);
    }
    return out;
}

// Packet construction is the Phase 5 hot path.  Keep its accounting in the
// isolated PF6 kernel so that the benchmark can separate mathematical work
// from the surrounding epoch stages.  Timing is opt-in; counters and
// certificates are collected on every PF6 call.
struct PacketBuildBreakdown {
    double endpoint_root_ms = 0.0;
    double algebraic_geometry_ms = 0.0;
    double log_derivative_ms = 0.0;
    double pfaffian_recurrence_ms = 0.0;
    double quality_tail_ms = 0.0;
    double total_ms = 0.0;
};

struct PacketArcProfile {
    int arc_index = -1;
    int cell_index = -1;
    double cell_lo = 0.0;
    double cell_hi = 0.0;
    double radial_center = 0.0;
    double theta_lo = 0.0;
    double theta_hi = 0.0;
    double event_gap_ratio = std::numeric_limits<double>::infinity();
    double xi_min_divisor = std::numeric_limits<double>::infinity();
    int nearest_event_kind = 0;  // 1=fold, 2=chart_p4, 3=D14, 4=other
    int chart2 = 0;
    int constructions = 0;
    int built = 0;
    int accepted = 0;
    int rejected = 0;
    int quality_rejected = 0;
    int build_failed = 0;
    int max_depth = 0;
    int accepted_node_sum = 0;
    int accepted_node_min = 0;
    int accepted_node_max = 0;
    int node_failed = 0;
    int fold_constructions = 0;
    int fold_accepted = 0;
    int fold_rejected = 0;
    int fold_failure_reason = 0;
    double fold_construction_ms = 0.0;
    double fold_taylor_tail = std::numeric_limits<double>::infinity();
    double fold_moment_tail = std::numeric_limits<double>::infinity();
    double fold_pair_residual = std::numeric_limits<double>::infinity();
    double fold_v_center = std::numeric_limits<double>::quiet_NaN();
    double fold_v_side = std::numeric_limits<double>::quiet_NaN();
    double fold_v_ratio = std::numeric_limits<double>::quiet_NaN();
    double fold_event_radius = std::numeric_limits<double>::quiet_NaN();
    double first_failed_R = std::numeric_limits<double>::quiet_NaN();
    double first_failed_packet_center = std::numeric_limits<double>::quiet_NaN();
    double first_failed_packet_lo = std::numeric_limits<double>::quiet_NaN();
    double first_failed_packet_hi = std::numeric_limits<double>::quiet_NaN();
    double first_failed_block_lo = std::numeric_limits<double>::quiet_NaN();
    double first_failed_block_hi = std::numeric_limits<double>::quiet_NaN();
    double first_failed_phase_error = std::numeric_limits<double>::quiet_NaN();
    double first_failed_packet_tail = std::numeric_limits<double>::quiet_NaN();
    double construction_ms = 0.0;
};

struct PacketProfile {
    bool timing_enabled = false;
    long long candidates = 0;
    long long built = 0;
    long long accepted = 0;
    long long rejected = 0;
    long long quality_rejected = 0;
    long long build_failed = 0;
    std::array<long long, 17> depth_hist{};
    std::array<long long, 65> accepted_node_hist{};
    double endpoint_root_ms = 0.0;
    double algebraic_geometry_ms = 0.0;
    double log_derivative_ms = 0.0;
    double pfaffian_recurrence_ms = 0.0;
    double quality_tail_ms = 0.0;
    double accepted_candidate_ms = 0.0;
    double rejected_candidate_ms = 0.0;
};

inline void record_packet_build(
    PacketProfile* profile, PacketArcProfile* arc,
    const PacketBuildBreakdown* timing, bool built, bool accepted,
    bool quality_rejected, int depth) {
    if (profile) {
        ++profile->candidates;
        if (built) ++profile->built;
        if (accepted) {
            ++profile->accepted;
        } else {
            ++profile->rejected;
        }
        if (quality_rejected) ++profile->quality_rejected;
        if (!built) ++profile->build_failed;
        const int d = std::max(0, std::min(depth,
                                           (int)profile->depth_hist.size() - 1));
        ++profile->depth_hist[d];
        if (timing) {
            profile->endpoint_root_ms += timing->endpoint_root_ms;
            profile->algebraic_geometry_ms += timing->algebraic_geometry_ms;
            profile->log_derivative_ms += timing->log_derivative_ms;
            profile->pfaffian_recurrence_ms += timing->pfaffian_recurrence_ms;
            profile->quality_tail_ms += timing->quality_tail_ms;
            if (accepted) profile->accepted_candidate_ms += timing->total_ms;
            else profile->rejected_candidate_ms += timing->total_ms;
        }
    }
    if (arc) {
        ++arc->constructions;
        if (built) ++arc->built;
        if (accepted) {
            ++arc->accepted;
        } else {
            ++arc->rejected;
        }
        if (quality_rejected) ++arc->quality_rejected;
        if (!built) ++arc->build_failed;
        arc->max_depth = std::max(arc->max_depth, depth);
        if (timing) arc->construction_ms += timing->total_ms;
    }
}

inline bool packet_profile_enabled() {
    const char* e = std::getenv("GM6_PACKET_PROFILE");
    return e && e[0] != '\0' && e[0] != '0';
}

inline double xi_divisor_distance(const Geometry<double>& geometry) {
    double d = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 5; ++i) {
        d = std::min(d, cabs_value(geometry.xi[i]));
        d = std::min(d, cabs_value(geometry.xi[i] -
                                   Complex<double>{1.0, 0.0}));
        for (int j = i + 1; j < 5; ++j)
            d = std::min(d, cabs_value(geometry.xi[i] - geometry.xi[j]));
    }
    return d;
}

inline int radial_event_kind(const std::string& kind) {
    if (kind == "physical_real") return 1;  // a physical fold
    if (kind == "chart_p4") return 2;
    if (kind == "physical_complex") return 3;  // D14 soft event
    return 4;
}

inline void classify_packet_arc(PacketArcProfile& arc,
                                const std::vector<RadialEvent>& events) {
    const double scale = std::max(arc.cell_hi - arc.cell_lo, 1e-300);
    for (const auto& event : events) {
        const double gap = std::fabs(event.radius - arc.radial_center) / scale;
        if (gap < arc.event_gap_ratio) {
            arc.event_gap_ratio = gap;
            arc.nearest_event_kind = radial_event_kind(event.kind);
        }
    }
}

inline bool cell_touches_physical_fold(
    const CellPlan& cell, const std::vector<RadialEvent>& events) {
    for (const auto& event : events) {
        if (event.kind != "physical_real") continue;
        const double scale = std::max(1.0, std::fabs(event.radius));
        const double tol = std::max(2e-9, 3e-7 * scale);
        if (std::fabs(event.radius - cell.r_lo) <= tol ||
            std::fabs(event.radius - cell.r_hi) <= tol)
            return true;
    }
    return false;
}

struct FoldSideCertificate {
    bool candidate = false;
    bool lower = false;
    double event_radius = std::numeric_limits<double>::quiet_NaN();
    double v_center = std::numeric_limits<double>::quiet_NaN();
    double v_side = std::numeric_limits<double>::quiet_NaN();
    double ratio = std::numeric_limits<double>::infinity();
};

inline double arc_angle_distance(double a, double b) {
    return std::fabs(std::atan2(std::sin(a - b), std::cos(a - b)));
}

inline FoldSideCertificate fold_side_certificate(
    const CellPlan& cell, const std::vector<RadialEvent>& events,
    double center, const PrimaryFrame& pf, const ArcSeed& center_seed) {
    FoldSideCertificate out;
    if (!(center_seed.t_hi > center_seed.t_lo)) return out;
    out.v_center = 0.25 * (center_seed.t_hi - center_seed.t_lo) *
                   (center_seed.t_hi - center_seed.t_lo);
    if (!(out.v_center > 0.0) || !std::isfinite(out.v_center)) return out;

    double boundary = 0.0;
    bool lower = false;
    double best = std::numeric_limits<double>::infinity();
    for (const auto& event : events) {
        if (event.kind != "physical_real") continue;
        const double scale = std::max(1.0, std::fabs(event.radius));
        const double tol = std::max(2e-9, 3e-7 * scale);
        if (std::fabs(event.radius - cell.r_lo) <= tol &&
            std::fabs(event.radius - cell.r_lo) < best) {
            best = std::fabs(event.radius - cell.r_lo);
            boundary = cell.r_lo;
            lower = true;
            out.lower = true;
            out.event_radius = event.radius;
        }
        if (std::fabs(event.radius - cell.r_hi) <= tol &&
            std::fabs(event.radius - cell.r_hi) < best) {
            best = std::fabs(event.radius - cell.r_hi);
            boundary = cell.r_hi;
            lower = false;
            out.lower = false;
            out.event_radius = event.radius;
        }
    }
    if (!std::isfinite(best)) return out;
    const double width = cell.r_hi - cell.r_lo;
    if (!(width > 0.0)) return out;
    const double delta = std::max(1e-7 * width,
                                  1e-12 * std::max(1.0, std::fabs(boundary)));
    const double side_radius = lower ? boundary + delta : boundary - delta;
    if (!(side_radius > cell.r_lo && side_radius < cell.r_hi)) return out;
    const ArcSet side_arcs = arc_intervals(side_radius, pf);
    if (side_arcs.kind != ArcKind::kArcs || side_arcs.arcs.empty()) return out;
    const double center_mid = 0.5 *
        (center_seed.theta_lo + center_seed.theta_hi);
    const std::array<double, 2>* side_arc = nullptr;
    double best_angle = std::numeric_limits<double>::infinity();
    for (const auto& arc : side_arcs.arcs) {
        double enter = arc[0], leave = arc[1];
        if (leave <= enter) leave += kTwoPi;
        const double mid = 0.5 * (enter + leave);
        const double d = arc_angle_distance(mid, center_mid);
        if (d < best_angle) {
            best_angle = d;
            side_arc = &arc;
        }
    }
    if (!side_arc) return out;
    ArcSeed side_seed = prepare_arc_chart(pf, *side_arc,
                                          center_seed.chart2);
    if (!side_seed.ok || !certify_arc(side_radius, pf, *side_arc, side_seed))
        return out;
    const double dt = side_seed.t_hi - side_seed.t_lo;
    out.v_side = 0.25 * dt * dt;
    out.ratio = out.v_side / out.v_center;
    out.candidate = std::isfinite(out.v_side) && out.v_side > 0.0 &&
                    std::isfinite(out.ratio) && out.ratio < 0.5;
    return out;
}

inline bool fold_packet_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("GM6_FOLD_PACKET");
        return !(e && e[0] == '0');
    }();
    return enabled;
}

// The fold certificate decides whether the regular representation is
// applicable.  This limit only controls how many certified radial nodes are
// grouped into the fold-side candidate; it is an A/B tuning knob, not a
// physical-parameter route.  The default covers the whole certified cell in
// one candidate, so a failed quality check does not create a rebuild tree.
inline int fold_packet_node_limit() {
    static const int limit = [] {
        const char* e = std::getenv("GM6_FOLD_NODE_LIMIT");
        if (!e || e[0] == '\0') return std::numeric_limits<int>::max();
        char* end = nullptr;
        const long value = std::strtol(e, &end, 10);
        if (end == e || *end != '\0')
            return std::numeric_limits<int>::max();
        return (int)std::max(1L, std::min(4096L, value));
    }();
    return limit;
}

template <int Order>
inline double eval_root(const GmSeries<Order, double>& root, double h,
                        int degree = Order) {
    return eval_series_value(root, h, degree);
}

template <int Order>
inline double state_tail(const ValueState& a, const ValueState& b) {
    double e = 0.0;
    for (int i = 0; i < kStateDim; ++i) {
        const double da = std::hypot(a[i].re, a[i].im);
        const double db = std::hypot(b[i].re, b[i].im);
        e = std::max(e, std::hypot(a[i].re - b[i].re,
                                   a[i].im - b[i].im) /
                             (1.0 + std::max(da, db)));
    }
    return e;
}

template <int Order>
inline double eval_series_derivative_value(
    const GmSeries<Order, double>& a, double h, int degree = Order) {
    if (degree <= 0) return 0.0;
    double r = degree * a.c[degree];
    for (int n = degree - 1; n >= 1; --n) r = r * h + n * a.c[n];
    return r;
}

inline double complex_rel_tail(const Complex<double>& a,
                               const Complex<double>& b) {
    const double na = std::hypot(a.re, a.im);
    const double nb = std::hypot(b.re, b.im);
    return std::hypot(a.re - b.re, a.im - b.im) /
           (1.0 + std::max(na, nb));
}

struct PacketProjection {
    Complex<double> value{};
    std::array<Complex<double>, 5> dF_dxi{};
};

template <int Order>
inline PacketProjection eval_packet_projection(
    const TaylorPacket<Order>& packet, double h, int degree) {
    const ValueState state = eval_state(packet, h, degree);
    PacketProjection out;
    // The prefactor C is a deterministic algebraic function of the same
    // geometry and is rebuilt by evaluate_node.  The transported scalar is
    // F; keeping the gate in the normalized state avoids adding a second
    // prefactor series to every packet.
    out.value = state[0];
    const auto xi_all = eval_xi(packet, h, degree);
    for (int i = 0; i < 5; ++i) {
        const auto xi = xi_all[i];
        const Complex<double> denominator{1.0 - xi.re, -xi.im};
        out.dF_dxi[i] = cscale(
            (state[0] - state[i + 1]) / denominator, kBeta[i]);
    }
    return out;
}

template <int Order>
inline double packet_projected_tail_at(const TaylorPacket<Order>& packet,
                                       double h, int degree) {
    const PacketProjection full = eval_packet_projection(packet, h, Order);
    const PacketProjection low = eval_packet_projection(packet, h, degree);
    double e = complex_rel_tail(full.value, low.value);
    for (int i = 0; i < 5; ++i) {
        e = std::max(e, complex_rel_tail(full.dF_dxi[i], low.dF_dxi[i]));
    }
    return e;
}

template <int Order>
inline double packet_divisor_distance_at(const TaylorPacket<Order>& packet,
                                         double h, int degree) {
    const auto xi = eval_xi(packet, h, degree);
    double d = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 5; ++i) {
        d = std::min(d, std::hypot(xi[i].re, xi[i].im));
        d = std::min(d, std::hypot(xi[i].re - 1.0, xi[i].im));
        for (int j = i + 1; j < 5; ++j)
            d = std::min(d, std::hypot(xi[i].re - xi[j].re,
                                       xi[i].im - xi[j].im));
    }
    return d;
}

template <int Order>
inline double packet_pfaffian_residual_at(const TaylorPacket<Order>& packet,
                                          double h) {
    const ValueState state = eval_state(packet, h, Order);
    ValueState derivative{};
    for (int i = 0; i < kStateDim; ++i) {
        derivative[i] = {};
        if constexpr (Order > 0) {
            derivative[i] = packet.coeff[i][Order];
            derivative[i] = cscale(derivative[i], (double)Order);
            for (int n = Order - 1; n >= 1; --n)
                derivative[i] = derivative[i] * Complex<double>{h, 0.0} +
                                cscale(packet.coeff[i][n], (double)n);
        }
    }
    std::array<Complex<double>, 5> xi{};
    std::array<Complex<double>, 5> xi_r{};
    for (int i = 0; i < 5; ++i) {
        xi[i] = {eval_series_value(packet.geometry.xi[i].re, h),
                 eval_series_value(packet.geometry.xi[i].im, h)};
        xi_r[i] = {eval_series_derivative_value(packet.geometry.xi[i].re, h),
                   eval_series_derivative_value(packet.geometry.xi[i].im, h)};
    }
    std::array<Complex<double>, 5> ell0{};
    std::array<Complex<double>, 5> ell1{};
    Complex<double> ellij[5][5]{};
    for (int i = 0; i < 5; ++i) {
        ell0[i] = xi_r[i] / xi[i];
        ell1[i] = xi_r[i] / Complex<double>{xi[i].re - 1.0, xi[i].im};
        for (int j = i + 1; j < 5; ++j) {
            ellij[i][j] = (xi_r[i] - xi_r[j]) / (xi[i] - xi[j]);
            ellij[j][i] = ellij[i][j];
        }
    }
    const Complex<double> F = state[0];
    Complex<double> M0 = F;
    for (int i = 0; i < 5; ++i)
        M0 = M0 - cscale(state[i + 1], kBeta[i] / kAlpha);
    ValueState rhs{};
    for (int i = 0; i < 5; ++i) {
        const Complex<double> Mi = state[i + 1];
        rhs[0] = rhs[0] + cscale(ell1[i] * (Mi - F), kBeta[i]);
        rhs[i + 1] = rhs[i + 1] -
                     cscale(ell0[i] * (Mi - M0), kAlpha);
        rhs[i + 1] = rhs[i + 1] +
                     cscale(ell1[i] * (Mi - F), kAlpha);
        for (int j = 0; j < 5; ++j) {
            if (j == i) continue;
            rhs[i + 1] = rhs[i + 1] -
                         cscale(ellij[i][j] * (Mi - state[j + 1]),
                                kBeta[j]);
        }
    }
    double e = 0.0;
    for (int i = 0; i < kStateDim; ++i) {
        e = std::max(e, std::hypot(derivative[i].re - rhs[i].re,
                                   derivative[i].im - rhs[i].im) /
                           (1.0 + std::max(std::hypot(derivative[i].re,
                                                       derivative[i].im),
                                           std::hypot(rhs[i].re, rhs[i].im))));
    }
    return e;
}

inline bool projected_packet_gate_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("GM6_PACKET_GATE");
        return e && std::string(e) == "projected";
    }();
    return enabled;
}

template <int Order>
inline void build_pfaffian_coefficients(
    const SeriesGeometry<Order, double>& geometry, const ValueState& seed,
    std::array<Complex<double>, Order + 1> coeff[kStateDim],
    PacketBuildBreakdown* timing = nullptr) {
    using Clock = std::chrono::steady_clock;
    for (int i = 0; i < kStateDim; ++i) coeff[i][0] = seed[i];
    if constexpr (Order == 0) return;
    const auto log_start = timing ? Clock::now() : Clock::time_point{};
    std::array<Complex<double>, Order> ell0[5]{};
    std::array<Complex<double>, Order> ell1[5]{};
    std::array<Complex<double>, Order> ellij[5][5]{};
    for (int i = 0; i < 5; ++i) {
        ell0[i] = log_derivative_series(geometry.xi[i]);
        auto g1 = geometry.xi[i];
        g1.re.c[0] -= 1.0;
        ell1[i] = log_derivative_series(g1);
        for (int j = i + 1; j < 5; ++j) {
            auto gij = geometry.xi[i] - geometry.xi[j];
            ellij[i][j] = log_derivative_series(gij);
            ellij[j][i] = ellij[i][j];
        }
    }
    if (timing) {
        timing->log_derivative_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - log_start).count();
    }
    const auto recurrence_start = timing ? Clock::now() : Clock::time_point{};

    for (int n = 0; n < Order; ++n) {
        ValueState rhs{};
        for (int j = 0; j <= n; ++j) {
            const int k = n - j;
            const Complex<double> F = coeff[0][k];
            Complex<double> M0 = F;
            for (int i = 0; i < 5; ++i)
                M0 = M0 - cscale(coeff[i + 1][k], kBeta[i] / kAlpha);
            for (int i = 0; i < 5; ++i) {
                const Complex<double> Mi = coeff[i + 1][k];
                rhs[0] = rhs[0] +
                         cscale(ell1[i][j] * (Mi - F), kBeta[i]);
                rhs[i + 1] = rhs[i + 1] -
                             cscale(ell0[i][j] * (Mi - M0), kAlpha);
                rhs[i + 1] = rhs[i + 1] +
                             cscale(ell1[i][j] * (Mi - F), kAlpha);
                for (int q = 0; q < 5; ++q) {
                    if (q == i) continue;
                    rhs[i + 1] = rhs[i + 1] -
                                 cscale(ellij[i][q][j] * (Mi - coeff[q + 1][k]),
                                        kBeta[q]);
                }
            }
        }
        for (int i = 0; i < kStateDim; ++i)
            coeff[i][n + 1] = cscale(rhs[i], 1.0 / (n + 1.0));
    }
    if (timing) {
        timing->pfaffian_recurrence_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - recurrence_start).count();
    }
}

template <int Order>
inline bool packet_quality(TaylorPacket<Order>& packet) {
    if constexpr (Order < 2) {
        packet.tail = 0.0;
        packet.strict_tail = 0.0;
        return packet.ok;
    } else {
        const double span = 0.98 *
            std::min(packet.center - packet.lo, packet.hi - packet.center);
        const auto full_p = eval_state(packet, span, Order);
        const auto low_p = eval_state(packet, span, Order - 2);
        const auto full_m = eval_state(packet, -span, Order);
        const auto low_m = eval_state(packet, -span, Order - 2);
        packet.strict_tail = std::max(state_tail<Order>(full_p, low_p),
                                      state_tail<Order>(full_m, low_m));
        packet.tail = packet.strict_tail;
        if (!projected_packet_gate_enabled()) {
            return packet.ok && std::isfinite(packet.tail) &&
                   packet.tail < 2e-11;
        }
        packet.projected_tail = std::max(
            packet_projected_tail_at(packet, span, Order - 2),
            packet_projected_tail_at(packet, -span, Order - 2));
        packet.pfaffian_residual = std::max(
            packet_pfaffian_residual_at(packet, span),
            packet_pfaffian_residual_at(packet, -span));
        const double divisor = std::min(
            packet_divisor_distance_at(packet, span, Order),
            packet_divisor_distance_at(packet, -span, Order));
        packet.tail = packet.projected_tail;
        return packet.ok && std::isfinite(packet.tail) &&
               packet.tail < 2e-11 && std::isfinite(packet.pfaffian_residual) &&
               packet.pfaffian_residual < 1e-8 && std::isfinite(divisor) &&
               divisor > 1e-14;
    }
}

template <int Order>
inline bool build_packet(double center, double lo, double hi, double tlo,
                         double thi, const Params<double>& params,
                         const ValueState& seed, TaylorPacket<Order>& out,
                         PacketBuildBreakdown* timing = nullptr) {
    using Clock = std::chrono::steady_clock;
    using Series = GmSeries<Order, double>;
    const auto build_start = timing ? Clock::now() : Clock::time_point{};
    auto finish = [&](bool ok) {
        if (timing) {
            timing->total_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - build_start).count();
        }
        return ok;
    };
    out = TaylorPacket<Order>{};
    out.center = center;
    out.lo = lo;
    out.hi = hi;
    const auto endpoint_start = timing ? Clock::now() : Clock::time_point{};
    Series R(center);
    if constexpr (Order >= 1) R.c[1] = 1.0;
    const Params<Series> ps{Series(params.a), Series(params.m0),
                            Series(params.X), Series(params.Y),
                            Series(params.rho)};
    const auto P = boundary_P_radius_series<Order>(center, ps);
    // `boundary_P_radius_series` is a degree-six polynomial in the packet
    // variable h even when the state order is 20.  Avoid reading the known
    // zero high coefficients during every implicit-root residual step.
    const bool roots_ok = root_series(P, tlo, out.t_lo, 6) &&
                          root_series(P, thi, out.t_hi, 6);
    if (timing) {
        timing->endpoint_root_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - endpoint_start).count();
    }
    if (!roots_ok) return finish(false);
    const auto geometry_start = timing ? Clock::now() : Clock::time_point{};
    out.geometry = geometry_series(R, out.t_lo, out.t_hi, ps, P);
    if (timing) {
        timing->algebraic_geometry_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - geometry_start).count();
    }
    if (!out.geometry.ok) return finish(false);
    build_pfaffian_coefficients(out.geometry, seed, out.coeff, timing);
    for (int i = 0; i < kStateDim; ++i)
        for (const auto& c : out.coeff[i])
            if (!std::isfinite(c.re) || !std::isfinite(c.im))
                return finish(false);
    out.ok = true;
    const auto quality_start = timing ? Clock::now() : Clock::time_point{};
    packet_quality(out);
    if (timing) {
        timing->quality_tail_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - quality_start).count();
    }
    return finish(true);
}

struct NodeResult {
    bool ok = false;
    double value = 0.0;
    std::array<double, 5> deriv{};
    double phase_error = 0.0;
    double value_ms = 0.0;
    double jacobian_ms = 0.0;
};

inline Dual5 endpoint_dual(double R, double t, const PrimaryFrame& pf,
                           bool chart2) {
    const double theta = 2.0 * std::atan(t) + (chart2 ? kPi : 0.0);
    const PhiGrad g = phi_grad(R, theta, pf);
    Dual5 out(t);
    if (!std::isfinite(g.dphi_dtheta) ||
        std::fabs(g.dphi_dtheta) < 1e-13)
        return Dual5(std::numeric_limits<double>::quiet_NaN());
    const double dt_dtheta = 0.5 * (1.0 + t * t);
    for (int j = 0; j < 5; ++j)
        out.deriv[j] = dt_dtheta * (-g.dP[j] / g.dphi_dtheta);
    return out;
}

inline double xi_distance(const Complex<double>& a,
                          const Complex<double>& b) {
    return std::hypot(a.re - b.re, a.im - b.im);
}

inline void align_s_roots(Geometry<Dual5>& geometry,
                          const std::array<Complex<double>, 5>& expected) {
    const double keep = xi_distance(
        {geometry.xi[0].re.value, geometry.xi[0].im.value}, expected[0]) +
        xi_distance({geometry.xi[1].re.value, geometry.xi[1].im.value},
                    expected[1]);
    const double swap = xi_distance(
        {geometry.xi[0].re.value, geometry.xi[0].im.value}, expected[1]) +
        xi_distance({geometry.xi[1].re.value, geometry.xi[1].im.value},
                    expected[0]);
    if (swap < keep) std::swap(geometry.xi[0], geometry.xi[1]);
}

template <int Order>
inline NodeResult evaluate_node(const TaylorPacket<Order>& packet,
                                double R, const PrimaryFrame& pf,
                                bool chart2, bool with_jacobian) {
    using Clock = std::chrono::steady_clock;
    NodeResult out;
    const auto t0 = Clock::now();
    const double h = R - packet.center;
    const auto state = eval_state(packet, h);
    const auto expected_xi = eval_xi(packet, h);
    const double l = eval_root(packet.t_lo, h);
    const double r = eval_root(packet.t_hi, h);
    const Params<double> pv = params_for_chart<double>(
        pf, chart2);
    const Geometry<double> gv = geometry_scalar(R, l, r, pv);
    if (!gv.ok) return out;
    const Complex<double> value_c = gv.C * state[0];
    out.value = value_c.re;
    out.phase_error = std::fabs(value_c.im) /
                      (1.0 + std::fabs(value_c.re));
    if (!std::isfinite(out.value) || !std::isfinite(out.phase_error) ||
        out.phase_error > 2e-7)
        return out;
    out.value_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t0).count();
    if (!with_jacobian) {
        out.ok = true;
        return out;
    }

    const auto tj = Clock::now();
    const Dual5 ld = endpoint_dual(R, l, pf, chart2);
    const Dual5 rd = endpoint_dual(R, r, pf, chart2);
    if (!scalar_finite(ld) || !scalar_finite(rd)) {
        out.jacobian_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - tj).count();
        return out;
    }
    const Params<Dual5> pd = dual_params_for_chart(pf, chart2);
    Geometry<Dual5> gd = geometry_scalar(Dual5(R), ld, rd, pd);
    if (!gd.ok) {
        out.jacobian_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - tj).count();
        return out;
    }
    align_s_roots(gd, expected_xi);
    const Complex<double> F = state[0];
    std::array<Complex<double>, 5> xi{};
    for (int i = 0; i < 5; ++i) xi[i] = expected_xi[i];
    for (int j = 0; j < 5; ++j) {
        const Complex<double> dC{gd.C.re.deriv[j], gd.C.im.deriv[j]};
        out.deriv[j] = (dC * F).re;
    }
    for (int i = 0; i < 5; ++i) {
        const Complex<double> Mi = state[i + 1];
        const Complex<double> dF_dxi =
            cscale((F - Mi) /
                       (Complex<double>{1.0 - xi[i].re, -xi[i].im}),
                   kBeta[i]);
        for (int j = 0; j < 5; ++j) {
            const Complex<double> dxi{gd.xi[i].re.deriv[j],
                                      gd.xi[i].im.deriv[j]};
            const Complex<double> dF = dF_dxi * dxi;
            const Complex<double> Cvalue{gd.C.re.value, gd.C.im.value};
            out.deriv[j] += (Cvalue * dF).re;
        }
    }
    out.jacobian_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - tj).count();
    for (double x : out.deriv)
        if (!std::isfinite(x)) return NodeResult{};
    out.ok = true;
    return out;
}

// The fold representation already transports the physical arc observable
// itself.  Dense output therefore evaluates one scalar series (and, when
// requested, its five coefficient-series derivatives) without rebuilding
// endpoint geometry or the six-state PF6 projection at each node.
template <int Order>
inline NodeResult evaluate_fold_node(const FoldPacket<Order>& packet,
                                     double R, bool with_jacobian) {
    using Clock = std::chrono::steady_clock;
    NodeResult out;
    const auto t0 = Clock::now();
    const double h = R - packet.center;
    out.value = eval_series_value(packet.coeff, h);
    out.phase_error = 0.0;
    out.value_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t0).count();
    if (!std::isfinite(out.value)) return out;
    if (!with_jacobian) {
        out.ok = true;
        return out;
    }
    const auto tj = Clock::now();
    for (int j = 0; j < 5; ++j) {
        out.deriv[j] = eval_series_value(packet.deriv[j], h);
        if (!std::isfinite(out.deriv[j])) {
            out.jacobian_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - tj).count();
            return out;
        }
    }
    out.jacobian_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - tj).count();
    out.ok = true;
    return out;
}

template <int Order>
struct Block {
    double lo = 0.0;
    double hi = 0.0;
    TaylorPacket<Order> packet{};
};

template <int Order>
inline bool cover_interval(const TaylorPacket<Order>& source,
                           double lo, double hi, const Params<double>& params,
                           int depth, std::vector<Block<Order>>& blocks,
                           int& constructions,
                           PacketProfile* profile = nullptr,
                           PacketArcProfile* arc_profile = nullptr,
                           bool reuse_source = false,
                           const PacketBuildBreakdown* source_timing = nullptr,
                           const std::vector<double>* node_radii = nullptr) {
    if (!(hi > lo)) return true;
    if (node_radii) {
        bool has_node = false;
        for (double R : *node_radii) {
            if (R >= lo && R <= hi) {
                has_node = true;
                break;
            }
        }
        if (!has_node) return true;
    }
    const double center = 0.5 * (lo + hi);
    TaylorPacket<Order> candidate;
    PacketBuildBreakdown timing;
    const bool centered_source = reuse_source &&
        std::fabs(source.center - center) <=
            4.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::fabs(center));
    bool built = false;
    PacketBuildBreakdown* build_timing_ptr = nullptr;
    const PacketBuildBreakdown* record_timing_ptr = nullptr;
    if (centered_source) {
        candidate = source;
        record_timing_ptr = source_timing;
        built = true;
    } else {
        const double h = center - source.center;
        const ValueState seed = eval_state(source, h);
        const double tlo = eval_root(source.t_lo, h);
        const double thi = eval_root(source.t_hi, h);
        ++constructions;
        build_timing_ptr = profile && profile->timing_enabled ? &timing : nullptr;
        record_timing_ptr = build_timing_ptr;
        if (!build_packet(center, lo, hi, tlo, thi, params, seed, candidate,
                          build_timing_ptr)) {
            record_packet_build(profile, arc_profile, record_timing_ptr,
                                false, false, false, depth);
            return false;
        }
        built = true;
    }

    const bool accepted = candidate.tail < 2e-11;
    if (accepted) {
        record_packet_build(profile, arc_profile, record_timing_ptr, built,
                            true, false, depth);
        blocks.push_back({lo, hi, std::move(candidate)});
        return true;
    }
    if (depth >= 16 || blocks.size() > 4096) {
        record_packet_build(profile, arc_profile, record_timing_ptr, built,
                            false, true, depth);
        return false;
    }
    record_packet_build(profile, arc_profile, record_timing_ptr, built, false,
                        true, depth);
    const double mid = center;
    return cover_interval(candidate, lo, mid, params, depth + 1, blocks,
                          constructions, profile, arc_profile, false, nullptr,
                          node_radii) &&
           cover_interval(candidate, mid, hi, params, depth + 1, blocks,
                          constructions, profile, arc_profile, false, nullptr,
                          node_radii);
}

template <int Order>
inline const TaylorPacket<Order>* find_block(
    const std::vector<Block<Order>>& blocks, double R) {
    for (const auto& b : blocks)
        if (R >= b.lo && (R <= b.hi || &b == &blocks.back())) return &b.packet;
    return nullptr;
}

template <int Order>
inline void record_packet_node_coverage(
    PacketProfile* profile, PacketArcProfile* arc,
    const std::vector<Block<Order>>& blocks,
    const std::vector<double>& radii) {
    if (!profile && !arc) return;
    for (const auto& block : blocks) {
        int count = 0;
        for (double R : radii)
            if (R >= block.lo && R <= block.hi)
                ++count;
        if (profile) {
            const int h = std::max(0, std::min(
                count, (int)profile->accepted_node_hist.size() - 1));
            ++profile->accepted_node_hist[h];
        }
        if (arc) {
            arc->accepted_node_sum += count;
            if (arc->accepted_node_min == 0 || count < arc->accepted_node_min)
                arc->accepted_node_min = count;
            arc->accepted_node_max = std::max(arc->accepted_node_max, count);
        }
    }
}

inline std::array<double, 5> internal_to_user(
    const std::array<double, 5>& d, const LensParams& p) {
    const double inv = 1.0 / (1.0 + p.q);
    const double dm0_dq = -inv * inv;
    const double dm1_dq = inv * inv;
    std::array<double, 5> out{d[0], d[1], d[2], 0.0, 0.0};
    if (p.barycentric) {
        out[3] = d[3] * dm0_dq + d[0] * (p.a * dm1_dq);
        out[4] = d[4] + d[0] * (p.q * inv);
    } else {
        out[3] = d[3] * dm0_dq;
        out[4] = d[4];
    }
    return out;
}

}  // namespace gm_l6_detail

using GmLauricella6PacketProfile = gm_l6_detail::PacketProfile;
using GmLauricella6ArcProfile = gm_l6_detail::PacketArcProfile;

struct GmLauricella6Cost {
    double total_ms = 0.0;
    double topology_ms = 0.0;
    double arc_geometry_ms = 0.0;
    double f0_arc_geometry_ms = 0.0;
    double f0_endpoint_ms = 0.0;
    double arc_prepare_ms = 0.0;
    double algebraic_geometry_ms = 0.0;
    double physical_seed_ms = 0.0;
    double connection_ms = 0.0;
    double packet_cover_ms = 0.0;
    double transport_ms = 0.0;
    double node_loop_ms = 0.0;
    double analytic_jacobian_ms = 0.0;
    double physical_fallback_ms = 0.0;
    double fold_packet_ms = 0.0;
    double fold_pair_series_ms = 0.0;
    double fold_smooth_series_ms = 0.0;
    double fold_quality_ms = 0.0;
    double fold_packet_evaluation_ms = 0.0;
    double unclassified_overhead_ms = 0.0;
    int cells = 0;
    int arcs = 0;
    int blocks = 0;
    int nodes = 0;
    int transported = 0;
    int connection_constructions = 0;
    int physical_seeds = 0;
    int fold_packet_constructions = 0;
    int fold_packet_accepted = 0;
    int fold_packet_rejected = 0;
    int fold_blocks = 0;
    int fold_nodes = 0;
    int precision_promotions = 0;
    int precision_failures = 0;
    int physical_fallback = 0;
    int reseeds = 0;
    int jacobian_nodes = 0;
    int jacobian_failed = 0;
    int chart_failed = 0;
    int branch_failed = 0;
    int chart_switches = 0;
    int geometry_failed = 0;
    int seed_failed = 0;
    int connection_failed = 0;
    int coverage_failed = 0;
    int node_failed = 0;
    std::vector<std::array<double, 2>> coverage_failed_ranges;
    GmLauricella6PacketProfile packet_profile{};
    std::vector<GmLauricella6ArcProfile> packet_arcs;
};

struct GmLauricella6Epoch {
    double F0 = 0.0;
    double F_half = 0.0;
    double mu = 0.0;
    double r_max = 0.0;
    std::array<double, 5> grad_mu{};
    Status status = Status::OK;
    Status local_status = Status::OK;
    bool all_true_transport = false;
    GmLauricella6Cost cost{};
};

template <int Order = 12>
inline GmLauricella6Epoch gm_lauricella6_epoch(
    const LensParams& p, double u, int n_r,
    std::vector<Cplx<__float128>>* warm_d14 = nullptr,
    bool with_jacobian = false) {
    using namespace gm_l6_detail;
    using Clock = std::chrono::steady_clock;
    GmLauricella6Epoch out;
    const auto epoch_started = Clock::now();
    if (n_r < 4 || !(p.rho > 0.0) || !(p.a > 0.0) || !(p.q >= 0.0)) {
        out.status = Status::BASIS_DEGENERATE;
        out.local_status = out.status;
        return out;
    }
    out.cost.packet_profile.timing_enabled = packet_profile_enabled();
    const PrimaryFrame pf = PrimaryFrame::from(p);
    const auto ttop = Clock::now();
    std::vector<Cplx<__float128>> roots;
    TopologyResult topo = classify_cells(
        pf, (warm_d14 && !warm_d14->empty()) ? warm_d14 : nullptr, &roots);
    if (warm_d14) *warm_d14 = std::move(roots);
    out.cost.topology_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - ttop).count();
    out.r_max = topo.r_max;
    out.status = topo.status;
    Cheb1Dyn rr(n_r);
    std::array<double, 5> dF0{}, dFh{};
    bool all = topo.status == Status::OK;

    for (const auto& cell : topo.cells) {
        ++out.cost.cells;
        const double width = cell.r_hi - cell.r_lo;
        if (!(width > 0.0)) {
            all = false;
            out.status = Status::EVENT_UNRESOLVED;
            continue;
        }
        const double inset = 1e-9 * width;
        const double lo = cell.r_lo + inset;
        const double hi = cell.r_hi - inset;
        const double center = 0.5 * (lo + hi);
        const double half = 0.5 * (hi - lo);
        const double node_lo = center + half *
            *std::min_element(rr.x.begin(), rr.x.end());
        const double node_hi = center + half *
            *std::max_element(rr.x.begin(), rr.x.end());
        std::vector<double> node_radii(n_r);
        for (int n = 0; n < n_r; ++n)
            node_radii[n] = center + half * rr.x[n];
        if (cell.kind == ArcKind::kEmpty) continue;
        if (cell.kind == ArcKind::kFull) {
            // A full circle has no finite endpoint pair.  Keep this explicit
            // physical-reference bucket instead of pretending it is a PF6
            // transport interval.
            ++out.cost.physical_fallback;
            all = false;
            out.status = Status::LOCAL_REFERENCE_USED;
            for (int n = 0; n < n_r; ++n) {
                const double R = center + half * rr.x[n];
                const double wk = half * rr.w[n];
                out.F0 += wk * R * gm_l6_detail::kTwoPi;
                if (u != 0.0) {
                    double fh = 0.0;
                    const auto ta = Clock::now();
                    bool ok = true;
                    constexpr int M = 512;
                    for (int k = 0; k < M; ++k) {
                        const double th = gm_l6_detail::kTwoPi * (k + 0.5) / M;
                        const double ph = phi_lens(R, th, pf);
                        if (!(ph > 0.0)) { ok = false; break; }
                        fh += std::sqrt(ph);
                    }
                    if (ok) {
                        fh *= R * gm_l6_detail::kTwoPi / M;
                        out.F_half += wk * fh;
                    } else {
                        out.status = Status::ARC_TOPOLOGY_INVALID;
                    }
                    out.cost.physical_fallback_ms +=
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - ta).count();
                }
            }
            continue;
        }
        if (cell.kind != ArcKind::kArcs || cell.status != Status::OK) {
            all = false;
            out.status = Status::TOPOLOGY_UNCERTAIN;
            continue;
        }

        const auto ta = Clock::now();
        const ArcSet center_arcs = arc_intervals(center, pf);
        out.cost.arc_geometry_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - ta).count();
        if (center_arcs.kind != ArcKind::kArcs || center_arcs.arcs.empty()) {
            all = false;
            out.status = Status::ARC_TOPOLOGY_INVALID;
            continue;
        }

        // F0 is an endpoint-geometry term.  It is kept in the same whole
        // epoch measurement, but never feeds the PF6 state.
        for (int n = 0; n < n_r; ++n) {
            const double R = center + half * rr.x[n];
            const double wk = half * rr.w[n];
            const auto tf0_geometry = Clock::now();
            const ArcSet as = arc_intervals(R, pf);
            out.cost.f0_arc_geometry_ms +=
                std::chrono::duration<double, std::milli>(
                    Clock::now() - tf0_geometry).count();
            if (as.kind != ArcKind::kArcs) {
                all = false;
                out.status = Status::ARC_TOPOLOGY_INVALID;
                continue;
            }
            for (const auto& arc : as.arcs) {
                const auto tf0_endpoint = Clock::now();
                const auto pe = polish_endpoint(R, arc[0], pf);
                const auto pl = polish_endpoint(R, arc[1], pf);
                double end = pl.theta;
                if (end <= pe.theta) end += gm_l6_detail::kTwoPi;
                out.F0 += wk * R * (end - pe.theta);
                if (with_jacobian) {
                    const auto ge = phi_val_dP(R, pe.theta, pf);
                    const auto gl = phi_val_dP(R, end, pf);
                    if (!pe.reliable || !pl.reliable ||
                        std::fabs(pe.dphi_dtheta) < 1e-13 ||
                        std::fabs(pl.dphi_dtheta) < 1e-13) {
                        all = false;
                        out.status = Status::GRADIENT_UNRELIABLE;
                    } else {
                        for (int j = 0; j < 5; ++j)
                            dF0[j] += wk * R *
                                      (ge.dP[j] / pe.dphi_dtheta -
                                       gl.dP[j] / pl.dphi_dtheta);
                    }
                }
                out.cost.f0_endpoint_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - tf0_endpoint).count();
            }
        }
        if (u == 0.0) continue;

        const bool fold_candidate = cell_touches_physical_fold(
            cell, topo.events);
        for (const auto& arc : center_arcs.arcs) {
            PacketArcProfile arc_profile;
            arc_profile.arc_index = out.cost.arcs;
            arc_profile.cell_index = cell.index;
            arc_profile.cell_lo = cell.r_lo;
            arc_profile.cell_hi = cell.r_hi;
            arc_profile.radial_center = center;
            arc_profile.theta_lo = arc[0];
            arc_profile.theta_hi = arc[1];
            classify_packet_arc(arc_profile, topo.events);
            auto finish_arc_profile = [&] {
                out.cost.packet_arcs.push_back(std::move(arc_profile));
            };
            ++out.cost.arcs;
            const auto tpreferred = Clock::now();
            const ArcSeed preferred = prepare_arc(pf, arc);
            out.cost.arc_prepare_ms +=
                std::chrono::duration<double, std::milli>(
                    Clock::now() - tpreferred).count();
            const bool first_chart = preferred.ok ? preferred.chart2 : false;
            ArcSeed seed_geometry;
            Params<double> params{};
            SeedResult seed;
            bool seeded = false;
            bool fold_have = false;
            bool fold_served = false;
            FoldPacket<Order> fold_packet;
            double fold_lo = 0.0;
            double fold_hi = 0.0;
            int fold_node_count = 0;
            auto try_fold_side_packet = [&](const ArcSeed& ref,
                                            const FoldSideCertificate& cert) {
                if (!cert.candidate || !fold_packet_enabled()) return false;
                std::vector<double> ordered;
                for (double R : node_radii)
                    if (R >= lo && R <= hi) ordered.push_back(R);
                if (ordered.empty()) return false;
                std::sort(ordered.begin(), ordered.end());
                if (!cert.lower) std::reverse(ordered.begin(), ordered.end());

                const int total_nodes = (int)ordered.size();
                const double target_mid = 0.5 *
                    (ref.theta_lo + ref.theta_hi);
                const int count = std::min(total_nodes,
                                           fold_packet_node_limit());
                const double edge = cert.lower ? lo : hi;
                const double cut = count >= total_nodes
                    ? (cert.lower ? hi : lo)
                    : 0.5 * (ordered[count - 1] + ordered[count]);
                const double plo = std::min(edge, cut);
                const double phi = std::max(edge, cut);
                const double pc = 0.5 * (plo + phi);
                if (!(phi > plo)) return false;
                const ArcSet side_center_arcs = arc_intervals(pc, pf);
                if (side_center_arcs.kind != ArcKind::kArcs ||
                    side_center_arcs.arcs.empty())
                    return false;
                const std::array<double, 2>* side_arc = nullptr;
                double best_angle = std::numeric_limits<double>::infinity();
                for (const auto& candidate_arc : side_center_arcs.arcs) {
                    double enter = candidate_arc[0];
                    double leave = candidate_arc[1];
                    if (leave <= enter) leave += gm_l6_detail::kTwoPi;
                    const double midpoint = 0.5 * (enter + leave);
                    const double distance = arc_angle_distance(
                        midpoint, target_mid);
                    if (distance < best_angle) {
                        best_angle = distance;
                        side_arc = &candidate_arc;
                    }
                }
                if (!side_arc) return false;
                ArcSeed side_seed = prepare_arc_chart(
                    pf, *side_arc, ref.chart2);
                if (!side_seed.ok || !certify_arc(
                        pc, pf, *side_arc, side_seed))
                    return false;
                const auto side_params = params_for_chart<double>(
                    side_seed.chart_pf, false);
                ++out.cost.fold_packet_constructions;
                ++arc_profile.fold_constructions;
                FoldPacket<Order> candidate;
                FoldPacketBuildBreakdown timing;
                const bool ok = build_fold_packet(
                    pc, plo, phi, side_seed.t_lo, side_seed.t_hi,
                    side_params, pf, side_seed.chart2, with_jacobian,
                    candidate, &timing);
                out.cost.fold_packet_ms += timing.total_ms;
                out.cost.fold_pair_series_ms += timing.pair_series_ms;
                out.cost.fold_smooth_series_ms += timing.smooth_series_ms;
                out.cost.fold_quality_ms += timing.quality_ms;
                arc_profile.fold_construction_ms += timing.total_ms;
                if (!ok) {
                    ++out.cost.fold_packet_rejected;
                    ++arc_profile.fold_rejected;
                    arc_profile.fold_failure_reason = candidate.failure_reason;
                    arc_profile.fold_taylor_tail = candidate.taylor_tail;
                    arc_profile.fold_moment_tail = candidate.moment_tail;
                    arc_profile.fold_pair_residual = candidate.pair_residual;
                    return false;
                }
                ++out.cost.fold_packet_accepted;
                ++arc_profile.fold_accepted;
                fold_packet = std::move(candidate);
                fold_lo = plo;
                fold_hi = phi;
                fold_node_count = count;
                arc_profile.fold_taylor_tail = fold_packet.taylor_tail;
                arc_profile.fold_moment_tail = fold_packet.moment_tail;
                arc_profile.fold_pair_residual = fold_packet.pair_residual;
                return true;
            };
            auto consume_fold_packet = [&](const FoldPacket<Order>& packet,
                                           const ArcSeed& cand,
                                           double packet_lo,
                                           double packet_hi) {
                ++out.cost.fold_blocks;
                arc_profile.fold_taylor_tail = packet.taylor_tail;
                arc_profile.fold_moment_tail = packet.moment_tail;
                arc_profile.fold_pair_residual = packet.pair_residual;
                arc_profile.chart2 = cand.chart2 ? 1 : 0;
                const auto tnodes = Clock::now();
                for (int n = 0; n < n_r; ++n) {
                    const double R = center + half * rr.x[n];
                    if (R < packet_lo || R > packet_hi) continue;
                    const double wk = half * rr.w[n];
                    ++out.cost.nodes;
                    ++out.cost.fold_nodes;
                    const NodeResult node = evaluate_fold_node(
                        packet, R, with_jacobian);
                    out.cost.fold_packet_evaluation_ms += node.value_ms;
                    out.cost.analytic_jacobian_ms += node.jacobian_ms;
                    if (!node.ok) {
                        if (arc_profile.node_failed == 0) {
                            arc_profile.first_failed_R = R;
                            arc_profile.first_failed_packet_center =
                                packet.center;
                            arc_profile.first_failed_packet_lo = packet.lo;
                            arc_profile.first_failed_packet_hi = packet.hi;
                            arc_profile.first_failed_phase_error =
                                node.phase_error;
                            arc_profile.first_failed_packet_tail =
                                packet.taylor_tail;
                        }
                        ++arc_profile.node_failed;
                        ++out.cost.node_failed;
                        all = false;
                        out.status = with_jacobian
                            ? Status::GRADIENT_UNRELIABLE
                            : Status::TRANSPORT_TOLERANCE_FAILED;
                        if (with_jacobian) ++out.cost.jacobian_failed;
                        continue;
                    }
                    ++out.cost.transported;
                    out.F_half += wk * node.value;
                    if (with_jacobian) {
                        ++out.cost.jacobian_nodes;
                        for (int j = 0; j < 5; ++j)
                            dFh[j] += wk * node.deriv[j];
                    }
                }
                out.cost.node_loop_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - tnodes).count();
            };
            for (int attempt = 0; attempt < 2 && !seeded; ++attempt) {
                if (attempt != 0) {
                    // A failed first-chart seed must not leave a fold packet
                    // from that chart attached to the second-chart attempt.
                    fold_have = false;
                    fold_node_count = 0;
                    fold_lo = 0.0;
                    fold_hi = 0.0;
                }
                const bool chart2 = attempt == 0 ? first_chart : !first_chart;
                const auto tprepare = Clock::now();
                ArcSeed cand = (attempt == 0 && preferred.ok)
                    ? preferred
                    : prepare_arc_chart(pf, arc, chart2);
                const bool cand_certified = cand.ok &&
                    certify_arc(center, pf, arc, cand);
                out.cost.arc_prepare_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - tprepare).count();
                if (!cand.ok) continue;
                if (!cand_certified) {
                    ++out.cost.branch_failed;
                    continue;
                }
                const Params<double> cand_params =
                    params_for_chart<double>(cand.chart_pf, false);
                FoldSideCertificate fold_cert;
                if (fold_candidate)
                    fold_cert = fold_side_certificate(
                        cell, topo.events, center, pf, cand);
                arc_profile.fold_v_center = fold_cert.v_center;
                arc_profile.fold_v_side = fold_cert.v_side;
                arc_profile.fold_v_ratio = fold_cert.ratio;
                arc_profile.fold_event_radius = fold_cert.event_radius;
                if (fold_cert.candidate && fold_packet_enabled())
                    fold_have = try_fold_side_packet(cand, fold_cert);
                if (fold_have && fold_node_count == n_r) {
                    consume_fold_packet(fold_packet, cand, fold_lo, fold_hi);
                    fold_served = true;
                    break;
                }
                const auto tg = Clock::now();
                const Geometry<double> cand_geometry = geometry_scalar(
                    center, cand.t_lo, cand.t_hi, cand_params);
                out.cost.algebraic_geometry_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - tg).count();
                if (cand_geometry.ok)
                    arc_profile.xi_min_divisor = xi_divisor_distance(cand_geometry);
                SeedResult cand_seed;
                if (cand_geometry.ok) {
                    const auto ts = Clock::now();
                    cand_seed = physical_seed(cand_geometry, 512);
                    out.cost.physical_seed_ms +=
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - ts).count();
                    ++out.cost.physical_seeds;
                }
                bool cand_ok = cand_geometry.ok && cand_seed.ok;
                if (!cand_ok) {
                    ++out.cost.precision_promotions;
                    const auto thg = Clock::now();
                    const Params<long double> high_params =
                        params_for_chart<long double>(cand.chart_pf, false);
                    const Geometry<long double> high_geometry = geometry_scalar(
                        (long double)center, (long double)cand.t_lo,
                        (long double)cand.t_hi, high_params);
                    out.cost.algebraic_geometry_ms +=
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - thg).count();
                    if (high_geometry.ok) {
                        const auto ths = Clock::now();
                        const SeedResult high_seed =
                            physical_seed_long_double(high_geometry, 512);
                        out.cost.physical_seed_ms +=
                            std::chrono::duration<double, std::milli>(
                                Clock::now() - ths).count();
                        ++out.cost.physical_seeds;
                        if (high_seed.ok) {
                            cand_seed = high_seed;
                            cand_ok = true;
                        }
                    }
                }
                if (!cand_geometry.ok && !cand_ok) {
                    ++out.cost.geometry_failed;
                }
                if (!cand_ok) {
                    ++out.cost.precision_failures;
                    ++out.cost.seed_failed;
                    continue;
                }
                if (attempt != 0) ++out.cost.chart_switches;
                seed_geometry = cand;
                params = cand_params;
                seed = cand_seed;
                seeded = true;
            }
            if (fold_served) {
                finish_arc_profile();
                continue;
            }
            if (!seeded) {
                ++out.cost.chart_failed;
                all = false;
                out.status = Status::BASIS_DEGENERATE;
                finish_arc_profile();
                continue;
            }
            arc_profile.chart2 = seed_geometry.chart2 ? 1 : 0;
            if (fold_have)
                consume_fold_packet(fold_packet, seed_geometry,
                                    fold_lo, fold_hi);
            std::vector<double> pf6_node_radii;
            pf6_node_radii.reserve(n_r);
            for (double R : node_radii)
                if (!fold_have || R < fold_lo || R > fold_hi)
                    pf6_node_radii.push_back(R);
            if (pf6_node_radii.empty()) {
                finish_arc_profile();
                continue;
            }
            const double pf6_lo = *std::min_element(
                pf6_node_radii.begin(), pf6_node_radii.end());
            const double pf6_hi = *std::max_element(
                pf6_node_radii.begin(), pf6_node_radii.end());
            const auto tc = Clock::now();
            TaylorPacket<Order> initial;
            PacketBuildBreakdown initial_timing;
            PacketBuildBreakdown* initial_timing_ptr =
                out.cost.packet_profile.timing_enabled ? &initial_timing : nullptr;
            const bool initial_ok = build_packet(
                center, node_lo, node_hi, seed_geometry.t_lo, seed_geometry.t_hi,
                params, seed.state.z, initial, initial_timing_ptr);
            ++out.cost.connection_constructions;
            if (!initial_ok) {
                record_packet_build(
                    &out.cost.packet_profile, &arc_profile, initial_timing_ptr,
                    false, false, false, 0);
                ++out.cost.connection_failed;
                all = false;
                out.status = Status::CONNECTION_ILL_CONDITIONED;
                finish_arc_profile();
                continue;
            }
            std::vector<Block<Order>> blocks;
            blocks.reserve(8);
            if (initial.tail < 2e-11) {
                record_packet_build(
                    &out.cost.packet_profile, &arc_profile, initial_timing_ptr,
                    true, true, false, 0);
                blocks.push_back({lo, hi, std::move(initial)});
            } else {
                blocks.clear();
                const auto tcover = Clock::now();
                const bool covered = cover_interval(
                    initial, pf6_lo, pf6_hi, params, 0, blocks,
                    out.cost.connection_constructions,
                    &out.cost.packet_profile, &arc_profile, true,
                    initial_timing_ptr, &pf6_node_radii);
                out.cost.packet_cover_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - tcover).count();
                if (!covered) {
                    ++out.cost.coverage_failed;
                    out.cost.coverage_failed_ranges.push_back(
                        {cell.r_lo, cell.r_hi});
                    all = false;
                    out.status = Status::TRANSPORT_TOLERANCE_FAILED;
                    finish_arc_profile();
                    continue;
                }
            }
            out.cost.blocks += (int)blocks.size();
            record_packet_node_coverage(&out.cost.packet_profile, &arc_profile,
                                        blocks, pf6_node_radii);
            out.cost.connection_ms +=
                std::chrono::duration<double, std::milli>(
                    Clock::now() - tc).count();

            const auto tnodes = Clock::now();
            for (int n = 0; n < n_r; ++n) {
                const double R = center + half * rr.x[n];
                const double wk = half * rr.w[n];
                if (fold_have && R >= fold_lo && R <= fold_hi) continue;
                const TaylorPacket<Order>* packet = find_block(blocks, R);
                ++out.cost.nodes;
                if (!packet) {
                    all = false;
                    out.status = Status::TRANSPORT_TOLERANCE_FAILED;
                    continue;
                }
                const NodeResult node = evaluate_node(
                    *packet, R, pf, seed_geometry.chart2, with_jacobian);
                out.cost.transport_ms += node.value_ms;
                out.cost.analytic_jacobian_ms += node.jacobian_ms;
                if (!node.ok) {
                    if (arc_profile.node_failed == 0 && packet) {
                        arc_profile.first_failed_R = R;
                        arc_profile.first_failed_packet_center = packet->center;
                        arc_profile.first_failed_packet_lo = packet->lo;
                        arc_profile.first_failed_packet_hi = packet->hi;
                        for (const auto& b : blocks) {
                            if (&b.packet == packet) {
                                arc_profile.first_failed_block_lo = b.lo;
                                arc_profile.first_failed_block_hi = b.hi;
                                break;
                            }
                        }
                        arc_profile.first_failed_phase_error = node.phase_error;
                        arc_profile.first_failed_packet_tail = packet->tail;
                    }
                    ++arc_profile.node_failed;
                    ++out.cost.node_failed;
                    all = false;
                    out.status = with_jacobian ? Status::GRADIENT_UNRELIABLE
                                               : Status::TRANSPORT_TOLERANCE_FAILED;
                    if (with_jacobian) ++out.cost.jacobian_failed;
                    continue;
                }
                ++out.cost.transported;
                out.F_half += wk * node.value;
                if (with_jacobian) {
                    ++out.cost.jacobian_nodes;
                    for (int j = 0; j < 5; ++j) dFh[j] += wk * node.deriv[j];
                }
            }
            out.cost.node_loop_ms +=
                std::chrono::duration<double, std::milli>(
                    Clock::now() - tnodes).count();
            finish_arc_profile();
        }
    }
    const double denom = gm_l6_detail::kPi * p.rho * p.rho * (1.0 - u / 3.0);
    if (!(denom > 0.0) || !std::isfinite(denom)) {
        out.status = Status::BASIS_DEGENERATE;
        all = false;
    } else {
        out.mu = ((1.0 - u) * out.F0 + u * out.F_half) / denom;
        if (with_jacobian) {
            std::array<double, 5> di{};
            for (int j = 0; j < 5; ++j)
                di[j] = ((1.0 - u) * dF0[j] + u * dFh[j]) / denom;
            out.grad_mu = internal_to_user(di, p);
            out.grad_mu[2] -= 2.0 * out.mu / p.rho;
        }
    }
    out.local_status = out.status;
    out.all_true_transport = all && out.cost.nodes == out.cost.transported &&
                             out.cost.physical_fallback == 0;
    if (!out.all_true_transport && out.status == Status::OK)
        out.status = Status::TRANSPORT_TOLERANCE_FAILED;
    if (with_jacobian && out.cost.jacobian_failed != 0 &&
        out.status == Status::OK)
        out.status = Status::GRADIENT_UNRELIABLE;
    out.cost.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - epoch_started).count();
    // These timers are disjoint at the epoch level.  The finer transport,
    // Jacobian, packet-cover, and fold sub-timers intentionally remain
    // nested diagnostics and are not subtracted a second time here.
    const double classified_ms =
        out.cost.topology_ms + out.cost.arc_geometry_ms +
        out.cost.f0_arc_geometry_ms + out.cost.f0_endpoint_ms +
        out.cost.arc_prepare_ms + out.cost.algebraic_geometry_ms +
        out.cost.physical_seed_ms + out.cost.connection_ms +
        out.cost.transport_ms + out.cost.fold_packet_ms +
        out.cost.fold_packet_evaluation_ms + out.cost.physical_fallback_ms;
    out.cost.unclassified_overhead_ms = out.cost.total_ms - classified_ms;
    return out;
}

}  // namespace lcbinint::holonomic

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
#include <limits>
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
    const GmSeries<Order, Scalar>& r, const Params<GmSeries<Order, Scalar>>& p) {
    using S = GmSeries<Order, Scalar>;
    using C = Complex<S>;
    SeriesGeometry<Order, Scalar> out;
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
    if (!(out.S_left.c[0] > 0.0) || !(Aleft.c[0] > 0.0) ||
        !(Bleft.c[0] > 0.0) || !(p.rho.c[0] > 0.0)) return out;
    const S S1left = -(d1 + S(2) * P[4] * l);
    const S L = delta * S1left / out.S_left;
    const S M = delta * delta * (-P[4]) / out.S_left;
    const C discr{L * L - S(4) * M, S(0)};
    const C sq = complex_sqrt_series_clean(discr);
    const C zS1{(-L + sq.re) / S(2), sq.im / S(2)};
    const C zS2{(-L - sq.re) / S(2), -sq.im / S(2)};
    const C dz{delta, S(0)};
    const C zstar = dz / C{-l, S(1)};
    const C zaminus = dz / C{-l, S(-1)};
    const C bnum{delta * (R + p.a), S(0)};
    const C zbplus{-l * (R + p.a), R - p.a};
    const C zbminus{-l * (R + p.a), -(R - p.a)};
    out.z = {zS1, zS2, zstar, zaminus, bnum / zbplus, bnum / zbminus};
    out.q = {S(1) - zstar.re, -zstar.im};
    const C den{S(1) - zstar.re, -zstar.im};
    for (int j = 0; j < 5; ++j)
        out.xi[j] = (out.z[j < 2 ? j : j + 1] - zstar) / den;
    out.ok = series_finite(out.v) && series_finite(out.S_left) &&
             complex_series_finite(out.q);
    for (const auto& x : out.xi) out.ok = out.ok && complex_series_finite(x);
    return out;
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
                        double t0, GmSeries<Order, double>& out) {
    out = GmSeries<Order, double>(t0);
    const double pt = poly_derivative_at(p, t0);
    if (!std::isfinite(pt) || std::fabs(pt) < 1e-15) return false;
    for (int n = 1; n <= Order; ++n) {
        // With t_n set to zero, the coefficient of h^n is the known part.
        // The missing linear term is P_t(t0) t_n and is divided out below.
        out.c[n] = 0.0;
        const auto residual = eval_poly_series(p, out);
        out.c[n] = -residual.c[n] / pt;
        if (!std::isfinite(out.c[n])) return false;
    }
    return true;
}

template <int Order>
inline std::array<Complex<double>, Order> log_derivative_series(
    const Complex<GmSeries<Order, double>>& g) {
    std::array<Complex<double>, Order> ell{};
    const Complex<double> g0{g.re.c[0], g.im.c[0]};
    for (int n = 0; n < Order; ++n) {
        Complex<double> rhs{(n + 1 < Order + 1 ? (n + 1) * g.re.c[n + 1]
                                                : 0.0),
                            (n + 1 < Order + 1 ? (n + 1) * g.im.c[n + 1]
                                                : 0.0)};
        for (int k = 1; k <= n; ++k) {
            const Complex<double> gk{g.re.c[k], g.im.c[k]};
            rhs = rhs - gk * ell[n - k];
        }
        ell[n] = rhs / g0;
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
inline void build_pfaffian_coefficients(
    const SeriesGeometry<Order, double>& geometry, const ValueState& seed,
    std::array<Complex<double>, Order + 1> coeff[kStateDim]) {
    for (int i = 0; i < kStateDim; ++i) coeff[i][0] = seed[i];
    if constexpr (Order == 0) return;
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
}

template <int Order>
inline bool packet_quality(TaylorPacket<Order>& packet) {
    if constexpr (Order < 2) {
        packet.tail = 0.0;
        return packet.ok;
    } else {
        const double span = 0.98 *
            std::min(packet.center - packet.lo, packet.hi - packet.center);
        const auto full_p = eval_state(packet, span, Order);
        const auto low_p = eval_state(packet, span, Order - 2);
        const auto full_m = eval_state(packet, -span, Order);
        const auto low_m = eval_state(packet, -span, Order - 2);
        packet.tail = std::max(state_tail<Order>(full_p, low_p),
                               state_tail<Order>(full_m, low_m));
        return packet.ok && std::isfinite(packet.tail) && packet.tail < 2e-11;
    }
}

template <int Order>
inline bool build_packet(double center, double lo, double hi, double tlo,
                         double thi, const Params<double>& params,
                         const ValueState& seed, TaylorPacket<Order>& out) {
    using Series = GmSeries<Order, double>;
    out = TaylorPacket<Order>{};
    out.center = center;
    out.lo = lo;
    out.hi = hi;
    Series R(center);
    if constexpr (Order >= 1) R.c[1] = 1.0;
    const Params<Series> ps{Series(params.a), Series(params.m0),
                            Series(params.X), Series(params.Y),
                            Series(params.rho)};
    const auto P = boundary_P(R, ps);
    if (!root_series(P, tlo, out.t_lo) || !root_series(P, thi, out.t_hi))
        return false;
    out.geometry = geometry_series(R, out.t_lo, out.t_hi, ps);
    if (!out.geometry.ok) return false;
    build_pfaffian_coefficients(out.geometry, seed, out.coeff);
    for (int i = 0; i < kStateDim; ++i)
        for (const auto& c : out.coeff[i])
            if (!std::isfinite(c.re) || !std::isfinite(c.im)) return false;
    out.ok = true;
    packet_quality(out);
    return true;
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
                           int& constructions) {
    const double center = 0.5 * (lo + hi);
    const double h = center - source.center;
    const ValueState seed = eval_state(source, h);
    const double tlo = eval_root(source.t_lo, h);
    const double thi = eval_root(source.t_hi, h);
    TaylorPacket<Order> candidate;
    ++constructions;
    if (!build_packet(center, lo, hi, tlo, thi, params, seed, candidate))
        return false;
    if (candidate.tail < 2e-11) {
        blocks.push_back({lo, hi, std::move(candidate)});
        return true;
    }
    if (depth >= 16 || !(hi > lo) || blocks.size() > 4096) return false;
    // The candidate is still used as a local source.  The two child centres
    // lie in its central region even when the edge tail rejected the parent.
    const double mid = center;
    return cover_interval(candidate, lo, mid, params, depth + 1, blocks,
                          constructions) &&
           cover_interval(candidate, mid, hi, params, depth + 1, blocks,
                          constructions);
}

template <int Order>
inline const TaylorPacket<Order>* find_block(
    const std::vector<Block<Order>>& blocks, double R) {
    for (const auto& b : blocks)
        if (R >= b.lo && (R <= b.hi || &b == &blocks.back())) return &b.packet;
    return nullptr;
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

struct GmLauricella6Cost {
    double topology_ms = 0.0;
    double arc_geometry_ms = 0.0;
    double algebraic_geometry_ms = 0.0;
    double physical_seed_ms = 0.0;
    double connection_ms = 0.0;
    double transport_ms = 0.0;
    double analytic_jacobian_ms = 0.0;
    double physical_fallback_ms = 0.0;
    int cells = 0;
    int arcs = 0;
    int blocks = 0;
    int nodes = 0;
    int transported = 0;
    int connection_constructions = 0;
    int physical_seeds = 0;
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
    if (n_r < 4 || !(p.rho > 0.0) || !(p.a > 0.0) || !(p.q >= 0.0)) {
        out.status = Status::BASIS_DEGENERATE;
        out.local_status = out.status;
        return out;
    }
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
            const ArcSet as = arc_intervals(R, pf);
            if (as.kind != ArcKind::kArcs) {
                all = false;
                out.status = Status::ARC_TOPOLOGY_INVALID;
                continue;
            }
            for (const auto& arc : as.arcs) {
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
            }
        }
        if (u == 0.0) continue;

        for (const auto& arc : center_arcs.arcs) {
            ++out.cost.arcs;
            const ArcSeed preferred = prepare_arc(pf, arc);
            const bool first_chart = preferred.ok ? preferred.chart2 : false;
            ArcSeed seed_geometry;
            Params<double> params{};
            SeedResult seed;
            bool seeded = false;
            for (int attempt = 0; attempt < 2 && !seeded; ++attempt) {
                const bool chart2 = attempt == 0 ? first_chart : !first_chart;
                ArcSeed cand = (attempt == 0 && preferred.ok)
                    ? preferred
                    : prepare_arc_chart(pf, arc, chart2);
                if (!cand.ok) continue;
                if (!certify_arc(center, pf, arc, cand)) {
                    ++out.cost.branch_failed;
                    continue;
                }
                const Params<double> cand_params =
                    params_for_chart<double>(cand.chart_pf, false);
                const auto tg = Clock::now();
                const Geometry<double> cand_geometry = geometry_scalar(
                    center, cand.t_lo, cand.t_hi, cand_params);
                out.cost.algebraic_geometry_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - tg).count();
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
            if (!seeded) {
                ++out.cost.chart_failed;
                all = false;
                out.status = Status::BASIS_DEGENERATE;
                continue;
            }
            const auto tc = Clock::now();
            TaylorPacket<Order> initial;
            const bool initial_ok = build_packet(
                center, node_lo, node_hi, seed_geometry.t_lo, seed_geometry.t_hi,
                params, seed.state.z, initial);
            ++out.cost.connection_constructions;
            if (!initial_ok) {
                ++out.cost.connection_failed;
                all = false;
                out.status = Status::CONNECTION_ILL_CONDITIONED;
                continue;
            }
            std::vector<Block<Order>> blocks;
            blocks.reserve(8);
            blocks.push_back({lo, hi, initial});
            if (initial.tail >= 2e-11) {
                blocks.clear();
                const bool covered = cover_interval(
                    initial, node_lo, node_hi, params, 0, blocks,
                    out.cost.connection_constructions);
                if (!covered) {
                    ++out.cost.coverage_failed;
                    out.cost.coverage_failed_ranges.push_back(
                        {cell.r_lo, cell.r_hi});
                    all = false;
                    out.status = Status::TRANSPORT_TOLERANCE_FAILED;
                    continue;
                }
            }
            out.cost.blocks += (int)blocks.size();
            out.cost.connection_ms +=
                std::chrono::duration<double, std::milli>(
                    Clock::now() - tc).count();

            for (int n = 0; n < n_r; ++n) {
                const double R = center + half * rr.x[n];
                const double wk = half * rr.w[n];
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
    return out;
}

}  // namespace lcbinint::holonomic

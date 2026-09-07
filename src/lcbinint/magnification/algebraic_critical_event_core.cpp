#include "lcbinint/magnification/algebraic_critical_event_core.hpp"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_poly.h>

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace lcbinint::magnification::critical_events {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

struct Interval {
    double lo;
    double hi;
};
using IPoly = std::vector<Interval>;  // ascending powers

Interval exact_interval(double value) { return {value, value}; }

Interval iadd(Interval a, Interval b)
{
    return {
        std::nextafter(a.lo + b.lo, -std::numeric_limits<double>::infinity()),
        std::nextafter(a.hi + b.hi, std::numeric_limits<double>::infinity())};
}

Interval ineg(Interval a) { return {-a.hi, -a.lo}; }
Interval isub(Interval a, Interval b) { return iadd(a, ineg(b)); }

Interval imul(Interval a, Interval b)
{
    const double values[4] = {
        a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    double lo = values[0];
    double hi = values[0];
    for (double value : values) {
        lo = std::min(lo, value);
        hi = std::max(hi, value);
    }
    return {
        std::nextafter(lo, -std::numeric_limits<double>::infinity()),
        std::nextafter(hi, std::numeric_limits<double>::infinity())};
}

bool finite_interval(Interval value);

bool positive_square_subset(Interval radius, Interval u)
{
    if (!finite_interval(radius) || !finite_interval(u) ||
        !(radius.lo > 0.0)) return false;
    const Interval squared = imul(radius, radius);
    return finite_interval(squared) && squared.lo >= u.lo && squared.hi <= u.hi;
}

Interval iscale(Interval a, double scale)
{
    return imul(a, exact_interval(scale));
}

bool contains_zero(Interval value)
{
    return value.lo <= 0.0 && value.hi >= 0.0;
}

bool finite_interval(Interval value)
{
    return std::isfinite(value.lo) && std::isfinite(value.hi) &&
        value.lo <= value.hi;
}

double midpoint(Interval value)
{
    return 0.5 * value.lo + 0.5 * value.hi;
}

double absmax(Interval value)
{
    return std::max(std::abs(value.lo), std::abs(value.hi));
}

double absmin(Interval value)
{
    return contains_zero(value)
        ? 0.0
        : std::min(std::abs(value.lo), std::abs(value.hi));
}

Interval idiv(Interval numerator, Interval denominator)
{
    if (contains_zero(denominator)) {
        return {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
    }
    const Interval inverse = {
        std::nextafter(
            1.0 / denominator.hi,
            -std::numeric_limits<double>::infinity()),
        std::nextafter(
            1.0 / denominator.lo,
            std::numeric_limits<double>::infinity())};
    return imul(numerator, inverse);
}

Interval iintersect(Interval a, Interval b)
{
    return {std::max(a.lo, b.lo), std::min(a.hi, b.hi)};
}

void itrim(IPoly* polynomial)
{
    while (polynomial->size() > 1 && polynomial->back().lo == 0.0 &&
           polynomial->back().hi == 0.0) {
        polynomial->pop_back();
    }
}

IPoly iadd_poly(const IPoly& a, const IPoly& b, double scale_b = 1.0)
{
    IPoly out(std::max(a.size(), b.size()), exact_interval(0.0));
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = iadd(out[i], a[i]);
    for (std::size_t i = 0; i < b.size(); ++i) {
        out[i] = iadd(out[i], iscale(b[i], scale_b));
    }
    itrim(&out);
    return out;
}

IPoly iscale_poly(IPoly polynomial, double scale)
{
    for (auto& value : polynomial) value = iscale(value, scale);
    return polynomial;
}

IPoly imul_poly(const IPoly& a, const IPoly& b)
{
    IPoly out(a.size() + b.size() - 1, exact_interval(0.0));
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            out[i + j] = iadd(out[i + j], imul(a[i], b[j]));
        }
    }
    itrim(&out);
    return out;
}

IPoly ipower(IPoly polynomial, int power)
{
    IPoly out{exact_interval(1.0)};
    while (power-- > 0) out = imul_poly(out, polynomial);
    return out;
}

IPoly iproduct(
    std::initializer_list<std::pair<const IPoly*, int>> factors)
{
    IPoly out{exact_interval(1.0)};
    for (const auto& factor : factors) {
        out = imul_poly(out, ipower(*factor.first, factor.second));
    }
    return out;
}

void iadd_term(
    IPoly* out,
    double scale,
    std::initializer_list<std::pair<const IPoly*, int>> factors)
{
    *out = iadd_poly(*out, iproduct(factors), scale);
}

using boost::multiprecision::cpp_int;

struct Dyadic {
    cpp_int numerator = 0;
    int exponent = 0;
    bool valid = true;

    Dyadic() = default;
    explicit Dyadic(long long value) : numerator(value) { normalize(); }

    static Dyadic from_double(double value)
    {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const bool negative = (bits >> 63) != 0;
        const int encoded_exponent = static_cast<int>((bits >> 52) & 0x7ff);
        const std::uint64_t fraction = bits & ((std::uint64_t(1) << 52) - 1);
        Dyadic out;
        if (encoded_exponent == 0x7ff) {
            out.valid = false;
            return out;
        }
        if (encoded_exponent == 0) {
            out.numerator = fraction;
            out.exponent = -1074;
        } else {
            out.numerator = (std::uint64_t(1) << 52) | fraction;
            out.exponent = encoded_exponent - 1023 - 52;
        }
        if (negative) out.numerator = -out.numerator;
        out.normalize();
        return out;
    }

    void normalize()
    {
        if (numerator == 0) {
            exponent = 0;
            return;
        }
        cpp_int magnitude = numerator < 0 ? -numerator : numerator;
        while ((magnitude & 1) == 0) {
            numerator >>= 1;
            magnitude >>= 1;
            ++exponent;
        }
    }
};

Dyadic dadd(Dyadic a, const Dyadic& b)
{
    if (!a.valid || !b.valid) {
        Dyadic invalid;
        invalid.valid = false;
        return invalid;
    }
    const int base = std::min(a.exponent, b.exponent);
    a.numerator <<= (a.exponent - base);
    a.numerator += b.numerator << (b.exponent - base);
    a.exponent = base;
    a.normalize();
    return a;
}

Dyadic dneg(Dyadic value)
{
    if (!value.valid) return value;
    value.numerator = -value.numerator;
    return value;
}

Dyadic dsub(Dyadic a, const Dyadic& b)
{
    return dadd(std::move(a), dneg(b));
}

Dyadic dmul(Dyadic a, const Dyadic& b)
{
    if (!a.valid || !b.valid) {
        Dyadic invalid;
        invalid.valid = false;
        return invalid;
    }
    a.numerator *= b.numerator;
    a.exponent += b.exponent;
    a.normalize();
    return a;
}

bool dcompare(const Dyadic& a, const Dyadic& b, int* result)
{
    if (!a.valid || !b.valid || result == nullptr) return false;
    const int base = std::min(a.exponent, b.exponent);
    const cpp_int an = a.numerator << (a.exponent - base);
    const cpp_int bn = b.numerator << (b.exponent - base);
    *result = an < bn ? -1 : (an > bn ? 1 : 0);
    return true;
}

double dyadic_to_double(const Dyadic& value)
{
    if (!value.valid) return std::numeric_limits<double>::quiet_NaN();
    if (value.numerator == 0) return 0.0;
    const bool negative = value.numerator < 0;
    cpp_int magnitude = negative ? -value.numerator : value.numerator;
    const int bits = boost::multiprecision::msb(magnitude) + 1;
    int shift = bits - 53;
    cpp_int mantissa;
    if (shift > 0) {
        mantissa = magnitude >> shift;
        const cpp_int remainder = magnitude - (mantissa << shift);
        const cpp_int half = cpp_int(1) << (shift - 1);
        if (remainder > half || (remainder == half && (mantissa & 1) != 0)) {
            ++mantissa;
        }
        if (boost::multiprecision::msb(mantissa) + 1 > 53) {
            mantissa >>= 1;
            ++shift;
        }
    } else {
        mantissa = magnitude << (-shift);
    }
    const double scaled = std::ldexp(
        mantissa.convert_to<double>(), value.exponent + shift);
    return negative ? -scaled : scaled;
}

Interval dyadic_adjacent_enclosure(const Dyadic& value)
{
    if (!value.valid) {
        return {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
    }
    const double center = dyadic_to_double(value);
    const double lo = std::nextafter(
        center, -std::numeric_limits<double>::infinity());
    const double hi = std::nextafter(
        center, std::numeric_limits<double>::infinity());
    if (!std::isfinite(center) || !std::isfinite(lo) || !std::isfinite(hi)) {
        return {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
    }
    const Dyadic rounded = Dyadic::from_double(center);
    int comparison = 0;
    if (!dcompare(value, rounded, &comparison)) {
        return {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
    }
    if (comparison == 0) return {center, center};
    int neighbor_comparison = 0;
    if (comparison < 0 &&
        dcompare(value, Dyadic::from_double(lo), &neighbor_comparison) &&
        neighbor_comparison >= 0) {
        return {lo, center};
    }
    if (comparison > 0 &&
        dcompare(value, Dyadic::from_double(hi), &neighbor_comparison) &&
        neighbor_comparison <= 0) {
        return {center, hi};
    }
    return {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};
}

using DPoly = std::vector<Dyadic>;  // ascending powers

void dtrim(DPoly* polynomial)
{
    while (polynomial->size() > 1 && polynomial->back().valid &&
           polynomial->back().numerator == 0) {
        polynomial->pop_back();
    }
}

DPoly dadd_poly(const DPoly& a, const DPoly& b, int sign_b = 1)
{
    DPoly out(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = dadd(out[i], a[i]);
    for (std::size_t i = 0; i < b.size(); ++i) {
        out[i] = dadd(out[i], sign_b > 0 ? b[i] : dneg(b[i]));
    }
    dtrim(&out);
    return out;
}

DPoly dscale_poly(DPoly polynomial, long long scale)
{
    for (auto& value : polynomial) value = dmul(value, Dyadic(scale));
    return polynomial;
}

DPoly dmul_poly(const DPoly& a, const DPoly& b)
{
    DPoly out(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            out[i + j] = dadd(out[i + j], dmul(a[i], b[j]));
        }
    }
    dtrim(&out);
    return out;
}

DPoly dpower(DPoly polynomial, int power)
{
    DPoly out{Dyadic(1)};
    while (power-- > 0) out = dmul_poly(out, polynomial);
    return out;
}

DPoly dproduct(
    std::initializer_list<std::pair<const DPoly*, int>> factors)
{
    DPoly out{Dyadic(1)};
    for (const auto& factor : factors) {
        out = dmul_poly(out, dpower(*factor.first, factor.second));
    }
    return out;
}

void dadd_term(
    DPoly* out,
    long long scale,
    std::initializer_list<std::pair<const DPoly*, int>> factors)
{
    *out = dadd_poly(*out, dscale_poly(dproduct(factors), scale));
}

Dyadic deval(const DPoly& polynomial, Dyadic x)
{
    Dyadic value(0);
    for (auto it = polynomial.rbegin(); it != polynomial.rend(); ++it) {
        value = dadd(dmul(value, x), *it);
    }
    return value;
}

DPoly dderivative(const DPoly& polynomial)
{
    if (polynomial.size() < 2) return {Dyadic(0)};
    DPoly out(polynomial.size() - 1);
    for (std::size_t i = 1; i < polynomial.size(); ++i) {
        out[i - 1] = dmul(polynomial[i], Dyadic(static_cast<long long>(i)));
    }
    return out;
}

std::array<DPoly, 5> dyadic_trig_polynomials(const FrozenGeometry& geometry)
{
    const Dyadic a = Dyadic::from_double(geometry.a);
    const Dyadic m2 = Dyadic::from_double(geometry.m2);
    // sx has already crossed the one authoritative binary64 rounding boundary.
    const Dyadic sx = Dyadic::from_double(geometry.sx);
    const Dyadic sy = Dyadic::from_double(geometry.sy);
    const Dyadic rho = Dyadic::from_double(geometry.rho);
    const Dyadic rho2 = dmul(rho, rho);
    const Dyadic kx = dadd(Dyadic(-1), dmul(a, sx));
    const Dyadic ky = dmul(a, sy);
    const Dyadic zero(0);
    const DPoly A{
        zero, zero, dneg(dmul(dmul(a, m2), sx)), zero, dmul(a, sx)};
    const DPoly B{
        zero, zero, dmul(dmul(a, m2), sy), zero, dneg(dmul(a, sy))};
    const Dyadic c3 = dadd(
        dadd(dneg(dmul(sx, kx)), dneg(dmul(sy, ky))),
        dadd(dmul(a, dsub(m2, kx)), dmul(rho2, a)));
    const DPoly C{
        zero, dmul(dmul(a, m2), kx), zero, c3, zero,
        dneg(dadd(sx, a))};
    const Dyadic d3 = dadd(
        dadd(dmul(sy, kx), dneg(dmul(sx, ky))), dmul(a, ky));
    const DPoly D{
        zero, dneg(dmul(dmul(a, m2), ky)), zero, d3, zero, sy};
    const Dyadic e0 = dmul(dmul(a, a), dmul(m2, m2));
    const Dyadic e2 = dsub(
        dsub(
            dadd(dmul(kx, kx), dmul(ky, ky)),
            dmul(Dyadic(2), dmul(dmul(a, a), m2))),
        dmul(rho2, dmul(a, a)));
    const Dyadic e4 = dadd(
        dadd(
            dadd(dmul(sx, sx), dmul(sy, sy)),
            dmul(Dyadic(2), kx)),
        dsub(dmul(a, a), rho2));
    const DPoly E{e0, zero, e2, zero, e4, zero, Dyadic(1)};
    return {{
        dadd_poly(dadd_poly(E, dscale_poly(C, 2)), dscale_poly(A, 2)),
        dadd_poly(dscale_poly(D, -4), dscale_poly(B, -8)),
        dadd_poly(dscale_poly(E, 2), dscale_poly(A, -12)),
        dadd_poly(dscale_poly(D, -4), dscale_poly(B, 8)),
        dadd_poly(dadd_poly(E, dscale_poly(C, -2)), dscale_poly(A, 2))}};
}

DPoly dyadic_discriminant(const std::array<DPoly, 5>& polynomial)
{
    const DPoly& e = polynomial[0];
    const DPoly& d = polynomial[1];
    const DPoly& c = polynomial[2];
    const DPoly& b = polynomial[3];
    const DPoly& a = polynomial[4];
    DPoly out{Dyadic(0)};
    dadd_term(&out, 256, {{&a, 3}, {&e, 3}});
    dadd_term(&out, -192, {{&a, 2}, {&b, 1}, {&d, 1}, {&e, 2}});
    dadd_term(&out, -128, {{&a, 2}, {&c, 2}, {&e, 2}});
    dadd_term(&out, 144, {{&a, 2}, {&c, 1}, {&d, 2}, {&e, 1}});
    dadd_term(&out, -27, {{&a, 2}, {&d, 4}});
    dadd_term(&out, 144, {{&a, 1}, {&b, 2}, {&c, 1}, {&e, 2}});
    dadd_term(&out, -6, {{&a, 1}, {&b, 2}, {&d, 2}, {&e, 1}});
    dadd_term(
        &out, -80,
        {{&a, 1}, {&b, 1}, {&c, 2}, {&d, 1}, {&e, 1}});
    dadd_term(&out, 18, {{&a, 1}, {&b, 1}, {&c, 1}, {&d, 3}});
    dadd_term(&out, 16, {{&a, 1}, {&c, 4}, {&e, 1}});
    dadd_term(&out, -4, {{&a, 1}, {&c, 3}, {&d, 2}});
    dadd_term(&out, -27, {{&b, 4}, {&e, 2}});
    dadd_term(&out, 18, {{&b, 3}, {&c, 1}, {&d, 1}, {&e, 1}});
    dadd_term(&out, -4, {{&b, 3}, {&d, 3}});
    dadd_term(&out, -4, {{&b, 2}, {&c, 3}, {&e, 1}});
    dadd_term(&out, 1, {{&b, 2}, {&c, 2}, {&d, 2}});
    return out;
}

using SeedPoly = std::vector<long double>;  // non-certifying GSL/polish path

SeedPoly dyadic_midpoint_poly(const DPoly& input)
{
    SeedPoly out;
    out.reserve(input.size());
    for (const Dyadic& value : input) {
        out.push_back(static_cast<long double>(dyadic_to_double(value)));
    }
    return out;
}

long double seed_eval(const SeedPoly& polynomial, long double x)
{
    long double value = 0.0L;
    for (auto it = polynomial.rbegin(); it != polynomial.rend(); ++it) {
        value = value * x + *it;
    }
    return value;
}

SeedPoly seed_derivative(const SeedPoly& polynomial)
{
    if (polynomial.size() < 2) return {0.0L};
    SeedPoly out(polynomial.size() - 1);
    for (std::size_t i = 1; i < polynomial.size(); ++i) {
        out[i - 1] = static_cast<long double>(i) * polynomial[i];
    }
    return out;
}

bool roots_of_real_poly(const SeedPoly& polynomial, std::vector<double>* roots)
{
    if (polynomial.size() < 2 || roots == nullptr) return false;
    std::vector<double> coefficients(polynomial.size());
    for (std::size_t i = 0; i < polynomial.size(); ++i) {
        coefficients[i] = static_cast<double>(polynomial[i]);
    }
    gsl_poly_complex_workspace* workspace =
        gsl_poly_complex_workspace_alloc(polynomial.size());
    if (workspace == nullptr) return false;
    std::vector<double> values(2 * (polynomial.size() - 1));
    const int status = gsl_poly_complex_solve(
        coefficients.data(), polynomial.size(), workspace, values.data());
    gsl_poly_complex_workspace_free(workspace);
    if (status != GSL_SUCCESS) return false;
    for (std::size_t i = 0; i + 1 < values.size(); i += 2) {
        if (std::isfinite(values[i]) && std::isfinite(values[i + 1]) &&
            std::abs(values[i + 1]) <=
                1.0e-8 * (1.0 + std::abs(values[i]))) {
            roots->push_back(values[i]);
        }
    }
    std::sort(roots->begin(), roots->end());
    roots->erase(
        std::unique(
            roots->begin(), roots->end(),
            [](double x, double y) {
                return std::abs(x - y) <=
                    1.0e-7 * (1.0 + std::abs(x));
            }),
        roots->end());
    return true;
}

Interval ieval(const IPoly& polynomial, Interval x)
{
    Interval value = exact_interval(0.0);
    for (auto it = polynomial.rbegin(); it != polynomial.rend(); ++it) {
        value = iadd(imul(value, x), *it);
    }
    return value;
}

IPoly iderivative(const IPoly& polynomial)
{
    if (polynomial.size() < 2) return {exact_interval(0.0)};
    IPoly out(polynomial.size() - 1);
    for (std::size_t i = 1; i < polynomial.size(); ++i) {
        out[i - 1] = iscale(polynomial[i], static_cast<double>(i));
    }
    return out;
}

int strict_sign(Interval value)
{
    return value.lo > 0.0 ? 1 : (value.hi < 0.0 ? -1 : 0);
}

Interval centered_eval(
    const IPoly& polynomial,
    const IPoly& derivative,
    double lo,
    double hi)
{
    const double mid = 0.5 * lo + 0.5 * hi;
    const Interval displacement = {
        std::nextafter(lo - mid, -std::numeric_limits<double>::infinity()),
        std::nextafter(hi - mid, std::numeric_limits<double>::infinity())};
    return iadd(
        ieval(polynomial, exact_interval(mid)),
        imul(displacement, ieval(derivative, {lo, hi})));
}

struct IsolatedRoot {
    double lo;
    double hi;
    double witness_lo;
    double witness_hi;
};

struct IsolationState {
    int budget = 128;
    int bisections = 0;
    int calls = 0;
    bool ok = true;
    const char* reason = "ok";

    void fail(const char* text)
    {
        if (ok) {
            ok = false;
            reason = text;
        }
    }
};

bool bisect_simple_root(
    const IPoly& polynomial,
    double lo,
    double hi,
    IsolationState* state,
    IsolatedRoot* box)
{
    int left = strict_sign(ieval(polynomial, exact_interval(lo)));
    int right = strict_sign(ieval(polynomial, exact_interval(hi)));
    if (left == 0 || right == 0 || left == right) {
        state->fail("endpoint_sign_uncertain");
        return false;
    }
    for (int iteration = 0; iteration < state->budget; ++iteration) {
        const double mid = 0.5 * lo + 0.5 * hi;
        if (!(mid > lo && mid < hi)) break;
        const int sign = strict_sign(ieval(polynomial, exact_interval(mid)));
        ++state->bisections;
        if (sign == 0) {
            const double qleft = 0.5 * (lo + mid);
            const double qright = 0.5 * (mid + hi);
            const int left_quarter =
                strict_sign(ieval(polynomial, exact_interval(qleft)));
            const int right_quarter =
                strict_sign(ieval(polynomial, exact_interval(qright)));
            if (left_quarter != 0) {
                if (left_quarter == left) {
                    lo = qleft;
                    left = left_quarter;
                } else {
                    hi = qleft;
                    right = left_quarter;
                }
            } else if (right_quarter != 0) {
                if (right_quarter == right) {
                    hi = qright;
                    right = right_quarter;
                } else {
                    lo = qright;
                    left = right_quarter;
                }
            } else {
                *box = {lo, hi, lo, hi};
                return true;
            }
        } else if (sign == left) {
            lo = mid;
            left = sign;
        } else {
            hi = mid;
            right = sign;
        }
        if (hi - lo <=
            2.0e-12 * (1.0 + std::max(std::abs(lo), std::abs(hi)))) {
            *box = {lo, hi, lo, hi};
            return true;
        }
    }
    state->fail("root_bisection_budget");
    return false;
}

bool isolate_positive_roots(
    const IPoly& polynomial,
    double lo,
    double hi,
    IsolationState* state,
    std::vector<IsolatedRoot>* roots)
{
    ++state->calls;
    if (!state->ok || polynomial.size() < 2 || !std::isfinite(lo) ||
        !std::isfinite(hi) || !(hi > lo)) {
        state->fail("invalid_isolation_domain");
        return false;
    }
    const IPoly derivative = iderivative(polynomial);
    if (derivative.size() != polynomial.size() - 1) {
        state->fail("derivative_degree_not_descending");
        return false;
    }
    std::vector<IsolatedRoot> critical;
    if (derivative.size() > 1 &&
        !isolate_positive_roots(derivative, lo, hi, state, &critical)) {
        return false;
    }
    std::sort(
        critical.begin(), critical.end(),
        [](const IsolatedRoot& a, const IsolatedRoot& b) {
            return a.lo < b.lo;
        });
    for (std::size_t i = 0; i < critical.size(); ++i) {
        if (!(critical[i].lo > lo && critical[i].hi < hi) ||
            critical[i].hi < critical[i].lo ||
            (i > 0 && critical[i - 1].hi >= critical[i].lo)) {
            state->fail("critical_boxes_overlap_or_escape");
            return false;
        }
        const Interval critical_value = centered_eval(
            polynomial, derivative, critical[i].lo, critical[i].hi);
        if (contains_zero(critical_value)) {
            state->fail("critical_value_contains_zero");
            return false;
        }
    }
    std::vector<std::pair<double, double>> cells;
    double left = lo;
    for (const auto& box : critical) {
        cells.push_back({left, box.lo});
        left = box.hi;
    }
    cells.push_back({left, hi});
    for (const auto& cell : cells) {
        if (!(cell.second > cell.first)) {
            state->fail("empty_monotone_cell");
            return false;
        }
        const int left_sign =
            strict_sign(ieval(polynomial, exact_interval(cell.first)));
        const int right_sign =
            strict_sign(ieval(polynomial, exact_interval(cell.second)));
        const double cell_midpoint = 0.5 * cell.first + 0.5 * cell.second;
        if (left_sign == 0 || right_sign == 0 ||
            strict_sign(
                ieval(derivative, exact_interval(cell_midpoint))) == 0) {
            state->fail("monotone_cell_sign_uncertain");
            return false;
        }
        if (left_sign == right_sign) continue;
        IsolatedRoot root;
        if (!bisect_simple_root(
                polynomial, cell.first, cell.second, state, &root)) {
            return false;
        }
        bool contracted_once = false;
        const IPoly second_derivative = iderivative(derivative);
        for (int contraction = 0; contraction < 8; ++contraction) {
            const Interval derivative_natural =
                ieval(derivative, {root.lo, root.hi});
            const Interval derivative_taylor = centered_eval(
                derivative, second_derivative, root.lo, root.hi);
            const Interval derivative_box =
                iintersect(derivative_natural, derivative_taylor);
            if (!(derivative_box.hi >= derivative_box.lo)) {
                state->fail("derivative_enclosure_intersection_empty");
                return false;
            }
            if (contains_zero(derivative_box)) {
                state->fail("interval_newton_derivative_contains_zero");
                return false;
            }
            const double mid = 0.5 * root.lo + 0.5 * root.hi;
            const Interval newton = isub(
                exact_interval(mid),
                idiv(
                    ieval(polynomial, exact_interval(mid)), derivative_box));
            const double new_lo = std::max(root.lo, newton.lo);
            const double new_hi = std::min(root.hi, newton.hi);
            if (!(new_hi > new_lo)) {
                state->fail("interval_newton_empty_intersection");
                return false;
            }
            if (new_lo == root.lo && new_hi == root.hi) {
                if (!contracted_once) {
                    state->fail("interval_newton_no_contraction");
                    return false;
                }
                break;
            }
            root.lo = new_lo;
            root.hi = new_hi;
            contracted_once = true;
            if (root.hi - root.lo <=
                2.0e-12 *
                    (1.0 + std::max(std::abs(root.lo), std::abs(root.hi)))) {
                break;
            }
        }
        roots->push_back(root);
    }
    return state->ok;
}

bool outward_cauchy_bound(const IPoly& polynomial, double* bound)
{
    if (polynomial.size() < 2 || !finite_interval(polynomial.back()) ||
        contains_zero(polynomial.back()) || bound == nullptr) {
        return false;
    }
    const double lead = absmin(polynomial.back());
    if (!(lead > 0.0) || !std::isfinite(lead)) return false;
    double result = 1.0;
    for (std::size_t i = 0; i + 1 < polynomial.size(); ++i) {
        const double ratio = std::nextafter(
            absmax(polynomial[i]) / lead,
            std::numeric_limits<double>::infinity());
        result = std::max(
            result,
            std::nextafter(
                1.0 + ratio, std::numeric_limits<double>::infinity()));
    }
    *bound = std::nextafter(
        result, std::numeric_limits<double>::infinity());
    return std::isfinite(*bound);
}

struct ChartSystemInterval {
    Interval f;
    Interval ft;
    Interval fr;
    Interval ftr;
    Interval ftt;
};

ChartSystemInterval chart_system_exact_center(
    const std::array<DPoly, 5>& polynomial,
    double radius,
    double chart)
{
    const Dyadic r = Dyadic::from_double(radius);
    const Dyadic t = Dyadic::from_double(chart);
    Dyadic q[5];
    Dyadic qr[5];
    for (int i = 0; i < 5; ++i) {
        q[i] = deval(polynomial[i], r);
        qr[i] = deval(dderivative(polynomial[i]), r);
    }
    const Dyadic t2 = dmul(t, t);
    const Dyadic t3 = dmul(t2, t);
    const Dyadic t4 = dmul(t2, t2);
    const Dyadic f = dadd(
        dadd(
            dadd(dadd(dmul(q[4], t4), dmul(q[3], t3)), dmul(q[2], t2)),
            dmul(q[1], t)),
        q[0]);
    const Dyadic ft = dadd(
        dadd(
            dadd(
                dmul(dmul(q[4], Dyadic(4)), t3),
                dmul(dmul(q[3], Dyadic(3)), t2)),
            dmul(dmul(q[2], Dyadic(2)), t)),
        q[1]);
    const Dyadic fr = dadd(
        dadd(
            dadd(
                dadd(dmul(qr[4], t4), dmul(qr[3], t3)),
                dmul(qr[2], t2)),
            dmul(qr[1], t)),
        qr[0]);
    const Dyadic ftr = dadd(
        dadd(
            dadd(
                dmul(dmul(qr[4], Dyadic(4)), t3),
                dmul(dmul(qr[3], Dyadic(3)), t2)),
            dmul(dmul(qr[2], Dyadic(2)), t)),
        qr[1]);
    const Dyadic ftt = dadd(
        dadd(
            dmul(dmul(q[4], Dyadic(12)), t2),
            dmul(dmul(q[3], Dyadic(6)), t)),
        dmul(q[2], Dyadic(2)));
    return {
        dyadic_adjacent_enclosure(f),
        dyadic_adjacent_enclosure(ft),
        dyadic_adjacent_enclosure(fr),
        dyadic_adjacent_enclosure(ftr),
        dyadic_adjacent_enclosure(ftt)};
}

ChartSystemInterval chart_system_interval(
    const std::array<IPoly, 5>& polynomial,
    Interval radius,
    Interval chart)
{
    Interval q[5];
    Interval qr[5];
    for (int i = 0; i < 5; ++i) {
        q[i] = ieval(polynomial[i], radius);
        qr[i] = ieval(iderivative(polynomial[i]), radius);
    }
    const Interval chart2 = imul(chart, chart);
    const Interval chart3 = imul(chart2, chart);
    const Interval chart4 = imul(chart2, chart2);
    const Interval f = iadd(
        iadd(
            iadd(iadd(imul(q[4], chart4), imul(q[3], chart3)),
                 imul(q[2], chart2)),
            imul(q[1], chart)),
        q[0]);
    const Interval ft = iadd(
        iadd(
            iadd(imul(iscale(q[4], 4.0), chart3),
                 imul(iscale(q[3], 3.0), chart2)),
            imul(iscale(q[2], 2.0), chart)),
        q[1]);
    const Interval fr = iadd(
        iadd(
            iadd(iadd(imul(qr[4], chart4), imul(qr[3], chart3)),
                 imul(qr[2], chart2)),
            imul(qr[1], chart)),
        qr[0]);
    const Interval ftr = iadd(
        iadd(
            iadd(imul(iscale(qr[4], 4.0), chart3),
                 imul(iscale(qr[3], 3.0), chart2)),
            imul(iscale(qr[2], 2.0), chart)),
        qr[1]);
    const Interval ftt = iadd(
        iadd(
            imul(iscale(q[4], 12.0), chart2),
            imul(iscale(q[3], 6.0), chart)),
        iscale(q[2], 2.0));
    return {f, ft, fr, ftr, ftt};
}

bool chart_krawczyk(
    const std::array<IPoly, 5>& polynomial,
    const std::array<DPoly, 5>& exact_polynomial,
    double radius0,
    double chart0,
    double radius_lo,
    double radius_hi,
    Interval* certified_radius,
    Interval* certified_chart)
{
    Interval radius{radius_lo, radius_hi};
    const double chart_padding = 1.0e-10 * (1.0 + std::abs(chart0));
    Interval chart{chart0 - chart_padding, chart0 + chart_padding};
    const auto center =
        chart_system_exact_center(exact_polynomial, radius0, chart0);
    if (!finite_interval(center.f) || !finite_interval(center.ft) ||
        !finite_interval(center.fr) || !finite_interval(center.ftr) ||
        !finite_interval(center.ftt)) {
        return false;
    }
    const double determinant =
        midpoint(center.fr) * midpoint(center.ftt) -
        midpoint(center.ft) * midpoint(center.ftr);
    if (!std::isfinite(determinant) || std::abs(determinant) < 1.0e-20) {
        return false;
    }
    const double c00 = midpoint(center.ftt) / determinant;
    const double c01 = -midpoint(center.ft) / determinant;
    const double c10 = -midpoint(center.ftr) / determinant;
    const double c11 = midpoint(center.fr) / determinant;
    const Interval y0 = isub(
        exact_interval(radius0),
        iadd(iscale(center.f, c00), iscale(center.ft, c01)));
    const Interval y1 = isub(
        exact_interval(chart0),
        iadd(iscale(center.f, c10), iscale(center.ft, c11)));
    bool contracted = false;
    for (int iteration = 0; iteration < 8; ++iteration) {
        const auto whole = chart_system_interval(polynomial, radius, chart);
        const Interval m00 = isub(
            exact_interval(1.0),
            iadd(iscale(whole.fr, c00), iscale(whole.ftr, c01)));
        const Interval m01 = ineg(
            iadd(iscale(whole.ft, c00), iscale(whole.ftt, c01)));
        const Interval m10 = ineg(
            iadd(iscale(whole.fr, c10), iscale(whole.ftr, c11)));
        const Interval m11 = isub(
            exact_interval(1.0),
            iadd(iscale(whole.ft, c10), iscale(whole.ftt, c11)));
        const Interval dr = isub(radius, exact_interval(radius0));
        const Interval dt = isub(chart, exact_interval(chart0));
        const Interval k0 =
            iadd(y0, iadd(imul(m00, dr), imul(m01, dt)));
        const Interval k1 =
            iadd(y1, iadd(imul(m10, dr), imul(m11, dt)));
        const bool enclosed =
            k0.lo > radius.lo && k0.hi < radius.hi &&
            k1.lo > chart.lo && k1.hi < chart.hi;
        if (!enclosed) return false;
        radius = k0;
        chart = k1;
        contracted = true;
        if (radius.hi - radius.lo <=
            2.0e-12 * (1.0 + std::abs(radius0))) {
            break;
        }
    }
    const bool accepted = contracted &&
        radius.hi - radius.lo <= 2.0e-12 * (1.0 + std::abs(radius0));
    if (accepted) {
        *certified_radius = radius;
        *certified_chart = chart;
    }
    return accepted;
}

double seed_eval_t(
    const std::array<SeedPoly, 5>& polynomial,
    double radius,
    double chart,
    int derivative_order)
{
    double q[5];
    for (int i = 0; i < 5; ++i) {
        q[i] = static_cast<double>(seed_eval(polynomial[i], radius));
    }
    if (derivative_order == 0) {
        return (((q[4] * chart + q[3]) * chart + q[2]) * chart + q[1]) *
            chart + q[0];
    }
    return ((4.0 * q[4] * chart + 3.0 * q[3]) * chart + 2.0 * q[2]) *
        chart + q[1];
}

bool polish_event_chart(
    const std::array<SeedPoly, 5>& polynomial,
    double* radius,
    double* chart)
{
    for (int iteration = 0; iteration < 32; ++iteration) {
        double q[5];
        double qr[5];
        for (int i = 0; i < 5; ++i) {
            q[i] = static_cast<double>(seed_eval(polynomial[i], *radius));
            qr[i] = static_cast<double>(
                seed_eval(seed_derivative(polynomial[i]), *radius));
        }
        const double z = *chart;
        const double f =
            (((q[4] * z + q[3]) * z + q[2]) * z + q[1]) * z + q[0];
        const double ft =
            ((4.0 * q[4] * z + 3.0 * q[3]) * z + 2.0 * q[2]) * z +
            q[1];
        const double fr =
            (((qr[4] * z + qr[3]) * z + qr[2]) * z + qr[1]) * z +
            qr[0];
        const double ftr =
            ((4.0 * qr[4] * z + 3.0 * qr[3]) * z + 2.0 * qr[2]) * z +
            qr[1];
        const double ftt =
            (12.0 * q[4] * z + 6.0 * q[3]) * z + 2.0 * q[2];
        const double determinant = fr * ftt - ft * ftr;
        if (!std::isfinite(determinant) ||
            std::abs(determinant) < 1.0e-15) {
            return false;
        }
        const double dr = (-f * ftt + ft * ft) / determinant;
        const double dz = (-fr * ft + ftr * f) / determinant;
        *radius += dr;
        *chart += dz;
        if (!std::isfinite(*radius) || !std::isfinite(*chart) ||
            !(*radius > 0.0)) {
            return false;
        }
        if (std::max(
                std::abs(dr) / (1.0 + std::abs(*radius)),
                std::abs(dz) / (1.0 + std::abs(*chart))) < 1.0e-13) {
            return true;
        }
    }
    return false;
}

bool finite_contact_proxy(
    const std::array<DPoly, 5>& exact_polynomial,
    double radius,
    double chart)
{
    const auto contact =
        chart_system_exact_center(exact_polynomial, radius, chart);
    if (!finite_interval(contact.fr) || !finite_interval(contact.ftr) ||
        !finite_interval(contact.ft) || !finite_interval(contact.ftt)) {
        return false;
    }
    const double determinant =
        midpoint(contact.fr) * midpoint(contact.ftt) -
        midpoint(contact.ft) * midpoint(contact.ftr);
    return std::isfinite(determinant) && std::abs(determinant) > 1.0e-20;
}

bool canonicalize_candidate_reports(
    const std::vector<CriticalEvent>& reports, CriticalEvent* canonical)
{
    // Preconditions supplied by the caller: each report is an original strict
    // lift of one simple positive H root.  This only checks consistency; it
    // does not synthesize or certify a replacement box.
    if (reports.empty() || canonical == nullptr) return false;
    const std::size_t id = reports.front().event_id;
    const CertifiedInterval u_box = reports.front().u_box;
    bool any_tan = false;
    for (const auto& report : reports) {
        if (report.event_id != id || report.u_box.lo != u_box.lo ||
            report.u_box.hi != u_box.hi ||
            !positive_square_subset(
                {report.radius_box.lo, report.radius_box.hi},
                {report.u_box.lo, report.u_box.hi})) return false;
        any_tan |= !report.cot_chart;
    }
    Interval common_radius{reports.front().radius_box.lo, reports.front().radius_box.hi};
    Interval common_chart{reports.front().chart_box.lo, reports.front().chart_box.hi};
    if (any_tan && reports.front().cot_chart) {
        common_chart = idiv(exact_interval(1.0), common_chart);
    }
    if (!finite_interval(common_chart)) return false;
    for (std::size_t i = 1; i < reports.size(); ++i) {
        const auto& report = reports[i];
        common_radius = iintersect(
            common_radius, {report.radius_box.lo, report.radius_box.hi});
        Interval chart{report.chart_box.lo, report.chart_box.hi};
        if (any_tan && report.cot_chart) chart = idiv(exact_interval(1.0), chart);
        if (!finite_interval(chart)) return false;
        common_chart = iintersect(common_chart, chart);
        if (!finite_interval(common_radius) || !finite_interval(common_chart)) return false;
    }
    *canonical = reports.front();  // Retain an original strict lift, not a synthetic box.
    return true;
}

bool arithmetic_unit_checks()
{
    bool pass = true;
    const auto interval_check = [&pass](
                                    const IPoly& polynomial,
                                    bool expected_success,
                                    int expected_roots) {
        double bound = 0.0;
        IsolationState state;
        std::vector<IsolatedRoot> roots;
        const bool bound_ok = outward_cauchy_bound(polynomial, &bound);
        const bool success = bound_ok && isolate_positive_roots(
            polynomial, 0.0, bound, &state, &roots);
        pass &= success == expected_success &&
            (!success || static_cast<int>(roots.size()) == expected_roots);
    };
    interval_check(
        {exact_interval(-2.0), exact_interval(1.0)}, true, 1);
    interval_check(
        {exact_interval(3.0), exact_interval(-4.0), exact_interval(1.0)},
        true, 2);
    interval_check(
        {exact_interval(1.0), exact_interval(1.0)}, true, 0);
    interval_check(
        {exact_interval(-2.0), exact_interval(-1.0), exact_interval(1.0)},
        true, 1);
    interval_check(
        {exact_interval(-0.5), exact_interval(1.0)}, true, 1);
    interval_check(
        {exact_interval(1.0000000000001),
         exact_interval(-2.0000000000001), exact_interval(1.0)},
        false, 0);
    interval_check(
        {exact_interval(1.0), {-1.0e-16, 1.0e-16}}, false, 0);
    interval_check(
        {exact_interval(0.0), exact_interval(0.0), exact_interval(1.0)},
        false, 0);

    const auto dyadic_check = [&pass](
                                  const Dyadic& value,
                                  bool expected_valid,
                                  bool expected_finite) {
        pass &= value.valid == expected_valid &&
            finite_interval(dyadic_adjacent_enclosure(value)) == expected_finite;
    };
    dyadic_check(Dyadic::from_double(0.0), true, true);
    dyadic_check(Dyadic::from_double(-1.25), true, true);
    dyadic_check(
        Dyadic::from_double(std::numeric_limits<double>::denorm_min()),
        true, true);
    dyadic_check(
        Dyadic::from_double(std::numeric_limits<double>::infinity()),
        false, false);
    dyadic_check(
        Dyadic::from_double(std::numeric_limits<double>::quiet_NaN()),
        false, false);
    Dyadic tie;
    tie.numerator = (cpp_int(1) << 53) + 1;
    tie.exponent = -53;
    tie.normalize();
    pass &= dyadic_to_double(tie) == 1.0;
    dyadic_check(
        Dyadic::from_double(std::numeric_limits<double>::max()), true, false);
    const Dyadic invalid =
        Dyadic::from_double(std::numeric_limits<double>::infinity());
    int ignored_compare = 0;
    const Dyadic sum = dadd(invalid, Dyadic(1));
    const Dyadic product = dmul(Dyadic(1), invalid);
    const Dyadic negated = dneg(invalid);
    const Dyadic difference = dsub(Dyadic(1), invalid);
    pass &= !sum.valid && !product.valid && !negated.valid &&
        !difference.valid &&
        !dcompare(invalid, Dyadic(0), &ignored_compare) &&
        !std::isfinite(dyadic_to_double(sum)) &&
        !finite_interval(dyadic_adjacent_enclosure(product));

    // A tiny nonzero chart center with the fixed Krawczyk padding crosses
    // zero.  The outward displacement must contain exact-Dyadic endpoint
    // differences, not just their round-to-nearest subtraction.
    const double chart0 = 1.0e-20;
    const Interval chart_box{-1.0e-10, 1.0e-10};
    const Interval displacement = isub(chart_box, exact_interval(chart0));
    const Interval plain_displacement{
        chart_box.lo - chart0, chart_box.hi - chart0};
    const auto exact_in = [](Interval enclosure, const Dyadic& value) {
        int cmp_lo = 0;
        int cmp_hi = 0;
        return dcompare(
                   Dyadic::from_double(enclosure.lo), value, &cmp_lo) &&
            dcompare(value, Dyadic::from_double(enclosure.hi), &cmp_hi) &&
            cmp_lo <= 0 && cmp_hi <= 0;
    };
    const Dyadic exact_lower = dsub(
        Dyadic::from_double(chart_box.lo), Dyadic::from_double(chart0));
    const Dyadic exact_upper = dsub(
        Dyadic::from_double(chart_box.hi), Dyadic::from_double(chart0));
    pass &= exact_in(displacement, exact_lower) &&
        exact_in(displacement, exact_upper) &&
        !exact_in(plain_displacement, exact_upper);

    // A midpoint exactly on the U lower bound cannot rescue a radius box that
    // straddles it by one representable value.
    const Interval u_boundary{1.0, 1.0};
    const Interval straddling_radius{
        std::nextafter(1.0, -std::numeric_limits<double>::infinity()),
        std::nextafter(1.0, std::numeric_limits<double>::infinity())};
    pass &= !positive_square_subset(straddling_radius, u_boundary);
    pass &= positive_square_subset({1.0, 1.0}, {0.9, 1.1});

    // Consistency-helper tests only: production calls this after strict lifts
    // and simple-root certification, never as a new certificate shortcut.
    const auto report = [](std::size_t id, Interval radius, Interval chart,
                           bool cot) {
        return CriticalEvent{id, {0.9, 1.1}, {radius.lo, radius.hi},
                             {chart.lo, chart.hi}, cot, 1.0, 0.0};
    };
    CriticalEvent canonical;
    const auto same = report(4, {1.0, 1.0}, {.4, .6}, false);
    pass &= canonicalize_candidate_reports({same, same}, &canonical);
    pass &= canonical.event_id == same.event_id &&
        canonical.radius_box.lo == same.radius_box.lo &&
        canonical.chart_box.lo == same.chart_box.lo;
    pass &= canonicalize_candidate_reports(
        {report(4, {.99, 1.01}, {.4, .6}, false),
         report(4, {1.0, 1.02}, {.5, .7}, false)}, &canonical);
    pass &= canonicalize_candidate_reports(
        {report(4, {1.0, 1.0}, {.49, .51}, false),
         report(4, {1.0, 1.0}, {1.99, 2.01}, true)}, &canonical);
    pass &= canonicalize_candidate_reports(
        {report(4, {1.0, 1.0}, {-.51, -.49}, false),
         report(4, {1.0, 1.0}, {-2.01, -1.99}, true)}, &canonical);
    pass &= canonicalize_candidate_reports(
        {report(4, {1.0, 1.0}, {1.99, 2.01}, true),
         report(4, {1.0, 1.0}, {2.0, 2.02}, true)}, &canonical);
    pass &= !canonicalize_candidate_reports(
        {report(4, {1.0, 1.0}, {.4, .6}, false),
         report(4, {1.0, 1.0}, {-.1, .1}, true)}, &canonical);
    pass &= !canonicalize_candidate_reports(
        {report(4, {1.0, 1.0}, {0.0, 2.0}, false),
         report(4, {1.0, 1.0}, {1.0, 3.0}, false),
         report(4, {1.0, 1.0}, {2.5, 4.0}, false)}, &canonical);
    pass &= !canonicalize_candidate_reports(
        {report(4, {1.0, 1.0}, {.1, .2}, false),
         report(4, {1.0, 1.0}, {.3, .4}, false)}, &canonical);
    pass &= !canonicalize_candidate_reports(
        {report(4, {1.0, 1.0}, {.4, .6}, false),
         report(5, {1.0, 1.0}, {.4, .6}, false)}, &canonical);
    auto different_u = report(4, {1.0, 1.0}, {.4, .6}, false);
    different_u.u_box.hi = 1.2;
    pass &= !canonicalize_candidate_reports({same, different_u}, &canonical);
    return pass;
}

}  // namespace

bool freeze_positive_geometry(
    const CriticalEventInput& input,
    FrozenGeometry* frozen,
    std::string* reason)
{
    if (frozen == nullptr) return false;
    *frozen = {};
    if (!std::isfinite(input.separation) || !std::isfinite(input.q) ||
        !std::isfinite(input.rho) || !std::isfinite(input.source.x) ||
        !std::isfinite(input.source.y) || !(input.separation > 0.0) ||
        !(input.q > 0.0) || !(input.rho > 0.0)) {
        if (reason != nullptr) {
            *reason = "critical-event policy requires finite positive geometry";
        }
        return false;
    }
    const double normalized_q = input.q < 1.0 ? input.q : 1.0 / input.q;
    const double a = input.q < 1.0 ? -input.separation : input.separation;
    const double m1 = 1.0 / (1.0 + normalized_q);
    const double m2 = normalized_q * m1;
    const double am1 = a * m1;
    const double sx = input.source.x + am1;
    if (!std::isfinite(normalized_q) || !std::isfinite(a) ||
        !std::isfinite(m1) || !std::isfinite(m2) || !std::isfinite(sx)) {
        if (reason != nullptr) {
            *reason = "critical-event canonicalization overflow";
        }
        return false;
    }
    *frozen = {a, m1, m2, sx, input.source.y, input.rho};
    return true;
}

PhysicalResidual physical_boundary_residual(
    const FrozenGeometry& frozen,
    SourcePosition source,
    double image_x,
    double image_y)
{
    PhysicalResidual out;
    const double d0 = image_x * image_x + image_y * image_y;
    const double xa = image_x - frozen.a;
    const double d1 = xa * xa + image_y * image_y;
    out.non_pole = std::isfinite(d0) && std::isfinite(d1) &&
        d0 > 1.0e-28 && d1 > 1.0e-28;
    if (!out.non_pole) return out;
    const double mapped_x = image_x - frozen.m1 * xa / d1 -
        frozen.m2 * image_x / d0 - frozen.a * frozen.m1;
    const double mapped_y = image_y - frozen.m1 * image_y / d1 -
        frozen.m2 * image_y / d0;
    const double dx = mapped_x - source.x;
    const double dy = mapped_y - source.y;
    out.value = dx * dx + dy * dy - frozen.rho * frozen.rho;
    out.scale = std::max(frozen.rho * frozen.rho, dx * dx + dy * dy);
    out.finite = std::isfinite(out.value) && std::isfinite(out.scale) &&
        out.scale > 0.0;
    return out;
}

CriticalEventNumericalResult enumerate_and_lift_critical_events(
    const CriticalEventInput& input)
{
    CriticalEventNumericalResult out;
    FrozenGeometry geometry;
    if (!freeze_positive_geometry(input, &geometry, &out.unsafe_reason)) {
        return out;
    }

    const auto exact_polynomial = dyadic_trig_polynomials(geometry);
    std::array<IPoly, 5> polynomial_enclosure;
    bool coefficients_valid = true;
    for (std::size_t i = 0; i < exact_polynomial.size(); ++i) {
        for (const Dyadic& coefficient : exact_polynomial[i]) {
            coefficients_valid &= coefficient.valid;
            polynomial_enclosure[i].push_back(
                dyadic_adjacent_enclosure(coefficient));
        }
    }
    const DPoly exact_discriminant = dyadic_discriminant(exact_polynomial);
    for (const Dyadic& coefficient : exact_discriminant) {
        coefficients_valid &= coefficient.valid;
    }
    if (!coefficients_valid) {
        out.unsafe_reason = "invalid_exact_coefficient";
        return out;
    }
    if (exact_discriminant.size() > 37) {
        out.unsafe_reason = "discriminant_degree_failure";
        return out;
    }
    IPoly h;
    for (std::size_t power = 0; power < 37; ++power) {
        const Dyadic coefficient = power < exact_discriminant.size()
            ? exact_discriminant[power]
            : Dyadic(0);
        const bool allowed =
            power >= 4 && power <= 32 && ((power - 4) % 2 == 0);
        if (!allowed) {
            if (!coefficient.valid || coefficient.numerator != 0) {
                out.unsafe_reason = "discriminant_structure_failure";
                return out;
            }
        } else {
            const Interval enclosure = dyadic_adjacent_enclosure(coefficient);
            if (!finite_interval(enclosure)) {
                out.unsafe_reason = "nonfinite_h_coefficient_enclosure";
                return out;
            }
            h.push_back(enclosure);
        }
    }
    if (h.size() != 15) {
        out.unsafe_reason = "canonical_h_degree_failure";
        return out;
    }

    // Exact power-of-two U=2V scaling retained from the accepted probe.
    IPoly h_v = h;
    for (std::size_t i = 0; i < h_v.size(); ++i) {
        h_v[i] = iscale(
            h_v[i], std::ldexp(1.0, static_cast<int>(i)));
    }
    double cauchy_bound = 0.0;
    IsolationState isolation;
    std::vector<IsolatedRoot> roots_v;
    if (!outward_cauchy_bound(h_v, &cauchy_bound) ||
        !isolate_positive_roots(
            h_v, 0.0, cauchy_bound, &isolation, &roots_v)) {
        out.isolation_calls = isolation.calls;
        out.root_bisections = isolation.bisections;
        out.unsafe_reason = isolation.ok
            ? "positive_root_bound_failure"
            : isolation.reason;
        return out;
    }
    out.isolation_calls = isolation.calls;
    out.root_bisections = isolation.bisections;
    std::sort(
        roots_v.begin(), roots_v.end(),
        [](const IsolatedRoot& a, const IsolatedRoot& b) {
            return a.lo < b.lo;
        });
    for (std::size_t i = 1; i < roots_v.size(); ++i) {
        const double previous_width =
            2.0 * (roots_v[i - 1].hi - roots_v[i - 1].lo);
        const double width = 2.0 * (roots_v[i].hi - roots_v[i].lo);
        const double gap = 2.0 * roots_v[i].lo - 2.0 * roots_v[i - 1].hi;
        if (!(gap >= 16.0 * std::max(previous_width, width))) {
            out.unsafe_reason = "positive_root_boxes_not_separated";
            return out;
        }
    }

    std::array<SeedPoly, 5> seed_polynomial;
    for (std::size_t i = 0; i < exact_polynomial.size(); ++i) {
        seed_polynomial[i] = dyadic_midpoint_poly(exact_polynomial[i]);
    }
    std::vector<std::vector<CriticalEvent>> candidate_events(roots_v.size());
    for (std::size_t candidate_id = 0; candidate_id < roots_v.size();
         ++candidate_id) {
        const double u_lo = 2.0 * roots_v[candidate_id].lo;
        const double u_hi = 2.0 * roots_v[candidate_id].hi;
        const double u_seed = 0.5 * u_lo + 0.5 * u_hi;
        if (!(u_seed > 0.0) || !std::isfinite(u_seed)) {
            out.unsafe_reason = "invalid_positive_root_box";
            return out;
        }
        const double radius_seed = std::sqrt(u_seed);
        SeedPoly tan_at_radius(5);
        SeedPoly cot_at_radius(5);
        for (int i = 0; i < 5; ++i) {
            tan_at_radius[i] = seed_eval(seed_polynomial[i], radius_seed);
            cot_at_radius[4 - i] = tan_at_radius[i];
        }
        for (int chart_kind : {0, 1}) {
            const SeedPoly& chart_polynomial =
                chart_kind == 0 ? tan_at_radius : cot_at_radius;
            std::vector<double> chart_roots;
            if (!roots_of_real_poly(
                    seed_derivative(chart_polynomial), &chart_roots)) {
                out.unsafe_reason = "angular_seed_failure";
                return out;
            }
            out.lift_candidates += static_cast<int>(chart_roots.size());
            std::array<SeedPoly, 5> chart_seed_polynomial = seed_polynomial;
            std::array<IPoly, 5> chart_polynomial_enclosure =
                polynomial_enclosure;
            std::array<DPoly, 5> chart_exact_polynomial = exact_polynomial;
            if (chart_kind == 1) {
                std::reverse(
                    chart_seed_polynomial.begin(), chart_seed_polynomial.end());
                std::reverse(
                    chart_polynomial_enclosure.begin(),
                    chart_polynomial_enclosure.end());
                std::reverse(
                    chart_exact_polynomial.begin(),
                    chart_exact_polynomial.end());
            }
            for (double chart_seed : chart_roots) {
                if (std::abs(chart_seed) > 1.25) continue;
                double event_radius = radius_seed;
                double event_chart = chart_seed;
                if (!polish_event_chart(
                        chart_seed_polynomial, &event_radius, &event_chart)) {
                    continue;
                }
                const double event_u = event_radius * event_radius;
                const double seed_p = seed_eval_t(
                    chart_seed_polynomial, event_radius, event_chart, 0);
                const double seed_pt = seed_eval_t(
                    chart_seed_polynomial, event_radius, event_chart, 1);
                if (!(event_u >= u_lo && event_u <= u_hi) ||
                    std::abs(seed_p) > 1.0e-8 ||
                    std::abs(seed_pt) > 1.0e-7) {
                    continue;
                }
                const double radius_lo = std::nextafter(
                    std::sqrt(u_lo), -std::numeric_limits<double>::infinity());
                const double radius_hi = std::nextafter(
                    std::sqrt(u_hi), std::numeric_limits<double>::infinity());
                Interval certified_radius{};
                Interval certified_chart{};
                if (!chart_krawczyk(
                        chart_polynomial_enclosure,
                        chart_exact_polynomial,
                        event_radius,
                        event_chart,
                        radius_lo,
                        radius_hi,
                        &certified_radius,
                        &certified_chart)) {
                    continue;
                }
                const double radius = midpoint(certified_radius);
                const double chart = midpoint(certified_chart);
                const Interval u_box{u_lo, u_hi};
                const Interval representative_square =
                    imul(exact_interval(radius), exact_interval(radius));
                const double p = seed_eval_t(
                    chart_seed_polynomial, radius, chart, 0);
                const double pt = seed_eval_t(
                    chart_seed_polynomial, radius, chart, 1);
                const bool box_inside =
                    certified_radius.lo >= radius_lo &&
                    certified_radius.hi <= radius_hi;
                if (!positive_square_subset(certified_radius, u_box) ||
                    !(representative_square.lo >= u_lo &&
                      representative_square.hi <= u_hi) ||
                    !box_inside || std::abs(p) > 1.0e-12 ||
                    std::abs(pt) > 1.0e-12 ||
                    !finite_contact_proxy(
                        chart_exact_polynomial, radius, chart)) {
                    continue;
                }
                const double theta = chart_kind == 0
                    ? std::fmod(2.0 * std::atan(chart) + kTwoPi, kTwoPi)
                    : 2.0 * std::atan2(1.0, chart);
                candidate_events[candidate_id].push_back({
                    candidate_id,
                    {u_lo, u_hi},
                    {certified_radius.lo, certified_radius.hi},
                    {certified_chart.lo, certified_chart.hi},
                    chart_kind == 1,
                    radius,
                    theta});
            }
        }
        CriticalEvent canonical_event;
        if (!canonicalize_candidate_reports(
                candidate_events[candidate_id], &canonical_event)) {
            out.unsafe_reason = candidate_events[candidate_id].empty()
                ? "unlifted_positive_candidate" : "ambiguous_projective_lift";
            return out;
        }
        out.events.push_back(canonical_event);
    }

    std::sort(
        out.events.begin(), out.events.end(),
        [](const CriticalEvent& a, const CriticalEvent& b) {
            return a.radius < b.radius;
        });
    for (std::size_t i = 0; i < out.events.size(); ++i) {
        out.events[i].event_id = i;
    }
    out.success = true;
    out.unsafe_reason.clear();
    return out;
}

bool critical_event_numerical_unit_checks()
{
    return arithmetic_unit_checks();
}

CriticalEventPlan find_critical_events(const CriticalEventInput& input)
{
    CriticalEventPlan out;
    const auto numerical = enumerate_and_lift_critical_events(input);
    out.events = numerical.events;
    out.root_work = numerical.root_bisections;
    out.lift_work = numerical.lift_candidates;
    if (!numerical.success) {
        out.unsafe_reason = numerical.unsafe_reason;
        return out;
    }
    // Deliberate fail-closed boundary: physical interval residual/pole,
    // scaled-IFT, topology, and event-tagged band gates are not in this port.
    out.unsafe_reason =
        "pending physical-residual, scaled-IFT, topology, and band gates";
    return out;
}

}  // namespace lcbinint::magnification::critical_events

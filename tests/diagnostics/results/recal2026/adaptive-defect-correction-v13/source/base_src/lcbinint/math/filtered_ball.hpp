#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace lcbinint::math {

// Center-radius enclosure for a floating-point sign filter. Same arithmetic
// assumptions as OutwardInterval. Radius evaluation is inflated to cover its
// own rounding; uncertain signs fall back to the endpoint interval evaluator.
struct FilteredBall {
    double value = 0.0;
    double radius = 0.0;
    FilteredBall() = default;
    FilteredBall(double x) : value(x) {}
    FilteredBall(double x, double r) : value(x), radius(r) {}
    bool finite() const { return std::isfinite(value) && std::isfinite(radius); }
    bool zero() const { return value == 0.0 && radius == 0.0; }
    bool contains_zero() const { return std::abs(value) <= radius; }
    double magnitude() const;
    double minimum_magnitude() const;
    bool negative() const { return value < 0.0; }
};

// Every nonnegative bound expression below has at most eight arithmetic
// operations. 16*epsilon (epsilon=2*unit roundoff) covers their rounding plus
// the rounding of this inflation. The additive term covers gradual underflow.
inline double ball_upper(double x) {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    return x*(1.0+16.0*eps)+16.0*std::numeric_limits<double>::denorm_min();
}
inline double FilteredBall::magnitude() const {
    return ball_upper(std::abs(value)+radius);
}
inline double FilteredBall::minimum_magnitude() const {
    // Pivot choice need not itself be certified; zero separation is checked
    // on the selected enclosure before division.
    return std::max(0.0, std::abs(value)-radius);
}
inline FilteredBall operator-(FilteredBall a) { return {-a.value,a.radius}; }
inline FilteredBall operator+(FilteredBall a, FilteredBall b) {
    if (a.zero()) return b;
    if (b.zero()) return a;
    const double v = a.value+b.value;
    // epsilon*abs(v)+denorm_min bounds rounding of the center operation.
    return {v,ball_upper(a.radius+b.radius+
        std::numeric_limits<double>::epsilon()*std::abs(v))};
}
inline FilteredBall operator-(FilteredBall a, FilteredBall b) { return a+(-b); }
inline FilteredBall operator*(FilteredBall a, FilteredBall b) {
    if (a.zero() || b.zero()) return {};
    const double v = a.value*b.value;
    return {v,ball_upper(std::abs(a.value)*b.radius+std::abs(b.value)*a.radius+
        a.radius*b.radius+std::numeric_limits<double>::epsilon()*std::abs(v))};
}
inline FilteredBall operator/(FilteredBall a, FilteredBall b) {
    const double abs_b = std::abs(b.value);
    // A lower bound on the exact abs(center)-radius, including subtraction
    // rounding. Reject an underflowing or nonpositive lower bound.
    const double lower = (abs_b-b.radius)-ball_upper(
        std::numeric_limits<double>::epsilon()*abs_b);
    if (!(lower>0.0)) return {0.0,std::numeric_limits<double>::infinity()};
    const double v = a.value/b.value;
    const double rounding = ball_upper(std::numeric_limits<double>::epsilon()*std::abs(v));
    // |a/b-v| <= (ra+|v|*rb+|a0-v*b0|)/(|b0|-rb).
    // |a0-v*b0| <= division-rounding-bound * |b0|.
    // Inflate the numerator BEFORE dividing: numerator underflow can be
    // amplified by a small denominator and cannot be covered at the end.
    const double numerator = ball_upper(a.radius+std::abs(v)*b.radius+rounding*abs_b);
    return {v,ball_upper(numerator/lower)};
}
inline FilteredBall square(FilteredBall a) { return a*a; }

} // namespace lcbinint::math

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace lcbinint::math {

// Small closed intervals for diagnostic validated arithmetic. Inputs are the
// exact real values of their binary64 representations. Requires ordinary IEEE
// arithmetic (no fast-math, round-to-nearest, gradual underflow).
struct OutwardInterval {
    double lo = 0.0;
    double hi = 0.0;

    OutwardInterval() = default;
    OutwardInterval(double value) : lo(value), hi(value) {}
    OutwardInterval(double lower, double upper) : lo(lower), hi(upper) {}

    bool finite() const { return std::isfinite(lo) && std::isfinite(hi); }
    bool contains_zero() const { return lo <= 0.0 && hi >= 0.0; }
    bool zero() const { return lo == 0.0 && hi == 0.0; }
    double magnitude() const { return std::max(std::abs(lo), std::abs(hi)); }
    double minimum_magnitude() const {
        return contains_zero() ? 0.0 : std::min(std::abs(lo), std::abs(hi));
    }
};

// Exactly nextafter(x,+/-infinity) for IEEE binary64, including signed zero
// and infinities. Avoid hundreds of libm calls per small determinant.
inline double adjacent(double x, bool upward) {
    if constexpr (std::numeric_limits<double>::is_iec559 && sizeof(double)==sizeof(std::uint64_t)) {
        if (std::isnan(x)) return x;
        if (x==0.0) return upward ? std::numeric_limits<double>::denorm_min()
                                : -std::numeric_limits<double>::denorm_min();
        if (x==(upward ? std::numeric_limits<double>::infinity()
                      : -std::numeric_limits<double>::infinity())) return x;
        std::uint64_t bits;
        std::memcpy(&bits,&x,sizeof(bits));
        if ((x>0.0)==upward) ++bits; else --bits;
        std::memcpy(&x,&bits,sizeof(bits));
        return x;
    } else {
        return std::nextafter(x,upward ? std::numeric_limits<double>::infinity()
                                      : -std::numeric_limits<double>::infinity());
    }
}
inline double down(double x) { return adjacent(x,false); }
inline double up(double x) { return adjacent(x,true); }
inline OutwardInterval operator+(OutwardInterval a, OutwardInterval b) {
    if (a.zero()) return b;
    if (b.zero()) return a;
    return {down(a.lo + b.lo), up(a.hi + b.hi)};
}
inline OutwardInterval operator-(OutwardInterval a) { return {-a.hi, -a.lo}; }
inline OutwardInterval operator-(OutwardInterval a, OutwardInterval b) {
    return a + (-b);
}
inline OutwardInterval operator*(OutwardInterval a, OutwardInterval b) {
    if (a.zero() || b.zero()) return {};
    const double products[] = {a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi};
    return {down(*std::min_element(products, products+4)),
            up(*std::max_element(products, products+4))};
}
inline OutwardInterval operator/(OutwardInterval a, OutwardInterval b) {
    if (b.contains_zero()) {
        return {-std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity()};
    }
    return a * OutwardInterval(down(1.0/b.hi), up(1.0/b.lo));
}
inline OutwardInterval square(OutwardInterval a) {
    if (a.zero()) return {};
    const double lower = a.minimum_magnitude();
    const double upper = a.magnitude();
    return {lower == 0.0 ? 0.0 : std::max(0.0, down(lower*lower)), up(upper*upper)};
}

} // namespace lcbinint::math

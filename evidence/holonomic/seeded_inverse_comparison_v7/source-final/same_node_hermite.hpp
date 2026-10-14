#pragma once

#include <array>
#include <cmath>
#include <limits>

namespace lcbinint::holonomic {

// Evaluate a short product/ratio without materializing an overflowing
// numerator or underflowing denominator intermediate.  This is used for the
// fold-regular combination J^2/sqrt(v), not as a general high-precision type.
template <std::size_t N, std::size_t M>
inline double hermite_scaled_product_ratio(const std::array<double,N>& num,
                                           const std::array<double,M>& den) {
    int sign=1;
    bool zero=false;
    for(double x:num) {
        if(!std::isfinite(x))return std::numeric_limits<double>::quiet_NaN();
        if(std::signbit(x))sign=-sign;
        zero=zero||(x==0.0);
    }
    for(double x:den) {
        if(!std::isfinite(x)||x==0.0)
            return std::numeric_limits<double>::quiet_NaN();
        if(std::signbit(x))sign=-sign;
    }
    if(zero)return std::copysign(0.0,sign<0?-1.0:1.0);

    double mantissa=1.0;
    int exponent=0;
    auto renormalize=[&]() {
        int e=0;
        mantissa=std::frexp(mantissa,&e);
        exponent+=e;
    };
    for(double x:num) {
        int e=0;
        mantissa*=std::frexp(std::fabs(x),&e);
        exponent+=e;
        renormalize();
    }
    for(double x:den) {
        int e=0;
        mantissa/=std::frexp(std::fabs(x),&e);
        exponent-=e;
        renormalize();
    }
    return std::copysign(std::scalbn(mantissa,exponent),sign<0?-1.0:1.0);
}

// Integral weights for the degree-13 Hermite interpolant on the seven
// Fejer-II level-3 nodes, ordered from +cos(pi/8) to -cos(pi/8).
// Generated at high precision from the cardinal Hermite basis.
struct SameNodeHermite7 {
    static constexpr std::array<double, 7> value_weights{{
        0.113598649409033012250142,
        0.271905871905871905871906,
        0.394337858527474924257794,
        0.440315240315240315240315,
        0.394337858527474924257794,
        0.271905871905871905871906,
        0.113598649409033012250142}};
    static constexpr std::array<double, 7> derivative_weights{{
        0.004903180563873449869723,
        0.015493715019005936432093,
        0.015283029566226087563276,
        0.0,
       -0.015283029566226087563276,
       -0.015493715019005936432093,
       -0.004903180563873449869723}};
    static constexpr std::array<double, 3> value3_weights{{
        8.0/15.0, 14.0/15.0, 8.0/15.0}};
    static constexpr std::array<double, 3> derivative3_weights{{
        0.04714045207910317, 0.0, -0.04714045207910317}};

    static double integrate7(const std::array<double, 7>& f,
                            const std::array<double, 7>& df) {
        double s=0,c=0;
        for(int i=0;i<7;++i) {
            const double x=value_weights[i]*f[i]+derivative_weights[i]*df[i];
            const double t=s+x;
            c += (std::fabs(s)>=std::fabs(x)) ? (s-t)+x : (x-t)+s;
            s=t;
        }
        return s+c;
    }
    static double integrate3(const std::array<double, 7>& f,
                            const std::array<double, 7>& df) {
        constexpr int ix[3]={1,3,5};
        double s=0,c=0;
        for(int i=0;i<3;++i) {
            const double x=value3_weights[i]*f[ix[i]]+
                           derivative3_weights[i]*df[ix[i]];
            const double t=s+x;
            c += (std::fabs(s)>=std::fabs(x)) ? (s-t)+x : (x-t)+s;
            s=t;
        }
        return s+c;
    }

    // Chain rule for a panel-local coordinate xi when the radial map is
    // evaluated in its cell coordinate x.  dR_dxi already contains h;
    // only the second-derivative term needs the explicit h^2 factor.
    static double mapped_slope(double F_over_norm, double FR_over_norm,
                               double dR_dxi, double d2R_dx2,
                               double panel_half_width) {
        return FR_over_norm*dR_dxi*dR_dxi +
               F_over_norm*d2R_dx2*panel_half_width*panel_half_width;
    }

    // Fold path: the callback already formed (F_R/N)*(dR/dxi)^2 in scaled
    // arithmetic, so a divergent fixed-R derivative is never multiplied by a
    // vanishing map Jacobian in binary64.
    static double mapped_slope_fold_regular(double FR_J2_over_norm,
                                            double F_over_norm,
                                            double d2R_dx2,
                                            double panel_half_width) {
        return FR_J2_over_norm +
               F_over_norm*d2R_dx2*panel_half_width*panel_half_width;
    }
};

} // namespace lcbinint::holonomic

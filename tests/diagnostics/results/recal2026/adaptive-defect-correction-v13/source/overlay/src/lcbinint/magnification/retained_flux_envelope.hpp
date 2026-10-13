#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace lcbinint::magnification::detail {
// If the old true residual lies in signed_error +/- remainder, changing
// the retained numerical flux by delta changes that residual by -delta.
// This transports an existing envelope; it does not certify its origin.
struct RetainedFluxEnvelope {
    double signed_error=0,remainder=0;
    bool valid=false;
    double indicator() const {
        if(!valid)return std::numeric_limits<double>::infinity();
        return std::nextafter(std::abs(signed_error)+remainder,
                              std::numeric_limits<double>::infinity());
    }
};
inline RetainedFluxEnvelope shift_retained_flux_envelope(
    double signed_error,double remainder,double delta,double delta_uncertainty=0) {
    if(!std::isfinite(signed_error)||!std::isfinite(remainder)||remainder<0||
       !std::isfinite(delta)||!std::isfinite(delta_uncertainty)||delta_uncertainty<0)return {};
    const double inf=std::numeric_limits<double>::infinity();
    const auto down=[&](double x){return std::nextafter(x,-inf);};
    const auto up=[&](double x){return std::nextafter(x,inf);};
    const double lower=down(down(signed_error-remainder)-up(delta+delta_uncertainty));
    const double upper=up(up(signed_error+remainder)-down(delta-delta_uncertainty));
    const double centre=signed_error-delta;
    const double width=up(std::max(up(centre-lower),up(upper-centre)));
    if(!std::isfinite(centre)||!std::isfinite(width))return {};
    return {centre,width,true};
}
}

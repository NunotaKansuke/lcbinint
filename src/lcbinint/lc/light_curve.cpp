#include "light_curve.hpp"
#include <stdexcept>

namespace lcbinint::lc {

LightCurve::LightCurve(lcbi_options opts, double ld_c, double ld_d)
    : opts_(opts), ld_c_(ld_c), ld_d_(ld_d)
{}

std::vector<double> LightCurve::magnification(
    const std::vector<double>& times,
    const lcbi_params&         params) const
{
    lcbi_params p = params;
    if (ld_c_ != 0.0 || ld_d_ != 0.0) {
        p.limb_darkening_c = ld_c_;
        p.limb_darkening_d = ld_d_;
    }

    const int n = static_cast<int>(times.size());
    std::vector<lcbi_result> results(n);
    const lcbi_status status =
        lcbi_magnification_array(times.data(), n, &p, &opts_, results.data());
    if (status != LCBI_OK)
        throw std::runtime_error(lcbi_status_string(status));

    std::vector<double> mags(n);
    for (int i = 0; i < n; ++i)
        mags[i] = results[i].magnification;
    return mags;
}

} // namespace lcbinint::lc

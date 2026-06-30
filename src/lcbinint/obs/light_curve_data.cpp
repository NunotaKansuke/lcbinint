#include "light_curve_data.hpp"
#include <stdexcept>

namespace lcbinint::obs {

LightCurveData::LightCurveData(
    std::vector<double>   time,
    std::vector<double>   flux,
    std::vector<double>   flux_err,
    std::string           name,
    std::string           band,
    std::string           observatory,
    std::shared_ptr<Site> site)
    : time_(std::move(time))
    , flux_(std::move(flux))
    , flux_err_(std::move(flux_err))
    , name_(std::move(name))
    , band_(std::move(band))
    , observatory_(std::move(observatory))
    , site_(std::move(site))
{
    if (flux_.size() != time_.size() || flux_err_.size() != time_.size())
        throw std::invalid_argument("time, flux, flux_err must have the same length");

    weight_.resize(time_.size());
    for (std::size_t i = 0; i < flux_err_.size(); ++i) {
        const double s = flux_err_[i];
        weight_[i] = (s > 0.0) ? 1.0 / (s * s) : 0.0;
    }
}

} // namespace lcbinint::obs

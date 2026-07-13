#include "light_curve_data.hpp"
#include <cmath>
#include <stdexcept>

namespace lcbinint::obs {

LightCurveData::LightCurveData(
    std::vector<double>   time,
    std::vector<double>   flux,
    std::vector<double>   flux_err,
    std::string           name,
    std::string           band,
    std::string           observatory,
    std::shared_ptr<Site> site,
    double                k,
    double                emin)
    : time_(std::move(time))
    , flux_(std::move(flux))
    , flux_err_(std::move(flux_err))
    , k_(k)
    , emin_(emin)
    , name_(std::move(name))
    , band_(std::move(band))
    , observatory_(std::move(observatory))
    , site_(std::move(site))
{
    if (flux_.size() != time_.size() || flux_err_.size() != time_.size())
        throw std::invalid_argument("time, flux, flux_err must have the same length");
    if (!std::isfinite(k_) || k_ <= 0.0)
        throw std::invalid_argument("LightCurveData: k must be finite and positive");
    if (!std::isfinite(emin_) || emin_ < 0.0)
        throw std::invalid_argument("LightCurveData: emin must be finite and non-negative");

    effective_flux_err_.resize(time_.size());
    weight_.resize(time_.size());
    for (std::size_t i = 0; i < flux_err_.size(); ++i) {
        const double s = flux_err_[i];
        const double s_eff = (s >= 0.0 && std::isfinite(s))
            ? k_ * std::sqrt(s * s + emin_ * emin_) : 0.0;
        effective_flux_err_[i] = s_eff;
        weight_[i] = (s_eff > 0.0) ? 1.0 / (s_eff * s_eff) : 0.0;
    }
}

} // namespace lcbinint::obs

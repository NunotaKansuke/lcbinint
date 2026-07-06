#pragma once
#include "coordinates.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace lcbinint::obs {

class LightCurveData {
public:
    LightCurveData(
        std::vector<double>   time,
        std::vector<double>   flux,
        std::vector<double>   flux_err,
        std::string           name        = {},
        std::string           band        = {},
        std::string           observatory = {},
        std::shared_ptr<Site> site        = nullptr,
        double                k           = 1.0,
        double                emin        = 0.0
    );

    const std::vector<double>& time()      const noexcept { return time_; }
    const std::vector<double>& flux()      const noexcept { return flux_; }
    const std::vector<double>& flux_err()  const noexcept { return flux_err_; }
    const std::vector<double>& effective_flux_err() const noexcept { return effective_flux_err_; }
    const std::vector<double>& weight()    const noexcept { return weight_; }
    std::size_t                size()      const noexcept { return time_.size(); }
    double                     k()         const noexcept { return k_; }
    double                     emin()      const noexcept { return emin_; }

    const std::string&           name()        const noexcept { return name_; }
    const std::string&           band()        const noexcept { return band_; }
    const std::string&           observatory() const noexcept { return observatory_; }
    const std::shared_ptr<Site>& site()        const noexcept { return site_; }

private:
    std::vector<double>   time_;
    std::vector<double>   flux_;
    std::vector<double>   flux_err_;
    std::vector<double>   effective_flux_err_;
    std::vector<double>   weight_;   // precomputed 1/sigma_eff^2
    double                k_ = 1.0;
    double                emin_ = 0.0;
    std::string           name_;
    std::string           band_;
    std::string           observatory_;
    std::shared_ptr<Site> site_;
};

} // namespace lcbinint::obs

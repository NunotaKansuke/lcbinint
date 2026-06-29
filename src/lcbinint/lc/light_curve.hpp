#pragma once
#include "evaluator.hpp"
#include "parameters.hpp"
#include "lcbinint/lcbinint.h"
#include <vector>

namespace lcbinint::lc {

// Concrete IEvaluator: calls lcbi_magnification_array with stored options.
// LimbDarkening stored here overrides params.limb_darkening_c/d when non-zero.
class LightCurve : public IEvaluator {
public:
    explicit LightCurve(
        lcbi_options opts = lcbi_default_options(),
        double       ld_c = 0.0,
        double       ld_d = 0.0
    );

    std::vector<double> magnification(
        const std::vector<double>& times,
        const lcbi_params&         params
    ) const override;

    std::vector<double> magnification(
        const std::vector<double>& times,
        const Parameters&          params
    ) const;

    const lcbi_options& options() const noexcept { return opts_; }
    double              ld_c()    const noexcept { return ld_c_; }
    double              ld_d()    const noexcept { return ld_d_; }

private:
    lcbi_options opts_;
    double       ld_c_;
    double       ld_d_;
};

} // namespace lcbinint::lc

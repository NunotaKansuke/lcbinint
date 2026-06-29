#pragma once
#include "lcbinint/lcbinint.h"
#include <vector>

namespace lcbinint::lc {

// Standalone Python-facing evaluator.
// Holds lcbi_options and optional limb-darkening; calls lcbi_magnification_array directly.
class LightCurve {
public:
    explicit LightCurve(
        lcbi_options opts = lcbi_default_options(),
        double       ld_c = 0.0,
        double       ld_d = 0.0
    );

    std::vector<double> magnification(
        const std::vector<double>& times,
        const lcbi_params&         params
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

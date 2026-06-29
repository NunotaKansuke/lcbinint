#pragma once
#include "lcbinint/lcbinint.h"
#include <vector>

namespace lcbinint::lc {

// Abstract interface implemented by LightCurve and used by bayes::Model
// to compute magnification arrays without knowing about lens type or options.
class IEvaluator {
public:
    virtual ~IEvaluator() = default;

    virtual std::vector<double> magnification(
        const std::vector<double>& times,
        const lcbi_params& params
    ) const = 0;
};

} // namespace lcbinint::lc

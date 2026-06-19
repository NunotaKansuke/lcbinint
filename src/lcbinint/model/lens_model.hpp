#pragma once

#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/types.hpp"

namespace lcbinint::model {

class LensModel {
public:
    LensModel(LensParameters params, ComputationOptions options);

    MagnificationResult magnification(double time) const;

private:
    LensParameters params_;
    ComputationOptions options_;
};

} // namespace lcbinint::model

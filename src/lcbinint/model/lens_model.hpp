#pragma once

#include "lcbinint/magnification/finite_source_magnifier.hpp"
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
    magnification::FiniteSourceMagnifier finite_magnifier_;
};

} // namespace lcbinint::model

#pragma once

#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/types.hpp"

namespace lcbinint::model {

class Trajectory {
public:
    explicit Trajectory(LensParameters params) : params_(params) {}

    SourcePosition source_position(double time, bool vbm_mode = false) const;

private:
    LensParameters params_;
};

} // namespace lcbinint::model

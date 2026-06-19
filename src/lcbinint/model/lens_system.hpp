#pragma once

#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/types.hpp"

namespace lcbinint::model {

struct LensBody {
    Complex position = {0.0, 0.0};
    double mass_fraction = 0.0;
};

class LensSystem {
public:
    static LensSystem from_parameters(const LensParameters &params);

    bool is_triple() const { return is_triple_; }
    const LensBody &primary() const { return primary_; }
    const LensBody &secondary() const { return secondary_; }
    const LensBody &tertiary() const { return tertiary_; }

private:
    bool is_triple_ = false;
    LensBody primary_;
    LensBody secondary_;
    LensBody tertiary_;
};

} // namespace lcbinint::model

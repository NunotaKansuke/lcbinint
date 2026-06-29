#include "parameters.hpp"

namespace lcbinint::lc {

Parameters::Parameters()
    : p_(lcbi_default_params())
{}

Parameters::Parameters(const lcbi_params& raw)
    : p_(raw)
{}

} // namespace lcbinint::lc

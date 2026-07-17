#include "lcbinint/lcbinint.h"

#include <cassert>
#include <cmath>

int main()
{
    const lcbi_params parameters = lcbi_default_params();
    const lcbi_options options = lcbi_default_options();
    lcbi_result result {};
    const lcbi_status status =
        lcbi_magnification(0.25, &parameters, &options, &result);
    assert(status == LCBI_OK);
    assert(std::isfinite(result.magnification));
    assert(result.magnification >= 1.0);
}

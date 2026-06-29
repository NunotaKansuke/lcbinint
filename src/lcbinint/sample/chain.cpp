#include "chain.hpp"

namespace lcbinint::sample {

void Chain::push(std::vector<double> theta, double lp)
{
    samples_.push_back(std::move(theta));
    log_prob_.push_back(lp);
}

} // namespace lcbinint::sample

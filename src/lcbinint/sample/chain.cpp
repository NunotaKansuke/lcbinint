#include "chain.hpp"
#include <stdexcept>

namespace lcbinint::sample {

void Chain::init(int nsteps, int nwalkers, int ndim)
{
    nsteps_     = nsteps;
    nwalkers_   = nwalkers;
    ndim_       = ndim;
    step_count_ = 0;
    flat_samples_ .assign(static_cast<std::size_t>(nsteps * nwalkers * ndim), 0.0);
    flat_log_prob_.assign(static_cast<std::size_t>(nsteps * nwalkers), 0.0);
}

void Chain::push_step(const std::vector<double>& positions,
                      const std::vector<double>& log_probs)
{
    const std::size_t base_s = static_cast<std::size_t>(step_count_ * nwalkers_ * ndim_);
    const std::size_t base_l = static_cast<std::size_t>(step_count_ * nwalkers_);
    std::copy(positions.begin(), positions.end(), flat_samples_.begin() + base_s);
    std::copy(log_probs.begin(), log_probs.end(), flat_log_prob_.begin() + base_l);
    ++step_count_;
}

} // namespace lcbinint::sample

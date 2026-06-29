#pragma once
#include "chain.hpp"
#include "../bayes/model.hpp"
#include "../optimize/result.hpp"

namespace lcbinint::sample {

class EnsembleSampler {
public:
    explicit EnsembleSampler(int nwalkers = 64, unsigned int seed = 0);

    Chain run(
        bayes::Model&               model,
        int                         nsteps = 1000,
        int                         burnin = 0,
        const optimize::Result*     start  = nullptr
    );

private:
    int          nwalkers_;
    unsigned int seed_;
};

} // namespace lcbinint::sample

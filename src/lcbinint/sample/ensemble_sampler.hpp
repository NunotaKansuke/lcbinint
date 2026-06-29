#pragma once
#include "chain.hpp"
#include "../bayes/model.hpp"
#include "../optimize/result.hpp"
#include <vector>

namespace lcbinint::sample {

// Affine-invariant ensemble sampler (emcee stretch-move).
// Operates entirely in C++ with no Python callbacks in the inner loop.
// All walker positions are in transformed space (log for LogUniform priors).
class EnsembleSampler {
public:
    explicit EnsembleSampler(
        int          nwalkers = 64,
        unsigned int seed     = 0,
        double       a        = 2.0   // stretch scale factor (default 2)
    );

    // Start from random positions sampled uniformly within prior bounds.
    Chain run(bayes::Model& model, int nsteps = 1000, int burnin = 0);

    // Start walkers in a tight gaussian ball around a best-fit result.
    Chain run(bayes::Model& model, const optimize::Result& start,
              int nsteps = 1000, int burnin = 0);

    // Start from explicit walker positions (shape: nwalkers × ndim).
    Chain run(bayes::Model& model,
              const std::vector<std::vector<double>>& start_pos,
              int nsteps = 1000, int burnin = 0);

private:
    Chain run_impl(bayes::Model& model,
                   std::vector<std::vector<double>> init_pos,
                   int nsteps, int burnin);

    int          nwalkers_;
    unsigned int seed_;
    double       a_;
};

} // namespace lcbinint::sample

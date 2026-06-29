#include "ensemble_sampler.hpp"
#include <stdexcept>

namespace lcbinint::sample {

EnsembleSampler::EnsembleSampler(int nwalkers, unsigned int seed)
    : nwalkers_(nwalkers), seed_(seed)
{}

Chain EnsembleSampler::run(
    bayes::Model& model,
    int nsteps,
    int burnin,
    const optimize::Result* start)
{
    // TODO: implement affine-invariant ensemble sampler (emcee-style)
    (void)model;
    (void)nsteps;
    (void)burnin;
    (void)start;
    throw std::runtime_error("EnsembleSampler::run: not yet implemented");
}

} // namespace lcbinint::sample

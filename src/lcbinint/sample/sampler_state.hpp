#pragma once
#include <random>
#include <vector>

namespace lcbinint::sample {

// Mutable state of an ensemble sampler run.
// Flat (row-major) storage for zero-copy numpy exposure.
// The RNG lives here so step-by-step results are reproducible
// regardless of how many steps are batched per call.
struct SamplerState {
    int nwalkers  = 0;
    int ndim      = 0;
    int n_fluxes  = 0;

    std::vector<double> pos;       // nwalkers * ndim,     row-major
    std::vector<double> log_prob;  // nwalkers
    std::vector<double> fluxes;    // nwalkers * n_fluxes

    long long n_accepted = 0;
    long long n_total    = 0;
    int       n_step     = 0;

    std::mt19937_64 rng;

    double* pos_row(int w)             { return pos.data()    + w * ndim; }
    const double* pos_row(int w) const { return pos.data()    + w * ndim; }
    double* flux_row(int w)            { return fluxes.data() + w * n_fluxes; }
    const double* flux_row(int w) const{ return fluxes.data() + w * n_fluxes; }

    std::vector<double> pos_vec(int w) const {
        return {pos_row(w), pos_row(w) + ndim};
    }
    double acceptance_fraction() const {
        return n_total > 0 ? static_cast<double>(n_accepted)
                           / static_cast<double>(n_total) : 0.0;
    }
};

} // namespace lcbinint::sample

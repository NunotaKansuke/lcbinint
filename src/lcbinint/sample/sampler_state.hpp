#pragma once
#include <random>
#include <vector>

namespace lcbinint::sample {

// Mutable state of an ensemble sampler run.
// Flat (row-major) storage for zero-copy numpy exposure.
// The RNG lives here so step-by-step results are reproducible
// regardless of how many steps are batched per call.
//
// History accumulation: every call to EnsembleSampler::step() appends
// the current walker positions to history/hist_lp/hist_fl, so the full
// chain is always available in memory. Call reset_history() to free it
// (e.g. after flushing a chunk to disk).
struct SamplerState {
    int nwalkers  = 0;
    int ndim      = 0;
    int n_fluxes  = 0;

    // Current walker positions (updated in-place by step())
    std::vector<double> pos;       // nwalkers * ndim,     row-major
    std::vector<double> log_prob;  // nwalkers
    std::vector<double> fluxes;    // nwalkers * n_fluxes

    // Accumulated history — grows by one step on every step() call.
    // Layout: history[s * nwalkers * ndim + w * ndim + j]
    std::vector<double> history;   // n_step * nwalkers * ndim
    std::vector<double> hist_lp;   // n_step * nwalkers
    std::vector<double> hist_fl;   // n_step * nwalkers * n_fluxes

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

    // Discard accumulated history (e.g. after writing a chunk to disk).
    // Does not reset n_step, n_accepted, n_total, or current positions.
    void reset_history() {
        history.clear(); history.shrink_to_fit();
        hist_lp.clear(); hist_lp.shrink_to_fit();
        hist_fl.clear(); hist_fl.shrink_to_fit();
    }
};

} // namespace lcbinint::sample

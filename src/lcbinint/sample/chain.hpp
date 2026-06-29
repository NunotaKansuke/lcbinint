#pragma once
#include <string>
#include <vector>

namespace lcbinint::sample {

// Flat-storage MCMC chain for an ensemble sampler.
// Layout: flat_samples_[step * nwalkers * ndim + walker * ndim + param]
//         flat_log_prob_[step * nwalkers + walker]
// (burn-in steps are stripped before storage)
class Chain {
public:
    Chain() = default;

    // Called by sampler to reserve storage before the run
    void init(int nsteps, int nwalkers, int ndim);

    // Called by sampler to store one completed step (all walkers)
    void push_step(const std::vector<double>& positions,  // nwalkers * ndim
                   const std::vector<double>& log_probs); // nwalkers

    void set_acceptance(double f)                      noexcept { acceptance_ = f; }
    void set_param_names(std::vector<std::string> ns)           { param_names_ = std::move(ns); }

    int    nsteps()   const noexcept { return nsteps_; }
    int    nwalkers() const noexcept { return nwalkers_; }
    int    ndim()     const noexcept { return ndim_; }
    double acceptance() const noexcept { return acceptance_; }

    // Flat views for Python/numpy export: (nsteps*nwalkers, ndim) and (nsteps*nwalkers,)
    const std::vector<double>& flat_samples()  const noexcept { return flat_samples_; }
    const std::vector<double>& flat_log_prob() const noexcept { return flat_log_prob_; }

    const std::vector<std::string>& param_names() const noexcept { return param_names_; }

private:
    int    nsteps_   = 0;
    int    nwalkers_ = 0;
    int    ndim_     = 0;
    double acceptance_ = 0.0;
    int    step_count_ = 0;

    std::vector<double>      flat_samples_;   // (nsteps * nwalkers * ndim)
    std::vector<double>      flat_log_prob_;  // (nsteps * nwalkers)
    std::vector<std::string> param_names_;
};

} // namespace lcbinint::sample

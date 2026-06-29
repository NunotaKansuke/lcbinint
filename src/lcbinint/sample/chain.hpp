#pragma once
#include <string>
#include <vector>

namespace lcbinint::sample {

// Flat-storage MCMC chain for an ensemble sampler.
// Layout: flat_samples_[step * nwalkers * ndim + walker * ndim + param]
//         flat_log_prob_[step * nwalkers + walker]
//         flat_fluxes_[step * nwalkers * n_fluxes + walker * n_fluxes + k]
//           where n_fluxes = n_datasets * 2  (Fs, Fb per dataset)
// (burn-in steps are stripped before storage)
class Chain {
public:
    Chain() = default;

    // Called by sampler to reserve storage before the run
    void init(int nsteps, int nwalkers, int ndim);
    void init_fluxes(int n_datasets,
                     std::vector<std::string> dataset_names);

    // Called by sampler to store one completed step (all walkers)
    void push_step(const std::vector<double>& positions,   // nwalkers * ndim
                   const std::vector<double>& log_probs);  // nwalkers

    void push_step(const std::vector<double>& positions,   // nwalkers * ndim
                   const std::vector<double>& log_probs,   // nwalkers
                   const std::vector<double>& fluxes);     // nwalkers * n_fluxes

    void set_acceptance(double f)                      noexcept { acceptance_ = f; }
    void set_param_names(std::vector<std::string> ns)           { param_names_ = std::move(ns); }
    void set_transforms(std::vector<std::string> ts)            { transforms_  = std::move(ts); }

    int    nsteps()    const noexcept { return nsteps_; }
    int    nwalkers()  const noexcept { return nwalkers_; }
    int    ndim()      const noexcept { return ndim_; }
    int    n_fluxes()  const noexcept { return n_fluxes_; }
    double acceptance() const noexcept { return acceptance_; }

    // Flat views for Python/numpy export
    const std::vector<double>& flat_samples()  const noexcept { return flat_samples_; }
    const std::vector<double>& flat_log_prob() const noexcept { return flat_log_prob_; }
    const std::vector<double>& flat_fluxes()   const noexcept { return flat_fluxes_; }

    const std::vector<std::string>& param_names()    const noexcept { return param_names_; }
    const std::vector<std::string>& transforms()     const noexcept { return transforms_; }
    const std::vector<std::string>& dataset_names()  const noexcept { return dataset_names_; }

    // Integrated autocorrelation time per parameter (Sokal auto-window).
    // Uses FFT-based estimator averaged over walkers.
    // c: window multiplier (emcee default = 5).
    // Returns NaN per parameter if chain is too short to converge.
    std::vector<double> tau(double c = 5.0) const;

    // Effective sample size = nsteps * nwalkers / tau.
    std::vector<double> ess() const;

private:
    int    nsteps_    = 0;
    int    nwalkers_  = 0;
    int    ndim_      = 0;
    int    n_fluxes_  = 0;   // n_datasets * 2
    double acceptance_ = 0.0;
    int    step_count_ = 0;

    std::vector<double>      flat_samples_;    // (nsteps * nwalkers * ndim)
    std::vector<double>      flat_log_prob_;   // (nsteps * nwalkers)
    std::vector<double>      flat_fluxes_;     // (nsteps * nwalkers * n_fluxes)
    std::vector<std::string> param_names_;
    std::vector<std::string> transforms_;      // "log" or "identity" per dim
    std::vector<std::string> dataset_names_;
};

} // namespace lcbinint::sample

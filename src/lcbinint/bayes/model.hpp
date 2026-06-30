#pragma once
#include "prior.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/lc/light_curve.hpp"
#include "lcbinint/obs/event.hpp"
#include <memory>
#include <string>
#include <vector>

namespace lcbinint::bayes {

// Optimizer/sampler space transform applied per parameter.
// identity : theta = physical value        (Uniform priors)
// log      : theta = ln(physical value)    (LogUniform priors — auto-assigned)
enum class Transform { identity, log };

struct ParameterDef {
    std::string            name;
    std::shared_ptr<Prior> prior;
    Transform              transform   = Transform::identity;
    bool                   fixed       = false;
    double                 fixed_value = 0.0;
};

// Bounds in optimizer/sampler space (after transform).
struct OptimizerBounds {
    double lo;
    double hi;
};

// Binary source parameters extracted alongside lcbi_params.
// In coupled xallarap+binary mode, t0_2/u0_2 are unused (both sources share
// the CoM trajectory); q_mass = m2/m1 sets source 2's xallarap amplitude.
struct BinarySourceExtras {
    double q_source = 1.0;
    double t0_2     = 0.0;  // independent binary source only
    double u0_2     = 0.0;  // independent binary source only
    double q_mass   = 1.0;  // coupled xallarap+binary only (m2/m1)
};

// Per-dataset sums precomputed at construction + reusable evaluation buffers.
struct DatasetCache {
    double                   S_w   = 0.0;
    double                   S_wf  = 0.0;
    double                   S_wf2 = 0.0;
    std::vector<double>      mag_buf;
    std::vector<lcbi_result> res_buf;
    std::vector<lcbi_result> res_buf2;  // binary source: second source buffer
};

// Central class for Bayesian inference.
// Delegates magnification computation to lc::LightCurve (which owns the options and mode flags).
// Parameters are registered with priors; LogUniform priors automatically use log transform.
// Optimizers and samplers work in transformed space (theta_internal); the model converts to
// physical params via theta_to_params before magnification evaluation.
class Model {
public:
    Model(std::shared_ptr<lc::LightCurve> light_curve, std::shared_ptr<obs::Event> event);
    Model(std::shared_ptr<lc::LightCurve> light_curve, std::shared_ptr<obs::LightCurveData> data);

    void param(std::string name, std::shared_ptr<Prior> prior);
    void flux(std::string mode = "linear_blend");
    void likelihood(std::string mode = "gaussian");

    int                              n_params()      const noexcept;
    const std::vector<ParameterDef>& param_defs()    const noexcept { return params_; }
    const lc::LightCurve&            light_curve()   const noexcept { return *light_curve_; }
    const lcbi_options&              options()       const noexcept { return light_curve_->options(); }
    const obs::Event&                event()         const noexcept { return *event_; }

    // Bounds in optimizer/sampler (transformed) space.
    std::vector<OptimizerBounds> optimizer_bounds() const;

    // Total number of data points across all datasets.
    int n_data() const noexcept;

    // All public evaluation methods accept theta in transformed space.
    double log_prior(      const std::vector<double>& theta) const;
    double log_likelihood( const std::vector<double>& theta) const;
    double log_prob(       const std::vector<double>& theta) const;
    double chi2(           const std::vector<double>& theta) const;

    // Flat weighted residual vector r_i = (flux_i - Fs*A_i - Fb) / sigma_i
    // across all datasets (used by LevenbergMarquardt for Jacobian construction).
    std::vector<double> residuals(const std::vector<double>& theta) const;

    // Linear flux parameters {Fs, Fb} per dataset, solved analytically.
    struct FluxSolution { double Fs; double Fb; };
    std::vector<FluxSolution> fluxes(const std::vector<double>& theta) const;

    // Compute log_prob and fluxes in a single magnification pass.
    // Used by EnsembleSampler to avoid double-calling lcbi_magnification_array on accept.
    double log_prob_and_fluxes(const std::vector<double>& theta,
                               std::vector<FluxSolution>& out_fluxes) const;

private:
    // Convert transformed theta → lcbi_params + binary source extras.
    // Applies sky coords from LightCurve/Event as defaults; free params override.
    lcbi_params theta_to_params(const std::vector<double>& theta,
                                BinarySourceExtras& bs_out) const;
    double              compute_chi2(const lcbi_params& p, const BinarySourceExtras& bs) const;
    std::vector<double> compute_residuals(const lcbi_params& p, const BinarySourceExtras& bs) const;
    void                build_cache();

    std::shared_ptr<lc::LightCurve> light_curve_;
    std::shared_ptr<obs::Event>     event_;
    std::vector<ParameterDef>   params_;
    std::string                 flux_mode_{"linear_blend"};
    std::string                 likelihood_mode_{"gaussian"};

    mutable std::vector<DatasetCache> cache_;
};

} // namespace lcbinint::bayes

#pragma once
#include "prior.hpp"
#include "lcbinint/lcbinint.h"
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

// Per-dataset sums precomputed at construction + reusable evaluation buffers.
struct DatasetCache {
    double                   S_w   = 0.0;
    double                   S_wf  = 0.0;
    double                   S_wf2 = 0.0;
    std::vector<double>      mag_buf;
    std::vector<lcbi_result> res_buf;
};

// Central class for Bayesian inference.
// Takes lcbi_options directly; calls lcbi_magnification_array in the C++ hot path.
// Parameters are registered with priors; LogUniform priors automatically use log transform.
// Optimizers and samplers work in transformed space (theta_internal); the model converts to
// physical params via theta_to_params before magnification evaluation.
class Model {
public:
    Model(lcbi_options options, std::shared_ptr<obs::Event> event);
    Model(lcbi_options options, std::shared_ptr<obs::LightCurveData> data);

    void param(std::string name, std::shared_ptr<Prior> prior);
    void flux(std::string mode = "linear_blend");
    void likelihood(std::string mode = "gaussian");

    int                              n_params()   const noexcept;
    const std::vector<ParameterDef>& param_defs() const noexcept { return params_; }
    const lcbi_options&              options()    const noexcept { return options_; }

    // Bounds in optimizer/sampler (transformed) space.
    std::vector<OptimizerBounds> optimizer_bounds() const;

    // All public evaluation methods accept theta in transformed space.
    double log_prior(      const std::vector<double>& theta) const;
    double log_likelihood( const std::vector<double>& theta) const;
    double log_prob(       const std::vector<double>& theta) const;
    double chi2(           const std::vector<double>& theta) const;

private:
    // Convert transformed theta → lcbi_params (physical values).
    lcbi_params theta_to_params(const std::vector<double>& theta) const;
    double      compute_chi2(const lcbi_params& p) const;
    void        build_cache();

    lcbi_options                options_;
    std::shared_ptr<obs::Event> event_;
    std::vector<ParameterDef>   params_;
    std::string                 flux_mode_{"linear_blend"};
    std::string                 likelihood_mode_{"gaussian"};

    mutable std::vector<DatasetCache> cache_;
};

} // namespace lcbinint::bayes

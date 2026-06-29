#include "model.hpp"
#include <cmath>
#include <stdexcept>

namespace lcbinint::bayes {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Model::Model(lcbi_options options, std::shared_ptr<obs::Event> event)
    : options_(options), event_(std::move(event))
{
    if (!event_) throw std::invalid_argument("event must not be null");
    build_cache();
}

Model::Model(lcbi_options options, std::shared_ptr<obs::LightCurveData> data)
    : options_(options)
    , event_(std::make_shared<obs::Event>())
{
    if (!data) throw std::invalid_argument("data must not be null");
    event_->add(std::move(data));
    build_cache();
}

void Model::build_cache()
{
    const std::size_t n_ds = event_->size();
    cache_.resize(n_ds);
    for (std::size_t k = 0; k < n_ds; ++k) {
        const auto& ds = event_->at(k);
        const std::size_t n = ds.size();
        const double* __restrict__ f = ds.flux().data();
        const double* __restrict__ w = ds.weight().data();

        DatasetCache& c = cache_[k];
        c.mag_buf.resize(n);
        c.res_buf.resize(n);

        double S_w = 0.0, S_wf = 0.0, S_wf2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double wi = w[i], fi = f[i];
            S_w   += wi;
            S_wf  += wi * fi;
            S_wf2 += wi * fi * fi;
        }
        c.S_w = S_w; c.S_wf = S_wf; c.S_wf2 = S_wf2;
    }
}

// ---------------------------------------------------------------------------
// Parameter registration
// ---------------------------------------------------------------------------

void Model::param(std::string name, std::shared_ptr<Prior> prior)
{
    if (!prior) throw std::invalid_argument("prior must not be null");
    // Auto-assign log transform for LogUniform priors.
    Transform tr = Transform::identity;
    if (dynamic_cast<LogUniform*>(prior.get())) tr = Transform::log;
    params_.push_back({std::move(name), std::move(prior), tr});
}

void Model::flux(std::string mode)
{
    if (mode != "linear_blend")
        throw std::invalid_argument("unsupported flux mode: " + mode);
    flux_mode_ = std::move(mode);
}

void Model::likelihood(std::string mode)
{
    if (mode != "gaussian")
        throw std::invalid_argument("unsupported likelihood mode: " + mode);
    likelihood_mode_ = std::move(mode);
}

int Model::n_params() const noexcept
{
    int n = 0;
    for (const auto& p : params_)
        if (!p.fixed) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// Optimizer bounds (in transformed space)
// ---------------------------------------------------------------------------

std::vector<OptimizerBounds> Model::optimizer_bounds() const
{
    std::vector<OptimizerBounds> out;
    out.reserve(params_.size());
    for (const auto& def : params_) {
        if (def.fixed) continue;
        auto b = def.prior->bounds();
        if (def.transform == Transform::log) {
            out.push_back({std::log(b.lo), std::log(b.hi)});
        } else {
            out.push_back({b.lo, b.hi});
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// theta_to_params: transformed space → lcbi_params (physical)
// ---------------------------------------------------------------------------

lcbi_params Model::theta_to_params(const std::vector<double>& theta) const
{
    lcbi_params p = lcbi_default_params();
    int idx = 0;
    for (const auto& def : params_) {
        double val;
        if (def.fixed) {
            val = def.fixed_value;
        } else {
            val = (def.transform == Transform::log)
                  ? std::exp(theta[idx]) : theta[idx];
            ++idx;
        }
        const std::string& n = def.name;
        if      (n == "t0")                   p.t0   = val;
        else if (n == "tE")                   p.tE   = val;
        else if (n == "u0"   || n == "umin")  p.umin = val;
        else if (n == "alpha"|| n == "theta") p.theta= val;
        else if (n == "s"    || n == "sep")   p.sep  = val;
        else if (n == "q")                    p.q    = val;
        else if (n == "rho")                  p.rho  = val;
        else if (n == "piEN")                 p.piEN = val;
        else if (n == "piEE")                 p.piEE = val;
        else if (n == "q2")                   p.q2   = val;
        else if (n == "sep2")                 p.sep2 = val;
        else if (n == "ang")                  p.ang  = val;
        else if (n == "ra")                   p.ra   = val;
        else if (n == "dec")                  p.dec  = val;
        else throw std::invalid_argument("Model: unknown parameter '" + n + "'");
    }
    return p;
}

// ---------------------------------------------------------------------------
// n_data
// ---------------------------------------------------------------------------

int Model::n_data() const noexcept
{
    int n = 0;
    for (std::size_t k = 0; k < event_->size(); ++k)
        n += static_cast<int>(event_->at(k).size());
    return n;
}

// ---------------------------------------------------------------------------
// compute_chi2: hot path — magnification + linear flux solve per dataset
// ---------------------------------------------------------------------------

double Model::compute_chi2(const lcbi_params& p) const
{
    double total = 0.0;
    for (std::size_t k = 0; k < event_->size(); ++k) {
        const auto& ds = event_->at(k);
        DatasetCache& c = cache_[k];
        const std::size_t n = ds.size();

        const lcbi_status status =
            lcbi_magnification_array(ds.time().data(), static_cast<int>(n),
                                     &p, &options_, c.res_buf.data());
        if (status != LCBI_OK)
            throw std::runtime_error(lcbi_status_string(status));

        for (std::size_t i = 0; i < n; ++i)
            c.mag_buf[i] = c.res_buf[i].magnification;

        const double* __restrict__ A = c.mag_buf.data();
        const double* __restrict__ f = ds.flux().data();
        const double* __restrict__ w = ds.weight().data();

        double S_wA = 0.0, S_wA2 = 0.0, S_wAf = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double wi = w[i], Ai = A[i], fi = f[i];
            S_wA  += wi * Ai;
            S_wA2 += wi * Ai * Ai;
            S_wAf += wi * Ai * fi;
        }
        const double D  = S_wA2 * c.S_w - S_wA * S_wA;
        const double Fs = (S_wAf * c.S_w  - S_wA  * c.S_wf) / D;
        const double Fb = (S_wA2 * c.S_wf - S_wA  * S_wAf)  / D;
        total += c.S_wf2 - Fs * S_wAf - Fb * c.S_wf;
    }
    return total;
}

// ---------------------------------------------------------------------------
// Public evaluation (theta in transformed space)
// ---------------------------------------------------------------------------

double Model::log_prior(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    double lp = 0.0;
    int idx = 0;
    for (const auto& def : params_) {
        if (def.fixed) continue;
        const double t = theta[idx++];
        if (def.transform == Transform::log) {
            // p(ln x) = p_physical(exp(t)) * exp(t)  (Jacobian factor)
            // log p(t) = log p_physical(exp(t)) + t
            lp += def.prior->log_prob(std::exp(t)) + t;
        } else {
            lp += def.prior->log_prob(t);
        }
        if (!std::isfinite(lp)) return lp;
    }
    return lp;
}

// ---------------------------------------------------------------------------
// compute_residuals: weighted residuals r_i = (flux_i - Fs*A_i - Fb) / sigma_i
// ---------------------------------------------------------------------------

std::vector<double> Model::compute_residuals(const lcbi_params& p) const
{
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(n_data()));

    for (std::size_t k = 0; k < event_->size(); ++k) {
        const auto& ds = event_->at(k);
        DatasetCache& c = cache_[k];
        const std::size_t n = ds.size();

        const lcbi_status status =
            lcbi_magnification_array(ds.time().data(), static_cast<int>(n),
                                     &p, &options_, c.res_buf.data());
        if (status != LCBI_OK)
            throw std::runtime_error(lcbi_status_string(status));
        for (std::size_t i = 0; i < n; ++i)
            c.mag_buf[i] = c.res_buf[i].magnification;

        const double* __restrict__ A = c.mag_buf.data();
        const double* __restrict__ f = ds.flux().data();
        const double* __restrict__ w = ds.weight().data();

        double S_wA = 0.0, S_wA2 = 0.0, S_wAf = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double wi = w[i], Ai = A[i];
            S_wA  += wi * Ai;
            S_wA2 += wi * Ai * Ai;
            S_wAf += wi * Ai * f[i];
        }
        const double D  = S_wA2 * c.S_w - S_wA * S_wA;
        const double Fs = (S_wAf * c.S_w  - S_wA  * c.S_wf) / D;
        const double Fb = (S_wA2 * c.S_wf - S_wA  * S_wAf)  / D;

        const double* __restrict__ sig = ds.flux_err().data();
        for (std::size_t i = 0; i < n; ++i)
            out.push_back((f[i] - Fs * A[i] - Fb) / sig[i]);
    }
    return out;
}

std::vector<double> Model::residuals(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    return compute_residuals(theta_to_params(theta));
}

double Model::chi2(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    return compute_chi2(theta_to_params(theta));
}

double Model::log_likelihood(const std::vector<double>& theta) const
{
    return -0.5 * chi2(theta);
}

std::vector<Model::FluxSolution>
Model::fluxes(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    const lcbi_params p = theta_to_params(theta);
    std::vector<FluxSolution> out;
    out.reserve(event_->size());
    for (std::size_t k = 0; k < event_->size(); ++k) {
        const auto& ds = event_->at(k);
        DatasetCache& c = cache_[k];
        const std::size_t n = ds.size();
        const lcbi_status status =
            lcbi_magnification_array(ds.time().data(), static_cast<int>(n),
                                     &p, &options_, c.res_buf.data());
        if (status != LCBI_OK)
            throw std::runtime_error(lcbi_status_string(status));
        for (std::size_t i = 0; i < n; ++i)
            c.mag_buf[i] = c.res_buf[i].magnification;
        const double* __restrict__ A = c.mag_buf.data();
        const double* __restrict__ f = ds.flux().data();
        const double* __restrict__ w = ds.weight().data();
        double S_wA = 0.0, S_wA2 = 0.0, S_wAf = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            S_wA  += w[i] * A[i];
            S_wA2 += w[i] * A[i] * A[i];
            S_wAf += w[i] * A[i] * f[i];
        }
        const double D  = S_wA2 * c.S_w - S_wA * S_wA;
        out.push_back({(S_wAf * c.S_w  - S_wA  * c.S_wf) / D,
                       (S_wA2 * c.S_wf - S_wA  * S_wAf)  / D});
    }
    return out;
}

double Model::log_prob(const std::vector<double>& theta) const
{
    const double lp = log_prior(theta);
    if (!std::isfinite(lp)) return lp;
    return lp + log_likelihood(theta);
}

} // namespace lcbinint::bayes

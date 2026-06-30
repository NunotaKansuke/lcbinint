#include "model.hpp"
#include <cmath>
#include <stdexcept>

namespace lcbinint::bayes {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Model::Model(std::shared_ptr<lc::LightCurve> light_curve, std::shared_ptr<obs::Event> event)
    : light_curve_(std::move(light_curve)), event_(std::move(event))
{
    if (!light_curve_) throw std::invalid_argument("light_curve must not be null");
    if (!event_)       throw std::invalid_argument("event must not be null");
    build_cache();
}

Model::Model(std::shared_ptr<lc::LightCurve> light_curve, std::shared_ptr<obs::LightCurveData> data)
    : light_curve_(std::move(light_curve))
    , event_(std::make_shared<obs::Event>())
{
    if (!light_curve_) throw std::invalid_argument("light_curve must not be null");
    if (!data)         throw std::invalid_argument("data must not be null");
    event_->add(std::move(data));
    build_cache();
}

void Model::build_cache()
{
    const bool is_binary = (light_curve_->source_kind() == lc::SourceKind::binary);
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
        if (is_binary) c.res_buf2.resize(n);

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
// theta_to_params: transformed space → lcbi_params + binary source extras
// ---------------------------------------------------------------------------

lcbi_params Model::theta_to_params(const std::vector<double>& theta,
                                    BinarySourceExtras& bs) const
{
    lcbi_params p = light_curve_->apply_coords(lcbi_default_params());
    if (!light_curve_->sky_coord() && event_->sky_coord()) {
        p.ra  = event_->sky_coord()->ra_deg();
        p.dec = event_->sky_coord()->dec_deg();
    }
    bs = {};  // reset binary source extras to defaults

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
        // --- standard microlensing ---
        if      (n == "t0"    || n == "t_0")   p.t0    = val;
        else if (n == "tE"    || n == "t_E")   p.tE    = val;
        else if (n == "u0"    || n == "umin")  p.umin  = val;
        else if (n == "alpha" || n == "theta") p.theta = val;
        else if (n == "s"     || n == "sep")   p.sep   = val;
        else if (n == "q")                     p.q     = val;
        else if (n == "rho")                   p.rho   = val;
        // --- parallax ---
        else if (n == "piEN")  p.piEN = val;
        else if (n == "piEE")  p.piEE = val;
        else if (n == "ra")    p.ra   = val;
        else if (n == "dec")   p.dec  = val;
        // --- triple lens ---
        else if (n == "q2")    p.q2   = val;
        else if (n == "sep2")  p.sep2 = val;
        else if (n == "ang")   p.ang  = val;
        // --- orbital motion (circular / Kepler) ---
        else if (n == "g1")      p.g1      = val;
        else if (n == "g2")      p.g2      = val;
        else if (n == "g3")      p.g3      = val;
        else if (n == "lom_szs") p.lom_szs = val;
        else if (n == "lom_ar")  p.lom_ar  = val;
        // --- xallarap (angular velocity mode) ---
        else if (n == "xi_1")     p.xi_1     = val;
        else if (n == "xi_2")     p.xi_2     = val;
        else if (n == "omega_xa") p.omega_xa = val;
        else if (n == "inc_xa")   p.inc_xa   = val;
        else if (n == "phi_xa")   p.phi_xa   = val;
        // --- xallarap (orbital elements mode) ---
        else if (n == "piEN_xa")  p.piEN_xa  = val;
        else if (n == "piEE_xa")  p.piEE_xa  = val;
        else if (n == "period_xa") p.period_xa = val;
        else if (n == "ecc_xa")   p.ecc_xa   = val;
        else if (n == "peri_xa")  p.peri_xa  = val;
        // --- binary source ---
        else if (n == "q_source" || n == "flux_ratio") bs.q_source = val;
        else if (n == "t0_2")  bs.t0_2  = val;
        else if (n == "u0_2")  bs.u0_2  = val;
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
// Magnification helper: fills c.mag_buf with effective magnification.
// For single source: one lcbi_magnification_array call.
// For binary source: two calls, combined as (A1 + q*A2)/(1+q).
// Per-dataset site override is applied zero-copy (pointer swap).
// ---------------------------------------------------------------------------

static void fill_magnification(
    const obs::LightCurveData& ds,
    DatasetCache&              c,
    const lcbi_params&         p,
    const BinarySourceExtras&  bs,
    const lcbi_options&        opts,
    bool                       is_binary,
    bool                       terrestrial,
    lcbi_params&               p_scratch)
{
    const std::size_t n = ds.size();
    const lcbi_params* pp = &p;
    // Dataset-level site overrides LightCurve-level site (already baked into p),
    // but only when terrestrial parallax is explicitly enabled.
    if (terrestrial && ds.site()) {
        p_scratch = p;
        p_scratch.obs_lat = ds.site()->lat_deg();
        p_scratch.obs_lon = ds.site()->lon_deg();
        pp = &p_scratch;
    }

    lcbi_status status = lcbi_magnification_array(
        ds.time().data(), static_cast<int>(n), pp, &opts, c.res_buf.data());
    if (status != LCBI_OK)
        throw std::runtime_error(lcbi_status_string(status));

    if (!is_binary) {
        for (std::size_t i = 0; i < n; ++i)
            c.mag_buf[i] = c.res_buf[i].magnification;
        return;
    }

    // Binary source: second source has different t0/u0.
    lcbi_params p2 = *pp;
    p2.t0   = bs.t0_2;
    p2.umin = bs.u0_2;
    status = lcbi_magnification_array(
        ds.time().data(), static_cast<int>(n), &p2, &opts, c.res_buf2.data());
    if (status != LCBI_OK)
        throw std::runtime_error(lcbi_status_string(status));

    const double denom = 1.0 + bs.q_source;
    for (std::size_t i = 0; i < n; ++i)
        c.mag_buf[i] = (c.res_buf[i].magnification
                        + bs.q_source * c.res_buf2[i].magnification) / denom;
}

// ---------------------------------------------------------------------------
// compute_chi2
// ---------------------------------------------------------------------------

double Model::compute_chi2(const lcbi_params& p, const BinarySourceExtras& bs) const
{
    const bool is_binary    = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial  = light_curve_->effects().terrestrial;
    const lcbi_options& opts = light_curve_->options();
    double total = 0.0;
    lcbi_params p_scratch;

    for (std::size_t k = 0; k < event_->size(); ++k) {
        const auto& ds = event_->at(k);
        DatasetCache& c = cache_[k];
        fill_magnification(ds, c, p, bs, opts, is_binary, terrestrial, p_scratch);

        const double* __restrict__ A = c.mag_buf.data();
        const double* __restrict__ f = ds.flux().data();
        const double* __restrict__ w = ds.weight().data();
        const std::size_t n = ds.size();

        double S_wA = 0.0, S_wA2 = 0.0, S_wAf = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double wi = w[i], Ai = A[i], fi = f[i];
            S_wA  += wi * Ai;
            S_wA2 += wi * Ai * Ai;
            S_wAf += wi * Ai * fi;
        }
        const double D = S_wA2 * c.S_w - S_wA * S_wA;
        if (D <= 0.0) return std::numeric_limits<double>::infinity();
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
            lp += def.prior->log_prob(std::exp(t)) + t;
        } else {
            lp += def.prior->log_prob(t);
        }
        if (!std::isfinite(lp)) return lp;
    }
    return lp;
}

// ---------------------------------------------------------------------------
// compute_residuals
// ---------------------------------------------------------------------------

std::vector<double> Model::compute_residuals(const lcbi_params& p,
                                              const BinarySourceExtras& bs) const
{
    const bool is_binary   = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial = light_curve_->effects().terrestrial;
    const lcbi_options& opts = light_curve_->options();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(n_data()));
    lcbi_params p_scratch;

    for (std::size_t k = 0; k < event_->size(); ++k) {
        const auto& ds = event_->at(k);
        DatasetCache& c = cache_[k];
        const std::size_t n = ds.size();
        fill_magnification(ds, c, p, bs, opts, is_binary, terrestrial, p_scratch);

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
    BinarySourceExtras bs;
    return compute_residuals(theta_to_params(theta, bs), bs);
}

double Model::chi2(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    BinarySourceExtras bs;
    return compute_chi2(theta_to_params(theta, bs), bs);
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
    BinarySourceExtras bs;
    const lcbi_params p = theta_to_params(theta, bs);
    const bool is_binary   = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial = light_curve_->effects().terrestrial;
    const lcbi_options& opts = light_curve_->options();

    std::vector<FluxSolution> out;
    out.reserve(event_->size());
    lcbi_params p_scratch;
    for (std::size_t k = 0; k < event_->size(); ++k) {
        const auto& ds = event_->at(k);
        DatasetCache& c = cache_[k];
        const std::size_t n = ds.size();
        fill_magnification(ds, c, p, bs, opts, is_binary, terrestrial, p_scratch);

        const double* __restrict__ A = c.mag_buf.data();
        const double* __restrict__ f = ds.flux().data();
        const double* __restrict__ w = ds.weight().data();
        double S_wA = 0.0, S_wA2 = 0.0, S_wAf = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            S_wA  += w[i] * A[i];
            S_wA2 += w[i] * A[i] * A[i];
            S_wAf += w[i] * A[i] * f[i];
        }
        const double D = S_wA2 * c.S_w - S_wA * S_wA;
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

double Model::log_prob_and_fluxes(const std::vector<double>& theta,
                                   std::vector<FluxSolution>& out_fluxes) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    const double lp = log_prior(theta);
    if (!std::isfinite(lp)) {
        out_fluxes.clear();
        return lp;
    }
    BinarySourceExtras bs;
    const lcbi_params p = theta_to_params(theta, bs);
    const bool is_binary   = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial = light_curve_->effects().terrestrial;
    const lcbi_options& opts = light_curve_->options();

    double chi2_total = 0.0;
    out_fluxes.clear();
    out_fluxes.reserve(event_->size());
    lcbi_params p_scratch;
    for (std::size_t k = 0; k < event_->size(); ++k) {
        const auto& ds = event_->at(k);
        DatasetCache& c = cache_[k];
        const std::size_t n = ds.size();
        fill_magnification(ds, c, p, bs, opts, is_binary, terrestrial, p_scratch);

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
        const double Fs = (S_wAf * c.S_w  - S_wA  * c.S_wf) / D;
        const double Fb = (S_wA2 * c.S_wf - S_wA  * S_wAf)  / D;
        chi2_total += c.S_wf2 - Fs * S_wAf - Fb * c.S_wf;
        out_fluxes.push_back({Fs, Fb});
    }
    return lp - 0.5 * chi2_total;
}

} // namespace lcbinint::bayes

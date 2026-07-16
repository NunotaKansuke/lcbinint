#include "model.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include <cmath>
#include <limits>
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

void Model::likelihood(std::string mode, double nu, std::string flux)
{
    if (mode != "gaussian" && mode != "student_t")
        throw std::invalid_argument("unsupported likelihood mode: " + mode);
    if (mode == "student_t" && (!std::isfinite(nu) || nu <= 0.0))
        throw std::invalid_argument("student_t likelihood requires finite nu > 0");
    if (flux != "fit" && flux != "sample" && flux != "marginalize")
        throw std::invalid_argument("unsupported flux treatment: " + flux);
    if (mode == "student_t" && flux == "marginalize")
        throw std::invalid_argument(
            "flux='marginalize' is currently supported only for gaussian likelihood");
    likelihood_mode_ = std::move(mode);
    likelihood_nu_ = nu;
    flux_treatment_ = std::move(flux);
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

namespace {

std::string flux_dataset_name(const obs::Event& event, std::size_t k)
{
    if (k < event.size() && !event.at(k).name().empty())
        return event.at(k).name();
    return "ds" + std::to_string(k);
}

bool match_flux_param(const obs::Event& event,
                      const std::string& name,
                      std::size_t& dataset_index,
                      bool& is_fs)
{
    for (std::size_t k = 0; k < event.size(); ++k) {
        const std::string ds_name = flux_dataset_name(event, k);
        if (name == "Fs_" + ds_name || name == "fs_" + ds_name) {
            dataset_index = k;
            is_fs = true;
            return true;
        }
        if (name == "Fb_" + ds_name || name == "fb_" + ds_name) {
            dataset_index = k;
            is_fs = false;
            return true;
        }
    }
    if (event.size() == 1) {
        if (name == "Fs" || name == "fs") {
            dataset_index = 0;
            is_fs = true;
            return true;
        }
        if (name == "Fb" || name == "fb") {
            dataset_index = 0;
            is_fs = false;
            return true;
        }
    }
    return false;
}

struct LinearFluxStats {
    Model::FluxSolution flux;
    double det  = std::numeric_limits<double>::quiet_NaN();
    double chi2 = std::numeric_limits<double>::quiet_NaN();
};

LinearFluxStats linear_flux_stats_for_dataset(const obs::LightCurveData& ds,
                                              const DatasetCache& c,
                                              const double* A)
{
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
    if (D <= 0.0)
        return {};
    const double Fs = (S_wAf * c.S_w  - S_wA  * c.S_wf) / D;
    const double Fb = (S_wA2 * c.S_wf - S_wA  * S_wAf)  / D;
    return {{Fs, Fb}, D, c.S_wf2 - Fs * S_wAf - Fb * c.S_wf};
}

Model::FluxSolution fit_flux_for_dataset(const obs::LightCurveData& ds,
                                         const DatasetCache& c,
                                         const double* A)
{
    return linear_flux_stats_for_dataset(ds, c, A).flux;
}

double marginalized_flux_scale_log_likelihood(const LinearFluxStats& stats,
                                              std::size_t n_data)
{
    constexpr std::size_t n_flux_params = 2;
    if (n_data <= n_flux_params)
        return -std::numeric_limits<double>::infinity();
    if (!std::isfinite(stats.chi2) || !std::isfinite(stats.det) || stats.det <= 0.0)
        return -std::numeric_limits<double>::infinity();
    if (stats.chi2 <= 0.0)
        return -std::numeric_limits<double>::infinity();

    const double dof = static_cast<double>(n_data - n_flux_params);
    // Constants that depend only on n_data and fixed error bars are omitted,
    // matching the existing Gaussian likelihood convention.
    return -0.5 * dof * std::log(stats.chi2) - 0.5 * std::log(stats.det);
}

} // namespace

lcbi_params Model::theta_to_params(const std::vector<double>& theta,
                                    BinarySourceExtras& bs,
                                    std::vector<FluxSolution>* fluxes_out) const
{
    lcbi_params p = light_curve_->apply_coords(lcbi_default_params());
    if (!light_curve_->sky_coord() && event_->sky_coord()) {
        p.ra  = event_->sky_coord()->ra_deg();
        p.dec = event_->sky_coord()->dec_deg();
    }
    bs = {};  // reset binary source extras to defaults
    if (fluxes_out) {
        fluxes_out->assign(
            event_->size(),
            {std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN()});
    }

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
        std::size_t flux_idx = 0;
        bool is_fs = false;
        if (match_flux_param(*event_, n, flux_idx, is_fs)) {
            if (flux_treatment_ != "sample")
                throw std::invalid_argument(
                    "flux parameter '" + n
                    + "' requires model.likelihood(..., flux='sample')");
            if (!fluxes_out)
                throw std::invalid_argument(
                    "flux='sample' parameter '" + n
                    + "' cannot be used in this context");
            if (is_fs)
                (*fluxes_out)[flux_idx].Fs = val;
            else
                (*fluxes_out)[flux_idx].Fb = val;
            continue;
        }
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
        // --- xallarap: amplitude/position (all modes use xi_1/xi_2) ---
        else if (n == "xi_1")     p.xi_1     = val;
        else if (n == "xi_2")     p.xi_2     = val;
        // orbital_elements / circular_elements: period-based orbit params
        else if (n == "period_xa") p.period_xa = val;
        else if (n == "ecc_xa")    p.ecc_xa    = val;
        else if (n == "peri_xa")   p.peri_xa   = val;
        else if (n == "inc_xa")    p.inc_xa    = val;
        // circular_velocity / kepler_velocity: w1/w2/w3 (mapped to omega/inc/phi fields)
        else if (n == "w1") p.omega_xa = val;
        else if (n == "w2") p.inc_xa   = val;
        else if (n == "w3") p.phi_xa   = val;
        // kepler_velocity: xa_szs/xa_ar (mapped to piEN_xa/piEE_xa fields)
        else if (n == "xa_szs") p.piEN_xa = val;
        else if (n == "xa_ar")  p.piEE_xa = val;
        // --- binary source ---
        // Coupled mode (xallarap + binary): q_mass sets source 2's xallarap
        // amplitude; t0_2/u0_2 are not free (both sources share CoM trajectory).
        // Independent mode: t0_2/u0_2 define source 2's separate trajectory.
        else if (n == "q_source" || n == "flux_ratio") bs.q_source = val;
        else if (n == "q_mass") bs.q_mass = val;
        else if (n == "t0_2")   bs.t0_2   = val;
        else if (n == "u0_2")   bs.u0_2   = val;
        // Auxiliary physical parameters consumed by Python-side priors.
        else if (n == "thetaS" || n == "DL" || n == "DS") continue;
        else throw std::invalid_argument("Model: unknown parameter '" + n + "'");
    }
    if (flux_treatment_ == "sample") {
        if (!fluxes_out)
            throw std::invalid_argument("flux='sample' requires sampled flux values");
        for (std::size_t k = 0; k < fluxes_out->size(); ++k) {
            if (!std::isfinite((*fluxes_out)[k].Fs)
                || !std::isfinite((*fluxes_out)[k].Fb)) {
                const std::string ds_name = flux_dataset_name(*event_, k);
                throw std::invalid_argument(
                    "flux='sample' requires parameters Fs_" + ds_name
                    + " and Fb_" + ds_name);
            }
        }
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

    // Coupled xallarap + binary source: both sources orbit the common centre of
    // mass. Source 1 uses the standard xallarap params (already in pp). Source 2
    // has the same CoM trajectory but the opposite xallarap displacement scaled
    // by 1/q_mass (q_mass = m2/m1, so the lighter source moves farther from CoM).
    const bool coupled = (opts.xallarap_param_type != LCBI_XALLARAP_NONE);
    lcbi_params p2 = *pp;
    if (coupled) {
        // All xallarap modes use xi_1/xi_2 as amplitude/position.
        // Source 2 orbits the CoM in the opposite direction, scaled by 1/q_mass.
        p2.xi_1 = -pp->xi_1 / bs.q_mass;
        p2.xi_2 = -pp->xi_2 / bs.q_mass;
        // t0/umin unchanged — source 2 is on the same CoM trajectory
    } else {
        // Independent binary source: source 2 has its own t0/u0.
        p2.t0   = bs.t0_2;
        p2.umin = bs.u0_2;
    }

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

double Model::compute_chi2(const lcbi_params& p,
                           const BinarySourceExtras& bs,
                           const std::vector<FluxSolution>* sampled_fluxes) const
{
    if (!model::from_c_params(p).is_valid())
        return std::numeric_limits<double>::infinity();

    const bool is_binary    = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial  = light_curve_->spec().terrestrial;
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

        const FluxSolution flux = sampled_fluxes
            ? sampled_fluxes->at(k)
            : fit_flux_for_dataset(ds, c, A);
        if (!std::isfinite(flux.Fs) || !std::isfinite(flux.Fb))
            return std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            const double r = f[i] - flux.Fs * A[i] - flux.Fb;
            total += w[i] * r * r;
        }
    }
    return total;
}

double Model::compute_log_likelihood(const lcbi_params& p,
                                      const BinarySourceExtras& bs,
                                      const std::vector<FluxSolution>* sampled_fluxes) const
{
    if (likelihood_mode_ == "gaussian" && flux_treatment_ != "marginalize")
        return -0.5 * compute_chi2(p, bs, sampled_fluxes);

    if (!model::from_c_params(p).is_valid())
        return -std::numeric_limits<double>::infinity();

    const bool is_binary    = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial  = light_curve_->spec().terrestrial;
    const lcbi_options& opts = light_curve_->options();
    const double nu = likelihood_nu_;
    const double norm = std::lgamma(0.5 * (nu + 1.0))
        - std::lgamma(0.5 * nu)
        - 0.5 * std::log(nu * M_PI);
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

        if (likelihood_mode_ == "gaussian" && flux_treatment_ == "marginalize") {
            const LinearFluxStats stats = linear_flux_stats_for_dataset(ds, c, A);
            const double ll = marginalized_flux_scale_log_likelihood(stats, n);
            if (!std::isfinite(ll))
                return -std::numeric_limits<double>::infinity();
            total += ll;
            continue;
        }

        const FluxSolution flux = sampled_fluxes
            ? sampled_fluxes->at(k)
            : fit_flux_for_dataset(ds, c, A);
        if (!std::isfinite(flux.Fs) || !std::isfinite(flux.Fb))
            return -std::numeric_limits<double>::infinity();

        const double* __restrict__ sig = ds.effective_flux_err().data();
        for (std::size_t i = 0; i < n; ++i) {
            const double sigma = sig[i];
            if (sigma <= 0.0) return -std::numeric_limits<double>::infinity();
            const double r = (f[i] - flux.Fs * A[i] - flux.Fb) / sigma;
            total += norm - std::log(sigma)
                - 0.5 * (nu + 1.0) * std::log1p((r * r) / nu);
        }
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

std::vector<double> Model::compute_residuals(
    const lcbi_params& p,
    const BinarySourceExtras& bs,
    const std::vector<FluxSolution>* sampled_fluxes) const
{
    if (!model::from_c_params(p).is_valid())
        return {};

    const bool is_binary   = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial = light_curve_->spec().terrestrial;
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

        const FluxSolution flux = sampled_fluxes
            ? sampled_fluxes->at(k)
            : fit_flux_for_dataset(ds, c, A);
        if (!std::isfinite(flux.Fs) || !std::isfinite(flux.Fb))
            return {};

        const double* __restrict__ sig = ds.effective_flux_err().data();
        for (std::size_t i = 0; i < n; ++i)
            out.push_back((f[i] - flux.Fs * A[i] - flux.Fb) / sig[i]);
    }
    return out;
}

std::vector<double> Model::residuals(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    BinarySourceExtras bs;
    std::vector<FluxSolution> sampled_fluxes;
    const lcbi_params p = theta_to_params(
        theta, bs, flux_treatment_ == "sample" ? &sampled_fluxes : nullptr);
    return compute_residuals(
        p, bs, flux_treatment_ == "sample" ? &sampled_fluxes : nullptr);
}

double Model::chi2(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    BinarySourceExtras bs;
    std::vector<FluxSolution> sampled_fluxes;
    const lcbi_params p = theta_to_params(
        theta, bs, flux_treatment_ == "sample" ? &sampled_fluxes : nullptr);
    return compute_chi2(
        p, bs, flux_treatment_ == "sample" ? &sampled_fluxes : nullptr);
}

double Model::log_likelihood(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    BinarySourceExtras bs;
    std::vector<FluxSolution> sampled_fluxes;
    const lcbi_params p = theta_to_params(
        theta, bs, flux_treatment_ == "sample" ? &sampled_fluxes : nullptr);
    return compute_log_likelihood(
        p, bs, flux_treatment_ == "sample" ? &sampled_fluxes : nullptr);
}

std::vector<Model::FluxSolution>
Model::fluxes(const std::vector<double>& theta) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    BinarySourceExtras bs;
    std::vector<FluxSolution> sampled_fluxes;
    const lcbi_params p = theta_to_params(
        theta, bs, flux_treatment_ == "sample" ? &sampled_fluxes : nullptr);
    if (!model::from_c_params(p).is_valid())
        return {};
    if (flux_treatment_ == "sample")
        return sampled_fluxes;
    const bool is_binary   = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial = light_curve_->spec().terrestrial;
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
        if (D <= 0.0) return {};
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
                                   std::vector<FluxSolution>& out_fluxes,
                                   std::vector<FluxConditional>* out_conditionals) const
{
    if (static_cast<int>(theta.size()) != n_params())
        throw std::invalid_argument("theta size mismatch");
    const double lp = log_prior(theta);
    if (!std::isfinite(lp)) {
        out_fluxes.clear();
        if (out_conditionals) out_conditionals->clear();
        return lp;
    }
    BinarySourceExtras bs;
    std::vector<FluxSolution> sampled_fluxes;
    const lcbi_params p = theta_to_params(
        theta, bs, flux_treatment_ == "sample" ? &sampled_fluxes : nullptr);
    if (!model::from_c_params(p).is_valid()) {
        out_fluxes.clear();
        if (out_conditionals) out_conditionals->clear();
        return -std::numeric_limits<double>::infinity();
    }
    const bool is_binary   = (light_curve_->source_kind() == lc::SourceKind::binary);
    const bool terrestrial = light_curve_->spec().terrestrial;
    const lcbi_options& opts = light_curve_->options();

    double chi2_total = 0.0;
    double log_likelihood_total = 0.0;
    const double nu = likelihood_nu_;
    const double student_t_norm = std::lgamma(0.5 * (nu + 1.0))
        - std::lgamma(0.5 * nu)
        - 0.5 * std::log(nu * M_PI);
    out_fluxes.clear();
    out_fluxes.reserve(event_->size());
    if (out_conditionals) {
        out_conditionals->clear();
        out_conditionals->reserve(event_->size());
    }
    lcbi_params p_scratch;
    for (std::size_t k = 0; k < event_->size(); ++k) {
        const auto& ds = event_->at(k);
        DatasetCache& c = cache_[k];
        const std::size_t n = ds.size();
        fill_magnification(ds, c, p, bs, opts, is_binary, terrestrial, p_scratch);

        const double* __restrict__ A = c.mag_buf.data();
        const double* __restrict__ f = ds.flux().data();
        const double* __restrict__ w = ds.weight().data();
        LinearFluxStats stats;
        const FluxSolution flux = flux_treatment_ == "sample"
            ? sampled_fluxes.at(k)
            : (stats = linear_flux_stats_for_dataset(ds, c, A)).flux;
        if (!std::isfinite(flux.Fs) || !std::isfinite(flux.Fb))
            return -std::numeric_limits<double>::infinity();
        if (likelihood_mode_ == "gaussian") {
            if (flux_treatment_ == "marginalize") {
                const double ll = marginalized_flux_scale_log_likelihood(stats, n);
                if (!std::isfinite(ll))
                    return -std::numeric_limits<double>::infinity();
                log_likelihood_total += ll;
                if (out_conditionals) {
                    const double df = static_cast<double>(n - 2);
                    const double variance = stats.chi2 / df * c.S_w / stats.det;
                    if (!std::isfinite(variance) || variance < 0.0)
                        return -std::numeric_limits<double>::infinity();
                    out_conditionals->push_back(
                        {flux.Fs, std::sqrt(variance), df});
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    const double r = f[i] - flux.Fs * A[i] - flux.Fb;
                    chi2_total += w[i] * r * r;
                }
            }
        } else {
            const double* __restrict__ sig = ds.effective_flux_err().data();
            for (std::size_t i = 0; i < n; ++i) {
                const double sigma = sig[i];
                if (sigma <= 0.0) return -std::numeric_limits<double>::infinity();
                const double r = (f[i] - flux.Fs * A[i] - flux.Fb) / sigma;
                log_likelihood_total += student_t_norm - std::log(sigma)
                    - 0.5 * (nu + 1.0) * std::log1p((r * r) / nu);
            }
        }
        out_fluxes.push_back(flux);
        if (out_conditionals && flux_treatment_ != "marginalize") {
            out_conditionals->push_back({flux.Fs, 0.0, 0.0});
        }
    }
    return lp + (
        likelihood_mode_ == "gaussian" && flux_treatment_ != "marginalize"
            ? -0.5 * chi2_total : log_likelihood_total);
}

} // namespace lcbinint::bayes

#include "light_curve.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/model/lens_model.hpp"
#include "lcbinint/model/trajectory.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>

#ifdef LCBININT_HAS_OPENMP
#include <omp.h>
#endif

namespace lcbinint::lc {
namespace {

constexpr double negative_infinity =
    -std::numeric_limits<double>::infinity();

int batch_thread_count(std::size_t rows, std::size_t columns)
{
    // The polynomial/root stack still contains shared scratch outside the
    // Skowron-Gould workspace. Independent parameter rows can therefore fail
    // numerically under OpenMP even though every row is deterministic alone.
    // Keep the native batch/GIL-free boundary but execute rows serially until
    // the complete root stack has a thread-safety audit.
    (void)rows;
    (void)columns;
    return 1;
}

struct LinearStats {
    double source = 0.0;
    double blend = 0.0;
    double chi2 = 0.0;
    double determinant = 0.0;
    double sum_weight = 0.0;
    bool valid = false;
};

LinearStats solve_flux(
    const std::vector<double>& magnification,
    const std::vector<double>& flux,
    const std::vector<double>& error)
{
    double sw = 0.0, swa = 0.0, swa2 = 0.0, swf = 0.0, swaf = 0.0;
    for (std::size_t index = 0; index < flux.size(); ++index) {
        if (!(std::isfinite(error[index]) && error[index] > 0.0)) return {};
        const double weight = 1.0 / (error[index] * error[index]);
        const double a = magnification[index];
        const double f = flux[index];
        sw += weight;
        swa += weight * a;
        swa2 += weight * a * a;
        swf += weight * f;
        swaf += weight * a * f;
    }
    const double determinant = swa2 * sw - swa * swa;
    if (!(std::isfinite(determinant) && determinant > 0.0)) return {};
    LinearStats result;
    result.source = (swaf * sw - swa * swf) / determinant;
    result.blend = (swa2 * swf - swa * swaf) / determinant;
    result.determinant = determinant;
    result.sum_weight = sw;
    result.valid = std::isfinite(result.source) && std::isfinite(result.blend);
    if (!result.valid) return result;
    for (std::size_t index = 0; index < flux.size(); ++index) {
        const double residual = flux[index]
            - result.source * magnification[index] - result.blend;
        result.chi2 += residual * residual
            / (error[index] * error[index]);
    }
    return result;
}

} // namespace

LightCurve::LightCurve(lcbi_options opts, double ld_c, double ld_d,
    std::shared_ptr<Model> model, std::shared_ptr<obs::Site> site)
    : opts_(opts), ld_c_(ld_c), ld_d_(ld_d), model_(std::move(model)),
      site_(std::move(site))
{
    if (!model_) model_ = std::make_shared<Model>();
    const bool binary_xallarap = model_->source == SourceKind::binary &&
        model_->xallarap != LCBI_XALLARAP_NONE;
    const bool velocity_xallarap = model_->xallarap == LCBI_XALLARAP_CIRCULAR_VEL ||
        model_->xallarap == LCBI_XALLARAP_KEPLER_VEL;
    if (!binary_xallarap &&
        model_->source_orbit_coordinates != SourceOrbitCoordinates::none) {
        throw std::invalid_argument(
            "LightCurve: source_orbit_coordinates requires binary source xallarap");
    }
    if (binary_xallarap && velocity_xallarap &&
        model_->source_orbit_coordinates == SourceOrbitCoordinates::none) {
        throw std::invalid_argument(
            "LightCurve: binary velocity xallarap requires source_orbit_coordinates");
    }
    if (binary_xallarap && !velocity_xallarap &&
        model_->source_orbit_coordinates != SourceOrbitCoordinates::none) {
        throw std::invalid_argument(
            "LightCurve: source_orbit_coordinates is available only with binary velocity xallarap");
    }
    // Model overrides lcbi_options for physics-mode fields.
    opts_.parallax_mode = model_->parallax ? 1 : 0;
    opts_.xallarap_param_type = model_->xallarap;
    // terrestrial is event-level and can be shared with a space curve.  It
    // requires coordinates only for a ground curve (or an unspecified site).
    if (model_->terrestrial && (!site_ ||
         (site_->kind() == obs::SiteKind::ground && !site_->has_ground_position()))) {
        throw std::invalid_argument(
            "LightCurve: terrestrial=True requires Site('ground', lat, lon)");
    }
}

lcbi_params LightCurve::apply_coords(const lcbi_params& params) const
{
    const bool needs_tref = model_->parallax
                         || (model_->orbital_motion != LCBI_ORBIT_STATIC)
                         || (model_->source == SourceKind::binary &&
                             model_->xallarap != LCBI_XALLARAP_NONE);
    if (needs_tref && !model_->t_ref.has_value())
        throw std::runtime_error(
            "LightCurve: t_ref must be set when using parallax, orbital motion, or binary-source xallarap");

    lcbi_params p = params;
    if (!model_->finite_source) p.rho = 0.0;
    if (model_->lens == LensKind::triple && p.q2 <= 0.0) {
        throw std::runtime_error(
            "LightCurve: lens='triple' requires a positive q2 parameter");
    }
    if (model_->lens == LensKind::binary && p.q2 > 0.0) {
        throw std::runtime_error(
            "LightCurve: lens='binary' cannot be used with a positive q2 parameter");
    }
    p.orbital_motion_mode = model_->orbital_motion;
    if (ld_c_ != 0.0 || ld_d_ != 0.0) {
        p.limb_darkening_c = ld_c_;
        p.limb_darkening_d = ld_d_;
    }
    if (model_->sky) {
        p.ra  = model_->sky->ra_deg();
        p.dec = model_->sky->dec_deg();
    }
    // Site is only applied when terrestrial parallax is explicitly enabled.
    if (model_->terrestrial && site_ && site_->kind() == obs::SiteKind::ground &&
        site_->has_ground_position()) {
        p.obs_lat = site_->lat_deg();
        p.obs_lon = site_->lon_deg();
    }
    if (model_->t_ref.has_value()) p.tfix = *model_->t_ref;
    return p;
}

lcbi_options LightCurve::runtime_options() const noexcept
{
    auto runtime = opts_;
    runtime.parallax_mode = model_->parallax ? 1 : 0;
    runtime.xallarap_param_type = model_->xallarap;
    return runtime;
}

std::vector<MagnificationResult> LightCurve::evaluate_routed(
    const std::vector<double>& times,
    const model::LensParameters& params,
    const model::ComputationOptions& options) const
{
    std::vector<MagnificationResult> results;
    results.reserve(times.size());
    model::LensModel lens_model(params, options, site_);
    for (double time : times) results.push_back(lens_model.magnification(time));
    return results;
}

void LightCurve::fill_routed_magnification(
    const std::vector<double>& times,
    const model::LensParameters& params,
    const model::ComputationOptions& options,
    double* output) const
{
    const auto results = evaluate_routed(times, params, options);
    for (std::size_t column = 0; column < results.size(); ++column) {
        if (results[column].status == EvaluationStatus::unsupported) {
            throw std::runtime_error("unsupported");
        }
        if (results[column].status == EvaluationStatus::numerical_error ||
            !std::isfinite(results[column].magnification)) {
            throw std::runtime_error("numerical error");
        }
        output[column] = results[column].magnification;
    }
}

std::vector<MagnificationResult> LightCurve::evaluate(
    const std::vector<double>& times,
    const lcbi_params& params) const
{
    const auto results = evaluate_diagnostic(times, params);
    for (const auto& result : results) {
        if (result.status == EvaluationStatus::unsupported) {
            throw std::runtime_error("unsupported");
        }
        if (result.status == EvaluationStatus::numerical_error ||
            !std::isfinite(result.magnification)) {
            throw std::runtime_error("numerical error");
        }
    }
    return results;
}

std::vector<MagnificationResult> LightCurve::evaluate_diagnostic(
    const std::vector<double>& times,
    const lcbi_params& params) const
{
    const lcbi_params p = apply_coords(params);
    const auto cpp_params = model::from_c_params(p);
    if (!cpp_params.is_valid()) {
        throw std::runtime_error("invalid argument");
    }
    const auto runtime_opts = runtime_options();
    return evaluate_routed(
        times, cpp_params, model::from_c_options(&runtime_opts));
}

std::vector<magnification::FiniteSourceGeometry> LightCurve::finite_source_geometry(
    const std::vector<double>& times, const lcbi_params& params) const
{
    const lcbi_params p = apply_coords(params);
    const auto cpp_params = model::from_c_params(p);
    if (!cpp_params.is_valid()) throw std::runtime_error("invalid argument");
    const auto runtime_opts = runtime_options();
    model::LensModel lens_model(
        cpp_params, model::from_c_options(&runtime_opts), site_);
    std::vector<magnification::FiniteSourceGeometry> output(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        if (!lens_model.finite_source_geometry(times[i], output[i]))
            throw std::runtime_error("finite-source geometry unavailable");
    }
    return output;
}

std::vector<double> LightCurve::magnification(
    const std::vector<double>& times,
    const lcbi_params&         params) const
{
    const auto results = evaluate(times, params);
    std::vector<double> mags;
    mags.reserve(results.size());
    for (const auto& result : results)
        mags.push_back(result.magnification);
    return mags;
}

std::vector<double> LightCurve::magnification_batch(
    const std::vector<double>& times,
    const std::vector<lcbi_params>& parameters,
    const std::vector<lcbi_params>& secondary_parameters,
    const std::vector<double>& source_ratios) const
{
    const bool binary_source = !secondary_parameters.empty();
    if (binary_source && (secondary_parameters.size() != parameters.size() ||
        source_ratios.size() != parameters.size())) {
        throw std::invalid_argument(
            "binary-source parameter and ratio rows must be aligned");
    }
    const std::size_t row_size = times.size();
    std::vector<double> output(parameters.size() * row_size);
    const auto runtime_opts = runtime_options();
    const auto cpp_options = model::from_c_options(&runtime_opts);
    std::vector<std::exception_ptr> failures(parameters.size());
    // lcbinint's root solver carries legacy shared-scratch symbols and is not
    // reentrant; rows are evaluated with a single thread until that stack has
    // a full audit (see batch_thread_count).
    const int batch_threads = batch_thread_count(parameters.size(), times.size());

    // Build one LensModel per independent parameter row, as the scalar API
    // does, but write magnifications directly into the final matrix. This
    // avoids two temporary vectors and a full row copy for every walker.
#ifdef LCBININT_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(batch_threads) if(batch_threads > 1)
#endif
    for (std::ptrdiff_t row_index = 0;
         row_index < static_cast<std::ptrdiff_t>(parameters.size()); ++row_index) {
        const std::size_t row = static_cast<std::size_t>(row_index);
        try {
            const lcbi_params params = apply_coords(parameters[row]);
            const auto cpp_params = model::from_c_params(params);
            if (!cpp_params.is_valid()) {
                throw std::runtime_error("invalid argument");
            }
            const std::size_t offset = row * row_size;
            fill_routed_magnification(
                times, cpp_params, cpp_options, output.data() + offset);
            if (binary_source) {
                const lcbi_params secondary =
                    apply_coords(secondary_parameters[row]);
                const auto secondary_cpp = model::from_c_params(secondary);
                if (!secondary_cpp.is_valid()) {
                    throw std::runtime_error("invalid argument");
                }
                std::vector<double> secondary_magnification(row_size);
                fill_routed_magnification(
                    times, secondary_cpp, cpp_options,
                    secondary_magnification.data());
                const double ratio = source_ratios[row];
                const double denominator = 1.0 + ratio;
                if (!(std::isfinite(ratio) && denominator != 0.0)) {
                    throw std::invalid_argument("invalid binary-source flux ratio");
                }
                for (std::size_t column = 0; column < row_size; ++column) {
                    output[offset + column] =
                        (output[offset + column]
                         + ratio * secondary_magnification[column])
                        / denominator;
                }
            }
        } catch (...) {
            failures[row] = std::current_exception();
        }
    }
    for (const auto& failure : failures) {
        if (failure) std::rethrow_exception(failure);
    }
    return output;
}

LikelihoodBatchResult LightCurve::light_curve_log_likelihood_batch(
    const std::vector<double>& times,
    const std::vector<double>& flux,
    const std::vector<double>& error,
    const std::vector<lcbi_params>& parameters,
    LikelihoodDistribution distribution,
    FluxMode flux_mode,
    double nu,
    const std::vector<double>& sampled_source,
    const std::vector<double>& sampled_blend,
    const std::vector<lcbi_params>& secondary_parameters,
    const std::vector<double>& source_ratios) const
{
    const bool binary_source = !secondary_parameters.empty();
    if (times.empty() || parameters.empty() || flux.size() != times.size() ||
        error.size() != times.size()) {
        throw std::invalid_argument(
            "times, flux, error, and parameter rows must be non-empty and aligned");
    }
    for (std::size_t index = 0; index < times.size(); ++index) {
        if (!std::isfinite(flux[index]) ||
            !(std::isfinite(error[index]) && error[index] > 0.0)) {
            throw std::invalid_argument(
                "flux must be finite and error must be finite and positive");
        }
    }
    if (distribution == LikelihoodDistribution::student_t &&
        (!(std::isfinite(nu)) || nu <= 0.0)) {
        throw std::invalid_argument("Student-t nu must be finite and positive");
    }
    if (distribution == LikelihoodDistribution::student_t &&
        flux_mode == FluxMode::marginalize) {
        throw std::invalid_argument(
            "Student-t flux marginalization is unsupported");
    }
    if (flux_mode == FluxMode::sample &&
        (sampled_source.size() != parameters.size() ||
         sampled_blend.size() != parameters.size())) {
        throw std::invalid_argument(
            "sampled flux arrays must match the parameter row count");
    }
    if (binary_source && (secondary_parameters.size() != parameters.size() ||
        source_ratios.size() != parameters.size())) {
        throw std::invalid_argument(
            "binary-source parameter and ratio rows must be aligned");
    }

    LikelihoodBatchResult output;
    const std::size_t rows = parameters.size();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    output.log_likelihood.assign(rows, negative_infinity);
    output.source_flux.assign(rows, nan);
    output.blend_flux.assign(rows, nan);
    output.conditional_scale.assign(rows, nan);
    output.conditional_df.assign(rows, nan);
    const double student_norm =
        distribution == LikelihoodDistribution::student_t
        ? std::lgamma(0.5 * (nu + 1.0)) - std::lgamma(0.5 * nu)
            - 0.5 * std::log(nu * std::acos(-1.0))
        : 0.0;
    const auto runtime_opts = runtime_options();
    const auto cpp_options = model::from_c_options(&runtime_opts);
    std::vector<std::exception_ptr> failures(rows);
    const int batch_threads = batch_thread_count(rows, times.size());

#ifdef LCBININT_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(batch_threads) if(batch_threads > 1)
#endif
    for (std::ptrdiff_t row_index = 0;
         row_index < static_cast<std::ptrdiff_t>(rows); ++row_index) {
        const std::size_t row = static_cast<std::size_t>(row_index);
        try {
            std::vector<double> magnification(times.size());
            const lcbi_params params = apply_coords(parameters[row]);
            const auto cpp_params = model::from_c_params(params);
            if (!cpp_params.is_valid()) {
                throw std::runtime_error("invalid argument");
            }
            fill_routed_magnification(
                times, cpp_params, cpp_options, magnification.data());
            if (binary_source) {
                const lcbi_params secondary =
                    apply_coords(secondary_parameters[row]);
                const auto secondary_cpp = model::from_c_params(secondary);
                if (!secondary_cpp.is_valid()) {
                    throw std::runtime_error("invalid argument");
                }
                std::vector<double> secondary_magnification(times.size());
                fill_routed_magnification(
                    times, secondary_cpp, cpp_options,
                    secondary_magnification.data());
                const double ratio = source_ratios[row];
                const double denominator = 1.0 + ratio;
                if (!(std::isfinite(ratio) && denominator != 0.0)) {
                    throw std::invalid_argument("invalid binary-source flux ratio");
                }
                for (std::size_t column = 0; column < times.size(); ++column) {
                    magnification[column] =
                        (magnification[column]
                         + ratio * secondary_magnification[column])
                        / denominator;
                }
            }
            const LinearStats stats = flux_mode == FluxMode::sample
                ? LinearStats{} : solve_flux(magnification, flux, error);
            double source = stats.source;
            double blend = stats.blend;
            if (flux_mode == FluxMode::sample) {
                source = sampled_source[row];
                blend = sampled_blend[row];
                if (!(std::isfinite(source) && std::isfinite(blend))) continue;
            } else if (!stats.valid) {
                continue;
            }
            output.source_flux[row] = source;
            output.blend_flux[row] = blend;

            if (distribution == LikelihoodDistribution::gaussian &&
                flux_mode == FluxMode::marginalize) {
                const double dof = static_cast<double>(times.size()) - 2.0;
                if (!(dof > 0.0 && stats.chi2 > 0.0)) continue;
                output.log_likelihood[row] =
                    -0.5 * dof * std::log(stats.chi2)
                    - 0.5 * std::log(stats.determinant);
                const double variance =
                    stats.chi2 / dof * stats.sum_weight / stats.determinant;
                if (variance > 0.0 && std::isfinite(variance)) {
                    output.conditional_scale[row] = std::sqrt(variance);
                    output.conditional_df[row] = dof;
                }
                continue;
            }

            double total = 0.0;
            for (std::size_t column = 0; column < times.size(); ++column) {
                const double residual = (flux[column]
                    - source * magnification[column] - blend) / error[column];
                if (distribution == LikelihoodDistribution::gaussian) {
                    total -= 0.5 * residual * residual;
                } else {
                    total += student_norm - std::log(error[column])
                        - 0.5 * (nu + 1.0)
                            * std::log1p(residual * residual / nu);
                }
            }
            output.log_likelihood[row] = total;
        } catch (...) {
            failures[row] = std::current_exception();
        }
    }
    for (const auto& failure : failures) {
        if (failure) std::rethrow_exception(failure);
    }
    return output;
}

std::vector<double> LightCurve::magnification_binary(
    const std::vector<double>& times,
    const lcbi_params&         params1,
    double                     flux_ratio,
    const lcbi_params&         params2) const
{
    const auto r1 = evaluate(times, params1);
    const auto r2 = evaluate(times, params2);

    const int n = static_cast<int>(times.size());
    std::vector<double> mags(n);
    const double denom = 1.0 + flux_ratio;
    for (int i = 0; i < n; ++i)
        mags[i] = (r1[i].magnification + flux_ratio * r2[i].magnification) / denom;
    return mags;
}

} // namespace lcbinint::lc

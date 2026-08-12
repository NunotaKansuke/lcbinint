#pragma once
#include "model.hpp"
#include "lcbinint/lcbinint.h"
#include "lcbinint/model/lens_model.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lcbinint::lc {

enum class LikelihoodDistribution { gaussian, student_t };
enum class FluxMode { fit, sample, marginalize };

struct LikelihoodBatchResult {
    std::vector<double> log_likelihood;
    std::vector<double> source_flux;
    std::vector<double> blend_flux;
    std::vector<double> conditional_scale;
    std::vector<double> conditional_df;
};

// Standalone Python-facing evaluator.
// Holds lcbi_options (numerics) + Model (physics) + limb-darkening coefficients.
// On construction, Model fields override the relevant lcbi_options flags.
// apply_coords() bakes Model into a params copy for every magnification call.
class LightCurve {
public:
    explicit LightCurve(
        lcbi_options opts    = lcbi_default_options(),
        double       ld_c   = 0.0,
        double       ld_d   = 0.0,
        std::shared_ptr<Model> model = std::make_shared<Model>(),
        std::shared_ptr<obs::Site> site = nullptr
    );

    // Apply stored Model + ld settings to a params copy.
    // Throws if parallax or orbital motion is active but t_ref is not set.
    lcbi_params apply_coords(const lcbi_params& params) const;

    std::vector<MagnificationResult> evaluate(
        const std::vector<double>& times,
        const lcbi_params& params
    ) const;

    // Return raw numerical diagnostics, including a finite last iterate when
    // the requested finite-source tolerance was not met.  Production
    // magnification paths use evaluate() and fail closed on that state.
    std::vector<MagnificationResult> evaluate_diagnostic(
        const std::vector<double>& times,
        const lcbi_params& params
    ) const;

    std::vector<MagnificationResult> evaluate_preplanned_diagnostic(
        const std::vector<double>& times,
        const lcbi_params& params,
        const std::vector<model::MagnificationExecutionPlan>& plan,
        std::vector<double>* epoch_seconds = nullptr
    ) const;

    // Diagnostic benchmark path for source coordinates supplied directly in
    // the internal lens frame.  It shares the LensModel construction and
    // preplanned execution machinery with evaluate_preplanned_diagnostic, but
    // does not reconstruct (x, y) from a time-domain trajectory per epoch.
    std::vector<MagnificationResult> evaluate_preplanned_xy_diagnostic(
        const std::vector<double>& source_x,
        const std::vector<double>& source_y,
        const lcbi_params& params,
        const std::vector<model::MagnificationExecutionPlan>& plan,
        std::vector<double>* epoch_seconds = nullptr
    ) const;

    // Root-solve-free, trajectory-resolved geometry for external finite-source
    // engines. Unlike the C API this preserves this LightCurve's sky/site and
    // physical model configuration.
    std::vector<magnification::FiniteSourceGeometry> finite_source_geometry(
        const std::vector<double>& times,
        const lcbi_params& params
    ) const;

    // Single-source magnification.
    std::vector<double> magnification(
        const std::vector<double>& times,
        const lcbi_params&         params
    ) const;

    std::vector<double> magnification_preplanned(
        const std::vector<double>& times,
        const lcbi_params& params,
        const std::vector<model::MagnificationExecutionPlan>& plan
    ) const;

    // Evaluate independent parameter rows into row-major [parameter, time]
    // storage, avoiding per-row Python call overhead for large batches. The
    // public scalar API (magnification()) remains unchanged.
    std::vector<double> magnification_batch(
        const std::vector<double>& times,
        const std::vector<lcbi_params>& parameters,
        const std::vector<lcbi_params>& secondary_parameters = {},
        const std::vector<double>& source_ratios = {}
    ) const;

    // Specialized inference path: stream one magnification row at a time into
    // flux solving and the likelihood, without materializing [parameter,time].
    LikelihoodBatchResult light_curve_log_likelihood_batch(
        const std::vector<double>& times,
        const std::vector<double>& flux,
        const std::vector<double>& error,
        const std::vector<lcbi_params>& parameters,
        LikelihoodDistribution distribution,
        FluxMode flux_mode,
        double nu,
        const std::vector<double>& sampled_source = {},
        const std::vector<double>& sampled_blend = {},
        const std::vector<lcbi_params>& secondary_parameters = {},
        const std::vector<double>& source_ratios = {}
    ) const;

    // Binary-source magnification. The caller supplies the independent
    // rectilinear parameters of both sources.
    std::vector<double> magnification_binary(
        const std::vector<double>& times,
        const lcbi_params&         params1,
        double                     flux_ratio,
        const lcbi_params&         params2
    ) const;

    const lcbi_options& options()  const noexcept { return opts_; }
    // Numerical options augmented with the currently shared Model state.
    // This is intentionally a value: a Model may be shared by several curves
    // and edited after those curves have been constructed.
    lcbi_options runtime_options() const noexcept;
    double              ld_c()     const noexcept { return ld_c_; }
    double              ld_d()     const noexcept { return ld_d_; }
    const Model&        model()     const noexcept { return *model_; }
    const std::shared_ptr<Model>& model_ptr() const noexcept { return model_; }

    // Convenience accessors (delegate to Model).
    LensKind                              lens_kind()      const noexcept { return model_->lens; }
    SourceKind                            source_kind()    const noexcept { return model_->source; }
    lcbi_orbital_motion_mode              orbital_motion() const noexcept { return model_->orbital_motion; }
    const std::shared_ptr<obs::SkyCoord>& sky_coord()      const noexcept { return model_->sky; }
    const std::shared_ptr<obs::Site>&     site()           const noexcept { return site_; }
    std::optional<double>                 t_ref()          const noexcept { return model_->t_ref; }

private:
    std::vector<MagnificationResult> evaluate_routed(
        const std::vector<double>& times,
        const model::LensParameters& params,
        const model::ComputationOptions& options) const;
    void fill_routed_magnification(
        const std::vector<double>& times,
        const model::LensParameters& params,
        const model::ComputationOptions& options,
        double* output) const;

    lcbi_options opts_;
    double       ld_c_;
    double       ld_d_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<obs::Site> site_;
};

} // namespace lcbinint::lc

#pragma once
#include "effects.hpp"
#include "lcbinint/lcbinint.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lcbinint::lc {

// Standalone Python-facing evaluator.
// Holds lcbi_options (numerics) + Effects (physics) + limb-darkening coefficients.
// On construction, Effects fields override the relevant lcbi_options flags.
// apply_coords() bakes Effects into a params copy for every magnification call.
class LightCurve {
public:
    explicit LightCurve(
        lcbi_options opts    = lcbi_default_options(),
        double       ld_c   = 0.0,
        double       ld_d   = 0.0,
        Effects      effects = {}
    );

    // Apply stored Effects + ld settings to a params copy.
    // Throws if parallax or orbital motion is active but t_ref is not set.
    lcbi_params apply_coords(const lcbi_params& params) const;

    // Single-source magnification.
    std::vector<double> magnification(
        const std::vector<double>& times,
        const lcbi_params&         params
    ) const;

    // Binary-source magnification: source 2 has different t0/u0, same tE/alpha.
    // A_eff = (A1 + q_source * A2) / (1 + q_source)
    std::vector<double> magnification_binary(
        const std::vector<double>& times,
        const lcbi_params&         params,
        double                     q_source,
        double                     t0_2,
        double                     u0_2
    ) const;

    // Binary-source magnification: caller supplies full params for source 2.
    // Used for coupled xallarap where source 2's xi_1/xi_2 differ from source 1's.
    std::vector<double> magnification_binary(
        const std::vector<double>& times,
        const lcbi_params&         params1,
        double                     q_source,
        const lcbi_params&         params2
    ) const;

    const lcbi_options& options()  const noexcept { return opts_; }
    double              ld_c()     const noexcept { return ld_c_; }
    double              ld_d()     const noexcept { return ld_d_; }
    const Effects&      effects()  const noexcept { return effects_; }

    // Convenience accessors (delegate to Effects).
    SourceKind                            source_kind()    const noexcept { return effects_.source; }
    lcbi_orbital_motion_mode              orbital_motion() const noexcept { return effects_.orbital_motion; }
    const std::shared_ptr<obs::SkyCoord>& sky_coord()      const noexcept { return effects_.sky; }
    const std::shared_ptr<obs::Site>&     site()           const noexcept { return effects_.site; }
    std::optional<double>                 t_ref()          const noexcept { return effects_.t_ref; }

private:
    lcbi_options opts_;
    double       ld_c_;
    double       ld_d_;
    Effects      effects_;
};

} // namespace lcbinint::lc

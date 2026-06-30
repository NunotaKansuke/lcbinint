#pragma once
#include "lcbinint/lcbinint.h"
#include "lcbinint/obs/coordinates.hpp"
#include <memory>
#include <optional>

namespace lcbinint::lc {

enum class SourceKind { single, binary };

// All physical higher-order effect settings, separate from numerical Options (lcbi_options).
// Rules:
//  - terrestrial must be explicitly true for site coords to be applied (even if site is set).
//  - parallax / xallarap / orbital_motion override the corresponding lcbi_options fields.
//  - sky/site at LightCurve level are defaults; per-dataset LightCurveData::site overrides
//    obs_lat/obs_lon when terrestrial=true.
struct Effects {
    SourceKind               source         = SourceKind::single;
    lcbi_orbital_motion_mode orbital_motion = LCBI_ORBIT_STATIC;
    lcbi_xallarap_param_type xallarap       = LCBI_XALLARAP_NONE;
    bool                     parallax       = false;
    bool                     terrestrial    = false;
    std::shared_ptr<obs::SkyCoord> sky      = nullptr;
    std::shared_ptr<obs::Site>     site     = nullptr;
    std::optional<double>          t_ref    = std::nullopt;
};

} // namespace lcbinint::lc

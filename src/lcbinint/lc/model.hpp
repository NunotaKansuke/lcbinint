#pragma once
#include "lcbinint/lcbinint.h"
#include "lcbinint/obs/coordinates.hpp"
#include <memory>
#include <optional>

namespace lcbinint::lc {

enum class LensKind { binary, triple };
enum class SourceKind { single, binary };

// Physical light-curve model, separate from numerical Options (lcbi_options).
// Rules:
//  - parallax / xallarap / orbital_motion override the corresponding lcbi_options fields.
//  - sky is event-level; each LightCurve carries its own observing site.
struct Model {
    LensKind                 lens           = LensKind::binary;
    SourceKind               source         = SourceKind::single;
    lcbi_orbital_motion_mode orbital_motion = LCBI_ORBIT_STATIC;
    lcbi_xallarap_param_type xallarap       = LCBI_XALLARAP_NONE;
    bool                     parallax       = false;
    bool                     terrestrial    = false;
    std::shared_ptr<obs::SkyCoord> sky      = nullptr;
    std::optional<double>          t_ref    = std::nullopt;
};

} // namespace lcbinint::lc

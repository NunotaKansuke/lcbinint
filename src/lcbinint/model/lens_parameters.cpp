#include "lcbinint/model/lens_parameters.hpp"

#include <cmath>

namespace lcbinint::model {

bool LensParameters::is_valid() const
{
    return std::isfinite(tE) && tE != 0.0 && std::isfinite(q) && std::isfinite(sep) &&
           orbital_motion_mode >= LCBI_ORBIT_STATIC && orbital_motion_mode <= LCBI_ORBIT_KEPLER;
}

LensParameters from_c_params(const lcbi_params &params)
{
    LensParameters out;
    out.t0 = params.t0;
    out.tE = params.tE;
    out.umin = params.umin;
    out.q = params.q;
    out.sep = params.sep;
    out.theta = params.theta;
    out.rho = params.rho;
    out.omega = params.omega;
    out.piEN = params.piEN;
    out.piEE = params.piEE;
    out.piEN_xa = params.piEN_xa;
    out.piEE_xa = params.piEE_xa;
    out.ra_xa = params.ra_xa;
    out.dec_xa = params.dec_xa;
    out.period_xa = params.period_xa;
    out.ecc_xa = params.ecc_xa;
    out.peri_xa = params.peri_xa;
    out.v_sep = params.v_sep;
    out.q2 = params.q2;
    out.sep2 = params.sep2;
    out.ang = params.ang;
    out.ra = params.ra;
    out.dec = params.dec;
    out.earth_axis = params.earth_axis;
    out.tfix = params.tfix;
    out.limb_darkening_c = params.limb_darkening_c;
    out.limb_darkening_d = params.limb_darkening_d;
    out.orbital_motion_mode = params.orbital_motion_mode;
    out.g1 = params.g1;
    out.g2 = params.g2;
    out.g3 = params.g3;
    out.lom_szs = params.lom_szs;
    out.lom_ar = params.lom_ar;
    return out;
}

ComputationOptions from_c_options(const lcbi_options *options)
{
    ComputationOptions out;
    if (options == nullptr) {
        return out;
    }
    out.parallax_mode = options->parallax_mode;
    out.orbit_pair = options->orbit_pair;
    out.center_of_mass = options->center_of_mass;
    out.caustic_bins = options->caustic_bins;
    out.source_bins = options->source_bins;
    out.mode = options->mode;
    out.vbbl_compatible = options->vbbl_compatible;
    out.grid_ratio = options->grid_ratio;
    out.point_source_threshold = options->point_source_threshold;
    out.hexadecapole_threshold = options->hexadecapole_threshold;
    return out;
}

} // namespace lcbinint::model

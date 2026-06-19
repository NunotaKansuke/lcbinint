#pragma once

#include "lcbinint/lcbinint.h"

namespace lcbinint::model {

struct LensParameters {
    double t0 = 0.0;
    double tE = 1.0;
    double umin = 0.0;
    double q = 1.0;
    double sep = 1.0;
    double theta = 0.0;
    double rho = 0.0;
    double omega = 0.0;
    double piEN = 0.0;
    double piEE = 0.0;
    double piEN_xa = 0.0;
    double piEE_xa = 0.0;
    double ra_xa = 0.0;
    double dec_xa = 0.0;
    double period_xa = 0.0;
    double ecc_xa = 0.0;
    double peri_xa = 0.0;
    double v_sep = 0.0;
    double q2 = 0.0;
    double sep2 = 0.0;
    double ang = 0.0;
    double ra = 0.0;
    double dec = 0.0;
    double earth_axis = 0.0;
    double tfix = 0.0;
    double limb_darkening_c = 0.0;
    double limb_darkening_d = 0.0;

    bool is_triple() const { return q2 > 0.0; }
    bool is_valid() const;
};

struct ComputationOptions {
    lcbi_finite_source_mode finite_source_mode = LCBI_POINT_SOURCE;
    int parallax_mode = 0;
    int orbit_pair = 23;
    int center_of_mass = 0;
    int caustic_bins = 1400;
    int source_bins = 20;
    double grid_ratio = 4.0;
    double finite_source_threshold = 9.0;
    double hexadecapole_threshold = 2.0;

    bool is_point_source() const { return finite_source_mode == LCBI_POINT_SOURCE; }
};

LensParameters from_c_params(const lcbi_params &params);
ComputationOptions from_c_options(const lcbi_options *options);

} // namespace lcbinint::model

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
    lcbi_orbital_motion_mode orbital_motion_mode = LCBI_ORBIT_STATIC;
    double g1 = 0.0;
    double g2 = 0.0;
    double g3 = 0.0;
    double lom_szs = 0.0;
    double lom_ar = 1.0;

    bool is_triple() const { return q2 > 0.0; }
    bool has_orbital_motion() const { return orbital_motion_mode != LCBI_ORBIT_STATIC; }
    bool is_valid() const;
};

struct ComputationOptions {
    int parallax_mode = 0;
    int orbit_pair = 23;
    int center_of_mass = 0;
    int caustic_bins = 1400;
    int source_bins = 50;
    int mode = 4;                        // internal: 1 = cartesian, 2 = polar, 4 = auto
    int vbm_compatible = 0;             // 0 = original lcbinint convention, 1 = VBM convention
    double grid_ratio = 4.0;
    double point_source_threshold = 20.0;
    double hexadecapole_threshold = 3.0;
    double adaptive_hex_threshold = 0.001;
    int adaptive_source_bins = 0;
    int max_source_bins = 400;
    double finite_source_tol = 0.0;
    double finite_source_reltol = 0.0;
};

LensParameters from_c_params(const lcbi_params &params);
ComputationOptions from_c_options(const lcbi_options *options);

} // namespace lcbinint::model

#ifndef LCBININT_H
#define LCBININT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum lcbi_status {
    LCBI_OK = 0,
    LCBI_INVALID_ARGUMENT = 1,
    LCBI_NUMERICAL_ERROR = 2,
    LCBI_UNSUPPORTED = 3
} lcbi_status;

typedef enum lcbi_orbital_motion_mode {
    LCBI_ORBIT_STATIC = 0,
    LCBI_ORBIT_CIRCULAR = 1,
    LCBI_ORBIT_KEPLER = 2
} lcbi_orbital_motion_mode;

typedef struct lcbi_params {
    double t0;
    double tE;
    double umin;
    double q;
    double sep;
    double theta;
    double rho;
    double omega;
    double piEN;
    double piEE;
    double piEN_xa;
    double piEE_xa;
    double ra_xa;
    double dec_xa;
    double period_xa;
    double ecc_xa;
    double peri_xa;
    double v_sep;

    /* Triple-lens parameters. Triple mode is enabled when q2 > 0. */
    double q2;
    double sep2;
    double ang;

    double ra;
    double dec;
    double earth_axis;
    double tfix;
    double obs_lat;  /* observatory geodetic latitude in degrees (enables terrestrial parallax) */
    double obs_lon;  /* observatory East longitude in degrees */
    double limb_darkening_c;
    double limb_darkening_d;

    lcbi_orbital_motion_mode orbital_motion_mode;
    double g1;
    double g2;
    double g3;
    double lom_szs;
    double lom_ar;

    /* Xallarap (source orbital motion) parameters.
       For xallarap_param_type == LCBI_XALLARAP_ANGULAR_VELOCITY:
         xi_1, xi_2: xallarap amplitude components
         omega_xa:   source orbital angular frequency (rad / time unit)
         inc_xa:     orbital inclination
         phi_xa:     orbital phase at t0
       For xallarap_param_type == LCBI_XALLARAP_ORBITAL_ELEMENTS:
         piEN_xa, piEE_xa: xallarap amplitude components
         period_xa, ecc_xa, peri_xa: period, eccentricity, periapsis angle at t0
         inc_xa: orbital inclination (shared with angular_velocity mode) */
    double xi_1;
    double xi_2;
    double omega_xa;
    double inc_xa;
    double phi_xa;
} lcbi_params;

typedef enum lcbi_xallarap_param_type {
    LCBI_XALLARAP_NONE              = 0,
    LCBI_XALLARAP_ANGULAR_VELOCITY  = 1,
    LCBI_XALLARAP_ORBITAL_ELEMENTS  = 2
} lcbi_xallarap_param_type;

typedef struct lcbi_options {
    int parallax_mode;
    int orbit_pair;
    int center_of_mass;
    int caustic_bins;
    int source_bins;
    int mode;                        /* internal finite-source method: 1 = cartesian, 2 = polar, 4 = auto (default) */
    int vbm_compatible;             /* 0 = original lcbinint convention, 1 = VBM convention */
    lcbi_xallarap_param_type xallarap_param_type;
    double grid_ratio;
    int polar_source_bins;          /* 0 = use source_bins for polar inverse-ray */
    double polar_grid_ratio;        /* <=0 = use grid_ratio for polar inverse-ray */
    double point_source_threshold;   /* bbox margin for fast PS exit (in units of rho) */
    double hexadecapole_threshold;   /* caustic-distance threshold; unused when adaptive_hex_threshold > 0 */
    double adaptive_hex_threshold;   /* VBM-style hex tolerance: |a4 correction|/mag > this => IR mode */
    int adaptive_source_bins;        /* 0 = fixed source_bins (default), 1 = refine source_bins from grid diagnostics */
    int max_source_bins;             /* maximum source_bins used when adaptive_source_bins is enabled */
    double finite_source_tol;        /* absolute adaptive IR tolerance target; 0 disables */
    double finite_source_reltol;     /* relative adaptive IR tolerance target; 0 disables */
} lcbi_options;

typedef struct lcbi_result {
    double magnification;
    double point_source_magnification;
    double finite_source_magnification;
    double source_x;
    double source_y;
    int image_count;
    double finite_source_error_estimate;
    int finite_source_refinement_level;
    int finite_source_converged;
} lcbi_result;

lcbi_params lcbi_default_params(void);
lcbi_options lcbi_default_options(void);

lcbi_status lcbi_magnification(
    double time,
    const lcbi_params *params,
    const lcbi_options *options,
    lcbi_result *result
);

lcbi_status lcbi_magnification_array(
    const double *times,
    int count,
    const lcbi_params *params,
    const lcbi_options *options,
    lcbi_result *results
);

const char *lcbi_status_string(lcbi_status status);

#ifdef __cplusplus
}
#endif

#endif

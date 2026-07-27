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
    double obs_lat;  /* observatory geodetic latitude in degrees; NaN means no site */
    double obs_lon;  /* observatory East longitude in degrees; NaN means no site */
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
    LCBI_XALLARAP_ANGULAR_VELOCITY  = 1,  /* deprecated — use circular_elements instead */
    LCBI_XALLARAP_ORBITAL_ELEMENTS  = 2,  /* Kepler:   xi_1/xi_2 amplitude, period/ecc/peri/inc_xa */
    LCBI_XALLARAP_CIRCULAR_ELEMENTS = 3,  /* circular: xi_1/xi_2 amplitude, period_xa, inc_xa (ecc=0) */
    LCBI_XALLARAP_CIRCULAR_VEL      = 4,  /* circular: xi_1/xi_2 pos at tref; omega_xa=w1, inc_xa=w2, phi_xa=w3 */
    LCBI_XALLARAP_KEPLER_VEL        = 5   /* Kepler:   xi_1/xi_2 pos at tref; w1/2/3, piEN_xa=szs, piEE_xa=ar */
} lcbi_xallarap_param_type;

typedef struct lcbi_options {
    int jax;                       /* Python execution backend: 0 = native, 1 = JAX */
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
    int automatic_source_bins;       /* 0 = fixed source_bins, 1 = calibrated one-shot auto nbin (default) */
    int max_source_bins;             /* maximum nbin selected by automatic mode */
    double finite_source_tol;        /* absolute term in tol + reltol*max(|A|,1); 0 means no abs term */
    double finite_source_reltol;     /* relative term; both terms 0 selects the calibrated default */
} lcbi_options;

typedef struct lcbi_result {
    double magnification;
    double point_source_magnification;
    double finite_source_magnification;
    double source_x;
    double source_y;
    int image_count;
    int finite_source_method;
    double finite_source_error_estimate;
    int finite_source_refinement_level;
    int finite_source_converged;
    int root_candidate_count;
    int root_duplicate_count;
    int root_polish_failure_count;
    int root_used_warm_start;
    int root_used_cold_retry;
    int root_used_high_precision;
    int root_needs_high_precision;
    double root_max_residual;
    double point_source_quadrupole_indicator;
    double point_source_cusp_indicator;
    double point_source_ghost_indicator;
    double point_source_planetary_distance2;
    double point_source_safety_tolerance;
    int point_source_ghost_count;
    int point_source_safety_flags; /* bit 0: quadrupole+cusp, bit 1: ghost, bit 2: planetary */
    double separation;       /* orbital-motion-corrected projected separation at this epoch */
    double mass_ratio;       /* effective mass ratio at this epoch */
    double caustic_distance; /* source-to-caustic distance; +inf outside the finite-source path */
} lcbi_result;

/* Engine-neutral, trajectory-resolved finite-source geometry for one epoch.
   Root-solve-free: cheap enough to call once per epoch on a hot path, unlike
   lcbi_magnification[_array]. Intended for an external caller that implements
   its own finite-source engine (e.g. contour integration) and needs the same
   trajectory/orbital-motion transformations lcbinint itself uses. */
typedef struct lcbi_geometry {
    double separation;
    double mass_ratio;
    double source_x;
    double source_y;
    double source_radius;
    double limb_darkening_c;
    double limb_darkening_d;
    double absolute_tolerance;
    double relative_tolerance;
    int valid;
} lcbi_geometry;

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

lcbi_status lcbi_finite_source_geometry(
    double time,
    const lcbi_params *params,
    const lcbi_options *options,
    lcbi_geometry *geometry
);

lcbi_status lcbi_finite_source_geometry_array(
    const double *times,
    int count,
    const lcbi_params *params,
    const lcbi_options *options,
    lcbi_geometry *geometries
);

const char *lcbi_status_string(lcbi_status status);

#ifdef __cplusplus
}
#endif

#endif

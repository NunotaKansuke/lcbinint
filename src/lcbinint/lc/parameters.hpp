#pragma once
#include "lcbinint/lcbinint.h"

namespace lcbinint::lc {

// Named wrapper around lcbi_params.
// Python-friendly aliases: u0=umin, alpha=theta, s=sep
class Parameters {
public:
    Parameters();
    explicit Parameters(const lcbi_params& raw);

    const lcbi_params& raw() const noexcept { return p_; }
    lcbi_params&       raw()       noexcept { return p_; }

    // Core
    double t0()    const { return p_.t0; }     void set_t0(double v)    { p_.t0 = v; }
    double tE()    const { return p_.tE; }     void set_tE(double v)    { p_.tE = v; }
    double u0()    const { return p_.umin; }   void set_u0(double v)    { p_.umin = v; }
    double alpha() const { return p_.theta; }  void set_alpha(double v) { p_.theta = v; }
    double s()     const { return p_.sep; }    void set_s(double v)     { p_.sep = v; }
    double q()     const { return p_.q; }      void set_q(double v)     { p_.q = v; }
    double rho()   const { return p_.rho; }    void set_rho(double v)   { p_.rho = v; }

    // Triple-lens (enabled when q2 > 0)
    double q2()   const { return p_.q2; }   void set_q2(double v)   { p_.q2 = v; }
    double sep2() const { return p_.sep2; } void set_sep2(double v) { p_.sep2 = v; }
    double ang()  const { return p_.ang; }  void set_ang(double v)  { p_.ang = v; }

    // Parallax
    double piEN() const { return p_.piEN; }   void set_piEN(double v) { p_.piEN = v; }
    double piEE() const { return p_.piEE; }   void set_piEE(double v) { p_.piEE = v; }
    double ra()   const { return p_.ra; }     void set_ra(double v)   { p_.ra = v; }
    double dec()  const { return p_.dec; }    void set_dec(double v)  { p_.dec = v; }
    double tfix() const { return p_.tfix; }   void set_tfix(double v) { p_.tfix = v; }

    // Terrestrial parallax
    double obs_lat() const { return p_.obs_lat; } void set_obs_lat(double v) { p_.obs_lat = v; }
    double obs_lon() const { return p_.obs_lon; } void set_obs_lon(double v) { p_.obs_lon = v; }

    // Orbital motion
    lcbi_orbital_motion_mode orbital_motion_mode() const { return p_.orbital_motion_mode; }
    void set_orbital_motion_mode(lcbi_orbital_motion_mode v) { p_.orbital_motion_mode = v; }
    double g1()      const { return p_.g1; }      void set_g1(double v)      { p_.g1 = v; }
    double g2()      const { return p_.g2; }      void set_g2(double v)      { p_.g2 = v; }
    double g3()      const { return p_.g3; }      void set_g3(double v)      { p_.g3 = v; }
    double lom_szs() const { return p_.lom_szs; } void set_lom_szs(double v) { p_.lom_szs = v; }
    double lom_ar()  const { return p_.lom_ar; }  void set_lom_ar(double v)  { p_.lom_ar = v; }
    double v_sep()   const { return p_.v_sep; }   void set_v_sep(double v)   { p_.v_sep = v; }

    // Xallarap (angular velocity mode)
    double xi_1()    const { return p_.xi_1; }    void set_xi_1(double v)    { p_.xi_1 = v; }
    double xi_2()    const { return p_.xi_2; }    void set_xi_2(double v)    { p_.xi_2 = v; }
    double omega_xa() const { return p_.omega_xa; } void set_omega_xa(double v) { p_.omega_xa = v; }
    double inc_xa()   const { return p_.inc_xa; }   void set_inc_xa(double v)   { p_.inc_xa = v; }
    double phi_xa()   const { return p_.phi_xa; }   void set_phi_xa(double v)   { p_.phi_xa = v; }

    // Xallarap (orbital elements mode)
    double piEN_xa()  const { return p_.piEN_xa; }  void set_piEN_xa(double v)  { p_.piEN_xa = v; }
    double piEE_xa()  const { return p_.piEE_xa; }  void set_piEE_xa(double v)  { p_.piEE_xa = v; }
    double period_xa() const { return p_.period_xa; } void set_period_xa(double v) { p_.period_xa = v; }
    double ecc_xa()    const { return p_.ecc_xa; }    void set_ecc_xa(double v)    { p_.ecc_xa = v; }
    double peri_xa()   const { return p_.peri_xa; }   void set_peri_xa(double v)   { p_.peri_xa = v; }

    // Limb darkening (per-params override; prefer LightCurve.limb_darkening for global setting)
    double limb_darkening_c() const { return p_.limb_darkening_c; }
    void set_limb_darkening_c(double v) { p_.limb_darkening_c = v; }
    double limb_darkening_d() const { return p_.limb_darkening_d; }
    void set_limb_darkening_d(double v) { p_.limb_darkening_d = v; }

private:
    lcbi_params p_;
};

} // namespace lcbinint::lc

#include "light_curve.hpp"
#include "lcbinint/model/lens_parameters.hpp"
#include "lcbinint/model/trajectory.hpp"

#include <stdexcept>

namespace lcbinint::lc {

LightCurve::LightCurve(lcbi_options opts, double ld_c, double ld_d, Effects effects)
    : opts_(opts), ld_c_(ld_c), ld_d_(ld_d), effects_(std::move(effects))
{
    // Effects override lcbi_options for physics-mode fields.
    if (effects_.parallax) opts_.parallax_mode = 1;
    opts_.xallarap_param_type = effects_.xallarap;
}

lcbi_params LightCurve::apply_coords(const lcbi_params& params) const
{
    const bool needs_tref = (opts_.parallax_mode != 0)
                         || (effects_.orbital_motion != LCBI_ORBIT_STATIC);
    if (needs_tref && !effects_.t_ref.has_value())
        throw std::runtime_error(
            "LightCurve: t_ref must be set when using parallax or orbital motion");

    lcbi_params p = params;
    p.orbital_motion_mode = effects_.orbital_motion;
    if (ld_c_ != 0.0 || ld_d_ != 0.0) {
        p.limb_darkening_c = ld_c_;
        p.limb_darkening_d = ld_d_;
    }
    if (effects_.sky) {
        p.ra  = effects_.sky->ra_deg();
        p.dec = effects_.sky->dec_deg();
    }
    // Site is only applied when terrestrial parallax is explicitly enabled.
    if (effects_.terrestrial && effects_.site) {
        p.obs_lat = effects_.site->lat_deg();
        p.obs_lon = effects_.site->lon_deg();
    }
    if (effects_.t_ref.has_value())
        p.tfix = *effects_.t_ref;
    return p;
}

std::vector<double> LightCurve::magnification(
    const std::vector<double>& times,
    const lcbi_params&         params) const
{
    const lcbi_params p = apply_coords(params);

    const int n = static_cast<int>(times.size());
    std::vector<lcbi_result> results(n);
    const lcbi_status status =
        lcbi_magnification_array(times.data(), n, &p, &opts_, results.data());
    if (status != LCBI_OK)
        throw std::runtime_error(lcbi_status_string(status));

    std::vector<double> mags(n);
    for (int i = 0; i < n; ++i)
        mags[i] = results[i].magnification;
    return mags;
}

std::vector<double> LightCurve::magnification_binary(
    const std::vector<double>& times,
    const lcbi_params&         params,
    double                     q_source,
    double                     t0_2,
    double                     u0_2) const
{
    const lcbi_params p1 = apply_coords(params);
    lcbi_params p2 = p1;
    p2.t0   = t0_2;
    p2.umin = u0_2;

    const int n = static_cast<int>(times.size());
    std::vector<lcbi_result> r1(n), r2(n);

    lcbi_status s = lcbi_magnification_array(times.data(), n, &p1, &opts_, r1.data());
    if (s != LCBI_OK) throw std::runtime_error(lcbi_status_string(s));
    s = lcbi_magnification_array(times.data(), n, &p2, &opts_, r2.data());
    if (s != LCBI_OK) throw std::runtime_error(lcbi_status_string(s));

    std::vector<double> mags(n);
    const double denom = 1.0 + q_source;
    for (int i = 0; i < n; ++i)
        mags[i] = (r1[i].magnification + q_source * r2[i].magnification) / denom;
    return mags;
}

std::vector<double> LightCurve::magnification_binary(
    const std::vector<double>& times,
    const lcbi_params&         params1,
    double                     q_source,
    const lcbi_params&         params2) const
{
    const lcbi_params p1 = apply_coords(params1);
    const lcbi_params p2 = apply_coords(params2);

    const int n = static_cast<int>(times.size());
    std::vector<lcbi_result> r1(n), r2(n);

    lcbi_status s = lcbi_magnification_array(times.data(), n, &p1, &opts_, r1.data());
    if (s != LCBI_OK) throw std::runtime_error(lcbi_status_string(s));
    s = lcbi_magnification_array(times.data(), n, &p2, &opts_, r2.data());
    if (s != LCBI_OK) throw std::runtime_error(lcbi_status_string(s));

    std::vector<double> mags(n);
    const double denom = 1.0 + q_source;
    for (int i = 0; i < n; ++i)
        mags[i] = (r1[i].magnification + q_source * r2[i].magnification) / denom;
    return mags;
}

} // namespace lcbinint::lc

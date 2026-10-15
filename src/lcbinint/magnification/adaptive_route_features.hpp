#pragma once

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <complex>

namespace lcbinint::magnification::adaptive_detail {

// Uses already computed centre images. No extra root solve or integration.
// The estimate describes local image geometry; it is not an error certificate.
inline std::array<double, 15> adaptive_route_features(
    double separation, double mass_ratio, double rho, double limb,
    double relative_tolerance, const std::vector<BinaryImage>& images)
{
    const double q = std::min(mass_ratio, 1.0 / mass_ratio);
    const double a = mass_ratio < 1.0 ? -std::abs(separation) : std::abs(separation);
    const double m1 = 1.0 / (1.0 + q), m2 = q * m1;
    double amp=0, radial=0, angular=0, axis=0, perimeter=0, curvature=0;
    double min_radius=1e100, max_radius=0, negative=0;
    for (const auto& image : images) {
        const std::complex<double> z(image.position.x, image.position.y), d=z-a;
        const auto shear=m1/(d*d)+m2/(z*z);
        const double det=image.jacobian_determinant;
        const double mu=1/std::max(1e-15,std::abs(det));
        // Inverse real Jacobian applied to radial and tangential image axes.
        const double j00=1+shear.real(), j01=-shear.imag(), j11=1-shear.real();
        const double radius=std::abs(z), ux=image.position.x/std::max(radius,1e-15);
        const double uy=image.position.y/std::max(radius,1e-15);
        const double radial_scale=std::hypot(j11*ux-j01*uy,-j01*ux+j00*uy)*mu;
        const double tangent_scale=std::hypot(-j11*uy-j01*ux,j01*uy+j00*ux)*mu;
        amp+=mu; radial+=radial_scale; angular+=tangent_scale/std::max(radius,rho);
        axis+=std::hypot(j11,j01)*mu+std::hypot(j00,j01)*mu;
        perimeter+=std::hypot(radial_scale,tangent_scale);
        curvature=std::max(curvature,rho*std::abs(2.0*m1/(d*d*d)+2.0*m2/(z*z*z))*mu);
        min_radius=std::min(min_radius,radius);max_radius=std::max(max_radius,radius);
        if(det<0)negative+=mu;
    }
    auto log=[](double x){return std::log10(std::max(1e-15,x));};
    return {log(std::abs(separation)),log(q),log(rho),limb,log(relative_tolerance),
        log(amp),double(images.size()),log(radial),log(angular),log(axis),
        log(perimeter),log(curvature),log(min_radius),log(max_radius),negative/std::max(amp,1e-15)};
}
} // namespace lcbinint::magnification::adaptive_detail

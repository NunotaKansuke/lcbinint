#include "lcbinint/model/triple_lens_geometry.hpp"

#include <cmath>

namespace lcbinint::model {

TripleLensGeometry make_triple_lens_geometry(
    double separation,
    double mass_ratio,
    double secondary_mass_ratio,
    double secondary_separation,
    double secondary_angle)
{
    TripleLensGeometry geometry;
    const double q = mass_ratio;
    const double q2 = secondary_mass_ratio;
    const double eps2 = q / (1.0 + q + q2);
    const double eps3 = q2 / (1.0 + q + q2);
    const double eps1 = 1.0 - eps2 - eps3;
    const double eps4 = eps2 + eps3;

    const double x1 = -eps4 * separation;
    const double y1 = 0.0;
    const double x4 = eps1 * separation;
    const double y4 = 0.0;
    const double dx = secondary_separation * std::cos(secondary_angle);
    const double dy = secondary_separation * std::sin(secondary_angle);

    geometry.lens_positions[0] = {x1, y1};
    geometry.lens_positions[1] = {x4 + eps3 / eps4 * dx, y4 + eps3 / eps4 * dy};
    geometry.lens_positions[2] = {x4 - eps2 / eps4 * dx, y4 - eps2 / eps4 * dy};
    geometry.masses = {eps1, eps2, eps3};
    return geometry;
}

TripleLensGeometry make_triple_lens_geometry_vbm(
    double d12, double q, double d13, double psi, double q2)
{
    const double total = 1.0 + q + q2;
    const double z1x = q * d12 / (1.0 + q);
    const double z2x = -d12 / (1.0 + q);
    const double z3x = z1x - d13 * std::cos(psi);
    const double z3y = -d13 * std::sin(psi);
    TripleLensGeometry geometry;
    geometry.lens_positions[0] = {z1x, 0.0};
    geometry.lens_positions[1] = {z2x, 0.0};
    geometry.lens_positions[2] = {z3x, z3y};
    geometry.masses = {1.0/total, q/total, q2/total};
    return geometry;
}

SourcePosition triple_lens_equation(
    const TripleLensGeometry& geometry,
    SourcePosition image)
{
    Complex source(image.x, image.y);
    const Complex image_conjugate = std::conj(source);
    for (std::size_t i = 0; i < geometry.lens_positions.size(); ++i) {
        const Complex lens(
            geometry.lens_positions[i].x,
            geometry.lens_positions[i].y);
        source -= geometry.masses[i] / (image_conjugate - std::conj(lens));
    }
    return {source.real(), source.imag()};
}

} // namespace lcbinint::model

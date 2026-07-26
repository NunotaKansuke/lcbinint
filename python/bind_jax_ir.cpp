#include "bind_jax_ir.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

enum class MomentMode {
    uniform = 1,
    linear = 2,
    two_coefficient = 3,
};

struct PhiDerivatives {
    double phi;
    double gradient_x;
    double gradient_y;
    double laplacian;
};

struct LensConstants {
    double lens_1_x;
    double lens_2_x;
    double mass_1;
    double mass_2;
};

struct KernelResult {
    std::array<double, 3> moments{0.0, 0.0, 0.0};
    std::int64_t boundary_cells = 0;
    std::int64_t active_cells = 0;
};

MomentMode parse_moment_mode(const std::string& value)
{
    if (value == "uniform") return MomentMode::uniform;
    if (value == "linear") return MomentMode::linear;
    if (value == "two_coefficient") return MomentMode::two_coefficient;
    throw std::invalid_argument(
        "moment_mode must be 'uniform', 'linear', or 'two_coefficient'");
}

int moment_count(MomentMode mode)
{
    return static_cast<int>(mode);
}

PhiDerivatives phi_derivatives(
    double image_x,
    double image_y,
    double source_x,
    double source_y,
    const LensConstants& lens,
    double inverse_source_radius_squared)
{
    const double dx_1 = image_x - lens.lens_1_x;
    const double dx_2 = image_x - lens.lens_2_x;
    const double y_squared = image_y * image_y;
    const double radius_1_squared = dx_1 * dx_1 + y_squared;
    const double radius_2_squared = dx_2 * dx_2 + y_squared;
    const double inverse_radius_1_squared = 1.0 / radius_1_squared;
    const double inverse_radius_2_squared = 1.0 / radius_2_squared;

    const double mapped_x =
        image_x - lens.mass_1 * dx_1 * inverse_radius_1_squared
        - lens.mass_2 * dx_2 * inverse_radius_2_squared;
    const double mapped_y =
        image_y - lens.mass_1 * image_y * inverse_radius_1_squared
        - lens.mass_2 * image_y * inverse_radius_2_squared;
    const double shear_real =
        lens.mass_1 * (dx_1 * dx_1 - y_squared)
            * inverse_radius_1_squared * inverse_radius_1_squared
        + lens.mass_2 * (dx_2 * dx_2 - y_squared)
            * inverse_radius_2_squared * inverse_radius_2_squared;
    const double shear_cross =
        2.0 * image_y
        * (lens.mass_1 * dx_1
               * inverse_radius_1_squared * inverse_radius_1_squared
           + lens.mass_2 * dx_2
               * inverse_radius_2_squared * inverse_radius_2_squared);
    const double du_dx = 1.0 + shear_real;
    const double du_dy = shear_cross;
    const double dv_dx = shear_cross;
    const double dv_dy = 1.0 - shear_real;
    const double residual_x = mapped_x - source_x;
    const double residual_y = mapped_y - source_y;

    return {
        1.0 - (residual_x * residual_x + residual_y * residual_y)
                  * inverse_source_radius_squared,
        -2.0 * inverse_source_radius_squared
            * (residual_x * du_dx + residual_y * dv_dx),
        -2.0 * inverse_source_radius_squared
            * (residual_x * du_dy + residual_y * dv_dy),
        -2.0 * inverse_source_radius_squared
            * (du_dx * du_dx + du_dy * du_dy
               + dv_dx * dv_dx + dv_dy * dv_dy),
    };
}

double positive_power(double value, double power)
{
    return value > 0.0 ? std::pow(value, power) : 0.0;
}

double affine_unit_square_moment(
    double lower_left,
    double delta_x,
    double delta_y,
    double power)
{
    const double scale = std::max(
        1.0e-14,
        std::max(std::abs(lower_left), std::max(std::abs(delta_x), std::abs(delta_y))));
    const double slope_threshold = 1.0e-6 * scale;
    const bool x_small = std::abs(delta_x) <= slope_threshold;
    const bool y_small = std::abs(delta_y) <= slope_threshold;

    if (x_small && y_small) {
        const double centre = lower_left + 0.5 * (delta_x + delta_y);
        if (power == 0.0) return centre > 0.0 ? 1.0 : 0.0;
        return positive_power(centre, power);
    }
    if (x_small || y_small) {
        const double delta = x_small ? delta_y : delta_x;
        const double intercept =
            lower_left + 0.5 * (x_small ? delta_x : delta_y);
        return (
            positive_power(intercept + delta, power + 1.0)
            - positive_power(intercept, power + 1.0))
            / (delta * (power + 1.0));
    }
    const double numerator =
        positive_power(lower_left + delta_x + delta_y, power + 2.0)
        - positive_power(lower_left + delta_x, power + 2.0)
        - positive_power(lower_left + delta_y, power + 2.0)
        + positive_power(lower_left, power + 2.0);
    return numerator
        / (delta_x * delta_y * (power + 1.0) * (power + 2.0));
}

void add_affine_moments(
    std::array<double, 3>& result,
    const PhiDerivatives& values,
    double cell_size,
    MomentMode mode)
{
    const double delta_x = values.gradient_x * cell_size;
    const double delta_y = values.gradient_y * cell_size;
    const double lower_left = values.phi - 0.5 * (delta_x + delta_y);
    const double area = cell_size * cell_size;
    constexpr std::array<double, 3> powers{0.0, 0.5, 0.25};
    for (int index = 0; index < moment_count(mode); ++index) {
        result[index] += area * affine_unit_square_moment(
            lower_left, delta_x, delta_y, powers[index]);
    }
}

void add_interior_moments(
    std::array<double, 3>& result,
    const PhiDerivatives& values,
    double cell_size,
    MomentMode mode)
{
    const double area = cell_size * cell_size;
    result[0] += area;
    if (mode == MomentMode::uniform) return;

    const double sqrt_phi = std::sqrt(values.phi);
    const double delta_squared =
        cell_size * cell_size
        * (values.gradient_x * values.gradient_x
           + values.gradient_y * values.gradient_y);
    const double laplacian_term = cell_size * cell_size * values.laplacian;
    result[1] += area * (
        sqrt_phi + laplacian_term / (48.0 * sqrt_phi)
        - delta_squared / (96.0 * values.phi * sqrt_phi));
    if (mode == MomentMode::linear) return;

    const double fourth_root = std::sqrt(sqrt_phi);
    result[2] += area * (
        fourth_root
        + laplacian_term / (96.0 * sqrt_phi * fourth_root)
        - delta_squared
            / (128.0 * values.phi * sqrt_phi * fourth_root));
}

KernelResult fixed_support_kernel(
    const py::detail::unchecked_reference<double, 2>& origins,
    const py::detail::unchecked_reference<bool, 1>& mask,
    double cell_size,
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    double limb_d,
    int tile_size,
    MomentMode mode,
    int boundary_subdivision)
{
    KernelResult result;
    const double inverse_source_radius_squared =
        1.0 / (source_radius * source_radius);
    const double total_mass = 1.0 + mass_ratio;
    const LensConstants lens{
        -mass_ratio / total_mass * separation,
        separation / total_mass,
        1.0 / total_mass,
        mass_ratio / total_mass,
    };
    const double subcell_size = cell_size / boundary_subdivision;

    for (py::ssize_t tile = 0; tile < origins.shape(0); ++tile) {
        if (!mask(tile)) continue;
        for (int iy = 0; iy < tile_size; ++iy) {
            const double image_y =
                origins(tile, 1) + (static_cast<double>(iy) + 0.5) * cell_size;
            for (int ix = 0; ix < tile_size; ++ix) {
                const double image_x =
                    origins(tile, 0) + (static_cast<double>(ix) + 0.5) * cell_size;
                const auto values = phi_derivatives(
                    image_x, image_y, source_x, source_y, lens,
                    inverse_source_radius_squared);
                const double half_delta_x =
                    0.5 * values.gradient_x * cell_size;
                const double half_delta_y =
                    0.5 * values.gradient_y * cell_size;
                const double extent =
                    std::abs(half_delta_x) + std::abs(half_delta_y);
                const bool fully_inside = values.phi - extent > 0.0;
                const bool fully_outside = values.phi + extent <= 0.0;
                const bool geometric_boundary = !(fully_inside || fully_outside);
                bool detailed = geometric_boundary;
                if (mode == MomentMode::two_coefficient && limb_d != 0.0
                    && fully_inside) {
                    const double relative_variation =
                        (extent + 0.125 * std::abs(values.laplacian)
                                      * cell_size * cell_size)
                        / std::max(values.phi, 1.0e-30);
                    detailed = relative_variation > 0.2;
                }

                if (detailed) {
                    ++result.boundary_cells;
                    ++result.active_cells;
                    for (int sy = 0; sy < boundary_subdivision; ++sy) {
                        const double offset_y =
                            ((static_cast<double>(sy) + 0.5)
                                 / boundary_subdivision
                             - 0.5)
                            * cell_size;
                        for (int sx = 0; sx < boundary_subdivision; ++sx) {
                            const double offset_x =
                                ((static_cast<double>(sx) + 0.5)
                                     / boundary_subdivision
                                 - 0.5)
                                * cell_size;
                            const auto sub_values = phi_derivatives(
                                image_x + offset_x, image_y + offset_y,
                                source_x, source_y, lens,
                                inverse_source_radius_squared);
                            add_affine_moments(
                                result.moments, sub_values, subcell_size, mode);
                        }
                    }
                } else if (fully_inside) {
                    ++result.active_cells;
                    add_interior_moments(
                        result.moments, values, cell_size, mode);
                }
            }
        }
    }
    return result;
}

py::tuple fixed_support_forward(
    py::array_t<double, py::array::c_style | py::array::forcecast> tile_origins,
    py::array_t<bool, py::array::c_style | py::array::forcecast> tile_mask,
    double cell_size,
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    double limb_c,
    double limb_d,
    int tile_size,
    const std::string& moment_mode,
    int boundary_subdivision)
{
    if (tile_origins.ndim() != 2 || tile_origins.shape(1) != 2) {
        throw std::invalid_argument("tile_origins must have shape (N, 2)");
    }
    if (tile_mask.ndim() != 1
        || tile_mask.shape(0) != tile_origins.shape(0)) {
        throw std::invalid_argument("tile_mask must have shape (N,)");
    }
    if (!(cell_size > 0.0) || !(source_radius > 0.0)) {
        throw std::invalid_argument("cell_size and source_radius must be positive");
    }
    if (tile_size <= 0) {
        throw std::invalid_argument("tile_size must be positive");
    }
    if (boundary_subdivision < 1 || boundary_subdivision > 4) {
        throw std::invalid_argument("boundary_subdivision must be 1, 2, 3, or 4");
    }
    const auto mode = parse_moment_mode(moment_mode);
    const auto result = fixed_support_kernel(
        tile_origins.unchecked<2>(), tile_mask.unchecked<1>(), cell_size,
        source_x, source_y, separation, mass_ratio, source_radius, limb_d,
        tile_size, mode, boundary_subdivision);

    py::array_t<double> moments(moment_count(mode));
    auto output = moments.mutable_unchecked<1>();
    for (int index = 0; index < moment_count(mode); ++index) {
        output(index) = result.moments[index];
    }

    const double pi = std::acos(-1.0);
    double magnification;
    if (mode == MomentMode::uniform) {
        magnification = result.moments[0]
            / (pi * source_radius * source_radius);
    } else if (mode == MomentMode::linear) {
        magnification =
            ((1.0 - limb_c) * result.moments[0]
             + limb_c * result.moments[1])
            / (pi * source_radius * source_radius * (1.0 - limb_c / 3.0));
    } else {
        const double source_flux =
            pi * source_radius * source_radius
            * (1.0 - limb_c / 3.0 - limb_d / 5.0);
        magnification =
            source_flux > 0.0
            ? ((1.0 - limb_c - limb_d) * result.moments[0]
               + limb_c * result.moments[1] + limb_d * result.moments[2])
                / source_flux
            : std::numeric_limits<double>::quiet_NaN();
    }
    return py::make_tuple(
        magnification, std::move(moments), result.boundary_cells,
        result.active_cells);
}

}  // namespace

void register_jax_ir_submodule(py::module_& parent)
{
    auto module = parent.def_submodule(
        "_jax_ir", "Experimental C++ execution backend for JAX inverse rays");
    module.def(
        "fixed_support_forward",
        &fixed_support_forward,
        py::arg("tile_origins"),
        py::arg("tile_mask"),
        py::arg("cell_size"),
        py::arg("source_x"),
        py::arg("source_y"),
        py::arg("separation"),
        py::arg("mass_ratio"),
        py::arg("source_radius"),
        py::arg("limb_c") = 0.0,
        py::arg("limb_d") = 0.0,
        py::kw_only(),
        py::arg("tile_size") = 8,
        py::arg("moment_mode") = "two_coefficient",
        py::arg("boundary_subdivision") = 4,
        "Evaluate the JAX fixed-support cell algorithm in native C++.");
}

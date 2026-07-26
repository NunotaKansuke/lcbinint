#include "bind_jax_ir.hpp"

#include "lcbinint/math/polynomial_roots.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef LCBININT_HAS_JAX_FFI
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#endif

namespace py = pybind11;

namespace {

enum class MomentMode {
    uniform = 1,
    linear = 2,
    two_coefficient = 3,
};

constexpr std::size_t kernel_derivative_count = 5;
constexpr std::size_t parameter_count = 7;

struct Jet {
    double value = 0.0;
    std::array<double, kernel_derivative_count> derivative{};

    Jet() = default;
    Jet(double scalar) : value(scalar) {}

    static Jet variable(double scalar, std::size_t index)
    {
        Jet result(scalar);
        result.derivative[index] = 1.0;
        return result;
    }
};

Jet operator+(const Jet& left, const Jet& right)
{
    Jet result(left.value + right.value);
    for (std::size_t index = 0; index < kernel_derivative_count; ++index) {
        result.derivative[index] =
            left.derivative[index] + right.derivative[index];
    }
    return result;
}

Jet operator-(const Jet& left, const Jet& right)
{
    Jet result(left.value - right.value);
    for (std::size_t index = 0; index < kernel_derivative_count; ++index) {
        result.derivative[index] =
            left.derivative[index] - right.derivative[index];
    }
    return result;
}

Jet operator-(const Jet& value)
{
    Jet result(-value.value);
    for (std::size_t index = 0; index < kernel_derivative_count; ++index) {
        result.derivative[index] = -value.derivative[index];
    }
    return result;
}

Jet operator*(const Jet& left, const Jet& right)
{
    Jet result(left.value * right.value);
    for (std::size_t index = 0; index < kernel_derivative_count; ++index) {
        result.derivative[index] =
            left.derivative[index] * right.value
            + left.value * right.derivative[index];
    }
    return result;
}

Jet operator/(const Jet& left, const Jet& right)
{
    Jet result(left.value / right.value);
    const double inverse_denominator_squared =
        1.0 / (right.value * right.value);
    for (std::size_t index = 0; index < kernel_derivative_count; ++index) {
        result.derivative[index] =
            (left.derivative[index] * right.value
             - left.value * right.derivative[index])
            * inverse_denominator_squared;
    }
    return result;
}

Jet& operator+=(Jet& left, const Jet& right)
{
    left = left + right;
    return left;
}

double scalar_value(double value)
{
    return value;
}

double scalar_value(const Jet& value)
{
    return value.value;
}

double scalar_sqrt(double value)
{
    return std::sqrt(value);
}

Jet scalar_sqrt(const Jet& value)
{
    const double root = std::sqrt(value.value);
    Jet result(root);
    const double scale = 0.5 / root;
    for (std::size_t index = 0; index < kernel_derivative_count; ++index) {
        result.derivative[index] = scale * value.derivative[index];
    }
    return result;
}

double scalar_pow(double value, double power)
{
    return std::pow(value, power);
}

Jet scalar_pow(const Jet& value, double power)
{
    const double powered = std::pow(value.value, power);
    Jet result(powered);
    const double scale = power * std::pow(value.value, power - 1.0);
    for (std::size_t index = 0; index < kernel_derivative_count; ++index) {
        result.derivative[index] = scale * value.derivative[index];
    }
    return result;
}

template <typename Scalar>
struct PhiDerivatives {
    Scalar phi;
    Scalar gradient_x;
    Scalar gradient_y;
    Scalar laplacian;
};

template <typename Scalar>
struct LensConstants {
    Scalar lens_1_x;
    Scalar lens_2_x;
    Scalar mass_1;
    Scalar mass_2;
};

template <typename Scalar>
struct KernelResult {
    std::array<Scalar, 3> moments{};
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

template <typename Scalar>
PhiDerivatives<Scalar> phi_derivatives(
    double image_x,
    double image_y,
    const Scalar& source_x,
    const Scalar& source_y,
    const LensConstants<Scalar>& lens,
    const Scalar& inverse_source_radius_squared)
{
    const Scalar dx_1 = image_x - lens.lens_1_x;
    const Scalar dx_2 = image_x - lens.lens_2_x;
    const double y_squared = image_y * image_y;
    const Scalar radius_1_squared = dx_1 * dx_1 + y_squared;
    const Scalar radius_2_squared = dx_2 * dx_2 + y_squared;
    const Scalar inverse_radius_1_squared = 1.0 / radius_1_squared;
    const Scalar inverse_radius_2_squared = 1.0 / radius_2_squared;

    const Scalar mapped_x =
        image_x - lens.mass_1 * dx_1 * inverse_radius_1_squared
        - lens.mass_2 * dx_2 * inverse_radius_2_squared;
    const Scalar mapped_y =
        image_y - lens.mass_1 * image_y * inverse_radius_1_squared
        - lens.mass_2 * image_y * inverse_radius_2_squared;
    const Scalar shear_real =
        lens.mass_1 * (dx_1 * dx_1 - y_squared)
            * inverse_radius_1_squared * inverse_radius_1_squared
        + lens.mass_2 * (dx_2 * dx_2 - y_squared)
            * inverse_radius_2_squared * inverse_radius_2_squared;
    const Scalar shear_cross =
        2.0 * image_y
        * (lens.mass_1 * dx_1
               * inverse_radius_1_squared * inverse_radius_1_squared
           + lens.mass_2 * dx_2
               * inverse_radius_2_squared * inverse_radius_2_squared);
    const Scalar du_dx = 1.0 + shear_real;
    const Scalar du_dy = shear_cross;
    const Scalar dv_dx = shear_cross;
    const Scalar dv_dy = 1.0 - shear_real;
    const Scalar residual_x = mapped_x - source_x;
    const Scalar residual_y = mapped_y - source_y;

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

template <typename Scalar>
Scalar positive_power(const Scalar& value, double power)
{
    return scalar_value(value) > 0.0 ? scalar_pow(value, power) : Scalar(0.0);
}

template <typename Scalar>
Scalar affine_unit_square_moment(
    const Scalar& lower_left,
    const Scalar& delta_x,
    const Scalar& delta_y,
    double power)
{
    const double scale = std::max(
        1.0e-14,
        std::max(
            std::abs(scalar_value(lower_left)),
            std::max(
                std::abs(scalar_value(delta_x)),
                std::abs(scalar_value(delta_y)))));
    const double slope_threshold = 1.0e-6 * scale;
    const bool x_small = std::abs(scalar_value(delta_x)) <= slope_threshold;
    const bool y_small = std::abs(scalar_value(delta_y)) <= slope_threshold;

    if (x_small && y_small) {
        const Scalar centre = lower_left + 0.5 * (delta_x + delta_y);
        if (power == 0.0) {
            return scalar_value(centre) > 0.0 ? Scalar(1.0) : Scalar(0.0);
        }
        return positive_power(centre, power);
    }
    if (x_small || y_small) {
        const Scalar delta = x_small ? delta_y : delta_x;
        const Scalar intercept =
            lower_left + 0.5 * (x_small ? delta_x : delta_y);
        return (
            positive_power(intercept + delta, power + 1.0)
            - positive_power(intercept, power + 1.0))
            / (delta * (power + 1.0));
    }
    const Scalar numerator =
        positive_power(lower_left + delta_x + delta_y, power + 2.0)
        - positive_power(lower_left + delta_x, power + 2.0)
        - positive_power(lower_left + delta_y, power + 2.0)
        + positive_power(lower_left, power + 2.0);
    return numerator
        / (delta_x * delta_y * (power + 1.0) * (power + 2.0));
}

template <typename Scalar>
void add_affine_moments(
    std::array<Scalar, 3>& result,
    const PhiDerivatives<Scalar>& values,
    double cell_size,
    MomentMode mode)
{
    const Scalar delta_x = values.gradient_x * cell_size;
    const Scalar delta_y = values.gradient_y * cell_size;
    const Scalar lower_left = values.phi - 0.5 * (delta_x + delta_y);
    const double area = cell_size * cell_size;
    constexpr std::array<double, 3> powers{0.0, 0.5, 0.25};
    for (int index = 0; index < moment_count(mode); ++index) {
        result[index] += area * affine_unit_square_moment(
            lower_left, delta_x, delta_y, powers[index]);
    }
}

template <typename Scalar>
void add_interior_moments(
    std::array<Scalar, 3>& result,
    const PhiDerivatives<Scalar>& values,
    double cell_size,
    MomentMode mode)
{
    const double area = cell_size * cell_size;
    result[0] += area;
    if (mode == MomentMode::uniform) return;

    const Scalar sqrt_phi = scalar_sqrt(values.phi);
    const Scalar delta_squared =
        cell_size * cell_size
        * (values.gradient_x * values.gradient_x
           + values.gradient_y * values.gradient_y);
    const Scalar laplacian_term = cell_size * cell_size * values.laplacian;
    result[1] += area * (
        sqrt_phi + laplacian_term / (48.0 * sqrt_phi)
        - delta_squared / (96.0 * values.phi * sqrt_phi));
    if (mode == MomentMode::linear) return;

    const Scalar fourth_root = scalar_sqrt(sqrt_phi);
    result[2] += area * (
        fourth_root
        + laplacian_term / (96.0 * sqrt_phi * fourth_root)
        - delta_squared
            / (128.0 * values.phi * sqrt_phi * fourth_root));
}

template <typename Scalar>
KernelResult<Scalar> fixed_support_kernel(
    const double* origins,
    const bool* mask,
    std::int64_t tile_count,
    double cell_size,
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& source_radius,
    const Scalar& limb_d,
    int tile_size,
    MomentMode mode,
    int boundary_subdivision)
{
    KernelResult<Scalar> result;
    const Scalar inverse_source_radius_squared =
        1.0 / (source_radius * source_radius);
    const Scalar total_mass = 1.0 + mass_ratio;
    const LensConstants<Scalar> lens{
        -mass_ratio / total_mass * separation,
        separation / total_mass,
        1.0 / total_mass,
        mass_ratio / total_mass,
    };
    const double subcell_size = cell_size / boundary_subdivision;

    for (std::int64_t tile = 0; tile < tile_count; ++tile) {
        if (!mask[tile]) continue;
        for (int iy = 0; iy < tile_size; ++iy) {
            const double image_y =
                origins[2 * tile + 1]
                + (static_cast<double>(iy) + 0.5) * cell_size;
            for (int ix = 0; ix < tile_size; ++ix) {
                const double image_x =
                    origins[2 * tile]
                    + (static_cast<double>(ix) + 0.5) * cell_size;
                const auto values = phi_derivatives(
                    image_x, image_y, source_x, source_y, lens,
                    inverse_source_radius_squared);
                const double half_delta_x =
                    0.5 * scalar_value(values.gradient_x) * cell_size;
                const double half_delta_y =
                    0.5 * scalar_value(values.gradient_y) * cell_size;
                const double extent =
                    std::abs(half_delta_x) + std::abs(half_delta_y);
                const double phi_value = scalar_value(values.phi);
                const bool fully_inside = phi_value - extent > 0.0;
                const bool fully_outside = phi_value + extent <= 0.0;
                const bool geometric_boundary = !(fully_inside || fully_outside);
                bool detailed = geometric_boundary;
                if (mode == MomentMode::two_coefficient
                    && scalar_value(limb_d) != 0.0
                    && fully_inside) {
                    const double relative_variation =
                        (extent
                         + 0.125 * std::abs(scalar_value(values.laplacian))
                                      * cell_size * cell_size)
                        / std::max(phi_value, 1.0e-30);
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

template <typename Scalar>
Scalar combine_magnification(
    const std::array<Scalar, 3>& moments,
    const Scalar& source_radius,
    const Scalar& limb_c,
    const Scalar& limb_d,
    MomentMode mode)
{
    const double pi = std::acos(-1.0);
    if (mode == MomentMode::uniform) {
        return moments[0] / (pi * source_radius * source_radius);
    }
    if (mode == MomentMode::linear) {
        return ((1.0 - limb_c) * moments[0] + limb_c * moments[1])
            / (pi * source_radius * source_radius
               * (1.0 - limb_c / 3.0));
    }
    const Scalar source_flux =
        pi * source_radius * source_radius
        * (1.0 - limb_c / 3.0 - limb_d / 5.0);
    if (scalar_value(source_flux) <= 0.0) {
        return Scalar(std::numeric_limits<double>::quiet_NaN());
    }
    return (
        (1.0 - limb_c - limb_d) * moments[0]
        + limb_c * moments[1] + limb_d * moments[2])
        / source_flux;
}

std::array<double, 2> limb_coefficient_derivatives(
    const std::array<Jet, 3>& moments,
    double source_radius,
    double limb_c,
    double limb_d,
    MomentMode mode)
{
    if (mode == MomentMode::uniform) return {0.0, 0.0};
    const double pi_rho_squared =
        std::acos(-1.0) * source_radius * source_radius;
    if (mode == MomentMode::linear) {
        const double numerator =
            (1.0 - limb_c) * moments[0].value
            + limb_c * moments[1].value;
        const double denominator =
            pi_rho_squared * (1.0 - limb_c / 3.0);
        const double numerator_derivative =
            moments[1].value - moments[0].value;
        const double denominator_derivative = -pi_rho_squared / 3.0;
        return {
            (numerator_derivative * denominator
             - numerator * denominator_derivative)
                / (denominator * denominator),
            0.0,
        };
    }

    const double numerator =
        (1.0 - limb_c - limb_d) * moments[0].value
        + limb_c * moments[1].value + limb_d * moments[2].value;
    const double denominator =
        pi_rho_squared * (1.0 - limb_c / 3.0 - limb_d / 5.0);
    const double denominator_squared = denominator * denominator;
    return {
        ((moments[1].value - moments[0].value) * denominator
         + numerator * pi_rho_squared / 3.0)
            / denominator_squared,
        ((moments[2].value - moments[0].value) * denominator
         + numerator * pi_rho_squared / 5.0)
            / denominator_squared,
    };
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
    const auto origins = tile_origins.unchecked<2>();
    const auto mask = tile_mask.unchecked<1>();
    const auto result = fixed_support_kernel(
        origins.data(0, 0), mask.data(0), tile_origins.shape(0),
        cell_size, source_x, source_y, separation, mass_ratio, source_radius,
        limb_d, tile_size, mode, boundary_subdivision);

    py::array_t<double> moments(moment_count(mode));
    auto output = moments.mutable_unchecked<1>();
    for (int index = 0; index < moment_count(mode); ++index) {
        output(index) = result.moments[index];
    }

    const double magnification = combine_magnification(
        result.moments, source_radius, limb_c, limb_d, mode);
    return py::make_tuple(
        magnification, std::move(moments), result.boundary_cells,
        result.active_cells);
}

#ifdef LCBININT_HAS_JAX_FFI
namespace ffi = xla::ffi;

constexpr std::size_t binary_root_count = 5;
constexpr std::size_t binary_root_parameter_count = 4;

template <std::size_t LeftSize, std::size_t RightSize>
std::array<lcbinint::Complex, LeftSize + RightSize - 1>
convolve_polynomials(
    const std::array<lcbinint::Complex, LeftSize>& left,
    const std::array<lcbinint::Complex, RightSize>& right)
{
    std::array<lcbinint::Complex, LeftSize + RightSize - 1> result{};
    for (std::size_t left_index = 0; left_index < LeftSize; ++left_index) {
        for (
            std::size_t right_index = 0;
            right_index < RightSize;
            ++right_index) {
            result[left_index + right_index] +=
                left[left_index] * right[right_index];
        }
    }
    return result;
}

std::array<lcbinint::Complex, 6> binary_lens_polynomial(
    lcbinint::Complex source,
    double separation,
    double mass_ratio)
{
    const double total_mass = 1.0 + mass_ratio;
    const double lens_1_x = -mass_ratio / total_mass * separation;
    const double lens_2_x = separation / total_mass;
    const double mass_1 = 1.0 / total_mass;
    const double mass_2 = mass_ratio / total_mass;
    const std::array<lcbinint::Complex, 2> factor_1{
        -lens_1_x, 1.0};
    const std::array<lcbinint::Complex, 2> factor_2{
        -lens_2_x, 1.0};
    const auto denominator = convolve_polynomials(factor_1, factor_2);
    std::array<lcbinint::Complex, 3> numerator{};
    for (std::size_t index = 0; index < denominator.size(); ++index) {
        numerator[index] = std::conj(source) * denominator[index];
    }
    numerator[0] += -mass_1 * lens_2_x - mass_2 * lens_1_x;
    numerator[1] += mass_1 + mass_2;
    auto conjugate_offset_1 = numerator;
    auto conjugate_offset_2 = numerator;
    for (std::size_t index = 0; index < denominator.size(); ++index) {
        conjugate_offset_1[index] -= lens_1_x * denominator[index];
        conjugate_offset_2[index] -= lens_2_x * denominator[index];
    }
    const std::array<lcbinint::Complex, 2> source_factor{
        -source, 1.0};
    auto polynomial = convolve_polynomials(
        convolve_polynomials(source_factor, conjugate_offset_1),
        conjugate_offset_2);
    const auto deflection_1 =
        convolve_polynomials(denominator, conjugate_offset_2);
    const auto deflection_2 =
        convolve_polynomials(denominator, conjugate_offset_1);
    for (std::size_t index = 0; index < deflection_1.size(); ++index) {
        polynomial[index] -=
            mass_1 * deflection_1[index] + mass_2 * deflection_2[index];
    }
    return polynomial;
}

struct BinaryRootResult {
    std::array<lcbinint::Complex, binary_root_count> roots{};
    std::array<bool, binary_root_count> converged{};
    std::array<bool, binary_root_count> physical{};
    std::array<double, binary_root_count> residuals{};
};

struct LensMapAtRoot {
    double mapped_x;
    double mapped_y;
    double du_dx;
    double du_dy;
    double dv_dx;
    double dv_dy;
};

LensMapAtRoot binary_lens_map_at_root(
    lcbinint::Complex root,
    double separation,
    double mass_ratio)
{
    const double image_x = root.real();
    const double image_y = root.imag();
    const double total_mass = 1.0 + mass_ratio;
    const double lens_1_x = -mass_ratio / total_mass * separation;
    const double lens_2_x = separation / total_mass;
    const double mass_1 = 1.0 / total_mass;
    const double mass_2 = mass_ratio / total_mass;
    const double dx_1 = image_x - lens_1_x;
    const double dx_2 = image_x - lens_2_x;
    const double y_squared = image_y * image_y;
    const double radius_1_squared = dx_1 * dx_1 + y_squared;
    const double radius_2_squared = dx_2 * dx_2 + y_squared;
    const double inverse_1 = 1.0 / radius_1_squared;
    const double inverse_2 = 1.0 / radius_2_squared;
    const double mapped_x =
        image_x - mass_1 * dx_1 * inverse_1
        - mass_2 * dx_2 * inverse_2;
    const double mapped_y =
        image_y - mass_1 * image_y * inverse_1
        - mass_2 * image_y * inverse_2;
    const double shear_real =
        mass_1 * (dx_1 * dx_1 - y_squared) * inverse_1 * inverse_1
        + mass_2 * (dx_2 * dx_2 - y_squared) * inverse_2 * inverse_2;
    const double shear_cross =
        2.0 * image_y
        * (mass_1 * dx_1 * inverse_1 * inverse_1
           + mass_2 * dx_2 * inverse_2 * inverse_2);
    return {
        mapped_x,
        mapped_y,
        1.0 + shear_real,
        shear_cross,
        shear_cross,
        1.0 - shear_real,
    };
}

BinaryRootResult solve_binary_images(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio)
{
    const lcbinint::Complex source{source_x, source_y};
    const auto coefficients = binary_lens_polynomial(
        source, separation, mass_ratio);
    double coefficient_scale = 0.0;
    for (const auto& coefficient : coefficients) {
        coefficient_scale =
            std::max(coefficient_scale, std::abs(coefficient));
    }
    const bool use_quartic =
        std::abs(coefficients[5])
        <= 1.0e-10 * std::max(coefficient_scale, 1.0e-30);
    const std::size_t coefficient_count = use_quartic ? 5 : 6;
    std::vector<lcbinint::Complex> active_coefficients(
        coefficients.begin(),
        coefficients.begin()
            + static_cast<std::ptrdiff_t>(coefficient_count));
    const auto solved =
        lcbinint::math::PolynomialRootSolver().solve(active_coefficients);
    BinaryRootResult result;
    result.residuals.fill(std::numeric_limits<double>::infinity());
    if (solved.status != lcbinint::math::RootSolverStatus::ok) return result;

    for (
        std::size_t index = 0;
        index < solved.roots.size() && index < binary_root_count;
        ++index) {
        lcbinint::Complex root = solved.roots[index];
        bool converged =
            std::isfinite(root.real()) && std::isfinite(root.imag());
        for (int polish = 0; polish < 6 && converged; ++polish) {
            const auto mapped =
                binary_lens_map_at_root(root, separation, mass_ratio);
            const double residual_x = mapped.mapped_x - source_x;
            const double residual_y = mapped.mapped_y - source_y;
            const double determinant =
                mapped.du_dx * mapped.dv_dy
                - mapped.du_dy * mapped.dv_dx;
            if (
                !(std::abs(determinant) > 1.0e-14)
                || !std::isfinite(determinant)) {
                break;
            }
            const double delta_x =
                (mapped.dv_dy * residual_x - mapped.du_dy * residual_y)
                / determinant;
            const double delta_y =
                (-mapped.dv_dx * residual_x + mapped.du_dx * residual_y)
                / determinant;
            const double step_scale =
                std::max(1.0, std::hypot(delta_x, delta_y) / 0.5);
            root -= lcbinint::Complex(delta_x, delta_y) / step_scale;
            converged =
                std::isfinite(root.real()) && std::isfinite(root.imag());
        }
        result.roots[index] = root;
        result.converged[index] = converged;
        if (!converged) continue;
        const auto mapped =
            binary_lens_map_at_root(root, separation, mass_ratio);
        result.residuals[index] = std::hypot(
            mapped.mapped_x - source_x,
            mapped.mapped_y - source_y);
        result.physical[index] =
            std::isfinite(result.residuals[index])
            && result.residuals[index]
                <= 1.0e-9 * (1.0 + std::abs(source));
    }
    for (std::size_t left = 0; left < binary_root_count; ++left) {
        if (!result.physical[left]) continue;
        for (std::size_t right = left + 1; right < binary_root_count; ++right) {
            if (!result.physical[right]) continue;
            const double duplicate_tolerance =
                1.0e-8
                * (
                    1.0
                    + std::max(
                        std::abs(result.roots[left]),
                        std::abs(result.roots[right])));
            if (
                std::abs(result.roots[left] - result.roots[right])
                > duplicate_tolerance) {
                continue;
            }
            if (result.residuals[right] < result.residuals[left]) {
                result.physical[left] = false;
                break;
            }
            result.physical[right] = false;
        }
    }
    return result;
}

template <typename Scalar>
std::array<Scalar, 2> binary_lens_map_at_fixed_root(
    lcbinint::Complex root,
    const Scalar& separation,
    const Scalar& mass_ratio)
{
    const Scalar total_mass = 1.0 + mass_ratio;
    const Scalar lens_1_x = -mass_ratio / total_mass * separation;
    const Scalar lens_2_x = separation / total_mass;
    const Scalar mass_1 = 1.0 / total_mass;
    const Scalar mass_2 = mass_ratio / total_mass;
    const double image_x = root.real();
    const double image_y = root.imag();
    const Scalar dx_1 = image_x - lens_1_x;
    const Scalar dx_2 = image_x - lens_2_x;
    const double y_squared = image_y * image_y;
    const Scalar radius_1_squared = dx_1 * dx_1 + y_squared;
    const Scalar radius_2_squared = dx_2 * dx_2 + y_squared;
    return {
        image_x - mass_1 * dx_1 / radius_1_squared
            - mass_2 * dx_2 / radius_2_squared,
        image_y - mass_1 * image_y / radius_1_squared
            - mass_2 * image_y / radius_2_squared,
    };
}

void write_binary_root_outputs(
    const BinaryRootResult& result,
    double* root_coordinates,
    bool* physical,
    double* residuals,
    bool* converged)
{
    for (std::size_t index = 0; index < binary_root_count; ++index) {
        root_coordinates[2 * index] = result.roots[index].real();
        root_coordinates[2 * index + 1] = result.roots[index].imag();
        physical[index] = result.physical[index];
        residuals[index] = result.residuals[index];
        converged[index] = result.converged[index];
    }
}

ffi::Error validate_binary_root_outputs(
    ffi::ResultBufferR2<ffi::F64>& roots,
    ffi::ResultBufferR1<ffi::PRED>& physical,
    ffi::ResultBufferR1<ffi::F64>& residuals,
    ffi::ResultBufferR1<ffi::PRED>& converged)
{
    const auto root_dimensions = roots->dimensions();
    if (
        root_dimensions[0] != binary_root_count || root_dimensions[1] != 2
        || physical->dimensions()[0] != binary_root_count
        || residuals->dimensions()[0] != binary_root_count
        || converged->dimensions()[0] != binary_root_count) {
        return ffi::Error::InvalidArgument(
            "binary-root outputs must have five root slots");
    }
    return ffi::Error::Success();
}

ffi::Error binary_image_roots_ffi_impl(
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::ResultBufferR2<ffi::F64> roots,
    ffi::ResultBufferR1<ffi::PRED> physical,
    ffi::ResultBufferR1<ffi::F64> residuals,
    ffi::ResultBufferR1<ffi::PRED> converged)
{
    auto validation =
        validate_binary_root_outputs(roots, physical, residuals, converged);
    if (validation.failure()) return validation;
    const auto result = solve_binary_images(
        *source_x.typed_data(), *source_y.typed_data(),
        *separation.typed_data(), *mass_ratio.typed_data());
    write_binary_root_outputs(
        result, roots->typed_data(), physical->typed_data(),
        residuals->typed_data(), converged->typed_data());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    binary_image_roots_ffi_handler,
    binary_image_roots_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::PRED>>());

ffi::Error binary_image_roots_jacobian_ffi_impl(
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::ResultBufferR2<ffi::F64> roots,
    ffi::ResultBufferR1<ffi::PRED> physical,
    ffi::ResultBufferR1<ffi::F64> residuals,
    ffi::ResultBufferR1<ffi::PRED> converged,
    ffi::ResultBufferR3<ffi::F64> root_jacobian)
{
    auto validation =
        validate_binary_root_outputs(roots, physical, residuals, converged);
    if (validation.failure()) return validation;
    const auto jacobian_dimensions = root_jacobian->dimensions();
    if (
        jacobian_dimensions[0] != binary_root_count
        || jacobian_dimensions[1] != 2
        || jacobian_dimensions[2] != binary_root_parameter_count) {
        return ffi::Error::InvalidArgument(
            "root Jacobian must have shape (5, 2, 4)");
    }
    const double x = *source_x.typed_data();
    const double y = *source_y.typed_data();
    const double s = *separation.typed_data();
    const double q = *mass_ratio.typed_data();
    const auto result = solve_binary_images(x, y, s, q);
    write_binary_root_outputs(
        result, roots->typed_data(), physical->typed_data(),
        residuals->typed_data(), converged->typed_data());
    auto* jacobian = root_jacobian->typed_data();
    std::fill(
        jacobian,
        jacobian
            + binary_root_count * 2 * binary_root_parameter_count,
        0.0);

    const Jet separation_jet = Jet::variable(s, 0);
    const Jet mass_ratio_jet = Jet::variable(q, 1);
    for (std::size_t index = 0; index < binary_root_count; ++index) {
        if (!result.physical[index]) continue;
        const auto mapped = binary_lens_map_at_root(
            result.roots[index], s, q);
        const double determinant =
            mapped.du_dx * mapped.dv_dy - mapped.du_dy * mapped.dv_dx;
        if (!(std::abs(determinant) > 1.0e-12)) continue;
        const auto parameter_map = binary_lens_map_at_fixed_root(
            result.roots[index], separation_jet, mass_ratio_jet);
        for (
            std::size_t parameter = 0;
            parameter < binary_root_parameter_count;
            ++parameter) {
            const double rhs_x =
                parameter == 0
                ? 1.0
                : (
                      parameter >= 2
                          ? -parameter_map[0].derivative[parameter - 2]
                          : 0.0);
            const double rhs_y =
                parameter == 1
                ? 1.0
                : (
                      parameter >= 2
                          ? -parameter_map[1].derivative[parameter - 2]
                          : 0.0);
            jacobian[
                (index * 2) * binary_root_parameter_count + parameter] =
                (mapped.dv_dy * rhs_x - mapped.du_dy * rhs_y)
                / determinant;
            jacobian[
                (index * 2 + 1) * binary_root_parameter_count + parameter] =
                (-mapped.dv_dx * rhs_x + mapped.du_dx * rhs_y)
                / determinant;
        }
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    binary_image_roots_jacobian_ffi_handler,
    binary_image_roots_jacobian_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR3<ffi::F64>>());

std::uint64_t tile_key(std::int32_t x, std::int32_t y)
{
    return (
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32)
        | static_cast<std::uint32_t>(y);
}

bool tile_has_inside_probe(
    std::int32_t tile_x,
    std::int32_t tile_y,
    double tile_width,
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius)
{
    const double total_mass = 1.0 + mass_ratio;
    const double lens_1_x = -mass_ratio / total_mass * separation;
    const double lens_2_x = separation / total_mass;
    const double mass_1 = 1.0 / total_mass;
    const double mass_2 = mass_ratio / total_mass;
    const double source_radius_squared = source_radius * source_radius;
    const double origin_x = static_cast<double>(tile_x) * tile_width;
    const double origin_y = static_cast<double>(tile_y) * tile_width;
    constexpr std::array<double, 3> fractions{0.0, 0.5, 1.0};

    for (double fraction_y : fractions) {
        const double image_y = origin_y + fraction_y * tile_width;
        for (double fraction_x : fractions) {
            const double image_x = origin_x + fraction_x * tile_width;
            const double dx_1 = image_x - lens_1_x;
            const double dx_2 = image_x - lens_2_x;
            const double radius_1_squared =
                dx_1 * dx_1 + image_y * image_y;
            const double radius_2_squared =
                dx_2 * dx_2 + image_y * image_y;
            const double mapped_x =
                image_x - mass_1 * dx_1 / radius_1_squared
                - mass_2 * dx_2 / radius_2_squared;
            const double mapped_y =
                image_y - mass_1 * image_y / radius_1_squared
                - mass_2 * image_y / radius_2_squared;
            const double residual_x = mapped_x - source_x;
            const double residual_y = mapped_y - source_y;
            const double distance_squared =
                residual_x * residual_x + residual_y * residual_y;
            if (
                std::isfinite(distance_squared)
                && distance_squared <= source_radius_squared) {
                return true;
            }
        }
    }
    return false;
}

ffi::Error macro_tile_discovery_ffi_impl(
    ffi::BufferR2<ffi::F64> seed_coordinates,
    ffi::BufferR1<ffi::PRED> seed_physical,
    ffi::BufferR0<ffi::F64> tile_width,
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::ResultBufferR2<ffi::S32> tile_indices,
    ffi::ResultBufferR2<ffi::F64> tile_origins,
    ffi::ResultBufferR1<ffi::PRED> tile_mask,
    ffi::ResultBufferR1<ffi::PRED> active_mask,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::S32> visited_count,
    ffi::ResultBufferR0<ffi::S32> active_count,
    ffi::ResultBufferR0<ffi::S32> seed_count)
{
    const auto seed_dimensions = seed_coordinates.dimensions();
    const auto index_dimensions = tile_indices->dimensions();
    const auto origin_dimensions = tile_origins->dimensions();
    const std::int64_t capacity = index_dimensions[0];
    if (
        seed_dimensions[1] != 2
        || seed_physical.dimensions()[0] != seed_dimensions[0]) {
        return ffi::Error::InvalidArgument(
            "seed arrays must have shapes (N, 2) and (N,)");
    }
    if (
        capacity <= 0 || index_dimensions[1] != 2
        || origin_dimensions[0] != capacity || origin_dimensions[1] != 2
        || tile_mask->dimensions()[0] != capacity
        || active_mask->dimensions()[0] != capacity) {
        return ffi::Error::InvalidArgument(
            "discovery outputs have inconsistent capacities");
    }
    const double width = *tile_width.typed_data();
    const double rho = *source_radius.typed_data();
    if (!(width > 0.0) || !(rho > 0.0)) {
        return ffi::Error::InvalidArgument(
            "tile width and source radius must be positive");
    }

    auto* output_indices = tile_indices->typed_data();
    auto* output_origins = tile_origins->typed_data();
    auto* output_mask = tile_mask->typed_data();
    auto* output_active = active_mask->typed_data();
    std::fill(output_indices, output_indices + 2 * capacity, 0);
    std::fill(output_origins, output_origins + 2 * capacity, 0.0);
    std::fill(output_mask, output_mask + capacity, false);
    std::fill(output_active, output_active + capacity, false);

    std::vector<std::array<std::int32_t, 2>> queue;
    queue.reserve(static_cast<std::size_t>(capacity));
    std::unordered_map<std::uint64_t, std::int32_t> visited;
    visited.reserve(static_cast<std::size_t>(capacity));
    std::unordered_set<std::uint64_t> seeds;
    seeds.reserve(static_cast<std::size_t>(seed_dimensions[0]));
    bool did_overflow = false;

    const auto insert = [&](std::int32_t x, std::int32_t y) {
        const std::uint64_t key = tile_key(x, y);
        if (visited.find(key) != visited.end()) return true;
        if (queue.size() >= static_cast<std::size_t>(capacity)) {
            did_overflow = true;
            return false;
        }
        visited.emplace(key, static_cast<std::int32_t>(queue.size()));
        queue.push_back({x, y});
        return true;
    };

    const auto* coordinates = seed_coordinates.typed_data();
    const auto* physical = seed_physical.typed_data();
    for (std::int64_t index = 0; index < seed_dimensions[0]; ++index) {
        if (!physical[index]) continue;
        const auto x = static_cast<std::int32_t>(
            std::floor(coordinates[2 * index] / width));
        const auto y = static_cast<std::int32_t>(
            std::floor(coordinates[2 * index + 1] / width));
        if (insert(x, y)) seeds.insert(tile_key(x, y));
    }
    const std::int32_t unique_seed_count =
        static_cast<std::int32_t>(queue.size());
    constexpr std::array<std::array<std::int32_t, 2>, 4> neighbours{{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}
    }};
    std::int32_t active_total = 0;
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const auto tile = queue[head];
        const bool active =
            seeds.find(tile_key(tile[0], tile[1])) != seeds.end()
            || tile_has_inside_probe(
                tile[0], tile[1], width, *source_x.typed_data(),
                *source_y.typed_data(), *separation.typed_data(),
                *mass_ratio.typed_data(), rho);
        output_active[head] = active;
        active_total += static_cast<std::int32_t>(active);
        if (active) {
            for (const auto& neighbour : neighbours) {
                insert(tile[0] + neighbour[0], tile[1] + neighbour[1]);
            }
        }
    }

    for (std::size_t index = 0; index < queue.size(); ++index) {
        output_indices[2 * index] = queue[index][0];
        output_indices[2 * index + 1] = queue[index][1];
        output_origins[2 * index] = queue[index][0] * width;
        output_origins[2 * index + 1] = queue[index][1] * width;
        output_mask[index] = true;
    }
    *overflow->typed_data() = did_overflow;
    *visited_count->typed_data() =
        static_cast<std::int32_t>(queue.size());
    *active_count->typed_data() = active_total;
    *seed_count->typed_data() = unique_seed_count;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    macro_tile_discovery_ffi_handler,
    macro_tile_discovery_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR0<ffi::PRED>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::S32>>());

ffi::Error fixed_support_forward_ffi_impl(
    std::int64_t tile_size,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR2<ffi::F64> tile_origins,
    ffi::BufferR1<ffi::PRED> tile_mask,
    ffi::BufferR0<ffi::F64> cell_size,
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR0<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> moments,
    ffi::ResultBufferR0<ffi::S32> boundary_cells,
    ffi::ResultBufferR0<ffi::S32> active_cells)
{
    const auto origin_dimensions = tile_origins.dimensions();
    if (origin_dimensions[1] != 2) {
        return ffi::Error::InvalidArgument(
            "tile_origins must have shape (N, 2)");
    }
    if (tile_mask.dimensions()[0] != origin_dimensions[0]) {
        return ffi::Error::InvalidArgument(
            "tile_mask must have shape (N,)");
    }
    if (tile_size <= 0) {
        return ffi::Error::InvalidArgument("tile_size must be positive");
    }
    if (mode_value < 1 || mode_value > 3) {
        return ffi::Error::InvalidArgument("invalid moment mode");
    }
    const auto mode = static_cast<MomentMode>(mode_value);
    if (moments->dimensions()[0] != moment_count(mode)) {
        return ffi::Error::InvalidArgument(
            "moments output has the wrong length");
    }
    if (boundary_subdivision < 1 || boundary_subdivision > 4) {
        return ffi::Error::InvalidArgument(
            "boundary_subdivision must be 1, 2, 3, or 4");
    }
    if (!(*cell_size.typed_data() > 0.0)
        || !(*source_radius.typed_data() > 0.0)) {
        return ffi::Error::InvalidArgument(
            "cell_size and source_radius must be positive");
    }

    const auto result = fixed_support_kernel(
        tile_origins.typed_data(), tile_mask.typed_data(),
        origin_dimensions[0], *cell_size.typed_data(), *source_x.typed_data(),
        *source_y.typed_data(), *separation.typed_data(),
        *mass_ratio.typed_data(), *source_radius.typed_data(),
        *limb_d.typed_data(), static_cast<int>(tile_size), mode,
        static_cast<int>(boundary_subdivision));
    for (int index = 0; index < moment_count(mode); ++index) {
        moments->typed_data()[index] = result.moments[index];
    }

    const double rho = *source_radius.typed_data();
    const double c = *limb_c.typed_data();
    const double d = *limb_d.typed_data();
    *magnification->typed_data() =
        combine_magnification(result.moments, rho, c, d, mode);
    *boundary_cells->typed_data() =
        static_cast<std::int32_t>(result.boundary_cells);
    *active_cells->typed_data() =
        static_cast<std::int32_t>(result.active_cells);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    fixed_support_forward_ffi_handler,
    fixed_support_forward_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("tile_size")
        .Attr<std::int64_t>("moment_mode")
        .Attr<std::int64_t>("boundary_subdivision")
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::S32>>());

ffi::Error fixed_support_value_jacobian_ffi_impl(
    std::int64_t tile_size,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR2<ffi::F64> tile_origins,
    ffi::BufferR1<ffi::PRED> tile_mask,
    ffi::BufferR0<ffi::F64> cell_size,
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR0<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> moments,
    ffi::ResultBufferR0<ffi::S32> boundary_cells,
    ffi::ResultBufferR0<ffi::S32> active_cells,
    ffi::ResultBufferR1<ffi::F64> magnification_jacobian,
    ffi::ResultBufferR2<ffi::F64> moments_jacobian)
{
    const auto origin_dimensions = tile_origins.dimensions();
    if (origin_dimensions[1] != 2
        || tile_mask.dimensions()[0] != origin_dimensions[0]) {
        return ffi::Error::InvalidArgument(
            "support arrays must have shapes (N, 2) and (N,)");
    }
    if (tile_size <= 0 || mode_value < 1 || mode_value > 3) {
        return ffi::Error::InvalidArgument(
            "invalid tile size or moment mode");
    }
    const auto mode = static_cast<MomentMode>(mode_value);
    const auto moment_jacobian_dimensions = moments_jacobian->dimensions();
    if (moments->dimensions()[0] != moment_count(mode)
        || magnification_jacobian->dimensions()[0] != parameter_count
        || moment_jacobian_dimensions[0] != moment_count(mode)
        || moment_jacobian_dimensions[1] != parameter_count) {
        return ffi::Error::InvalidArgument(
            "value/Jacobian output shapes do not match the moment mode");
    }
    if (boundary_subdivision < 1 || boundary_subdivision > 4) {
        return ffi::Error::InvalidArgument(
            "boundary_subdivision must be 1, 2, 3, or 4");
    }
    if (!(*cell_size.typed_data() > 0.0)
        || !(*source_radius.typed_data() > 0.0)) {
        return ffi::Error::InvalidArgument(
            "cell_size and source_radius must be positive");
    }

    const Jet source_x_jet = Jet::variable(*source_x.typed_data(), 0);
    const Jet source_y_jet = Jet::variable(*source_y.typed_data(), 1);
    const Jet separation_jet = Jet::variable(*separation.typed_data(), 2);
    const Jet mass_ratio_jet = Jet::variable(*mass_ratio.typed_data(), 3);
    const Jet source_radius_jet =
        Jet::variable(*source_radius.typed_data(), 4);
    const Jet limb_c_jet(*limb_c.typed_data());
    const Jet limb_d_jet(*limb_d.typed_data());

    const auto result = fixed_support_kernel(
        tile_origins.typed_data(), tile_mask.typed_data(),
        origin_dimensions[0], *cell_size.typed_data(), source_x_jet,
        source_y_jet, separation_jet, mass_ratio_jet, source_radius_jet,
        limb_d_jet, static_cast<int>(tile_size), mode,
        static_cast<int>(boundary_subdivision));
    const Jet magnification_result = combine_magnification(
        result.moments, source_radius_jet, limb_c_jet, limb_d_jet, mode);
    const auto limb_derivatives = limb_coefficient_derivatives(
        result.moments, *source_radius.typed_data(), *limb_c.typed_data(),
        *limb_d.typed_data(), mode);

    *magnification->typed_data() = magnification_result.value;
    for (
        std::size_t parameter = 0;
        parameter < kernel_derivative_count;
        ++parameter) {
        magnification_jacobian->typed_data()[parameter] =
            magnification_result.derivative[parameter];
    }
    magnification_jacobian->typed_data()[5] = limb_derivatives[0];
    magnification_jacobian->typed_data()[6] = limb_derivatives[1];
    for (int moment = 0; moment < moment_count(mode); ++moment) {
        moments->typed_data()[moment] = result.moments[moment].value;
        for (
            std::size_t parameter = 0;
            parameter < parameter_count;
            ++parameter) {
            moments_jacobian->typed_data()[
                moment * parameter_count + parameter] =
                parameter < kernel_derivative_count
                ? result.moments[moment].derivative[parameter]
                : 0.0;
        }
    }
    *boundary_cells->typed_data() =
        static_cast<std::int32_t>(result.boundary_cells);
    *active_cells->typed_data() =
        static_cast<std::int32_t>(result.active_cells);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    fixed_support_value_jacobian_ffi_handler,
    fixed_support_value_jacobian_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("tile_size")
        .Attr<std::int64_t>("moment_mode")
        .Attr<std::int64_t>("boundary_subdivision")
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>());

py::capsule fixed_support_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(fixed_support_forward_ffi_handler));
}

py::capsule binary_image_roots_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(binary_image_roots_ffi_handler));
}

py::capsule binary_image_roots_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(binary_image_roots_jacobian_ffi_handler));
}

py::capsule macro_tile_discovery_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(macro_tile_discovery_ffi_handler));
}

py::capsule fixed_support_value_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            fixed_support_value_jacobian_ffi_handler));
}
#endif

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
#ifdef LCBININT_HAS_JAX_FFI
    module.def(
        "binary_image_roots_ffi",
        &binary_image_roots_ffi_capsule,
        "Return the typed XLA binary-image root FFI handler capsule.");
    module.def(
        "binary_image_roots_jacobian_ffi",
        &binary_image_roots_jacobian_ffi_capsule,
        "Return the typed XLA binary-image root/Jacobian FFI handler capsule.");
    module.def(
        "macro_tile_discovery_ffi",
        &macro_tile_discovery_ffi_capsule,
        "Return the typed XLA macro-tile discovery FFI handler capsule.");
    module.def(
        "fixed_support_forward_ffi",
        &fixed_support_forward_ffi_capsule,
        "Return the typed XLA FFI handler capsule.");
    module.def(
        "fixed_support_value_jacobian_ffi",
        &fixed_support_value_jacobian_ffi_capsule,
        "Return the typed XLA value/Jacobian FFI handler capsule.");
#endif
}

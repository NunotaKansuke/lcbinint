#include "bind_jax_ir.hpp"

#include "lcbinint/math/polynomial_roots.hpp"
#include "lcbinint/magnification/component_certificate.hpp"
#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/model/triple_lens_geometry.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef LCBININT_HAS_OPENMP
#include <omp.h>
#endif

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
constexpr std::size_t triple_kernel_derivative_count = 8;
constexpr std::size_t triple_parameter_count = 10;

template <std::size_t DerivativeCount>
struct JetBase {
    double value = 0.0;
    std::array<double, DerivativeCount> derivative{};

    JetBase() = default;
    JetBase(double scalar) : value(scalar) {}

    static JetBase variable(double scalar, std::size_t index)
    {
        JetBase result(scalar);
        result.derivative[index] = 1.0;
        return result;
    }
};

using Jet = JetBase<kernel_derivative_count>;
using DirectionalJet = JetBase<1>;
using TripleJet = JetBase<triple_kernel_derivative_count>;

template <std::size_t N>
JetBase<N> operator+(const JetBase<N>& left, const JetBase<N>& right)
{
    JetBase<N> result(left.value + right.value);
    for (std::size_t index = 0; index < N; ++index) {
        result.derivative[index] =
            left.derivative[index] + right.derivative[index];
    }
    return result;
}

template <std::size_t N>
JetBase<N> operator-(const JetBase<N>& left, const JetBase<N>& right)
{
    JetBase<N> result(left.value - right.value);
    for (std::size_t index = 0; index < N; ++index) {
        result.derivative[index] =
            left.derivative[index] - right.derivative[index];
    }
    return result;
}

template <std::size_t N>
JetBase<N> operator-(const JetBase<N>& value)
{
    JetBase<N> result(-value.value);
    for (std::size_t index = 0; index < N; ++index) {
        result.derivative[index] = -value.derivative[index];
    }
    return result;
}

template <std::size_t N>
JetBase<N> operator*(const JetBase<N>& left, const JetBase<N>& right)
{
    JetBase<N> result(left.value * right.value);
    for (std::size_t index = 0; index < N; ++index) {
        result.derivative[index] =
            left.derivative[index] * right.value
            + left.value * right.derivative[index];
    }
    return result;
}

template <std::size_t N>
JetBase<N> operator/(const JetBase<N>& left, const JetBase<N>& right)
{
    JetBase<N> result(left.value / right.value);
    const double inverse_denominator_squared =
        1.0 / (right.value * right.value);
    for (std::size_t index = 0; index < N; ++index) {
        result.derivative[index] =
            (left.derivative[index] * right.value
             - left.value * right.derivative[index])
            * inverse_denominator_squared;
    }
    return result;
}

template <std::size_t N>
JetBase<N>& operator+=(JetBase<N>& left, const JetBase<N>& right)
{
    left = left + right;
    return left;
}

template <std::size_t N>
JetBase<N>& operator+=(JetBase<N>& left, double right)
{
    left.value += right;
    return left;
}

template <std::size_t N>
JetBase<N> operator+(double left, const JetBase<N>& right)
{
    return JetBase<N>(left) + right;
}

template <std::size_t N>
JetBase<N> operator+(const JetBase<N>& left, double right)
{
    return left + JetBase<N>(right);
}

template <std::size_t N>
JetBase<N> operator-(double left, const JetBase<N>& right)
{
    return JetBase<N>(left) - right;
}

template <std::size_t N>
JetBase<N> operator-(const JetBase<N>& left, double right)
{
    return left - JetBase<N>(right);
}

template <std::size_t N>
JetBase<N> operator*(double left, const JetBase<N>& right)
{
    return JetBase<N>(left) * right;
}

template <std::size_t N>
JetBase<N> operator*(const JetBase<N>& left, double right)
{
    return left * JetBase<N>(right);
}

template <std::size_t N>
JetBase<N> operator/(double left, const JetBase<N>& right)
{
    return JetBase<N>(left) / right;
}

template <std::size_t N>
JetBase<N> operator/(const JetBase<N>& left, double right)
{
    return left / JetBase<N>(right);
}

double scalar_value(double value)
{
    return value;
}

template <std::size_t N>
double scalar_value(const JetBase<N>& value)
{
    return value.value;
}

double scalar_sqrt(double value)
{
    return std::sqrt(value);
}

template <std::size_t N>
JetBase<N> scalar_sqrt(const JetBase<N>& value)
{
    const double root = std::sqrt(value.value);
    JetBase<N> result(root);
    const double scale = 0.5 / root;
    for (std::size_t index = 0; index < N; ++index) {
        result.derivative[index] = scale * value.derivative[index];
    }
    return result;
}

double scalar_sin(double value)
{
    return std::sin(value);
}

template <std::size_t N>
JetBase<N> scalar_sin(const JetBase<N>& value)
{
    JetBase<N> result(std::sin(value.value));
    const double scale = std::cos(value.value);
    for (std::size_t index = 0; index < N; ++index) {
        result.derivative[index] = scale * value.derivative[index];
    }
    return result;
}

double scalar_cos(double value)
{
    return std::cos(value);
}

template <std::size_t N>
JetBase<N> scalar_cos(const JetBase<N>& value)
{
    JetBase<N> result(std::cos(value.value));
    const double scale = -std::sin(value.value);
    for (std::size_t index = 0; index < N; ++index) {
        result.derivative[index] = scale * value.derivative[index];
    }
    return result;
}

double scalar_pow(double value, double power)
{
    if (power == 0.0) return 1.0;
    if (power == 1.0) return value;
    if (power == 2.0) return value * value;
    if (power == 0.5) return std::sqrt(value);
    if (power == 1.5) return value * std::sqrt(value);
    if (power == 2.5) return value * value * std::sqrt(value);
    if (power == 0.25) return std::sqrt(std::sqrt(value));
    if (power == 1.25) return value * std::sqrt(std::sqrt(value));
    if (power == 2.25) {
        return value * value * std::sqrt(std::sqrt(value));
    }
    return std::pow(value, power);
}

template <std::size_t N>
JetBase<N> scalar_pow(const JetBase<N>& value, double power)
{
    const double powered = scalar_pow(value.value, power);
    JetBase<N> result(powered);
    const double scale =
        power == 0.0 ? 0.0 : power * powered / value.value;
    for (std::size_t index = 0; index < N; ++index) {
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
struct TripleLensConstants {
    std::array<Scalar, 3> lens_x{};
    std::array<Scalar, 3> lens_y{};
    std::array<Scalar, 3> mass{};
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

constexpr int moment_count(MomentMode mode)
{
    return static_cast<int>(mode);
}

template <bool ComputeLaplacian = true, typename Scalar>
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

    Scalar laplacian = 0.0;
    if constexpr (ComputeLaplacian) {
        laplacian = -2.0 * inverse_source_radius_squared
            * (du_dx * du_dx + du_dy * du_dy
               + dv_dx * dv_dx + dv_dy * dv_dy);
    }
    return {
        1.0 - (residual_x * residual_x + residual_y * residual_y)
                  * inverse_source_radius_squared,
        -2.0 * inverse_source_radius_squared
            * (residual_x * du_dx + residual_y * dv_dx),
        -2.0 * inverse_source_radius_squared
            * (residual_x * du_dy + residual_y * dv_dy),
        laplacian,
    };
}

template <typename Scalar>
TripleLensConstants<Scalar> make_triple_lens_constants(
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& tertiary_mass_ratio,
    const Scalar& tertiary_separation,
    const Scalar& tertiary_angle,
    std::int64_t convention)
{
    const Scalar total = 1.0 + mass_ratio + tertiary_mass_ratio;
    const Scalar mass_1 = 1.0 / total;
    const Scalar mass_2 = mass_ratio / total;
    const Scalar mass_3 = tertiary_mass_ratio / total;
    TripleLensConstants<Scalar> result;
    result.mass = {mass_1, mass_2, mass_3};
    if (convention == 0) {
        const Scalar group_mass = mass_2 + mass_3;
        result.lens_x[0] = -group_mass * separation;
        result.lens_y[0] = 0.0;
        const Scalar group_x = mass_1 * separation;
        const Scalar delta_x =
            tertiary_separation * scalar_cos(tertiary_angle);
        const Scalar delta_y =
            tertiary_separation * scalar_sin(tertiary_angle);
        result.lens_x[1] =
            group_x + mass_3 / group_mass * delta_x;
        result.lens_y[1] = mass_3 / group_mass * delta_y;
        result.lens_x[2] =
            group_x - mass_2 / group_mass * delta_x;
        result.lens_y[2] = -mass_2 / group_mass * delta_y;
    } else {
        result.lens_x[0] =
            mass_ratio * separation / (1.0 + mass_ratio);
        result.lens_y[0] = 0.0;
        result.lens_x[1] = -separation / (1.0 + mass_ratio);
        result.lens_y[1] = 0.0;
        result.lens_x[2] = result.lens_x[0]
            - tertiary_separation * scalar_cos(tertiary_angle);
        result.lens_y[2] =
            -tertiary_separation * scalar_sin(tertiary_angle);
    }
    return result;
}

template <bool ComputeLaplacian = true, typename Scalar>
PhiDerivatives<Scalar> triple_phi_derivatives(
    double image_x,
    double image_y,
    const Scalar& source_x,
    const Scalar& source_y,
    const TripleLensConstants<Scalar>& lens,
    const Scalar& inverse_source_radius_squared)
{
    Scalar mapped_x = image_x;
    Scalar mapped_y = image_y;
    Scalar shear_real = 0.0;
    Scalar shear_cross = 0.0;
    for (std::size_t lens_index = 0; lens_index < 3; ++lens_index) {
        const Scalar dx = image_x - lens.lens_x[lens_index];
        const Scalar dy = image_y - lens.lens_y[lens_index];
        const Scalar radius_squared = dx * dx + dy * dy;
        const Scalar inverse_radius_squared = 1.0 / radius_squared;
        const Scalar inverse_radius_fourth =
            inverse_radius_squared * inverse_radius_squared;
        mapped_x =
            mapped_x - lens.mass[lens_index] * dx * inverse_radius_squared;
        mapped_y =
            mapped_y - lens.mass[lens_index] * dy * inverse_radius_squared;
        shear_real += lens.mass[lens_index]
            * (dx * dx - dy * dy) * inverse_radius_fourth;
        shear_cross += 2.0 * lens.mass[lens_index]
            * dx * dy * inverse_radius_fourth;
    }
    const Scalar du_dx = 1.0 + shear_real;
    const Scalar du_dy = shear_cross;
    const Scalar dv_dx = shear_cross;
    const Scalar dv_dy = 1.0 - shear_real;
    const Scalar residual_x = mapped_x - source_x;
    const Scalar residual_y = mapped_y - source_y;
    Scalar laplacian = 0.0;
    if constexpr (ComputeLaplacian) {
        laplacian = -2.0 * inverse_source_radius_squared
            * (du_dx * du_dx + du_dy * du_dy
               + dv_dx * dv_dx + dv_dy * dv_dy);
    }
    return {
        1.0 - (residual_x * residual_x + residual_y * residual_y)
                  * inverse_source_radius_squared,
        -2.0 * inverse_source_radius_squared
            * (residual_x * du_dx + residual_y * dv_dx),
        -2.0 * inverse_source_radius_squared
            * (residual_x * du_dy + residual_y * dv_dy),
        laplacian,
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

template <MomentMode Mode, typename Scalar>
void add_affine_moments(
    std::array<Scalar, 3>& result,
    const PhiDerivatives<Scalar>& values,
    double cell_size)
{
    const Scalar delta_x = values.gradient_x * cell_size;
    const Scalar delta_y = values.gradient_y * cell_size;
    const Scalar lower_left = values.phi - 0.5 * (delta_x + delta_y);
    const double area = cell_size * cell_size;
    constexpr std::array<double, 3> powers{0.0, 0.5, 0.25};
    for (int index = 0; index < moment_count(Mode); ++index) {
        result[index] += area * affine_unit_square_moment(
            lower_left, delta_x, delta_y, powers[index]);
    }
}

template <MomentMode Mode, typename Scalar>
void add_interior_moments(
    std::array<Scalar, 3>& result,
    const PhiDerivatives<Scalar>& values,
    double cell_size)
{
    const double area = cell_size * cell_size;
    result[0] += area;
    if constexpr (Mode != MomentMode::uniform) {
        const Scalar sqrt_phi = scalar_sqrt(values.phi);
        const Scalar delta_squared =
            cell_size * cell_size
            * (values.gradient_x * values.gradient_x
               + values.gradient_y * values.gradient_y);
        const Scalar laplacian_term =
            cell_size * cell_size * values.laplacian;
        result[1] += area * (
            sqrt_phi + laplacian_term / (48.0 * sqrt_phi)
            - delta_squared / (96.0 * values.phi * sqrt_phi));
        if constexpr (Mode == MomentMode::two_coefficient) {
            const Scalar fourth_root = scalar_sqrt(sqrt_phi);
            result[2] += area * (
                fourth_root
                + laplacian_term / (96.0 * sqrt_phi * fourth_root)
                - delta_squared
                    / (128.0 * values.phi * sqrt_phi * fourth_root));
        }
    }
}

template <MomentMode Mode, int BoundarySubdivision, typename Scalar>
KernelResult<Scalar> fixed_support_kernel_for_mode(
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
    int tile_size)
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
    const LensConstants<double> classification_lens{
        scalar_value(lens.lens_1_x),
        scalar_value(lens.lens_2_x),
        scalar_value(lens.mass_1),
        scalar_value(lens.mass_2),
    };
    const double classification_inverse_source_radius_squared =
        scalar_value(inverse_source_radius_squared);
    const double subcell_size = cell_size / BoundarySubdivision;
    std::array<double, BoundarySubdivision> subcell_offsets{};
    for (int subcell = 0; subcell < BoundarySubdivision; ++subcell) {
        subcell_offsets[subcell] =
            ((static_cast<double>(subcell) + 0.5) / BoundarySubdivision
             - 0.5)
            * cell_size;
    }
    std::vector<double> classification_phi(
        static_cast<std::size_t>(tile_size));
    std::vector<double> classification_gradient_x(
        static_cast<std::size_t>(tile_size));
    std::vector<double> classification_gradient_y(
        static_cast<std::size_t>(tile_size));
    std::vector<double> classification_laplacian(
        static_cast<std::size_t>(tile_size));

    for (std::int64_t tile = 0; tile < tile_count; ++tile) {
        if (mask != nullptr && !mask[tile]) continue;
        for (int iy = 0; iy < tile_size; ++iy) {
            const double image_y =
                origins[2 * tile + 1]
                + (static_cast<double>(iy) + 0.5) * cell_size;
#ifdef LCBININT_HAS_OPENMP
#pragma omp simd
#endif
            for (int ix = 0; ix < tile_size; ++ix) {
                const double image_x =
                    origins[2 * tile]
                    + (static_cast<double>(ix) + 0.5) * cell_size;
                const auto classification = phi_derivatives<
                    Mode != MomentMode::uniform>(
                    image_x, image_y, scalar_value(source_x),
                    scalar_value(source_y), classification_lens,
                    classification_inverse_source_radius_squared);
                classification_phi[static_cast<std::size_t>(ix)] =
                    classification.phi;
                classification_gradient_x[static_cast<std::size_t>(ix)] =
                    classification.gradient_x;
                classification_gradient_y[static_cast<std::size_t>(ix)] =
                    classification.gradient_y;
                classification_laplacian[static_cast<std::size_t>(ix)] =
                    classification.laplacian;
            }
            for (int ix = 0; ix < tile_size; ++ix) {
                const double image_x =
                    origins[2 * tile]
                    + (static_cast<double>(ix) + 0.5) * cell_size;
                const PhiDerivatives<double> classification{
                    classification_phi[static_cast<std::size_t>(ix)],
                    classification_gradient_x[static_cast<std::size_t>(ix)],
                    classification_gradient_y[static_cast<std::size_t>(ix)],
                    classification_laplacian[static_cast<std::size_t>(ix)],
                };
                const double half_delta_x =
                    0.5 * classification.gradient_x * cell_size;
                const double half_delta_y =
                    0.5 * classification.gradient_y * cell_size;
                const double extent =
                    std::abs(half_delta_x) + std::abs(half_delta_y);
                const double phi_value = classification.phi;
                const bool fully_inside = phi_value - extent > 0.0;
                const bool fully_outside = phi_value + extent <= 0.0;
                const bool geometric_boundary = !(fully_inside || fully_outside);
                bool detailed = geometric_boundary;
                if constexpr (Mode == MomentMode::two_coefficient) {
                    if (scalar_value(limb_d) != 0.0 && fully_inside) {
                        const double relative_variation =
                            (extent
                             + 0.125 * std::abs(classification.laplacian)
                                          * cell_size * cell_size)
                            / std::max(phi_value, 1.0e-30);
                        detailed = relative_variation > 0.2;
                    }
                }

                if (detailed) {
                    ++result.boundary_cells;
                    ++result.active_cells;
                    for (int sy = 0; sy < BoundarySubdivision; ++sy) {
                        const double offset_y = subcell_offsets[sy];
                        for (int sx = 0; sx < BoundarySubdivision; ++sx) {
                            const double offset_x = subcell_offsets[sx];
                            const auto sub_values = phi_derivatives<false>(
                                image_x + offset_x, image_y + offset_y,
                                source_x, source_y, lens,
                                inverse_source_radius_squared);
                            add_affine_moments<Mode>(
                                result.moments, sub_values, subcell_size);
                        }
                    }
                } else if (fully_inside) {
                    ++result.active_cells;
                    if constexpr (Mode == MomentMode::uniform) {
                        result.moments[0] += cell_size * cell_size;
                    } else if constexpr (std::is_same_v<Scalar, double>) {
                        add_interior_moments<Mode>(
                            result.moments, classification, cell_size);
                    } else {
                        const auto values = phi_derivatives(
                            image_x, image_y, source_x, source_y, lens,
                            inverse_source_radius_squared);
                        add_interior_moments<Mode>(
                            result.moments, values, cell_size);
                    }
                }
            }
        }
    }
    return result;
}

template <MomentMode Mode, typename Scalar>
KernelResult<Scalar> dispatch_fixed_support_kernel_for_mode(
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
    int boundary_subdivision)
{
    if (boundary_subdivision == 1) {
        return fixed_support_kernel_for_mode<Mode, 1>(
            origins, mask, tile_count, cell_size, source_x, source_y,
            separation, mass_ratio, source_radius, limb_d, tile_size);
    }
    if (boundary_subdivision == 2) {
        return fixed_support_kernel_for_mode<Mode, 2>(
            origins, mask, tile_count, cell_size, source_x, source_y,
            separation, mass_ratio, source_radius, limb_d, tile_size);
    }
    if (boundary_subdivision == 3) {
        return fixed_support_kernel_for_mode<Mode, 3>(
            origins, mask, tile_count, cell_size, source_x, source_y,
            separation, mass_ratio, source_radius, limb_d, tile_size);
    }
    if (boundary_subdivision == 8) {
        return fixed_support_kernel_for_mode<Mode, 8>(
            origins, mask, tile_count, cell_size, source_x, source_y,
            separation, mass_ratio, source_radius, limb_d, tile_size);
    }
    return fixed_support_kernel_for_mode<Mode, 4>(
        origins, mask, tile_count, cell_size, source_x, source_y,
        separation, mass_ratio, source_radius, limb_d, tile_size);
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
    if (mode == MomentMode::uniform) {
        return dispatch_fixed_support_kernel_for_mode<MomentMode::uniform>(
            origins, mask, tile_count, cell_size, source_x, source_y,
            separation, mass_ratio, source_radius, limb_d, tile_size,
            boundary_subdivision);
    }
    if (mode == MomentMode::linear) {
        return dispatch_fixed_support_kernel_for_mode<MomentMode::linear>(
            origins, mask, tile_count, cell_size, source_x, source_y,
            separation, mass_ratio, source_radius, limb_d, tile_size,
            boundary_subdivision);
    }
    return dispatch_fixed_support_kernel_for_mode<
        MomentMode::two_coefficient>(
        origins, mask, tile_count, cell_size, source_x, source_y,
        separation, mass_ratio, source_radius, limb_d, tile_size,
        boundary_subdivision);
}

template <MomentMode Mode, int BoundarySubdivision, typename Scalar>
KernelResult<Scalar> triple_fixed_support_kernel_for_mode(
    const double* origins,
    std::int64_t tile_count,
    double cell_size,
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& tertiary_mass_ratio,
    const Scalar& tertiary_separation,
    const Scalar& tertiary_angle,
    const Scalar& source_radius,
    const Scalar& limb_d,
    int tile_size,
    std::int64_t convention)
{
    KernelResult<Scalar> result;
    const Scalar inverse_source_radius_squared =
        1.0 / (source_radius * source_radius);
    const auto lens = make_triple_lens_constants(
        separation, mass_ratio, tertiary_mass_ratio,
        tertiary_separation, tertiary_angle, convention);
    TripleLensConstants<double> classification_lens;
    for (std::size_t index = 0; index < 3; ++index) {
        classification_lens.lens_x[index] =
            scalar_value(lens.lens_x[index]);
        classification_lens.lens_y[index] =
            scalar_value(lens.lens_y[index]);
        classification_lens.mass[index] =
            scalar_value(lens.mass[index]);
    }
    const double classification_inverse_radius_squared =
        scalar_value(inverse_source_radius_squared);
    const double subcell_size = cell_size / BoundarySubdivision;
    std::array<double, BoundarySubdivision> subcell_offsets{};
    for (int subcell = 0; subcell < BoundarySubdivision; ++subcell) {
        subcell_offsets[subcell] =
            ((static_cast<double>(subcell) + 0.5) / BoundarySubdivision
             - 0.5) * cell_size;
    }
    std::vector<double> classification_phi(
        static_cast<std::size_t>(tile_size));
    std::vector<double> classification_gradient_x(
        static_cast<std::size_t>(tile_size));
    std::vector<double> classification_gradient_y(
        static_cast<std::size_t>(tile_size));
    std::vector<double> classification_laplacian(
        static_cast<std::size_t>(tile_size));

    for (std::int64_t tile = 0; tile < tile_count; ++tile) {
        for (int iy = 0; iy < tile_size; ++iy) {
            const double image_y =
                origins[2 * tile + 1]
                + (static_cast<double>(iy) + 0.5) * cell_size;
#ifdef LCBININT_HAS_OPENMP
#pragma omp simd
#endif
            for (int ix = 0; ix < tile_size; ++ix) {
                const double image_x =
                    origins[2 * tile]
                    + (static_cast<double>(ix) + 0.5) * cell_size;
                const auto classification = triple_phi_derivatives<
                    Mode != MomentMode::uniform>(
                    image_x, image_y, scalar_value(source_x),
                    scalar_value(source_y), classification_lens,
                    classification_inverse_radius_squared);
                classification_phi[static_cast<std::size_t>(ix)] =
                    classification.phi;
                classification_gradient_x[static_cast<std::size_t>(ix)] =
                    classification.gradient_x;
                classification_gradient_y[static_cast<std::size_t>(ix)] =
                    classification.gradient_y;
                classification_laplacian[static_cast<std::size_t>(ix)] =
                    classification.laplacian;
            }
            for (int ix = 0; ix < tile_size; ++ix) {
                const double image_x =
                    origins[2 * tile]
                    + (static_cast<double>(ix) + 0.5) * cell_size;
                const PhiDerivatives<double> classification{
                    classification_phi[static_cast<std::size_t>(ix)],
                    classification_gradient_x[static_cast<std::size_t>(ix)],
                    classification_gradient_y[static_cast<std::size_t>(ix)],
                    classification_laplacian[static_cast<std::size_t>(ix)],
                };
                const double extent = 0.5 * cell_size * (
                    std::abs(classification.gradient_x)
                    + std::abs(classification.gradient_y));
                const bool fully_inside =
                    classification.phi - extent > 0.0;
                const bool fully_outside =
                    classification.phi + extent <= 0.0;
                bool detailed = !(fully_inside || fully_outside);
                if constexpr (Mode == MomentMode::two_coefficient) {
                    if (scalar_value(limb_d) != 0.0 && fully_inside) {
                        const double relative_variation =
                            (extent
                             + 0.125 * std::abs(classification.laplacian)
                                 * cell_size * cell_size)
                            / std::max(classification.phi, 1.0e-30);
                        detailed = relative_variation > 0.2;
                    }
                }
                if (detailed) {
                    ++result.boundary_cells;
                    ++result.active_cells;
                    for (int sy = 0; sy < BoundarySubdivision; ++sy) {
                        for (int sx = 0; sx < BoundarySubdivision; ++sx) {
                            const auto values = triple_phi_derivatives<false>(
                                image_x + subcell_offsets[sx],
                                image_y + subcell_offsets[sy],
                                source_x, source_y, lens,
                                inverse_source_radius_squared);
                            add_affine_moments<Mode>(
                                result.moments, values, subcell_size);
                        }
                    }
                } else if (fully_inside) {
                    ++result.active_cells;
                    if constexpr (Mode == MomentMode::uniform) {
                        result.moments[0] += cell_size * cell_size;
                    } else if constexpr (std::is_same_v<Scalar, double>) {
                        add_interior_moments<Mode>(
                            result.moments, classification, cell_size);
                    } else {
                        const auto values = triple_phi_derivatives(
                            image_x, image_y, source_x, source_y, lens,
                            inverse_source_radius_squared);
                        add_interior_moments<Mode>(
                            result.moments, values, cell_size);
                    }
                }
            }
        }
    }
    return result;
}

template <MomentMode Mode, typename Scalar>
KernelResult<Scalar> dispatch_triple_fixed_support_kernel(
    const double* origins,
    std::int64_t tile_count,
    double cell_size,
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& tertiary_mass_ratio,
    const Scalar& tertiary_separation,
    const Scalar& tertiary_angle,
    const Scalar& source_radius,
    const Scalar& limb_d,
    int tile_size,
    std::int64_t convention,
    int boundary_subdivision)
{
#define LCBININT_TRIPLE_KERNEL(N) \
    return triple_fixed_support_kernel_for_mode<Mode, N>( \
        origins, tile_count, cell_size, source_x, source_y, separation, \
        mass_ratio, tertiary_mass_ratio, tertiary_separation, \
        tertiary_angle, source_radius, limb_d, tile_size, convention)
    if (boundary_subdivision == 1) { LCBININT_TRIPLE_KERNEL(1); }
    if (boundary_subdivision == 2) { LCBININT_TRIPLE_KERNEL(2); }
    if (boundary_subdivision == 3) { LCBININT_TRIPLE_KERNEL(3); }
    if (boundary_subdivision == 8) { LCBININT_TRIPLE_KERNEL(8); }
    LCBININT_TRIPLE_KERNEL(4);
#undef LCBININT_TRIPLE_KERNEL
}

template <typename Scalar>
KernelResult<Scalar> triple_fixed_support_kernel(
    const double* origins,
    std::int64_t tile_count,
    double cell_size,
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& tertiary_mass_ratio,
    const Scalar& tertiary_separation,
    const Scalar& tertiary_angle,
    const Scalar& source_radius,
    const Scalar& limb_d,
    int tile_size,
    std::int64_t convention,
    MomentMode mode,
    int boundary_subdivision)
{
#define LCBININT_TRIPLE_MODE(MODE) \
    return dispatch_triple_fixed_support_kernel<MODE>( \
        origins, tile_count, cell_size, source_x, source_y, separation, \
        mass_ratio, tertiary_mass_ratio, tertiary_separation, \
        tertiary_angle, source_radius, limb_d, tile_size, convention, \
        boundary_subdivision)
    if (mode == MomentMode::uniform) {
        LCBININT_TRIPLE_MODE(MomentMode::uniform);
    }
    if (mode == MomentMode::linear) {
        LCBININT_TRIPLE_MODE(MomentMode::linear);
    }
    LCBININT_TRIPLE_MODE(MomentMode::two_coefficient);
#undef LCBININT_TRIPLE_MODE
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

template <typename Scalar>
std::array<double, 2> limb_coefficient_derivatives(
    const std::array<Scalar, 3>& moments,
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
    if (boundary_subdivision < 1 || boundary_subdivision > 8
        || (boundary_subdivision > 4 && boundary_subdivision != 8)) {
        throw std::invalid_argument("boundary_subdivision must be 1, 2, 3, 4, or 8");
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

struct BinaryRootContinuation {
    bool valid = false;
    double source_x = 0.0;
    double source_y = 0.0;
    double separation = 0.0;
    double mass_ratio = 0.0;
    std::array<lcbinint::Complex, binary_root_count> polynomial_roots{};
    std::array<bool, binary_root_count> physical{};
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

bool binary_polynomial_roots_are_usable(
    const std::vector<lcbinint::Complex>& coefficients,
    const std::vector<lcbinint::Complex>& roots)
{
    if (coefficients.size() != roots.size() + 1 || roots.empty()) {
        return false;
    }
    constexpr double residual_tolerance = 1.0e-8;
    for (const auto& root : roots) {
        if (!std::isfinite(root.real()) || !std::isfinite(root.imag())) {
            return false;
        }
        lcbinint::Complex value = 0.0;
        double scale = 0.0;
        const double radius = std::abs(root);
        for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
            value = value * root + *it;
            scale = scale * radius + std::abs(*it);
        }
        if (std::abs(value) > residual_tolerance * std::max(scale, 1.0)) {
            return false;
        }
    }

    std::vector<lcbinint::Complex> reconstructed(roots.size() + 1, 0.0);
    reconstructed[0] = 1.0;
    for (std::size_t index = 0; index < roots.size(); ++index) {
        for (std::size_t coefficient = index + 1; coefficient > 0; --coefficient) {
            reconstructed[coefficient] =
                reconstructed[coefficient - 1]
                - roots[index] * reconstructed[coefficient];
        }
        reconstructed[0] *= -roots[index];
    }
    const auto leading = coefficients.back();
    if (!(std::abs(leading) > 0.0) ||
        !std::isfinite(leading.real()) || !std::isfinite(leading.imag())) {
        return false;
    }
    constexpr double coefficient_tolerance = 1.0e-7;
    for (std::size_t index = 0; index < reconstructed.size(); ++index) {
        const auto expected = coefficients[index] / leading;
        if (std::abs(reconstructed[index] - expected) >
            coefficient_tolerance * std::max(std::abs(expected), 1.0)) {
            return false;
        }
    }
    return true;
}

std::vector<lcbinint::Complex> predicted_binary_roots(
    const BinaryRootContinuation& continuation,
    double source_x,
    double source_y,
    double separation,
    double mass_ratio)
{
    std::vector<lcbinint::Complex> roots(
        continuation.polynomial_roots.begin(),
        continuation.polynomial_roots.end());
    if (continuation.separation != separation ||
        continuation.mass_ratio != mass_ratio) {
        return roots;
    }
    const lcbinint::Complex delta_source {
        source_x - continuation.source_x,
        source_y - continuation.source_y,
    };
    const double source_step = std::abs(delta_source);
    if (!(source_step > 0.0) || source_step > 0.25) {
        return roots;
    }

    const double total_mass = 1.0 + mass_ratio;
    const double lens_1_x = -mass_ratio / total_mass * separation;
    const double lens_2_x = separation / total_mass;
    const double mass_1 = 1.0 / total_mass;
    const double mass_2 = mass_ratio / total_mass;
    constexpr double minimum_abs_jacobian = 1.0e-4;
    constexpr double maximum_image_step = 0.5;
    for (std::size_t index = 0; index < roots.size(); ++index) {
        if (!continuation.physical[index]) continue;
        const auto zbar = std::conj(roots[index]);
        const auto offset_1 = zbar - lens_1_x;
        const auto offset_2 = zbar - lens_2_x;
        if (std::abs(offset_1) == 0.0 || std::abs(offset_2) == 0.0) continue;
        const auto kappa =
            mass_1 / (offset_1 * offset_1)
            + mass_2 / (offset_2 * offset_2);
        const double jacobian = 1.0 - std::norm(kappa);
        if (!std::isfinite(jacobian) ||
            std::abs(jacobian) < minimum_abs_jacobian) continue;
        const auto delta_image =
            (delta_source - kappa * std::conj(delta_source)) / jacobian;
        if (!std::isfinite(delta_image.real()) ||
            !std::isfinite(delta_image.imag()) ||
            std::abs(delta_image) > maximum_image_step) continue;
        roots[index] += delta_image;
    }
    return roots;
}

BinaryRootResult solve_binary_images(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    BinaryRootContinuation* continuation = nullptr)
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
    lcbinint::math::PolynomialRootSolver solver;
    auto solved = continuation != nullptr && continuation->valid && !use_quartic
        ? solver.solve_from_roots(
            active_coefficients,
            predicted_binary_roots(
                *continuation, source_x, source_y, separation, mass_ratio),
            {true, true})
        : solver.solve(active_coefficients);
    if (solved.status != lcbinint::math::RootSolverStatus::ok ||
        !binary_polynomial_roots_are_usable(active_coefficients, solved.roots)) {
        solved = solver.solve(active_coefficients);
    }
    BinaryRootResult result;
    result.residuals.fill(std::numeric_limits<double>::infinity());
    if (solved.status != lcbinint::math::RootSolverStatus::ok) {
        if (continuation != nullptr) continuation->valid = false;
        return result;
    }
    const auto polynomial_roots = solved.roots;

    const auto retain_continuation = [&]() {
        if (continuation == nullptr || use_quartic ||
            polynomial_roots.size() != binary_root_count) return;
        continuation->valid = true;
        continuation->source_x = source_x;
        continuation->source_y = source_y;
        continuation->separation = separation;
        continuation->mass_ratio = mass_ratio;
        std::copy(
            polynomial_roots.begin(), polynomial_roots.end(),
            continuation->polynomial_roots.begin());
        continuation->physical = result.physical;
    };

    const auto canonicalise_result_order = [&]() {
        // SG does not promise a root ordering, and warm continuation can
        // therefore return the same physical image set in a different order
        // from a cold solve.  Discovery is a breadth-first traversal seeded
        // in this order, so an arbitrary root permutation changes the tile
        // summation order and can leak out as a one-ulp moment difference.
        // Put physical images first in coordinate order; this keeps the FFI
        // result and every discovery consumer independent of the SG start.
        std::array<std::size_t, binary_root_count> order{};
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(
            order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
                if (result.physical[left] != result.physical[right]) {
                    return result.physical[left] > result.physical[right];
                }
                const auto left_root = result.roots[left];
                const auto right_root = result.roots[right];
                const bool left_finite =
                    std::isfinite(left_root.real())
                    && std::isfinite(left_root.imag());
                const bool right_finite =
                    std::isfinite(right_root.real())
                    && std::isfinite(right_root.imag());
                if (left_finite != right_finite) return left_finite;
                if (!left_finite) return false;
                if (left_root.real() != right_root.real()) {
                    return left_root.real() < right_root.real();
                }
                return left_root.imag() < right_root.imag();
            });
        const auto unordered = result;
        for (std::size_t index = 0; index < binary_root_count; ++index) {
            const auto source_index = order[index];
            result.roots[index] = unordered.roots[source_index];
            result.converged[index] = unordered.converged[source_index];
            result.physical[index] = unordered.physical[source_index];
            result.residuals[index] = unordered.residuals[source_index];
        }
    };

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
            const auto next =
                root - lcbinint::Complex(delta_x, delta_y) / step_scale;
            if (next == root) break;
            root = next;
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
        result.physical[index] = false;
    }

    // Polynomial ghost roots can polish onto an already found physical image.
    // Distinct physical images really do coalesce at folds and cusps, so
    // coordinate distance alone cannot distinguish the two cases -- but the
    // images that merge at a fold or a cusp are always one positive- and one
    // negative-parity image, so two *same-parity* candidates at the same place
    // are never a physical degeneracy.  They are one image and a ghost that
    // polished onto it.  That makes distance an exact test once it is applied
    // per parity class, and it is applied in both places below, because either
    // one alone leaves the other counting one image twice.
    //
    // The rest of the classification uses the binary-lens parity invariant: a
    // five-image solution has two positive- and three negative-parity images;
    // a three-image solution has one and two.
    const double physical_tolerance = 1.0e-8 * (1.0 + std::abs(source));
    // Two polished roots this close are the same root.  The measured margin is
    // wide: the ghosts land within 1e-15 of their host, and widening this to
    // 1e-6 changes nothing over 3000 sampled positions, half of them crowded
    // onto the central caustic where the five-image solutions live.
    const double coincidence_tolerance = 1.0e-9;
    const auto same_parity_coincidence =
        [&](std::size_t first, std::size_t second, const auto& parities) {
            return parities[first] == parities[second]
                && std::abs(result.roots[first] - result.roots[second])
                    <= coincidence_tolerance
                        * (1.0 + std::abs(result.roots[first]));
        };
    std::array<bool, binary_root_count> candidate{};
    std::array<int, binary_root_count> parity{};
    std::size_t candidate_count = 0;
    int positive_count = 0;
    int negative_count = 0;
    for (std::size_t index = 0; index < binary_root_count; ++index) {
        candidate[index] =
            result.converged[index]
            && std::isfinite(result.residuals[index])
            && result.residuals[index] <= physical_tolerance;
        if (!candidate[index]) continue;
        ++candidate_count;
        const auto mapped = binary_lens_map_at_root(
            result.roots[index], separation, mass_ratio);
        const double determinant =
            mapped.du_dx * mapped.dv_dy - mapped.du_dy * mapped.dv_dx;
        parity[index] = determinant > 0.0 ? 1 : -1;
        positive_count += static_cast<int>(parity[index] > 0);
        negative_count += static_cast<int>(parity[index] < 0);
    }

    // Two coincident ghosts can forge the five-image parity signature: a
    // three-image solution whose faint negative and bright positive image each
    // attracted one ghost presents as two positive and three negative
    // candidates, and the count check alone accepts all five.
    bool forged_five = false;
    for (std::size_t first = 0; first < binary_root_count && !forged_five; ++first) {
        if (!candidate[first]) continue;
        for (std::size_t second = first + 1; second < binary_root_count; ++second) {
            if (!candidate[second]) continue;
            if (same_parity_coincidence(first, second, parity)) {
                forged_five = true;
                break;
            }
        }
    }

    if (
        candidate_count == 5
        && positive_count == 2
        && negative_count == 3
        && !forged_five) {
        result.physical = candidate;
        retain_continuation();
        canonicalise_result_order();
        return result;
    }

    // Outside a caustic, choose the unique parity-compatible three-image
    // subset.  Residual selects the positive image.  For the two negative
    // images, residual divided by their separation ranks the pairs, and a
    // coincident pair is rejected outright: the ratio alone does not reject it,
    // because two ghosts that polished onto the same image both reach residual
    // exactly zero, and 1e-30 over the 1e-15 separation floor is then a finite
    // score that beats the true pair's.  That is one image counted twice, and
    // it was wrong on 5.7% of sampled point-source epochs, by up to 94%.
    std::size_t positive_index = binary_root_count;
    double positive_residual = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < binary_root_count; ++index) {
        if (
            candidate[index]
            && parity[index] > 0
            && result.residuals[index] < positive_residual) {
            positive_index = index;
            positive_residual = result.residuals[index];
        }
    }
    std::size_t negative_1 = binary_root_count;
    std::size_t negative_2 = binary_root_count;
    double negative_score = std::numeric_limits<double>::infinity();
    for (std::size_t first = 0; first < binary_root_count; ++first) {
        if (!candidate[first] || parity[first] >= 0) continue;
        for (std::size_t second = first + 1; second < binary_root_count; ++second) {
            if (!candidate[second] || parity[second] >= 0) continue;
            if (same_parity_coincidence(first, second, parity)) continue;
            const double separation_between_roots = std::abs(
                result.roots[first] - result.roots[second]);
            const double score =
                (result.residuals[first] + result.residuals[second]
                 + 1.0e-30)
                / std::max(separation_between_roots, 1.0e-15);
            if (score < negative_score) {
                negative_score = score;
                negative_1 = first;
                negative_2 = second;
            }
        }
    }
    if (
        positive_index < binary_root_count
        && negative_1 < binary_root_count
        && negative_2 < binary_root_count) {
        result.physical[positive_index] = true;
        result.physical[negative_1] = true;
        result.physical[negative_2] = true;
    }
    retain_continuation();
    canonicalise_result_order();
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

constexpr std::int64_t triple_root_count = 10;

ffi::Error triple_image_roots_ffi_impl(
    std::int64_t convention,
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::ResultBufferR2<ffi::F64> roots,
    ffi::ResultBufferR1<ffi::PRED> physical,
    ffi::ResultBufferR1<ffi::F64> residuals,
    ffi::ResultBufferR1<ffi::PRED> converged)
{
    if (
        roots->dimensions()[0] != triple_root_count
        || roots->dimensions()[1] != 2
        || physical->dimensions()[0] != triple_root_count
        || residuals->dimensions()[0] != triple_root_count
        || converged->dimensions()[0] != triple_root_count
        || (convention != 0 && convention != 1)) {
        return ffi::Error::InvalidArgument(
            "invalid triple-root output shape or convention");
    }
    const double s = *separation.typed_data();
    const double q = *mass_ratio.typed_data();
    const double q2 = *tertiary_mass_ratio.typed_data();
    const double s2 = *tertiary_separation.typed_data();
    const double angle = *tertiary_angle.typed_data();
    const auto geometry = convention == 0
        ? lcbinint::model::make_triple_lens_geometry(s, q, q2, s2, angle)
        : lcbinint::model::make_triple_lens_geometry_vbm(s, q, s2, angle, q2);
    const lcbinint::magnification::PointSourceMagnifier magnifier;
    const auto candidates = magnifier.triple_image_candidates(
        geometry, {*source_x.typed_data(), *source_y.typed_data()});
    auto* output_roots = roots->typed_data();
    auto* output_physical = physical->typed_data();
    auto* output_residuals = residuals->typed_data();
    auto* output_converged = converged->typed_data();
    std::fill(output_roots, output_roots + 2 * triple_root_count, 0.0);
    std::fill(output_physical, output_physical + triple_root_count, false);
    std::fill(
        output_residuals, output_residuals + triple_root_count,
        std::numeric_limits<double>::infinity());
    std::fill(output_converged, output_converged + triple_root_count, false);
    const std::size_t count = std::min<std::size_t>(
        candidates.size(), static_cast<std::size_t>(triple_root_count));
    for (std::size_t index = 0; index < count; ++index) {
        const auto& candidate = candidates[index];
        output_roots[2 * index] = candidate.position.x;
        output_roots[2 * index + 1] = candidate.position.y;
        output_physical[index] = candidate.physical;
        output_residuals[index] = candidate.residual;
        output_converged[index] =
            std::isfinite(candidate.residual) && candidate.residual <= 1.0e-7;
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    triple_image_roots_ffi_handler,
    triple_image_roots_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("convention")
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
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

template <typename Scalar>
Scalar binary_jacobian_determinant(
    const Scalar& image_x,
    const Scalar& image_y,
    const Scalar& separation,
    const Scalar& mass_ratio)
{
    const Scalar total_mass = 1.0 + mass_ratio;
    const Scalar lens_1_x = -mass_ratio / total_mass * separation;
    const Scalar lens_2_x = separation / total_mass;
    const Scalar mass_1 = 1.0 / total_mass;
    const Scalar mass_2 = mass_ratio / total_mass;
    const Scalar dx_1 = image_x - lens_1_x;
    const Scalar dx_2 = image_x - lens_2_x;
    const Scalar radius_1_squared = dx_1 * dx_1 + image_y * image_y;
    const Scalar radius_2_squared = dx_2 * dx_2 + image_y * image_y;
    const Scalar inverse_1 = 1.0 / radius_1_squared;
    const Scalar inverse_2 = 1.0 / radius_2_squared;
    const Scalar shear_real =
        mass_1 * (dx_1 * dx_1 - image_y * image_y)
            * inverse_1 * inverse_1
        + mass_2 * (dx_2 * dx_2 - image_y * image_y)
            * inverse_2 * inverse_2;
    const Scalar shear_cross =
        2.0 * image_y
        * (
            mass_1 * dx_1 * inverse_1 * inverse_1
            + mass_2 * dx_2 * inverse_2 * inverse_2);
    return 1.0 - shear_real * shear_real - shear_cross * shear_cross;
}

Jet binary_point_magnification_jet(
    const Jet& source_x,
    const Jet& source_y,
    const Jet& separation,
    const Jet& mass_ratio,
    std::int32_t& image_count,
    bool& root_failure,
    BinaryRootContinuation* continuation = nullptr)
{
    const double x = source_x.value;
    const double y = source_y.value;
    const double s = separation.value;
    const double q = mass_ratio.value;
    const auto images = solve_binary_images(x, y, s, q, continuation);
    std::int32_t physical_count = 0;
    bool all_converged = true;
    Jet magnification(0.0);
    for (std::size_t root = 0; root < binary_root_count; ++root) {
        physical_count += static_cast<std::int32_t>(images.physical[root]);
        all_converged = all_converged && images.converged[root];
        if (!images.physical[root]) continue;
        const auto mapped =
            binary_lens_map_at_root(images.roots[root], s, q);
        const double determinant =
            mapped.du_dx * mapped.dv_dy - mapped.du_dy * mapped.dv_dx;
        if (!(std::abs(determinant) > 1.0e-12)) continue;
        const auto parameter_map = binary_lens_map_at_fixed_root(
            images.roots[root], separation, mass_ratio);
        Jet image_x(images.roots[root].real());
        Jet image_y(images.roots[root].imag());
        for (
            std::size_t parameter = 0;
            parameter < kernel_derivative_count;
            ++parameter) {
            const double rhs_x =
                source_x.derivative[parameter]
                - parameter_map[0].derivative[parameter];
            const double rhs_y =
                source_y.derivative[parameter]
                - parameter_map[1].derivative[parameter];
            image_x.derivative[parameter] =
                (mapped.dv_dy * rhs_x - mapped.du_dy * rhs_y)
                / determinant;
            image_y.derivative[parameter] =
                (-mapped.dv_dx * rhs_x + mapped.du_dx * rhs_y)
                / determinant;
        }
        const Jet active_determinant = binary_jacobian_determinant(
            image_x, image_y, separation, mass_ratio);
        magnification +=
            1.0
            / (
                determinant >= 0.0
                    ? active_determinant
                    : -active_determinant);
    }
    root_failure =
        root_failure
        || !all_converged
        || (physical_count != 3 && physical_count != 5);
    image_count = physical_count;
    return magnification;
}

struct HexadecapoleKernelResult {
    Jet magnification;
    Jet point_magnification;
    Jet quadrupole_correction;
    Jet hexadecapole_correction;
    bool topology_stable = false;
    bool root_failure = false;
    double limb_c_derivative = 0.0;
    double limb_d_derivative = 0.0;
    double quadrupole_limb_c_derivative = 0.0;
    double quadrupole_limb_d_derivative = 0.0;
    double hexadecapole_limb_c_derivative = 0.0;
    double hexadecapole_limb_d_derivative = 0.0;
};

HexadecapoleKernelResult hexadecapole_kernel(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    double limb_c,
    double limb_d)
{
    constexpr std::array<double, 13> unit_x{
        0.0, 1.0, 0.0, -1.0, 0.0,
        0.5, 0.0, -0.5, 0.0,
        0.7071067811865475244, -0.7071067811865475244,
        -0.7071067811865475244, 0.7071067811865475244};
    constexpr std::array<double, 13> unit_y{
        0.0, 0.0, 1.0, 0.0, -1.0,
        0.0, 0.5, 0.0, -0.5,
        0.7071067811865475244, 0.7071067811865475244,
        -0.7071067811865475244, -0.7071067811865475244};
    const Jet centre_x = Jet::variable(source_x, 0);
    const Jet centre_y = Jet::variable(source_y, 1);
    const Jet active_separation = Jet::variable(separation, 2);
    const Jet active_mass_ratio = Jet::variable(mass_ratio, 3);
    const Jet active_radius = Jet::variable(source_radius, 4);
    std::array<Jet, 13> samples;
    std::array<std::int32_t, 13> image_counts{};
    HexadecapoleKernelResult result;
    BinaryRootContinuation continuation;
    for (std::size_t sample = 0; sample < samples.size(); ++sample) {
        samples[sample] = binary_point_magnification_jet(
            centre_x + unit_x[sample] * active_radius,
            centre_y + unit_y[sample] * active_radius,
            active_separation, active_mass_ratio, image_counts[sample],
            result.root_failure, &continuation);
    }
    const Jet a0 = samples[0];
    const Jet a1_plus =
        0.25 * (samples[1] + samples[2] + samples[3] + samples[4]) - a0;
    const Jet a2_plus =
        0.25 * (samples[5] + samples[6] + samples[7] + samples[8]) - a0;
    const Jet a1_cross =
        0.25 * (samples[9] + samples[10] + samples[11] + samples[12]) - a0;
    const Jet a2rho2 = (16.0 * a2_plus - a1_plus) / 3.0;
    const Jet a4rho4 = 0.5 * (a1_plus + a1_cross) - a2rho2;
    const double denominator = 15.0 - 5.0 * limb_c - 3.0 * limb_d;
    const double gamma = denominator != 0.0
        ? 10.0 * limb_c / denominator
        : 0.0;
    const double lambda = denominator != 0.0
        ? 12.0 * limb_d / denominator
        : 0.0;
    result.point_magnification = a0;
    result.quadrupole_correction =
        0.5 * a2rho2 * (1.0 - 0.2 * gamma - lambda / 9.0);
    result.hexadecapole_correction =
        a4rho4 / 3.0
        * (1.0 - 11.0 * gamma / 35.0 - 7.0 * lambda / 39.0);
    result.magnification =
        a0 + result.quadrupole_correction
        + result.hexadecapole_correction;
    result.topology_stable =
        !result.root_failure
        && std::all_of(
            image_counts.begin() + 1,
            image_counts.end(),
            [&](std::int32_t count) {
                return count == image_counts[0];
            });
    if (denominator != 0.0) {
        const double inverse_denominator_squared =
            1.0 / (denominator * denominator);
        const double gamma_c =
            10.0 / denominator
            + 50.0 * limb_c * inverse_denominator_squared;
        const double gamma_d =
            30.0 * limb_c * inverse_denominator_squared;
        const double lambda_c =
            60.0 * limb_d * inverse_denominator_squared;
        const double lambda_d =
            12.0 / denominator
            + 36.0 * limb_d * inverse_denominator_squared;
        result.quadrupole_limb_c_derivative =
            -0.5 * a2rho2.value
            * (0.2 * gamma_c + lambda_c / 9.0);
        result.quadrupole_limb_d_derivative =
            -0.5 * a2rho2.value
            * (0.2 * gamma_d + lambda_d / 9.0);
        result.hexadecapole_limb_c_derivative =
            -a4rho4.value / 3.0
            * (11.0 * gamma_c / 35.0 + 7.0 * lambda_c / 39.0);
        result.hexadecapole_limb_d_derivative =
            -a4rho4.value / 3.0
            * (11.0 * gamma_d / 35.0 + 7.0 * lambda_d / 39.0);
        result.limb_c_derivative =
            result.quadrupole_limb_c_derivative
            + result.hexadecapole_limb_c_derivative;
        result.limb_d_derivative =
            result.quadrupole_limb_d_derivative
            + result.hexadecapole_limb_d_derivative;
    }
    return result;
}

template <typename Scalar>
struct TripleMapValues {
    Scalar source_x;
    Scalar source_y;
    Scalar du_dx;
    Scalar du_dy;
    Scalar dv_dx;
    Scalar dv_dy;
    Scalar determinant;
};

template <typename Scalar>
struct TripleMappedSource {
    Scalar source_x;
    Scalar source_y;
};

// Support discovery and radial boundary probes need only the lens-mapped
// source coordinate.  Reusing `triple_map_values` there also accumulated the
// full shear/Jacobian at every flood cell and then discarded it.  Keep this
// small mapper separate so the differentiable integration path can still ask
// for the complete map when its derivatives are actually consumed.
template <typename Scalar>
TripleMappedSource<Scalar> triple_map_source(
    const Scalar& image_x,
    const Scalar& image_y,
    const TripleLensConstants<Scalar>& lens)
{
    Scalar source_x = image_x;
    Scalar source_y = image_y;
    for (std::size_t index = 0; index < 3; ++index) {
        const Scalar dx = image_x - lens.lens_x[index];
        const Scalar dy = image_y - lens.lens_y[index];
        const Scalar inverse_radius_squared = 1.0 / (dx * dx + dy * dy);
        source_x =
            source_x - lens.mass[index] * dx * inverse_radius_squared;
        source_y =
            source_y - lens.mass[index] * dy * inverse_radius_squared;
    }
    return {source_x, source_y};
}

template <typename Scalar>
TripleMapValues<Scalar> triple_map_values(
    const Scalar& image_x,
    const Scalar& image_y,
    const TripleLensConstants<Scalar>& lens)
{
    Scalar source_x = image_x;
    Scalar source_y = image_y;
    Scalar shear_real = 0.0;
    Scalar shear_cross = 0.0;
    for (std::size_t index = 0; index < 3; ++index) {
        const Scalar dx = image_x - lens.lens_x[index];
        const Scalar dy = image_y - lens.lens_y[index];
        const Scalar radius_squared = dx * dx + dy * dy;
        const Scalar inverse_radius_squared = 1.0 / radius_squared;
        const Scalar inverse_radius_fourth =
            inverse_radius_squared * inverse_radius_squared;
        source_x =
            source_x - lens.mass[index] * dx * inverse_radius_squared;
        source_y =
            source_y - lens.mass[index] * dy * inverse_radius_squared;
        shear_real += lens.mass[index]
            * (dx * dx - dy * dy) * inverse_radius_fourth;
        shear_cross += 2.0 * lens.mass[index]
            * dx * dy * inverse_radius_fourth;
    }
    const Scalar du_dx = 1.0 + shear_real;
    const Scalar du_dy = shear_cross;
    const Scalar dv_dx = shear_cross;
    const Scalar dv_dy = 1.0 - shear_real;
    return {
        source_x,
        source_y,
        du_dx,
        du_dy,
        dv_dx,
        dv_dy,
        du_dx * dv_dy - du_dy * dv_dx,
    };
}

TripleJet triple_point_magnification_jet(
    const TripleJet& source_x,
    const TripleJet& source_y,
    const TripleJet& separation,
    const TripleJet& mass_ratio,
    const TripleJet& tertiary_mass_ratio,
    const TripleJet& tertiary_separation,
    const TripleJet& tertiary_angle,
    std::int64_t convention,
    std::int32_t& image_count,
    bool& root_failure,
    double* derivative_indicator = nullptr)
{
    const auto native_geometry = convention == 0
        ? lcbinint::model::make_triple_lens_geometry(
            separation.value, mass_ratio.value, tertiary_mass_ratio.value,
            tertiary_separation.value, tertiary_angle.value)
        : lcbinint::model::make_triple_lens_geometry_vbm(
            separation.value, mass_ratio.value, tertiary_separation.value,
            tertiary_angle.value, tertiary_mass_ratio.value);
    const lcbinint::magnification::PointSourceMagnifier magnifier;
    const auto candidates = magnifier.triple_image_candidates(
        native_geometry, {source_x.value, source_y.value});
    if (derivative_indicator != nullptr) {
        *derivative_indicator = 0.0;
        for (const auto& candidate : candidates) {
            if (!candidate.physical) continue;
            const std::complex<double> image(
                candidate.position.x, candidate.position.y);
            std::complex<double> j1(0.0, 0.0);
            std::complex<double> j2(0.0, 0.0);
            std::complex<double> j3(0.0, 0.0);
            for (std::size_t index = 0; index < 3; ++index) {
                const std::complex<double> lens(
                    native_geometry.lens_positions[index].x,
                    native_geometry.lens_positions[index].y);
                const auto displacement = image - lens;
                const auto displacement_2 = displacement * displacement;
                const auto displacement_3 =
                    displacement_2 * displacement;
                const auto displacement_4 =
                    displacement_3 * displacement;
                const double mass = native_geometry.masses[index];
                j1 += mass / displacement_2;
                j2 -= 2.0 * mass / displacement_3;
                j3 += 6.0 * mass / displacement_4;
            }
            const auto j1_conjugate = std::conj(j1);
            const double determinant =
                1.0 - std::real(j1 * j1_conjugate);
            const double determinant_2 = determinant * determinant;
            const double denominator = std::abs(
                determinant * determinant_2 * determinant_2);
            if (
                denominator == 0.0 || !std::isfinite(denominator)
                || !std::isfinite(determinant)) {
                *derivative_indicator =
                    std::numeric_limits<double>::infinity();
                break;
            }
            const auto j1_conjugate_2 =
                j1_conjugate * j1_conjugate;
            const auto j3_modified = j3 * j1_conjugate_2;
            const double ob2 = std::norm(j2)
                * (6.0 - 6.0 * determinant + determinant_2);
            const auto j2_modified =
                j2 * j2 * j1_conjugate_2 * j1_conjugate;
            const double contribution = 0.5 * (
                std::abs(
                    ob2 - 6.0 * std::real(j2_modified)
                    - 2.0 * std::real(j3_modified) * determinant)
                + 3.0 * std::abs(std::imag(j2_modified)))
                / denominator;
            if (!std::isfinite(contribution)) {
                *derivative_indicator =
                    std::numeric_limits<double>::infinity();
                break;
            }
            *derivative_indicator += contribution;
        }
    }
    const auto active_lens = make_triple_lens_constants(
        separation, mass_ratio, tertiary_mass_ratio,
        tertiary_separation, tertiary_angle, convention);
    TripleLensConstants<double> value_lens;
    for (std::size_t index = 0; index < 3; ++index) {
        value_lens.lens_x[index] = active_lens.lens_x[index].value;
        value_lens.lens_y[index] = active_lens.lens_y[index].value;
        value_lens.mass[index] = active_lens.mass[index].value;
    }
    TripleJet magnification(0.0);
    std::int32_t physical_count = 0;
    bool physical_converged = true;
    for (const auto& candidate : candidates) {
        if (!candidate.physical) continue;
        ++physical_count;
        physical_converged = physical_converged
            && std::isfinite(candidate.residual)
            && candidate.residual <= 1.0e-7;
        const auto value_map = triple_map_values(
            candidate.position.x, candidate.position.y, value_lens);
        const double determinant = value_map.determinant;
        if (!(std::abs(determinant) > 1.0e-12)) {
            root_failure = true;
            continue;
        }
        const auto fixed_root_map = triple_map_values(
            TripleJet(candidate.position.x),
            TripleJet(candidate.position.y), active_lens);
        TripleJet image_x(candidate.position.x);
        TripleJet image_y(candidate.position.y);
        for (
            std::size_t parameter = 0;
            parameter < triple_kernel_derivative_count;
            ++parameter) {
            const double rhs_x =
                source_x.derivative[parameter]
                - fixed_root_map.source_x.derivative[parameter];
            const double rhs_y =
                source_y.derivative[parameter]
                - fixed_root_map.source_y.derivative[parameter];
            image_x.derivative[parameter] =
                (value_map.dv_dy * rhs_x - value_map.du_dy * rhs_y)
                / determinant;
            image_y.derivative[parameter] =
                (-value_map.dv_dx * rhs_x + value_map.du_dx * rhs_y)
                / determinant;
        }
        const auto active_map =
            triple_map_values(image_x, image_y, active_lens);
        magnification += 1.0 / (
            determinant >= 0.0
                ? active_map.determinant
                : -active_map.determinant);
    }
    // Extreme mass hierarchies can numerically lose demagnified roots at a
    // lens pole even after the native high-precision retry.  The native point
    // path deliberately retains the converged physical subset; rejecting it
    // here made otherwise valid finite-source support unusable.
    const bool valid_count = physical_count > 0;
    root_failure =
        root_failure || !physical_converged || !valid_count;
    image_count = physical_count;
    return magnification;
}

struct TripleHexadecapoleKernelResult {
    TripleJet magnification;
    TripleJet point_magnification;
    TripleJet quadrupole_correction;
    TripleJet hexadecapole_correction;
    bool topology_stable = false;
    bool root_failure = false;
    double limb_c_derivative = 0.0;
    double limb_d_derivative = 0.0;
    double quadrupole_limb_c_derivative = 0.0;
    double quadrupole_limb_d_derivative = 0.0;
    double hexadecapole_limb_c_derivative = 0.0;
    double hexadecapole_limb_d_derivative = 0.0;
};

TripleHexadecapoleKernelResult triple_hexadecapole_kernel(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double tertiary_mass_ratio,
    double tertiary_separation,
    double tertiary_angle,
    double source_radius,
    double limb_c,
    double limb_d,
    std::int64_t convention)
{
    constexpr std::array<double, 13> unit_x{
        0.0, 1.0, 0.0, -1.0, 0.0,
        0.5, 0.0, -0.5, 0.0,
        0.7071067811865475244, -0.7071067811865475244,
        -0.7071067811865475244, 0.7071067811865475244};
    constexpr std::array<double, 13> unit_y{
        0.0, 0.0, 1.0, 0.0, -1.0,
        0.0, 0.5, 0.0, -0.5,
        0.7071067811865475244, 0.7071067811865475244,
        -0.7071067811865475244, -0.7071067811865475244};
    const TripleJet centre_x = TripleJet::variable(source_x, 0);
    const TripleJet centre_y = TripleJet::variable(source_y, 1);
    const TripleJet active_separation =
        TripleJet::variable(separation, 2);
    const TripleJet active_mass_ratio =
        TripleJet::variable(mass_ratio, 3);
    const TripleJet active_tertiary_mass_ratio =
        TripleJet::variable(tertiary_mass_ratio, 4);
    const TripleJet active_tertiary_separation =
        TripleJet::variable(tertiary_separation, 5);
    const TripleJet active_tertiary_angle =
        TripleJet::variable(tertiary_angle, 6);
    const TripleJet active_radius =
        TripleJet::variable(source_radius, 7);
    std::array<TripleJet, 13> samples;
    std::array<std::int32_t, 13> image_counts{};
    TripleHexadecapoleKernelResult result;
    for (std::size_t sample = 0; sample < samples.size(); ++sample) {
        samples[sample] = triple_point_magnification_jet(
            centre_x + unit_x[sample] * active_radius,
            centre_y + unit_y[sample] * active_radius,
            active_separation, active_mass_ratio,
            active_tertiary_mass_ratio, active_tertiary_separation,
            active_tertiary_angle, convention, image_counts[sample],
            result.root_failure);
    }
    const TripleJet a0 = samples[0];
    const TripleJet a1_plus =
        0.25 * (samples[1] + samples[2] + samples[3] + samples[4]) - a0;
    const TripleJet a2_plus =
        0.25 * (samples[5] + samples[6] + samples[7] + samples[8]) - a0;
    const TripleJet a1_cross =
        0.25 * (samples[9] + samples[10] + samples[11] + samples[12]) - a0;
    const TripleJet a2rho2 = (16.0 * a2_plus - a1_plus) / 3.0;
    const TripleJet a4rho4 =
        0.5 * (a1_plus + a1_cross) - a2rho2;
    const double denominator = 15.0 - 5.0 * limb_c - 3.0 * limb_d;
    const double gamma = denominator != 0.0
        ? 10.0 * limb_c / denominator
        : 0.0;
    const double lambda = denominator != 0.0
        ? 12.0 * limb_d / denominator
        : 0.0;
    result.point_magnification = a0;
    result.quadrupole_correction =
        0.5 * a2rho2 * (1.0 - 0.2 * gamma - lambda / 9.0);
    result.hexadecapole_correction =
        a4rho4 / 3.0
        * (1.0 - 11.0 * gamma / 35.0 - 7.0 * lambda / 39.0);
    result.magnification =
        a0 + result.quadrupole_correction
        + result.hexadecapole_correction;
    result.topology_stable =
        !result.root_failure
        && std::all_of(
            image_counts.begin() + 1,
            image_counts.end(),
            [&](std::int32_t count) {
                return count == image_counts[0];
            });
    if (denominator != 0.0) {
        const double inverse_denominator_squared =
            1.0 / (denominator * denominator);
        const double gamma_c =
            10.0 / denominator
            + 50.0 * limb_c * inverse_denominator_squared;
        const double gamma_d =
            30.0 * limb_c * inverse_denominator_squared;
        const double lambda_c =
            60.0 * limb_d * inverse_denominator_squared;
        const double lambda_d =
            12.0 / denominator
            + 36.0 * limb_d * inverse_denominator_squared;
        result.quadrupole_limb_c_derivative =
            -0.5 * a2rho2.value
            * (0.2 * gamma_c + lambda_c / 9.0);
        result.quadrupole_limb_d_derivative =
            -0.5 * a2rho2.value
            * (0.2 * gamma_d + lambda_d / 9.0);
        result.hexadecapole_limb_c_derivative =
            -a4rho4.value / 3.0
            * (11.0 * gamma_c / 35.0 + 7.0 * lambda_c / 39.0);
        result.hexadecapole_limb_d_derivative =
            -a4rho4.value / 3.0
            * (11.0 * gamma_d / 35.0 + 7.0 * lambda_d / 39.0);
        result.limb_c_derivative =
            result.quadrupole_limb_c_derivative
            + result.hexadecapole_limb_c_derivative;
        result.limb_d_derivative =
            result.quadrupole_limb_d_derivative
            + result.hexadecapole_limb_d_derivative;
    }
    return result;
}

std::uint64_t tile_key(std::int32_t x, std::int32_t y)
{
    return (
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32)
        | static_cast<std::uint32_t>(y);
}

// Whether the tile may contain a point that maps into the source disk, and so
// has to stay on the flood-fill frontier.
//
// Sampling nine points in the tile answers a different question -- whether one
// of those nine happens to land in the disk -- and a thin image component that
// merely passes between them fails all nine, which stops the expansion on the
// component the walk was following.  That is the same defect class as the limb
// raster the certificate replaced: a sample count used as a decision about a
// set it does not cover.  It is silent, too; the walk still reports
// `support_valid`.  Measured on the tangent cusp at resolution 128, tile_size
// 2/4/8/16/32 gave 3.960953/3.960857/3.959864/3.955731/3.945949 against a
// reference of 3.960888.
//
// Bound the map instead of sampling it.  The tile is convex, so for every `z`
// in it
//
//     |f(z) - zeta| >= |f(c) - zeta| - L |z - c| >= |f(c) - zeta| - L r
//
// with `c` the centre and `r` the half-diagonal.  For
// `f(z) = z - sum_i m_i / conj(z - z_i)` the differential is the identity plus
// `df/dconj(z) = sum_i m_i / conj(z - z_i)^2`, so
//
//     L = 1 + sum_i m_i / d_i^2,   d_i = dist(tile, z_i)
//
// bounds it over the whole tile.  Rejecting only when the lower bound exceeds
// the source radius can over-admit but never under-admit, which is the
// direction that keeps the support honest.  A tile containing a lens has
// `d_i = 0`; it is admitted outright, as its neighbourhood of images requires.
// One lens-map evaluation replaces nine.
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
    const double half_width = 0.5 * tile_width;
    const double centre_x =
        static_cast<double>(tile_x) * tile_width + half_width;
    const double centre_y =
        static_cast<double>(tile_y) * tile_width + half_width;

    // Both lenses sit on the real axis, so the distance from the tile to a lens
    // is the usual clamped box distance.
    const auto lens_distance_squared = [&](double lens_x) {
        const double dx = std::max(0.0, std::abs(centre_x - lens_x) - half_width);
        const double dy = std::max(0.0, std::abs(centre_y) - half_width);
        return dx * dx + dy * dy;
    };
    const double lens_1_distance_squared = lens_distance_squared(lens_1_x);
    const double lens_2_distance_squared = lens_distance_squared(lens_2_x);
    if (lens_1_distance_squared <= 0.0 || lens_2_distance_squared <= 0.0) {
        return true;
    }

    const double dx_1 = centre_x - lens_1_x;
    const double dx_2 = centre_x - lens_2_x;
    const double radius_1_squared = dx_1 * dx_1 + centre_y * centre_y;
    const double radius_2_squared = dx_2 * dx_2 + centre_y * centre_y;
    const double mapped_x =
        centre_x - mass_1 * dx_1 / radius_1_squared
        - mass_2 * dx_2 / radius_2_squared;
    const double mapped_y =
        centre_y - mass_1 * centre_y / radius_1_squared
        - mass_2 * centre_y / radius_2_squared;
    const double distance =
        std::hypot(mapped_x - source_x, mapped_y - source_y);
    if (!std::isfinite(distance)) {
        return true;
    }

    const double lipschitz =
        1.0 + mass_1 / lens_1_distance_squared + mass_2 / lens_2_distance_squared;
    const double half_diagonal = half_width * std::sqrt(2.0);
    return distance - lipschitz * half_diagonal <= source_radius;
}

struct CartesianDiscovery {
    // During the flood fill this is the full visited queue, including the
    // one-tile inactive frontier.  `compact_active_tiles` turns it into the
    // active support consumed by the integration kernel after preserving the
    // visited count used by public diagnostics.
    std::vector<std::array<std::int32_t, 2>> queue;
    bool overflow = false;
    bool root_failure = false;
    // False when the caustic geometry certifies a component that no probe
    // could reach; the value is then not trustworthy at any resolution.
    bool support_proven = true;
    std::uint64_t support_fingerprint = 0;
    std::int32_t visited_count = 0;
    std::int32_t active_count = 0;
    std::int32_t seed_count = 0;
};

void compact_active_tiles(
    CartesianDiscovery& discovery,
    const std::vector<std::uint8_t>& active)
{
    discovery.visited_count =
        static_cast<std::int32_t>(discovery.queue.size());
    std::size_t destination = 0;
    for (std::size_t source = 0; source < discovery.queue.size(); ++source) {
        if (!active[source]) continue;
        if (destination != source) {
            discovery.queue[destination] = discovery.queue[source];
        }
        ++destination;
    }
    discovery.queue.resize(destination);
    discovery.active_count = static_cast<std::int32_t>(destination);
}

// The caustic polyline depends only on (separation, mass_ratio) and is cached
// inside the magnifier, so one instance per thread keeps the certificate off
// the per-epoch cost.  Rebuilding it per call would re-run the whole
// critical-curve phase scan for every light-curve point.
const std::vector<std::vector<lcbinint::SourcePosition>>& cached_caustic_branches(
    double separation, double mass_ratio)
{
    thread_local lcbinint::magnification::FiniteSourceMagnifier magnifier {{}};
    return magnifier.binary_caustic_branches(separation, mass_ratio);
}

struct CartesianSeedSupport {
    // Image coordinates are independent of the Cartesian resolution.  Keep
    // them in solver/probe order so every rung quantises exactly the same seed
    // stream the standalone discovery would have inserted.
    std::vector<lcbinint::Complex> image_coordinates;
    bool root_failure = false;
    bool support_proven = true;
    std::uint64_t support_fingerprint = 0;
};

CartesianSeedSupport prepare_cartesian_seed_support(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    std::int64_t limb_samples)
{
    CartesianSeedSupport prepared;
    BinaryRootContinuation continuation;
    prepared.image_coordinates.reserve(
        static_cast<std::size_t>((limb_samples + 1) * binary_root_count));
    const double two_pi = 2.0 * std::acos(-1.0);
    for (std::int64_t sample = 0; sample <= limb_samples; ++sample) {
        double sample_x = source_x;
        double sample_y = source_y;
        if (sample > 0) {
            const double angle =
                two_pi * static_cast<double>(sample - 1)
                / static_cast<double>(limb_samples);
            sample_x += source_radius * std::cos(angle);
            sample_y += source_radius * std::sin(angle);
        }
        const auto images = solve_binary_images(
            sample_x, sample_y, separation, mass_ratio, &continuation);
        std::int32_t physical_count = 0;
        for (std::size_t root = 0; root < binary_root_count; ++root) {
            physical_count += static_cast<std::int32_t>(images.physical[root]);
            if (!images.physical[root]) continue;
            prepared.image_coordinates.push_back(images.roots[root]);
        }
        prepared.root_failure =
            prepared.root_failure
            || physical_count < 3
            || physical_count > 5;
    }
    // The limb set above is useful redundancy but is not a completeness
    // criterion: a component born in a cap of the source disk subtends an arc
    // of the limb that shrinks to zero as the caustic approaches tangency, so
    // no fixed limb_samples can be relied on to sample it.  The certificate is
    // derived from the caustic geometry instead and is therefore independent
    // of limb_samples and of the tile resolution.
    const auto support = lcbinint::magnification::certify_disk_support(
        cached_caustic_branches(separation, mass_ratio),
        {source_x, source_y}, source_radius);
    prepared.support_fingerprint = support.fingerprint;
    prepared.support_proven = lcbinint::magnification::resolve_certified_probes(
        support, [&](lcbinint::SourcePosition probe) {
            const auto images = solve_binary_images(
                probe.x, probe.y, separation, mass_ratio, &continuation);
            std::int32_t physical_count = 0;
            for (std::size_t root = 0; root < binary_root_count; ++root) {
                physical_count += static_cast<std::int32_t>(images.physical[root]);
            }
            for (std::size_t root = 0; root < binary_root_count; ++root) {
                if (!images.physical[root]) continue;
                prepared.image_coordinates.push_back(images.roots[root]);
            }
            prepared.root_failure =
                prepared.root_failure || physical_count > 5;
            return static_cast<int>(physical_count);
        });
    // An extremum inside the disk whose probes all saw the same image count
    // means one of the two components the caustic arc separates is thinner
    // than the finest probe offset, so a component exists that this support
    // does not cover.  Fold that into root_failure here rather than at each
    // consumer: every caller already treats root_failure as "the support could
    // not be established", and a support that is known to be incomplete must
    // never be reported as valid at any resolution.
    prepared.root_failure =
        prepared.root_failure || !prepared.support_proven;
    return prepared;
}

CartesianSeedSupport cached_cartesian_seed_support(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    std::int64_t limb_samples)
{
    struct Entry {
        double source_x;
        double source_y;
        double separation;
        double mass_ratio;
        double source_radius;
        std::int64_t limb_samples;
        CartesianSeedSupport support;
    };
    constexpr std::size_t maximum_entries = 2048;
    static std::deque<Entry> entries;
    static std::shared_mutex mutex;
    {
        std::shared_lock lock(mutex);
        for (const auto& entry : entries) {
            if (entry.source_x == source_x && entry.source_y == source_y &&
                entry.separation == separation &&
                entry.mass_ratio == mass_ratio &&
                entry.source_radius == source_radius &&
                entry.limb_samples == limb_samples) {
                return entry.support;
            }
        }
    }

    auto support = prepare_cartesian_seed_support(
        source_x, source_y, separation, mass_ratio, source_radius,
        limb_samples);
    {
        std::unique_lock lock(mutex);
        // Another FFI worker may have populated the same exact epoch while
        // this thread was solving probes. Reuse that canonical ordering.
        for (const auto& entry : entries) {
            if (entry.source_x == source_x && entry.source_y == source_y &&
                entry.separation == separation &&
                entry.mass_ratio == mass_ratio &&
                entry.source_radius == source_radius &&
                entry.limb_samples == limb_samples) {
                return entry.support;
            }
        }
        if (entries.size() >= maximum_entries) entries.pop_front();
        entries.push_back({
            source_x, source_y, separation, mass_ratio, source_radius,
            limb_samples, support});
    }
    return support;
}

CartesianDiscovery discover_cartesian_support_from_prepared(
    double tile_width,
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    std::int64_t tile_capacity,
    const CartesianSeedSupport& prepared)
{
    CartesianDiscovery result;
    // `tile_capacity` is an overflow ceiling, not an expected size.  Reserving
    // the whole ceiling made every epoch pay a multi-megabyte zeroed bucket
    // array at the fine rungs, so grow from a modest floor the way the triple
    // discovery already does.
    const auto initial_capacity = static_cast<std::size_t>(
        std::min<std::int64_t>(tile_capacity, 4096));
    result.queue.reserve(initial_capacity);
    std::unordered_map<std::uint64_t, std::int32_t> visited;
    visited.reserve(initial_capacity);
    std::unordered_set<std::uint64_t> seeds;
    seeds.reserve(prepared.image_coordinates.size());

    const auto insert = [&](std::int32_t x, std::int32_t y) {
        const std::uint64_t key = tile_key(x, y);
        if (visited.find(key) != visited.end()) return true;
        if (result.queue.size() >= static_cast<std::size_t>(tile_capacity)) {
            result.overflow = true;
            return false;
        }
        visited.emplace(
            key, static_cast<std::int32_t>(result.queue.size()));
        result.queue.push_back({x, y});
        return true;
    };

    for (const auto& image : prepared.image_coordinates) {
        const auto tile_x = static_cast<std::int32_t>(
            std::floor(image.real() / tile_width));
        const auto tile_y = static_cast<std::int32_t>(
            std::floor(image.imag() / tile_width));
        if (insert(tile_x, tile_y)) seeds.insert(tile_key(tile_x, tile_y));
    }
    result.root_failure = prepared.root_failure;
    result.support_proven = prepared.support_proven;
    result.support_fingerprint = prepared.support_fingerprint;
    result.seed_count = static_cast<std::int32_t>(result.queue.size());

    constexpr std::array<std::array<std::int32_t, 2>, 4> neighbours{{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}
    }};
    std::vector<std::uint8_t> active_tiles;
    active_tiles.reserve(initial_capacity);
    for (std::size_t head = 0; head < result.queue.size(); ++head) {
        const auto tile = result.queue[head];
        const bool active =
            seeds.find(tile_key(tile[0], tile[1])) != seeds.end()
            || tile_has_inside_probe(
                tile[0], tile[1], tile_width, source_x, source_y,
                separation, mass_ratio, source_radius);
        active_tiles.push_back(static_cast<std::uint8_t>(active));
        if (!active) continue;
        for (const auto& neighbour : neighbours) {
            insert(tile[0] + neighbour[0], tile[1] + neighbour[1]);
        }
    }
    compact_active_tiles(result, active_tiles);
    return result;
}

CartesianDiscovery discover_cartesian_support(
    double tile_width,
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    std::int64_t tile_capacity,
    std::int64_t limb_samples)
{
    const auto prepared = cached_cartesian_seed_support(
        source_x, source_y, separation, mass_ratio, source_radius,
        limb_samples);
    return discover_cartesian_support_from_prepared(
        tile_width, source_x, source_y, separation, mass_ratio,
        source_radius, tile_capacity, prepared);
}

// The triple-lens frontier test, bounded exactly as the binary one above.  The
// three lenses are not collinear here, so the tile-to-lens distance is the
// clamped box distance in both coordinates.  `phi` is
// `1 - |f(z) - zeta|^2 / rho^2`, so the distance the bound needs is
// `rho * sqrt(1 - phi)`.
bool triple_tile_has_inside_probe(
    std::int32_t tile_x,
    std::int32_t tile_y,
    double tile_width,
    double source_x,
    double source_y,
    const TripleLensConstants<double>& lens,
    double source_radius)
{
    const double half_width = 0.5 * tile_width;
    const double centre_x =
        static_cast<double>(tile_x) * tile_width + half_width;
    const double centre_y =
        static_cast<double>(tile_y) * tile_width + half_width;

    double lipschitz = 1.0;
    for (std::size_t lens_index = 0; lens_index < 3; ++lens_index) {
        const double dx = std::max(
            0.0, std::abs(centre_x - lens.lens_x[lens_index]) - half_width);
        const double dy = std::max(
            0.0, std::abs(centre_y - lens.lens_y[lens_index]) - half_width);
        const double distance_squared = dx * dx + dy * dy;
        if (distance_squared <= 0.0) {
            return true;
        }
        lipschitz += lens.mass[lens_index] / distance_squared;
    }

    const auto values = triple_phi_derivatives<false>(
        centre_x, centre_y, source_x, source_y, lens,
        1.0 / (source_radius * source_radius));
    if (!std::isfinite(values.phi)) {
        return true;
    }
    const double distance =
        source_radius * std::sqrt(std::max(0.0, 1.0 - values.phi));
    const double half_diagonal = half_width * std::sqrt(2.0);
    return distance - lipschitz * half_diagonal <= source_radius;
}

CartesianDiscovery discover_triple_cartesian_support(
    double tile_width,
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double tertiary_mass_ratio,
    double tertiary_separation,
    double tertiary_angle,
    double source_radius,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t convention)
{
    CartesianDiscovery result;
    const auto initial_capacity = static_cast<std::size_t>(
        std::min<std::int64_t>(tile_capacity, 4096));
    result.queue.reserve(initial_capacity);
    std::unordered_map<std::uint64_t, std::int32_t> visited;
    visited.reserve(initial_capacity);
    std::unordered_set<std::uint64_t> seeds;
    seeds.reserve(
        static_cast<std::size_t>((limb_samples + 1) * triple_root_count));
    const auto insert = [&](std::int32_t x, std::int32_t y) {
        const auto key = tile_key(x, y);
        if (visited.find(key) != visited.end()) return true;
        if (result.queue.size() >= static_cast<std::size_t>(tile_capacity)) {
            result.overflow = true;
            return false;
        }
        visited.emplace(
            key, static_cast<std::int32_t>(result.queue.size()));
        result.queue.push_back({x, y});
        return true;
    };
    const auto native_geometry = convention == 0
        ? lcbinint::model::make_triple_lens_geometry(
            separation, mass_ratio, tertiary_mass_ratio,
            tertiary_separation, tertiary_angle)
        : lcbinint::model::make_triple_lens_geometry_vbm(
            separation, mass_ratio, tertiary_separation,
            tertiary_angle, tertiary_mass_ratio);
    const auto classification_lens = make_triple_lens_constants(
        separation, mass_ratio, tertiary_mass_ratio,
        tertiary_separation, tertiary_angle, convention);
    const lcbinint::magnification::PointSourceMagnifier magnifier;
    const double two_pi = 2.0 * std::acos(-1.0);
    for (std::int64_t sample = 0; sample <= limb_samples; ++sample) {
        double sample_x = source_x;
        double sample_y = source_y;
        if (sample > 0) {
            const double angle =
                two_pi * static_cast<double>(sample - 1)
                / static_cast<double>(limb_samples);
            sample_x += source_radius * std::cos(angle);
            sample_y += source_radius * std::sin(angle);
        }
        const auto candidates = magnifier.triple_image_candidates(
            native_geometry, {sample_x, sample_y});
        std::int32_t physical_count = 0;
        for (const auto& candidate : candidates) {
            if (!candidate.physical) continue;
            ++physical_count;
            const auto tile_x = static_cast<std::int32_t>(
                std::floor(candidate.position.x / tile_width));
            const auto tile_y = static_cast<std::int32_t>(
                std::floor(candidate.position.y / tile_width));
            if (insert(tile_x, tile_y)) {
                seeds.insert(tile_key(tile_x, tile_y));
            }
        }
        const bool valid_count = physical_count > 0;
        // The native high-precision classifier can retain a useful
        // demagnified image with a conservative residual flag.  Flood support
        // only needs one valid seed per connected component; the 65 source
        // probes provide redundancy, so do not invalidate the whole epoch
        // for one flagged probe.
        result.root_failure = result.root_failure || !valid_count;
    }
    result.seed_count = static_cast<std::int32_t>(result.queue.size());
    constexpr std::array<std::array<std::int32_t, 2>, 4> neighbours{{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}
    }};
    std::vector<std::uint8_t> active_tiles;
    active_tiles.reserve(initial_capacity);
    for (std::size_t head = 0; head < result.queue.size(); ++head) {
        const auto tile = result.queue[head];
        const bool active =
            seeds.find(tile_key(tile[0], tile[1])) != seeds.end()
            || triple_tile_has_inside_probe(
                tile[0], tile[1], tile_width, source_x, source_y,
                classification_lens, source_radius);
        active_tiles.push_back(static_cast<std::uint8_t>(active));
        if (!active) continue;
        for (const auto& neighbour : neighbours) {
            insert(tile[0] + neighbour[0], tile[1] + neighbour[1]);
        }
    }
    compact_active_tiles(result, active_tiles);
    return result;
}

template <typename Scalar>
struct CartesianEpochResult {
    KernelResult<Scalar> integration;
    std::int32_t tile_count = 0;
    bool overflow = false;
    bool root_failure = false;
};

template <typename Scalar>
CartesianEpochResult<Scalar> cartesian_epoch_kernel_from_prepared(
    double cell_size,
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& source_radius,
    const Scalar& limb_d,
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    MomentMode mode,
    std::int64_t boundary_subdivision,
    const CartesianSeedSupport& prepared)
{
    const double tile_width =
        cell_size * static_cast<double>(tile_size);
    const auto discovery = discover_cartesian_support_from_prepared(
        tile_width,
        scalar_value(source_x), scalar_value(source_y),
        scalar_value(separation), scalar_value(mass_ratio),
        scalar_value(source_radius), tile_capacity, prepared);
    std::vector<double> origins(2 * discovery.queue.size());
    for (std::size_t index = 0; index < discovery.queue.size(); ++index) {
        origins[2 * index] =
            discovery.queue[index][0] * tile_width;
        origins[2 * index + 1] =
            discovery.queue[index][1] * tile_width;
    }
    CartesianEpochResult<Scalar> result;
    result.integration = fixed_support_kernel(
        origins.data(), nullptr,
        static_cast<std::int64_t>(discovery.queue.size()), cell_size,
        source_x, source_y, separation, mass_ratio, source_radius, limb_d,
        static_cast<int>(tile_size), mode,
        static_cast<int>(boundary_subdivision));
    result.tile_count =
        discovery.visited_count;
    result.overflow = discovery.overflow;
    result.root_failure = discovery.root_failure;
    return result;
}

template <typename Scalar>
CartesianEpochResult<Scalar> cartesian_epoch_kernel(
    double cell_size,
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& source_radius,
    const Scalar& limb_d,
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    MomentMode mode,
    std::int64_t boundary_subdivision)
{
    const auto prepared = cached_cartesian_seed_support(
        scalar_value(source_x), scalar_value(source_y),
        scalar_value(separation), scalar_value(mass_ratio),
        scalar_value(source_radius), limb_samples);
    return cartesian_epoch_kernel_from_prepared(
        cell_size, source_x, source_y, separation, mass_ratio,
        source_radius, limb_d, tile_size, tile_capacity, mode,
        boundary_subdivision, prepared);
}

template <typename Scalar>
CartesianEpochResult<Scalar> triple_cartesian_epoch_kernel(
    double cell_size,
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& tertiary_mass_ratio,
    const Scalar& tertiary_separation,
    const Scalar& tertiary_angle,
    const Scalar& source_radius,
    const Scalar& limb_d,
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t convention,
    MomentMode mode,
    std::int64_t boundary_subdivision)
{
    const double tile_width =
        cell_size * static_cast<double>(tile_size);
    const auto discovery = discover_triple_cartesian_support(
        tile_width, scalar_value(source_x), scalar_value(source_y),
        scalar_value(separation), scalar_value(mass_ratio),
        scalar_value(tertiary_mass_ratio),
        scalar_value(tertiary_separation), scalar_value(tertiary_angle),
        scalar_value(source_radius), tile_capacity, limb_samples, convention);
    std::vector<double> origins(2 * discovery.queue.size());
    for (std::size_t index = 0; index < discovery.queue.size(); ++index) {
        origins[2 * index] = discovery.queue[index][0] * tile_width;
        origins[2 * index + 1] = discovery.queue[index][1] * tile_width;
    }
    CartesianEpochResult<Scalar> result;
    result.integration = triple_fixed_support_kernel(
        origins.data(), static_cast<std::int64_t>(discovery.queue.size()),
        cell_size, source_x, source_y, separation, mass_ratio,
        tertiary_mass_ratio, tertiary_separation, tertiary_angle,
        source_radius, limb_d, static_cast<int>(tile_size), convention,
        mode, static_cast<int>(boundary_subdivision));
    result.tile_count =
        discovery.visited_count;
    result.overflow = discovery.overflow;
    result.root_failure = discovery.root_failure;
    return result;
}

struct PolarSeed {
    double radius = 0.0;
    double angle = 0.0;
    bool physical = false;
};

struct PolarDiscovery {
    std::vector<PolarSeed> seeds;
    bool root_failure = false;
};

PolarDiscovery discover_triple_polar_support(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double tertiary_mass_ratio,
    double tertiary_separation,
    double tertiary_angle,
    double source_radius,
    std::int64_t limb_samples,
    std::int64_t convention)
{
    PolarDiscovery result;
    result.seeds.reserve(
        static_cast<std::size_t>((limb_samples + 1) * triple_root_count));
    const auto geometry = convention == 0
        ? lcbinint::model::make_triple_lens_geometry(
            separation, mass_ratio, tertiary_mass_ratio,
            tertiary_separation, tertiary_angle)
        : lcbinint::model::make_triple_lens_geometry_vbm(
            separation, mass_ratio, tertiary_separation,
            tertiary_angle, tertiary_mass_ratio);
    const lcbinint::magnification::PointSourceMagnifier magnifier;
    const double two_pi = 2.0 * std::acos(-1.0);
    for (std::int64_t sample = 0; sample <= limb_samples; ++sample) {
        double sample_x = source_x;
        double sample_y = source_y;
        if (sample > 0) {
            const double angle =
                two_pi * static_cast<double>(sample - 1)
                / static_cast<double>(limb_samples);
            sample_x += source_radius * std::cos(angle);
            sample_y += source_radius * std::sin(angle);
        }
        const auto candidates = magnifier.triple_image_candidates(
            geometry, {sample_x, sample_y});
        std::int32_t physical_count = 0;
        for (const auto& candidate : candidates) {
            const double radius = std::hypot(
                candidate.position.x, candidate.position.y);
            const double angle = std::atan2(
                candidate.position.y, candidate.position.x);
            result.seeds.push_back(
                {radius, angle, candidate.physical});
            if (!candidate.physical) continue;
            ++physical_count;
        }
        const bool valid_count = physical_count > 0;
        result.root_failure = result.root_failure || !valid_count;
    }
    return result;
}

struct PolarFloodRun {
    std::int32_t angular_index = 0;
    std::int32_t left = 0;
    std::int32_t right = 0;
    double left_inside_distance_squared = -1.0;
    double right_inside_distance_squared = -1.0;
    double left_outside_distance_squared = -1.0;
    double right_outside_distance_squared = -1.0;
};

struct PolarFloodSupport {
    std::vector<PolarFloodRun> runs;
    std::array<double, 3> moments{};
    bool overflow = false;
    bool root_failure = false;
    std::int64_t cell_count = 0;
};

// A polar grid may have millions of possible angular columns even though the
// discovered image support occupies only a few connected arcs.  Keeping one
// vector object per possible angle makes the cost of a fine angular grid
// proportional to the *global* circumference.  Store interval state in lazy
// pages instead: lookups remain O(1), while untouched angles allocate nothing.
// The same structure is also used for the run index during boundary assembly.
class PolarVisitedCellIntervals {
public:
    using Interval = std::array<std::int32_t, 2>;

    struct Column {
        Interval first{1, 0};
        std::vector<Interval> overflow;

        bool contains(std::int32_t radial) const
        {
            if (first[0] > first[1] || radial < first[0]) return false;
            if (radial <= first[1]) return true;
            for (const auto& interval : overflow) {
                if (radial < interval[0]) return false;
                if (radial <= interval[1]) return true;
            }
            return false;
        }

        std::int64_t first_unvisited_at_or_after(std::int64_t radial) const
        {
            if (first[0] > first[1] || radial < first[0]) return radial;
            if (radial <= first[1]) {
                radial = static_cast<std::int64_t>(first[1]) + 1;
            }
            for (const auto& interval : overflow) {
                if (radial < interval[0]) return radial;
                if (radial <= interval[1]) {
                    radial = static_cast<std::int64_t>(interval[1]) + 1;
                }
            }
            return radial;
        }

        void add(std::int32_t left, std::int32_t right)
        {
            if (right < left) return;
            if (first[0] > first[1]) {
                first = {left, right};
                return;
            }
            if (static_cast<std::int64_t>(right) + 1 < first[0]) {
                overflow.insert(overflow.begin(), first);
                first = {left, right};
                return;
            }
            if (static_cast<std::int64_t>(left)
                <= static_cast<std::int64_t>(first[1]) + 1) {
                first[0] = std::min(first[0], left);
                first[1] = std::max(first[1], right);
                std::size_t merged = 0;
                while (
                    merged < overflow.size()
                    && static_cast<std::int64_t>(overflow[merged][0])
                        <= static_cast<std::int64_t>(first[1]) + 1) {
                    first[1] = std::max(first[1], overflow[merged][1]);
                    ++merged;
                }
                if (merged != 0) {
                    overflow.erase(
                        overflow.begin(),
                        overflow.begin() + static_cast<std::ptrdiff_t>(merged));
                }
                return;
            }

            auto iterator = std::lower_bound(
                overflow.begin(), overflow.end(), left,
                [](const Interval& interval, std::int32_t value) {
                    return interval[0] < value;
                });
            if (
                iterator != overflow.begin()
                && static_cast<std::int64_t>(std::prev(iterator)->at(1)) + 1
                    >= left) {
                --iterator;
                iterator->at(1) = std::max(iterator->at(1), right);
            } else {
                iterator = overflow.insert(iterator, {left, right});
            }
            while (
                std::next(iterator) != overflow.end()
                && static_cast<std::int64_t>(std::next(iterator)->at(0))
                    <= static_cast<std::int64_t>(iterator->at(1)) + 1) {
                iterator->at(1) = std::max(
                    iterator->at(1), std::next(iterator)->at(1));
                overflow.erase(std::next(iterator));
            }
        }

        template <typename Function>
        void for_each_interval(Function&& function) const
        {
            if (first[0] <= first[1]) function(first);
            for (const auto& interval : overflow) function(interval);
        }
    };

    explicit PolarVisitedCellIntervals(std::int64_t angular_bins)
        : pages_(static_cast<std::size_t>(
            (angular_bins + kColumnsPerPage - 1) / kColumnsPerPage))
    {}

    const Column* find(std::int32_t angular) const
    {
        const auto index = static_cast<std::size_t>(angular);
        const auto& page = pages_[index / kColumnsPerPage];
        return page == nullptr
            ? nullptr
            : &page->columns[index % kColumnsPerPage];
    }

    Column& get_or_create(std::int32_t angular)
    {
        const auto index = static_cast<std::size_t>(angular);
        auto& page = pages_[index / kColumnsPerPage];
        if (page == nullptr) page = std::make_unique<Page>();
        return page->columns[index % kColumnsPerPage];
    }

    void direction(
        std::int32_t angular, double dtheta, double& cosine, double& sine)
    {
        const auto index = static_cast<std::size_t>(angular);
        auto& page = pages_[index / kColumnsPerPage];
        if (page == nullptr) page = std::make_unique<Page>();
        const auto offset = index % kColumnsPerPage;
        const std::uint64_t bit = std::uint64_t{1} << offset;
        if ((page->direction_mask & bit) == 0) {
            const double theta =
                (static_cast<double>(angular) + 0.5) * dtheta;
            page->cosine[offset] = std::cos(theta);
            page->sine[offset] = std::sin(theta);
            page->direction_mask |= bit;
        }
        cosine = page->cosine[offset];
        sine = page->sine[offset];
    }

private:
    static constexpr std::size_t kColumnsPerPage = 64;

    struct Page {
        std::array<Column, kColumnsPerPage> columns;
        std::array<double, kColumnsPerPage> cosine{};
        std::array<double, kColumnsPerPage> sine{};
        std::uint64_t direction_mask = 0;
    };

    std::vector<std::unique_ptr<Page>> pages_;
};

template <MomentMode Mode, bool AccumulateMoments>
PolarFloodSupport discover_triple_polar_flood_support(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double tertiary_mass_ratio,
    double tertiary_separation,
    double tertiary_angle,
    double source_radius,
    double dr,
    std::int64_t angular_bins,
    std::int64_t limb_samples,
    std::int64_t cell_capacity,
    std::int64_t convention)
{
    const auto seeds = discover_triple_polar_support(
        source_x, source_y, separation, mass_ratio,
        tertiary_mass_ratio, tertiary_separation, tertiary_angle,
        source_radius, limb_samples, convention);
    PolarFloodSupport result;
    result.root_failure = seeds.root_failure;
    const auto lens = make_triple_lens_constants(
        separation, mass_ratio, tertiary_mass_ratio,
        tertiary_separation, tertiary_angle, convention);
    const double source_radius_squared = source_radius * source_radius;
    const double inverse_source_radius_squared =
        1.0 / source_radius_squared;
    const double two_pi = 2.0 * std::acos(-1.0);
    const double dtheta = two_pi / angular_bins;
    PolarVisitedCellIntervals visited(angular_bins);
    struct PolarFrontier {
        std::int32_t left = 0;
        std::int32_t right = -1;
        std::int32_t angular = 0;
    };
    std::deque<PolarFrontier> queue;
    const auto wrap_angle = [&](std::int32_t index) {
        std::int64_t wrapped = index % angular_bins;
        if (wrapped < 0) wrapped += angular_bins;
        return static_cast<std::int32_t>(wrapped);
    };
    const auto cell_visited = [&](std::int32_t radial, std::int32_t angular) {
        if (radial < 0) return true;
        const auto* intervals = visited.find(wrap_angle(angular));
        return intervals != nullptr && intervals->contains(radial);
    };
    const auto first_unvisited_at_or_after = [&visited, &wrap_angle](
        std::int32_t radial, std::int32_t angular) {
        const auto* intervals = visited.find(wrap_angle(angular));
        return intervals == nullptr
            ? static_cast<std::int64_t>(radial)
            : intervals->first_unvisited_at_or_after(radial);
    };
    const auto add_visited = [&](
        std::int32_t angular, std::int32_t left, std::int32_t right) {
        visited.get_or_create(wrap_angle(angular)).add(left, right);
    };
    const auto mapped_distance_squared = [&](
        std::int32_t radial, std::int32_t angular) {
        if (radial < 0) return std::numeric_limits<double>::infinity();
        double cosine = 1.0;
        double sine = 0.0;
        visited.direction(wrap_angle(angular), dtheta, cosine, sine);
        const double radius = (static_cast<double>(radial) + 0.5) * dr;
        const auto mapped = triple_map_source(
            radius * cosine, radius * sine, lens);
        const double dx = mapped.source_x - source_x;
        const double dy = mapped.source_y - source_y;
        return dx * dx + dy * dy;
    };
    const auto enqueue_run = [&first_unvisited_at_or_after, &queue, &wrap_angle](
        std::int32_t left, std::int32_t right, std::int32_t angular) {
        left = std::max<std::int32_t>(left, 0);
        left = first_unvisited_at_or_after(left, angular);
        if (left <= right) {
            queue.push_back({left, right, wrap_angle(angular)});
        }
    };
    for (const auto& seed : seeds.seeds) {
        if (!seed.physical) continue;
        const auto radial = static_cast<std::int32_t>(
            std::max(0.0, std::floor(seed.radius / dr)));
        const auto angular = static_cast<std::int32_t>(
            std::floor(
                (seed.angle < 0.0 ? seed.angle + two_pi : seed.angle)
                / dtheta));
        bool found = false;
        for (int da = -2; da <= 2 && !found; ++da) {
            for (int dr_index = -2; dr_index <= 2; ++dr_index) {
                const std::int32_t candidate_radial = radial + dr_index;
                const std::int32_t candidate_angular =
                    wrap_angle(angular + da);
                if (
                    mapped_distance_squared(
                        candidate_radial, candidate_angular)
                    <= source_radius_squared * (1.0 + 1.0e-10)) {
                    enqueue_run(
                        candidate_radial, candidate_radial,
                        candidate_angular);
                    found = true;
                    break;
                }
            }
        }
    }
    std::vector<double> left_inside_distances;
    std::vector<double> right_inside_distances;
    left_inside_distances.reserve(
        static_cast<std::size_t>(std::max(1.0, source_radius / dr)));
    right_inside_distances.reserve(left_inside_distances.capacity());
    const auto accumulate_inside = [&](
        std::int32_t radial, double distance_squared) {
        if constexpr (AccumulateMoments) {
            const double radius = (static_cast<double>(radial) + 0.5) * dr;
            const double area = radius * dr * dtheta;
            result.moments[0] += area;
            if constexpr (Mode != MomentMode::uniform) {
                const double phi =
                    1.0
                    - distance_squared * inverse_source_radius_squared;
                const double sqrt_phi = std::sqrt(phi);
                result.moments[1] += area * sqrt_phi;
                if constexpr (Mode == MomentMode::two_coefficient) {
                    result.moments[2] += area * std::sqrt(sqrt_phi);
                }
            }
        }
    };
    while (!queue.empty()) {
        const auto frontier = queue.front();
        queue.pop_front();
        const std::int32_t angular = frontier.angular;
        double cosine = 1.0;
        double sine = 0.0;
        visited.direction(angular, dtheta, cosine, sine);
        const auto column_distance_squared = [&](std::int32_t radial) {
            if (radial < 0) {
                return std::numeric_limits<double>::infinity();
            }
            const double radius =
                (static_cast<double>(radial) + 0.5) * dr;
            const auto mapped = triple_map_source(
                radius * cosine, radius * sine, lens);
            const double dx = mapped.source_x - source_x;
            const double dy = mapped.source_y - source_y;
            return dx * dx + dy * dy;
        };
        std::int64_t frontier_radial = frontier.left;
        while (frontier_radial <= frontier.right) {
            frontier_radial = first_unvisited_at_or_after(
                static_cast<std::int32_t>(frontier_radial), angular);
            if (frontier_radial > frontier.right) break;
            const double start_distance = column_distance_squared(
                static_cast<std::int32_t>(frontier_radial));
            if (start_distance > source_radius_squared) {
                ++frontier_radial;
                continue;
            }
            const std::int32_t radial =
                static_cast<std::int32_t>(frontier_radial);
            left_inside_distances.clear();
            std::int32_t left = radial;
            double left_outside = -1.0;
            while (left > 0 && !cell_visited(left - 1, angular)) {
                const double distance = column_distance_squared(left - 1);
                if (distance > source_radius_squared) {
                    left_outside = distance;
                    break;
                }
                --left;
                left_inside_distances.push_back(distance);
            }
            right_inside_distances.clear();
            std::int32_t right = radial;
            double right_outside = -1.0;
            while (!cell_visited(right + 1, angular)) {
                const double distance = column_distance_squared(right + 1);
                if (distance > source_radius_squared) {
                    right_outside = distance;
                    break;
                }
                ++right;
                right_inside_distances.push_back(distance);
            }
            const std::int64_t run_size =
                static_cast<std::int64_t>(right) - left + 1;
            if (result.cell_count + run_size > cell_capacity) {
                result.overflow = true;
                break;
            }
            add_visited(angular, left, right);
            const double left_inside = left_inside_distances.empty()
                ? start_distance
                : left_inside_distances.back();
            const double right_inside = right_inside_distances.empty()
                ? start_distance
                : right_inside_distances.back();
            result.runs.push_back({
                angular, left, right, left_inside, right_inside,
                left_outside, right_outside});
            result.cell_count += run_size;
            std::int32_t current = left;
            for (auto iterator = left_inside_distances.rbegin();
                 iterator != left_inside_distances.rend();
                 ++iterator, ++current) {
                accumulate_inside(current, *iterator);
            }
            accumulate_inside(radial, start_distance);
            current = radial + 1;
            for (const double distance : right_inside_distances) {
                accumulate_inside(current++, distance);
            }
            enqueue_run(left, right, angular - 1);
            enqueue_run(left, right, angular + 1);
            frontier_radial = std::max<std::int64_t>(
                frontier_radial + 1,
                static_cast<std::int64_t>(right) + 1);
        }
        if (result.overflow) break;
    }
    return result;
}

template <typename MakeColumnDistance>
PolarFloodSupport discover_binary_polar_flood_support(
    const CartesianSeedSupport& prepared,
    double source_radius,
    double dr,
    std::int64_t angular_bins,
    std::int64_t cell_capacity,
    MakeColumnDistance make_column_distance)
{
    PolarFloodSupport result;
    result.root_failure = prepared.root_failure;
    const double two_pi = 2.0 * std::acos(-1.0);
    const double dtheta = two_pi / angular_bins;
    PolarVisitedCellIntervals visited(angular_bins);
    struct PolarFrontier {
        std::int32_t left = 0;
        std::int32_t right = -1;
        std::int32_t angular = 0;
    };
    std::deque<PolarFrontier> queue;
    const auto wrap_angle = [angular_bins](std::int64_t index) {
        index %= angular_bins;
        if (index < 0) index += angular_bins;
        return static_cast<std::int32_t>(index);
    };
    const auto cell_visited = [&](std::int32_t radial, std::int32_t angular) {
        if (radial < 0) return true;
        const auto* intervals = visited.find(wrap_angle(angular));
        return intervals != nullptr && intervals->contains(radial);
    };
    const auto first_unvisited_at_or_after = [&](
        std::int32_t radial, std::int32_t angular) {
        const auto* intervals = visited.find(wrap_angle(angular));
        return intervals == nullptr
            ? static_cast<std::int64_t>(radial)
            : intervals->first_unvisited_at_or_after(radial);
    };
    const auto add_visited = [&](std::int32_t angular, std::int32_t left,
                                 std::int32_t right) {
        visited.get_or_create(wrap_angle(angular)).add(left, right);
    };
    const auto enqueue_run = [&](std::int32_t left, std::int32_t right,
                                 std::int32_t angular) {
        left = std::max<std::int32_t>(left, 0);
        left = first_unvisited_at_or_after(left, angular);
        if (left <= right) {
            queue.push_back({left, right, wrap_angle(angular)});
        }
    };

    // A root lies inside the continuous image, but not necessarily at the
    // centre of its quantised polar cell. Search a small resolution-scaled
    // neighbourhood and start the flood from the first centre-inside cell.
    // The centre, limb, and certificate roots are deliberately all retained:
    // redundant roots are cheap after the visited check, while omitting one
    // can lose a fold component absent at the source centre.
    for (const auto& coordinate : prepared.image_coordinates) {
        const double seed_radius = std::abs(coordinate);
        double seed_angle = std::arg(coordinate);
        if (seed_angle < 0.0) seed_angle += two_pi;
        const auto radial = static_cast<std::int32_t>(
            std::max(0.0, std::floor(seed_radius / dr)));
        const auto angular = static_cast<std::int32_t>(
            std::floor(seed_angle / dtheta));
        bool found = false;
        for (int da = -2; da <= 2 && !found; ++da) {
            const std::int32_t candidate_angular = wrap_angle(angular + da);
            double cosine = 1.0;
            double sine = 0.0;
            visited.direction(candidate_angular, dtheta, cosine, sine);
            const auto column_distance =
                make_column_distance(cosine, sine);
            for (int dr_index = -2; dr_index <= 2; ++dr_index) {
                const std::int32_t candidate_radial = radial + dr_index;
                if (candidate_radial < 0) continue;
                if (column_distance(candidate_radial)
                    <= source_radius * source_radius * (1.0 + 1.0e-10)) {
                    enqueue_run(
                        candidate_radial, candidate_radial,
                        candidate_angular);
                    found = true;
                    break;
                }
            }
        }
    }

    std::vector<double> left_inside_distances;
    std::vector<double> right_inside_distances;
    left_inside_distances.reserve(64);
    right_inside_distances.reserve(64);
    const double source_radius_squared = source_radius * source_radius;
    while (!queue.empty()) {
        const auto frontier = queue.front();
        queue.pop_front();
        const std::int32_t angular = frontier.angular;
        double cosine = 1.0;
        double sine = 0.0;
        visited.direction(angular, dtheta, cosine, sine);
        const auto column_distance = make_column_distance(cosine, sine);
        std::int64_t frontier_radial = frontier.left;
        while (frontier_radial <= frontier.right) {
            frontier_radial = first_unvisited_at_or_after(
                static_cast<std::int32_t>(frontier_radial), angular);
            if (frontier_radial > frontier.right) break;
            const double start_distance = column_distance(
                static_cast<std::int32_t>(frontier_radial));
            if (start_distance > source_radius_squared) {
                ++frontier_radial;
                continue;
            }
            const std::int32_t radial =
                static_cast<std::int32_t>(frontier_radial);
            left_inside_distances.clear();
            std::int32_t left = radial;
            double left_outside = -1.0;
            while (left > 0 && !cell_visited(left - 1, angular)) {
                const double distance = column_distance(left - 1);
                if (distance > source_radius_squared) {
                    left_outside = distance;
                    break;
                }
                --left;
                left_inside_distances.push_back(distance);
            }
            right_inside_distances.clear();
            std::int32_t right = radial;
            double right_outside = -1.0;
            while (!cell_visited(right + 1, angular)) {
                const double distance = column_distance(right + 1);
                if (distance > source_radius_squared) {
                    right_outside = distance;
                    break;
                }
                ++right;
                right_inside_distances.push_back(distance);
            }
            const std::int64_t run_size =
                static_cast<std::int64_t>(right) - left + 1;
            if (result.cell_count + run_size > cell_capacity) {
                result.overflow = true;
                break;
            }
            add_visited(angular, left, right);
            result.runs.push_back({
                angular,
                left,
                right,
                left_inside_distances.empty()
                    ? start_distance : left_inside_distances.back(),
                right_inside_distances.empty()
                    ? start_distance : right_inside_distances.back(),
                left_outside,
                right_outside});
            result.cell_count += run_size;
            enqueue_run(left, right, angular - 1);
            enqueue_run(left, right, angular + 1);
            frontier_radial = std::max<std::int64_t>(
                frontier_radial + 1,
                static_cast<std::int64_t>(right) + 1);
        }
        if (result.overflow) break;
    }
    if (result.runs.empty()) result.root_failure = true;
    return result;
}

template <MomentMode Mode, typename Scalar>
void add_polar_interior(
    std::array<Scalar, 3>& moments,
    const PhiDerivatives<Scalar>& values,
    const Scalar& delta_r,
    const Scalar& delta_theta,
    double area)
{
    moments[0] += area;
    if constexpr (Mode != MomentMode::uniform) {
        const Scalar sqrt_phi = scalar_sqrt(values.phi);
        const Scalar delta_squared =
            delta_r * delta_r + delta_theta * delta_theta;
        moments[1] += area * (
            sqrt_phi
            - delta_squared / (96.0 * values.phi * sqrt_phi));
        if constexpr (Mode == MomentMode::two_coefficient) {
            const Scalar fourth_root = scalar_sqrt(sqrt_phi);
            moments[2] += area * (
                fourth_root
                - delta_squared
                    / (128.0 * values.phi * sqrt_phi * fourth_root));
        }
    }
}

template <MomentMode Mode, typename Scalar>
void add_polar_affine(
    std::array<Scalar, 3>& moments,
    const PhiDerivatives<Scalar>& values,
    const Scalar& delta_r,
    const Scalar& delta_theta,
    double area)
{
    const Scalar lower_left =
        values.phi - 0.5 * (delta_r + delta_theta);
    constexpr std::array<double, 3> powers{0.0, 0.5, 0.25};
    for (int moment = 0; moment < moment_count(Mode); ++moment) {
        moments[moment] += area * affine_unit_square_moment(
            lower_left, delta_r, delta_theta, powers[moment]);
    }
}

template <MomentMode Mode, int BoundarySubdivision, typename Scalar>
CartesianEpochResult<Scalar> binary_polar_flood_epoch_kernel(
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& source_radius,
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    double angular_grid_ratio,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity)
{
    const double source_x_value = scalar_value(source_x);
    const double source_y_value = scalar_value(source_y);
    const double separation_value = scalar_value(separation);
    const double mass_ratio_value = scalar_value(mass_ratio);
    const double source_radius_value = scalar_value(source_radius);
    const double dr = source_radius_value / resolution;
    const double two_pi = 2.0 * std::acos(-1.0);

    const auto prepared = cached_cartesian_seed_support(
        source_x_value, source_y_value, separation_value,
        mass_ratio_value, source_radius_value, limb_samples);
    if (angular_bins == 0) {
        // A fixed angular floor is not resolution-independent for small
        // sources: the image-plane tangential cell is r*dtheta, while the
        // radial cell is rho/resolution.  The root support is already built
        // for this epoch, so use its maximum image radius to choose one
        // geometry-aware angular grid without an extra trial evaluation.
        double maximum_radius = 1.0;
        for (const auto& coordinate : prepared.image_coordinates) {
            maximum_radius = std::max(
                maximum_radius,
                std::abs(coordinate) + 4.0 * source_radius_value);
        }
        // Match the tangential cell width to the radial cell width with one
        // caller-supplied, dimensionless aspect ratio.  The ratio is a
        // numerical-policy knob, not a lens-specific branch: convergence
        // controllers can reduce it uniformly when the angular discretisation
        // is the limiting error term.
        const double polar_grid_ratio = std::max(angular_grid_ratio, 1.0);
        const double requested = std::ceil(
            two_pi * maximum_radius / (dr * polar_grid_ratio));
        constexpr std::int64_t maximum_angular_bins = 16'777'216;
        angular_bins = std::min<std::int64_t>(
            maximum_angular_bins,
            std::max<std::int64_t>(16, static_cast<std::int64_t>(requested)));
    }
    const double dtheta = two_pi / angular_bins;

    const double total_mass_value = 1.0 + mass_ratio_value;
    const LensConstants<double> classification_lens{
        -mass_ratio_value / total_mass_value * separation_value,
        separation_value / total_mass_value,
        1.0 / total_mass_value,
        mass_ratio_value / total_mass_value};
    const auto make_column_distance = [=](double cosine, double sine) {
        return [=](std::int32_t radial) {
            const double radius =
                (static_cast<double>(radial) + 0.5) * dr;
            const double image_x = radius * cosine;
            const double image_y = radius * sine;
            const double dx_1 = image_x - classification_lens.lens_1_x;
            const double dx_2 = image_x - classification_lens.lens_2_x;
            const double y_squared = image_y * image_y;
            const double radius_1_squared = dx_1 * dx_1 + y_squared;
            const double radius_2_squared = dx_2 * dx_2 + y_squared;
            const double mapped_x =
                image_x
                - classification_lens.mass_1 * dx_1 / radius_1_squared
                - classification_lens.mass_2 * dx_2 / radius_2_squared;
            const double mapped_y =
                image_y
                - classification_lens.mass_1 * image_y / radius_1_squared
                - classification_lens.mass_2 * image_y / radius_2_squared;
            const double residual_x = mapped_x - source_x_value;
            const double residual_y = mapped_y - source_y_value;
            return residual_x * residual_x + residual_y * residual_y;
        };
    };
    const auto capacity_product = [](std::int64_t left, std::int64_t right) {
        const auto maximum = std::numeric_limits<std::int64_t>::max();
        return left > maximum / right ? maximum : left * right;
    };
    const std::int64_t cell_capacity = capacity_product(
        capacity_product(angular_bins, radial_capacity), band_capacity);
    const auto support = discover_binary_polar_flood_support(
        prepared, source_radius_value, dr, angular_bins,
        cell_capacity, make_column_distance);

    CartesianEpochResult<Scalar> result;
    result.tile_count = static_cast<std::int32_t>(std::min<std::size_t>(
        support.runs.size(),
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())));
    result.overflow = support.overflow;
    result.root_failure = support.root_failure;
    const Scalar total_mass = 1.0 + mass_ratio;
    const LensConstants<Scalar> lens{
        -mass_ratio / total_mass * separation,
        separation / total_mass,
        1.0 / total_mass,
        mass_ratio / total_mass};
    const Scalar inverse_source_radius_squared =
        1.0 / (source_radius * source_radius);

    PolarVisitedCellIntervals runs_by_angle(angular_bins);
    for (const auto& run : support.runs) {
        runs_by_angle.get_or_create(run.angular_index).add(
            run.left, run.right);
    }
    const auto wrap_angle = [angular_bins](std::int64_t angular) {
        angular %= angular_bins;
        if (angular < 0) angular += angular_bins;
        return static_cast<std::int32_t>(angular);
    };
    const auto boundary_key = [&](std::int32_t radial, std::int32_t angular) {
        return (
            static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(wrap_angle(angular)))
            << 32)
            | static_cast<std::uint32_t>(radial);
    };
    std::unordered_set<std::uint64_t> boundary_candidates;
    boundary_candidates.reserve(
        static_cast<std::size_t>(std::max<std::int64_t>(
            16, 4 * static_cast<std::int64_t>(support.runs.size()))));
    for (const auto& run : support.runs) {
        if (run.left > 0) {
            boundary_candidates.insert(
                boundary_key(run.left - 1, run.angular_index));
        }
        boundary_candidates.insert(
            boundary_key(run.right + 1, run.angular_index));
        for (const int direction : {-1, 1}) {
            const std::int32_t neighbor_angle =
                wrap_angle(run.angular_index + direction);
            const auto* neighbor_intervals =
                runs_by_angle.find(neighbor_angle);
            if (neighbor_intervals == nullptr) {
                for (
                    std::int32_t radial = run.left;
                    radial <= run.right;
                    ++radial) {
                    boundary_candidates.insert(
                        boundary_key(radial, neighbor_angle));
                }
                continue;
            }
            std::int32_t cursor = run.left;
            neighbor_intervals->for_each_interval([&](const auto& interval) {
                if (cursor > run.right || interval[1] < cursor) return;
                if (interval[0] > run.right) {
                    cursor = run.right + 1;
                    return;
                }
                const std::int32_t uncovered_right =
                    std::min<std::int32_t>(run.right, interval[0] - 1);
                for (
                    std::int32_t radial = cursor;
                    radial <= uncovered_right;
                    ++radial) {
                    boundary_candidates.insert(
                        boundary_key(radial, neighbor_angle));
                }
                cursor = std::max<std::int32_t>(
                    cursor, interval[1] + 1);
            });
            for (
                std::int32_t radial = cursor;
                radial <= run.right;
                ++radial) {
                boundary_candidates.insert(
                    boundary_key(radial, neighbor_angle));
            }
        }
    }

    const std::int64_t chunk_count =
        (angular_bins + angular_chunk_size - 1) / angular_chunk_size;
    std::vector<std::int64_t> boundaries_per_chunk(
        static_cast<std::size_t>(chunk_count), 0);
    const auto integrate_cell = [&](
        std::int32_t radial, std::int32_t angular, bool centre_inside) {
        const double theta =
            (static_cast<double>(angular) + 0.5) * dtheta;
        double cosine = 1.0;
        double sine = 0.0;
        runs_by_angle.direction(angular, dtheta, cosine, sine);
        const double radius =
            (static_cast<double>(radial) + 0.5) * dr;
        const auto values = phi_derivatives<false>(
            radius * cosine, radius * sine,
            source_x, source_y, lens, inverse_source_radius_squared);
        const Scalar delta_r =
            (values.gradient_x * cosine + values.gradient_y * sine) * dr;
        const Scalar delta_theta =
            radius
            * (-values.gradient_x * sine + values.gradient_y * cosine)
            * dtheta;
        const double extent = 0.5 * (
            std::abs(scalar_value(delta_r))
            + std::abs(scalar_value(delta_theta)));
        if (centre_inside && scalar_value(values.phi) > extent) {
            add_polar_interior<Mode>(
                result.integration.moments, values,
                delta_r, delta_theta, radius * dr * dtheta);
            ++result.integration.active_cells;
            return;
        }
        if (scalar_value(values.phi) + extent <= 0.0) return;
        ++result.integration.active_cells;
        ++result.integration.boundary_cells;
        ++boundaries_per_chunk[static_cast<std::size_t>(
            angular / angular_chunk_size)];
        constexpr double subdivision =
            static_cast<double>(BoundarySubdivision);
        for (int sr = 0; sr < BoundarySubdivision; ++sr) {
            const double sub_radius =
                radius
                + ((static_cast<double>(sr) + 0.5) / subdivision - 0.5)
                    * dr;
            for (int st = 0; st < BoundarySubdivision; ++st) {
                const double sub_theta =
                    theta
                    + ((static_cast<double>(st) + 0.5) / subdivision - 0.5)
                        * dtheta;
                const double sub_cosine = std::cos(sub_theta);
                const double sub_sine = std::sin(sub_theta);
                const auto sub_values = phi_derivatives<false>(
                    sub_radius * sub_cosine,
                    sub_radius * sub_sine,
                    source_x, source_y, lens,
                    inverse_source_radius_squared);
                const Scalar sub_delta_r =
                    (sub_values.gradient_x * sub_cosine
                     + sub_values.gradient_y * sub_sine)
                    * dr / subdivision;
                const Scalar sub_delta_theta =
                    sub_radius
                    * (-sub_values.gradient_x * sub_sine
                       + sub_values.gradient_y * sub_cosine)
                    * dtheta / subdivision;
                add_polar_affine<Mode>(
                    result.integration.moments, sub_values,
                    sub_delta_r, sub_delta_theta,
                    sub_radius * dr * dtheta
                        / (subdivision * subdivision));
            }
        }
    };
    for (const auto& run : support.runs) {
        for (
            std::int32_t radial = run.left;
            radial <= run.right;
            ++radial) {
            integrate_cell(radial, run.angular_index, true);
        }
    }
    for (const auto key : boundary_candidates) {
        integrate_cell(
            static_cast<std::int32_t>(key & 0xffffffffU),
            static_cast<std::int32_t>(key >> 32), false);
    }
    for (const auto count : boundaries_per_chunk) {
        if (count > boundary_capacity) result.overflow = true;
    }
    return result;
}

template <MomentMode Mode, typename Scalar>
CartesianEpochResult<Scalar> triple_polar_flood_epoch_kernel(
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& tertiary_mass_ratio,
    const Scalar& tertiary_separation,
    const Scalar& tertiary_angle,
    const Scalar& source_radius,
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t limb_samples,
    std::int64_t convention,
    double angular_grid_ratio)
{
    // Triple-lens auto-grids retain the calibrated production aspect ratio.
    // The caller-supplied angular factor is used by the binary auto-grid, but
    // changing the triple ratio would invalidate the existing polar
    // convergence calibration for extreme mass-ratio configurations.
    (void) angular_grid_ratio;
    const double dr = scalar_value(source_radius) / resolution;
    if (angular_bins == 0) {
        const auto geometry = convention == 0
            ? lcbinint::model::make_triple_lens_geometry(
                scalar_value(separation), scalar_value(mass_ratio),
                scalar_value(tertiary_mass_ratio),
                scalar_value(tertiary_separation),
                scalar_value(tertiary_angle))
            : lcbinint::model::make_triple_lens_geometry_vbm(
                scalar_value(separation), scalar_value(mass_ratio),
                scalar_value(tertiary_separation),
                scalar_value(tertiary_angle),
                scalar_value(tertiary_mass_ratio));
        const lcbinint::magnification::PointSourceMagnifier magnifier;
        const auto candidates = magnifier.triple_image_candidates(
            geometry,
            {scalar_value(source_x), scalar_value(source_y)});
        double maximum_radius = 1.0;
        for (const auto& candidate : candidates) {
            if (!candidate.physical) continue;
            maximum_radius = std::max(
                maximum_radius,
                std::hypot(
                    candidate.position.x, candidate.position.y)
                    + 4.0 * scalar_value(source_radius));
        }
        constexpr double polar_grid_ratio = 32.0;
        angular_bins = std::max<std::int64_t>(
            16,
            static_cast<std::int64_t>(std::ceil(
                2.0 * std::acos(-1.0) * maximum_radius
                / (dr * polar_grid_ratio))));
    }
    const double dtheta = 2.0 * std::acos(-1.0) / angular_bins;
    constexpr std::int64_t cell_capacity = 50'000'000;
    const auto support = discover_triple_polar_flood_support<
        Mode, std::is_same_v<Scalar, double>>(
        scalar_value(source_x), scalar_value(source_y),
        scalar_value(separation), scalar_value(mass_ratio),
        scalar_value(tertiary_mass_ratio),
        scalar_value(tertiary_separation), scalar_value(tertiary_angle),
        scalar_value(source_radius), dr, angular_bins, limb_samples,
        cell_capacity, convention);
    CartesianEpochResult<Scalar> result;
    result.tile_count = static_cast<std::int32_t>(support.runs.size());
    result.overflow = support.overflow;
    result.root_failure = support.root_failure;
    if constexpr (std::is_same_v<Scalar, double>) {
        result.integration.moments = support.moments;
    }
    const auto lens = make_triple_lens_constants(
        separation, mass_ratio, tertiary_mass_ratio,
        tertiary_separation, tertiary_angle, convention);
    const Scalar inverse_radius_squared =
        1.0 / (source_radius * source_radius);
    const auto distance_squared = [&](
        std::int32_t radial, std::int32_t angular) {
        std::int64_t wrapped = angular % angular_bins;
        if (wrapped < 0) wrapped += angular_bins;
        const double theta =
            (static_cast<double>(wrapped) + 0.5) * dtheta;
        const double radius = (static_cast<double>(radial) + 0.5) * dr;
        const auto mapped = triple_map_source(
            Scalar(radius * std::cos(theta)),
            Scalar(radius * std::sin(theta)),
            lens);
        const Scalar dx = mapped.source_x - source_x;
        const Scalar dy = mapped.source_y - source_y;
        return dx * dx + dy * dy;
    };
    // The production flood uses centre-inside radial runs.  That discrete
    // support is sufficient for the fast primal after the radial end
    // correction below, but differentiating it omits motion through the
    // azimuthal run caps.  For the Jet-only Jacobian path, integrate every
    // inside cell with the affine level-set rule and add the one-cell outside
    // halo.  The halo supplies both radial and azimuthal boundary motion while
    // keeping the discovered topology stopped-gradient.
    std::unordered_set<std::uint64_t> derivative_boundary_candidates;
    PolarVisitedCellIntervals derivative_runs_by_angle(angular_bins);
    const auto boundary_key = [&](std::int32_t radial, std::int32_t angular) {
        std::int64_t wrapped = angular % angular_bins;
        if (wrapped < 0) wrapped += angular_bins;
        return (
            static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(wrapped))
            << 32)
            | static_cast<std::uint32_t>(radial);
    };
    const auto add_derivative_cell = [&](
        std::int32_t radial, std::int32_t angular, bool centre_inside) {
        std::int64_t wrapped = angular % angular_bins;
        if (wrapped < 0) wrapped += angular_bins;
        const double theta =
            (static_cast<double>(wrapped) + 0.5) * dtheta;
        double cosine = 1.0;
        double sine = 0.0;
        derivative_runs_by_angle.direction(
            static_cast<std::int32_t>(wrapped), dtheta, cosine, sine);
        const double radius = (static_cast<double>(radial) + 0.5) * dr;
        const auto values = triple_phi_derivatives<false>(
            radius * cosine, radius * sine,
            source_x, source_y, lens, inverse_radius_squared);
        const Scalar delta_r =
            (values.gradient_x * cosine + values.gradient_y * sine) * dr;
        const Scalar delta_theta =
            radius
            * (-values.gradient_x * sine + values.gradient_y * cosine)
            * dtheta;
        const double extent = 0.5 * (
            std::abs(scalar_value(delta_r))
            + std::abs(scalar_value(delta_theta)));
        if (centre_inside) {
            if (scalar_value(values.phi) > extent) {
                add_polar_interior<Mode>(
                    result.integration.moments, values,
                    delta_r, delta_theta, radius * dr * dtheta);
            } else {
                add_polar_affine<Mode>(
                    result.integration.moments, values,
                    delta_r, delta_theta, radius * dr * dtheta);
            }
        } else if (scalar_value(values.phi) + extent > 0.0) {
            add_polar_affine<Mode>(
                result.integration.moments, values,
                delta_r, delta_theta, radius * dr * dtheta);
            ++result.integration.boundary_cells;
        }
    };
    if constexpr (!std::is_same_v<Scalar, double>) {
        for (const auto& run : support.runs) {
            derivative_runs_by_angle.get_or_create(run.angular_index).add(
                run.left, run.right);
        }
    }
    for (const auto& run : support.runs) {
        const Scalar left_inside_distance_squared =
            run.left_inside_distance_squared;
        const Scalar right_inside_distance_squared =
            run.right_inside_distance_squared;
        if constexpr (!std::is_same_v<Scalar, double>) {
            for (
                std::int32_t radial = run.left;
                radial <= run.right;
                ++radial) {
                add_derivative_cell(radial, run.angular_index, true);
            }
        }
        if constexpr (!std::is_same_v<Scalar, double>) {
            if (run.left > 0) {
                derivative_boundary_candidates.insert(
                    boundary_key(run.left - 1, run.angular_index));
            }
            derivative_boundary_candidates.insert(
                boundary_key(run.right + 1, run.angular_index));
            for (const int direction : {-1, 1}) {
                std::int64_t neighbor_angle =
                    (run.angular_index + direction) % angular_bins;
                if (neighbor_angle < 0) neighbor_angle += angular_bins;
                const auto* neighbor_intervals =
                    derivative_runs_by_angle.find(
                        static_cast<std::int32_t>(neighbor_angle));
                std::int32_t cursor = run.left;
                if (neighbor_intervals != nullptr) {
                    neighbor_intervals->for_each_interval([&](const auto& interval) {
                        if (cursor > run.right || interval[1] < cursor) return;
                        if (interval[0] > run.right) {
                            cursor = run.right + 1;
                            return;
                        }
                    const std::int32_t uncovered_right =
                        std::min<std::int32_t>(
                            run.right, interval[0] - 1);
                    for (
                        std::int32_t radial = cursor;
                        radial <= uncovered_right;
                        ++radial) {
                        derivative_boundary_candidates.insert(
                            boundary_key(radial, neighbor_angle));
                    }
                    cursor = std::max<std::int32_t>(
                        cursor, interval[1] + 1);
                    });
                }
                for (
                    std::int32_t radial = cursor;
                    radial <= run.right;
                    ++radial) {
                    derivative_boundary_candidates.insert(
                        boundary_key(radial, neighbor_angle));
                }
            }
        }
        if constexpr (std::is_same_v<Scalar, double>) {
            if (run.left > 0 && run.left_outside_distance_squared >= 0.0) {
                const Scalar outside =
                    run.left_outside_distance_squared;
                const Scalar inside_radius =
                    scalar_sqrt(left_inside_distance_squared);
                const Scalar outside_radius = scalar_sqrt(outside);
                const Scalar fraction =
                    (source_radius - inside_radius)
                    / (outside_radius - inside_radius);
                result.integration.moments[0] +=
                    (fraction - 0.5)
                    * (static_cast<double>(run.left) * dr) * dr * dtheta;
            }
            if (run.right_outside_distance_squared >= 0.0) {
                const Scalar outside =
                    run.right_outside_distance_squared;
                const Scalar inside_radius =
                    scalar_sqrt(right_inside_distance_squared);
                const Scalar outside_radius = scalar_sqrt(outside);
                const Scalar fraction =
                    (source_radius - inside_radius)
                    / (outside_radius - inside_radius);
                result.integration.moments[0] +=
                    (fraction - 0.5)
                    * (static_cast<double>(run.right + 1) * dr)
                    * dr * dtheta;
            }
        }
        result.integration.active_cells +=
            static_cast<std::int64_t>(run.right) - run.left + 1;
        result.integration.boundary_cells +=
            static_cast<std::int64_t>(
                run.left_outside_distance_squared >= 0.0)
            + static_cast<std::int64_t>(
                run.right_outside_distance_squared >= 0.0);
    }
    if constexpr (!std::is_same_v<Scalar, double>) {
        for (const auto key : derivative_boundary_candidates) {
            const auto radial =
                static_cast<std::int32_t>(key & 0xffffffffU);
            const auto angular =
                static_cast<std::int32_t>(key >> 32);
            const Scalar candidate_distance =
                distance_squared(radial, angular);
            if (
                scalar_value(candidate_distance)
                <= scalar_value(source_radius * source_radius)) {
                continue;
            }
            add_derivative_cell(radial, angular, false);
        }
    }
    return result;
}


template <typename Scalar>
CartesianEpochResult<Scalar> polar_epoch_kernel(
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& source_radius,
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    double padding_factor,
    double angular_padding_factor,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity,
    MomentMode mode,
    std::int64_t boundary_subdivision)
{
    // Retained in the public FFI ABI for compatibility with the pure-JAX
    // band fallback. The angular factor controls the generic auto-grid
    // aspect ratio when angular_bins==0.
    (void) padding_factor;
#define LCBININT_POLAR_CASE(active_mode, subdivision) \
    return binary_polar_flood_epoch_kernel<active_mode, subdivision>( \
        source_x, source_y, separation, mass_ratio, source_radius, \
        resolution, angular_bins, radial_capacity, band_capacity, \
        limb_samples, std::max(angular_padding_factor, 1.0), \
        angular_chunk_size, boundary_capacity)
    if (mode == MomentMode::uniform) {
        if (boundary_subdivision == 1) LCBININT_POLAR_CASE(MomentMode::uniform, 1);
        if (boundary_subdivision == 2) LCBININT_POLAR_CASE(MomentMode::uniform, 2);
        if (boundary_subdivision == 3) LCBININT_POLAR_CASE(MomentMode::uniform, 3);
        LCBININT_POLAR_CASE(MomentMode::uniform, 4);
    }
    if (mode == MomentMode::linear) {
        if (boundary_subdivision == 1) LCBININT_POLAR_CASE(MomentMode::linear, 1);
        if (boundary_subdivision == 2) LCBININT_POLAR_CASE(MomentMode::linear, 2);
        if (boundary_subdivision == 3) LCBININT_POLAR_CASE(MomentMode::linear, 3);
        LCBININT_POLAR_CASE(MomentMode::linear, 4);
    }
    if (boundary_subdivision == 1) LCBININT_POLAR_CASE(MomentMode::two_coefficient, 1);
    if (boundary_subdivision == 2) LCBININT_POLAR_CASE(MomentMode::two_coefficient, 2);
    if (boundary_subdivision == 3) LCBININT_POLAR_CASE(MomentMode::two_coefficient, 3);
    LCBININT_POLAR_CASE(MomentMode::two_coefficient, 4);
#undef LCBININT_POLAR_CASE
}

template <typename Scalar>
CartesianEpochResult<Scalar> triple_polar_epoch_kernel(
    const Scalar& source_x,
    const Scalar& source_y,
    const Scalar& separation,
    const Scalar& mass_ratio,
    const Scalar& tertiary_mass_ratio,
    const Scalar& tertiary_separation,
    const Scalar& tertiary_angle,
    const Scalar& source_radius,
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    double padding_factor,
    double angular_padding_factor,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity,
    std::int64_t convention,
    MomentMode mode,
    std::int64_t boundary_subdivision)
{
#define LCBININT_TRIPLE_POLAR_CASE(active_mode, subdivision) \
    return triple_polar_flood_epoch_kernel<active_mode>( \
        source_x, source_y, separation, mass_ratio, \
        tertiary_mass_ratio, tertiary_separation, tertiary_angle, \
        source_radius, resolution, angular_bins, limb_samples, convention, \
        std::max(angular_padding_factor, 1.0))
    if (mode == MomentMode::uniform) {
        if (boundary_subdivision == 1) {
            LCBININT_TRIPLE_POLAR_CASE(MomentMode::uniform, 1);
        }
        if (boundary_subdivision == 2) {
            LCBININT_TRIPLE_POLAR_CASE(MomentMode::uniform, 2);
        }
        if (boundary_subdivision == 3) {
            LCBININT_TRIPLE_POLAR_CASE(MomentMode::uniform, 3);
        }
        LCBININT_TRIPLE_POLAR_CASE(MomentMode::uniform, 4);
    }
    if (mode == MomentMode::linear) {
        if (boundary_subdivision == 1) {
            LCBININT_TRIPLE_POLAR_CASE(MomentMode::linear, 1);
        }
        if (boundary_subdivision == 2) {
            LCBININT_TRIPLE_POLAR_CASE(MomentMode::linear, 2);
        }
        if (boundary_subdivision == 3) {
            LCBININT_TRIPLE_POLAR_CASE(MomentMode::linear, 3);
        }
        LCBININT_TRIPLE_POLAR_CASE(MomentMode::linear, 4);
    }
    if (boundary_subdivision == 1) {
        LCBININT_TRIPLE_POLAR_CASE(MomentMode::two_coefficient, 1);
    }
    if (boundary_subdivision == 2) {
        LCBININT_TRIPLE_POLAR_CASE(MomentMode::two_coefficient, 2);
    }
    if (boundary_subdivision == 3) {
        LCBININT_TRIPLE_POLAR_CASE(MomentMode::two_coefficient, 3);
    }
    LCBININT_TRIPLE_POLAR_CASE(MomentMode::two_coefficient, 4);
#undef LCBININT_TRIPLE_POLAR_CASE
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
    if (boundary_subdivision < 1 || boundary_subdivision > 8
        || (boundary_subdivision > 4 && boundary_subdivision != 8)) {
        return ffi::Error::InvalidArgument(
            "boundary_subdivision must be 1, 2, 3, 4, or 8");
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
    if (boundary_subdivision < 1 || boundary_subdivision > 8
        || (boundary_subdivision > 4 && boundary_subdivision != 8)) {
        return ffi::Error::InvalidArgument(
            "boundary_subdivision must be 1, 2, 3, 4, or 8");
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

ffi::Error validate_cartesian_epoch_arguments(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    double cell_size,
    double source_radius,
    ffi::ResultBufferR1<ffi::F64>& moments)
{
    if (
        tile_size <= 0 || tile_capacity <= 0 || limb_samples <= 0
        || mode_value < 1 || mode_value > 3) {
        return ffi::Error::InvalidArgument(
            "invalid Cartesian epoch static configuration");
    }
    if (boundary_subdivision < 1 || boundary_subdivision > 8
        || (boundary_subdivision > 4 && boundary_subdivision != 8)) {
        return ffi::Error::InvalidArgument(
            "boundary_subdivision must be 1, 2, 3, 4, or 8");
    }
    if (!(cell_size > 0.0) || !(source_radius > 0.0)) {
        return ffi::Error::InvalidArgument(
            "cell_size and source_radius must be positive");
    }
    if (
        moments->dimensions()[0]
        != moment_count(static_cast<MomentMode>(mode_value))) {
        return ffi::Error::InvalidArgument(
            "moments output has the wrong length");
    }
    return ffi::Error::Success();
}

ffi::Error cartesian_epoch_forward_ffi_impl(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
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
    ffi::ResultBufferR0<ffi::S32> visited_tiles,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::PRED> root_failure)
{
    auto validation = validate_cartesian_epoch_arguments(
        tile_size, tile_capacity, limb_samples, mode_value,
        boundary_subdivision, *cell_size.typed_data(),
        *source_radius.typed_data(), moments);
    if (validation.failure()) return validation;
    const auto mode = static_cast<MomentMode>(mode_value);
    const auto result = cartesian_epoch_kernel(
        *cell_size.typed_data(), *source_x.typed_data(),
        *source_y.typed_data(), *separation.typed_data(),
        *mass_ratio.typed_data(), *source_radius.typed_data(),
        *limb_d.typed_data(), tile_size, tile_capacity, limb_samples, mode,
        boundary_subdivision);
    for (int index = 0; index < moment_count(mode); ++index) {
        moments->typed_data()[index] = result.integration.moments[index];
    }
    *magnification->typed_data() = combine_magnification(
        result.integration.moments, *source_radius.typed_data(),
        *limb_c.typed_data(), *limb_d.typed_data(), mode);
    *boundary_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.boundary_cells);
    *active_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.active_cells);
    *visited_tiles->typed_data() = result.tile_count;
    *overflow->typed_data() = result.overflow;
    *root_failure->typed_data() = result.root_failure;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    cartesian_epoch_forward_ffi_handler,
    cartesian_epoch_forward_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("tile_size")
        .Attr<std::int64_t>("tile_capacity")
        .Attr<std::int64_t>("limb_samples")
        .Attr<std::int64_t>("moment_mode")
        .Attr<std::int64_t>("boundary_subdivision")
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
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::PRED>>()
        .Ret<ffi::BufferR0<ffi::PRED>>());

ffi::Error cartesian_epoch_value_jacobian_ffi_impl(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
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
    ffi::ResultBufferR0<ffi::S32> visited_tiles,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::PRED> root_failure,
    ffi::ResultBufferR1<ffi::F64> magnification_jacobian,
    ffi::ResultBufferR2<ffi::F64> moments_jacobian)
{
    auto validation = validate_cartesian_epoch_arguments(
        tile_size, tile_capacity, limb_samples, mode_value,
        boundary_subdivision, *cell_size.typed_data(),
        *source_radius.typed_data(), moments);
    if (validation.failure()) return validation;
    const auto mode = static_cast<MomentMode>(mode_value);
    const auto moment_jacobian_dimensions = moments_jacobian->dimensions();
    if (
        magnification_jacobian->dimensions()[0] != parameter_count
        || moment_jacobian_dimensions[0] != moment_count(mode)
        || moment_jacobian_dimensions[1] != parameter_count) {
        return ffi::Error::InvalidArgument(
            "Cartesian epoch Jacobian outputs have incorrect shapes");
    }

    const Jet source_x_jet = Jet::variable(*source_x.typed_data(), 0);
    const Jet source_y_jet = Jet::variable(*source_y.typed_data(), 1);
    const Jet separation_jet = Jet::variable(*separation.typed_data(), 2);
    const Jet mass_ratio_jet = Jet::variable(*mass_ratio.typed_data(), 3);
    const Jet source_radius_jet =
        Jet::variable(*source_radius.typed_data(), 4);
    const Jet limb_c_jet(*limb_c.typed_data());
    const Jet limb_d_jet(*limb_d.typed_data());
    const auto result = cartesian_epoch_kernel(
        *cell_size.typed_data(), source_x_jet, source_y_jet,
        separation_jet, mass_ratio_jet, source_radius_jet, limb_d_jet,
        tile_size, tile_capacity, limb_samples, mode,
        boundary_subdivision);
    const Jet magnification_result = combine_magnification(
        result.integration.moments, source_radius_jet, limb_c_jet,
        limb_d_jet, mode);
    const auto limb_derivatives = limb_coefficient_derivatives(
        result.integration.moments, *source_radius.typed_data(),
        *limb_c.typed_data(), *limb_d.typed_data(), mode);

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
        moments->typed_data()[moment] =
            result.integration.moments[moment].value;
        for (
            std::size_t parameter = 0;
            parameter < parameter_count;
            ++parameter) {
            moments_jacobian->typed_data()[
                moment * parameter_count + parameter] =
                parameter < kernel_derivative_count
                ? result.integration.moments[moment].derivative[parameter]
                : 0.0;
        }
    }
    *boundary_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.boundary_cells);
    *active_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.active_cells);
    *visited_tiles->typed_data() = result.tile_count;
    *overflow->typed_data() = result.overflow;
    *root_failure->typed_data() = result.root_failure;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    cartesian_epoch_value_jacobian_ffi_handler,
    cartesian_epoch_value_jacobian_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("tile_size")
        .Attr<std::int64_t>("tile_capacity")
        .Attr<std::int64_t>("limb_samples")
        .Attr<std::int64_t>("moment_mode")
        .Attr<std::int64_t>("boundary_subdivision")
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
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::PRED>>()
        .Ret<ffi::BufferR0<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>());

ffi::Error triple_cartesian_epoch_forward_ffi_impl(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t convention,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR0<ffi::F64> cell_size,
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR0<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> moments,
    ffi::ResultBufferR0<ffi::S32> boundary_cells,
    ffi::ResultBufferR0<ffi::S32> active_cells,
    ffi::ResultBufferR0<ffi::S32> visited_tiles,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::PRED> root_failure)
{
    auto validation = validate_cartesian_epoch_arguments(
        tile_size, tile_capacity, limb_samples, mode_value,
        boundary_subdivision, *cell_size.typed_data(),
        *source_radius.typed_data(), moments);
    if (validation.failure()) return validation;
    if (convention != 0 && convention != 1) {
        return ffi::Error::InvalidArgument("invalid triple convention");
    }
    const auto mode = static_cast<MomentMode>(mode_value);
    const auto result = triple_cartesian_epoch_kernel(
        *cell_size.typed_data(), *source_x.typed_data(),
        *source_y.typed_data(), *separation.typed_data(),
        *mass_ratio.typed_data(), *tertiary_mass_ratio.typed_data(),
        *tertiary_separation.typed_data(), *tertiary_angle.typed_data(),
        *source_radius.typed_data(), *limb_d.typed_data(), tile_size,
        tile_capacity, limb_samples, convention, mode,
        boundary_subdivision);
    for (int index = 0; index < moment_count(mode); ++index) {
        moments->typed_data()[index] = result.integration.moments[index];
    }
    *magnification->typed_data() = combine_magnification(
        result.integration.moments, *source_radius.typed_data(),
        *limb_c.typed_data(), *limb_d.typed_data(), mode);
    *boundary_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.boundary_cells);
    *active_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.active_cells);
    *visited_tiles->typed_data() = result.tile_count;
    *overflow->typed_data() = result.overflow;
    *root_failure->typed_data() = result.root_failure;
    return ffi::Error::Success();
}

ffi::Error triple_cartesian_epoch_value_jacobian_ffi_impl(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t convention,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR0<ffi::F64> cell_size,
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR0<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> moments,
    ffi::ResultBufferR0<ffi::S32> boundary_cells,
    ffi::ResultBufferR0<ffi::S32> active_cells,
    ffi::ResultBufferR0<ffi::S32> visited_tiles,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::PRED> root_failure,
    ffi::ResultBufferR1<ffi::F64> magnification_jacobian,
    ffi::ResultBufferR2<ffi::F64> moments_jacobian)
{
    auto validation = validate_cartesian_epoch_arguments(
        tile_size, tile_capacity, limb_samples, mode_value,
        boundary_subdivision, *cell_size.typed_data(),
        *source_radius.typed_data(), moments);
    if (validation.failure()) return validation;
    const auto mode = static_cast<MomentMode>(mode_value);
    if (
        (convention != 0 && convention != 1)
        || magnification_jacobian->dimensions()[0]
            != triple_parameter_count
        || moments_jacobian->dimensions()[0] != moment_count(mode)
        || moments_jacobian->dimensions()[1] != triple_parameter_count) {
        return ffi::Error::InvalidArgument(
            "invalid triple Cartesian Jacobian configuration");
    }
    const TripleJet source_x_jet =
        TripleJet::variable(*source_x.typed_data(), 0);
    const TripleJet source_y_jet =
        TripleJet::variable(*source_y.typed_data(), 1);
    const TripleJet separation_jet =
        TripleJet::variable(*separation.typed_data(), 2);
    const TripleJet mass_ratio_jet =
        TripleJet::variable(*mass_ratio.typed_data(), 3);
    const TripleJet tertiary_mass_ratio_jet =
        TripleJet::variable(*tertiary_mass_ratio.typed_data(), 4);
    const TripleJet tertiary_separation_jet =
        TripleJet::variable(*tertiary_separation.typed_data(), 5);
    const TripleJet tertiary_angle_jet =
        TripleJet::variable(*tertiary_angle.typed_data(), 6);
    const TripleJet source_radius_jet =
        TripleJet::variable(*source_radius.typed_data(), 7);
    const TripleJet limb_c_jet(*limb_c.typed_data());
    const TripleJet limb_d_jet(*limb_d.typed_data());
    const auto result = triple_cartesian_epoch_kernel(
        *cell_size.typed_data(), source_x_jet, source_y_jet,
        separation_jet, mass_ratio_jet, tertiary_mass_ratio_jet,
        tertiary_separation_jet, tertiary_angle_jet, source_radius_jet,
        limb_d_jet, tile_size, tile_capacity, limb_samples, convention,
        mode, boundary_subdivision);
    const TripleJet magnification_result = combine_magnification(
        result.integration.moments, source_radius_jet,
        limb_c_jet, limb_d_jet, mode);
    const auto limb_derivatives = limb_coefficient_derivatives(
        result.integration.moments, *source_radius.typed_data(),
        *limb_c.typed_data(), *limb_d.typed_data(), mode);
    *magnification->typed_data() = magnification_result.value;
    for (
        std::size_t parameter = 0;
        parameter < triple_kernel_derivative_count;
        ++parameter) {
        magnification_jacobian->typed_data()[parameter] =
            magnification_result.derivative[parameter];
    }
    magnification_jacobian->typed_data()[8] = limb_derivatives[0];
    magnification_jacobian->typed_data()[9] = limb_derivatives[1];
    for (int moment = 0; moment < moment_count(mode); ++moment) {
        moments->typed_data()[moment] =
            result.integration.moments[moment].value;
        for (
            std::size_t parameter = 0;
            parameter < triple_parameter_count;
            ++parameter) {
            moments_jacobian->typed_data()[
                moment * triple_parameter_count + parameter] =
                parameter < triple_kernel_derivative_count
                ? result.integration.moments[moment].derivative[parameter]
                : 0.0;
        }
    }
    *boundary_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.boundary_cells);
    *active_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.active_cells);
    *visited_tiles->typed_data() = result.tile_count;
    *overflow->typed_data() = result.overflow;
    *root_failure->typed_data() = result.root_failure;
    return ffi::Error::Success();
}

#define LCBININT_TRIPLE_CARTESIAN_FFI_BINDING \
    ffi::Ffi::Bind() \
        .Attr<std::int64_t>("tile_size") \
        .Attr<std::int64_t>("tile_capacity") \
        .Attr<std::int64_t>("limb_samples") \
        .Attr<std::int64_t>("convention") \
        .Attr<std::int64_t>("moment_mode") \
        .Attr<std::int64_t>("boundary_subdivision") \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR0<ffi::S32>>() \
        .Ret<ffi::BufferR0<ffi::S32>>() \
        .Ret<ffi::BufferR0<ffi::S32>>() \
        .Ret<ffi::BufferR0<ffi::PRED>>() \
        .Ret<ffi::BufferR0<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    triple_cartesian_epoch_forward_ffi_handler,
    triple_cartesian_epoch_forward_ffi_impl,
    LCBININT_TRIPLE_CARTESIAN_FFI_BINDING);

XLA_FFI_DEFINE_HANDLER(
    triple_cartesian_epoch_value_jacobian_ffi_handler,
    triple_cartesian_epoch_value_jacobian_ffi_impl,
    LCBININT_TRIPLE_CARTESIAN_FFI_BINDING
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>());

#undef LCBININT_TRIPLE_CARTESIAN_FFI_BINDING

ffi::Error validate_triple_cartesian_batch(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t convention,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR1<ffi::F64>& source_x,
    ffi::BufferR1<ffi::F64>& source_y,
    ffi::BufferR1<ffi::PRED>& active,
    ffi::ResultBufferR1<ffi::F64>& magnification,
    ffi::ResultBufferR2<ffi::F64>& moments)
{
    const auto batch_size = source_x.dimensions()[0];
    if (
        source_y.dimensions()[0] != batch_size
        || active.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || moments->dimensions()[0] != batch_size
        || mode_value < 1 || mode_value > 3
        || moments->dimensions()[1]
            != moment_count(static_cast<MomentMode>(mode_value))
        || tile_size <= 0 || tile_capacity <= 0 || limb_samples <= 0
        || (convention != 0 && convention != 1)
        || boundary_subdivision < 1 || boundary_subdivision > 8
        || (boundary_subdivision > 4 && boundary_subdivision != 8)) {
        return ffi::Error::InvalidArgument(
            "invalid triple Cartesian batch configuration");
    }
    return ffi::Error::Success();
}

ffi::Error triple_cartesian_batch_forward_ffi_impl(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t convention,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> cell_size,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR2<ffi::F64> moments,
    ffi::ResultBufferR1<ffi::S32> boundary_cells,
    ffi::ResultBufferR1<ffi::S32> active_cells,
    ffi::ResultBufferR1<ffi::S32> visited_tiles,
    ffi::ResultBufferR1<ffi::PRED> overflow,
    ffi::ResultBufferR1<ffi::PRED> root_failure)
{
    auto validation = validate_triple_cartesian_batch(
        tile_size, tile_capacity, limb_samples, convention, mode_value,
        boundary_subdivision, source_x, source_y, active,
        magnification, moments);
    if (validation.failure()) return validation;
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        boundary_cells->dimensions()[0] != batch_size
        || active_cells->dimensions()[0] != batch_size
        || visited_tiles->dimensions()[0] != batch_size
        || overflow->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "invalid triple Cartesian batch diagnostic shapes");
    }
    const auto mode = static_cast<MomentMode>(mode_value);
    const int output_moments = moment_count(mode);
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1, std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        auto* moment_output =
            moments->typed_data() + index * output_moments;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            std::fill(moment_output, moment_output + output_moments, 0.0);
            boundary_cells->typed_data()[index] = 0;
            active_cells->typed_data()[index] = 0;
            visited_tiles->typed_data()[index] = 0;
            overflow->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            continue;
        }
        const auto result = triple_cartesian_epoch_kernel(
            *cell_size.typed_data(), source_x.typed_data()[index],
            source_y.typed_data()[index], *separation.typed_data(),
            *mass_ratio.typed_data(), *tertiary_mass_ratio.typed_data(),
            *tertiary_separation.typed_data(), *tertiary_angle.typed_data(),
            *source_radius.typed_data(), *limb_d.typed_data(), tile_size,
            tile_capacity, limb_samples, convention, mode,
            boundary_subdivision);
        for (int moment = 0; moment < output_moments; ++moment) {
            moment_output[moment] = result.integration.moments[moment];
        }
        magnification->typed_data()[index] = combine_magnification(
            result.integration.moments, *source_radius.typed_data(),
            *limb_c.typed_data(), *limb_d.typed_data(), mode);
        boundary_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.boundary_cells);
        active_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.active_cells);
        visited_tiles->typed_data()[index] = result.tile_count;
        overflow->typed_data()[index] = result.overflow;
        root_failure->typed_data()[index] = result.root_failure;
    }
    return ffi::Error::Success();
}

ffi::Error triple_cartesian_batch_value_jacobian_ffi_impl(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t convention,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> cell_size,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR2<ffi::F64> moments,
    ffi::ResultBufferR1<ffi::S32> boundary_cells,
    ffi::ResultBufferR1<ffi::S32> active_cells,
    ffi::ResultBufferR1<ffi::S32> visited_tiles,
    ffi::ResultBufferR1<ffi::PRED> overflow,
    ffi::ResultBufferR1<ffi::PRED> root_failure,
    ffi::ResultBufferR2<ffi::F64> magnification_jacobian,
    ffi::ResultBufferR3<ffi::F64> moments_jacobian)
{
    auto validation = validate_triple_cartesian_batch(
        tile_size, tile_capacity, limb_samples, convention, mode_value,
        boundary_subdivision, source_x, source_y, active,
        magnification, moments);
    if (validation.failure()) return validation;
    const std::int64_t batch_size = source_x.dimensions()[0];
    const auto mode = static_cast<MomentMode>(mode_value);
    const int output_moments = moment_count(mode);
    if (
        magnification_jacobian->dimensions()[0] != batch_size
        || magnification_jacobian->dimensions()[1]
            != triple_parameter_count
        || moments_jacobian->dimensions()[0] != batch_size
        || moments_jacobian->dimensions()[1] != output_moments
        || moments_jacobian->dimensions()[2] != triple_parameter_count) {
        return ffi::Error::InvalidArgument(
            "invalid triple Cartesian batch Jacobian shapes");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1, std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        auto* moment_output =
            moments->typed_data() + index * output_moments;
        auto* mag_jacobian = magnification_jacobian->typed_data()
            + index * triple_parameter_count;
        auto* moment_jacobian = moments_jacobian->typed_data()
            + index * output_moments * triple_parameter_count;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            std::fill(moment_output, moment_output + output_moments, 0.0);
            std::fill(
                mag_jacobian,
                mag_jacobian + triple_parameter_count, 0.0);
            std::fill(
                moment_jacobian,
                moment_jacobian
                    + output_moments * triple_parameter_count,
                0.0);
            boundary_cells->typed_data()[index] = 0;
            active_cells->typed_data()[index] = 0;
            visited_tiles->typed_data()[index] = 0;
            overflow->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            continue;
        }
        const TripleJet source_x_jet =
            TripleJet::variable(source_x.typed_data()[index], 0);
        const TripleJet source_y_jet =
            TripleJet::variable(source_y.typed_data()[index], 1);
        const TripleJet separation_jet =
            TripleJet::variable(*separation.typed_data(), 2);
        const TripleJet mass_ratio_jet =
            TripleJet::variable(*mass_ratio.typed_data(), 3);
        const TripleJet tertiary_mass_ratio_jet =
            TripleJet::variable(*tertiary_mass_ratio.typed_data(), 4);
        const TripleJet tertiary_separation_jet =
            TripleJet::variable(*tertiary_separation.typed_data(), 5);
        const TripleJet tertiary_angle_jet =
            TripleJet::variable(*tertiary_angle.typed_data(), 6);
        const TripleJet source_radius_jet =
            TripleJet::variable(*source_radius.typed_data(), 7);
        const TripleJet limb_c_jet(*limb_c.typed_data());
        const TripleJet limb_d_jet(*limb_d.typed_data());
        const auto result = triple_cartesian_epoch_kernel(
            *cell_size.typed_data(), source_x_jet, source_y_jet,
            separation_jet, mass_ratio_jet, tertiary_mass_ratio_jet,
            tertiary_separation_jet, tertiary_angle_jet,
            source_radius_jet, limb_d_jet, tile_size, tile_capacity,
            limb_samples, convention, mode, boundary_subdivision);
        const auto magnification_result = combine_magnification(
            result.integration.moments, source_radius_jet,
            limb_c_jet, limb_d_jet, mode);
        const auto limb_derivatives = limb_coefficient_derivatives(
            result.integration.moments, *source_radius.typed_data(),
            *limb_c.typed_data(), *limb_d.typed_data(), mode);
        magnification->typed_data()[index] = magnification_result.value;
        for (
            std::size_t parameter = 0;
            parameter < triple_kernel_derivative_count;
            ++parameter) {
            mag_jacobian[parameter] =
                magnification_result.derivative[parameter];
        }
        mag_jacobian[8] = limb_derivatives[0];
        mag_jacobian[9] = limb_derivatives[1];
        for (int moment = 0; moment < output_moments; ++moment) {
            moment_output[moment] =
                result.integration.moments[moment].value;
            for (
                std::size_t parameter = 0;
                parameter < triple_parameter_count;
                ++parameter) {
                moment_jacobian[
                    moment * triple_parameter_count + parameter] =
                    parameter < triple_kernel_derivative_count
                    ? result.integration.moments[moment]
                          .derivative[parameter]
                    : 0.0;
            }
        }
        boundary_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.boundary_cells);
        active_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.active_cells);
        visited_tiles->typed_data()[index] = result.tile_count;
        overflow->typed_data()[index] = result.overflow;
        root_failure->typed_data()[index] = result.root_failure;
    }
    return ffi::Error::Success();
}

#define LCBININT_TRIPLE_CARTESIAN_BATCH_BINDING \
    ffi::Ffi::Bind() \
        .Attr<std::int64_t>("tile_size") \
        .Attr<std::int64_t>("tile_capacity") \
        .Attr<std::int64_t>("limb_samples") \
        .Attr<std::int64_t>("convention") \
        .Attr<std::int64_t>("moment_mode") \
        .Attr<std::int64_t>("boundary_subdivision") \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::PRED>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR2<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::S32>>() \
        .Ret<ffi::BufferR1<ffi::S32>>() \
        .Ret<ffi::BufferR1<ffi::S32>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    triple_cartesian_batch_forward_ffi_handler,
    triple_cartesian_batch_forward_ffi_impl,
    LCBININT_TRIPLE_CARTESIAN_BATCH_BINDING);

XLA_FFI_DEFINE_HANDLER(
    triple_cartesian_batch_value_jacobian_ffi_handler,
    triple_cartesian_batch_value_jacobian_ffi_impl,
    LCBININT_TRIPLE_CARTESIAN_BATCH_BINDING
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR3<ffi::F64>>());

#undef LCBININT_TRIPLE_CARTESIAN_BATCH_BINDING

ffi::Error triple_caustic_distance_batch_ffi_impl(
    std::int64_t convention,
    std::int64_t caustic_bins,
    double refine_factor,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::ResultBufferR1<ffi::F64> distances)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        (convention != 0 && convention != 1)
        || caustic_bins < 64 || refine_factor < 0.0
        || source_y.dimensions()[0] != batch_size
        || distances->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "invalid triple caustic-distance configuration");
    }
    const auto geometry = convention == 0
        ? lcbinint::model::make_triple_lens_geometry(
            *separation.typed_data(), *mass_ratio.typed_data(),
            *tertiary_mass_ratio.typed_data(),
            *tertiary_separation.typed_data(),
            *tertiary_angle.typed_data())
        : lcbinint::model::make_triple_lens_geometry_vbm(
            *separation.typed_data(), *mass_ratio.typed_data(),
            *tertiary_separation.typed_data(),
            *tertiary_angle.typed_data(),
            *tertiary_mass_ratio.typed_data());
    lcbinint::magnification::FiniteSourceSettings settings;
    settings.caustic_bins = static_cast<int>(caustic_bins);
    const lcbinint::magnification::FiniteSourceMagnifier magnifier(settings);
    const double refine_within =
        refine_factor * std::abs(*source_radius.typed_data());
    for (std::int64_t index = 0; index < batch_size; ++index) {
        distances->typed_data()[index] =
            magnifier.triple_caustic_distance_for_source(
            geometry,
            {source_x.typed_data()[index], source_y.typed_data()[index]},
            refine_within);
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    triple_caustic_distance_batch_ffi_handler,
    triple_caustic_distance_batch_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("convention")
        .Attr<std::int64_t>("caustic_bins")
        .Attr<double>("refine_factor")
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>());

ffi::Error binary_caustic_distance_batch_ffi_impl(
    std::int64_t caustic_bins,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::ResultBufferR1<ffi::F64> distances)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        caustic_bins < 64
        || source_y.dimensions()[0] != batch_size
        || distances->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "invalid binary caustic-distance configuration");
    }
    lcbinint::magnification::FiniteSourceSettings settings;
    settings.caustic_bins = static_cast<int>(caustic_bins);
    // One local magnifier owns one mutable caustic cache.  Keep this loop
    // sequential: sharing that cache across OpenMP workers is not safe.
    const lcbinint::magnification::FiniteSourceMagnifier magnifier(settings);
    for (std::int64_t index = 0; index < batch_size; ++index) {
        distances->typed_data()[index] =
            magnifier.binary_caustic_distance_for_source(
                *separation.typed_data(),
                *mass_ratio.typed_data(),
                {source_x.typed_data()[index], source_y.typed_data()[index]});
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    binary_caustic_distance_batch_ffi_handler,
    binary_caustic_distance_batch_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("caustic_bins")
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>());

ffi::Error binary_routing_diagnostics_batch_ffi_impl(
    std::int64_t caustic_bins,
    double hex_threshold,
    double adaptive_hex_threshold,
    double kinji_threshold,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::F64> point_magnification,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> absolute_tolerance,
    ffi::BufferR0<ffi::F64> relative_tolerance,
    ffi::ResultBufferR2<ffi::F64> floating_diagnostics,
    ffi::ResultBufferR2<ffi::S32> integer_diagnostics,
    ffi::ResultBufferR2<ffi::PRED> routing_flags)
{
    constexpr std::int64_t floating_count = 9;
    constexpr std::int64_t integer_count = 3;
    constexpr std::int64_t flag_count = 8;
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        caustic_bins < 64 || hex_threshold < 0.0
        || adaptive_hex_threshold < 0.0 || kinji_threshold < 0.0
        || source_y.dimensions()[0] != batch_size
        || point_magnification.dimensions()[0] != batch_size
        || floating_diagnostics->dimensions()[0] != batch_size
        || floating_diagnostics->dimensions()[1] != floating_count
        || integer_diagnostics->dimensions()[0] != batch_size
        || integer_diagnostics->dimensions()[1] != integer_count
        || routing_flags->dimensions()[0] != batch_size
        || routing_flags->dimensions()[1] != flag_count) {
        return ffi::Error::InvalidArgument(
            "invalid binary routing-diagnostics configuration");
    }
    lcbinint::magnification::FiniteSourceSettings settings;
    settings.caustic_bins = static_cast<int>(caustic_bins);
    settings.hex_threshold = hex_threshold;
    settings.adaptive_hex_threshold = adaptive_hex_threshold;
    settings.kinji_threshold = kinji_threshold;
    settings.finite_source_tol = *absolute_tolerance.typed_data();
    settings.finite_source_reltol = *relative_tolerance.typed_data();
    const lcbinint::magnification::FiniteSourceMagnifier magnifier(settings);
    const lcbinint::magnification::PointSourceMagnifier point_magnifier;
    for (std::int64_t index = 0; index < batch_size; ++index) {
        const auto diagnostic =
            magnifier.binary_routing_diagnostics_for_source(
                *separation.typed_data(),
                *mass_ratio.typed_data(),
                {source_x.typed_data()[index], source_y.typed_data()[index]},
                *source_radius.typed_data(),
                point_magnification.typed_data()[index],
                &point_magnifier);
        double* floating =
            floating_diagnostics->typed_data() + index * floating_count;
        floating[0] = diagnostic.point_magnification;
        floating[1] = diagnostic.point_error_estimate;
        floating[2] = diagnostic.point_absolute_tolerance;
        floating[3] = diagnostic.caustic_distance;
        floating[4] = diagnostic.scan_min_distance;
        floating[5] = diagnostic.quadrupole_indicator;
        floating[6] = diagnostic.cusp_indicator;
        floating[7] = diagnostic.ghost_indicator;
        floating[8] = diagnostic.planetary_distance2;
        std::int32_t* integers =
            integer_diagnostics->typed_data() + index * integer_count;
        integers[0] = diagnostic.image_count;
        integers[1] = diagnostic.ghost_count;
        integers[2] = diagnostic.safety_flags;
        bool* flags = routing_flags->typed_data() + index * flag_count;
        flags[0] = diagnostic.point_preflight_safe;
        flags[1] = diagnostic.point_safe;
        flags[2] = diagnostic.scan_performed;
        flags[3] = diagnostic.any_vertex_inside;
        flags[4] = diagnostic.has_crossing_probes;
        flags[5] = diagnostic.chord_band;
        flags[6] = diagnostic.tangent_band;
        flags[7] = diagnostic.grazing_ring_band;
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    binary_routing_diagnostics_batch_ffi_handler,
    binary_routing_diagnostics_batch_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("caustic_bins")
        .Attr<double>("hex_threshold")
        .Attr<double>("adaptive_hex_threshold")
        .Attr<double>("kinji_threshold")
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR2<ffi::PRED>>());

ffi::Error polar_epoch_forward_ffi_impl(
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    double padding_factor,
    double angular_padding_factor,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
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
    ffi::ResultBufferR0<ffi::S32> band_count,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::PRED> root_failure)
{
    if (
        resolution <= 0 || angular_bins < 0 || radial_capacity <= 0
        || band_capacity <= 0 || limb_samples <= 0
        || angular_chunk_size <= 0 || boundary_capacity <= 0
        || mode_value < 1 || mode_value > 3
        || boundary_subdivision < 1 || boundary_subdivision > 4
        || moments->dimensions()[0]
            != moment_count(static_cast<MomentMode>(mode_value))) {
        return ffi::Error::InvalidArgument("invalid polar epoch configuration");
    }
    const auto mode = static_cast<MomentMode>(mode_value);
    const auto result = polar_epoch_kernel(
        *source_x.typed_data(), *source_y.typed_data(),
        *separation.typed_data(), *mass_ratio.typed_data(),
        *source_radius.typed_data(), resolution, angular_bins,
        radial_capacity, band_capacity, limb_samples, padding_factor,
        angular_padding_factor, angular_chunk_size, boundary_capacity,
        mode, boundary_subdivision);
    for (int moment = 0; moment < moment_count(mode); ++moment) {
        moments->typed_data()[moment] = result.integration.moments[moment];
    }
    *magnification->typed_data() = combine_magnification(
        result.integration.moments, *source_radius.typed_data(),
        *limb_c.typed_data(), *limb_d.typed_data(), mode);
    *boundary_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.boundary_cells);
    *active_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.active_cells);
    *band_count->typed_data() = result.tile_count;
    *overflow->typed_data() = result.overflow;
    *root_failure->typed_data() = result.root_failure;
    return ffi::Error::Success();
}

#define LCBININT_POLAR_FFI_BINDING \
    ffi::Ffi::Bind() \
        .Attr<std::int64_t>("resolution") \
        .Attr<std::int64_t>("angular_bins") \
        .Attr<std::int64_t>("radial_capacity") \
        .Attr<std::int64_t>("band_capacity") \
        .Attr<std::int64_t>("limb_samples") \
        .Attr<double>("padding_factor") \
        .Attr<double>("angular_padding_factor") \
        .Attr<std::int64_t>("angular_chunk_size") \
        .Attr<std::int64_t>("boundary_capacity") \
        .Attr<std::int64_t>("moment_mode") \
        .Attr<std::int64_t>("boundary_subdivision") \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR0<ffi::S32>>() \
        .Ret<ffi::BufferR0<ffi::S32>>() \
        .Ret<ffi::BufferR0<ffi::S32>>() \
        .Ret<ffi::BufferR0<ffi::PRED>>() \
        .Ret<ffi::BufferR0<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    polar_epoch_forward_ffi_handler,
    polar_epoch_forward_ffi_impl,
    LCBININT_POLAR_FFI_BINDING);

ffi::Error polar_epoch_jacobian_ffi_impl(
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    double padding_factor,
    double angular_padding_factor,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
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
    ffi::ResultBufferR0<ffi::S32> band_count,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::PRED> root_failure,
    ffi::ResultBufferR1<ffi::F64> magnification_jacobian,
    ffi::ResultBufferR2<ffi::F64> moments_jacobian)
{
    const auto mode = static_cast<MomentMode>(mode_value);
    if (
        mode_value < 1 || mode_value > 3
        || moments->dimensions()[0] != moment_count(mode)
        || magnification_jacobian->dimensions()[0] != parameter_count
        || moments_jacobian->dimensions()[0] != moment_count(mode)
        || moments_jacobian->dimensions()[1] != parameter_count) {
        return ffi::Error::InvalidArgument(
            "invalid polar epoch Jacobian shapes");
    }
    const Jet x = Jet::variable(*source_x.typed_data(), 0);
    const Jet y = Jet::variable(*source_y.typed_data(), 1);
    const Jet s = Jet::variable(*separation.typed_data(), 2);
    const Jet q = Jet::variable(*mass_ratio.typed_data(), 3);
    const Jet rho = Jet::variable(*source_radius.typed_data(), 4);
    const auto result = polar_epoch_kernel(
        x, y, s, q, rho, resolution, angular_bins, radial_capacity,
        band_capacity, limb_samples, padding_factor,
        angular_padding_factor, angular_chunk_size, boundary_capacity,
        mode, boundary_subdivision);
    const Jet c(*limb_c.typed_data());
    const Jet d(*limb_d.typed_data());
    const Jet active_magnification = combine_magnification(
        result.integration.moments, rho, c, d, mode);
    const auto limb_derivatives = limb_coefficient_derivatives(
        result.integration.moments, rho.value, c.value, d.value, mode);
    *magnification->typed_data() = active_magnification.value;
    for (
        std::size_t parameter = 0;
        parameter < kernel_derivative_count;
        ++parameter) {
        magnification_jacobian->typed_data()[parameter] =
            active_magnification.derivative[parameter];
    }
    magnification_jacobian->typed_data()[5] = limb_derivatives[0];
    magnification_jacobian->typed_data()[6] = limb_derivatives[1];
    for (int moment = 0; moment < moment_count(mode); ++moment) {
        moments->typed_data()[moment] =
            result.integration.moments[moment].value;
        for (
            std::size_t parameter = 0;
            parameter < parameter_count;
            ++parameter) {
            moments_jacobian->typed_data()[
                moment * parameter_count + parameter] =
                parameter < kernel_derivative_count
                ? result.integration.moments[moment]
                      .derivative[parameter]
                : 0.0;
        }
    }
    *boundary_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.boundary_cells);
    *active_cells->typed_data() =
        static_cast<std::int32_t>(result.integration.active_cells);
    *band_count->typed_data() = result.tile_count;
    *overflow->typed_data() = result.overflow;
    *root_failure->typed_data() = result.root_failure;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    polar_epoch_jacobian_ffi_handler,
    polar_epoch_jacobian_ffi_impl,
    LCBININT_POLAR_FFI_BINDING
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>());

// A trajectory JVP has one directional tangent, but the ordinary polar
// Jacobian above computes five kernel derivatives (plus two limb coefficients)
// for every active cell and contracts them afterwards.  That is needlessly
// expensive for the dominant dA/dt use case.  Keep the same stopped-gradient
// support and integration algorithm, but propagate one directional derivative
// through the FFI kernel directly.
ffi::Error polar_epoch_directional_ffi_impl(
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    double padding_factor,
    double angular_padding_factor,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR0<ffi::F64> source_x,
    ffi::BufferR0<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::BufferR0<ffi::F64> source_x_tangent,
    ffi::BufferR0<ffi::F64> source_y_tangent,
    ffi::BufferR0<ffi::F64> separation_tangent,
    ffi::BufferR0<ffi::F64> mass_ratio_tangent,
    ffi::BufferR0<ffi::F64> source_radius_tangent,
    ffi::BufferR0<ffi::F64> limb_c_tangent,
    ffi::BufferR0<ffi::F64> limb_d_tangent,
    ffi::ResultBufferR0<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> moments,
    ffi::ResultBufferR0<ffi::S32> boundary_cells,
    ffi::ResultBufferR0<ffi::S32> active_cells,
    ffi::ResultBufferR0<ffi::S32> band_count,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::PRED> root_failure,
    ffi::ResultBufferR0<ffi::F64> directional_magnification,
    ffi::ResultBufferR1<ffi::F64> directional_moments)
{
    if (
        resolution <= 0 || angular_bins < 0 || radial_capacity <= 0
        || band_capacity <= 0 || limb_samples <= 0
        || angular_chunk_size <= 0 || boundary_capacity <= 0
        || mode_value < 1 || mode_value > 3
        || boundary_subdivision < 1 || boundary_subdivision > 4
        || moments->dimensions()[0] != moment_count(
            static_cast<MomentMode>(mode_value))
        || directional_moments->dimensions()[0] != moment_count(
            static_cast<MomentMode>(mode_value))) {
        return ffi::Error::InvalidArgument(
            "invalid polar epoch directional configuration");
    }
    const auto make_directional = [](double value, double tangent) {
        DirectionalJet result(value);
        result.derivative[0] = tangent;
        return result;
    };
    const auto mode = static_cast<MomentMode>(mode_value);
    const auto result = polar_epoch_kernel(
        make_directional(
            *source_x.typed_data(), *source_x_tangent.typed_data()),
        make_directional(
            *source_y.typed_data(), *source_y_tangent.typed_data()),
        make_directional(
            *separation.typed_data(), *separation_tangent.typed_data()),
        make_directional(
            *mass_ratio.typed_data(), *mass_ratio_tangent.typed_data()),
        make_directional(
            *source_radius.typed_data(), *source_radius_tangent.typed_data()),
        resolution, angular_bins, radial_capacity, band_capacity,
        limb_samples, padding_factor, angular_padding_factor,
        angular_chunk_size, boundary_capacity, mode, boundary_subdivision);
    const auto active_magnification = combine_magnification(
        result.integration.moments,
        make_directional(*source_radius.typed_data(), *source_radius_tangent.typed_data()),
        make_directional(*limb_c.typed_data(), *limb_c_tangent.typed_data()),
        make_directional(*limb_d.typed_data(), *limb_d_tangent.typed_data()),
        mode);
    *magnification->typed_data() = active_magnification.value;
    *directional_magnification->typed_data() =
        active_magnification.derivative[0];
    for (int moment = 0; moment < moment_count(mode); ++moment) {
        moments->typed_data()[moment] =
            result.integration.moments[moment].value;
        directional_moments->typed_data()[moment] =
            result.integration.moments[moment].derivative[0];
    }
    *boundary_cells->typed_data() = static_cast<std::int32_t>(
        result.integration.boundary_cells);
    *active_cells->typed_data() = static_cast<std::int32_t>(
        result.integration.active_cells);
    *band_count->typed_data() = result.tile_count;
    *overflow->typed_data() = result.overflow;
    *root_failure->typed_data() = result.root_failure;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    polar_epoch_directional_ffi_handler,
    polar_epoch_directional_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("resolution")
        .Attr<std::int64_t>("angular_bins")
        .Attr<std::int64_t>("radial_capacity")
        .Attr<std::int64_t>("band_capacity")
        .Attr<std::int64_t>("limb_samples")
        .Attr<double>("padding_factor")
        .Attr<double>("angular_padding_factor")
        .Attr<std::int64_t>("angular_chunk_size")
        .Attr<std::int64_t>("boundary_capacity")
        .Attr<std::int64_t>("moment_mode")
        .Attr<std::int64_t>("boundary_subdivision")
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
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
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::PRED>>()
        .Ret<ffi::BufferR0<ffi::PRED>>()
        .Ret<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>());

#undef LCBININT_POLAR_FFI_BINDING

ffi::Error validate_triple_polar_batch(
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    std::int64_t convention,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::ResultBufferR2<ffi::F64>& moments)
{
    const auto dimensions = moments->dimensions();
    if (
        resolution <= 0 || angular_bins < 0 || radial_capacity <= 0
        || band_capacity <= 0 || limb_samples <= 0
        || angular_chunk_size <= 0 || boundary_capacity <= 0
        || mode_value < 1 || mode_value > 3
        || boundary_subdivision < 1 || boundary_subdivision > 4
        || (convention != 0 && convention != 1)
        || source_y.dimensions()[0] != source_x.dimensions()[0]
        || active.dimensions()[0] != source_x.dimensions()[0]
        || dimensions[0] != source_x.dimensions()[0]
        || dimensions[1]
            != moment_count(static_cast<MomentMode>(mode_value))) {
        return ffi::Error::InvalidArgument(
            "invalid triple polar batch configuration");
    }
    return ffi::Error::Success();
}

ffi::Error triple_polar_batch_forward_ffi_impl(
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    double padding_factor,
    double angular_padding_factor,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    std::int64_t convention,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR2<ffi::F64> moments,
    ffi::ResultBufferR1<ffi::S32> boundary_cells,
    ffi::ResultBufferR1<ffi::S32> active_cells,
    ffi::ResultBufferR1<ffi::S32> band_count,
    ffi::ResultBufferR1<ffi::PRED> overflow,
    ffi::ResultBufferR1<ffi::PRED> root_failure)
{
    auto validation = validate_triple_polar_batch(
        resolution, angular_bins, radial_capacity, band_capacity,
        limb_samples, angular_chunk_size, boundary_capacity, mode_value,
        boundary_subdivision, convention, source_x, source_y, active, moments);
    if (validation.failure()) return validation;
    const auto mode = static_cast<MomentMode>(mode_value);
    const std::int64_t batch_size = source_x.dimensions()[0];
    const int moment_size = moment_count(mode);
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1,
        std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        double* output_moments =
            moments->typed_data() + index * moment_size;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            std::fill(output_moments, output_moments + moment_size, 0.0);
            boundary_cells->typed_data()[index] = 0;
            active_cells->typed_data()[index] = 0;
            band_count->typed_data()[index] = 0;
            overflow->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            continue;
        }
        const auto result = triple_polar_epoch_kernel(
            source_x.typed_data()[index], source_y.typed_data()[index],
            *separation.typed_data(), *mass_ratio.typed_data(),
            *tertiary_mass_ratio.typed_data(),
            *tertiary_separation.typed_data(), *tertiary_angle.typed_data(),
            *source_radius.typed_data(), resolution, angular_bins,
            radial_capacity, band_capacity, limb_samples, padding_factor,
            angular_padding_factor, angular_chunk_size, boundary_capacity,
            convention, mode, boundary_subdivision);
        for (int moment = 0; moment < moment_size; ++moment) {
            output_moments[moment] = result.integration.moments[moment];
        }
        magnification->typed_data()[index] = combine_magnification(
            result.integration.moments, *source_radius.typed_data(),
            *limb_c.typed_data(), *limb_d.typed_data(), mode);
        boundary_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.boundary_cells);
        active_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.active_cells);
        band_count->typed_data()[index] = result.tile_count;
        overflow->typed_data()[index] = result.overflow;
        root_failure->typed_data()[index] = result.root_failure;
    }
    return ffi::Error::Success();
}

ffi::Error triple_polar_batch_jacobian_ffi_impl(
    std::int64_t resolution,
    std::int64_t angular_bins,
    std::int64_t radial_capacity,
    std::int64_t band_capacity,
    std::int64_t limb_samples,
    double padding_factor,
    double angular_padding_factor,
    std::int64_t angular_chunk_size,
    std::int64_t boundary_capacity,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    std::int64_t convention,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR2<ffi::F64> moments,
    ffi::ResultBufferR1<ffi::S32> boundary_cells,
    ffi::ResultBufferR1<ffi::S32> active_cells,
    ffi::ResultBufferR1<ffi::S32> band_count,
    ffi::ResultBufferR1<ffi::PRED> overflow,
    ffi::ResultBufferR1<ffi::PRED> root_failure,
    ffi::ResultBufferR2<ffi::F64> magnification_jacobian,
    ffi::ResultBufferR3<ffi::F64> moments_jacobian)
{
    auto validation = validate_triple_polar_batch(
        resolution, angular_bins, radial_capacity, band_capacity,
        limb_samples, angular_chunk_size, boundary_capacity, mode_value,
        boundary_subdivision, convention, source_x, source_y, active, moments);
    if (validation.failure()) return validation;
    const auto mode = static_cast<MomentMode>(mode_value);
    const std::int64_t batch_size = source_x.dimensions()[0];
    const int moment_size = moment_count(mode);
    const auto jacobian_dimensions = magnification_jacobian->dimensions();
    const auto moment_jacobian_dimensions = moments_jacobian->dimensions();
    if (
        jacobian_dimensions[0] != batch_size
        || jacobian_dimensions[1] != triple_parameter_count
        || moment_jacobian_dimensions[0] != batch_size
        || moment_jacobian_dimensions[1] != moment_size
        || moment_jacobian_dimensions[2] != triple_parameter_count) {
        return ffi::Error::InvalidArgument(
            "invalid triple polar Jacobian shapes");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1,
        std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        double* output_moments =
            moments->typed_data() + index * moment_size;
        double* output_jacobian =
            magnification_jacobian->typed_data()
            + index * triple_parameter_count;
        double* output_moment_jacobian =
            moments_jacobian->typed_data()
            + index * moment_size * triple_parameter_count;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            std::fill(output_moments, output_moments + moment_size, 0.0);
            std::fill(
                output_jacobian,
                output_jacobian + triple_parameter_count, 0.0);
            std::fill(
                output_moment_jacobian,
                output_moment_jacobian
                    + moment_size * triple_parameter_count,
                0.0);
            boundary_cells->typed_data()[index] = 0;
            active_cells->typed_data()[index] = 0;
            band_count->typed_data()[index] = 0;
            overflow->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            continue;
        }
        const TripleJet x =
            TripleJet::variable(source_x.typed_data()[index], 0);
        const TripleJet y =
            TripleJet::variable(source_y.typed_data()[index], 1);
        const TripleJet s =
            TripleJet::variable(*separation.typed_data(), 2);
        const TripleJet q =
            TripleJet::variable(*mass_ratio.typed_data(), 3);
        const TripleJet q2 =
            TripleJet::variable(*tertiary_mass_ratio.typed_data(), 4);
        const TripleJet s2 =
            TripleJet::variable(*tertiary_separation.typed_data(), 5);
        const TripleJet angle =
            TripleJet::variable(*tertiary_angle.typed_data(), 6);
        const TripleJet rho =
            TripleJet::variable(*source_radius.typed_data(), 7);
        const auto result = triple_polar_epoch_kernel(
            x, y, s, q, q2, s2, angle, rho, resolution, angular_bins,
            radial_capacity, band_capacity, limb_samples, padding_factor,
            angular_padding_factor, angular_chunk_size, boundary_capacity,
            convention, mode, boundary_subdivision);
        const TripleJet c(*limb_c.typed_data());
        const TripleJet d(*limb_d.typed_data());
        const TripleJet value = combine_magnification(
            result.integration.moments, rho, c, d, mode);
        const auto limb_derivatives = limb_coefficient_derivatives(
            result.integration.moments, rho.value, c.value, d.value, mode);
        magnification->typed_data()[index] = value.value;
        for (
            std::size_t parameter = 0;
            parameter < triple_kernel_derivative_count;
            ++parameter) {
            output_jacobian[parameter] = value.derivative[parameter];
        }
        output_jacobian[8] = limb_derivatives[0];
        output_jacobian[9] = limb_derivatives[1];
        for (int moment = 0; moment < moment_size; ++moment) {
            output_moments[moment] =
                result.integration.moments[moment].value;
            for (
                std::size_t parameter = 0;
                parameter < triple_parameter_count;
                ++parameter) {
                output_moment_jacobian[
                    moment * triple_parameter_count + parameter] =
                    parameter < triple_kernel_derivative_count
                    ? result.integration.moments[moment]
                          .derivative[parameter]
                    : 0.0;
            }
        }
        boundary_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.boundary_cells);
        active_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.active_cells);
        band_count->typed_data()[index] = result.tile_count;
        overflow->typed_data()[index] = result.overflow;
        root_failure->typed_data()[index] = result.root_failure;
    }
    return ffi::Error::Success();
}

#define LCBININT_TRIPLE_POLAR_BATCH_BINDING \
    ffi::Ffi::Bind() \
        .Attr<std::int64_t>("resolution") \
        .Attr<std::int64_t>("angular_bins") \
        .Attr<std::int64_t>("radial_capacity") \
        .Attr<std::int64_t>("band_capacity") \
        .Attr<std::int64_t>("limb_samples") \
        .Attr<double>("padding_factor") \
        .Attr<double>("angular_padding_factor") \
        .Attr<std::int64_t>("angular_chunk_size") \
        .Attr<std::int64_t>("boundary_capacity") \
        .Attr<std::int64_t>("moment_mode") \
        .Attr<std::int64_t>("boundary_subdivision") \
        .Attr<std::int64_t>("convention") \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::PRED>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR2<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::S32>>() \
        .Ret<ffi::BufferR1<ffi::S32>>() \
        .Ret<ffi::BufferR1<ffi::S32>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    triple_polar_batch_forward_ffi_handler,
    triple_polar_batch_forward_ffi_impl,
    LCBININT_TRIPLE_POLAR_BATCH_BINDING);

XLA_FFI_DEFINE_HANDLER(
    triple_polar_batch_jacobian_ffi_handler,
    triple_polar_batch_jacobian_ffi_impl,
    LCBININT_TRIPLE_POLAR_BATCH_BINDING
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR3<ffi::F64>>());

#undef LCBININT_TRIPLE_POLAR_BATCH_BINDING

ffi::Error point_batch_ffi_impl(
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::S32> image_count,
    ffi::ResultBufferR1<ffi::PRED> root_failure)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        source_y.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || image_count->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "point-source batch arrays must have a common length");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = batch_size >= 256
        ? std::max(
            1,
            std::min(
                {static_cast<int>(batch_size), omp_get_max_threads(), 8}))
        : 1;
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        std::int32_t count = 0;
        bool failed = false;
        const auto value = binary_point_magnification_jet(
            Jet::variable(source_x.typed_data()[index], 0),
            Jet::variable(source_y.typed_data()[index], 1),
            Jet::variable(*separation.typed_data(), 2),
            Jet::variable(*mass_ratio.typed_data(), 3),
            count, failed);
        magnification->typed_data()[index] = value.value;
        image_count->typed_data()[index] = count;
        root_failure->typed_data()[index] = failed;
    }
    return ffi::Error::Success();
}

ffi::Error point_batch_jacobian_ffi_impl(
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::S32> image_count,
    ffi::ResultBufferR1<ffi::PRED> root_failure,
    ffi::ResultBufferR2<ffi::F64> output_jacobian)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        source_y.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || image_count->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size
        || output_jacobian->dimensions()[0] != batch_size
        || output_jacobian->dimensions()[1] != 4) {
        return ffi::Error::InvalidArgument(
            "point-source batch Jacobian arrays have invalid shapes");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = batch_size >= 256
        ? std::max(
            1,
            std::min(
                {static_cast<int>(batch_size), omp_get_max_threads(), 8}))
        : 1;
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        std::int32_t count = 0;
        bool failed = false;
        const auto value = binary_point_magnification_jet(
            Jet::variable(source_x.typed_data()[index], 0),
            Jet::variable(source_y.typed_data()[index], 1),
            Jet::variable(*separation.typed_data(), 2),
            Jet::variable(*mass_ratio.typed_data(), 3),
            count, failed);
        magnification->typed_data()[index] = value.value;
        image_count->typed_data()[index] = count;
        root_failure->typed_data()[index] = failed;
        std::copy_n(
            value.derivative.begin(), 4,
            output_jacobian->typed_data() + 4 * index);
    }
    return ffi::Error::Success();
}

#define LCBININT_POINT_BATCH_BINDING \
    ffi::Ffi::Bind() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::S32>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    point_batch_ffi_handler,
    point_batch_ffi_impl,
    LCBININT_POINT_BATCH_BINDING);

XLA_FFI_DEFINE_HANDLER(
    point_batch_jacobian_ffi_handler,
    point_batch_jacobian_ffi_impl,
    LCBININT_POINT_BATCH_BINDING.Ret<ffi::BufferR2<ffi::F64>>());

#undef LCBININT_POINT_BATCH_BINDING

ffi::Error hexadecapole_batch_ffi_impl(
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> point_magnification,
    ffi::ResultBufferR1<ffi::F64> quadrupole_correction,
    ffi::ResultBufferR1<ffi::F64> hexadecapole_correction,
    ffi::ResultBufferR1<ffi::PRED> topology_stable,
    ffi::ResultBufferR1<ffi::PRED> root_failure)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        source_y.dimensions()[0] != batch_size
        || active.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || point_magnification->dimensions()[0] != batch_size
        || quadrupole_correction->dimensions()[0] != batch_size
        || hexadecapole_correction->dimensions()[0] != batch_size
        || topology_stable->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "hexadecapole batch arrays must have a common length");
    }
    const auto active_count = static_cast<std::int64_t>(std::count(
        active.typed_data(), active.typed_data() + batch_size, true));
    if (active_count == 0) {
        std::fill_n(magnification->typed_data(), batch_size, 0.0);
        std::fill_n(point_magnification->typed_data(), batch_size, 0.0);
        std::fill_n(quadrupole_correction->typed_data(), batch_size, 0.0);
        std::fill_n(hexadecapole_correction->typed_data(), batch_size, 0.0);
        std::fill_n(topology_stable->typed_data(), batch_size, false);
        std::fill_n(root_failure->typed_data(), batch_size, false);
        return ffi::Error::Success();
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1,
        std::min(
            {static_cast<int>(active_count), omp_get_max_threads(), 8}));
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            point_magnification->typed_data()[index] = 0.0;
            quadrupole_correction->typed_data()[index] = 0.0;
            hexadecapole_correction->typed_data()[index] = 0.0;
            topology_stable->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            continue;
        }
        const auto result = hexadecapole_kernel(
            source_x.typed_data()[index], source_y.typed_data()[index],
            *separation.typed_data(), *mass_ratio.typed_data(),
            *source_radius.typed_data(), *limb_c.typed_data(),
            *limb_d.typed_data());
        magnification->typed_data()[index] = result.magnification.value;
        point_magnification->typed_data()[index] =
            result.point_magnification.value;
        quadrupole_correction->typed_data()[index] =
            result.quadrupole_correction.value;
        hexadecapole_correction->typed_data()[index] =
            result.hexadecapole_correction.value;
        topology_stable->typed_data()[index] = result.topology_stable;
        root_failure->typed_data()[index] = result.root_failure;
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    hexadecapole_batch_ffi_handler,
    hexadecapole_batch_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::PRED>>());

ffi::Error hexadecapole_batch_jacobian_ffi_impl(
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> point_magnification,
    ffi::ResultBufferR1<ffi::F64> quadrupole_correction,
    ffi::ResultBufferR1<ffi::F64> hexadecapole_correction,
    ffi::ResultBufferR1<ffi::PRED> topology_stable,
    ffi::ResultBufferR1<ffi::PRED> root_failure,
    ffi::ResultBufferR3<ffi::F64> output_jacobian)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    const auto jacobian_dimensions = output_jacobian->dimensions();
    if (
        source_y.dimensions()[0] != batch_size
        || active.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || point_magnification->dimensions()[0] != batch_size
        || quadrupole_correction->dimensions()[0] != batch_size
        || hexadecapole_correction->dimensions()[0] != batch_size
        || topology_stable->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size
        || jacobian_dimensions[0] != batch_size
        || jacobian_dimensions[1] != 4
        || jacobian_dimensions[2] != parameter_count) {
        return ffi::Error::InvalidArgument(
            "hexadecapole batch Jacobian arrays have invalid shapes");
    }
    const auto active_count = static_cast<std::int64_t>(std::count(
        active.typed_data(), active.typed_data() + batch_size, true));
    if (active_count == 0) {
        std::fill_n(magnification->typed_data(), batch_size, 0.0);
        std::fill_n(point_magnification->typed_data(), batch_size, 0.0);
        std::fill_n(quadrupole_correction->typed_data(), batch_size, 0.0);
        std::fill_n(hexadecapole_correction->typed_data(), batch_size, 0.0);
        std::fill_n(topology_stable->typed_data(), batch_size, false);
        std::fill_n(root_failure->typed_data(), batch_size, false);
        std::fill_n(
            output_jacobian->typed_data(),
            batch_size * 4 * parameter_count,
            0.0);
        return ffi::Error::Success();
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1,
        std::min(
            {static_cast<int>(active_count), omp_get_max_threads(), 8}));
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        double* jacobian =
            output_jacobian->typed_data() + index * 4 * parameter_count;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            point_magnification->typed_data()[index] = 0.0;
            quadrupole_correction->typed_data()[index] = 0.0;
            hexadecapole_correction->typed_data()[index] = 0.0;
            topology_stable->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            std::fill(jacobian, jacobian + 4 * parameter_count, 0.0);
            continue;
        }
        const auto result = hexadecapole_kernel(
            source_x.typed_data()[index], source_y.typed_data()[index],
            *separation.typed_data(), *mass_ratio.typed_data(),
            *source_radius.typed_data(), *limb_c.typed_data(),
            *limb_d.typed_data());
        magnification->typed_data()[index] = result.magnification.value;
        point_magnification->typed_data()[index] =
            result.point_magnification.value;
        quadrupole_correction->typed_data()[index] =
            result.quadrupole_correction.value;
        hexadecapole_correction->typed_data()[index] =
            result.hexadecapole_correction.value;
        topology_stable->typed_data()[index] = result.topology_stable;
        root_failure->typed_data()[index] = result.root_failure;
        const std::array<const Jet*, 4> values{
            &result.magnification,
            &result.point_magnification,
            &result.quadrupole_correction,
            &result.hexadecapole_correction};
        for (std::size_t output = 0; output < values.size(); ++output) {
            for (
                std::size_t parameter = 0;
                parameter < kernel_derivative_count;
                ++parameter) {
                jacobian[output * parameter_count + parameter] =
                    values[output]->derivative[parameter];
            }
            jacobian[output * parameter_count + 5] = 0.0;
            jacobian[output * parameter_count + 6] = 0.0;
        }
        jacobian[5] = result.limb_c_derivative;
        jacobian[6] = result.limb_d_derivative;
        jacobian[2 * parameter_count + 5] =
            result.quadrupole_limb_c_derivative;
        jacobian[2 * parameter_count + 6] =
            result.quadrupole_limb_d_derivative;
        jacobian[3 * parameter_count + 5] =
            result.hexadecapole_limb_c_derivative;
        jacobian[3 * parameter_count + 6] =
            result.hexadecapole_limb_d_derivative;
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    hexadecapole_batch_jacobian_ffi_handler,
    hexadecapole_batch_jacobian_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR3<ffi::F64>>());

ffi::Error validate_triple_hexadecapole_outputs(
    ffi::BufferR1<ffi::F64>& source_x,
    ffi::BufferR1<ffi::F64>& source_y,
    ffi::ResultBufferR1<ffi::F64>& magnification,
    ffi::ResultBufferR1<ffi::F64>& point_magnification,
    ffi::ResultBufferR1<ffi::F64>& quadrupole_correction,
    ffi::ResultBufferR1<ffi::F64>& hexadecapole_correction,
    ffi::ResultBufferR1<ffi::PRED>& topology_stable,
    ffi::ResultBufferR1<ffi::PRED>& root_failure)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        source_y.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || point_magnification->dimensions()[0] != batch_size
        || quadrupole_correction->dimensions()[0] != batch_size
        || hexadecapole_correction->dimensions()[0] != batch_size
        || topology_stable->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "triple hexadecapole arrays must have a common length");
    }
    return ffi::Error::Success();
}

ffi::Error triple_hexadecapole_batch_ffi_impl(
    std::int64_t convention,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> point_magnification,
    ffi::ResultBufferR1<ffi::F64> quadrupole_correction,
    ffi::ResultBufferR1<ffi::F64> hexadecapole_correction,
    ffi::ResultBufferR1<ffi::PRED> topology_stable,
    ffi::ResultBufferR1<ffi::PRED> root_failure)
{
    auto validation = validate_triple_hexadecapole_outputs(
        source_x, source_y, magnification, point_magnification,
        quadrupole_correction, hexadecapole_correction,
        topology_stable, root_failure);
    if (validation.failure()) return validation;
    if (
        (convention != 0 && convention != 1)
        || active.dimensions()[0] != source_x.dimensions()[0]) {
        return ffi::Error::InvalidArgument("invalid triple convention");
    }
    const std::int64_t batch_size = source_x.dimensions()[0];
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1, std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            point_magnification->typed_data()[index] = 0.0;
            quadrupole_correction->typed_data()[index] = 0.0;
            hexadecapole_correction->typed_data()[index] = 0.0;
            topology_stable->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            continue;
        }
        const auto result = triple_hexadecapole_kernel(
            source_x.typed_data()[index], source_y.typed_data()[index],
            *separation.typed_data(), *mass_ratio.typed_data(),
            *tertiary_mass_ratio.typed_data(),
            *tertiary_separation.typed_data(),
            *tertiary_angle.typed_data(), *source_radius.typed_data(),
            *limb_c.typed_data(), *limb_d.typed_data(), convention);
        magnification->typed_data()[index] = result.magnification.value;
        point_magnification->typed_data()[index] =
            result.point_magnification.value;
        quadrupole_correction->typed_data()[index] =
            result.quadrupole_correction.value;
        hexadecapole_correction->typed_data()[index] =
            result.hexadecapole_correction.value;
        topology_stable->typed_data()[index] = result.topology_stable;
        root_failure->typed_data()[index] = result.root_failure;
    }
    return ffi::Error::Success();
}

ffi::Error triple_hexadecapole_batch_jacobian_ffi_impl(
    std::int64_t convention,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> point_magnification,
    ffi::ResultBufferR1<ffi::F64> quadrupole_correction,
    ffi::ResultBufferR1<ffi::F64> hexadecapole_correction,
    ffi::ResultBufferR1<ffi::PRED> topology_stable,
    ffi::ResultBufferR1<ffi::PRED> root_failure,
    ffi::ResultBufferR3<ffi::F64> output_jacobian)
{
    auto validation = validate_triple_hexadecapole_outputs(
        source_x, source_y, magnification, point_magnification,
        quadrupole_correction, hexadecapole_correction,
        topology_stable, root_failure);
    if (validation.failure()) return validation;
    const std::int64_t batch_size = source_x.dimensions()[0];
    const auto dimensions = output_jacobian->dimensions();
    if (
        (convention != 0 && convention != 1)
        || active.dimensions()[0] != batch_size
        || dimensions[0] != batch_size || dimensions[1] != 4
        || dimensions[2] != triple_parameter_count) {
        return ffi::Error::InvalidArgument(
            "invalid triple hexadecapole Jacobian shape");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1, std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        auto* jacobian = output_jacobian->typed_data()
            + index * 4 * triple_parameter_count;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            point_magnification->typed_data()[index] = 0.0;
            quadrupole_correction->typed_data()[index] = 0.0;
            hexadecapole_correction->typed_data()[index] = 0.0;
            topology_stable->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            std::fill(
                jacobian,
                jacobian + 4 * triple_parameter_count,
                0.0);
            continue;
        }
        const auto result = triple_hexadecapole_kernel(
            source_x.typed_data()[index], source_y.typed_data()[index],
            *separation.typed_data(), *mass_ratio.typed_data(),
            *tertiary_mass_ratio.typed_data(),
            *tertiary_separation.typed_data(),
            *tertiary_angle.typed_data(), *source_radius.typed_data(),
            *limb_c.typed_data(), *limb_d.typed_data(), convention);
        magnification->typed_data()[index] = result.magnification.value;
        point_magnification->typed_data()[index] =
            result.point_magnification.value;
        quadrupole_correction->typed_data()[index] =
            result.quadrupole_correction.value;
        hexadecapole_correction->typed_data()[index] =
            result.hexadecapole_correction.value;
        topology_stable->typed_data()[index] = result.topology_stable;
        root_failure->typed_data()[index] = result.root_failure;
        const std::array<const TripleJet*, 4> values{
            &result.magnification,
            &result.point_magnification,
            &result.quadrupole_correction,
            &result.hexadecapole_correction};
        for (std::size_t output = 0; output < values.size(); ++output) {
            for (
                std::size_t parameter = 0;
                parameter < triple_kernel_derivative_count;
                ++parameter) {
                jacobian[output * triple_parameter_count + parameter] =
                    values[output]->derivative[parameter];
            }
            jacobian[output * triple_parameter_count + 8] = 0.0;
            jacobian[output * triple_parameter_count + 9] = 0.0;
        }
        jacobian[8] = result.limb_c_derivative;
        jacobian[9] = result.limb_d_derivative;
        jacobian[2 * triple_parameter_count + 8] =
            result.quadrupole_limb_c_derivative;
        jacobian[2 * triple_parameter_count + 9] =
            result.quadrupole_limb_d_derivative;
        jacobian[3 * triple_parameter_count + 8] =
            result.hexadecapole_limb_c_derivative;
        jacobian[3 * triple_parameter_count + 9] =
            result.hexadecapole_limb_d_derivative;
    }
    return ffi::Error::Success();
}

#define LCBININT_TRIPLE_HEX_BINDING \
    ffi::Ffi::Bind() \
        .Attr<std::int64_t>("convention") \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::PRED>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    triple_hexadecapole_batch_ffi_handler,
    triple_hexadecapole_batch_ffi_impl,
    LCBININT_TRIPLE_HEX_BINDING);

XLA_FFI_DEFINE_HANDLER(
    triple_hexadecapole_batch_jacobian_ffi_handler,
    triple_hexadecapole_batch_jacobian_ffi_impl,
    LCBININT_TRIPLE_HEX_BINDING
        .Ret<ffi::BufferR3<ffi::F64>>());

#undef LCBININT_TRIPLE_HEX_BINDING

ffi::Error triple_point_batch_ffi_impl(
    std::int64_t convention,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> derivative_indicator,
    ffi::ResultBufferR1<ffi::PRED> root_failure)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        (convention != 0 && convention != 1)
        || source_y.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || derivative_indicator->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "invalid triple point-source batch shapes");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1, std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        const TripleJet x =
            TripleJet::variable(source_x.typed_data()[index], 0);
        const TripleJet y =
            TripleJet::variable(source_y.typed_data()[index], 1);
        const TripleJet s =
            TripleJet::variable(*separation.typed_data(), 2);
        const TripleJet q =
            TripleJet::variable(*mass_ratio.typed_data(), 3);
        const TripleJet q2 =
            TripleJet::variable(*tertiary_mass_ratio.typed_data(), 4);
        const TripleJet s2 =
            TripleJet::variable(*tertiary_separation.typed_data(), 5);
        const TripleJet angle =
            TripleJet::variable(*tertiary_angle.typed_data(), 6);
        std::int32_t image_count = 0;
        bool failed = false;
        double indicator = 0.0;
        const auto result = triple_point_magnification_jet(
            x, y, s, q, q2, s2, angle, convention, image_count, failed,
            &indicator);
        magnification->typed_data()[index] = result.value;
        derivative_indicator->typed_data()[index] = indicator;
        root_failure->typed_data()[index] = failed;
    }
    return ffi::Error::Success();
}

ffi::Error triple_point_batch_jacobian_ffi_impl(
    std::int64_t convention,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_mass_ratio,
    ffi::BufferR0<ffi::F64> tertiary_separation,
    ffi::BufferR0<ffi::F64> tertiary_angle,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR1<ffi::F64> derivative_indicator,
    ffi::ResultBufferR1<ffi::PRED> root_failure,
    ffi::ResultBufferR2<ffi::F64> jacobian)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        (convention != 0 && convention != 1)
        || source_y.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || derivative_indicator->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size
        || jacobian->dimensions()[0] != batch_size
        || jacobian->dimensions()[1] != 7) {
        return ffi::Error::InvalidArgument(
            "invalid triple point-source Jacobian shapes");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1, std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(static) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        const TripleJet x =
            TripleJet::variable(source_x.typed_data()[index], 0);
        const TripleJet y =
            TripleJet::variable(source_y.typed_data()[index], 1);
        const TripleJet s =
            TripleJet::variable(*separation.typed_data(), 2);
        const TripleJet q =
            TripleJet::variable(*mass_ratio.typed_data(), 3);
        const TripleJet q2 =
            TripleJet::variable(*tertiary_mass_ratio.typed_data(), 4);
        const TripleJet s2 =
            TripleJet::variable(*tertiary_separation.typed_data(), 5);
        const TripleJet angle =
            TripleJet::variable(*tertiary_angle.typed_data(), 6);
        std::int32_t image_count = 0;
        bool failed = false;
        double indicator = 0.0;
        const auto result = triple_point_magnification_jet(
            x, y, s, q, q2, s2, angle, convention, image_count, failed,
            &indicator);
        magnification->typed_data()[index] = result.value;
        derivative_indicator->typed_data()[index] = indicator;
        root_failure->typed_data()[index] = failed;
        for (std::size_t parameter = 0; parameter < 7; ++parameter) {
            jacobian->typed_data()[index * 7 + parameter] =
                result.derivative[parameter];
        }
    }
    return ffi::Error::Success();
}

#define LCBININT_TRIPLE_POINT_BINDING \
    ffi::Ffi::Bind() \
        .Attr<std::int64_t>("convention") \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    triple_point_batch_ffi_handler,
    triple_point_batch_ffi_impl,
    LCBININT_TRIPLE_POINT_BINDING);

XLA_FFI_DEFINE_HANDLER(
    triple_point_batch_jacobian_ffi_handler,
    triple_point_batch_jacobian_ffi_impl,
    LCBININT_TRIPLE_POINT_BINDING
        .Ret<ffi::BufferR2<ffi::F64>>());

#undef LCBININT_TRIPLE_POINT_BINDING

struct TrajectoryEpochDecision {
    double magnification = std::numeric_limits<double>::quiet_NaN();
    double estimated_error = std::numeric_limits<double>::quiet_NaN();
    std::int32_t method = 1;
    bool support_valid = false;
    bool used_multipole = false;
    bool used_polar = false;
    bool needs_fallback = false;
    std::array<double, parameter_count> jacobian{};
};

bool accept_hexadecapole(
    const HexadecapoleKernelResult& hex,
    double source_radius,
    double absolute_tolerance,
    double relative_tolerance,
    double safety_factor)
{
    const double magnitude = std::max(
        std::abs(hex.magnification.value), 1.0);
    const double budget =
        absolute_tolerance + relative_tolerance * magnitude;
    const double error = std::abs(hex.hexadecapole_correction.value);
    const double correction_scale = std::max(
        std::abs(hex.quadrupole_correction.value), budget);
    return source_radius >= 0.0
        && hex.topology_stable && !hex.root_failure
        && std::isfinite(hex.magnification.value)
        && error <= 0.25 * correction_scale
        && std::abs(hex.quadrupole_correction.value) <= 0.1 * magnitude
        && safety_factor * error <= budget;
}

template <typename Scalar>
void write_selected_jacobian(
    TrajectoryEpochDecision& decision,
    const Scalar& magnification,
    const std::array<double, 2>& limb_derivatives)
{
    if constexpr (std::is_same_v<Scalar, Jet>) {
        for (
            std::size_t parameter = 0;
            parameter < kernel_derivative_count;
            ++parameter) {
            decision.jacobian[parameter] =
                magnification.derivative[parameter];
        }
        decision.jacobian[5] = limb_derivatives[0];
        decision.jacobian[6] = limb_derivatives[1];
    }
}

template <bool WithJacobian>
TrajectoryEpochDecision trajectory_epoch_kernel(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    double limb_c,
    double limb_d,
    double absolute_tolerance,
    double relative_tolerance,
    double multipole_safety_factor,
    double polar_magnification_threshold,
    double polar_max_source_radius,
    double polar_min_mass_ratio,
    std::int64_t cartesian_resolution,
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t cartesian_limb_samples,
    std::int64_t polar_resolution,
    std::int64_t polar_angular_bins,
    std::int64_t polar_radial_capacity,
    std::int64_t polar_band_capacity,
    std::int64_t polar_limb_samples,
    double polar_padding_factor,
    double polar_angular_padding_factor,
    std::int64_t polar_angular_chunk_size,
    std::int64_t polar_boundary_capacity,
    std::int64_t polar_boundary_subdivision,
    bool polar_fallback_on_overflow,
    MomentMode mode)
{
    TrajectoryEpochDecision decision;
    const auto hex = hexadecapole_kernel(
        source_x, source_y, separation, mass_ratio, source_radius,
        limb_c, limb_d);
    if (accept_hexadecapole(
            hex, source_radius, absolute_tolerance, relative_tolerance,
            multipole_safety_factor)) {
        decision.magnification = hex.magnification.value;
        decision.estimated_error =
            std::abs(hex.hexadecapole_correction.value);
        decision.method = 0;
        decision.support_valid = true;
        decision.used_multipole = true;
        if constexpr (WithJacobian) {
            for (
                std::size_t parameter = 0;
                parameter < kernel_derivative_count;
                ++parameter) {
                decision.jacobian[parameter] =
                    hex.magnification.derivative[parameter];
            }
            decision.jacobian[5] = hex.limb_c_derivative;
            decision.jacobian[6] = hex.limb_d_derivative;
        }
        return decision;
    }
    const double absolute_q = std::abs(mass_ratio);
    const double symmetric_q = absolute_q > 0.0
        ? std::min(absolute_q, 1.0 / absolute_q)
        : 0.0;
    const bool polar_allowed =
        hex.topology_stable && symmetric_q >= polar_min_mass_ratio;
    bool use_polar =
        hex.point_magnification.value >= polar_magnification_threshold
        && source_radius <= polar_max_source_radius
        && source_radius > 0.0 && polar_allowed;

    const auto evaluate_polar = [&]() {
        decision.method = 2;
        decision.used_polar = true;
        if constexpr (WithJacobian) {
            const Jet x = Jet::variable(source_x, 0);
            const Jet y = Jet::variable(source_y, 1);
            const Jet s = Jet::variable(separation, 2);
            const Jet q = Jet::variable(mass_ratio, 3);
            const Jet rho = Jet::variable(source_radius, 4);
            const auto result = polar_epoch_kernel(
                x, y, s, q, rho, polar_resolution, polar_angular_bins,
                polar_radial_capacity, polar_band_capacity,
                polar_limb_samples, polar_padding_factor,
                polar_angular_padding_factor, polar_angular_chunk_size,
                polar_boundary_capacity, mode,
                polar_boundary_subdivision);
            const Jet magnification = combine_magnification(
                result.integration.moments, rho, Jet(limb_c), Jet(limb_d),
                mode);
            decision.magnification = magnification.value;
            decision.support_valid =
                !(result.overflow || result.root_failure);
            write_selected_jacobian(
                decision, magnification,
                limb_coefficient_derivatives(
                    result.integration.moments, source_radius, limb_c,
                    limb_d, mode));
        } else {
            const auto result = polar_epoch_kernel(
                source_x, source_y, separation, mass_ratio, source_radius,
                polar_resolution, polar_angular_bins,
                polar_radial_capacity, polar_band_capacity,
                polar_limb_samples, polar_padding_factor,
                polar_angular_padding_factor, polar_angular_chunk_size,
                polar_boundary_capacity, mode,
                polar_boundary_subdivision);
            decision.magnification = combine_magnification(
                result.integration.moments, source_radius, limb_c, limb_d,
                mode);
            decision.support_valid =
                !(result.overflow || result.root_failure);
        }
    };
    if (use_polar) {
        evaluate_polar();
    } else if constexpr (WithJacobian) {
        const Jet x = Jet::variable(source_x, 0);
        const Jet y = Jet::variable(source_y, 1);
        const Jet s = Jet::variable(separation, 2);
        const Jet q = Jet::variable(mass_ratio, 3);
        const Jet rho = Jet::variable(source_radius, 4);
        const auto result = cartesian_epoch_kernel(
            source_radius / cartesian_resolution, x, y, s, q, rho,
            Jet(limb_d), tile_size, tile_capacity,
            cartesian_limb_samples, mode,
            mode == MomentMode::two_coefficient ? 4 : 3);
        if (
            result.overflow && polar_allowed
            && polar_fallback_on_overflow) {
            use_polar = true;
            evaluate_polar();
        } else {
            const Jet magnification = combine_magnification(
                result.integration.moments, rho, Jet(limb_c), Jet(limb_d),
                mode);
            decision.magnification = magnification.value;
            decision.support_valid =
                !(result.overflow || result.root_failure);
            write_selected_jacobian(
                decision, magnification,
                limb_coefficient_derivatives(
                    result.integration.moments, source_radius, limb_c,
                    limb_d, mode));
        }
    } else {
        const auto result = cartesian_epoch_kernel(
            source_radius / cartesian_resolution, source_x, source_y,
            separation, mass_ratio, source_radius, limb_d, tile_size,
            tile_capacity, cartesian_limb_samples, mode,
            mode == MomentMode::two_coefficient ? 4 : 3);
        if (
            result.overflow && polar_allowed
            && polar_fallback_on_overflow) {
            use_polar = true;
            evaluate_polar();
        } else {
            decision.magnification = combine_magnification(
                result.integration.moments, source_radius, limb_c, limb_d,
                mode);
            decision.support_valid =
                !(result.overflow || result.root_failure);
        }
    }
    decision.needs_fallback = !decision.support_valid;
    return decision;
}

#define LCBININT_TRAJECTORY_ARGUMENTS \
    std::int64_t cartesian_resolution, \
    std::int64_t tile_size, \
    std::int64_t tile_capacity, \
    std::int64_t cartesian_limb_samples, \
    std::int64_t polar_resolution, \
    std::int64_t polar_angular_bins, \
    std::int64_t polar_radial_capacity, \
    std::int64_t polar_band_capacity, \
    std::int64_t polar_limb_samples, \
    double polar_padding_factor, \
    double polar_angular_padding_factor, \
    std::int64_t polar_angular_chunk_size, \
    std::int64_t polar_boundary_capacity, \
    std::int64_t polar_boundary_subdivision, \
    std::int64_t polar_fallback_on_overflow, \
    std::int64_t mode_value, \
    ffi::BufferR1<ffi::F64> source_x, \
    ffi::BufferR1<ffi::F64> source_y, \
    ffi::BufferR1<ffi::F64> separation, \
    ffi::BufferR0<ffi::F64> mass_ratio, \
    ffi::BufferR0<ffi::F64> source_radius, \
    ffi::BufferR0<ffi::F64> limb_c, \
    ffi::BufferR0<ffi::F64> limb_d, \
    ffi::BufferR0<ffi::F64> absolute_tolerance, \
    ffi::BufferR0<ffi::F64> relative_tolerance, \
    ffi::BufferR0<ffi::F64> multipole_safety_factor, \
    ffi::BufferR0<ffi::F64> polar_magnification_threshold, \
    ffi::BufferR0<ffi::F64> polar_max_source_radius, \
    ffi::BufferR0<ffi::F64> polar_min_mass_ratio, \
    ffi::ResultBufferR1<ffi::F64> magnification, \
    ffi::ResultBufferR1<ffi::S32> method, \
    ffi::ResultBufferR1<ffi::F64> estimated_error, \
    ffi::ResultBufferR1<ffi::PRED> support_valid, \
    ffi::ResultBufferR1<ffi::PRED> used_multipole, \
    ffi::ResultBufferR1<ffi::PRED> used_polar, \
    ffi::ResultBufferR1<ffi::PRED> needs_fallback

#define LCBININT_TRAJECTORY_KERNEL_ARGUMENTS(index) \
    source_x.typed_data()[index], source_y.typed_data()[index], \
    separation.typed_data()[index], *mass_ratio.typed_data(), \
    *source_radius.typed_data(), *limb_c.typed_data(), \
    *limb_d.typed_data(), *absolute_tolerance.typed_data(), \
    *relative_tolerance.typed_data(), *multipole_safety_factor.typed_data(), \
    *polar_magnification_threshold.typed_data(), \
    *polar_max_source_radius.typed_data(), *polar_min_mass_ratio.typed_data(), \
    cartesian_resolution, tile_size, tile_capacity, \
    cartesian_limb_samples, polar_resolution, polar_angular_bins, \
    polar_radial_capacity, polar_band_capacity, polar_limb_samples, \
    polar_padding_factor, polar_angular_padding_factor, \
    polar_angular_chunk_size, polar_boundary_capacity, \
    polar_boundary_subdivision, polar_fallback_on_overflow, \
    static_cast<MomentMode>(mode_value)

ffi::Error trajectory_forward_ffi_impl(LCBININT_TRAJECTORY_ARGUMENTS)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        source_y.dimensions()[0] != batch_size
        || separation.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || method->dimensions()[0] != batch_size
        || estimated_error->dimensions()[0] != batch_size
        || support_valid->dimensions()[0] != batch_size
        || used_multipole->dimensions()[0] != batch_size
        || used_polar->dimensions()[0] != batch_size
        || needs_fallback->dimensions()[0] != batch_size
        || mode_value < 1 || mode_value > 3) {
        return ffi::Error::InvalidArgument(
            "invalid trajectory FFI arrays or mode");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1,
        std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        const auto result = trajectory_epoch_kernel<false>(
            LCBININT_TRAJECTORY_KERNEL_ARGUMENTS(index));
        magnification->typed_data()[index] = result.magnification;
        method->typed_data()[index] = result.method;
        estimated_error->typed_data()[index] = result.estimated_error;
        support_valid->typed_data()[index] = result.support_valid;
        used_multipole->typed_data()[index] = result.used_multipole;
        used_polar->typed_data()[index] = result.used_polar;
        needs_fallback->typed_data()[index] = result.needs_fallback;
    }
    return ffi::Error::Success();
}

ffi::Error trajectory_jacobian_ffi_impl(
    LCBININT_TRAJECTORY_ARGUMENTS,
    ffi::ResultBufferR2<ffi::F64> output_jacobian)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        output_jacobian->dimensions()[0] != batch_size
        || output_jacobian->dimensions()[1] != parameter_count) {
        return ffi::Error::InvalidArgument(
            "trajectory Jacobian must have shape (N, 7)");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1,
        std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        const auto result = trajectory_epoch_kernel<true>(
            LCBININT_TRAJECTORY_KERNEL_ARGUMENTS(index));
        magnification->typed_data()[index] = result.magnification;
        method->typed_data()[index] = result.method;
        estimated_error->typed_data()[index] = result.estimated_error;
        support_valid->typed_data()[index] = result.support_valid;
        used_multipole->typed_data()[index] = result.used_multipole;
        used_polar->typed_data()[index] = result.used_polar;
        needs_fallback->typed_data()[index] = result.needs_fallback;
        std::copy(
            result.jacobian.begin(), result.jacobian.end(),
            output_jacobian->typed_data() + index * parameter_count);
    }
    return ffi::Error::Success();
}

#define LCBININT_TRAJECTORY_BINDING \
    ffi::Ffi::Bind() \
        .Attr<std::int64_t>("cartesian_resolution") \
        .Attr<std::int64_t>("tile_size") \
        .Attr<std::int64_t>("tile_capacity") \
        .Attr<std::int64_t>("cartesian_limb_samples") \
        .Attr<std::int64_t>("polar_resolution") \
        .Attr<std::int64_t>("polar_angular_bins") \
        .Attr<std::int64_t>("polar_radial_capacity") \
        .Attr<std::int64_t>("polar_band_capacity") \
        .Attr<std::int64_t>("polar_limb_samples") \
        .Attr<double>("polar_padding_factor") \
        .Attr<double>("polar_angular_padding_factor") \
        .Attr<std::int64_t>("polar_angular_chunk_size") \
        .Attr<std::int64_t>("polar_boundary_capacity") \
        .Attr<std::int64_t>("polar_boundary_subdivision") \
        .Attr<std::int64_t>("polar_fallback_on_overflow") \
        .Attr<std::int64_t>("moment_mode") \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::S32>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    trajectory_forward_ffi_handler,
    trajectory_forward_ffi_impl,
    LCBININT_TRAJECTORY_BINDING);
XLA_FFI_DEFINE_HANDLER(
    trajectory_jacobian_ffi_handler,
    trajectory_jacobian_ffi_impl,
    LCBININT_TRAJECTORY_BINDING.Ret<ffi::BufferR2<ffi::F64>>());

#undef LCBININT_TRAJECTORY_BINDING
#undef LCBININT_TRAJECTORY_KERNEL_ARGUMENTS
#undef LCBININT_TRAJECTORY_ARGUMENTS

struct CartesianLadderRung {
    double magnification = 0.0;
    bool support_valid = false;
    std::array<double, parameter_count> jacobian{};
};

struct CartesianLadderDecision {
    double magnification = 0.0;
    double estimated_error = std::numeric_limits<double>::infinity();
    bool support_valid = false;
    bool converged = false;
    std::array<double, parameter_count> magnification_jacobian{};
    std::array<double, parameter_count> error_jacobian{};
};

template <bool WithJacobian>
CartesianLadderRung evaluate_cartesian_ladder_rung(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    double limb_c,
    double limb_d,
    std::int32_t resolution,
    std::int32_t tile_capacity,
    std::int64_t tile_size,
    MomentMode mode,
    std::int64_t boundary_subdivision,
    const CartesianSeedSupport& prepared)
{
    CartesianLadderRung rung;
    if constexpr (WithJacobian) {
        const Jet x = Jet::variable(source_x, 0);
        const Jet y = Jet::variable(source_y, 1);
        const Jet s = Jet::variable(separation, 2);
        const Jet q = Jet::variable(mass_ratio, 3);
        const Jet rho = Jet::variable(source_radius, 4);
        const auto result = cartesian_epoch_kernel_from_prepared(
            source_radius / static_cast<double>(resolution),
            x, y, s, q, rho, Jet(limb_d), tile_size, tile_capacity,
            mode, boundary_subdivision, prepared);
        const Jet magnification = combine_magnification(
            result.integration.moments, rho, Jet(limb_c), Jet(limb_d), mode);
        rung.magnification = magnification.value;
        for (
            std::size_t parameter = 0;
            parameter < kernel_derivative_count;
            ++parameter) {
            rung.jacobian[parameter] = magnification.derivative[parameter];
        }
        const auto limb_derivatives = limb_coefficient_derivatives(
            result.integration.moments, source_radius, limb_c, limb_d, mode);
        rung.jacobian[5] = limb_derivatives[0];
        rung.jacobian[6] = limb_derivatives[1];
        rung.support_valid = !(result.overflow || result.root_failure);
    } else {
        const auto result = cartesian_epoch_kernel_from_prepared(
            source_radius / static_cast<double>(resolution),
            source_x, source_y, separation, mass_ratio, source_radius,
            limb_d, tile_size, tile_capacity, mode,
            boundary_subdivision, prepared);
        rung.magnification = combine_magnification(
            result.integration.moments, source_radius, limb_c, limb_d, mode);
        rung.support_valid = !(result.overflow || result.root_failure);
    }
    return rung;
}

template <bool WithJacobian>
CartesianLadderDecision cartesian_ladder_epoch_kernel(
    double source_x,
    double source_y,
    double separation,
    double mass_ratio,
    double source_radius,
    double limb_c,
    double limb_d,
    double absolute_tolerance,
    double relative_tolerance,
    std::int32_t selected_index,
    const std::int32_t* resolutions,
    const std::int32_t* tile_capacities,
    std::int64_t bucket_count,
    std::int64_t tile_size,
    std::int64_t limb_samples,
    MomentMode mode,
    std::int64_t boundary_subdivision)
{
    const auto prepared = cached_cartesian_seed_support(
        source_x, source_y, separation, mass_ratio, source_radius,
        limb_samples);
    const auto evaluate = [&](std::int32_t index) {
        return evaluate_cartesian_ladder_rung<WithJacobian>(
            source_x, source_y, separation, mass_ratio, source_radius,
            limb_c, limb_d, resolutions[index], tile_capacities[index],
            tile_size, mode, boundary_subdivision, prepared);
    };
    const auto absolute_difference = [](
        const CartesianLadderRung& left,
        const CartesianLadderRung& right) {
        CartesianLadderRung difference;
        const double signed_difference =
            left.magnification - right.magnification;
        difference.magnification = std::abs(signed_difference);
        // jnp.abs uses the +1 subgradient at exactly zero.  Matching it keeps
        // the fused custom JVP bit-for-bit consistent at equal rungs.
        const double sign = signed_difference < 0.0 ? -1.0 : 1.0;
        if constexpr (WithJacobian) {
            for (std::size_t parameter = 0; parameter < parameter_count;
                 ++parameter) {
                difference.jacobian[parameter] = sign * (
                    left.jacobian[parameter] - right.jacobian[parameter]);
            }
        }
        return difference;
    };

    const std::int32_t lower_index = std::max(selected_index - 1, 0);
    const std::int32_t upper_index = std::min(
        selected_index + 1, static_cast<std::int32_t>(bucket_count - 1));
    const CartesianLadderRung selected = evaluate(selected_index);
    const CartesianLadderRung lower = lower_index == selected_index
        ? selected
        : evaluate(lower_index);
    const bool use_upper =
        selected_index == 0 || !lower.support_valid;
    const auto lower_error = absolute_difference(selected, lower);
    const double lower_budget =
        absolute_tolerance
        + relative_tolerance * std::max(std::abs(selected.magnification), 1.0);
    const bool lower_certified =
        !use_upper
        && selected.support_valid
        && lower.support_valid
        && std::isfinite(lower_error.magnification)
        && lower_error.magnification <= lower_budget;
    const bool at_top = upper_index == selected_index;
    const bool needs_upper = !lower_certified && !at_top;
    const CartesianLadderRung upper = at_top
        ? selected
        : (needs_upper ? evaluate(upper_index) : CartesianLadderRung{});
    const std::int32_t comparison_index =
        use_upper ? upper_index : lower_index;
    const CartesianLadderRung& comparison = use_upper ? upper : lower;
    const auto initial_error = absolute_difference(selected, comparison);
    const double initial_budget =
        absolute_tolerance
        + relative_tolerance * std::max(std::abs(selected.magnification), 1.0);
    const bool initial_converged =
        comparison_index != selected_index
        && selected.support_valid
        && comparison.support_valid
        && std::isfinite(initial_error.magnification)
        && initial_error.magnification <= initial_budget;
    const bool retry = !initial_converged && upper_index != selected_index;
    const auto retry_error = absolute_difference(upper, selected);

    CartesianLadderDecision decision;
    const CartesianLadderRung& chosen = retry ? upper : selected;
    const CartesianLadderRung& error = retry ? retry_error : initial_error;
    decision.magnification = chosen.magnification;
    decision.estimated_error = error.magnification;
    decision.support_valid = chosen.support_valid;
    const double retry_budget =
        absolute_tolerance
        + relative_tolerance * std::max(std::abs(chosen.magnification), 1.0);
    decision.converged = initial_converged || (
        retry
        && selected.support_valid
        && upper.support_valid
        && std::isfinite(retry_error.magnification)
        && retry_error.magnification <= retry_budget);
    if constexpr (WithJacobian) {
        decision.magnification_jacobian = chosen.jacobian;
        decision.error_jacobian = error.jacobian;
    }
    return decision;
}

ffi::Error validate_cartesian_ladder_arguments(
    std::int64_t tile_size,
    std::int64_t limb_samples,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR1<ffi::F64>& source_x,
    ffi::BufferR1<ffi::F64>& source_y,
    ffi::BufferR1<ffi::PRED>& active,
    ffi::BufferR1<ffi::S32>& selected_index,
    ffi::BufferR1<ffi::S32>& resolutions,
    ffi::BufferR1<ffi::S32>& tile_capacities,
    ffi::ResultBufferR1<ffi::F64>& magnification,
    ffi::ResultBufferR1<ffi::F64>& estimated_error,
    ffi::ResultBufferR1<ffi::PRED>& support_valid,
    ffi::ResultBufferR1<ffi::PRED>& converged)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    const std::int64_t bucket_count = resolutions.dimensions()[0];
    if (
        source_y.dimensions()[0] != batch_size
        || active.dimensions()[0] != batch_size
        || selected_index.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || estimated_error->dimensions()[0] != batch_size
        || support_valid->dimensions()[0] != batch_size
        || converged->dimensions()[0] != batch_size
        || tile_capacities.dimensions()[0] != bucket_count
        || bucket_count <= 0) {
        return ffi::Error::InvalidArgument(
            "Cartesian ladder arrays have incompatible shapes");
    }
    if (
        tile_size <= 0 || limb_samples <= 0
        || mode_value < 1 || mode_value > 3
        || boundary_subdivision < 1 || boundary_subdivision > 4) {
        return ffi::Error::InvalidArgument(
            "invalid Cartesian ladder static configuration");
    }
    for (std::int64_t index = 0; index < bucket_count; ++index) {
        if (
            resolutions.typed_data()[index] <= 0
            || tile_capacities.typed_data()[index] <= 0) {
            return ffi::Error::InvalidArgument(
                "Cartesian ladder buckets must be positive");
        }
    }
    for (std::int64_t index = 0; index < batch_size; ++index) {
        if (
            selected_index.typed_data()[index] < 0
            || selected_index.typed_data()[index] >= bucket_count) {
            return ffi::Error::InvalidArgument(
                "Cartesian ladder selected index is out of range");
        }
    }
    return ffi::Error::Success();
}

#define LCBININT_CARTESIAN_LADDER_ARGUMENTS \
    std::int64_t tile_size, \
    std::int64_t limb_samples, \
    std::int64_t mode_value, \
    std::int64_t boundary_subdivision, \
    ffi::BufferR1<ffi::F64> source_x, \
    ffi::BufferR1<ffi::F64> source_y, \
    ffi::BufferR1<ffi::PRED> active, \
    ffi::BufferR1<ffi::S32> selected_index, \
    ffi::BufferR1<ffi::S32> resolutions, \
    ffi::BufferR1<ffi::S32> tile_capacities, \
    ffi::BufferR0<ffi::F64> separation, \
    ffi::BufferR0<ffi::F64> mass_ratio, \
    ffi::BufferR0<ffi::F64> source_radius, \
    ffi::BufferR0<ffi::F64> limb_c, \
    ffi::BufferR0<ffi::F64> limb_d, \
    ffi::BufferR0<ffi::F64> absolute_tolerance, \
    ffi::BufferR0<ffi::F64> relative_tolerance, \
    ffi::ResultBufferR1<ffi::F64> magnification, \
    ffi::ResultBufferR1<ffi::F64> estimated_error, \
    ffi::ResultBufferR1<ffi::PRED> support_valid, \
    ffi::ResultBufferR1<ffi::PRED> converged

#define LCBININT_CARTESIAN_LADDER_VALIDATE \
    validate_cartesian_ladder_arguments( \
        tile_size, limb_samples, mode_value, boundary_subdivision, \
        source_x, source_y, active, selected_index, resolutions, \
        tile_capacities, magnification, estimated_error, support_valid, \
        converged)

#define LCBININT_CARTESIAN_LADDER_KERNEL(index, with_jacobian) \
    cartesian_ladder_epoch_kernel<with_jacobian>( \
        source_x.typed_data()[index], source_y.typed_data()[index], \
        *separation.typed_data(), *mass_ratio.typed_data(), \
        *source_radius.typed_data(), *limb_c.typed_data(), \
        *limb_d.typed_data(), *absolute_tolerance.typed_data(), \
        *relative_tolerance.typed_data(), selected_index.typed_data()[index], \
        resolutions.typed_data(), tile_capacities.typed_data(), \
        resolutions.dimensions()[0], tile_size, limb_samples, \
        static_cast<MomentMode>(mode_value), boundary_subdivision)

ffi::Error cartesian_ladder_forward_ffi_impl(
    LCBININT_CARTESIAN_LADDER_ARGUMENTS)
{
    auto validation = LCBININT_CARTESIAN_LADDER_VALIDATE;
    if (validation.failure()) return validation;
    if (!(*source_radius.typed_data() > 0.0)) {
        return ffi::Error::InvalidArgument("source_radius must be positive");
    }
    const std::int64_t batch_size = source_x.dimensions()[0];
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1, std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            estimated_error->typed_data()[index] =
                std::numeric_limits<double>::infinity();
            support_valid->typed_data()[index] = false;
            converged->typed_data()[index] = false;
            continue;
        }
        const auto result = LCBININT_CARTESIAN_LADDER_KERNEL(index, false);
        magnification->typed_data()[index] = result.magnification;
        estimated_error->typed_data()[index] = result.estimated_error;
        support_valid->typed_data()[index] = result.support_valid;
        converged->typed_data()[index] = result.converged;
    }
    return ffi::Error::Success();
}

ffi::Error cartesian_ladder_jacobian_ffi_impl(
    LCBININT_CARTESIAN_LADDER_ARGUMENTS,
    ffi::ResultBufferR2<ffi::F64> magnification_jacobian,
    ffi::ResultBufferR2<ffi::F64> error_jacobian)
{
    auto validation = LCBININT_CARTESIAN_LADDER_VALIDATE;
    if (validation.failure()) return validation;
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        magnification_jacobian->dimensions()[0] != batch_size
        || magnification_jacobian->dimensions()[1] != parameter_count
        || error_jacobian->dimensions()[0] != batch_size
        || error_jacobian->dimensions()[1] != parameter_count) {
        return ffi::Error::InvalidArgument(
            "Cartesian ladder Jacobians must have shape (N, 7)");
    }
    if (!(*source_radius.typed_data() > 0.0)) {
        return ffi::Error::InvalidArgument("source_radius must be positive");
    }
#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1, std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        double* output_magnification_jacobian =
            magnification_jacobian->typed_data() + index * parameter_count;
        double* output_error_jacobian =
            error_jacobian->typed_data() + index * parameter_count;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            estimated_error->typed_data()[index] =
                std::numeric_limits<double>::infinity();
            support_valid->typed_data()[index] = false;
            converged->typed_data()[index] = false;
            std::fill(
                output_magnification_jacobian,
                output_magnification_jacobian + parameter_count, 0.0);
            std::fill(
                output_error_jacobian,
                output_error_jacobian + parameter_count, 0.0);
            continue;
        }
        const auto result = LCBININT_CARTESIAN_LADDER_KERNEL(index, true);
        magnification->typed_data()[index] = result.magnification;
        estimated_error->typed_data()[index] = result.estimated_error;
        support_valid->typed_data()[index] = result.support_valid;
        converged->typed_data()[index] = result.converged;
        std::copy(
            result.magnification_jacobian.begin(),
            result.magnification_jacobian.end(),
            output_magnification_jacobian);
        std::copy(
            result.error_jacobian.begin(), result.error_jacobian.end(),
            output_error_jacobian);
    }
    return ffi::Error::Success();
}

#define LCBININT_CARTESIAN_LADDER_BINDING \
    ffi::Ffi::Bind() \
        .Attr<std::int64_t>("tile_size") \
        .Attr<std::int64_t>("limb_samples") \
        .Attr<std::int64_t>("moment_mode") \
        .Attr<std::int64_t>("boundary_subdivision") \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::F64>>() \
        .Arg<ffi::BufferR1<ffi::PRED>>() \
        .Arg<ffi::BufferR1<ffi::S32>>() \
        .Arg<ffi::BufferR1<ffi::S32>>() \
        .Arg<ffi::BufferR1<ffi::S32>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Arg<ffi::BufferR0<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::F64>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>() \
        .Ret<ffi::BufferR1<ffi::PRED>>()

XLA_FFI_DEFINE_HANDLER(
    cartesian_ladder_forward_ffi_handler,
    cartesian_ladder_forward_ffi_impl,
    LCBININT_CARTESIAN_LADDER_BINDING);
XLA_FFI_DEFINE_HANDLER(
    cartesian_ladder_jacobian_ffi_handler,
    cartesian_ladder_jacobian_ffi_impl,
    LCBININT_CARTESIAN_LADDER_BINDING
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>());

#undef LCBININT_CARTESIAN_LADDER_BINDING
#undef LCBININT_CARTESIAN_LADDER_KERNEL
#undef LCBININT_CARTESIAN_LADDER_VALIDATE
#undef LCBININT_CARTESIAN_LADDER_ARGUMENTS

ffi::Error validate_cartesian_batch_arguments(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR1<ffi::F64>& source_x,
    ffi::BufferR1<ffi::F64>& source_y,
    ffi::BufferR1<ffi::PRED>& active,
    ffi::ResultBufferR1<ffi::F64>& magnification,
    ffi::ResultBufferR2<ffi::F64>& moments)
{
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        source_y.dimensions()[0] != batch_size
        || active.dimensions()[0] != batch_size
        || magnification->dimensions()[0] != batch_size
        || moments->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "Cartesian batch arrays must have a common leading dimension");
    }
    if (
        mode_value < 1 || mode_value > 3
        || moments->dimensions()[1]
            != moment_count(static_cast<MomentMode>(mode_value))) {
        return ffi::Error::InvalidArgument(
            "Cartesian batch moments have an invalid shape");
    }
    if (
        tile_size <= 0 || tile_capacity <= 0 || limb_samples <= 0
        || boundary_subdivision < 1 || boundary_subdivision > 8
        || (boundary_subdivision > 4 && boundary_subdivision != 8)) {
        return ffi::Error::InvalidArgument(
            "invalid Cartesian batch static configuration");
    }
    return ffi::Error::Success();
}

ffi::Error cartesian_batch_forward_ffi_impl(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> cell_size,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR2<ffi::F64> moments,
    ffi::ResultBufferR1<ffi::S32> boundary_cells,
    ffi::ResultBufferR1<ffi::S32> active_cells,
    ffi::ResultBufferR1<ffi::S32> visited_tiles,
    ffi::ResultBufferR1<ffi::PRED> overflow,
    ffi::ResultBufferR1<ffi::PRED> root_failure)
{
    auto validation = validate_cartesian_batch_arguments(
        tile_size, tile_capacity, limb_samples, mode_value,
        boundary_subdivision, source_x, source_y, active, magnification,
        moments);
    if (validation.failure()) return validation;
    if (!(*cell_size.typed_data() > 0.0)
        || !(*source_radius.typed_data() > 0.0)) {
        return ffi::Error::InvalidArgument(
            "cell_size and source_radius must be positive");
    }
    const std::int64_t batch_size = source_x.dimensions()[0];
    if (
        boundary_cells->dimensions()[0] != batch_size
        || active_cells->dimensions()[0] != batch_size
        || visited_tiles->dimensions()[0] != batch_size
        || overflow->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size) {
        return ffi::Error::InvalidArgument(
            "Cartesian batch diagnostic outputs have invalid shapes");
    }
    const auto mode = static_cast<MomentMode>(mode_value);
    const int output_moment_count = moment_count(mode);

#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1,
        std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        double* output_moments =
            moments->typed_data() + index * output_moment_count;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            std::fill(
                output_moments,
                output_moments + output_moment_count,
                0.0);
            boundary_cells->typed_data()[index] = 0;
            active_cells->typed_data()[index] = 0;
            visited_tiles->typed_data()[index] = 0;
            overflow->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            continue;
        }
        const auto result = cartesian_epoch_kernel(
            *cell_size.typed_data(), source_x.typed_data()[index],
            source_y.typed_data()[index], *separation.typed_data(),
            *mass_ratio.typed_data(), *source_radius.typed_data(),
            *limb_d.typed_data(), tile_size, tile_capacity, limb_samples,
            mode, boundary_subdivision);
        for (int moment = 0; moment < output_moment_count; ++moment) {
            output_moments[moment] = result.integration.moments[moment];
        }
        magnification->typed_data()[index] = combine_magnification(
            result.integration.moments, *source_radius.typed_data(),
            *limb_c.typed_data(), *limb_d.typed_data(), mode);
        boundary_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.boundary_cells);
        active_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.active_cells);
        visited_tiles->typed_data()[index] = result.tile_count;
        overflow->typed_data()[index] = result.overflow;
        root_failure->typed_data()[index] = result.root_failure;
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    cartesian_batch_forward_ffi_handler,
    cartesian_batch_forward_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("tile_size")
        .Attr<std::int64_t>("tile_capacity")
        .Attr<std::int64_t>("limb_samples")
        .Attr<std::int64_t>("moment_mode")
        .Attr<std::int64_t>("boundary_subdivision")
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::PRED>>());

ffi::Error cartesian_batch_value_jacobian_ffi_impl(
    std::int64_t tile_size,
    std::int64_t tile_capacity,
    std::int64_t limb_samples,
    std::int64_t mode_value,
    std::int64_t boundary_subdivision,
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::PRED> active,
    ffi::BufferR0<ffi::F64> cell_size,
    ffi::BufferR0<ffi::F64> separation,
    ffi::BufferR0<ffi::F64> mass_ratio,
    ffi::BufferR0<ffi::F64> source_radius,
    ffi::BufferR0<ffi::F64> limb_c,
    ffi::BufferR0<ffi::F64> limb_d,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR2<ffi::F64> moments,
    ffi::ResultBufferR1<ffi::S32> boundary_cells,
    ffi::ResultBufferR1<ffi::S32> active_cells,
    ffi::ResultBufferR1<ffi::S32> visited_tiles,
    ffi::ResultBufferR1<ffi::PRED> overflow,
    ffi::ResultBufferR1<ffi::PRED> root_failure,
    ffi::ResultBufferR2<ffi::F64> magnification_jacobian,
    ffi::ResultBufferR3<ffi::F64> moments_jacobian)
{
    auto validation = validate_cartesian_batch_arguments(
        tile_size, tile_capacity, limb_samples, mode_value,
        boundary_subdivision, source_x, source_y, active, magnification,
        moments);
    if (validation.failure()) return validation;
    if (!(*cell_size.typed_data() > 0.0)
        || !(*source_radius.typed_data() > 0.0)) {
        return ffi::Error::InvalidArgument(
            "cell_size and source_radius must be positive");
    }
    const std::int64_t batch_size = source_x.dimensions()[0];
    const auto mode = static_cast<MomentMode>(mode_value);
    const int output_moment_count = moment_count(mode);
    const auto moment_jacobian_dimensions = moments_jacobian->dimensions();
    if (
        boundary_cells->dimensions()[0] != batch_size
        || active_cells->dimensions()[0] != batch_size
        || visited_tiles->dimensions()[0] != batch_size
        || overflow->dimensions()[0] != batch_size
        || root_failure->dimensions()[0] != batch_size
        || magnification_jacobian->dimensions()[0] != batch_size
        || magnification_jacobian->dimensions()[1] != parameter_count
        || moment_jacobian_dimensions[0] != batch_size
        || moment_jacobian_dimensions[1] != output_moment_count
        || moment_jacobian_dimensions[2] != parameter_count) {
        return ffi::Error::InvalidArgument(
            "Cartesian batch Jacobian outputs have invalid shapes");
    }

#ifdef LCBININT_HAS_OPENMP
    const int batch_threads = std::max(
        1,
        std::min(
            {static_cast<int>(batch_size), omp_get_max_threads(), 32}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(batch_threads) if (batch_threads > 1)
#endif
    for (std::int64_t index = 0; index < batch_size; ++index) {
        double* output_moments =
            moments->typed_data() + index * output_moment_count;
        double* output_magnification_jacobian =
            magnification_jacobian->typed_data() + index * parameter_count;
        double* output_moments_jacobian =
            moments_jacobian->typed_data()
            + index * output_moment_count * parameter_count;
        if (!active.typed_data()[index]) {
            magnification->typed_data()[index] = 0.0;
            std::fill(
                output_moments,
                output_moments + output_moment_count,
                0.0);
            std::fill(
                output_magnification_jacobian,
                output_magnification_jacobian + parameter_count,
                0.0);
            std::fill(
                output_moments_jacobian,
                output_moments_jacobian
                    + output_moment_count * parameter_count,
                0.0);
            boundary_cells->typed_data()[index] = 0;
            active_cells->typed_data()[index] = 0;
            visited_tiles->typed_data()[index] = 0;
            overflow->typed_data()[index] = false;
            root_failure->typed_data()[index] = false;
            continue;
        }
        const Jet source_x_jet =
            Jet::variable(source_x.typed_data()[index], 0);
        const Jet source_y_jet =
            Jet::variable(source_y.typed_data()[index], 1);
        const Jet separation_jet =
            Jet::variable(*separation.typed_data(), 2);
        const Jet mass_ratio_jet =
            Jet::variable(*mass_ratio.typed_data(), 3);
        const Jet source_radius_jet =
            Jet::variable(*source_radius.typed_data(), 4);
        const Jet limb_c_jet(*limb_c.typed_data());
        const Jet limb_d_jet(*limb_d.typed_data());
        const auto result = cartesian_epoch_kernel(
            *cell_size.typed_data(), source_x_jet, source_y_jet,
            separation_jet, mass_ratio_jet, source_radius_jet, limb_d_jet,
            tile_size, tile_capacity, limb_samples, mode,
            boundary_subdivision);
        const Jet magnification_result = combine_magnification(
            result.integration.moments, source_radius_jet, limb_c_jet,
            limb_d_jet, mode);
        const auto limb_derivatives = limb_coefficient_derivatives(
            result.integration.moments, *source_radius.typed_data(),
            *limb_c.typed_data(), *limb_d.typed_data(), mode);
        magnification->typed_data()[index] = magnification_result.value;
        for (
            std::size_t parameter = 0;
            parameter < kernel_derivative_count;
            ++parameter) {
            output_magnification_jacobian[parameter] =
                magnification_result.derivative[parameter];
        }
        output_magnification_jacobian[5] = limb_derivatives[0];
        output_magnification_jacobian[6] = limb_derivatives[1];
        for (int moment = 0; moment < output_moment_count; ++moment) {
            output_moments[moment] =
                result.integration.moments[moment].value;
            for (
                std::size_t parameter = 0;
                parameter < parameter_count;
                ++parameter) {
                output_moments_jacobian[
                    moment * parameter_count + parameter] =
                    parameter < kernel_derivative_count
                    ? result.integration.moments[moment]
                          .derivative[parameter]
                    : 0.0;
            }
        }
        boundary_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.boundary_cells);
        active_cells->typed_data()[index] =
            static_cast<std::int32_t>(result.integration.active_cells);
        visited_tiles->typed_data()[index] = result.tile_count;
        overflow->typed_data()[index] = result.overflow;
        root_failure->typed_data()[index] = result.root_failure;
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    cartesian_batch_value_jacobian_ffi_handler,
    cartesian_batch_value_jacobian_ffi_impl,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("tile_size")
        .Attr<std::int64_t>("tile_capacity")
        .Attr<std::int64_t>("limb_samples")
        .Attr<std::int64_t>("moment_mode")
        .Attr<std::int64_t>("boundary_subdivision")
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR3<ffi::F64>>());

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

py::capsule triple_image_roots_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(triple_image_roots_ffi_handler));
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

py::capsule cartesian_epoch_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(cartesian_epoch_forward_ffi_handler));
}

py::capsule cartesian_epoch_value_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            cartesian_epoch_value_jacobian_ffi_handler));
}

py::capsule triple_cartesian_epoch_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_cartesian_epoch_forward_ffi_handler));
}

py::capsule triple_cartesian_epoch_value_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_cartesian_epoch_value_jacobian_ffi_handler));
}

py::capsule triple_cartesian_batch_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_cartesian_batch_forward_ffi_handler));
}

py::capsule triple_cartesian_batch_value_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_cartesian_batch_value_jacobian_ffi_handler));
}

py::capsule cartesian_batch_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(cartesian_batch_forward_ffi_handler));
}

py::capsule cartesian_batch_value_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            cartesian_batch_value_jacobian_ffi_handler));
}

py::capsule cartesian_ladder_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(cartesian_ladder_forward_ffi_handler));
}

py::capsule cartesian_ladder_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(cartesian_ladder_jacobian_ffi_handler));
}

py::capsule hexadecapole_batch_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(hexadecapole_batch_ffi_handler));
}

py::capsule point_batch_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(point_batch_ffi_handler));
}

py::capsule point_batch_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(point_batch_jacobian_ffi_handler));
}

py::capsule hexadecapole_batch_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(hexadecapole_batch_jacobian_ffi_handler));
}

py::capsule triple_hexadecapole_batch_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(triple_hexadecapole_batch_ffi_handler));
}

py::capsule triple_hexadecapole_batch_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_hexadecapole_batch_jacobian_ffi_handler));
}

py::capsule triple_point_batch_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(triple_point_batch_ffi_handler));
}

py::capsule triple_point_batch_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_point_batch_jacobian_ffi_handler));
}

py::capsule triple_caustic_distance_batch_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_caustic_distance_batch_ffi_handler));
}

py::capsule binary_caustic_distance_batch_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            binary_caustic_distance_batch_ffi_handler));
}

py::capsule binary_routing_diagnostics_batch_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            binary_routing_diagnostics_batch_ffi_handler));
}

py::capsule polar_epoch_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(polar_epoch_forward_ffi_handler));
}

py::capsule polar_epoch_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(polar_epoch_jacobian_ffi_handler));
}

py::capsule polar_epoch_directional_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(polar_epoch_directional_ffi_handler));
}

py::capsule triple_polar_batch_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_polar_batch_forward_ffi_handler));
}

py::capsule triple_polar_batch_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(
            triple_polar_batch_jacobian_ffi_handler));
}

py::capsule trajectory_forward_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(trajectory_forward_ffi_handler));
}

py::capsule trajectory_jacobian_ffi_capsule()
{
    return py::capsule(
        reinterpret_cast<void*>(trajectory_jacobian_ffi_handler));
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
        "triple_image_roots_ffi",
        &triple_image_roots_ffi_capsule,
        "Return the typed XLA triple-image root FFI handler capsule.");
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
    module.def(
        "cartesian_epoch_forward_ffi",
        &cartesian_epoch_forward_ffi_capsule,
        "Return the fused Cartesian epoch FFI handler capsule.");
    module.def(
        "cartesian_epoch_value_jacobian_ffi",
        &cartesian_epoch_value_jacobian_ffi_capsule,
        "Return the fused Cartesian epoch value/Jacobian FFI capsule.");
    module.def(
        "triple_cartesian_epoch_forward_ffi",
        &triple_cartesian_epoch_forward_ffi_capsule,
        "Return the fused triple Cartesian epoch FFI capsule.");
    module.def(
        "triple_cartesian_epoch_value_jacobian_ffi",
        &triple_cartesian_epoch_value_jacobian_ffi_capsule,
        "Return the fused triple Cartesian value/Jacobian FFI capsule.");
    module.def(
        "triple_cartesian_batch_forward_ffi",
        &triple_cartesian_batch_forward_ffi_capsule,
        "Return the fused triple Cartesian batch FFI capsule.");
    module.def(
        "triple_cartesian_batch_value_jacobian_ffi",
        &triple_cartesian_batch_value_jacobian_ffi_capsule,
        "Return the fused triple Cartesian batch Jacobian FFI capsule.");
    module.def(
        "cartesian_batch_forward_ffi",
        &cartesian_batch_forward_ffi_capsule,
        "Return the masked Cartesian batch FFI handler capsule.");
    module.def(
        "cartesian_batch_value_jacobian_ffi",
        &cartesian_batch_value_jacobian_ffi_capsule,
        "Return the masked Cartesian batch value/Jacobian FFI capsule.");
    module.def(
        "cartesian_ladder_forward_ffi",
        &cartesian_ladder_forward_ffi_capsule,
        "Return the adaptive multi-resolution Cartesian FFI capsule.");
    module.def(
        "cartesian_ladder_jacobian_ffi",
        &cartesian_ladder_jacobian_ffi_capsule,
        "Return the adaptive Cartesian value/error Jacobian FFI capsule.");
    module.def(
        "hexadecapole_batch_ffi",
        &hexadecapole_batch_ffi_capsule,
        "Return the batched hexadecapole FFI capsule.");
    module.def(
        "hexadecapole_batch_jacobian_ffi",
        &hexadecapole_batch_jacobian_ffi_capsule,
        "Return the batched hexadecapole value/Jacobian FFI capsule.");
    module.def(
        "point_batch_ffi",
        &point_batch_ffi_capsule,
        "Return the batched binary point-source FFI capsule.");
    module.def(
        "point_batch_jacobian_ffi",
        &point_batch_jacobian_ffi_capsule,
        "Return the batched binary point-source Jacobian FFI capsule.");
    module.def(
        "triple_hexadecapole_batch_ffi",
        &triple_hexadecapole_batch_ffi_capsule,
        "Return the batched triple hexadecapole FFI capsule.");
    module.def(
        "triple_hexadecapole_batch_jacobian_ffi",
        &triple_hexadecapole_batch_jacobian_ffi_capsule,
        "Return the batched triple hexadecapole Jacobian FFI capsule.");
    module.def(
        "triple_point_batch_ffi",
        &triple_point_batch_ffi_capsule,
        "Return the batched triple point-source FFI capsule.");
    module.def(
        "triple_point_batch_jacobian_ffi",
        &triple_point_batch_jacobian_ffi_capsule,
        "Return the batched triple point-source Jacobian FFI capsule.");
    module.def(
        "triple_caustic_distance_batch_ffi",
        &triple_caustic_distance_batch_ffi_capsule,
        "Return the batched triple caustic-distance FFI capsule.");
    module.def(
        "binary_caustic_distance_batch_ffi",
        &binary_caustic_distance_batch_ffi_capsule,
        "Return the batched binary caustic-distance FFI capsule.");
    module.def(
        "binary_routing_diagnostics_batch_ffi",
        &binary_routing_diagnostics_batch_ffi_capsule,
        "Return the batched binary native-routing diagnostics FFI capsule.");
    module.def(
        "polar_epoch_forward_ffi",
        &polar_epoch_forward_ffi_capsule,
        "Return the fused polar epoch FFI capsule.");
    module.def(
        "polar_epoch_jacobian_ffi",
        &polar_epoch_jacobian_ffi_capsule,
        "Return the fused polar epoch value/Jacobian FFI capsule.");
    module.def(
        "polar_epoch_directional_ffi",
        &polar_epoch_directional_ffi_capsule,
        "Return the fused polar epoch directional-JVP FFI capsule.");
    module.def(
        "triple_polar_batch_forward_ffi",
        &triple_polar_batch_forward_ffi_capsule,
        "Return the masked triple polar batch FFI capsule.");
    module.def(
        "triple_polar_batch_jacobian_ffi",
        &triple_polar_batch_jacobian_ffi_capsule,
        "Return the masked triple polar batch Jacobian FFI capsule.");
    module.def(
        "trajectory_forward_ffi",
        &trajectory_forward_ffi_capsule,
        "Return the integrated trajectory dispatcher FFI capsule.");
    module.def(
        "trajectory_jacobian_ffi",
        &trajectory_jacobian_ffi_capsule,
        "Return the integrated trajectory dispatcher Jacobian FFI capsule.");
#endif
}

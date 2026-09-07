#pragma once

// Forward-mode analytic derivatives for the experimental polar
// algebraic-boundary finite-source kernel.
//
// The primal (`experimental_algebraic_boundary_integral`) runs unchanged in
// `double` and fixes every discrete decision (band count, interval
// classification, real-quartic-vs-Skowron path, adaptive panel partition).
// This header provides the smooth pieces -- the boundary trig-quartic and its
// angular derivatives -- templated on a scalar type so the same code path can
// be re-evaluated with a forward-mode `ForwardJet` to carry parameter
// derivatives d(.)/d(source_x, source_y, rho, q, separation).
//
// See docs/design/algebraic-boundary-autodiff-20260828.md.

#include "lcbinint/magnification/algebraic_boundary.hpp"
#include "lcbinint/types.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace lcbinint::magnification::autodiff {

// ---------------------------------------------------------------------------
// Minimal forward-mode dual number.
// ---------------------------------------------------------------------------

template <std::size_t N>
struct ForwardJet {
    double value = 0.0;
    std::array<double, N> derivative{};

    ForwardJet() = default;
    ForwardJet(double scalar) : value(scalar) {}  // NOLINT: implicit on purpose

    static ForwardJet variable(double scalar, std::size_t index)
    {
        ForwardJet result(scalar);
        result.derivative[index] = 1.0;
        return result;
    }
};

template <std::size_t N>
ForwardJet<N> operator+(const ForwardJet<N>& a, const ForwardJet<N>& b)
{
    ForwardJet<N> r(a.value + b.value);
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = a.derivative[i] + b.derivative[i];
    return r;
}

template <std::size_t N>
ForwardJet<N> operator-(const ForwardJet<N>& a, const ForwardJet<N>& b)
{
    ForwardJet<N> r(a.value - b.value);
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = a.derivative[i] - b.derivative[i];
    return r;
}

template <std::size_t N>
ForwardJet<N> operator-(const ForwardJet<N>& a)
{
    ForwardJet<N> r(-a.value);
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = -a.derivative[i];
    return r;
}

template <std::size_t N>
ForwardJet<N> operator*(const ForwardJet<N>& a, const ForwardJet<N>& b)
{
    ForwardJet<N> r(a.value * b.value);
    for (std::size_t i = 0; i < N; ++i)
        r.derivative[i] = a.derivative[i] * b.value + a.value * b.derivative[i];
    return r;
}

template <std::size_t N>
ForwardJet<N> operator/(const ForwardJet<N>& a, const ForwardJet<N>& b)
{
    ForwardJet<N> r(a.value / b.value);
    const double inv2 = 1.0 / (b.value * b.value);
    for (std::size_t i = 0; i < N; ++i)
        r.derivative[i] = (a.derivative[i] * b.value - a.value * b.derivative[i]) * inv2;
    return r;
}

// double op jet / jet op double -- the implicit ForwardJet(double) ctor makes
// the jet/jet overloads applicable, but providing these avoids fabricating a
// zero-derivative temporary in the hot path.
template <std::size_t N>
ForwardJet<N> operator+(const ForwardJet<N>& a, double b) { ForwardJet<N> r(a); r.value += b; return r; }
template <std::size_t N>
ForwardJet<N> operator+(double a, const ForwardJet<N>& b) { return b + a; }
template <std::size_t N>
ForwardJet<N> operator-(const ForwardJet<N>& a, double b) { ForwardJet<N> r(a); r.value -= b; return r; }
template <std::size_t N>
ForwardJet<N> operator-(double a, const ForwardJet<N>& b) { return (-b) + a; }
template <std::size_t N>
ForwardJet<N> operator*(const ForwardJet<N>& a, double b)
{
    ForwardJet<N> r(a.value * b);
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = a.derivative[i] * b;
    return r;
}
template <std::size_t N>
ForwardJet<N> operator*(double a, const ForwardJet<N>& b) { return b * a; }
template <std::size_t N>
ForwardJet<N> operator/(const ForwardJet<N>& a, double b) { return a * (1.0 / b); }
template <std::size_t N>
ForwardJet<N> operator/(double a, const ForwardJet<N>& b)
{
    ForwardJet<N> r(a / b.value);
    const double factor = -a / (b.value * b.value);
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = factor * b.derivative[i];
    return r;
}

// ---------------------------------------------------------------------------
// Elementary functions. Overloaded for `double` so the templated core also
// compiles as the plain primal.
// ---------------------------------------------------------------------------

inline double scalar_sin(double x) { return std::sin(x); }
inline double scalar_cos(double x) { return std::cos(x); }
inline double scalar_sqrt(double x) { return std::sqrt(x); }
inline double scalar_pow(double x, double p) { return std::pow(x, p); }

template <std::size_t N>
ForwardJet<N> scalar_sin(const ForwardJet<N>& a)
{
    ForwardJet<N> r(std::sin(a.value));
    const double c = std::cos(a.value);
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = c * a.derivative[i];
    return r;
}

template <std::size_t N>
ForwardJet<N> scalar_cos(const ForwardJet<N>& a)
{
    ForwardJet<N> r(std::cos(a.value));
    const double s = -std::sin(a.value);
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = s * a.derivative[i];
    return r;
}

template <std::size_t N>
ForwardJet<N> scalar_sqrt(const ForwardJet<N>& a)
{
    const double root = std::sqrt(a.value);
    ForwardJet<N> r(root);
    const double factor = root > 0.0 ? 0.5 / root : 0.0;
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = factor * a.derivative[i];
    return r;
}

template <std::size_t N>
ForwardJet<N> scalar_pow(const ForwardJet<N>& a, double p)
{
    const double v = std::pow(a.value, p);
    ForwardJet<N> r(v);
    const double factor = a.value != 0.0 ? p * v / a.value : 0.0;
    for (std::size_t i = 0; i < N; ++i) r.derivative[i] = factor * a.derivative[i];
    return r;
}

// ---------------------------------------------------------------------------
// Lens geometry in HMC parameters.
//
// `make_binary_lens_mapper` folds the raw inputs through
//   s = |separation|,  q = q_raw<1 ? q_raw : 1/q_raw,  a = (q_raw<1 ? -s : s),
//   m1 = 1/(1+q),  m2 = q*m1.
// The two branches (separation sign, q<1) are discrete: the primal decides
// them and passes the outcome in as `separation_sign` / `q_below_one`.
// ---------------------------------------------------------------------------

struct FrozenLensBranch {
    double separation_sign = 1.0;  // sign(separation_raw)
    bool q_below_one = true;       // |q_raw| < 1
};

template <typename Scalar>
struct BinaryGeometry {
    Scalar a;    // signed lens separation in the native frame
    Scalar m1;
    Scalar m2;
    Scalar zeta_x;  // shifted source: source_x + a*m1
    Scalar zeta_y;  // source_y
    Scalar rho;
    Scalar src_x;   // raw source_x (map_lens folds the -a*m1 shift itself)
    Scalar src_y;   // raw source_y
};

// params: [source_x, source_y, rho, q_raw, separation_raw]
template <typename Scalar>
BinaryGeometry<Scalar> make_geometry(
    const std::array<Scalar, 5>& params, const FrozenLensBranch& branch)
{
    const Scalar& source_x = params[0];
    const Scalar& source_y = params[1];
    const Scalar& rho = params[2];
    const Scalar& q_raw = params[3];
    const Scalar& separation_raw = params[4];

    const Scalar s = branch.separation_sign * separation_raw;  // |separation|
    const Scalar q = branch.q_below_one ? q_raw : (Scalar(1.0) / q_raw);
    const Scalar a = branch.q_below_one ? Scalar(-1.0) * s : s;
    const Scalar m1 = Scalar(1.0) / (Scalar(1.0) + q);
    const Scalar m2 = q * m1;

    BinaryGeometry<Scalar> g;
    g.a = a;
    g.m1 = m1;
    g.m2 = m2;
    g.zeta_x = source_x + a * m1;
    g.zeta_y = source_y;
    g.rho = rho;
    g.src_x = source_x;
    g.src_y = source_y;
    return g;
}

// Clamp-to-zero mirroring the primal `std::max(0.0, phi)`. The derivative is
// taken to be zero once the value is non-positive (the sine map's vanishing
// Jacobian already suppresses the contribution at a boundary node).
inline double scalar_max0(double v) { return v > 0.0 ? v : 0.0; }
template <std::size_t N>
ForwardJet<N> scalar_max0(const ForwardJet<N>& a)
{
    return a.value > 0.0 ? a : ForwardJet<N>(0.0);
}

// Binary lens map, templated. Mirrors `map_lens` in algebraic_boundary.cpp with
// lens {a, m1, m2} and the native -a*m1 source-frame shift folded in.
template <typename Scalar>
void map_lens_jet(
    const BinaryGeometry<Scalar>& g, const Scalar& x, const Scalar& y,
    Scalar* mx, Scalar* my)
{
    const Scalar& a = g.a;
    const Scalar xa = x - a;
    const Scalar den1 = xa * xa + y * y;
    const Scalar den2 = x * x + y * y;
    *mx = x - g.m1 * xa / den1 - g.m2 * x / den2 - a * g.m1;
    *my = y - g.m1 * y / den1 - g.m2 * y / den2;
}

// phi = 1 - |lens_map(R,theta) - source|^2 / rho^2, templated.
template <typename Scalar>
Scalar phi_value_jet(
    const BinaryGeometry<Scalar>& g, const Scalar& radius, const Scalar& angle)
{
    const Scalar x = radius * scalar_cos(angle);
    const Scalar y = radius * scalar_sin(angle);
    Scalar mx;
    Scalar my;
    map_lens_jet(g, x, y, &mx, &my);
    const Scalar dx = mx - g.src_x;
    const Scalar dy = my - g.src_y;
    const Scalar rho2 = g.rho * g.rho;
    return Scalar(1.0) - (dx * dx + dy * dy) / rho2;
}

// ---------------------------------------------------------------------------
// Boundary self-inversive trig-quartic, real form.
//
// Reproduces `algebraic_boundary_quartic_fixed` followed by
// `self_inversive_trig_quartic`, expanded from std::complex<double> to explicit
// real/imaginary parts so it templates on Scalar. The primal reads
//   A = Re(p0), B = -Im(p0), C = Re(p1), D = -Im(p1), Epsilon = p2.
// ---------------------------------------------------------------------------

template <typename Scalar>
struct BoundaryTrigForm {
    Scalar A;
    Scalar B;
    Scalar C;
    Scalar D;
    Scalar Epsilon;
};

template <typename Scalar>
BoundaryTrigForm<Scalar> make_trig_form(
    const BinaryGeometry<Scalar>& g, const Scalar& image_radius)
{
    const Scalar& R = image_radius;
    const Scalar R2 = R * R;
    const Scalar R3 = R2 * R;
    const Scalar R4 = R2 * R2;

    const Scalar& zx = g.zeta_x;
    const Scalar& zy = g.zeta_y;
    const Scalar& a = g.a;
    const Scalar rho2 = g.rho * g.rho;

    // n0 = -zeta * R^2
    const Scalar n0x = Scalar(-1.0) * zx * R2;
    const Scalar n0y = Scalar(-1.0) * zy * R2;
    // n1 = R * (R^2 - 1 + a*zeta)
    const Scalar n1x = R * (R2 - 1.0 + a * zx);
    const Scalar n1y = R * (a * zy);
    // n2 = a * (m2 - R^2)   (real)
    const Scalar n2 = a * (g.m2 - R2);
    // E = rho^2 * a * R^3   (real)
    const Scalar E = rho2 * a * R3;

    // p0 = n0 * n2
    const Scalar p0x = n0x * n2;
    const Scalar p0y = n0y * n2;
    // p1 = n0 * conj(n1) + n1 * n2 + E
    const Scalar p1x = n0x * n1x + n0y * n1y + n1x * n2 + E;
    const Scalar p1y = n0y * n1x - n0x * n1y + n1y * n2;
    // p2 = |n0|^2 + |n1|^2 + n2^2 - rho^2 (R^4 + a^2 R^2)
    const Scalar p2 =
        n0x * n0x + n0y * n0y + n1x * n1x + n1y * n1y + n2 * n2
        - rho2 * (R4 + a * a * R2);

    BoundaryTrigForm<Scalar> form;
    form.A = p0x;
    form.B = Scalar(-1.0) * p0y;
    form.C = p1x;
    form.D = Scalar(-1.0) * p1y;
    form.Epsilon = p2;
    return form;
}

// F(theta), F'(theta), F''(theta): the three entries of `evaluate_trig_quartic`.
// `theta` stays a plain double -- the crossing angle is frozen from the primal;
// its parameter derivative comes from the quartic IFT, not from carrying theta
// as a jet.
template <typename Scalar>
Scalar trig_value(const BoundaryTrigForm<Scalar>& q, double theta)
{
    const double c1 = std::cos(theta);
    const double s1 = std::sin(theta);
    const double c2 = c1 * c1 - s1 * s1;
    const double s2 = 2.0 * s1 * c1;
    return q.Epsilon + 2.0 * (q.C * c1 - q.D * s1) + 2.0 * (q.A * c2 - q.B * s2);
}

template <typename Scalar>
Scalar trig_first(const BoundaryTrigForm<Scalar>& q, double theta)
{
    const double c1 = std::cos(theta);
    const double s1 = std::sin(theta);
    const double c2 = c1 * c1 - s1 * s1;
    const double s2 = 2.0 * s1 * c1;
    return 2.0 * (Scalar(-1.0) * q.C * s1 - q.D * c1)
        + 4.0 * (Scalar(-1.0) * q.A * s2 - q.B * c2);
}

template <typename Scalar>
Scalar trig_second(const BoundaryTrigForm<Scalar>& q, double theta)
{
    const double c1 = std::cos(theta);
    const double s1 = std::sin(theta);
    const double c2 = c1 * c1 - s1 * s1;
    const double s2 = 2.0 * s1 * c1;
    return 2.0 * (Scalar(-1.0) * q.C * c1 + q.D * s1)
        + 8.0 * (Scalar(-1.0) * q.A * c2 + q.B * s2);
}

// ---------------------------------------------------------------------------
// Analytic Jacobian driver.
//
// Consumes the frozen discrete plan of the primal
// (`experimental_algebraic_boundary_integral`) -- radial bands, per-radius
// boundary-angle classification, real-quartic-vs-Skowron decision -- and runs
// one forward-mode pass to produce d(mag)/d(source_x, source_y, rho, q_raw,
// separation_raw). Any measure-zero transition (F'(theta_k) ~ 0, a singular
// band-endpoint tangency system, or a two-form derivative disagreement over
// `grad_tol`) marks the whole-epoch row unreliable.
// ---------------------------------------------------------------------------

struct AlgebraicBoundaryJacobian {
    bool success = false;
    std::string unsafe_reason;
    double magnification = 0.0;
    // Absolute magnification error from the authoritative primal integral
    // consumed by this driver. It is already in magnification units.
    double value_error = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 5> jacobian{};  // wrt [source_x, source_y, rho, q, sep]
    // Maximum absolute K/G disagreement of the fully normalized
    // d(magnification)/d(parameter) row. This is directly comparable with the
    // caller's absolute grad_tol, including the rho normalization derivative.
    double grad_error = 0.0;
    bool grad_reliable = false;        // false => caller should NaN the row
    AlgebraicBoundaryDiagnostics diagnostics{};
};

// params: [source_x, source_y, rho, q_raw, separation_raw].
// grad_tol < 0 selects the automatic budget (10 * relative_tolerance, clamped
// to [1e-4, 1e-1]).
AlgebraicBoundaryJacobian experimental_algebraic_boundary_jacobian(
    const std::array<double, 5>& params,
    const FrozenLensBranch& branch,
    double limb_darkening_c,
    double limb_darkening_d,
    const std::vector<SourcePosition>& certified_image_seeds,
    const AlgebraicBoundarySettings& settings = {},
    double grad_tol = -1.0);

// Analytic Jacobian of the shared point-source / hexadecapole approximation.
// Used when binary_mag()'s shared routing resolves the epoch without a deep
// inverse-ray solve: the derivative is a ForwardJet<5> pass over the same
// 1- or 13-point stencil the router evaluated, so it costs ~1% of the deep
// driver.  `authoritative_magnification` is the value the router returned and
// is copied through verbatim; only the derivative row is computed here.
// `use_hexadecapole == false` selects the bare point-source stencil.
// params: [source_x, source_y, rho, q_raw, separation_raw].
AlgebraicBoundaryJacobian experimental_algebraic_boundary_multipole_jacobian(
    const std::array<double, 5>& params,
    const FrozenLensBranch& branch,
    double limb_darkening_c,
    double limb_darkening_d,
    bool use_hexadecapole,
    double authoritative_magnification);

}  // namespace lcbinint::magnification::autodiff

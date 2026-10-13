#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/cartesian_run_fill.hpp"
#include "lcbinint/magnification/component_certificate.hpp"
#include "lcbinint/magnification/caustic_probe_links.hpp"
#include "lcbinint/magnification/critical_support_certificate.hpp"
#include "lcbinint/magnification/polar_run_walker.hpp"

#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/probe_diagnostics.hpp"
#include "lcbinint/magnification/binary_image_discriminant.hpp"
#include "lcbinint/math/polynomial_roots.hpp"
#include "lcbinint/math/outward_interval.hpp"
#include "lcbinint/math/filtered_ball.hpp"
#include <type_traits>


#include <algorithm>
#include <array>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lcbinint::magnification {

namespace {

constexpr double kSqrtHalf = 0.70710678118654752440;
constexpr double kPi = 3.14159265358979323846;
constexpr int kHexadecapoleEvaluations = 13;
constexpr int kLimbDarkeningTableSize = 20000;
constexpr double kDefaultFiniteSourceAbsoluteTolerance = 1.0e-4;
constexpr double kDefaultFiniteSourceRelativeTolerance = 1.0e-3;
constexpr double kEmpiricalResolutionBaselineTolerance = 1.0e-3;
constexpr double kEmpiricalResolutionMinimumRelativeTolerance = 1.0e-4;
constexpr double kEmpiricalResolutionMinimumAbsoluteTolerance = 1.0e-4;
constexpr double kEmpiricalResolutionMaximumTolerance = 1.0e-2;
// Frozen production coefficients from
// tests/diagnostics/results/recal2026/REPORT_empirical_resolution_law.md and
// final_apoint_validation/final_apoint_validation.json.  The relative
// constants include the smallest
// discovery-side safety offset that gives >=99% exact-row coverage at every
// calibration level.  The absolute constants use the conservative
// discovery-side all-exact-row envelope; this is needed because the raw
// Apoint fit under-covered the independent holdout at several levels.
constexpr double kCartesianRelativeResolutionC = 49.5929807101336;
constexpr double kCartesianRelativeResolutionBeta = 0.47670215379590497;
constexpr double kPolarRelativeResolutionC = 105.29723705815378;
constexpr double kPolarRelativeResolutionBeta = 0.5952070961585817;
constexpr double kCartesianAbsoluteResolutionC = 138.06382198454384;
constexpr double kCartesianAbsoluteResolutionBeta = 0.4265493297299796;
constexpr double kCartesianAbsoluteResolutionGamma = 0.34119845152344075;
constexpr double kPolarAbsoluteResolutionC = 396.47500160748996;
constexpr double kPolarAbsoluteResolutionBeta = 0.5337641762207631;
constexpr double kPolarAbsoluteResolutionGamma = 0.2458039343900396;
constexpr std::array<int, 14> kFiniteSourceResolutionBuckets {{
    16, 24, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400,
}};

bool has_explicit_finite_source_tolerance(const FiniteSourceSettings& settings)
{
    return settings.finite_source_tol > 0.0 || settings.finite_source_reltol > 0.0;
}

double finite_source_error_budget(
    const FiniteSourceSettings& settings,
    double magnification)
{
    const double scale = std::max(std::abs(magnification), 1.0);
    if (has_explicit_finite_source_tolerance(settings)) {
        // Absolute and relative tolerances are alternative allowances.  The
        // common dimensional budget is the larger allowance, not their sum.
        return std::max(
            std::max(settings.finite_source_tol, 0.0),
            std::max(settings.finite_source_reltol, 0.0) * scale);
    }
    return std::max(
        kDefaultFiniteSourceAbsoluteTolerance,
        kDefaultFiniteSourceRelativeTolerance * scale);
}

bool finite_source_error_within_budget(
    const FiniteSourceSettings& settings,
    double magnification,
    double error_estimate)
{
    return std::isfinite(error_estimate) &&
        error_estimate <= finite_source_error_budget(settings, magnification);
}

struct FiniteSourceErrorAssessment {
    double budget = 0.0;
    double estimate = std::numeric_limits<double>::infinity();
    double shortfall = std::numeric_limits<double>::infinity();
    bool support_valid = false;
    bool converged = false;
};

FiniteSourceErrorAssessment assess_finite_source_error(
    const FiniteSourceSettings& settings,
    double magnification,
    double error_estimate,
    bool support_valid = true)
{
    const double budget = finite_source_error_budget(settings, magnification);
    const bool finite = std::isfinite(error_estimate);
    const double shortfall = budget > 0.0 && finite
        ? std::max(error_estimate / budget, 0.0)
        : std::numeric_limits<double>::infinity();
    return {
        budget,
        support_valid && finite
            ? error_estimate
            : std::numeric_limits<double>::infinity(),
        support_valid ? shortfall : std::numeric_limits<double>::infinity(),
        support_valid,
        support_valid && finite && error_estimate <= budget,
    };
}

int next_finite_source_resolution(
    int current_bins,
    int maximum_bins,
    double shortfall,
    int convergence_order)
{
    const int cap = std::max(maximum_bins, 1);
    if (current_bins >= cap) {
        return current_bins;
    }
    const int order = std::max(convergence_order, 1);
    const double growth = std::isfinite(shortfall)
        ? std::pow(std::max(shortfall, 1.0), 1.0 / static_cast<double>(order))
        : 2.0;
    const int requested = std::max(
        current_bins + 1,
        static_cast<int>(std::ceil(static_cast<double>(current_bins) * growth)));
    for (const int bucket : kFiniteSourceResolutionBuckets) {
        if (bucket >= requested) {
            return std::min(bucket, cap);
        }
    }
    return cap;
}

// For A(h) = A(0) + C h^p + o(h^p), two nested grids give the Richardson
// estimate |A_fine - A_coarse| / (r^p - 1) at the returned (fine) resolution,
// where r is the fine/coarse resolution ratio.  The caller supplies the
// convergence order p appropriate to its integration rule.
double grid_pair_error_estimate(
    double fine_magnification,
    double coarse_magnification,
    int fine_bins,
    int coarse_bins,
    int convergence_order = 1)
{
    const double difference = std::abs(fine_magnification - coarse_magnification);
    const double ratio = coarse_bins > 0
        ? static_cast<double>(fine_bins) / static_cast<double>(coarse_bins)
        : 2.0;
    const double denominator =
        std::pow(std::max(ratio, 1.0), std::max(convergence_order, 1)) - 1.0;
    return denominator > std::numeric_limits<double>::epsilon()
        ? difference / denominator
        : std::numeric_limits<double>::infinity();
}

// The boundary-area indicator counts the cells the source limb and the caustic
// cut, so it decays like 1/source_bins however fast the area itself converges.
// On the binary tangency it is 6x pessimistic by 4096 bins and on the triple
// cusp 1000x, which is enough to report a value that is right to 1e-14 as
// unconverged and hand back a NaN. Adaptive non-binary routes may use this
// independent half-resolution cross-check after their ladder, and a fixed
// nbin with an explicit tolerance may request it as a compatibility path.
// The calibrated binary automatic path does neither.
template <typename EvaluateAt>
void reconcile_with_half_resolution(
    const FiniteSourceSettings& settings,
    const FiniteSourceSettings& active_settings,
    double magnification,
    EvaluateAt&& evaluate_at,
    double& error_estimate,
    bool& converged,
    int convergence_order = 1)
{
    if (active_settings.source_bins <= 1) {
        return;
    }
    FiniteSourceSettings coarse_settings = active_settings;
    coarse_settings.source_bins = std::max(1, active_settings.source_bins / 2);
    if (coarse_settings.polar_source_bins > 0) {
        coarse_settings.polar_source_bins = coarse_settings.source_bins;
    }
    const double coarse_magnification = evaluate_at(coarse_settings);
    if (!std::isfinite(coarse_magnification)) {
        if (converged) {
            error_estimate = std::numeric_limits<double>::infinity();
            converged = false;
        }
        return;
    }
    const double measured = grid_pair_error_estimate(
        magnification, coarse_magnification,
        active_settings.source_bins, coarse_settings.source_bins,
        convergence_order);
    if (converged) {
        error_estimate = std::max(error_estimate, measured);
        converged = finite_source_error_within_budget(
            settings, magnification, error_estimate);
    } else if (finite_source_error_within_budget(settings, magnification, measured)) {
        error_estimate = measured;
        converged = true;
    }
}

double explicit_finite_source_relative_budget(
    const FiniteSourceSettings& settings,
    double magnification)
{
    if (!has_explicit_finite_source_tolerance(settings)) {
        return 0.0;
    }
    return finite_source_error_budget(settings, magnification) /
        std::max(std::abs(magnification), 1.0);
}

struct BinaryLensMapper {
    Complex separation;
    double m1 = 0.0;
    double m2 = 0.0;
    double source_origin_shift = 0.0;
    double input_mass_ratio=0.0;
};

struct TripleLensMapper {
    std::array<double, 3> lens_x {};
    std::array<double, 3> lens_y {};
    std::array<double, 3> mass {};
};

BinaryLensMapper make_binary_lens_mapper(double separation, double mass_ratio)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const Complex lens_separation = q_input < 1.0 ? Complex(-s, 0.0) : Complex(s, 0.0);
    const double m1 = 1.0 / (1.0 + q);
    const double m2 = q * m1;
    return {lens_separation, m1, m2, lens_separation.real() * m1,q_input};
}

struct EnclosedBinaryGeometry {
    math::OutwardInterval m1,m2,shift;
};
EnclosedBinaryGeometry enclosed_binary_geometry(const BinaryLensMapper& mapper) {
    using I=math::OutwardInterval;
    const I q=mapper.input_mass_ratio<1 ? I(mapper.input_mass_ratio) : I(1)/I(mapper.input_mass_ratio);
    const I first=I(1)/(I(1)+q),second=q*first,shift=I(mapper.separation.real())*first;
    const auto hull=[](I exact,double rounded) {return I(std::min(exact.lo,rounded),std::max(exact.hi,rounded));};
    // Cover both the exact-input mathematical model and the existing rounded
    // mapper; neither model may hide an unvisited interior lattice point.
    return {hull(first,mapper.m1),hull(second,mapper.m2),hull(shift,mapper.source_origin_shift)};
}

struct CertifiedSeedSupportIdentity {
    double s=0,q=0,rho=0;
    SourcePosition source{};
    bool proven=false;
    std::vector<SourcePosition> seeds;
};
thread_local CertifiedSeedSupportIdentity certified_seed_support;
bool matching_seed_support(const BinaryLensMapper& mapper,SourcePosition source,double rho) {
    const auto& c=certified_seed_support;
    return critical_enclosure_detail::supported_arithmetic() && c.proven && c.s==std::abs(mapper.separation.real()) && c.q==mapper.input_mass_ratio &&
           c.source.x==source.x && c.source.y==source.y && c.rho==rho;
}
using ImageFenceBox=std::array<double,4>;
// Every physical component has a certified seed. A closed rectangle whose
// entire boundary maps outside the open source disk traps any component
// containing one of those seeds. These fences restrict coverage work; they
// do not estimate an integral error or infer topology from sampled edges.
std::vector<ImageFenceBox> certified_image_fences(const BinaryLensMapper& mapper,
    SourcePosition source,double rho,double spacing,std::vector<ImageFenceBox> boxes,
    const std::vector<SourcePosition>& seeds)
{
    using I=math::OutwardInterval;
    if (!matching_seed_support(mapper,source,rho)) return {};
    for (const auto& z:seeds) boxes.push_back({z.x,z.x,z.y,z.y});
    for (const auto& z:certified_seed_support.seeds) boxes.push_back({z.x,z.x,z.y,z.y});
    if (boxes.empty() || !(spacing>0)) return {};
    for (auto& b:boxes) {
        const double padding=std::max(4*spacing,std::max(b[1]-b[0],b[3]-b[2])/8);
        b={math::down(b[0]-padding),math::up(b[1]+padding),math::down(b[2]-padding),math::up(b[3]+padding)};
    }
    const auto geometry=enclosed_binary_geometry(mapper);
    const auto outside=[&](const ImageFenceBox& b) {
        const I x(b[0],b[1]),y(b[2],b[3]),xa=x-I(mapper.separation.real());
        const I inv1=I(1)/(math::square(xa)+math::square(y)),inv2=I(1)/(math::square(x)+math::square(y));
        const I fx=x-geometry.m1*xa*inv1-geometry.m2*x*inv2-geometry.shift-I(source.x);
        const I fy=y-geometry.m1*y*inv1-geometry.m2*y*inv2-I(source.y);
        const I distance=math::square(fx)+math::square(fy);
        return distance.finite() && distance.lo>math::square(I(rho)).hi;
    };
    for (int attempt=0;attempt<4;++attempt) {
        bool changed=true;
        while (changed) {
            changed=false;
            for (std::size_t i=0;i<boxes.size() && !changed;++i)
                for (std::size_t j=0;j<i;++j) {
                    auto& a=boxes[j];const auto b=boxes[i];
                    if (a[0]>b[1] || b[0]>a[1] || a[2]>b[3] || b[2]>a[3]) continue;
                    a={std::min(a[0],b[0]),std::max(a[1],b[1]),std::min(a[2],b[2]),std::max(a[3],b[3])};
                    boxes.erase(boxes.begin()+i);changed=true;break;
                }
        }
        bool all_proven=true;
        for (auto& b:boxes) {
            std::vector<ImageFenceBox> pending{{b[0],b[0],b[2],b[3]},{b[1],b[1],b[2],b[3]},
                                               {b[0],b[1],b[2],b[2]},{b[0],b[1],b[3],b[3]}};
            int queries=0;
            while (!pending.empty() && queries++<512) {
                const auto edge=pending.back();pending.pop_back();
                if (outside(edge)) continue;
                const int k=edge[0]==edge[1] ? 2 : 0;
                const double midpoint=edge[k]+(edge[k+1]-edge[k])/2;
                if (!(midpoint>edge[k] && midpoint<edge[k+1])) {pending.push_back(edge);break;}
                auto a=edge,c=edge;a[k+1]=midpoint;c[k]=midpoint;
                pending.push_back(a);pending.push_back(c);
            }
            if (!pending.empty()) {
                all_proven=false;
                const double padding=std::max(4*spacing,std::max(b[1]-b[0],b[3]-b[2])/4);
                b={math::down(b[0]-padding),math::up(b[1]+padding),math::down(b[2]-padding),math::up(b[3]+padding)};
            }
        }
        if (all_proven) return boxes;
    }
    return {};
}

// Seed-independent angular scale for an experimental binary polar grid.
// If R exceeds every lens radius L, the lens equation gives
// R <= B + M/(R-L), B=max source radius about the lens-coordinate origin.
// Thus R <= (B+L+sqrt((B-L)^2+4M))/2. This is an image-radius bound,
// not a flux-error bound or a source-component completeness certificate.
double binary_polar_geometry_radius(
    const BinaryLensMapper& mapper, SourcePosition source, double source_radius)
{
    using I=math::OutwardInterval;
    const auto sqrt_interval=[](I value) {
        return I(std::max(0.0,math::down(std::sqrt(std::max(0.0,value.lo)))),
                 math::up(std::sqrt(value.hi)));
    };
    const I sx=I(source.x)+I(mapper.source_origin_shift), sy(source.y);
    const I b=sqrt_interval(math::square(sx)+math::square(sy))+I(source_radius);
    const I l(std::abs(mapper.separation.real())), mass=I(mapper.m1)+I(mapper.m2);
    const I bound=(b+l+sqrt_interval(math::square(b-l)+I(4)*mass))/I(2);
    // Keep the existing four-source-radius angular padding. Angular columns
    // use this scale only; the radial walk is not truncated at this radius.
    const I padded=bound+I(4)*I(source_radius);
    return padded.finite() ? std::max(1.0,padded.hi) : std::nan("");
}

TripleLensMapper make_triple_lens_mapper(const model::TripleLensGeometry& geometry)
{
    TripleLensMapper mapper;
    for (std::size_t i = 0; i < geometry.lens_positions.size(); ++i) {
        mapper.lens_x[i] = geometry.lens_positions[i].x;
        mapper.lens_y[i] = geometry.lens_positions[i].y;
        mapper.mass[i] = geometry.masses[i];
    }
    return mapper;
}

double mapped_binary_lens_distance2(
    const BinaryLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double y2 = y * y;
    const double den1 = xa * xa + y2;
    const double den2 = x * x + y2;
    // Both mapped coordinates use the same two lens denominators.  Spell out
    // their reciprocals so the hot inverse-ray loop performs two divisions per
    // cell instead of relying on the compiler to discover common divisions
    // across the x/y expressions (GCC does not do so under strict FP rules).
    const double inv_den1 = 1.0 / den1;
    const double inv_den2 = 1.0 / den2;
    const double mapped_x =
        x - mapper.m1 * xa * inv_den1 - mapper.m2 * x * inv_den2 -
        mapper.source_origin_shift;
    const double mapped_y =
        y - mapper.m1 * y * inv_den1 - mapper.m2 * y * inv_den2;
    const double dx = mapped_x - source.x;
    const double dy = mapped_y - source.y;
    return dx * dx + dy * dy;
}

SourcePosition map_triple_lens_real(
    const TripleLensMapper& mapper,
    double x,
    double y)
{
    double mapped_x = x;
    double mapped_y = y;
    for (std::size_t i = 0; i < mapper.mass.size(); ++i) {
        const double dx = x - mapper.lens_x[i];
        const double dy = y - mapper.lens_y[i];
        const double den = dx * dx + dy * dy;
        const double mass_over_den = mapper.mass[i] / den;
        mapped_x -= mass_over_den * dx;
        mapped_y -= mass_over_den * dy;
    }
    return {mapped_x, mapped_y};
}

double mapped_triple_lens_distance2(
    const TripleLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    const auto mapped = map_triple_lens_real(mapper, x, y);
    const double dx = mapped.x - source.x;
    const double dy = mapped.y - source.y;
    return dx * dx + dy * dy;
}

// Optional phase tracing: no clock reads in ordinary evaluations.
struct AdaptiveStageTimer {
    double* elapsed;
    std::chrono::steady_clock::time_point start;
    explicit AdaptiveStageTimer(double* field):elapsed(field) {
        if(elapsed)start=std::chrono::steady_clock::now();
    }
    ~AdaptiveStageTimer() {
        if(elapsed)*elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
};

struct BoundaryRootEvaluation {
    double residual = std::numeric_limits<double>::infinity();
    double derivative_x = 0.0;
};

BoundaryRootEvaluation boundary_root_evaluation(
    const BinaryLensMapper& mapper,
    double x,
    double y,
    SourcePosition source,
    double source_radius2)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double y2 = y * y;
    const double den1 = xa * xa + y2;
    const double den2 = x * x + y2;
    if (den1 < 1.0e-20 || den2 < 1.0e-20) {
        return {};
    }
    const double inv_den1 = 1.0 / den1;
    const double inv_den2 = 1.0 / den2;
    const double mapped_x =
        x - mapper.m1 * xa * inv_den1 - mapper.m2 * x * inv_den2 -
        mapper.source_origin_shift;
    const double mapped_y =
        y - mapper.m1 * y * inv_den1 - mapper.m2 * y * inv_den2;
    const double dx = mapped_x - source.x;
    const double dy = mapped_y - source.y;
    const double inv_den1sq = inv_den1 * inv_den1;
    const double inv_den2sq = inv_den2 * inv_den2;
    const double mapped_dx =
        1.0 + mapper.m1 * (xa * xa - y2) * inv_den1sq +
        mapper.m2 * (x * x - y2) * inv_den2sq;
    const double mapped_dy =
        2.0 * y * (mapper.m1 * xa * inv_den1sq +
                   mapper.m2 * x * inv_den2sq);
    return {
        dx * dx + dy * dy - source_radius2,
        2.0 * (dx * mapped_dx + dy * mapped_dy),
    };
}

BoundaryRootEvaluation boundary_root_evaluation(
    const TripleLensMapper& mapper,
    double x,
    double y,
    SourcePosition source,
    double source_radius2)
{
    double mapped_x = x;
    double mapped_y = y;
    double mapped_dx = 1.0;
    double mapped_dy = 0.0;
    for (std::size_t i = 0; i < mapper.mass.size(); ++i) {
        const double dx_lens = x - mapper.lens_x[i];
        const double dy_lens = y - mapper.lens_y[i];
        const double den = dx_lens * dx_lens + dy_lens * dy_lens;
        if (den < 1.0e-20) {
            return {};
        }
        const double inv_den = 1.0 / den;
        const double inv_densq = inv_den * inv_den;
        const double mass = mapper.mass[i];
        mapped_x -= mass * dx_lens * inv_den;
        mapped_y -= mass * dy_lens * inv_den;
        mapped_dx += mass *
            (dx_lens * dx_lens - dy_lens * dy_lens) * inv_densq;
        mapped_dy += 2.0 * mass * dx_lens * dy_lens * inv_densq;
    }
    const double dx = mapped_x - source.x;
    const double dy = mapped_y - source.y;
    return {
        dx * dx + dy * dy - source_radius2,
        2.0 * (dx * mapped_dx + dy * mapped_dy),
    };
}

double boundary_root_residual(
    const BinaryLensMapper& mapper,
    double x,
    double y,
    SourcePosition source,
    double source_radius2)
{
    return mapped_binary_lens_distance2(mapper, x, y, source) - source_radius2;
}

double boundary_root_residual(
    const TripleLensMapper& mapper,
    double x,
    double y,
    SourcePosition source,
    double source_radius2)
{
    return mapped_triple_lens_distance2(mapper, x, y, source) - source_radius2;
}

// Reuse the floating-point sign brackets already visited by the solver.
// These are numerical diagnostics, not interval-arithmetic certificates.
struct BoundaryBracketDiagnostic {
    double lower=0,upper=1;
    bool rounded_zero=false;
};

// Locate a source-limb crossing within a known inside/outside image-plane
// cell.  A converged Newton step is accepted only after two residual-only
// probes explicitly bracket it to the requested width; otherwise the bounded
// bisection tail supplies the same guarantee where the derivative is locally
// ill-conditioned.
template <typename ImageMap>
double refine_boundary_crossing_fraction(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius2,
    double inside_x,
    double outside_x,
    double y,
    double initial,
    double grid_spacing,
    BoundaryBracketDiagnostic* bracket_diagnostic=nullptr,
    double* limb_inverse_gradient=nullptr)
{
    const auto finish=[&](double value,double lo,double hi,bool rounded_zero=false,double derivative_hint=0.) {
        // A bracket midpoint has a systematic area bias when boundary errors
        // cancel the outer quadrature residual. Polish within the existing
        // bracket once; no tighter bisection tolerance or image solve is needed.
        if (!rounded_zero && std::isfinite(value)) {
            const auto evaluation=boundary_root_evaluation(mapper,
                inside_x+value*(outside_x-inside_x),y,source,source_radius2);
            derivative_hint=evaluation.derivative_x;
            const double derivative=evaluation.derivative_x*(outside_x-inside_x);
            if (std::isfinite(evaluation.residual)&&std::isfinite(derivative)&&std::abs(derivative)>1e-30) {
                const double candidate=value-evaluation.residual/derivative;
                if (candidate>=lo&&candidate<=hi&&std::isfinite(candidate))value=candidate;
            }
        }
        if(limb_inverse_gradient!=nullptr)*limb_inverse_gradient=std::isfinite(derivative_hint)&&derivative_hint!=0?1/std::abs(derivative_hint):0;
        if (bracket_diagnostic!=nullptr) *bracket_diagnostic={lo,hi,rounded_zero};
        return value;
    };
    double inside_fraction = 0.0;
    double outside_fraction = 1.0;
    double fraction = std::clamp(initial, 0.0, 1.0);
    // Bennett locates the boundary to 0.1 h^2 in physical coordinates.  The
    // fraction along a one-cell segment therefore needs a bracket of 0.1 h.
    // Keeping this tied to h is important: a fixed fraction leaves an O(h)
    // boundary-location error when the source grid is refined.
    const double maximum_boundary_bracket = std::clamp(
        0.1 * std::abs(grid_spacing),
        1.0e-14,
        0.25);
    constexpr int kBoundaryNewtonIterations = 3;
    bool convergence_probed = false;
    for (int iteration = 0; iteration < kBoundaryNewtonIterations; ++iteration) {
        const double x = inside_x + fraction * (outside_x - inside_x);
        const auto evaluation = boundary_root_evaluation(
            mapper, x, y, source, source_radius2);
        if (evaluation.residual == 0.0) {
            return finish(fraction,inside_fraction,outside_fraction,true,evaluation.derivative_x);
        }
        if (evaluation.residual <= 0.0) {
            inside_fraction = fraction;
        } else {
            outside_fraction = fraction;
        }
        if (outside_fraction - inside_fraction <= maximum_boundary_bracket) {
            return finish(0.5 * (inside_fraction + outside_fraction),inside_fraction,outside_fraction);
        }
        const double fraction_derivative =
            evaluation.derivative_x * (outside_x - inside_x);
        double next = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(fraction_derivative) &&
            std::abs(fraction_derivative) > 1.0e-30) {
            next = fraction - evaluation.residual / fraction_derivative;
        }
        if (!(next > inside_fraction && next < outside_fraction)) {
            next = 0.5 * (inside_fraction + outside_fraction);
        }
        if (!convergence_probed &&
            std::abs(next - fraction) <= maximum_boundary_bracket) {
            convergence_probed = true;
            // The current Newton evaluation already supplies one bracket sign.
            // Probe only the other side before paying for a two-sided check.
            if (std::getenv("LCBININT_DIAGNOSTIC_ONE_SIDED_BOUNDARY")) {
                const bool inside = evaluation.residual <= 0.0;
                const double other = inside
                    ? std::min(outside_fraction, fraction + maximum_boundary_bracket)
                    : std::max(inside_fraction, fraction - maximum_boundary_bracket);
                const double lo = inside ? fraction : other;
                const double hi = inside ? other : fraction;
                if (next >= lo && next <= hi && hi > lo &&
                    hi - lo <= maximum_boundary_bracket) {
                    const double residual = boundary_root_residual(mapper,
                        inside_x + other * (outside_x - inside_x), y, source, source_radius2);
                    if (std::isfinite(residual)) {
                        if ((inside && residual > 0.0) || (!inside && residual <= 0.0))
                            return finish(0.5 * (lo + hi), lo, hi);
                        if (inside) inside_fraction = other;
                        else outside_fraction = other;
                        if (!(next > inside_fraction && next < outside_fraction))
                            next = 0.5 * (inside_fraction + outside_fraction);
                    }
                }
            }
            const double half_width = 0.5 * maximum_boundary_bracket;
            const double local_inside = std::max(inside_fraction, next - half_width);
            const double local_outside = std::min(outside_fraction, next + half_width);
            if (local_outside - local_inside <= maximum_boundary_bracket) {
                const double inside_x_probe =
                    inside_x + local_inside * (outside_x - inside_x);
                const double outside_x_probe =
                    inside_x + local_outside * (outside_x - inside_x);
                const double inside_residual = boundary_root_residual(
                    mapper, inside_x_probe, y, source, source_radius2);
                const double outside_residual = boundary_root_residual(
                    mapper, outside_x_probe, y, source, source_radius2);
                if (inside_residual <= 0.0 && outside_residual > 0.0) {
                    return finish(0.5 * (local_inside + local_outside),local_inside,local_outside);
                }
                // Failed local certification still supplies evaluated signs.
                // Retain them for the safeguarded continuation rather than
                // bisecting the original, much wider interval again.
                for (const auto& probe : {std::pair<double,double>{local_inside,inside_residual},
                                          std::pair<double,double>{local_outside,outside_residual}}) {
                    if(probe.first>inside_fraction && probe.first<outside_fraction && std::isfinite(probe.second)) {
                        if(probe.second<=0)inside_fraction=probe.first;
                        else outside_fraction=probe.first;
                    }
                }
                if(!(next>inside_fraction && next<outside_fraction))
                    next=0.5*(inside_fraction+outside_fraction);
            }
        }
        fraction = next;
    }
    for (int iteration = 0;
         iteration < 64 &&
             outside_fraction - inside_fraction > maximum_boundary_bracket;
         ++iteration) {
        const double midpoint = 0.5 * (inside_fraction + outside_fraction);
        const double x = inside_x + midpoint * (outside_x - inside_x);
        const double residual = boundary_root_residual(
            mapper, x, y, source, source_radius2);
        if (residual <= 0.0) {
            inside_fraction = midpoint;
        } else {
            outside_fraction = midpoint;
        }
    }
    return finish(0.5 * (inside_fraction + outside_fraction),inside_fraction,outside_fraction);
}

// Correct a polar run edge along its known radial inside/outside segment.
// Endpoint interpolation supplies the first secant estimate; one extra
// lens-map probe updates the bracket and removes its leading curvature error
// without paying for a full boundary root solve in every angular column.
template <typename ImageMap>
double refine_polar_boundary_crossing_fraction(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius2,
    SourcePosition inside,
    SourcePosition outside,
    double inside_residual,
    double outside_residual,
    double initial,
    double grid_spacing)
{
    // Use the same 0.1 h^2 physical boundary-location target as the Cartesian
    // implementation.  The old single-secant probe was an improvement over
    // linear interpolation but did not provide Bennett's resolution target.
    const double maximum_boundary_bracket = std::clamp(
        0.1 * std::abs(grid_spacing),
        1.0e-14,
        0.25);
    double inside_fraction = 0.0;
    double outside_fraction = 1.0;
    double inside_value = inside_residual;
    double outside_value = outside_residual;
    double fraction = std::clamp(initial, 0.0, 1.0);
    int previous_side=0;
    for (int iteration = 0; iteration < 64; ++iteration) {
        if (outside_fraction - inside_fraction <= maximum_boundary_bracket) {
            return 0.5 * (inside_fraction + outside_fraction);
        }
        const double denominator = outside_value - inside_value;
        double candidate = denominator > 0.0
            ? inside_fraction - inside_value *
                (outside_fraction - inside_fraction) / denominator
            : fraction;
        if (!(candidate > inside_fraction && candidate < outside_fraction)) {
            candidate = 0.5 * (inside_fraction + outside_fraction);
        }
        const SourcePosition probe {
            inside.x + candidate * (outside.x - inside.x),
            inside.y + candidate * (outside.y - inside.y),
        };
        const double residual = boundary_root_residual(
            mapper, probe.x, probe.y, source, source_radius2);
        if (residual == 0.0) {
            return candidate;
        }
        if (residual <= 0.0) {
            inside_fraction = candidate;
            inside_value = residual;
            if(previous_side==-1)outside_value*=0.5;
            previous_side=-1;
        } else {
            outside_fraction = candidate;
            outside_value = residual;
            if(previous_side==1)inside_value*=0.5;
            previous_side=1;
        }
        fraction = candidate;
    }
    return 0.5 * (inside_fraction + outside_fraction);
}

// Locate a source-limb crossing between two neighbouring angular rays.  This
// is the polar analogue of the Cartesian boundary solve and is used only for
// the outer-coordinate rule (Bennett 2010, equation 15).
template <typename ImageMap>
double refine_angular_boundary_crossing_fraction(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius2,
    SourcePosition inside,
    SourcePosition outside,
    double inside_residual,
    double outside_residual,
    double initial,
    double grid_spacing)
{
    const double maximum_boundary_bracket = std::clamp(
        0.1 * std::abs(grid_spacing),
        1.0e-14,
        0.25);
    double inside_fraction = 0.0;
    double outside_fraction = 1.0;
    double inside_value = inside_residual;
    double outside_value = outside_residual;
    double fraction = std::clamp(initial, 0.0, 1.0);
    int previous_side=0;
    for (int iteration = 0; iteration < 64; ++iteration) {
        if (outside_fraction - inside_fraction <= maximum_boundary_bracket) {
            return 0.5 * (inside_fraction + outside_fraction);
        }
        const double denominator = outside_value - inside_value;
        double candidate = denominator > 0.0
            ? inside_fraction - inside_value *
                (outside_fraction - inside_fraction) / denominator
            : fraction;
        if (!(candidate > inside_fraction && candidate < outside_fraction)) {
            candidate = 0.5 * (inside_fraction + outside_fraction);
        }
        const SourcePosition probe {
            inside.x + candidate * (outside.x - inside.x),
            inside.y + candidate * (outside.y - inside.y),
        };
        const double residual = boundary_root_residual(
            mapper, probe.x, probe.y, source, source_radius2);
        if (residual == 0.0) {
            return candidate;
        }
        if (residual <= 0.0) {
            inside_fraction = candidate;
            inside_value = residual;
            if(previous_side==-1)outside_value*=0.5;
            previous_side=-1;
        } else {
            outside_fraction = candidate;
            outside_value = residual;
            if(previous_side==1)inside_value*=0.5;
            previous_side=1;
        }
        fraction = candidate;
    }
    return 0.5 * (inside_fraction + outside_fraction);
}

template <typename ImageMap>
double refine_vertical_boundary_crossing_fraction(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius2,
    double x,
    double inside_y,
    double outside_y,
    double inside_residual,
    double outside_residual,
    double initial,
    double grid_spacing)
{
    const double maximum_boundary_bracket = std::clamp(
        0.1 * std::abs(grid_spacing),
        1.0e-14,
        0.25);
    double inside_fraction = 0.0;
    double outside_fraction = 1.0;
    double inside_value = inside_residual;
    double outside_value = outside_residual;
    double fraction = std::clamp(initial, 0.0, 1.0);
    for (int iteration = 0; iteration < 64; ++iteration) {
        if (outside_fraction - inside_fraction <= maximum_boundary_bracket) {
            return 0.5 * (inside_fraction + outside_fraction);
        }
        const double denominator = outside_value - inside_value;
        double candidate = denominator > 0.0
            ? inside_fraction - inside_value *
                (outside_fraction - inside_fraction) / denominator
            : fraction;
        if (!(candidate > inside_fraction && candidate < outside_fraction)) {
            candidate = 0.5 * (inside_fraction + outside_fraction);
        }
        const double y = inside_y + candidate * (outside_y - inside_y);
        const double residual = boundary_root_residual(
            mapper, x, y, source, source_radius2);
        if (residual == 0.0) {
            return candidate;
        }
        if (residual <= 0.0) {
            inside_fraction = candidate;
            inside_value = residual;
        } else {
            outside_fraction = candidate;
            outside_value = residual;
        }
        fraction = candidate;
    }
    return 0.5 * (inside_fraction + outside_fraction);
}

struct BinaryLensEvaluation {
    double mapped_distance2 = 0.0;
    double jacobian = 0.0;
};

struct TripleLensEvaluation {
    double mapped_distance2 = 0.0;
    double jacobian = 0.0;
};

BinaryLensEvaluation evaluate_binary_lens_cell(
    const BinaryLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double y2 = y * y;
    const double den1 = xa * xa + y2;
    const double den2 = x * x + y2;
    const double inv_den1 = 1.0 / den1;
    const double inv_den2 = 1.0 / den2;

    const double mapped_x =
        x - mapper.m1 * xa * inv_den1 - mapper.m2 * x * inv_den2 -
        mapper.source_origin_shift;
    const double mapped_y =
        y - mapper.m1 * y * inv_den1 - mapper.m2 * y * inv_den2;
    const double dx = mapped_x - source.x;
    const double dy = mapped_y - source.y;

    double jacobian = 0.0;
    if (den1 >= 1.0e-20 && den2 >= 1.0e-20) {
        const double inv_den1sq = inv_den1 * inv_den1;
        const double inv_den2sq = inv_den2 * inv_den2;
        const double re_f = mapper.m1 * (xa * xa - y2) * inv_den1sq
                          + mapper.m2 * (x * x - y2) * inv_den2sq;
        const double im_f =
            -2.0 * y * (mapper.m1 * xa * inv_den1sq + mapper.m2 * x * inv_den2sq);
        jacobian = 1.0 - re_f * re_f - im_f * im_f;
    }
    return {dx * dx + dy * dy, jacobian};
}

TripleLensEvaluation evaluate_triple_lens_cell(
    const TripleLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    double mapped_x = x;
    double mapped_y = y;
    double re_f = 0.0;
    double im_f = 0.0;
    bool valid = true;
    for (std::size_t i = 0; i < mapper.mass.size(); ++i) {
        const double dx_lens = x - mapper.lens_x[i];
        const double dy_lens = y - mapper.lens_y[i];
        const double den = dx_lens * dx_lens + dy_lens * dy_lens;
        if (den < 1.0e-20) {
            valid = false;
            break;
        }
        const double inv_den = 1.0 / den;
        mapped_x -= mapper.mass[i] * dx_lens * inv_den;
        mapped_y -= mapper.mass[i] * dy_lens * inv_den;

        const double inv_densq = inv_den * inv_den;
        re_f += mapper.mass[i] * (dx_lens * dx_lens - dy_lens * dy_lens) * inv_densq;
        im_f += -2.0 * mapper.mass[i] * dx_lens * dy_lens * inv_densq;
    }
    if (!valid) {
        return {std::numeric_limits<double>::infinity(), 0.0};
    }
    const SourcePosition mapped {mapped_x, mapped_y};
    const double dx = mapped.x - source.x;
    const double dy = mapped.y - source.y;

    const double jacobian = 1.0 - re_f * re_f - im_f * im_f;
    return {dx * dx + dy * dy, jacobian};
}

// Boundary-seed deduplication used to call map_triple_lens_real and
// triple_jacobian separately.  Fuse those traversals while retaining their
// exact arithmetic forms: mass / den for the map and 1 / (den * den) for the
// Jacobian.  The distinction matters only at roundoff-sized dedup thresholds,
// but keeping it makes this optimization seed/order preserving.
TripleLensEvaluation evaluate_triple_lens_cell_strict(
    const TripleLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    double mapped_x = x;
    double mapped_y = y;
    double re_f = 0.0;
    double im_f = 0.0;
    bool jacobian_valid = true;
    for (std::size_t i = 0; i < mapper.mass.size(); ++i) {
        const double dx_lens = x - mapper.lens_x[i];
        const double dy_lens = y - mapper.lens_y[i];
        const double den = dx_lens * dx_lens + dy_lens * dy_lens;
        const double mass_over_den = mapper.mass[i] / den;
        mapped_x -= mass_over_den * dx_lens;
        mapped_y -= mass_over_den * dy_lens;
        if (den < 1.0e-20) {
            jacobian_valid = false;
            continue;
        }
        const double inv_densq = 1.0 / (den * den);
        re_f += mapper.mass[i] *
            (dx_lens * dx_lens - dy_lens * dy_lens) * inv_densq;
        im_f += -2.0 * mapper.mass[i] * dx_lens * dy_lens * inv_densq;
    }
    const double dx = mapped_x - source.x;
    const double dy = mapped_y - source.y;
    const double jacobian = jacobian_valid
        ? 1.0 - re_f * re_f - im_f * im_f
        : 0.0;
    return {dx * dx + dy * dy, jacobian};
}

// Jacobian determinant J = 1 - |m1/(z-a)^2 + m2/z^2|^2 at image position (x,y).
// J > 0: standard-parity image; J < 0: flipped-parity image.
// Returns 0.0 when image is too close to a lens (degenerate).
double binary_jacobian(const BinaryLensMapper& mapper, double x, double y)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double den1 = xa * xa + y * y;
    const double den2 = x * x + y * y;
    if (den1 < 1.0e-20 || den2 < 1.0e-20) return 0.0;
    const double inv_den1sq = 1.0 / (den1 * den1);
    const double inv_den2sq = 1.0 / (den2 * den2);
    const double re_f = mapper.m1 * (xa * xa - y * y) * inv_den1sq
                      + mapper.m2 * (x * x - y * y) * inv_den2sq;
    const double im_f = -2.0 * y *
        (mapper.m1 * xa * inv_den1sq + mapper.m2 * x * inv_den2sq);
    return 1.0 - re_f * re_f - im_f * im_f;
}

// Returns the sign of the binary lens Jacobian: +1, -1, or 0 (degenerate).
int binary_jacobian_sign(const BinaryLensMapper& mapper, double x, double y)
{
    const double J = binary_jacobian(mapper, x, y);
    return J > 0.0 ? 1 : J < 0.0 ? -1 : 0;
}

double triple_jacobian(const TripleLensMapper& mapper, double x, double y)
{
    double re_f = 0.0;
    double im_f = 0.0;
    for (std::size_t i = 0; i < mapper.mass.size(); ++i) {
        const double dx_lens = x - mapper.lens_x[i];
        const double dy_lens = y - mapper.lens_y[i];
        const double den = dx_lens * dx_lens + dy_lens * dy_lens;
        if (den < 1.0e-20) {
            return 0.0;
        }
        const double inv_densq = 1.0 / (den * den);
        re_f += mapper.mass[i] * (dx_lens * dx_lens - dy_lens * dy_lens) * inv_densq;
        im_f += -2.0 * mapper.mass[i] * dx_lens * dy_lens * inv_densq;
    }
    return 1.0 - re_f * re_f - im_f * im_f;
}

int triple_jacobian_sign(const TripleLensMapper& mapper, double x, double y)
{
    const double J = triple_jacobian(mapper, x, y);
    return J > 0.0 ? 1 : J < 0.0 ? -1 : 0;
}

double mapped_lens_distance2(
    const BinaryLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    return mapped_binary_lens_distance2(mapper, x, y, source);
}

double mapped_lens_distance2(
    const TripleLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    return mapped_triple_lens_distance2(mapper, x, y, source);
}

BinaryLensEvaluation evaluate_lens_cell(
    const BinaryLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    return evaluate_binary_lens_cell(mapper, x, y, source);
}

TripleLensEvaluation evaluate_lens_cell(
    const TripleLensMapper& mapper,
    double x,
    double y,
    SourcePosition source)
{
    return evaluate_triple_lens_cell(mapper, x, y, source);
}

double lens_jacobian(const BinaryLensMapper& mapper, double x, double y)
{
    return binary_jacobian(mapper, x, y);
}

double lens_jacobian(const TripleLensMapper& mapper, double x, double y)
{
    return triple_jacobian(mapper, x, y);
}

SourcePosition map_binary_lens_real(
    const BinaryLensMapper& mapper,
    double x,
    double y)
{
    const double a = mapper.separation.real();
    const double xa = x - a;
    const double y2 = y * y;
    const double inv_den1 = 1.0 / (xa * xa + y2);
    const double inv_den2 = 1.0 / (x * x + y2);
    return {
        x - mapper.m1 * xa * inv_den1 - mapper.m2 * x * inv_den2 -
            mapper.source_origin_shift,
        y - mapper.m1 * y * inv_den1 - mapper.m2 * y * inv_den2,
    };
}

SourcePosition map_lens_real(const BinaryLensMapper& mapper, double x, double y)
{
    return map_binary_lens_real(mapper, x, y);
}

SourcePosition map_lens_real(const TripleLensMapper& mapper, double x, double y)
{
    return map_triple_lens_real(mapper, x, y);
}

double source_distance(SourcePosition source)
{
    return std::hypot(source.x, source.y);
}

double binary_topology_boundary_margin(double separation, double mass_ratio)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input <= 1.0 ? q_input : 1.0 / q_input;
    if (!(s > 0.0) || !(q > 0.0) || !std::isfinite(s) || !std::isfinite(q)) {
        return 0.0;
    }
    const double q13 = std::cbrt(q);
    const double normalization = std::sqrt(1.0 + q);
    const double close_boundary =
        std::pow(1.0 - q13 + q13 * q13, 0.75) / normalization;
    const double wide_boundary =
        std::pow(1.0 + q13, 1.5) / normalization;
    return std::min(std::abs(s - close_boundary), std::abs(s - wide_boundary));
}

struct PointSourceSafetyEvaluation {
    PointSourceSafetyDiagnostic diagnostic;
    double absolute_tolerance = 0.0;
    double planetary_distance2 = std::numeric_limits<double>::infinity();
    bool quadrupole_cusp_safe = false;
    bool ghost_safe = false;
    bool planetary_safe = false;

    bool point_source_safe() const
    {
        return quadrupole_cusp_safe && ghost_safe && planetary_safe;
    }

    bool topology_safe() const
    {
        return ghost_safe && planetary_safe;
    }
};

PointSourceSafetyEvaluation evaluate_point_source_safety(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double point_source_magnification,
    const FiniteSourceSettings& settings)
{
    PointSourceSafetyEvaluation evaluation;
    evaluation.diagnostic = point_magnifier.binary_safety_diagnostic_cached(
        separation, mass_ratio, source);

    evaluation.absolute_tolerance = has_explicit_finite_source_tolerance(settings)
        ? finite_source_error_budget(settings, point_source_magnification)
        : std::max(
            kDefaultFiniteSourceAbsoluteTolerance,
            (settings.adaptive_hex_threshold > 0.0
                ? settings.adaptive_hex_threshold
                : kDefaultFiniteSourceRelativeTolerance) *
                std::max(std::abs(point_source_magnification), 1.0));

    constexpr double kQuadrupoleCuspSafety = 6.0;
    constexpr double kGhostSafety = 3.0;
    constexpr double kPlanetarySafety = 2.0;
    constexpr double kMinimumSafetyRadius = 1.0e-3;
    const double safety_radius = source_radius + kMinimumSafetyRadius;
    const double local_indicator =
        evaluation.diagnostic.quadrupole_indicator + evaluation.diagnostic.cusp_indicator;
    evaluation.quadrupole_cusp_safe =
        std::isfinite(local_indicator) &&
        kQuadrupoleCuspSafety * local_indicator * safety_radius * safety_radius <
            evaluation.absolute_tolerance;
    evaluation.ghost_safe =
        evaluation.diagnostic.ghost_count == 0 ||
        (std::isfinite(evaluation.diagnostic.ghost_indicator) &&
            kGhostSafety * safety_radius * evaluation.diagnostic.ghost_indicator < 1.0);

    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : (q_input > 0.0 ? 1.0 / q_input : 0.0);
    evaluation.planetary_safe = q >= 0.01;
    if (!evaluation.planetary_safe && s > 0.0) {
        const double signed_separation = q_input < 1.0 ? -s : s;
        const double primary_mass = 1.0 / (1.0 + q);
        const double shifted_source_x = source.x + signed_separation * primary_mass;
        const double planetary_caustic_x = 1.0 / signed_separation;
        const double dx = shifted_source_x - planetary_caustic_x;
        evaluation.planetary_distance2 = dx * dx + source.y * source.y;
        const double caustic_half_extent = 3.0 * std::sqrt(q) / s;
        evaluation.planetary_safe =
            evaluation.planetary_distance2 >
            kPlanetarySafety *
                (source_radius * source_radius + caustic_half_extent * caustic_half_extent);
    }
    return evaluation;
}

int estimate_cartesian_cost(const FiniteSourceSettings& settings)
{
    const int bins = settings.source_bins > 0 ? settings.source_bins : 1;
    return bins * bins * 16;
}

int active_polar_source_bins(const FiniteSourceSettings& settings)
{
    return std::max(settings.polar_source_bins > 0 ? settings.polar_source_bins : settings.source_bins, 1);
}

double active_polar_grid_ratio(const FiniteSourceSettings& settings)
{
    const double ratio = settings.polar_grid_ratio > 0.0 ? settings.polar_grid_ratio : settings.grid_ratio;
    return std::max(ratio, 1.0e-12);
}

int estimate_polar_cost(const FiniteSourceSettings& settings)
{
    const int radial_bins = active_polar_source_bins(settings);
    const int angular_bins = static_cast<int>(
        std::ceil(2.0 * M_PI * radial_bins / active_polar_grid_ratio(settings)));
    return radial_bins * angular_bins * 8;
}

double distance_squared(SourcePosition lhs, SourcePosition rhs)
{
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
}

double point_segment_distance(SourcePosition point, SourcePosition start, SourcePosition end)
{
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double length2 = dx * dx + dy * dy;
    if (length2 == 0.0) {
        return std::sqrt(distance_squared(point, start));
    }

    const double t = std::clamp(
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / length2, 0.0, 1.0);
    return std::hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

double limb_darkening_flux_factor(const FiniteSourceSettings& settings)
{
    return 1.0 - settings.limb_darkening_c / 3.0 - settings.limb_darkening_d / 5.0;
}

double limb_darkening_gamma(const FiniteSourceSettings& settings)
{
    const double denominator =
        15.0 - 5.0 * settings.limb_darkening_c - 3.0 * settings.limb_darkening_d;
    if (denominator == 0.0) {
        return 0.0;
    }
    return 10.0 * settings.limb_darkening_c / denominator;
}

double limb_darkening_lambda(const FiniteSourceSettings& settings)
{
    const double denominator =
        15.0 - 5.0 * settings.limb_darkening_c - 3.0 * settings.limb_darkening_d;
    if (denominator == 0.0) {
        return 0.0;
    }
    return 12.0 * settings.limb_darkening_d / denominator;
}

double source_surface_brightness(double normalized_radius2, const FiniteSourceSettings& settings)
{
    const double bounded_radius2 = std::clamp(normalized_radius2, 0.0, 1.0);
    const double mu = std::sqrt(std::max(0.0, 1.0 - bounded_radius2));
    if (settings.limb_darkening_d == 0.0) {
        return 1.0 - settings.limb_darkening_c * (1.0 - mu);
    }
    return 1.0 - settings.limb_darkening_c * (1.0 - mu) -
           settings.limb_darkening_d * (1.0 - std::sqrt(mu));
}

double source_flux(double source_radius, const FiniteSourceSettings& settings)
{
    const double flux_factor = limb_darkening_flux_factor(settings);
    if (flux_factor <= 0.0 || !std::isfinite(flux_factor)) {
        return std::nan("");
    }
    return kPi * source_radius * source_radius * flux_factor;
}

using LocalPolynomial = std::vector<Complex>;

LocalPolynomial multiply_local_polynomial(const LocalPolynomial& lhs, const LocalPolynomial& rhs)
{
    LocalPolynomial out(lhs.size() + rhs.size() - 1, 0.0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        for (std::size_t j = 0; j < rhs.size(); ++j) {
            out[i + j] += lhs[i] * rhs[j];
        }
    }
    return out;
}

void add_scaled_polynomial(LocalPolynomial& lhs, const LocalPolynomial& rhs, Complex scale)
{
    if (lhs.size() < rhs.size()) {
        lhs.resize(rhs.size(), 0.0);
    }
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        lhs[i] += scale * rhs[i];
    }
}

LocalPolynomial triple_critical_curve_polynomial_coefficients(
    const model::TripleLensGeometry& geometry,
    Complex phase)
{
    std::array<Complex, 3> lens_positions;
    for (std::size_t i = 0; i < lens_positions.size(); ++i) {
        lens_positions[i] = {
            geometry.lens_positions[i].x,
            geometry.lens_positions[i].y};
    }

    LocalPolynomial all = {1.0};
    for (const auto& lens : lens_positions) {
        const LocalPolynomial factor = {-lens, 1.0};
        all = multiply_local_polynomial(all, multiply_local_polynomial(factor, factor));
    }

    LocalPolynomial coefficients(all.size(), 0.0);
    add_scaled_polynomial(coefficients, all, -phase);
    for (std::size_t excluded = 0; excluded < lens_positions.size(); ++excluded) {
        LocalPolynomial term = {1.0};
        for (std::size_t i = 0; i < lens_positions.size(); ++i) {
            if (i == excluded) {
                continue;
            }
            const LocalPolynomial factor = {-lens_positions[i], 1.0};
            term = multiply_local_polynomial(term, multiply_local_polynomial(factor, factor));
        }
        add_scaled_polynomial(coefficients, term, geometry.masses[excluded]);
    }
    return coefficients;
}

std::vector<SourcePosition> triple_caustic_points_at_phase(
    const model::TripleLensGeometry& geometry,
    double phase_angle)
{
    math::PolynomialRootSolver solver;
    const auto roots = solver.solve(triple_critical_curve_polynomial_coefficients(
        geometry,
        std::polar(1.0, phase_angle)));
    if (roots.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> points;
    points.reserve(roots.roots.size());
    for (const auto& root : roots.roots) {
        points.push_back(model::triple_lens_equation(
            geometry,
            {root.real(), root.imag()}));
    }
    return points;
}

std::vector<SourcePosition> triple_critical_curve_points_at_phase(
    const model::TripleLensGeometry& geometry,
    double phase_angle)
{
    math::PolynomialRootSolver solver;
    const auto roots = solver.solve(triple_critical_curve_polynomial_coefficients(
        geometry,
        std::polar(1.0, phase_angle)));
    if (roots.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> points;
    points.reserve(roots.roots.size());
    for (const auto& root : roots.roots) {
        points.push_back({root.real(), root.imag()});
    }
    return points;
}

struct SourcePlaneQuadratureResult {
    double magnification = std::numeric_limits<double>::quiet_NaN();
    int sample_count = 0;
    int image_count = 0;
};

SourcePlaneQuadratureResult triple_source_plane_quadrature(
    const PointSourceMagnifier& point_magnifier,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    int radial_bins)
{
    const int bins = std::max(radial_bins, 1);
    const double angular_scale = active_polar_grid_ratio(settings);
    double weighted_magnification = 0.0;
    double brightness_sum = 0.0;
    int sample_count = 0;
    int max_image_count = 0;

    for (int ir = 0; ir < bins; ++ir) {
        const double normalized_radius2 =
            (static_cast<double>(ir) + 0.5) / static_cast<double>(bins);
        const double normalized_radius = std::sqrt(normalized_radius2);
        const double brightness = source_surface_brightness(normalized_radius2, settings);
        if (!std::isfinite(brightness)) {
            return {};
        }
        const int angular_bins = std::max(
            8,
            static_cast<int>(
                std::ceil(2.0 * kPi * normalized_radius * bins / angular_scale)));
        double ring_magnification = 0.0;
        for (int ia = 0; ia < angular_bins; ++ia) {
            const double angle = 2.0 * kPi *
                (static_cast<double>(ia) + 0.5) / static_cast<double>(angular_bins);
            const SourcePosition sample {
                source.x + source_radius * normalized_radius * std::cos(angle),
                source.y + source_radius * normalized_radius * std::sin(angle),
            };
            const auto point = point_magnifier.triple_mag0(geometry, sample);
            if (!std::isfinite(point.magnification)) {
                return {};
            }
            ring_magnification += point.magnification;
            max_image_count = std::max(max_image_count, point.image_count);
            ++sample_count;
        }
        weighted_magnification +=
            brightness * ring_magnification / static_cast<double>(angular_bins);
        brightness_sum += brightness;
    }

    if (brightness_sum <= 0.0 || !std::isfinite(brightness_sum)) {
        return {};
    }
    return {
        weighted_magnification / brightness_sum,
        sample_count,
        max_image_count,
    };
}

// Gauss-Legendre nodes/weights on [-1, 1], computed once per order via
// Newton iteration on the Legendre recurrence and cached.
const std::pair<std::vector<double>, std::vector<double>>& gauss_legendre_rule(int n)
{
    thread_local std::unordered_map<int, std::pair<std::vector<double>, std::vector<double>>>
        cache;
    auto it = cache.find(n);
    if (it != cache.end()) {
        return it->second;
    }
    std::vector<double> nodes(static_cast<std::size_t>(n));
    std::vector<double> weights(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        double x = std::cos(kPi * (static_cast<double>(i) + 0.75) /
                            (static_cast<double>(n) + 0.5));
        double dp = 0.0;
        for (int iter = 0; iter < 100; ++iter) {
            double p0 = 1.0;
            double p1 = x;
            for (int k = 2; k <= n; ++k) {
                const double p2 =
                    ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / static_cast<double>(k);
                p0 = p1;
                p1 = p2;
            }
            dp = static_cast<double>(n) * (x * p1 - p0) / (x * x - 1.0);
            const double dx = p1 / dp;
            x -= dx;
            if (std::abs(dx) < 1.0e-15) {
                break;
            }
        }
        nodes[static_cast<std::size_t>(i)] = x;
        weights[static_cast<std::size_t>(i)] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    return cache.emplace(n, std::make_pair(std::move(nodes), std::move(weights)))
        .first->second;
}

// Composite tensor Gauss-Legendre chord quadrature of the point-source
// magnification over the disk.  A single high-order rule is not reliable here:
// its nodes do not adapt to an off-centre fold spike, and orders 48/96/160 can
// agree while remaining wrong by 1e-3 relative.  Fixed order-eight rules on a
// panel ladder localise that integrable singularity; this is the same
// discretisation as the independent tangency arbiter (the root solver is the
// only code it shares).
double binary_source_plane_chord_quadrature(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    int panels)
{
    constexpr int order = 8;
    const int panel_count = std::max(panels, 1);
    const auto& rule = gauss_legendre_rule(order);
    const auto& nodes = rule.first;
    const auto& weights = rule.second;
    const std::size_t row_size = static_cast<std::size_t>(panel_count * order);
    std::vector<SourcePosition> row_sources(row_size);
    std::vector<double> row_weights(row_size);
    std::vector<double> row_magnifications(row_size);
    double weighted = 0.0;
    double brightness_norm = 0.0;
    const double panel_width = 2.0 / static_cast<double>(panel_count);
    const double half_panel = 0.5 * panel_width;
    for (int jy = 0; jy < panel_count; ++jy) {
        const double eta_mid = -1.0 + (static_cast<double>(jy) + 0.5) * panel_width;
        for (int ny = 0; ny < order; ++ny) {
            const double eta = eta_mid + half_panel * nodes[static_cast<std::size_t>(ny)];
            const double eta_weight = half_panel * weights[static_cast<std::size_t>(ny)];
            const double half_chord = std::sqrt(std::max(0.0, 1.0 - eta * eta));
            if (half_chord <= 0.0) {
                continue;
            }
            const double y = source.y + source_radius * eta;
            std::size_t k = 0;
            for (int jx = 0; jx < panel_count; ++jx) {
                const double xi_mid = -1.0 +
                    (static_cast<double>(jx) + 0.5) * panel_width;
                for (int nx = 0; nx < order; ++nx, ++k) {
                    const double unit_xi = xi_mid +
                        half_panel * nodes[static_cast<std::size_t>(nx)];
                    const double xi = half_chord * unit_xi;
                    row_sources[k] = {
                        source.x + source_radius * xi,
                        y,
                    };
                    row_weights[k] =
                        half_panel * weights[static_cast<std::size_t>(nx)];
                }
            }
            point_magnifier.binary_mag0_batch(
                separation, mass_ratio, row_sources.data(), row_magnifications.data(),
                row_sources.size());
            double row_acc = 0.0;
            double row_norm = 0.0;
            k = 0;
            for (int jx = 0; jx < panel_count; ++jx) {
                for (int nx = 0; nx < order; ++nx, ++k) {
                    if (!std::isfinite(row_magnifications[k])) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    const double xi =
                        (row_sources[k].x - source.x) / source_radius;
                    const double normalized_radius2 = xi * xi + eta * eta;
                    const double brightness =
                        source_surface_brightness(normalized_radius2, settings);
                    row_acc += row_weights[k] * brightness * row_magnifications[k];
                    row_norm += row_weights[k] * brightness;
                }
            }
            weighted += eta_weight * half_chord * row_acc;
            brightness_norm += eta_weight * half_chord * row_norm;
        }
    }
    if (brightness_norm <= 0.0 || !std::isfinite(brightness_norm)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return weighted / brightness_norm;
}

// Triple-lens counterpart of the tangent-caustic chord rule.  This is kept
// separate from the binary batch implementation because triple point-source
// solves use a different polynomial/cache, while the quadrature and
// normalization are identical.
double triple_source_plane_chord_quadrature(
    const PointSourceMagnifier& point_magnifier,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    int order)
{
    const auto& rule = gauss_legendre_rule(order);
    const auto& nodes = rule.first;
    const auto& weights = rule.second;
    double weighted = 0.0;
    double brightness_norm = 0.0;
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        const double eta = nodes[j];
        const double half_chord = std::sqrt(std::max(0.0, 1.0 - eta * eta));
        if (half_chord <= 0.0) {
            continue;
        }
        double row_acc = 0.0;
        double row_norm = 0.0;
        for (std::size_t k = 0; k < nodes.size(); ++k) {
            const double xi = half_chord * nodes[k];
            const SourcePosition sample {
                source.x + source_radius * xi,
                source.y + source_radius * eta,
            };
            const double magnification = point_magnifier.triple_mag0(
                geometry, sample).magnification;
            if (!std::isfinite(magnification)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double normalized_radius2 = xi * xi + eta * eta;
            const double brightness = source_surface_brightness(normalized_radius2, settings);
            row_acc += weights[k] * brightness * magnification;
            row_norm += weights[k] * brightness;
        }
        weighted += weights[j] * half_chord * row_acc;
        brightness_norm += weights[j] * half_chord * row_norm;
    }
    if (brightness_norm <= 0.0 || !std::isfinite(brightness_norm)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return weighted / brightness_norm;
}

// Binary twin of triple_source_plane_quadrature: equal-area radial rings of
// point-source magnifications.  Used for the grazing-caustic regime where the
// caustic passes within a few source radii but never enters the disk: the
// magnification is smooth (if steep) over the disk, so ring quadrature
// converges, whereas image-plane inverse rays truncate the sub-cell-thick
// image fingers of the near-caustic limb.
SourcePlaneQuadratureResult binary_source_plane_quadrature(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    int radial_bins)
{
    const int bins = std::max(radial_bins, 1);
    const double angular_scale = active_polar_grid_ratio(settings);
    double weighted_magnification = 0.0;
    double brightness_sum = 0.0;
    int sample_count = 0;

    std::vector<SourcePosition> ring_sources;
    std::vector<double> ring_magnifications;
    for (int ir = 0; ir < bins; ++ir) {
        const double normalized_radius2 =
            (static_cast<double>(ir) + 0.5) / static_cast<double>(bins);
        const double normalized_radius = std::sqrt(normalized_radius2);
        const double brightness = source_surface_brightness(normalized_radius2, settings);
        if (!std::isfinite(brightness)) {
            return {};
        }
        const int angular_bins = std::max(
            8,
            static_cast<int>(
                std::ceil(2.0 * kPi * normalized_radius * bins / angular_scale)));
        ring_sources.resize(static_cast<std::size_t>(angular_bins));
        ring_magnifications.resize(static_cast<std::size_t>(angular_bins));
        for (int ia = 0; ia < angular_bins; ++ia) {
            const double angle = 2.0 * kPi *
                (static_cast<double>(ia) + 0.5) / static_cast<double>(angular_bins);
            ring_sources[static_cast<std::size_t>(ia)] = {
                source.x + source_radius * normalized_radius * std::cos(angle),
                source.y + source_radius * normalized_radius * std::sin(angle),
            };
        }
        point_magnifier.binary_mag0_batch(
            separation, mass_ratio, ring_sources.data(), ring_magnifications.data(),
            ring_sources.size());
        double ring_magnification = 0.0;
        for (int ia = 0; ia < angular_bins; ++ia) {
            const double magnification = ring_magnifications[static_cast<std::size_t>(ia)];
            if (!std::isfinite(magnification)) {
                return {};
            }
            ring_magnification += magnification;
            ++sample_count;
        }
        weighted_magnification +=
            brightness * ring_magnification / static_cast<double>(angular_bins);
        brightness_sum += brightness;
    }

    if (brightness_sum <= 0.0 || !std::isfinite(brightness_sum)) {
        return {};
    }
    return {
        weighted_magnification / brightness_sum,
        sample_count,
        0,
    };
}

std::vector<Complex> critical_curve_polynomial_coefficients(double separation, double mass_ratio, Complex phase)
{
    const double s = std::abs(separation);
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const Complex a = q_input < 1.0 ? Complex(-s, 0.0) : Complex(s, 0.0);
    const Complex m1 = 1.0 / (1.0 + q);
    const Complex m2 = q * m1;
    const Complex a2 = a * a;

    return {
        m2 * a2,
        -2.0 * m2 * a,
        m1 + m2 - phase * a2,
        2.0 * phase * a,
        -phase,
    };
}

std::vector<SourcePosition> caustic_points_at_phase(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    double phase_angle)
{
    const Complex phase = std::polar(1.0, phase_angle);
    math::PolynomialRootSolver solver;
    const auto roots = solver.solve(critical_curve_polynomial_coefficients(separation, mass_ratio, phase));
    if (roots.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> points;
    points.reserve(roots.roots.size());
    for (const auto& root : roots.roots) {
        points.push_back(point_magnifier.binary_lens_equation(
            separation, mass_ratio, {root.real(), root.imag()}));
    }
    return points;
}

std::vector<SourcePosition> critical_curve_points_at_phase(
    double separation,
    double mass_ratio,
    double phase_angle)
{
    const Complex phase = std::polar(1.0, phase_angle);
    math::PolynomialRootSolver solver;
    const auto roots = solver.solve(critical_curve_polynomial_coefficients(separation, mass_ratio, phase));
    if (roots.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> points;
    points.reserve(roots.roots.size());
    for (const auto& root : roots.roots) {
        points.push_back({root.real(), root.imag()});
    }
    return points;
}

void append_tracked_caustic_points(
    std::vector<std::vector<SourcePosition>>& branches,
    std::vector<SourcePosition> points)
{
    const std::size_t n = branches.size();
    if (points.size() != n) {
        return;
    }

    if (branches[0].empty()) {
        std::sort(points.begin(), points.end(), [](const auto& lhs, const auto& rhs) {
            return std::atan2(lhs.y, lhs.x) < std::atan2(rhs.y, rhs.x);
        });
        for (std::size_t i = 0; i < n; ++i) {
            branches[i].push_back(points[i]);
        }
        return;
    }

    // Find the permutation of `points` that minimises total squared step length.
    // Greedy nearest-neighbour can swap inner/outer caustic branches when they
    // come close; the global optimum never makes a swap unless the two assignments
    // have identical total cost (branches genuinely coincide), avoiding spurious
    // long segments in the branch grid.  For n=4 this is 4!=24 permutations.
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<std::size_t> best_perm = perm;
    double best_cost = std::numeric_limits<double>::infinity();
    do {
        double cost = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            cost += distance_squared(branches[i].back(), points[perm[i]]);
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_perm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    for (std::size_t i = 0; i < n; ++i) {
        branches[i].push_back(points[best_perm[i]]);
    }
}

// A root followed through one 0..2pi phase sweep need not return to its own
// starting point.  At a branch point it can continue on another tracked root,
// so the raw root-index branches are not necessarily physical closed curves.
// Reconstruct the monodromy cycles using the same endpoint rule as
// VBMicrolensing::PlotCrit: continue onto the closest remaining branch when
// that is nearer than closing the current curve back to its own start.
std::vector<std::vector<SourcePosition>> merge_tracked_phase_branches(
    std::vector<std::vector<SourcePosition>> branches)
{
    std::vector<std::vector<SourcePosition>> curves;
    curves.reserve(branches.size());
    while (!branches.empty()) {
        std::vector<SourcePosition> curve = std::move(branches.front());
        branches.erase(branches.begin());
        if (curve.empty()) {
            continue;
        }

        while (!branches.empty()) {
            const double closing_distance =
                distance_squared(curve.back(), curve.front());
            std::size_t closest = branches.size();
            double closest_distance = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < branches.size(); ++i) {
                if (branches[i].empty()) {
                    continue;
                }
                const double candidate =
                    distance_squared(curve.back(), branches[i].front());
                if (candidate < closest_distance) {
                    closest_distance = candidate;
                    closest = i;
                }
            }
            if (closest == branches.size() ||
                !(closest_distance < closing_distance)) {
                break;
            }

            auto& continuation = branches[closest];
            curve.insert(
                curve.end(),
                std::make_move_iterator(continuation.begin()),
                std::make_move_iterator(continuation.end()));
            branches.erase(branches.begin() + static_cast<std::ptrdiff_t>(closest));
        }
        curves.push_back(std::move(curve));
    }
    return curves;
}

std::vector<std::vector<SourcePosition>> build_binary_critical_curves(
    double separation,
    double mass_ratio,
    int bins)
{
    std::vector<std::vector<SourcePosition>> branches(4);
    // Experimental proposal path. Residual/separation checks are numerical
    // guards, not a continuous-arc certificate. Keep the default quartic path.
    // Below 512 phases the measured prediction cost did not consistently
    // beat the quartic solver. This is a cost gate, not an accuracy criterion.
    const bool continuation = bins>=512 &&
        std::getenv("LCBININT_CRITICAL_CONTINUATION") != nullptr;
    const bool stats=probe_stats_enabled();
    const auto started=stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const double q_input = std::abs(mass_ratio);
    const double q = q_input < 1.0 ? q_input : 1.0 / q_input;
    const double m1 = 1.0 / (1.0 + q), m2 = q * m1;
    const Complex a(q_input < 1.0 ? -std::abs(separation) : std::abs(separation), 0.0);
    const double step = 2.0 * kPi / static_cast<double>(bins);
    struct Jet { Complex value,derivative,second; };
    auto inverse = [](Complex z) {
        const double norm=std::norm(z);
        if (!(norm>0.0) || !std::isfinite(norm))
            return Complex(std::numeric_limits<double>::quiet_NaN(),0.0);
        return std::conj(z)*(1.0/norm);
    };
    auto evaluate = [&](Complex z) {
        const Complex u=inverse(z-a),v=inverse(z);
        const Complex p=m1*u*u,t=m2*v*v;
        return Jet{p+t,-2.0*(p*u+t*v),6.0*(p*u*u+t*v*v)};
    };
    std::array<Jet,4> previous_jets{};
    bool jets_valid=false;
    for (int i = 0; i < bins; ++i) {
        const double phase_angle = 2.0 * kPi * static_cast<double>(i) /
                                   static_cast<double>(bins);
        std::vector<SourcePosition> points;
        bool preserve_order=false;
        if (continuation && jets_valid && !branches[0].empty()) {
            const Complex target = std::polar(1.0, phase_angle);
            const Complex previous = std::polar(1.0, phase_angle-step);
            std::size_t index=0;
            const Complex phase_delta=target-previous;
            for (const auto& branch : branches) {
                Complex z(branch.back().x, branch.back().y);
                const Jet old=previous_jets[index++];
                const Complex inv=inverse(old.derivative);
                const Complex linear=phase_delta*inv;
                z += linear-0.5*old.second*inv*linear*linear;
                bool converged = false;
                for (int iteration=0; iteration<8; ++iteration) {
                    if (stats) ++probe_counters().critical_newton_iterations;
                    const Jet jet=evaluate(z);
                    const Complex correction=(jet.value-target)*inverse(jet.derivative);
                    z -= correction;
                    if (!std::isfinite(std::abs(z))) break;
                    if (std::abs(correction) <= 8.0*std::numeric_limits<double>::epsilon()
                        *std::max(1.0,std::abs(z))) {
                        converged = std::abs(evaluate(z).value-target) <= 1.0e-11;
                        break;
                    }
                }
                if (!converged) { points.clear(); break; }
                points.push_back({z.real(),z.imag()});
            }
            for (std::size_t j=0; j<points.size(); ++j)
                for (std::size_t k=0; k<j; ++k)
                    if (distance_squared(points[j],points[k]) <= 1.0e-20
                        *std::max(1.0,points[j].x*points[j].x+points[j].y*points[j].y)) {
                        points.clear(); break;
                    }
            if (points.size()==4) {
                // If every proposal remains in its own disjoint neighbourhood,
                // preserve the known labels instead of enumerating 24 matches.
                preserve_order=true;
                for (std::size_t j=0;j<4;++j) {
                    double nearest=std::numeric_limits<double>::infinity();
                    for (std::size_t k=0;k<4;++k) if (j!=k)
                        nearest=std::min(nearest,distance_squared(branches[j].back(),branches[k].back()));
                    if (!(16.0*distance_squared(points[j],branches[j].back())<nearest))
                        preserve_order=false;
                }
            }
        }
        // Only this phase is unresolved. Previously sampled phases, their
        // branch correspondence and subsequent cached seeds remain intact.
        if (points.size()!=4) {
            if (stats) {
                ++probe_counters().critical_phase_solves;
                if (continuation && i>0) ++probe_counters().critical_tracking_failures;
            }
            points=critical_curve_points_at_phase(separation,mass_ratio,phase_angle);
        } else if (stats) ++probe_counters().critical_phase_tracked;
        if (preserve_order) {
            for (std::size_t j=0;j<4;++j) branches[j].push_back(points[j]);
        } else append_tracked_caustic_points(branches,std::move(points));
        jets_valid=continuation && !branches[0].empty();
        if (jets_valid) for (std::size_t j=0;j<4;++j)
            previous_jets[j]=evaluate(Complex(branches[j].back().x,branches[j].back().y));
    }
    if (stats) probe_counters().critical_build_seconds+=
        std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return merge_tracked_phase_branches(std::move(branches));
}

std::vector<std::vector<SourcePosition>> map_binary_critical_curves_to_caustics(
    const std::vector<std::vector<SourcePosition>>& critical_curves,
    double separation,
    double mass_ratio)
{
    const PointSourceMagnifier point_magnifier;
    std::vector<std::vector<SourcePosition>> caustics;
    caustics.reserve(critical_curves.size());
    for (const auto& critical_curve : critical_curves) {
        auto& caustic = caustics.emplace_back();
        caustic.reserve(critical_curve.size());
        for (const auto& image : critical_curve) {
            caustic.push_back(point_magnifier.binary_lens_equation(
                separation, mass_ratio, image));
        }
    }
    return caustics;
}

struct TripleCausticBranches {
    std::vector<std::vector<SourcePosition>> branches;
    int bins = 0;
};

TripleCausticBranches build_triple_caustic_branches(
    const model::TripleLensGeometry& geometry,
    int caustic_bins)
{
    const int bins = std::max(caustic_bins, 64);
    TripleCausticBranches result;
    result.bins = bins;
    std::vector<std::vector<SourcePosition>> critical_branches(6);
    for (int i = 0; i < bins; ++i) {
        const double phase_angle = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(bins);
        append_tracked_caustic_points(
            critical_branches,
            triple_critical_curve_points_at_phase(geometry, phase_angle));
    }
    critical_branches = merge_tracked_phase_branches(std::move(critical_branches));
    result.branches.reserve(critical_branches.size());
    for (const auto& critical_curve : critical_branches) {
        auto& caustic = result.branches.emplace_back();
        caustic.reserve(critical_curve.size());
        for (const auto& image : critical_curve) {
            caustic.push_back(model::triple_lens_equation(geometry, image));
        }
    }
    return result;
}

bool same_triple_geometry(
    const model::TripleLensGeometry& lhs,
    const model::TripleLensGeometry& rhs)
{
    for (std::size_t i = 0; i < lhs.lens_positions.size(); ++i) {
        if (lhs.lens_positions[i].x != rhs.lens_positions[i].x ||
            lhs.lens_positions[i].y != rhs.lens_positions[i].y ||
            lhs.masses[i] != rhs.masses[i]) {
            return false;
        }
    }
    return true;
}

const TripleCausticBranches& cached_triple_caustic_branches(
    const model::TripleLensGeometry& geometry,
    int caustic_bins)
{
    struct Cache {
        bool valid = false;
        model::TripleLensGeometry geometry;
        int caustic_bins = 0;
        TripleCausticBranches caustics;
    };
    thread_local Cache cache;
    if (!cache.valid ||
        cache.caustic_bins != caustic_bins ||
        !same_triple_geometry(cache.geometry, geometry)) {
        cache.geometry = geometry;
        cache.caustic_bins = caustic_bins;
        cache.caustics = build_triple_caustic_branches(geometry, caustic_bins);
        cache.valid = true;
    }
    return cache.caustics;
}

double nearest_triple_caustic_distance_at_phase(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double phase_angle)
{
    double best = std::numeric_limits<double>::infinity();
    const auto points = triple_caustic_points_at_phase(geometry, phase_angle);
    for (const auto& point : points) {
        best = std::min(best, std::sqrt(distance_squared(source, point)));
    }
    return best;
}

double refine_triple_caustic_distance(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double center_phase,
    double phase_step)
{
    double left = center_phase - phase_step;
    double right = center_phase + phase_step;
    constexpr double golden = 0.61803398874989484820;
    double x1 = right - golden * (right - left);
    double x2 = left + golden * (right - left);
    double f1 = nearest_triple_caustic_distance_at_phase(geometry, source, x1);
    double f2 = nearest_triple_caustic_distance_at_phase(geometry, source, x2);
    for (int iter = 0; iter < 24; ++iter) {
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + golden * (right - left);
            f2 = nearest_triple_caustic_distance_at_phase(geometry, source, x2);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = right - golden * (right - left);
            f1 = nearest_triple_caustic_distance_at_phase(geometry, source, x1);
        }
    }
    return std::min(f1, f2);
}

double triple_caustic_distance(
    const model::TripleLensGeometry& geometry,
    const TripleCausticBranches& caustics,
    SourcePosition source,
    double refine_within = std::numeric_limits<double>::infinity())
{
    double best = std::numeric_limits<double>::infinity();
    double best_phase = 0.0;
    const double phase_step = caustics.bins > 0
        ? 2.0 * kPi / static_cast<double>(caustics.bins)
        : 2.0 * kPi / 64.0;
    for (const auto& branch : caustics.branches) {
        if (branch.size() < 2) {
            continue;
        }
        for (std::size_t i = 1; i < branch.size(); ++i) {
            const double distance = point_segment_distance(source, branch[i - 1], branch[i]);
            if (distance < best) {
                best = distance;
                best_phase = (static_cast<double>(i) - 0.5) * phase_step;
            }
        }
        const double wrap_distance =
            point_segment_distance(source, branch.back(), branch.front());
        if (wrap_distance < best) {
            best = wrap_distance;
            best_phase = 0.0;
        }
    }
    // The golden-section refinement costs ~26 degree-10 root solves; it only
    // matters when the polyline distance sits near a decision threshold, so
    // skip it for clearly-far sources.
    if (std::isfinite(best) && best < refine_within) {
        best = std::min(best, refine_triple_caustic_distance(
            geometry,
            source,
            best_phase,
            phase_step));
    }
    return best;
}

double binary_caustic_distance(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    int caustic_bins)
{
    const int bins = std::max(caustic_bins, 32);
    std::vector<std::vector<SourcePosition>> branches(4);
    for (int i = 0; i < bins; ++i) {
        const double phase_angle = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(bins);
        append_tracked_caustic_points(
            branches, caustic_points_at_phase(point_magnifier, separation, mass_ratio, phase_angle));
    }

    double distance = std::numeric_limits<double>::infinity();
    for (const auto& branch : branches) {
        if (branch.size() < 2) {
            continue;
        }
        for (std::size_t i = 1; i < branch.size(); ++i) {
            distance = std::min(distance, point_segment_distance(source, branch[i - 1], branch[i]));
        }
        distance = std::min(distance, point_segment_distance(source, branch.back(), branch.front()));
    }
    return distance;
}

double image_radius(double source_radius, double determinant)
{
    const double abs_det = std::max(std::abs(determinant), 1.0e-8);
    return std::max(2.5 * source_radius / std::sqrt(abs_det), 2.0 * source_radius);
}

struct HexResult {
    double magnification;
    double relative_error; // |a4 correction| / |magnification|, used for VBM-style mode switch
};

HexResult hexadecapole_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const double* known_point_magnification = nullptr)
{
    const double sineta[8] = {0.0, 1.0, 0.0, -1.0, kSqrtHalf, kSqrtHalf, -kSqrtHalf, -kSqrtHalf};
    const double coseta[8] = {1.0, 0.0, -1.0, 0.0, kSqrtHalf, -kSqrtHalf, -kSqrtHalf, kSqrtHalf};

    const double a0 = known_point_magnification != nullptr ?
        *known_point_magnification :
        point_magnifier.binary_mag0(separation, mass_ratio, source).magnification;
    std::array<SourcePosition, 12> sample_sources;
    std::array<double, 12> sample_magnifications;
    int sample_index = 0;
    for (int i = 0; i < 4; ++i) {
        sample_sources[static_cast<std::size_t>(sample_index++)] =
            {source.x + source_radius * coseta[i],
                source.y + source_radius * sineta[i]};
        sample_sources[static_cast<std::size_t>(sample_index++)] =
            {source.x + 0.5 * source_radius * coseta[i],
                source.y + 0.5 * source_radius * sineta[i]};
        sample_sources[static_cast<std::size_t>(sample_index++)] =
            {source.x + source_radius * coseta[i + 4],
                source.y + source_radius * sineta[i + 4]};
    }
    point_magnifier.binary_mag0_batch(
        separation, mass_ratio, sample_sources.data(), sample_magnifications.data(),
        sample_sources.size());

    double a1_plus = 0.0;
    double a2_plus = 0.0;
    double a1_cross = 0.0;
    sample_index = 0;
    for (int i = 0; i < 4; ++i) {
        a1_plus += sample_magnifications[static_cast<std::size_t>(sample_index++)];
        a2_plus += sample_magnifications[static_cast<std::size_t>(sample_index++)];
        a1_cross += sample_magnifications[static_cast<std::size_t>(sample_index++)];
    }
    a1_plus = a1_plus / 4.0 - a0;
    a2_plus = a2_plus / 4.0 - a0;
    a1_cross = a1_cross / 4.0 - a0;

    const double a2rho2 = (16.0 * a2_plus - a1_plus) / 3.0;
    const double a4rho4 = (a1_plus + a1_cross) / 2.0 - a2rho2;
    const double gamma = limb_darkening_gamma(settings);
    const double lambda = limb_darkening_lambda(settings);
    const double quad_corr = 0.5 * a2rho2 * (1.0 - 0.2 * gamma - lambda / 9.0);
    const double hex_corr = a4rho4 / 3.0 * (1.0 - 11.0 * gamma / 35.0 - 7.0 * lambda / 39.0);
    const double magnification = a0 + quad_corr + hex_corr;
    const double rel_err = std::abs(hex_corr) / std::max(std::abs(magnification), 1.0e-10);
    return {magnification, rel_err};
}

HexResult hexadecapole_triple(
    const PointSourceMagnifier& point_magnifier,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const double* known_point_magnification = nullptr)
{
    const double sineta[8] = {0.0, 1.0, 0.0, -1.0, kSqrtHalf, kSqrtHalf, -kSqrtHalf, -kSqrtHalf};
    const double coseta[8] = {1.0, 0.0, -1.0, 0.0, kSqrtHalf, -kSqrtHalf, -kSqrtHalf, kSqrtHalf};

    const double a0 = known_point_magnification != nullptr ?
        *known_point_magnification :
        point_magnifier.triple_mag0(geometry, source).magnification;

    double a1_plus = 0.0;
    double a2_plus = 0.0;
    double a1_cross = 0.0;
    for (int i = 0; i < 4; ++i) {
        SourcePosition sample {
            source.x + source_radius * coseta[i],
            source.y + source_radius * sineta[i],
        };
        a1_plus += point_magnifier.triple_mag0(geometry, sample).magnification;

        sample = {
            source.x + 0.5 * source_radius * coseta[i],
            source.y + 0.5 * source_radius * sineta[i],
        };
        a2_plus += point_magnifier.triple_mag0(geometry, sample).magnification;

        sample = {
            source.x + source_radius * coseta[i + 4],
            source.y + source_radius * sineta[i + 4],
        };
        a1_cross += point_magnifier.triple_mag0(geometry, sample).magnification;
    }
    a1_plus = a1_plus / 4.0 - a0;
    a2_plus = a2_plus / 4.0 - a0;
    a1_cross = a1_cross / 4.0 - a0;

    const double a2rho2 = (16.0 * a2_plus - a1_plus) / 3.0;
    const double a4rho4 = (a1_plus + a1_cross) / 2.0 - a2rho2;
    const double gamma = limb_darkening_gamma(settings);
    const double lambda = limb_darkening_lambda(settings);
    const double quad_corr = 0.5 * a2rho2 * (1.0 - 0.2 * gamma - lambda / 9.0);
    const double hex_corr = a4rho4 / 3.0 * (1.0 - 11.0 * gamma / 35.0 - 7.0 * lambda / 39.0);
    const double magnification = a0 + quad_corr + hex_corr;
    const double rel_err = std::abs(hex_corr) / std::max(std::abs(magnification), 1.0e-10);
    return {magnification, rel_err};
}

// A polar grid can have millions of possible angular columns when rho is
// small, even though each connected image visits contiguous angular arcs.
// Allocate the direct-index table lazily in fixed pages: lookup stays O(1)
// without hashing, while unvisited parts of the circle construct no Columns.
class PolarVisitedCellIntervals {
public:
    using Interval = std::pair<int, int>;

    struct Column {
        Interval first {1, 0};
        std::vector<Interval> overflow;

        bool contains(int ir) const
        {
            if (first.first > first.second || ir < first.first) {
                return false;
            }
            if (ir <= first.second) {
                return true;
            }
            for (const auto& interval : overflow) {
                if (ir < interval.first) {
                    return false;
                }
                if (ir <= interval.second) {
                    return true;
                }
            }
            return false;
        }

        std::int64_t first_unvisited_at_or_after(std::int64_t ir) const
        {
            if (first.first > first.second || ir < first.first) {
                return ir;
            }
            if (ir <= first.second) {
                ir = static_cast<std::int64_t>(first.second) + 1;
            }
            for (const auto& interval : overflow) {
                if (ir < interval.first) {
                    return ir;
                }
                if (ir <= interval.second) {
                    ir = static_cast<std::int64_t>(interval.second) + 1;
                }
            }
            return ir;
        }

        void add(int left, int right)
        {
            if (right < left) {
                return;
            }
            if (first.first > first.second) {
                first = {left, right};
                return;
            }
            if (static_cast<std::int64_t>(right) + 1 < first.first) {
                overflow.insert(overflow.begin(), first);
                first = {left, right};
                return;
            }
            if (static_cast<std::int64_t>(left) <=
                static_cast<std::int64_t>(first.second) + 1) {
                first.first = std::min(first.first, left);
                first.second = std::max(first.second, right);
                std::size_t merged = 0;
                while (merged < overflow.size() &&
                       static_cast<std::int64_t>(overflow[merged].first) <=
                           static_cast<std::int64_t>(first.second) + 1) {
                    first.second = std::max(first.second, overflow[merged].second);
                    ++merged;
                }
                if (merged != 0) {
                    overflow.erase(
                        overflow.begin(),
                        overflow.begin() + static_cast<std::ptrdiff_t>(merged));
                }
                return;
            }

            auto it = std::lower_bound(
                overflow.begin(), overflow.end(), left,
                [](const Interval& interval, int value) {
                    return interval.first < value;
                });
            if (it != overflow.begin() &&
                static_cast<std::int64_t>(std::prev(it)->second) + 1 >= left) {
                --it;
                it->second = std::max(it->second, right);
            } else {
                it = overflow.insert(it, {left, right});
            }
            while (std::next(it) != overflow.end() &&
                   static_cast<std::int64_t>(std::next(it)->first) <=
                       static_cast<std::int64_t>(it->second) + 1) {
                it->second = std::max(it->second, std::next(it)->second);
                overflow.erase(std::next(it));
            }
        }
    };

    explicit PolarVisitedCellIntervals(int phi_bins)
        : pages_((static_cast<std::size_t>(phi_bins) + kColumnsPerPage - 1) /
                 kColumnsPerPage)
    {}

    const Column* find(int iphi) const
    {
        const std::size_t index = static_cast<std::size_t>(iphi);
        const auto& page = pages_[index / kColumnsPerPage];
        return page == nullptr
            ? nullptr
            : &page->columns[index % kColumnsPerPage];
    }

    Column& get_or_create(int iphi)
    {
        const std::size_t index = static_cast<std::size_t>(iphi);
        auto& page = pages_[index / kColumnsPerPage];
        if (page == nullptr) {
            page = std::make_unique<Page>();
        }
        return page->columns[index % kColumnsPerPage];
    }

    void direction(int iphi, double dphi, double& cosine, double& sine)
    {
        const std::size_t index = static_cast<std::size_t>(iphi);
        auto& page = pages_[index / kColumnsPerPage];
        if (page == nullptr) {
            page = std::make_unique<Page>();
        }
        const std::size_t offset = index % kColumnsPerPage;
        const std::uint64_t bit = std::uint64_t {1} << offset;
        if ((page->direction_mask & bit) == 0) {
            const double phi = (static_cast<double>(iphi) + 0.5) * dphi;
            page->cosine[offset] = std::cos(phi);
            page->sine[offset] = std::sin(phi);
            page->direction_mask |= bit;
        }
        cosine = page->cosine[offset];
        sine = page->sine[offset];
    }

private:
    static constexpr std::size_t kColumnsPerPage = 64;
    struct Page {
        std::array<Column, kColumnsPerPage> columns;
        std::array<double, kColumnsPerPage> cosine {};
        std::array<double, kColumnsPerPage> sine {};
        std::uint64_t direction_mask = 0;
    };
    std::vector<std::unique_ptr<Page>> pages_;
};

struct PolarAreaDiagnostics {
    bool seed_coverage_proven=true;
    double estimated_error = std::numeric_limits<double>::infinity();
};

struct LegacyImageAreaScratch {
    std::vector<double> xmin;
    std::vector<double> xmax;
    std::vector<double> ax;
    std::vector<double> y;
    std::vector<double> dys;

    void reset()
    {
        // clear() preserves capacity.  Values must nevertheless be discarded:
        // several gap-detection entries rely on resize value-initialising a
        // row that the current component has not visited yet.
        xmin.clear();
        xmax.clear();
        ax.clear();
        y.clear();
        dys.clear();
    }

    void ensure(std::size_t index)
    {
        if (xmin.size() <= index) {
            const std::size_t size = index + 1;
            xmin.resize(size);
            xmax.resize(size);
            ax.resize(size);
            y.resize(size);
            dys.resize(size);
        }
    }
};

// Per-epoch registry of grid cells already counted by any flood-fill.  All
// fills are anchored on the shared lattice x = ix*incr, y = iy*incr, so cell
// identity is exact; counting only unclaimed cells makes the integrated area
// independent of the seed set and seed order by construction and replaces
// the former row-bbox overlap heuristics.  Rows hold few merged intervals,
// so linear scans dominate hash-map cost only in degenerate cases.
// `owner` records which fill first counted the run.  A fill that walks into a
// run owned by *another* fill has been stopped by a neighbour rather than by
// its own geometry -- the fold-pair case, where one member's territory ends
// where the other's begins -- and that is what disqualifies it from being
// re-integrated on a private lattice.  Adjacent runs are merged only when they
// share an owner: merging across owners would erase the boundary between two
// components, which is the one thing the refinement has to be able to see.
struct ClaimedRun {
    std::int64_t lo;
    std::int64_t hi;
    int owner;
};

struct ClaimedCellRuns {
    using Row = std::vector<ClaimedRun>;

    std::unordered_map<std::int64_t, Row> rows;
    int owner = 0;  // stamped on runs claimed from now on

    const Row* find_row(std::int64_t iy) const
    {
        const auto it = rows.find(iy);
        return it == rows.end() ? nullptr : &it->second;
    }

    static const ClaimedRun* find_in_row(const Row* row, std::int64_t ix)
    {
        if (row == nullptr) {
            return nullptr;
        }
        for (const auto& interval : *row) {
            if (ix < interval.lo) {
                return nullptr;
            }
            if (ix >= interval.lo && ix <= interval.hi) {
                return &interval;
            }
        }
        return nullptr;
    }

    const ClaimedRun* find(std::int64_t iy, std::int64_t ix) const
    {
        return find_in_row(find_row(iy), ix);
    }

    void claim(std::int64_t iy, std::int64_t lo, std::int64_t hi)
    {
        if (hi < lo) {
            return;
        }
        auto& intervals = rows[iy];
        // Rows are maintained in increasing-lo order.  A new claim can only
        // change its immediate same-owner neighbours, because earlier claims
        // have already been merged.  Inserting locally avoids sorting and
        // rescanning the complete row for every contiguous run.
        auto inserted = std::lower_bound(
            intervals.begin(), intervals.end(), lo,
            [](const ClaimedRun& run, std::int64_t value) {
                return run.lo < value;
            });
        inserted = intervals.insert(inserted, {lo, hi, owner});

        if (inserted != intervals.begin()) {
            auto previous = inserted - 1;
            if (previous->owner == inserted->owner &&
                inserted->lo <= previous->hi + 1) {
                previous->hi = std::max(previous->hi, inserted->hi);
                inserted = intervals.erase(inserted);
                inserted = previous;
            }
        }
        auto next = inserted + 1;
        while (next != intervals.end() &&
               next->owner == inserted->owner &&
               next->lo <= inserted->hi + 1) {
            inserted->hi = std::max(inserted->hi, next->hi);
            next = intervals.erase(next);
        }
    }
};

// Lattice coordinates can be much larger than source_bins when rho is tiny
// or an image lies far from the origin.  Keep them 64-bit throughout, but stay
// well inside double's exact-integer range so x and x +/- one cell remain
// distinguishable after converting the index back to an image coordinate.
// More extreme inputs fail closed instead of silently sampling one coordinate
// under several different cell identities.
constexpr double kMaximumLatticeIndexMagnitude = 0x1p51;

bool rounded_lattice_index(double scaled, std::int64_t& index)
{
    if (!std::isfinite(scaled) ||
        !(scaled > -kMaximumLatticeIndexMagnitude &&
          scaled < kMaximumLatticeIndexMagnitude)) {
        return false;
    }
    index = static_cast<std::int64_t>(std::llround(scaled));
    return true;
}

struct LatticeCellIndex {
    std::int64_t x;
    std::int64_t y;

    bool operator==(const LatticeCellIndex& other) const
    {
        return x == other.x && y == other.y;
    }
};

struct LatticeCellIndexHash {
    std::size_t operator()(const LatticeCellIndex& cell) const noexcept
    {
        const std::size_t hx = std::hash<std::int64_t> {}(cell.x);
        const std::size_t hy = std::hash<std::int64_t> {}(cell.y);
        return hx ^ (hy + static_cast<std::size_t>(0x9e3779b9U) +
                     (hx << 6) + (hx >> 2));
    }
};

struct LegacyAreaDiagnostics {
    bool seed_coverage_proven=true;
    int seed_count = 0;
    int processed_images = 0;
    int fold_seed_count = 0;
    int boundary_rows = 0;
    int gap_repairs = 0;
    int overlaps = 0;
    int refined_components = 0;
    int refinement_factor = 0;  // largest per-component factor applied
    int interval_count = 0;
    std::int64_t mapped_cell_evaluations = 0;
    std::int64_t frontier_intervals_popped = 0;
    std::int64_t rows_with_multiple_runs = 0;
    std::size_t maximum_runs_in_row = 0;
    std::int64_t unique_seed_cells = 0;
    std::int64_t duplicate_seed_starts_avoided = 0;
    std::int64_t provisional_components = 0;
    std::int64_t merged_components = 0;
    std::int64_t fine_grid_cell_evaluations = 0;
    std::int64_t walk_budget_failures = 0;
    std::int64_t integrated_cells = 0;
    double max_jump_cells = 0.0;
    double estimated_error = 0.0;
    // The run fill integrates the union of fold partners, so a near-critical
    // seed is diagnostic information but does not create the artificial
    // parity boundary used by the legacy walker.
    bool parity_free_topology = false;
};

int cartesian_error_convergence_order(const LegacyAreaDiagnostics& diagnostics)
{
    return (diagnostics.parity_free_topology || diagnostics.fold_seed_count == 0) &&
        diagnostics.max_jump_cells <= 20.0 &&
        diagnostics.rows_with_multiple_runs == 0
        ? 2
        : 1;
}

double high_magnification_floor_coefficient(
    const LegacyAreaDiagnostics& diagnostics,
    double magnification,
    double source_radius)
{
    const std::int64_t topology_events =
        static_cast<std::int64_t>(diagnostics.gap_repairs) +
        diagnostics.rows_with_multiple_runs;
    if (std::abs(magnification) > 1000.0 &&
        diagnostics.rows_with_multiple_runs > 1000) {
        // A coarse lattice that repeatedly resolves more than one interval in
        // a row has real topology but not enough transverse samples to certify
        // a percent-level, very-high-magnification answer.  Keep the adaptive
        // caller fail-closed until a finer rung removes that ambiguity.
        return 1.0;
    }
    if (source_radius >= 0.1 &&
        (diagnostics.gap_repairs > 0 || diagnostics.max_jump_cells > 50.0)) {
        return 0.06;
    }
    if (source_radius < 1.0e-2 &&
        diagnostics.seed_count <= 5 &&
        std::abs(magnification) <= 10.0 &&
        diagnostics.gap_repairs > 0 &&
        diagnostics.max_jump_cells > 20.0) {
        return 0.035;
    }
    if (source_radius < 1.0e-2 &&
        diagnostics.seed_count <= 5 &&
        std::abs(magnification) <= 10.0 &&
        diagnostics.max_jump_cells > 1000.0) {
        return 0.04;
    }
    if (source_radius < 1.0e-2 && std::abs(magnification) <= 300.0) {
        return 0.02;
    }
    if (source_radius < 1.0e-2 && std::abs(magnification) <= 1000.0) {
        return 0.04;
    }
    if (source_radius < 1.0e-2 &&
        diagnostics.seed_count <= 5 &&
        diagnostics.gap_repairs > 0 &&
        diagnostics.max_jump_cells > 20.0) {
        return 0.04;
    }
    if (std::abs(magnification) <= 80.0 || topology_events <= 1000) {
        return 0.0;
    }
    return source_radius >= 1.0e-2 ? 0.04 : 0.05;
}

double cartesian_area_error_indicator(
    const LegacyAreaDiagnostics& diagnostics,
    double source_radius,
    const FiniteSourceSettings& settings)
{
    const int bins = std::max(settings.source_bins, 1);
    const double flux = source_flux(source_radius, settings);
    if (!std::isfinite(flux) || flux <= 0.0 || source_radius <= 0.0) {
        return 0.0;
    }

    // The (t - 0.5) edge correction makes an ordinary smooth image boundary
    // second order.  Fold components and large row-to-row jumps can degrade
    // the legacy, parity-split walk back toward first order.  The run fill
    // integrates fold partners as one union, so its near-critical seeds do not
    // by themselves lower the convergence order.  A small-jump, no-fold scan
    // gets the extra cell-width factor expected from the corrected boundary
    // rule.  Gap repairs participate in the same scaling there: without a
    // large jump they are scan-continuation events, not evidence of missing
    // image area.  With the claimed-cell registry cross-seed overlaps can no
    // longer occur, so the old overlap terms are gone.
    const double gap_weight = source_radius >= 2.0e-2
        ? 0.005
        : ((source_radius < 1.0e-2 && diagnostics.seed_count >= 16) ? 0.015 : 0.03);
    const double boundary_weight = source_radius >= 2.0e-2 ? 0.012 : 0.03;
    double uncertain_cells =
        boundary_weight * static_cast<double>(diagnostics.boundary_rows) +
        gap_weight * static_cast<double>(diagnostics.gap_repairs) +
        0.02 * static_cast<double>(std::max(0, diagnostics.fold_seed_count)) +
        0.002 * static_cast<double>(std::max(0, diagnostics.seed_count - 5));
    if (source_radius >= 1.0e-3 && diagnostics.max_jump_cells > 10.0) {
        const double jump_weight = source_radius >= 2.0e-2 ? 0.2 : 4.0;
        uncertain_cells += jump_weight * std::log10(diagnostics.max_jump_cells / 10.0);
    }
    if (cartesian_error_convergence_order(diagnostics) == 2) {
        constexpr double kSmoothBoundarySafetyCells = 8.0;
        uncertain_cells *= std::min(
            1.0, kSmoothBoundarySafetyCells / static_cast<double>(bins));
    }
    const double cell_area = (source_radius / static_cast<double>(bins)) *
                             (source_radius / static_cast<double>(bins));
    const double estimate = 1.25 * uncertain_cells * cell_area / flux;
    return std::isfinite(estimate) ? estimate : 0.0;
}

double wrap_angle(double angle)
{
    while (angle < 0.0) {
        angle += 2.0 * kPi;
    }
    while (angle >= 2.0 * kPi) {
        angle -= 2.0 * kPi;
    }
    return angle;
}

template <typename ImageMap>
bool find_polar_inside_start(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    SourcePosition image_seed,
    double dr,
    double dphi,
    int phi_bins,
    double* start_radius,
    double* start_phi);

template <typename ImageMap>
double inverse_ray_polar_core(
    const ImageMap& mapper,
    const std::vector<SourcePosition>& image_positions,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    PolarAreaDiagnostics* diagnostics = nullptr)
{
    if (image_positions.empty()) {
        return std::nan("");
    }

    const bool fill_stats=probe_stats_enabled();
    const auto fill_started=fill_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    long long map_queries=0,fill_cells=0;
    const int source_bins = active_polar_source_bins(settings);
    const double polar_grid_ratio = active_polar_grid_ratio(settings);
    const double dr = source_radius / static_cast<double>(source_bins);
    double max_image_radius = 1.0;
    for (const auto& image_position : image_positions) {
        max_image_radius = std::max(
            max_image_radius,
            std::hypot(image_position.x, image_position.y) + 4.0 * source_radius);
    }
    if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
        if (std::getenv("LCBININT_POLAR_GEOMETRY_GRID")!=nullptr) {
            max_image_radius=binary_polar_geometry_radius(mapper,source,source_radius);
            if (!std::isfinite(max_image_radius)) return std::nan("");
        }
    }
    // Diagnostic override isolates seed-dependent angular resolution from
    // seed-dependent coverage. It is not an image-support/error certificate.
    if (const char* radius_override=std::getenv("LCBININT_POLAR_GRID_RADIUS")) {
        char* end=nullptr;
        const double radius=std::strtod(radius_override,&end);
        if (end!=radius_override && *end=='\0' && std::isfinite(radius) && radius>0.0)
            max_image_radius=std::max(1.0,radius);
    }
    const bool seed_trace=std::getenv("LCBININT_POLAR_SEED_TRACE")!=nullptr;
    std::uint64_t trace_cells=0, trace_signature=0;
    // Choose angular resolution from the tangential cell size at the outermost
    // relevant image.  Using dphi ~ dr alone undersamples low-magnification
    // images far from the origin.
    const int phi_bins = std::max(
        16,
        static_cast<int>(std::ceil(2.0 * kPi * max_image_radius /
                                   (dr * polar_grid_ratio))));
    const double dphi = 2.0 * kPi / static_cast<double>(phi_bins);
    // Eight-neighbour reachability is required by reduced seeds independently
    // of any continuous-coverage audit. Never pay for that audit just to enable
    // the correct frontier.
    const char* reduced_seed_mode=std::getenv("LCBININT_REPRESENTATIVE_PROBES");
    const bool diagonal_frontier=std::is_same_v<ImageMap,BinaryLensMapper> &&
        (std::getenv("LCBININT_POLAR_DIAGONAL_FRONTIER")
         || (reduced_seed_mode && std::string(reduced_seed_mode)=="critical-seeds"));
    const bool uniform_source = settings.limb_darkening_c == 0.0 && settings.limb_darkening_d == 0.0;
    if (!uniform_source && finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }
    const double total_source_flux = source_flux(source_radius, settings);
    if (!std::isfinite(total_source_flux)) {
        return std::nan("");
    }

    PolarVisitedCellIntervals visited(phi_bins);
    struct PolarFrontier {
        int left = 0;
        int right = -1;
        int iphi = 0;
    };
    std::deque<PolarFrontier> queue;
    const double source_radius2 = source_radius * source_radius;
    const double inv_source_radius2 = 1.0 / source_radius2;
    // Frontier columns use canonical phi indices.  A frontier carries a whole
    // adjacent radial interval instead of one queue entry per inside cell.
    // The receiving column still tests every unvisited cell in that interval,
    // so disconnected runs and topology remain fully discoverable while the
    // common long-arc case avoids a queue push followed by a visited discard
    // for almost every cell.
    auto cell_visited_normalized = [&](int ir, int iphi) {
        if (ir < 0) {
            return true;
        }
        const auto* intervals = visited.find(iphi);
        return intervals != nullptr && intervals->contains(ir);
    };
    auto add_visited_run = [&](int iphi, int left, int right) {
        if (right < left) {
            return;
        }
        visited.get_or_create(iphi).add(left, right);
    };
    auto enqueue_run_normalized = [&](int left, int right, int iphi) {
        left = std::max(left, 0);
        const auto* intervals = visited.find(iphi);
        if (intervals != nullptr) {
            const std::int64_t first_unvisited =
                intervals->first_unvisited_at_or_after(left);
            if (first_unvisited > right) {
                return;
            }
            left = static_cast<int>(first_unvisited);
        }
        if (right >= left) {
            queue.push_back({left, right, iphi});
        }
    };
    // Cells in one radial run share the same phi column, so the column unit
    // vector is computed once per run instead of per cell (sincos dominated
    // the high-magnification profile otherwise).
    double column_cos = 1.0;
    double column_sin = 0.0;
    auto set_column = [&](int iphi) {
        visited.direction(iphi, dphi, column_cos, column_sin);
    };
    auto cell_inside = [&](int ir, double* dz2_out = nullptr) {
        if (ir < 0) {
            return false;
        }
        const double radius = (static_cast<double>(ir) + 0.5) * dr;
        if (fill_stats) ++map_queries;
        const SourcePosition mapped = map_lens_real(
            mapper, radius * column_cos, radius * column_sin);
        const double dz2 = distance_squared(mapped, source);
        if (dz2_out != nullptr) {
            *dz2_out = dz2;
        }
        return dz2 <= source_radius2;
    };

    // Second-order radial boundary correction: use the mapped distances at the
    // last inside cell and its outside neighbour for an initial source-edge
    // crossing, then take one bracketed secant correction for lens-map
    // curvature.
    // Midpoint counting covered the run out to the cell face, so the residual
    // strip is (t - 0.5) cells with area ~ dr * edge_radius * dphi.  t is
    // confined to [0, 1] by r_in <= rho < r_out, bounding the correction even
    // near folds.  High-magnification arcs are radially thin, so the radial
    // run ends dominate the boundary; the short azimuthal caps are left at
    // midpoint accuracy.
    const double edge_brightness = uniform_source ? 1.0 :
        (finite_magnifier != nullptr ?
            finite_magnifier->limb_darkening_table_brightness(1.0) :
            source_surface_brightness(1.0, settings));
    const double* brightness_table = !uniform_source && finite_magnifier != nullptr
        ? finite_magnifier->limb_darkening_table_data()
        : nullptr;
    const auto brightness_at = [&](double dz2) {
        if (uniform_source) {
            return 1.0;
        }
        const double normalized_radius2 = dz2 * inv_source_radius2;
        if (brightness_table != nullptr && settings.limb_darkening_d != 0.0) {
            const int index = static_cast<int>(
                normalized_radius2 *
                static_cast<double>(kLimbDarkeningTableSize) + 0.5);
            return brightness_table[std::clamp(
                index, 0, kLimbDarkeningTableSize)];
        }
        return source_surface_brightness(normalized_radius2, settings);
    };
    auto radial_edge_correction = [&](
        double inside_dz2,
        double outside_dz2,
        double inside_radius,
        double outside_radius,
        double face_radius,
        bool has_second_node,
        double second_radius,
        double second_dz2,
        double* absolute_geometric_correction) {
        const double r_in = std::sqrt(inside_dz2);
        const double r_out = std::sqrt(outside_dz2);
        const double dr_mapped = r_out - r_in;
        const double initial_delta = dr_mapped > 0.0
            ? std::clamp((source_radius - r_in) / dr_mapped, 0.0, 1.0)
            : 0.5;
        const double delta = refine_polar_boundary_crossing_fraction(
            mapper,
            source,
            source_radius2,
            {inside_radius * column_cos, inside_radius * column_sin},
            {outside_radius * column_cos, outside_radius * column_sin},
            inside_dz2 - source_radius2,
            outside_dz2 - source_radius2,
            initial_delta,
            dr);
        const double geometric_correction = (delta - 0.5) * face_radius;
        *absolute_geometric_correction += std::abs(geometric_correction);
        const double node_value = brightness_at(inside_dz2) * inside_radius;
        const double limb_radius = inside_radius +
            delta * (outside_radius - inside_radius);
        const double limb_value = edge_brightness * limb_radius;
        if (boundary_diagnostics_enabled()) {
            auto& counters = probe_counters();
            if (has_second_node && settings.limb_darkening_d == 0.0) {
                const auto model = second_radius > 0.0 && second_dz2 <= source_radius2
                    ? detail::bennett_boundary_model_residual(
                        delta, node_value, brightness_at(second_dz2)*second_radius,
                        limb_value, settings.bennett_delta_c)
                    : detail::BennettBoundaryModelResidual {};
                if (model.resolved) {
                    ++counters.boundary_model_edges;
                    const double flux_residual = model.signed_residual*dr*dphi;
                    counters.boundary_model_signed_flux_residual += flux_residual;
                    counters.boundary_model_absolute_flux_residual += std::abs(flux_residual);
                } else ++counters.boundary_model_unresolved;
            } else ++counters.boundary_model_unresolved;
        }
        return detail::bennett_boundary_residual(
            delta, node_value, limb_value, settings.bennett_delta_c);
    };

    // Keep the radial integral for every angular column.  Bennett's equation
    // (15) is an outer-coordinate rule, so it must replace endpoint stencils
    // rather than be collapsed into a single global midpoint sum.
    std::vector<double> column_integrals(static_cast<std::size_t>(phi_bins), 0.0);
    double total_count = 0.0;
    double absolute_radial_geometry = 0.0;
    // Expansion already maps every inside cell in a radial run to locate its
    // ends.  Retain those distances for the brightness pass; remapping the
    // same run used to double the dominant lens-equation work.  The two
    // buffers are reused across runs and preserve the original left-to-right
    // accumulation order.
    std::vector<double> left_inside_distances;
    std::vector<double> right_inside_distances;
    left_inside_distances.reserve(static_cast<std::size_t>(source_bins));
    right_inside_distances.reserve(static_cast<std::size_t>(source_bins));
    struct PolarIntegratedRun {
        int iphi = 0;
        int left = 0;
        int right = -1;
        double radial_integral = 0.0;
    };
    std::vector<PolarIntegratedRun> integrated_runs;
    const bool bennett_outer_complex_diagnostic =
        std::getenv("LCBININT_ENABLE_BENNETT_OUTER_SEGMENTS") != nullptr;
    const bool bennett_outer_enabled =
        std::getenv("LCBININT_DISABLE_BENNETT_OUTER") == nullptr &&
        (std::getenv("LCBININT_ENABLE_BENNETT_OUTER") != nullptr ||
         bennett_outer_complex_diagnostic ||
         image_positions.size() <= 3U);
    const bool retain_polar_runs=bennett_outer_enabled || std::getenv("LCBININT_GRID_REPAIR_TRACE");
    if (retain_polar_runs) {
        integrated_runs.reserve(image_positions.size() * 4U);
    }
    const auto drain_frontier=[&]() {
    while (!queue.empty()) {
        const auto frontier = queue.front();
        queue.pop_front();
        const int iphi = frontier.iphi;
        set_column(iphi);
        std::int64_t frontier_ir = frontier.left;
        while (frontier_ir <= frontier.right) {
            const auto* intervals = visited.find(iphi);
            if (intervals != nullptr) {
                frontier_ir = intervals->first_unvisited_at_or_after(frontier_ir);
            }
            if (frontier_ir > frontier.right) {
                break;
            }
            double start_dz2 = 0.0;
            if (!cell_inside(static_cast<int>(frontier_ir), &start_dz2)) {
                ++frontier_ir;
                continue;
            }
            const int ir = static_cast<int>(frontier_ir);

            left_inside_distances.clear();
            int left = ir;
            double left_outside_dz2 = -1.0;
            while (left > 0 && !cell_visited_normalized(left - 1, iphi)) {
                double neighbor_dz2 = 0.0;
                if (!cell_inside(left - 1, &neighbor_dz2)) {
                    left_outside_dz2 = neighbor_dz2;
                    break;
                }
                --left;
                left_inside_distances.push_back(neighbor_dz2);
            }
            right_inside_distances.clear();
            int right = ir;
            double right_outside_dz2 = -1.0;
            while (!cell_visited_normalized(right + 1, iphi)) {
                double neighbor_dz2 = 0.0;
                if (!cell_inside(right + 1, &neighbor_dz2)) {
                    right_outside_dz2 = neighbor_dz2;
                    break;
                }
                ++right;
                right_inside_distances.push_back(neighbor_dz2);
            }

            add_visited_run(iphi, left, right);
            if (fill_stats) fill_cells+=right-left+1;
            if (seed_trace) {
                const auto square_prefix=[](std::uint64_t n) {
                    std::uint64_t a=n,b=n+1,c=2*n+1;
                    if (a%2==0) a/=2; else b/=2;
                    if (a%3==0) a/=3; else if (b%3==0) b/=3; else c/=3;
                    return a*b*c; // sum of squares modulo 2^64
                };
                const auto n=static_cast<std::uint64_t>(right-left+1);
                const auto sum=n*(static_cast<std::uint64_t>(left)+static_cast<std::uint64_t>(right))/2;
                const auto square_sum=square_prefix(right)-(left>0 ? square_prefix(left-1) : 0);
                const auto offset=static_cast<std::uint64_t>(iphi)*UINT64_C(0x9e3779b97f4a7c15);
                trace_cells+=n;
                trace_signature+=square_sum+2*offset*sum+n*offset*offset;
            }

            const double left_inside_dz2 = left_inside_distances.empty()
                ? start_dz2
                : left_inside_distances.back();
            const double right_inside_dz2 = right_inside_distances.empty()
                ? start_dz2
                : right_inside_distances.back();
            double radial_integral = 0.0;
            const auto accumulate_cell = [&](int current, double dz2) {
                const double radius = (static_cast<double>(current) + 0.5) * dr;
                radial_integral += brightness_at(dz2) * radius;
            };
            int current = left;
            for (auto it = left_inside_distances.rbegin();
                 it != left_inside_distances.rend(); ++it, ++current) {
                accumulate_cell(current, *it);
            }
            accumulate_cell(ir, start_dz2);
            current = ir + 1;
            for (const double dz2 : right_inside_distances) {
                accumulate_cell(current++, dz2);
            }
            const int previous_phi = iphi == 0 ? phi_bins - 1 : iphi - 1;
            const int next_phi = iphi + 1 == phi_bins ? 0 : iphi + 1;
            // Polar run components use 8-neighbour adjacency. Include the
            // diagonal endpoints too; equal radial spans alone implement only
            // 4-neighbour reachability and require redundant endpoint seeds.
            const int frontier_left=diagonal_frontier && left>0 ? left-1 : left;
            const int frontier_right=diagonal_frontier && right<std::numeric_limits<int>::max() ? right+1 : right;
            enqueue_run_normalized(frontier_left, frontier_right, previous_phi);
            enqueue_run_normalized(frontier_left, frontier_right, next_phi);
            if (diagonal_frontier && fill_stats) ++probe_counters().polar_diagonal_runs;
            // Boundary diagnostics use the samples retained by the expansion;
            // no additional ray is needed to fit the nearest two inside nodes.
            const auto cached_distance = [&](int radial_index) {
                if (radial_index == ir) return start_dz2;
                if (radial_index < ir) return left_inside_distances[
                    static_cast<std::size_t>(ir-radial_index-1)];
                return right_inside_distances[static_cast<std::size_t>(radial_index-ir-1)];
            };
            if (left_outside_dz2 >= 0.0) {
                const double correction = radial_edge_correction(
                    left_inside_dz2,
                    left_outside_dz2,
                    (static_cast<double>(left) + 0.5) * dr,
                    (static_cast<double>(left) - 0.5) * dr,
                    static_cast<double>(left) * dr,
                    right > left,
                    (static_cast<double>(left)+1.5)*dr,
                    right > left && boundary_diagnostics_enabled() ? cached_distance(left+1) : -1.0,
                    &absolute_radial_geometry);
                radial_integral += correction;
            }
            if (right_outside_dz2 >= 0.0) {
                const double correction = radial_edge_correction(
                    right_inside_dz2,
                    right_outside_dz2,
                    (static_cast<double>(right) + 0.5) * dr,
                    (static_cast<double>(right) + 1.5) * dr,
                    static_cast<double>(right + 1) * dr,
                    right > left,
                    (static_cast<double>(right)-0.5)*dr,
                    right > left && boundary_diagnostics_enabled() ? cached_distance(right-1) : -1.0,
                    &absolute_radial_geometry);
                radial_integral += correction;
            }
            column_integrals[static_cast<std::size_t>(iphi)] += radial_integral;
            if (retain_polar_runs) {
                integrated_runs.push_back({iphi, left, right, radial_integral});
            }
            frontier_ir = std::max(
                frontier_ir + 1,
                static_cast<std::int64_t>(right) + 1);
        }
    }

    };
    // Process seeds sequentially using the same visited run registry.
    for (const auto& image_position : image_positions) {
        double grid_radius = 0.0;
        double image_phi = 0.0;
        if (!find_polar_inside_start(
                mapper, source, source_radius, image_position, dr, dphi, phi_bins,
                &grid_radius, &image_phi)) {
            continue;
        }
        const int ir = std::max(0, static_cast<int>(std::floor(grid_radius / dr)));
        const int iphi = std::clamp(static_cast<int>(image_phi / dphi), 0, phi_bins - 1);
        if (cell_visited_normalized(ir, iphi)) continue;
        enqueue_run_normalized(ir, ir, iphi);
        drain_frontier();
    }

    if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
        if ((std::getenv("LCBININT_POLAR_COVERAGE_CERTIFICATE") || std::getenv("LCBININT_SEED_LATTICE_CERTIFICATE")) && phi_bins<=1048576 &&
            std::numeric_limits<double>::is_iec559 && std::fegetround()==FE_TONEAREST) {
            using I=math::OutwardInterval;
            const auto geometry=enclosed_binary_geometry(mapper);
            auto& counters=probe_counters();++counters.polar_coverage_calls;
            const auto proof_started=std::chrono::steady_clock::now();
            int tree_size=1;while(tree_size<phi_bins) tree_size*=2;
            using Direction=std::array<double,4>;
            const double inf=std::numeric_limits<double>::infinity();
            std::vector<Direction> tree(2*static_cast<std::size_t>(tree_size),{inf,-inf,inf,-inf});
            double minimum_norm=1;
            for (int p=0;p<phi_bins;++p) {
                double c,t;visited.direction(p,dphi,c,t);tree[tree_size+p]={c,c,t,t};
                const I norm=math::square(I(c))+math::square(I(t));
                minimum_norm=std::min(minimum_norm,math::down(std::sqrt(norm.lo)));
            }
            for (int p=tree_size-1;p>0;--p) {
                const auto& a=tree[2*p];const auto& b=tree[2*p+1];
                tree[p]={std::min(a[0],b[0]),std::max(a[1],b[1]),std::min(a[2],b[2]),std::max(a[3],b[3])};
            }
            const auto directions=[&](int lo,int hi) {
                Direction out{inf,-inf,inf,-inf};
                const auto merge=[&](const Direction& a) {
                    out[0]=std::min(out[0],a[0]);out[1]=std::max(out[1],a[1]);
                    out[2]=std::min(out[2],a[2]);out[3]=std::max(out[3],a[3]);
                };
                for (lo+=tree_size,hi+=tree_size+1;lo<hi;lo/=2,hi/=2) {
                    if (lo%2) merge(tree[lo++]);if (hi%2) merge(tree[--hi]);
                }
                return out;
            };
            const I cx=I(source.x)+geometry.shift;
            const double source_norm=math::up(std::sqrt((math::square(cx)+math::square(I(source.y))).hi));
            const I bound=I(2)*(I(std::abs(mapper.separation.real()))+I(source_norm)+I(source_radius)+geometry.m1+geometry.m2+I(2));
            const double radial_limit=std::ceil((bound/(I(dr)*I(minimum_norm))).hi);
            long long boxes=0,unresolved=0,repairs=0;
            using Box=std::array<int,4>;
            std::vector<Box> pending;
            if (std::isfinite(radial_limit) && radial_limit>=0 && radial_limit<1e9 && minimum_norm>0)
                pending.push_back({0,static_cast<int>(radial_limit),0,phi_bins-1});
            else ++unresolved;
            while (!pending.empty()) {
                if (boxes>=65536) {unresolved+=pending.size();break;}
                const Box box=pending.back();pending.pop_back();++boxes;
                bool covered=true;
                for (int p=box[2];p<=box[3];++p) {
                    const auto* column=visited.find(p);
                    if (!column || column->first_unvisited_at_or_after(box[0])<=box[1]) {covered=false;break;}
                }
                if (covered) continue;
                const auto angle=directions(box[2],box[3]);
                const I r((static_cast<double>(box[0])+.5)*dr,(static_cast<double>(box[1])+.5)*dr);
                const I x=r*I(angle[0],angle[1]),y=r*I(angle[2],angle[3]);
                const I xa=x-I(mapper.separation.real()),y2=math::square(y);
                const I inv1=I(1)/(math::square(xa)+y2),inv2=I(1)/(math::square(x)+y2);
                const I fx=x-geometry.m1*xa*inv1-geometry.m2*x*inv2-geometry.shift-I(source.x);
                const I fy=y-geometry.m1*y*inv1-geometry.m2*y*inv2-I(source.y);
                const I d=math::square(fx)+math::square(fy);
                if (d.finite() && d.lo>math::square(I(source_radius)).hi) continue;
                if (box[0]==box[1] && box[2]==box[3]) {
                    set_column(box[2]);
                    if (cell_inside(box[0])) {
                        if (std::getenv("LCBININT_GRID_REPAIR_TRACE")) {
                            std::fprintf(stderr,"POLAR_GRID_REPAIR r=%d phi=%d phi_bins=%d dr=%.17g\n",box[0],box[2],phi_bins,dr);
                            for (const auto& run:integrated_runs) {
                                const int delta=std::abs(run.iphi-box[2]);
                                if (delta<=2 || delta>=phi_bins-2)
                                    std::fprintf(stderr,"POLAR_GRID_NEIGHBOUR phi=%d left=%d right=%d\n",run.iphi,run.left,run.right);
                            }
                        }
                        ++repairs;enqueue_run_normalized(box[0],box[0],box[2]);drain_frontier();
                    } else ++unresolved;
                    continue;
                }
                if (box[1]-box[0]>=box[3]-box[2]) {
                    const int m=box[0]+(box[1]-box[0])/2;
                    pending.push_back({box[0],m,box[2],box[3]});pending.push_back({m+1,box[1],box[2],box[3]});
                } else {
                    const int m=box[2]+(box[3]-box[2])/2;
                    pending.push_back({box[0],box[1],box[2],m});pending.push_back({box[0],box[1],m+1,box[3]});
                }
            }
            counters.polar_coverage_boxes+=boxes;counters.polar_coverage_unresolved+=unresolved;
            counters.polar_coverage_repairs+=repairs;counters.polar_coverage_certified+=unresolved==0;
            counters.polar_coverage_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-proof_started).count();
            if (diagnostics) diagnostics->seed_coverage_proven=unresolved==0;
        } else if (std::getenv("LCBININT_POLAR_COVERAGE_CERTIFICATE") || std::getenv("LCBININT_SEED_LATTICE_CERTIFICATE")) {
            if(diagnostics) diagnostics->seed_coverage_proven=false;
            ++probe_counters().polar_coverage_calls;++probe_counters().polar_coverage_unresolved;
        }
    }

    const double trace_radial_sum=seed_trace
        ? std::accumulate(column_integrals.begin(),column_integrals.end(),0.0) : 0.0;
    if (bennett_outer_enabled) {
    // Reconstruct connected image components from the radial runs.  The
    // visited table is deliberately a union of all flood fills, so its
    // columns alone cannot tell us whether two separated image arcs happen to
    // overlap in phi.  Equation (15) is a one-dimensional rule for one
    // component's outer boundary; applying it to the union of all columns is
    // wrong in precisely that situation.
    const std::size_t run_count = integrated_runs.size();
    std::vector<std::size_t> run_parent(run_count);
    std::iota(run_parent.begin(), run_parent.end(), std::size_t {0});
    const auto find_run_root = [&](std::size_t value) {
        std::size_t root = value;
        while (run_parent[root] != root) {
            root = run_parent[root];
        }
        while (run_parent[value] != value) {
            const std::size_t next = run_parent[value];
            run_parent[value] = root;
            value = next;
        }
        return root;
    };
    const auto unite_run_roots = [&](std::size_t first, std::size_t second) {
        first = find_run_root(first);
        second = find_run_root(second);
        if (first == second) {
            return;
        }
        run_parent[second] = first;
    };

    std::vector<std::vector<std::size_t>> runs_by_phi(
        static_cast<std::size_t>(phi_bins));
    for (std::size_t index = 0; index < run_count; ++index) {
        runs_by_phi[static_cast<std::size_t>(integrated_runs[index].iphi)]
            .push_back(index);
    }
    const auto run_precedes = [&](std::size_t first, std::size_t second) {
        const auto& lhs = integrated_runs[first];
        const auto& rhs = integrated_runs[second];
        return lhs.left < rhs.left ||
            (lhs.left == rhs.left && lhs.right < rhs.right);
    };
    for (auto& runs : runs_by_phi) {
        std::sort(runs.begin(), runs.end(), run_precedes);
        for (std::size_t index = 1; index < runs.size(); ++index) {
            const auto& previous = integrated_runs[runs[index - 1]];
            const auto& current = integrated_runs[runs[index]];
            if (static_cast<std::int64_t>(current.left) <=
                    static_cast<std::int64_t>(previous.right) + 1) {
                unite_run_roots(runs[index - 1], runs[index]);
            }
        }
    }
    for (int iphi = 0; iphi < phi_bins; ++iphi) {
        const int next_iphi = iphi + 1 == phi_bins ? 0 : iphi + 1;
        const auto& first_column = runs_by_phi[static_cast<std::size_t>(iphi)];
        const auto& second_column = runs_by_phi[static_cast<std::size_t>(next_iphi)];
        std::size_t first_index = 0;
        std::size_t second_index = 0;
        while (first_index < first_column.size() &&
               second_index < second_column.size()) {
            const auto& first = integrated_runs[first_column[first_index]];
            const auto& second = integrated_runs[second_column[second_index]];
            const bool overlap_or_touch =
                static_cast<std::int64_t>(first.left) <=
                    static_cast<std::int64_t>(second.right) + 1 &&
                static_cast<std::int64_t>(second.left) <=
                    static_cast<std::int64_t>(first.right) + 1;
            if (overlap_or_touch) {
                unite_run_roots(first_column[first_index], second_column[second_index]);
            }
            if (first.right < second.right) {
                ++first_index;
            } else {
                ++second_index;
            }
        }
    }

    struct PolarOuterComponent {
        std::vector<std::size_t> run_indices;
    };
    std::unordered_map<std::size_t, std::size_t> component_lookup;
    std::vector<PolarOuterComponent> outer_components;
    for (std::size_t index = 0; index < run_count; ++index) {
        const std::size_t root = find_run_root(index);
        const auto found = component_lookup.find(root);
        std::size_t component_id = 0;
        if (found == component_lookup.end()) {
            component_id = outer_components.size();
            component_lookup.emplace(root, component_id);
            outer_components.emplace_back();
        } else {
            component_id = found->second;
        }
        outer_components[component_id].run_indices.push_back(index);
    }

    // Apply Bennett's outer-coordinate construction (equation 15 in its
    // smooth p=0 limit) to one-to-one angular image segments.  The radial pass
    // above supplies F(phi), the radial integral for each angular run.  The
    // segment walker is the polar analogue of Dave's check_bktrack: a split
    // or merge ends one segment and starts another, so an endpoint rule is
    // never silently continued across a radial backtrack.
    struct PolarOuterBoundaryEstimate {
        double minimum = 1.0;
        double maximum = 0.0;
    };
    const auto angular_boundary_delta = [&](std::size_t run_index,
                                            int direction)
        -> std::optional<PolarOuterBoundaryEstimate> {
        if (run_index >= integrated_runs.size()) {
            return std::nullopt;
        }
        const auto& run = integrated_runs[run_index];
        const int inside_iphi = run.iphi;
        const double inside_phi =
            (static_cast<double>(inside_iphi) + 0.5) * dphi;
        const double outside_phi = inside_phi +
            static_cast<double>(direction) * dphi;
        const double inside_cos = std::cos(inside_phi);
        const double inside_sin = std::sin(inside_phi);
        const double outside_cos = std::cos(outside_phi);
        const double outside_sin = std::sin(outside_phi);
        double smallest_fraction = 1.0;
        double largest_fraction = 0.0;
        bool found = false;
        for (int ir = run.left; ir <= run.right; ++ir) {
            const double radius = (static_cast<double>(ir) + 0.5) * dr;
            const SourcePosition inside_position {
                radius * inside_cos, radius * inside_sin};
            const SourcePosition outside_position {
                radius * outside_cos, radius * outside_sin};
            const double inside_residual = boundary_root_residual(
                mapper, inside_position.x, inside_position.y,
                source, source_radius2);
            const double outside_residual = boundary_root_residual(
                mapper, outside_position.x, outside_position.y,
                source, source_radius2);
            // This is Dave's hmax/hmin criterion: only radial samples that
            // are inside on the endpoint column and cross the source limb in
            // the neighbouring column contribute.  Restricting the scan to
            // this run is important for a segment adjacent to a split or
            // merge; scanning every run in the column would mix branches.
            if (!(std::isfinite(inside_residual) &&
                  std::isfinite(outside_residual) &&
                  inside_residual <= 0.0 && outside_residual > 0.0)) {
                continue;
            }
            const double denominator = outside_residual - inside_residual;
            const double initial = denominator > 0.0
                ? std::clamp(-inside_residual / denominator, 0.0, 1.0)
                : 0.5;
            const double fraction = refine_angular_boundary_crossing_fraction(
                mapper, source, source_radius2,
                inside_position, outside_position,
                inside_residual, outside_residual,
                initial, dphi);
            smallest_fraction = std::min(smallest_fraction, fraction);
            largest_fraction = std::max(largest_fraction, fraction);
            found = true;
        }
        return found
            ? std::optional<PolarOuterBoundaryEstimate>(
                PolarOuterBoundaryEstimate {
                    smallest_fraction,
                    largest_fraction,
                })
            : std::nullopt;
    };

    int outer_polar_arcs = 0;
    int outer_polar_segments_considered = 0;
    int outer_polar_boundary_candidates = 0;
    int outer_polar_applied = 0;
    int outer_polar_components_considered = 0;
    int outer_polar_components_rejected_span = 0;
    int outer_polar_components_rejected_topology = 0;
    // The automatic route is limited to the clean two-image regime.  The
    // segment switch allows A/B experiments on more complicated maps, but it
    // only admits segments attached to a detected junction; clean multi-image
    // arcs remain fail-closed because their endpoint topology is not the
    // problem that check_bktrack is designed to solve.
    for (std::size_t component_id = 0;
         component_id < outer_components.size(); ++component_id) {
            ++outer_polar_components_considered;
            const auto& component = outer_components[component_id];
            if (!bennett_outer_complex_diagnostic &&
                (outer_components.size() > 2U || image_positions.size() > 8U)) {
                ++outer_polar_components_rejected_span;
                continue;
            }

            std::vector<detail::PolarRunInterval> component_runs;
            component_runs.reserve(component.run_indices.size());
            for (const std::size_t run_index : component.run_indices) {
                const auto& run = integrated_runs[run_index];
                component_runs.push_back({
                    run.iphi, run.left, run.right, run_index});
            }
            const auto walk = detail::walk_polar_outer_segments(
                component_runs, phi_bins);
            if (!walk.complete || walk.segments.empty()) {
                ++outer_polar_components_rejected_topology;
                continue;
            }
            // Preserve the production polar route's clean-arc contract.  A
            // component with a split or merge is exactly the case for which
            // the new check_bktrack-style decomposition is experimental; it
            // must not become active merely because the component happens to
            // contain only two point-image seeds.
            if (walk.junction_nodes > 0U &&
                !bennett_outer_complex_diagnostic) {
                ++outer_polar_components_rejected_topology;
                continue;
            }
            if (bennett_outer_complex_diagnostic &&
                std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr &&
                walk.junction_nodes > 0U) {
                std::fprintf(
                    stderr,
                    "BENNETT_OUTER_POLAR_WALK component=%zu segments=%zu edges=%zu junctions=%zu complete=%d\n",
                    component_id, walk.segments.size(), walk.adjacency_edges,
                    walk.junction_nodes, walk.complete ? 1 : 0);
            }

            for (std::size_t segment_id = 0;
                 segment_id < walk.segments.size(); ++segment_id) {
                const auto& segment = walk.segments[segment_id];
                ++outer_polar_segments_considered;
                if (segment.closed || segment.run_indices.size() < 6U) {
                    ++outer_polar_components_rejected_span;
                    continue;
                }
                if ((outer_components.size() > 2U || image_positions.size() > 8U) &&
                    !bennett_outer_complex_diagnostic) {
                    ++outer_polar_components_rejected_span;
                    continue;
                }
                if ((outer_components.size() > 2U || image_positions.size() > 8U) &&
                    !segment.lower_junction && !segment.upper_junction) {
                    ++outer_polar_components_rejected_topology;
                    continue;
                }

                double radial_width_sum = 0.0;
                bool valid_widths = true;
                for (const std::size_t run_index : segment.run_indices) {
                    const auto& run = integrated_runs[run_index];
                    const std::int64_t width =
                        static_cast<std::int64_t>(run.right) -
                        static_cast<std::int64_t>(run.left) + 1;
                    if (width <= 0) {
                        valid_widths = false;
                        break;
                    }
                    radial_width_sum += static_cast<double>(width);
                }
                if (!valid_widths) {
                    ++outer_polar_components_rejected_span;
                    continue;
                }
                const double mean_radial_width = radial_width_sum /
                    static_cast<double>(segment.run_indices.size());
                const double image_aspect = mean_radial_width > 0.0
                    ? static_cast<double>(segment.run_indices.size()) /
                        mean_radial_width
                    : 0.0;
                if (image_aspect < 8.0) {
                    ++outer_polar_components_rejected_span;
                    continue;
                }
                ++outer_polar_arcs;

                std::optional<PolarOuterBoundaryEstimate> lower_delta;
                std::optional<PolarOuterBoundaryEstimate> upper_delta;
                const std::size_t first = segment.run_indices.front();
                const std::size_t second = segment.run_indices[1];
                const std::size_t last = segment.run_indices.back();
                const std::size_t penultimate =
                    segment.run_indices[segment.run_indices.size() - 2];
                if (segment.lower_boundary) {
                    lower_delta = angular_boundary_delta(first, -1);
                }
                if (segment.upper_boundary) {
                    upper_delta = angular_boundary_delta(last, +1);
                }
                if (!lower_delta.has_value() && !upper_delta.has_value()) {
                    continue;
                }
                ++outer_polar_boundary_candidates;

                // Bennett's Eq. (15) is the two-point p=0 rule used by
                // eesunhong.  It is applied separately at each endpoint;
                // each run integral already contains the midpoint baseline.
                if (diagnostics != nullptr &&
                    std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr) {
                    std::fprintf(
                        stderr,
                        "BENNETT_OUTER_POLAR_APPLY component=%zu segment=%zu length=%zu aspect=%.6g junction=%d/%d lower=[%.9g,%.9g] upper=[%.9g,%.9g]\n",
                        component_id, segment_id, segment.run_indices.size(),
                        image_aspect,
                        segment.lower_junction ? 1 : 0,
                        segment.upper_junction ? 1 : 0,
                        lower_delta.has_value() ? lower_delta->minimum : -1.0,
                        lower_delta.has_value() ? lower_delta->maximum : -1.0,
                        upper_delta.has_value() ? upper_delta->minimum : -1.0,
                        upper_delta.has_value() ? upper_delta->maximum : -1.0);
                }
                if (lower_delta.has_value()) {
                    const auto weights =
                        detail::bennett_outer_boundary_weights(
                            lower_delta->maximum);
                    column_integrals[static_cast<std::size_t>(
                        integrated_runs[first].iphi)] +=
                        (weights.first - 1.0) *
                            integrated_runs[first].radial_integral +
                        (weights.second - 1.0) *
                            integrated_runs[second].radial_integral;
                }
                if (upper_delta.has_value()) {
                    const auto weights =
                        detail::bennett_outer_boundary_weights(
                            upper_delta->maximum);
                    column_integrals[static_cast<std::size_t>(
                        integrated_runs[last].iphi)] +=
                        (weights.first - 1.0) *
                            integrated_runs[last].radial_integral +
                        (weights.second - 1.0) *
                            integrated_runs[penultimate].radial_integral;
                }
                ++outer_polar_applied;
            }
        }

        if (diagnostics != nullptr &&
            std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr) {
            std::fprintf(
                stderr,
                "BENNETT_OUTER_POLAR phi_bins=%d components=%zu considered=%d segments=%d rejected_span=%d topology_reject=%d arcs=%d candidates=%d applied=%d\n",
                phi_bins, outer_components.size(), outer_polar_components_considered,
                outer_polar_segments_considered,
                outer_polar_components_rejected_span,
                outer_polar_components_rejected_topology, outer_polar_arcs,
                outer_polar_boundary_candidates, outer_polar_applied);
        }
    }

    total_count = std::accumulate(column_integrals.begin(), column_integrals.end(), 0.0);
    const double image_flux = total_count * dr * dphi;
    if (seed_trace) {
        std::fprintf(stderr,
            "POLAR_SEED_TRACE source_x=%.17g source_y=%.17g rho=%.17g nbin=%d seeds=%zu radius=%.17g phi_bins=%d cells=%llu signature=%llu outer_enabled=%d radial_A=%.17g total_A=%.17g\n",
            source.x,source.y,source_radius,source_bins,image_positions.size(),max_image_radius,phi_bins,
            static_cast<unsigned long long>(trace_cells),static_cast<unsigned long long>(trace_signature),
            bennett_outer_enabled,trace_radial_sum*dr*dphi/total_source_flux,image_flux/total_source_flux);
    }
    if (diagnostics != nullptr) {
        // The boundary correction removes the O(h) radial boundary term.  Its
        // accumulated absolute geometric magnitude therefore supplies the
        // local scale for the remaining O(h^2) term; multiplying by
        // h/rho = 1/N gives an embedded estimate without another lens-map
        // pass.  Keeping this scale geometric prevents a dark source limb
        // from collapsing the error estimate to zero.
        diagnostics->estimated_error =
            absolute_radial_geometry * dr * dphi /
            (total_source_flux * static_cast<double>(source_bins));
    }
    if (fill_stats) {
        auto& c=probe_counters();++c.polar_fill_calls;c.polar_map_queries+=map_queries;
        c.polar_fill_cells+=fill_cells;c.polar_seed_positions+=image_positions.size();
        c.polar_fill_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-fill_started).count();
    }
    return image_flux / total_source_flux;
}

double inverse_ray_polar_boundary_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<SourcePosition>* seed_positions = nullptr,
    PolarAreaDiagnostics* diagnostics = nullptr)
{
    std::vector<SourcePosition> image_positions;
    if (seed_positions != nullptr) {
        image_positions = *seed_positions;
    } else {
        const auto images = point_magnifier.binary_images(separation, mass_ratio, source);
        image_positions.reserve(images.size());
        for (const auto& image : images) {
            image_positions.push_back(image.position);
        }
    }
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    return inverse_ray_polar_core(
        mapper, image_positions, source, source_radius, settings, finite_magnifier,
        diagnostics);
}

struct PolarToleranceEvaluation {
    double magnification = std::numeric_limits<double>::quiet_NaN();
    double error_estimate = std::numeric_limits<double>::infinity();
    int refinement_level = 0;
    bool converged = false;
    bool seed_coverage_proven = true;
};

template <typename Evaluator>
PolarToleranceEvaluation evaluate_polar_to_tolerance(
    const FiniteSourceSettings& settings,
    bool support_proven,
    Evaluator&& evaluator,
    bool allow_automatic_retry = true)
{
    FiniteSourceSettings active = settings;
    PolarAreaDiagnostics fine_diagnostics;
    double fine = evaluator(active, &fine_diagnostics);
    if (!std::isfinite(fine)) {
        return {fine, std::numeric_limits<double>::infinity(), 0, false};
    }
    if (!support_proven || !fine_diagnostics.seed_coverage_proven) {
        return {fine, std::numeric_limits<double>::infinity(), 0, false, fine_diagnostics.seed_coverage_proven};
    }

    // Polar diagnostics use the embedded boundary-error estimate from the fine
    // pass. Callers that still own an adaptive ladder may opt into retries;
    // calibrated binary auto deliberately opts out. Fixed calls without a
    // tolerance retain their historical one-shot behaviour, while fixed calls
    // with an explicit tolerance keep the nested-grid compatibility check.
    if (!settings.automatic_source_bins &&
        !has_explicit_finite_source_tolerance(settings)) {
        return {fine, 0.0, 0, true};
    }

    int fine_bins = active_polar_source_bins(active);
    double error_estimate = fine_diagnostics.estimated_error;
    if (!settings.automatic_source_bins &&
        has_explicit_finite_source_tolerance(settings)) {
        FiniteSourceSettings coarse_settings = active;
        const int coarse_bins = std::max(1, fine_bins / 2);
        coarse_settings.source_bins = coarse_bins;
        coarse_settings.polar_source_bins = coarse_bins;
        PolarAreaDiagnostics coarse_diagnostics;
        const double coarse = evaluator(coarse_settings, &coarse_diagnostics);
        error_estimate = std::isfinite(coarse) && coarse_diagnostics.seed_coverage_proven
            ? std::max(
                error_estimate,
                grid_pair_error_estimate(fine, coarse, fine_bins, coarse_bins, 2))
            : std::numeric_limits<double>::infinity();
    }
    auto assessment = assess_finite_source_error(
        settings, fine, error_estimate, support_proven);
    const int maximum_bins = std::max(settings.max_source_bins, 1);
    int refinement_level = 0;
    while (allow_automatic_retry && settings.automatic_source_bins &&
           !assessment.converged &&
           fine_bins < maximum_bins) {
        const int retry_bins = next_finite_source_resolution(
            fine_bins, maximum_bins, assessment.shortfall, 2);
        if (retry_bins <= fine_bins) {
            break;
        }

        active.source_bins = retry_bins;
        active.polar_source_bins = retry_bins;
        PolarAreaDiagnostics retry_diagnostics;
        const double retry = evaluator(active, &retry_diagnostics);
        if (!std::isfinite(retry)) {
            break;
        }
        fine = retry;
        fine_bins = retry_bins;
        if (!retry_diagnostics.seed_coverage_proven) {
            return {fine, std::numeric_limits<double>::infinity(), refinement_level+1, false, false};
        }
        error_estimate = retry_diagnostics.estimated_error;
        ++refinement_level;
        assessment = assess_finite_source_error(
            settings, fine, error_estimate, support_proven);
    }
    return {
        fine,
        assessment.estimate,
        refinement_level,
        assessment.converged,
    };
}

template <typename ImageMap>
bool find_polar_inside_start(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    SourcePosition image_seed,
    double dr,
    double dphi,
    int phi_bins,
    double* start_radius,
    double* start_phi)
{
    const double seed_radius = std::hypot(image_seed.x, image_seed.y);
    const double seed_phi = wrap_angle(std::atan2(image_seed.y, image_seed.x));
    const int seed_ir = static_cast<int>(std::floor(seed_radius / dr));
    const int seed_iphi = std::clamp(static_cast<int>(seed_phi / dphi), 0, phi_bins - 1);
    const double source_radius2 = source_radius * source_radius;
    constexpr int max_shell = 10;
    for (int shell = 0; shell <= max_shell; ++shell) {
        for (int dir = -1; dir <= 1; dir += 2) {
            for (int d_ir = -shell; d_ir <= shell; ++d_ir) {
                const int d_iphi = (shell - std::abs(d_ir)) * dir;
                const int ir = seed_ir + d_ir;
                if (ir < 0) {
                    continue;
                }
                int iphi = seed_iphi + d_iphi;
                iphi %= phi_bins;
                if (iphi < 0) {
                    iphi += phi_bins;
                }
                const double radius = (static_cast<double>(ir) + 0.5) * dr;
                const double phi = (static_cast<double>(iphi) + 0.5) * dphi;
                const SourcePosition image {radius * std::cos(phi), radius * std::sin(phi)};
                const SourcePosition mapped = map_lens_real(mapper, image.x, image.y);
                if (distance_squared(mapped, source) <= source_radius2) {
                    *start_radius = radius;
                    *start_phi = phi;
                    return true;
                }
            }
        }
    }
    return false;
}

double source_limb_brightness(
    double normalized_radius2,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier)
{
    if (settings.limb_darkening_c == 0.0 && settings.limb_darkening_d == 0.0) {
        return 1.0;
    }
    return finite_magnifier != nullptr ?
        finite_magnifier->limb_darkening_table_brightness(normalized_radius2) :
        source_surface_brightness(normalized_radius2, settings);
}

std::vector<SourcePosition> selected_point_images(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source);

bool append_valid_seed(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    SourcePosition image,
    std::vector<SourcePosition>& seeds)
{
    if (source_radius <= 0.0) {
        return false;
    }
    const double source_radius2 = source_radius * source_radius;
    const SourcePosition mapped = map_binary_lens_real(mapper, image.x, image.y);
    if (distance_squared(mapped, source) > source_radius2 * (1.0 + 1.0e-8)) {
        return false;
    }
    seeds.push_back(image);
    return true;
}

// Seed-set half of a probe, split out so a caller that already rooted the
// probe (the certified stage does, to read off the image count) does not have
// to solve the lens equation a second time.
void append_probe_images_as_seeds(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const std::vector<SourcePosition>& probe_images,
    std::vector<SourcePosition>& seeds)
{
    const double source_radius2 = source_radius * source_radius;
    // Preserve a true union.  Proximity deduplication used to replace an
    // earlier representative when a later probe mapped closer to the source
    // centre.  At finite grid spacing those representatives need not reach the
    // same connected run, so adding probes could remove coverage.  Exact cell
    // deduplication happens after snapping to the integration lattice.
    for (const auto& img : probe_images) {
        const SourcePosition mapped = map_binary_lens_real(mapper, img.x, img.y);
        const double mapped_distance2 = distance_squared(mapped, source);
        if (mapped_distance2 > source_radius2 * (1.0 + 1.0e-8)) {
            continue;
        }
        seeds.push_back(img);
    }
}

void append_valid_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    SourcePosition probe_source,
    std::vector<SourcePosition>& seeds)
{
    const double source_radius2 = source_radius * source_radius;
    if (source_radius <= 0.0 ||
        distance_squared(probe_source, source) >= source_radius2 * (1.0 + 1.0e-10)) {
        return;
    }

    const auto probe_images =
        selected_point_images(point_magnifier, separation, mass_ratio, probe_source);
    if (probe_images.size() <= 3) {
        return;
    }
    append_probe_images_as_seeds(mapper, source, source_radius, probe_images, seeds);
}

void append_caustic_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    SourcePosition critical_source,
    std::vector<SourcePosition>& seeds)
{
    const double dx = critical_source.x - source.x;
    const double dy = critical_source.y - source.y;
    const double distance = std::hypot(dx, dy);
    if (distance <= 0.0 || distance >= source_radius) {
        return;
    }

    const double ux = dx / distance;
    const double uy = dy / distance;
    const double steps[] = {
        0.02 * source_radius,
        0.05 * source_radius,
        0.15 * source_radius,
        0.35 * source_radius,
    };
    for (const double step : steps) {
        const SourcePosition probes[2] = {
            {critical_source.x + ux * step, critical_source.y + uy * step},
            {critical_source.x - ux * step, critical_source.y - uy * step},
        };
        for (const auto& probe_source : probes) {
            append_valid_probe_image_seeds(
                point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                probe_source, seeds);
        }
    }

    if (seeds.size() <= 3) {
        constexpr int angular_probes = 16;
        const double radial_steps[] = {
            0.02 * source_radius,
            0.05 * source_radius,
            0.10 * source_radius,
            0.20 * source_radius,
            0.35 * source_radius,
        };
        for (const double step : radial_steps) {
            for (int i = 0; i < angular_probes; ++i) {
                const double theta =
                    2.0 * kPi * static_cast<double>(i) / static_cast<double>(angular_probes);
                const SourcePosition probe_source {
                    critical_source.x + step * std::cos(theta),
                    critical_source.y + step * std::sin(theta),
                };
                append_valid_probe_image_seeds(
                    point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                    probe_source, seeds);
                if (seeds.size() > 3) {
                    return;
                }
            }
        }
    }
}

void append_boundary_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    std::vector<SourcePosition>& seeds)
{
    constexpr std::size_t max_seeds = 128;
    if (source_radius <= 0.0 || seeds.size() >= max_seeds) {
        return;
    }
    constexpr int samples = 400;
    constexpr double inward_fraction = 0.02;
    const double probe_radius = source_radius * (1.0 - inward_fraction);
    for (int i = 0; i < samples && seeds.size() < max_seeds; ++i) {
        const double phi = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(samples);
        const SourcePosition probe_source {
            source.x + probe_radius * std::cos(phi),
            source.y + probe_radius * std::sin(phi),
        };
        append_valid_probe_image_seeds(
            point_magnifier, mapper, separation, mass_ratio, source, source_radius,
            probe_source, seeds);
    }
}

SourcePosition closest_point_on_segment(
    SourcePosition point,
    SourcePosition start,
    SourcePosition end);

// Walks the cached caustic branch polylines once and gathers everything the
// seeding phases need: the distance to the nearest caustic segment, the
// outside->inside transition points of each branch relative to the source
// disk, and whether any sampled vertex lies inside the disk.
struct CausticBranchScan {
    double min_distance = std::numeric_limits<double>::infinity();
    SourcePosition nearest {};
    SourcePosition nearest_segment_start {};
    SourcePosition nearest_segment_end {};
    // Outside->inside transition points of each branch (entry side only).
    std::vector<SourcePosition> crossing_probes;
    // Strided sample of all inside vertices.  Transition vertices sit near
    // the disk edge by construction, where the +-step probing of
    // append_caustic_probe_image_seeds often leaves the disk and yields
    // nothing; vertices deeper inside the disk are the reliable first
    // contact.
    std::vector<SourcePosition> first_contact_probes;
    // Sparse samples along engulfed arcs for large sources (arc seeds);
    // only probed once a first crossing has established fold seeds.
    std::vector<SourcePosition> arc_probes;
    bool any_vertex_inside = false;
    // Raw certificate descriptor accumulated during this same compulsory
    // branch walk.  Keeping it here removes a second O(Ncaustic) traversal in
    // inverse-ray seeding while `finalize_disk_support` preserves the exact
    // historical sort/merge/probe rules.
    DiskSupport support;
};

CausticBranchScan scan_caustic_branches_full(
    const std::vector<std::vector<SourcePosition>>& branches,
    SourcePosition source,
    double source_radius,
    bool build_support = false)
{
    constexpr std::size_t max_probes = 64;
    // Retain a defensive long-segment rejection for ill-conditioned root
    // solves. Physical curves are already joined by their phase monodromy, so
    // the closing segment is treated exactly like every other segment.
    constexpr double kPhantomLengthRatio2 = 625.0;  // 25x either neighbour
    CausticBranchScan scan;
    double min_distance2 = std::numeric_limits<double>::infinity();
    const double source_radius2 = source_radius * source_radius;
    std::vector<double> segment_length2;
    std::vector<double> vertex_radius;
    for (std::size_t branch_index=0;branch_index<branches.size();++branch_index) {
        const auto& branch=branches[branch_index];
        if (branch.size() < 2) {
            continue;
        }
        segment_length2.assign(branch.size(), 0.0);
        if (build_support && branch.size() >= 3) {
            vertex_radius.assign(branch.size(), 0.0);
        }
        for (std::size_t i = 0; i < branch.size(); ++i) {
            const std::size_t previous = i == 0 ? branch.size() - 1 : i - 1;
            segment_length2[i] = distance_squared(branch[previous], branch[i]);
            if (build_support && branch.size() >= 3) {
                vertex_radius[i] = std::sqrt(distance_squared(branch[i], source));
                scan.support.min_caustic_distance = std::min(
                    scan.support.min_caustic_distance, vertex_radius[i]);
            }
        }
        auto is_phantom_segment = [&](std::size_t i) {
            const double prev = segment_length2[
                i == 0 ? branch.size() - 1 : i - 1];
            const double next = segment_length2[(i + 1) % branch.size()];
            const double reference = std::max(prev, next);
            return reference > 0.0 &&
                   segment_length2[i] > kPhantomLengthRatio2 * reference;
        };
        bool prev_inside = distance_squared(branch.back(), source) < source_radius2;
        for (std::size_t i = 0; i < branch.size(); ++i) {
            const std::size_t segment_previous =
                i == 0 ? branch.size() - 1 : i - 1;
            const SourcePosition p0 = branch[segment_previous];
            const SourcePosition p1 = branch[i];
            if (!is_phantom_segment(i)) {
                const SourcePosition candidate = closest_point_on_segment(source, p0, p1);
                const double candidate_distance2 = distance_squared(candidate, source);
                if (candidate_distance2 < min_distance2) {
                    min_distance2 = candidate_distance2;
                    scan.nearest = candidate;
                    scan.nearest_segment_start = p0;
                    scan.nearest_segment_end = p1;
                }
                if (build_support && branch.size() >= 3) {
                    const SourcePosition segment {p1.x - p0.x, p1.y - p0.y};
                    const double segment_length2 =
                        segment.x * segment.x + segment.y * segment.y;
                    const double projection = segment_length2 > 0.0
                        ? ((source.x - p0.x) * segment.x +
                           (source.y - p0.y) * segment.y) / segment_length2
                        : 0.0;
                    // A radial minimum of the piecewise-linear caustic may lie
                    // in a segment interior while both sampled endpoints stay
                    // outside the source.  Add exactly that blind crossing;
                    // ordinary interior projections are already represented
                    // by retained vertex extrema and would only perturb the
                    // integration seed order with redundant probes.
                    if (projection > 1.0e-12 && projection < 1.0 - 1.0e-12 &&
                        candidate_distance2 < source_radius2 &&
                        distance_squared(p0, source) >= source_radius2 &&
                        distance_squared(p1, source) >= source_radius2) {
                        const std::size_t before_previous = segment_previous == 0
                            ? branch.size() - 1 : segment_previous - 1;
                        const std::size_t next = (i + 1) % branch.size();
                        const SourcePosition curvature0 {
                            branch[before_previous].x - 2.0 * p0.x + p1.x,
                            branch[before_previous].y - 2.0 * p0.y + p1.y,
                        };
                        const SourcePosition curvature1 {
                            p0.x - 2.0 * p1.x + branch[next].x,
                            p0.y - 2.0 * p1.y + branch[next].y,
                        };
                        const double margin = std::max(
                            std::hypot(curvature0.x, curvature0.y),
                            std::hypot(curvature1.x, curvature1.y));
                        const double candidate_distance =
                            std::sqrt(candidate_distance2);
                        const double segment_length = std::sqrt(segment_length2);
                        DiskExtremum extremum;
                        extremum.position = candidate;
                        extremum.tangent = {
                            segment.x / segment_length,
                            segment.y / segment_length};
                        extremum.normal = {
                            -extremum.tangent.y, extremum.tangent.x};
                        extremum.distance = candidate_distance;
                        extremum.polyline_margin = margin;
                        extremum.inside_disk = true;
                        extremum.branch_index=static_cast<int>(branch_index);
                        extremum.vertex_index=static_cast<int>(i);
                        scan.support.extrema.push_back(extremum);
                    }
                }
            }
            const bool inside = distance_squared(p1, source) < source_radius2;
            if (inside) {
                scan.any_vertex_inside = true;
                if (i % 5 == 0 && scan.first_contact_probes.size() < max_probes) {
                    scan.first_contact_probes.push_back(p1);
                }
            }
            if (inside && !prev_inside && scan.crossing_probes.size() < max_probes) {
                scan.crossing_probes.push_back(p1);
            } else if (inside && source_radius >= 2.0e-2 && i % 20 == 0 &&
                       scan.arc_probes.size() < max_probes) {
                scan.arc_probes.push_back(p1);
            }
            prev_inside = inside;

            // certify_disk_support historically ignores degenerate branches
            // with fewer than three vertices.  It is also intentionally
            // optional here: most callers need only the segment scan, and
            // making them pay for radial extrema would turn the fusion into a
            // regression outside augmented inverse-ray seeding.
            if (!build_support || branch.size() < 3) {
                continue;
            }

            const std::size_t previous =
                i == 0 ? branch.size() - 1 : i - 1;
            const std::size_t next = (i + 1) % branch.size();
            const bool radial_minimum =
                vertex_radius[i] <= vertex_radius[previous] &&
                vertex_radius[i] < vertex_radius[next];
            const bool radial_maximum =
                vertex_radius[i] >= vertex_radius[previous] &&
                vertex_radius[i] > vertex_radius[next];
            if (!radial_minimum && !radial_maximum) {
                continue;
            }
            const SourcePosition curvature {
                branch[previous].x - 2.0 * p1.x + branch[next].x,
                branch[previous].y - 2.0 * p1.y + branch[next].y,
            };
            const double margin = std::hypot(curvature.x, curvature.y);
            if (vertex_radius[i] >= source_radius + margin) {
                continue;
            }
            SourcePosition extremum_position = p1;
            SourcePosition chord {
                branch[next].x - branch[previous].x,
                branch[next].y - branch[previous].y,
            };
            // A shallow cap can cross the source limb between two cached
            // vertices while its sampled radial minimum remains just outside.
            // For a minimum, use the actual closest point on either adjacent
            // polyline segment.  Treating only the vertex as the extremum made
            // the certificate vacuous in exactly that case: it was labelled
            // outside and therefore did not have to straddle the caustic.
            if (radial_minimum) {
                const SourcePosition left = closest_point_on_segment(
                    source, branch[previous], p1);
                const SourcePosition right = closest_point_on_segment(
                    source, p1, branch[next]);
                if (distance_squared(left, source) <
                    distance_squared(right, source)) {
                    extremum_position = left;
                    chord = {p1.x - branch[previous].x,
                             p1.y - branch[previous].y};
                } else {
                    extremum_position = right;
                    chord = {branch[next].x - p1.x,
                             branch[next].y - p1.y};
                }
            }
            const double chord_length = std::hypot(chord.x, chord.y);
            if (!(chord_length > 0.0)) {
                continue;
            }
            DiskExtremum extremum;
            extremum.position = extremum_position;
            extremum.tangent = {
                chord.x / chord_length, chord.y / chord_length};
            extremum.normal = {-extremum.tangent.y, extremum.tangent.x};
            extremum.distance = std::sqrt(distance_squared(extremum_position, source));
            extremum.polyline_margin = margin;
            extremum.inside_disk = extremum.distance < source_radius;
            extremum.branch_index=static_cast<int>(branch_index);
            extremum.vertex_index=static_cast<int>(i);
            scan.support.extrema.push_back(extremum);
        }
    }
    scan.min_distance = std::sqrt(min_distance2);
    return scan;
}


CausticBranchScan scan_caustic_branches(
    const std::vector<std::vector<SourcePosition>>& branches,
    SourcePosition source,
    double source_radius,
    bool build_support = false,
    const std::vector<std::vector<std::size_t>>* selected_vertices=nullptr)
{
    if(!selected_vertices)return scan_caustic_branches_full(branches,source,source_radius,build_support);

    constexpr std::size_t max_probes = 64;
    // Retain a defensive long-segment rejection for ill-conditioned root
    // solves. Physical curves are already joined by their phase monodromy, so
    // the closing segment is treated exactly like every other segment.
    constexpr double kPhantomLengthRatio2 = 625.0;  // 25x either neighbour
    CausticBranchScan scan;
    double min_distance2 = std::numeric_limits<double>::infinity();
    const double source_radius2 = source_radius * source_radius;
    std::vector<double> segment_length2;
    std::vector<double> vertex_radius;
    for (std::size_t branch_index=0;branch_index<branches.size();++branch_index) {
        const auto& branch=branches[branch_index];
        const auto* selected=selected_vertices?&(*selected_vertices)[branch_index]:nullptr;
        if(selected && selected->empty())continue;
        if (branch.size() < 2) {
            continue;
        }
        if(!selected) {
        segment_length2.assign(branch.size(), 0.0);
        if (build_support && branch.size() >= 3) {
            vertex_radius.assign(branch.size(), 0.0);
        }
        for (std::size_t i = 0; i < branch.size(); ++i) {
            const std::size_t previous = i == 0 ? branch.size() - 1 : i - 1;
            segment_length2[i] = distance_squared(branch[previous], branch[i]);
            if (build_support && branch.size() >= 3) {
                vertex_radius[i] = std::sqrt(distance_squared(branch[i], source));
                scan.support.min_caustic_distance = std::min(
                    scan.support.min_caustic_distance, vertex_radius[i]);
            }
        }
        }
        const auto segment_squared=[&](std::size_t i) {
            return selected?distance_squared(branch[i==0?branch.size()-1:i-1],branch[i]):segment_length2[i];
        };
        const auto vertex_distance=[&](std::size_t i) {
            return selected?std::sqrt(distance_squared(branch[i],source)):vertex_radius[i];
        };
        auto is_phantom_segment = [&](std::size_t i) {
            const double prev = segment_squared(i == 0 ? branch.size() - 1 : i - 1);
            const double next = segment_squared((i + 1) % branch.size());
            const double reference = std::max(prev, next);
            return reference > 0.0 &&
                   segment_squared(i) > kPhantomLengthRatio2 * reference;
        };
        bool prev_inside = distance_squared(branch.back(), source) < source_radius2;
        for(std::size_t j=0;j<(selected?selected->size():branch.size());++j) {
            const std::size_t i=selected?(*selected)[j]:j;
            const std::size_t segment_previous =
                i == 0 ? branch.size() - 1 : i - 1;
            const SourcePosition p0 = branch[segment_previous];
            if(selected)prev_inside=distance_squared(p0,source)<source_radius2;
            const SourcePosition p1 = branch[i];
            if (!is_phantom_segment(i)) {
                const SourcePosition candidate = closest_point_on_segment(source, p0, p1);
                const double candidate_distance2 = distance_squared(candidate, source);
                if (candidate_distance2 < min_distance2) {
                    min_distance2 = candidate_distance2;
                    scan.nearest = candidate;
                    scan.nearest_segment_start = p0;
                    scan.nearest_segment_end = p1;
                }
                if (build_support && branch.size() >= 3) {
                    const SourcePosition segment {p1.x - p0.x, p1.y - p0.y};
                    const double segment_length2 =
                        segment.x * segment.x + segment.y * segment.y;
                    const double projection = segment_length2 > 0.0
                        ? ((source.x - p0.x) * segment.x +
                           (source.y - p0.y) * segment.y) / segment_length2
                        : 0.0;
                    // A radial minimum of the piecewise-linear caustic may lie
                    // in a segment interior while both sampled endpoints stay
                    // outside the source.  Add exactly that blind crossing;
                    // ordinary interior projections are already represented
                    // by retained vertex extrema and would only perturb the
                    // integration seed order with redundant probes.
                    if (projection > 1.0e-12 && projection < 1.0 - 1.0e-12 &&
                        candidate_distance2 < source_radius2 &&
                        distance_squared(p0, source) >= source_radius2 &&
                        distance_squared(p1, source) >= source_radius2) {
                        const std::size_t before_previous = segment_previous == 0
                            ? branch.size() - 1 : segment_previous - 1;
                        const std::size_t next = (i + 1) % branch.size();
                        const SourcePosition curvature0 {
                            branch[before_previous].x - 2.0 * p0.x + p1.x,
                            branch[before_previous].y - 2.0 * p0.y + p1.y,
                        };
                        const SourcePosition curvature1 {
                            p0.x - 2.0 * p1.x + branch[next].x,
                            p0.y - 2.0 * p1.y + branch[next].y,
                        };
                        const double margin = std::max(
                            std::hypot(curvature0.x, curvature0.y),
                            std::hypot(curvature1.x, curvature1.y));
                        const double candidate_distance =
                            std::sqrt(candidate_distance2);
                        const double segment_length = std::sqrt(segment_length2);
                        DiskExtremum extremum;
                        extremum.position = candidate;
                        extremum.tangent = {
                            segment.x / segment_length,
                            segment.y / segment_length};
                        extremum.normal = {
                            -extremum.tangent.y, extremum.tangent.x};
                        extremum.distance = candidate_distance;
                        extremum.polyline_margin = margin;
                        extremum.inside_disk = true;
                        extremum.branch_index=static_cast<int>(branch_index);
                        extremum.vertex_index=static_cast<int>(i);
                        scan.support.extrema.push_back(extremum);
                    }
                }
            }
            const bool inside = distance_squared(p1, source) < source_radius2;
            if (inside) {
                scan.any_vertex_inside = true;
                if (i % 5 == 0 && scan.first_contact_probes.size() < max_probes) {
                    scan.first_contact_probes.push_back(p1);
                }
            }
            if (inside && !prev_inside && scan.crossing_probes.size() < max_probes) {
                scan.crossing_probes.push_back(p1);
            } else if (inside && source_radius >= 2.0e-2 && i % 20 == 0 &&
                       scan.arc_probes.size() < max_probes) {
                scan.arc_probes.push_back(p1);
            }
            prev_inside = inside;

            // certify_disk_support historically ignores degenerate branches
            // with fewer than three vertices.  It is also intentionally
            // optional here: most callers need only the segment scan, and
            // making them pay for radial extrema would turn the fusion into a
            // regression outside augmented inverse-ray seeding.
            if (!build_support || branch.size() < 3) {
                continue;
            }

            const std::size_t previous =
                i == 0 ? branch.size() - 1 : i - 1;
            const std::size_t next = (i + 1) % branch.size();
            const bool radial_minimum =
                vertex_distance(i) <= vertex_distance(previous) &&
                vertex_distance(i) < vertex_distance(next);
            const bool radial_maximum =
                vertex_distance(i) >= vertex_distance(previous) &&
                vertex_distance(i) > vertex_distance(next);
            if (!radial_minimum && !radial_maximum) {
                continue;
            }
            const SourcePosition curvature {
                branch[previous].x - 2.0 * p1.x + branch[next].x,
                branch[previous].y - 2.0 * p1.y + branch[next].y,
            };
            const double margin = std::hypot(curvature.x, curvature.y);
            if (vertex_distance(i) >= source_radius + margin) {
                continue;
            }
            SourcePosition extremum_position = p1;
            SourcePosition chord {
                branch[next].x - branch[previous].x,
                branch[next].y - branch[previous].y,
            };
            // A shallow cap can cross the source limb between two cached
            // vertices while its sampled radial minimum remains just outside.
            // For a minimum, use the actual closest point on either adjacent
            // polyline segment.  Treating only the vertex as the extremum made
            // the certificate vacuous in exactly that case: it was labelled
            // outside and therefore did not have to straddle the caustic.
            if (radial_minimum) {
                const SourcePosition left = closest_point_on_segment(
                    source, branch[previous], p1);
                const SourcePosition right = closest_point_on_segment(
                    source, p1, branch[next]);
                if (distance_squared(left, source) <
                    distance_squared(right, source)) {
                    extremum_position = left;
                    chord = {p1.x - branch[previous].x,
                             p1.y - branch[previous].y};
                } else {
                    extremum_position = right;
                    chord = {branch[next].x - p1.x,
                             branch[next].y - p1.y};
                }
            }
            const double chord_length = std::hypot(chord.x, chord.y);
            if (!(chord_length > 0.0)) {
                continue;
            }
            DiskExtremum extremum;
            extremum.position = extremum_position;
            extremum.tangent = {
                chord.x / chord_length, chord.y / chord_length};
            extremum.normal = {-extremum.tangent.y, extremum.tangent.x};
            extremum.distance = std::sqrt(distance_squared(extremum_position, source));
            extremum.polyline_margin = margin;
            extremum.inside_disk = extremum.distance < source_radius;
            extremum.branch_index=static_cast<int>(branch_index);
            extremum.vertex_index=static_cast<int>(i);
            scan.support.extrema.push_back(extremum);
        }
    }
    scan.min_distance = std::sqrt(min_distance2);
    return scan;
}

void append_interior_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    std::vector<SourcePosition>& seeds)
{
    if (source_radius < 2.0e-2) {
        return;
    }
    constexpr int angle_samples = 64;
    constexpr double radii[] = {0.25, 0.5, 0.75};
    for (const double radius_fraction : radii) {
        const double probe_radius = source_radius * radius_fraction;
        for (int i = 0; i < angle_samples; ++i) {
            const double phi = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(angle_samples);
            const SourcePosition probe_source {
                source.x + probe_radius * std::cos(phi),
                source.y + probe_radius * std::sin(phi),
            };
            append_valid_probe_image_seeds(
                point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                probe_source, seeds);
        }
    }
}

std::vector<SourcePosition> critical_sources_at_phase(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    double phase_angle)
{
    math::PolynomialRootSolver solver;
    const auto root_result = solver.solve(critical_curve_polynomial_coefficients(
        separation, mass_ratio, std::polar(1.0, phase_angle)));
    if (root_result.status != math::RootSolverStatus::ok) {
        return {};
    }

    std::vector<SourcePosition> sources;
    sources.reserve(root_result.roots.size());
    for (const auto& root : root_result.roots) {
        sources.push_back(map_binary_lens_real(mapper, root.real(), root.imag()));
    }
    return sources;
}

double nearest_critical_source_distance2(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double phase_angle,
    SourcePosition* nearest_source)
{
    double best = std::numeric_limits<double>::infinity();
    const auto critical_sources =
        critical_sources_at_phase(mapper, separation, mass_ratio, phase_angle);
    for (const auto& critical_source : critical_sources) {
        const double distance2 = distance_squared(critical_source, source);
        if (distance2 < best) {
            best = distance2;
            if (nearest_source != nullptr) {
                *nearest_source = critical_source;
            }
        }
    }
    return best;
}

SourcePosition refine_nearest_critical_source(
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double center_phase,
    double phase_step)
{
    double left = center_phase - phase_step;
    double right = center_phase + phase_step;
    constexpr double golden = 0.61803398874989484820;
    double x1 = right - golden * (right - left);
    double x2 = left + golden * (right - left);
    SourcePosition nearest1;
    SourcePosition nearest2;
    double f1 = nearest_critical_source_distance2(
        mapper, separation, mass_ratio, source, x1, &nearest1);
    double f2 = nearest_critical_source_distance2(
        mapper, separation, mass_ratio, source, x2, &nearest2);

    for (int iter = 0; iter < 32; ++iter) {
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            nearest1 = nearest2;
            x2 = left + golden * (right - left);
            f2 = nearest_critical_source_distance2(
                mapper, separation, mass_ratio, source, x2, &nearest2);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            nearest2 = nearest1;
            x1 = right - golden * (right - left);
            f1 = nearest_critical_source_distance2(
                mapper, separation, mass_ratio, source, x1, &nearest1);
        }
    }

    return f1 < f2 ? nearest1 : nearest2;
}

std::vector<SourcePosition> selected_point_images(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source);

// Physical lens-map continuation, accepted only after a local Krawczyk
// inclusion for every distinct image and an independently enclosed image count.
struct ProbeContinuationImages {
    SourcePosition source{};
    std::vector<SourcePosition> images;
    bool certified=false;
};
struct ProbeLensEvaluation {
    math::FilteredBall fx{},fy{},jxx{},jxy{},jyy{};
};
template<bool Residual,bool Jacobian>
ProbeLensEvaluation enclosed_probe_lens(double separation,double mass_ratio,
    math::FilteredBall x,math::FilteredBall y,SourcePosition source)
{
    using B=math::FilteredBall;
    const double qi=std::abs(mass_ratio);
    struct Geometry { double s=0,q=0; B a{},m1{},m2{},shift{}; };
    thread_local Geometry g;
    if (g.s!=separation || g.q!=qi) {
        g.s=separation; g.q=qi;
        const B q=qi<1 ? B(qi) : B(1)/B(qi);
        g.m1=B(1)/(B(1)+q); g.m2=q*g.m1;
        g.a=B(qi<1 ? -std::abs(separation) : std::abs(separation));
        g.shift=g.a*g.m1;
    }
    const B a=g.a,m1=g.m1,m2=g.m2;
    ProbeLensEvaluation out{x-g.shift-B(source.x),y-B(source.y),B(1),B(0),B(1)};
    for (int k=0;k<2;++k) {
        const B dx=k==0 ? x-a : x, mass=k==0 ? m1 : m2;
        const B dx2=math::square(dx),y2=math::square(y),r2=dx2+y2;
        if constexpr (Residual) {
            const B inv=mass/r2;
            out.fx=out.fx-dx*inv; out.fy=out.fy-y*inv;
        }
        if constexpr (Jacobian) {
            const B inv2=mass/math::square(r2),shear=(dx2-y2)*inv2;
            out.jxx=out.jxx+shear; out.jyy=out.jyy-shear;
            out.jxy=out.jxy+B(2)*dx*y*inv2;
        }
    }
    return out;
}
bool continue_probe_images(const BinaryLensMapper& mapper,double separation,double mass_ratio,
    SourcePosition source,const std::vector<SourcePosition>& previous,
    std::vector<SourcePosition>& images,std::vector<double>* verified_radii=nullptr)
{
    using B=math::FilteredBall;
    images.clear();
    std::vector<double> radii;
    for (auto z:previous) {
        double cxx=0,cxy=0,cyy=0;
        bool converged=false;
        double initial_parity=0;
        for (int iteration=0;iteration<16;++iteration) {
            // Newton's proposal need not be certified. All acceptance uses
            // exact-input enclosures below, including geometry rounding.
            const auto mapped=map_binary_lens_real(mapper,z.x,z.y);
            double jxx=1,jxy=0,jyy=1;
            for (int k=0;k<2;++k) {
                const double x=k==0 ? z.x-mapper.separation.real() : z.x;
                const double m=k==0 ? mapper.m1 : mapper.m2;
                const double r2=x*x+z.y*z.y, inv2=m/(r2*r2);
                const double shear=(x*x-z.y*z.y)*inv2;
                jxx+=shear; jyy-=shear; jxy+=2*x*z.y*inv2;
            }
            const double det=jxx*jyy-jxy*jxy;
            if (!std::isfinite(det) || det==0) return false;
            if (iteration==0) initial_parity=det;
            cxx=jyy/det; cxy=-jxy/det; cyy=jxx/det;
            const double fx=mapped.x-source.x,fy=mapped.y-source.y;
            const double dx=cxx*fx+cxy*fy;
            const double dy=cxy*fx+cyy*fy;
            if (!std::isfinite(dx) || !std::isfinite(dy)) return false;
            if (std::max(std::abs(dx),std::abs(dy))<1e-11*(1+std::hypot(z.x,z.y))) {
                converged=true; break;
            }
            // Damped physical-map Newton keeps fold partners in their
            // respective parity basins. This is only a proposal strategy;
            // the image count and Krawczyk checks still decide acceptance.
            bool moved=false;
            double damping=1;
            for (int backtrack=0;backtrack<10;++backtrack,damping*=0.5) {
                const SourcePosition candidate{z.x-damping*dx,z.y-damping*dy};
                const auto w=map_binary_lens_real(mapper,candidate.x,candidate.y);
                const double rx=w.x-source.x,ry=w.y-source.y;
                if (!(rx*rx+ry*ry<fx*fx+fy*fy)) continue;
                double xx=1,xy=0,yy=1;
                for (int k=0;k<2;++k) {
                    const double x=k==0 ? candidate.x-mapper.separation.real() : candidate.x;
                    const double m=k==0 ? mapper.m1 : mapper.m2;
                    const double y=candidate.y,r2=x*x+y*y, inv2=m/(r2*r2);
                    const double shear=(x*x-y*y)*inv2;
                    xx+=shear; yy-=shear; xy+=2*x*y*inv2;
                }
                const double candidate_det=xx*yy-xy*xy;
                if (!std::isfinite(candidate_det) || candidate_det*initial_parity<=0) continue;
                z=candidate; moved=true; break;
            }
            if (!moved) break;
        }
        if (!converged) ++probe_counters().continuation_newton_stagnations;
        // Floating Newton may stagnate near a fold while its current proposal
        // is already certifiable. Only the inclusion below decides acceptance.
        const auto f=enclosed_probe_lens<true,false>(separation,mass_ratio,B(z.x),B(z.y),source);
        const B correction_x=B(cxx)*f.fx+B(cxy)*f.fy;
        const B correction_y=B(cxy)*f.fx+B(cyy)*f.fy;
        const double scale=1+std::hypot(z.x,z.y);
        const double radius=std::max((verified_radii ? 32*std::numeric_limits<double>::epsilon() : 1e-10)*scale,8*std::max(correction_x.magnitude(),correction_y.magnitude()));
        if (!std::isfinite(radius) || radius>1e-9*scale) return false;
        const auto j=enclosed_probe_lens<false,true>(separation,mass_ratio,B(z.x,radius),B(z.y,radius),source);
        const B qxx=B(1)-B(cxx)*j.jxx-B(cxy)*j.jxy;
        const B qxy=-B(cxx)*j.jxy-B(cxy)*j.jyy;
        const B qyx=-B(cxy)*j.jxx-B(cyy)*j.jxy;
        const B qyy=B(1)-B(cxy)*j.jxy-B(cyy)*j.jyy;
        const B r(0,radius);
        const B kx=-B(cxx)*f.fx-B(cxy)*f.fy+qxx*r+qxy*r;
        const B ky=-B(cxy)*f.fx-B(cyy)*f.fy+qyx*r+qyy*r;
        if (!kx.finite() || !ky.finite() || kx.magnitude()>=radius || ky.magnitude()>=radius
            || (qxx+ B(0,qxy.magnitude())).magnitude()>=1
            || (qyy+ B(0,qyx.magnitude())).magnitude()>=1) { ++probe_counters().continuation_inclusion_failures; return false; }
        for (std::size_t i=0;i<images.size();++i) {
            // Closed infinity-norm boxes must be disjoint.
            using I=math::OutwardInterval;
            const I dx=I(z.x)-I(images[i].x),dy=I(z.y)-I(images[i].y);
            const double gap=std::max(dx.lo>0 ? dx.lo : (dx.hi<0 ? -dx.hi : 0),
                                      dy.lo>0 ? dy.lo : (dy.hi<0 ? -dy.hi : 0));
            if (gap<=(I(radius)+I(radii[i])).hi) {
                ++probe_counters().continuation_duplicate_failures; return false;
            }
        }
        images.push_back(z); radii.push_back(radius);
    }
    if (verified_radii) *verified_radii=std::move(radii);
    return true;
}

bool certify_centre_seed_positions(double separation,double mass_ratio,SourcePosition source,double rho,const std::vector<SourcePosition>& proposals,
    const SourcePosition* support_source=nullptr) {
    using B=math::FilteredBall;using I=math::OutwardInterval;
    if (!(rho>0 && std::isfinite(rho)) || std::fegetround()!=FE_TONEAREST) return false;
    const auto count=binary_image_count_rounded_geometry(separation,mass_ratio,source);
    if (count.image_count<0) return false;
    if (static_cast<int>(proposals.size())!=count.image_count) return false;
    std::vector<SourcePosition> verified;
    const auto mapper=make_binary_lens_mapper(separation,mass_ratio);
    std::vector<double> radii;
    if (!continue_probe_images(mapper,separation,mass_ratio,source,proposals,verified,&radii)) return false;
    for (std::size_t j=0;j<verified.size();++j) {
        const auto z=verified[j];
        const double radius=math::up(std::max({radii[j],(I(z.x)-I(proposals[j].x)).magnitude(),
                                                        (I(z.y)-I(proposals[j].y)).magnitude()}));
        const auto f=enclosed_probe_lens<true,false>(separation,mass_ratio,B(z.x,radius),B(z.y,radius),support_source ? *support_source : source);
        const B d=math::square(f.fx)+math::square(f.fy);
        if (!d.finite() || (I(d.value)+I(d.radius)).hi>=math::square(I(rho)).lo) return false;
    }
    return true;
}

// Conditional sampled-critical-curve route. A regular component of f^-1(D)
// which contains no critical point maps onto all of D (proper local covering),
// so it contains a centre image. Other components meet the critical curve.
// Seeding every connected critical arc inside D therefore suffices in the
// continuous setting. Certifying that sampled arcs cover the *continuous*
// curve remains a separate task; do not promote this diagnostic route yet.
bool append_critical_arc_seeds(const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,double separation,double mass_ratio,SourcePosition source,double source_radius,
    const DiskSupport& support,const std::vector<std::vector<SourcePosition>>& caustics,
    const std::vector<std::vector<SourcePosition>>& critical,
    std::vector<SourcePosition>& seeds,int /*centre_count*/)
{
    using B=math::FilteredBall;using I=math::OutwardInterval;
    auto& counters=probe_counters();
    const double radius2=source_radius*source_radius;

    for (std::size_t branch=0;branch<caustics.size();++branch) {
        const auto& caustic=caustics[branch];const auto n=caustic.size();
        if (branch>=critical.size() || critical[branch].size()!=n || n==0) continue;
        std::vector<std::size_t> arc;
        const auto flush=[&]() {
            if (arc.empty()) return;
            auto best=arc.front();
            for (auto index:arc)
                if (distance_squared(caustic[index],source)<distance_squared(caustic[best],source)) best=index;
            const auto point=critical[branch][best];
            const auto mapped=enclosed_probe_lens<true,false>(separation,mass_ratio,B(point.x),B(point.y),source);
            const B distance=math::square(mapped.fx)+math::square(mapped.fy);
            if (distance.finite() && (I(distance.value)+I(distance.radius)).hi<math::square(I(source_radius)).lo) {
                seeds.push_back(point);
                ++counters.critical_arc_seeds;
            }
            arc.clear();
        };
        // Begin just after an outside vertex, so a wraparound arc is treated
        // as one interval. A fully inside loop likewise supplies one seed.
        std::size_t start=0;
        for (std::size_t i=0;i<n;++i)
            if (distance_squared(caustic[i],source)>=radius2) {start=(i+1)%n;break;}
        for (std::size_t k=0;k<n;++k) {
            const auto index=(start+k)%n;
            if (distance_squared(caustic[index],source)<radius2) arc.push_back(index);
            else flush();
        }
        flush();
    }
    // Classify the historical traversal first. Keep every extremum/direction
    // separate; equal counts on different rays are never a grouping witness.
    struct ClassifiedProbe {
        DiskProbe probe;
        int count=-1;
        bool rooted=false;
    };
    std::vector<ClassifiedProbe> classified;
    const auto root_probe=[&](const DiskProbe& probe) {
        const auto begin=std::chrono::steady_clock::now();
        const auto images=selected_point_images(point_magnifier,separation,mass_ratio,probe.position);
        ++counters.certified_solves;
        ++counters.critical_incremental_solves;
        counters.certified_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        append_probe_images_as_seeds(mapper,source,source_radius,images,seeds);
        return static_cast<int>(images.size());
    };
    bool proven=resolve_certified_probe_candidates(support,[&](const DiskProbe& probe) {
        if (distance_squared(probe.position,source)>=radius2*(1+1e-10)) return -1;
        const auto begin=std::chrono::steady_clock::now();
        const auto count=binary_image_count_caustic_filter(separation,mass_ratio,probe.position);
        counters.representative_count_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        ClassifiedProbe item{probe,count.image_count,false};
        if (item.count<0) {
            // Resolve only this ambiguous count, retaining its roots as seeds.
            item.count=root_probe(probe);item.rooted=true;
        }
        classified.push_back(item);
        return item.count;
    });
    std::vector<std::vector<std::size_t>> rays(support.extrema.size()*kProbeDirections);
    for (std::size_t i=0;i<classified.size();++i) {
        const auto& p=classified[i].probe;
        rays[p.extremum*kProbeDirections+static_cast<int>(p.direction)].push_back(i);
    }
    std::vector<unsigned char> selected(classified.size(),0);
    int representative_limit=2;
    if (const char* limit=std::getenv("LCBININT_PROBE_INTERVAL_REPRESENTATIVES"))
        representative_limit=std::clamp(std::atoi(limit),2,8);
    for (auto& ray:rays) {
        std::sort(ray.begin(),ray.end(),[&](auto a,auto b) {
            return classified[a].probe.offset<classified[b].probe.offset;
        });
        // Preserve both ends of each sampled constant-count interval. In
        // particular, retain both sides of every detected count transition.
        // A count match says nothing about a different extremum or direction.
        for (std::size_t first=0;first<ray.size();) {
            std::size_t last=first+1;
            while (last<ray.size() && classified[ray[last]].count==classified[ray[first]].count) ++last;
            selected[ray[first]]=1;
            selected[ray[last-1]]=1;
            // Keep the historical half-room representative too. Endpoints
            // alone can lie in thin portions of an otherwise regular domain.
            auto representative=ray[first];
            for (std::size_t j=first;j<last;++j)
                if (ray[j]<representative) representative=ray[j];
            if (representative_limit>=3) selected[representative]=1;
            for (int pass=3;pass<representative_limit;++pass) {
                double best_gap=-1;
                std::size_t best=ray.size();
                for (std::size_t j=first;j<last;++j) {
                    if (selected[ray[j]]) continue;
                    double gap=std::numeric_limits<double>::infinity();
                    for (std::size_t k=first;k<last;++k)
                        if (selected[ray[k]]) gap=std::min(gap,std::abs(
                            classified[ray[j]].probe.offset-classified[ray[k]].probe.offset));
                    if (gap>best_gap) {best_gap=gap;best=j;}
                }
                if (best==ray.size()) break;
                selected[ray[best]]=1;
            }
            first=last;
        }
    }
    for (std::size_t i=0;i<classified.size();++i) {
        if (classified[i].rooted) continue;
        if (selected[i]) {
            const int actual=root_probe(classified[i].probe);
            if (actual!=classified[i].count) {
                // selected_point_images retains near-caustic candidates by
                // residual, not an exact image-count certificate. Their
                // positions are valid disk seeds after the map check; retain
                // the discriminant count for the support traversal.
                ++counters.discriminant_mismatches;
            }
        }
        else ++counters.representative_skipped;
    }
    if (!proven) ++counters.unproven;
    return proven;
}

// Roots the certified support probes and adds every resulting image to the
// seed set.  Returns whether the support was proven; see
// resolve_certified_probes for the rule.
bool append_certified_component_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const DiskSupport& support,
    std::vector<SourcePosition>& seeds,
    const std::vector<std::vector<SourcePosition>>* branches=nullptr,
    int centre_image_count=-1)
{
    const double source_radius2 = source_radius * source_radius;
    const bool stats = probe_stats_enabled();
    // Reduce repeated roots only with a caustic-free source-domain witness.
    // The production Cartesian multi-run fill exhausts each reached lattice
    // component; continuous support coverage and finite-grid coverage still
    // need separate audits when removing historical seeds.
    struct ProbeGroup {
        SourcePosition lower{}, upper{}, representative{};
        bool present=false, checked=false, rooted=false;
        int count=-1;
        BinaryCausticGradient gradient{};
        double maximum_offset=0;
        double minimum_offset=std::numeric_limits<double>::infinity();
        double representative_offset=0;
        SourcePosition ray_first{},ray_last{};
        std::vector<DiskProbe> ray_probes;
        BinaryCausticRay ray{};
    };
    const bool continuation=std::getenv("LCBININT_PROBE_CONTINUATION")!=nullptr;
    std::vector<ProbeContinuationImages> continuation_cache;
    if (continuation && centre_image_count>0 && seeds.size()>=static_cast<std::size_t>(centre_image_count)) {
        continuation_cache.push_back({source,std::vector<SourcePosition>(seeds.begin(),seeds.begin()+centre_image_count),false});
    }
    const char* representative_mode=std::getenv("LCBININT_REPRESENTATIVE_PROBES");
    const bool monotone=representative_mode && std::string(representative_mode)=="monotone-rays";
    const bool fine_boxes=representative_mode && std::string(representative_mode)=="fine-boxes";
    const bool ray_polynomial=representative_mode && std::string(representative_mode)=="ray-polynomial";
    const bool source_boxes=representative_mode && std::string(representative_mode)=="source-boxes";
    const bool representatives=representative_probes_enabled();
    const bool count_ablation=representative_count_ablation_enabled();
    const bool polyline_links=representative_polyline_links_enabled();
    const bool verified_links=representative_verified_links_enabled();
    const bool pooled_links=verified_links && std::getenv("LCBININT_POOLED_PROBE_LINKS");
    const auto disk_inside=[&](SourcePosition position) {
        using I=math::OutwardInterval;
        const I distance=math::square(I(position.x)-I(source.x))+math::square(I(position.y)-I(source.y));
        return distance.finite() && distance.hi<math::square(I(source_radius)).lo;
    };
    struct SolvedSource {SourcePosition source;int count;};
    std::vector<SolvedSource> solved_sources;
    std::vector<BinarySourceBox> connected_boxes;
    if (source_boxes && centre_image_count>0) solved_sources.push_back({source,centre_image_count});
    const auto contains=[](const BinarySourceBox& box,SourcePosition p) {
        return p.x>=box.lower.x && p.x<=box.upper.x && p.y>=box.lower.y && p.y<=box.upper.y;
    };
    const auto connected_in_disk=[&](const BinarySourceBox& a,const BinarySourceBox& b) {
        const double lx=std::max(a.lower.x,b.lower.x),ux=std::min(a.upper.x,b.upper.x);
        const double ly=std::max(a.lower.y,b.lower.y),uy=std::min(a.upper.y,b.upper.y);
        if (lx>ux || ly>uy) return false;
        return disk_inside({std::clamp(source.x,lx,ux),std::clamp(source.y,ly,uy)});
    };
    const auto retain_box=[&](const BinarySourceBox& box) {
        // Discard boxes contained by a new larger witness. They prove no
        // additional connections, and retaining them only grows search cost.
        for (const auto& old:connected_boxes)
            if (old.image_count==box.image_count && contains(old,box.lower) && contains(old,box.upper)) return;
        connected_boxes.erase(std::remove_if(connected_boxes.begin(),connected_boxes.end(),[&](const auto& old) {
            return old.image_count==box.image_count && contains(box,old.lower) && contains(box,old.upper);
        }),connected_boxes.end());
        if (connected_boxes.size()<256) connected_boxes.push_back(box);
    };
    std::optional<detail::CausticProbeLinks> links;
    if (polyline_links && !pooled_links && branches) links.emplace(*branches);
    std::vector<ProbeGroup> groups(representatives ? kProbeDirections*support.extrema.size()+(pooled_links ? 1 : 0) : 0);
    if (pooled_links && centre_image_count>=0) {
        // Positions of these images are already in seeds; do not root the
        // centre again. Hinted seeds never masquerade as centre images.
        auto& centre=groups.back();
        centre.rooted=true; centre.count=centre_image_count;
        centre.representative=source;
    }
    if (representatives) {
        for (const auto& candidate: support.probes) {
            if (candidate.extremum<0 || static_cast<std::size_t>(candidate.extremum)>=support.extrema.size()) continue;
            auto& g=groups[kProbeDirections*candidate.extremum+static_cast<int>(candidate.direction)];
            if (ray_polynomial) {
                if (candidate.offset<g.minimum_offset) {g.minimum_offset=candidate.offset;g.ray_first=candidate.position;}
                if (candidate.offset>=g.maximum_offset) g.ray_last=candidate.position;
                g.ray_probes.push_back(candidate);
            }
            g.maximum_offset=std::max(g.maximum_offset,candidate.offset);
            if (!g.present) { g.lower=g.upper=candidate.position; g.present=true; }
            g.lower.x=std::min(g.lower.x,candidate.position.x);
            g.lower.y=std::min(g.lower.y,candidate.position.y);
            g.upper.x=std::max(g.upper.x,candidate.position.x);
            g.upper.y=std::max(g.upper.y,candidate.position.y);
        }
    }
    if (monotone || fine_boxes) {
        // Spend the shared certificate only on the fine part of the ladder.
        // Wide coarse spans resolved almost no derivative signs in the pilot.
        for (auto& g:groups) g.present=false;
        for (const auto& candidate:support.probes) {
            if (candidate.extremum<0 || static_cast<std::size_t>(candidate.extremum)>=support.extrema.size()) continue;
            auto& g=groups[kProbeDirections*candidate.extremum+static_cast<int>(candidate.direction)];
            if (candidate.offset>0.1*g.maximum_offset) continue;
            if (!g.present) { g.lower=g.upper=candidate.position; g.present=true; }
            g.lower.x=std::min(g.lower.x,candidate.position.x); g.lower.y=std::min(g.lower.y,candidate.position.y);
            g.upper.x=std::max(g.upper.x,candidate.position.x); g.upper.y=std::max(g.upper.y,candidate.position.y);
        }
    }
    const bool proven = resolve_certified_probe_candidates(support, [&](const DiskProbe& candidate) {
        const auto probe=candidate.position;

        if (distance_squared(probe, source) >= source_radius2 * (1.0 + 1.0e-10)) {
            return -1;
        }
        BinaryImageCount cheap_count{};
        BinarySourceBox source_box{};
        ProbeGroup* group=nullptr;
        if (representatives) {
            group=&groups[kProbeDirections*candidate.extremum+static_cast<int>(candidate.direction)];
            if (source_boxes) {
                // Reuse completed work before attempting any new certificate.
                for (const auto& previous:solved_sources) {
                    if (previous.source.x==probe.x && previous.source.y==probe.y) {
                        ++probe_counters().representative_skipped;
                        ++probe_counters().source_cell_reuses;
                        return previous.count;
                    }
                }
                if (disk_inside(probe)) {
                    for (const auto& box:connected_boxes) {
                        if (contains(box,probe)) {
                            ++probe_counters().representative_skipped;
                            ++probe_counters().source_cell_reuses;
                            return box.image_count;
                        }
                    }
                    const auto begin=std::chrono::steady_clock::now();
                    source_box=binary_source_box(separation,mass_ratio,probe,source_radius);
                    auto& counters=probe_counters();
                    counters.source_cell_cache_hits+=source_box.cache_hits;
                    counters.source_cell_enclosures+=source_box.enclosure_evaluations;
                    counters.representative_box_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                    cheap_count.image_count=source_box.image_count;
                    bool reached=false;
                    if (source_box.image_count>=0) {
                        for (const auto& box:connected_boxes) {
                            if (box.image_count==source_box.image_count && connected_in_disk(box,source_box)) {reached=true;break;}
                        }
                        if (!reached) for (const auto& previous:solved_sources) {
                            if (previous.count==source_box.image_count && disk_inside(previous.source) && contains(source_box,previous.source)) {reached=true;break;}
                        }
                    }
                    if (reached) {
                        retain_box(source_box);
                        ++counters.representative_skipped;
                        ++counters.source_cell_reuses;
                        return source_box.image_count;
                    }
                }
                // Incremental fallback: only this unresolved/unreached probe
                // needs positions. Keep every earlier source and cell proof.
                ++probe_counters().source_cell_fallbacks;
            } else if (ray_polynomial) {
                // Pay setup only if this group actually consumes a second
                // probe. The first probe already supplies the representative.
                if (group->rooted && !group->checked && group->maximum_offset>group->minimum_offset) {
                    const auto begin=std::chrono::steady_clock::now();
                    using B=math::FilteredBall;
                    const B span=B(group->maximum_offset)-B(group->minimum_offset);
                    const B dx=B(group->ray_last.x)-B(group->ray_first.x);
                    const B dy=B(group->ray_last.y)-B(group->ray_first.y);
                    SourcePosition residual{};
                    for (const auto& p:group->ray_probes) {
                        const B t=(B(p.offset)-B(group->minimum_offset))/span;
                        residual.x=std::max(residual.x,(B(p.position.x)-(B(group->ray_first.x)+t*dx)).magnitude());
                        residual.y=std::max(residual.y,(B(p.position.y)-(B(group->ray_first.y)+t*dy)).magnitude());
                    }
                    group->ray=binary_caustic_ray(separation,mass_ratio,group->ray_first,group->ray_last,residual);
                    group->checked=true;
                    ++probe_counters().representative_groups;
                    probe_counters().representative_box_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                }
                if (group->rooted && group->ray.valid && group->count>=0
                    && disk_inside(group->representative) && disk_inside(probe)) {
                    using I=math::OutwardInterval; using B=math::FilteredBall;
                    const I span=I(group->maximum_offset)-I(group->minimum_offset);
                    const I a=(I(group->representative_offset)-I(group->minimum_offset))/span;
                    const I b=(I(candidate.offset)-I(group->minimum_offset))/span;
                    const double lo=std::min(a.lo,b.lo),hi=std::max(a.hi,b.hi),mid=lo*0.5+hi*0.5;
                    const B parameter(mid,std::max((I(mid)-I(lo)).hi,(I(hi)-I(mid)).hi));
                    const auto begin=std::chrono::steady_clock::now();
                    const auto count=group->ray.count(parameter);
                    probe_counters().representative_count_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                    if (count.image_count==group->count) {
                        ++probe_counters().representative_skipped;
                        return group->count;
                    }
                }
            } else if (fine_boxes && candidate.offset<=0.1*group->maximum_offset) {
                if (!group->checked) {
                    const auto begin=std::chrono::steady_clock::now();
                    const auto box=binary_image_count_box(separation,mass_ratio,group->lower,group->upper);
                    group->count=box.image_count;
                    group->checked=true;
                    ++probe_counters().representative_groups;
                    probe_counters().representative_boxes_resolved+=box.image_count>=0;
                    probe_counters().representative_box_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                }
                if (group->rooted && group->count>=0
                    && group->representative.x>=group->lower.x && group->representative.x<=group->upper.x
                    && group->representative.y>=group->lower.y && group->representative.y<=group->upper.y
                    && disk_inside(group->representative) && disk_inside(probe)) {
                    ++probe_counters().representative_skipped;
                    return group->count;
                }
            } else if (fine_boxes) {
                // Keep coarse topology witnesses; certify only the fine box.
            } else if (monotone && candidate.offset<=0.1*group->maximum_offset) {
                if (!group->checked) {
                    const auto begin=std::chrono::steady_clock::now();
                    group->gradient=binary_caustic_gradient_box(separation,mass_ratio,group->lower,group->upper);
                    group->checked=true;
                    ++probe_counters().representative_groups;
                    probe_counters().representative_box_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                }
                const auto begin=std::chrono::steady_clock::now();
                cheap_count=binary_image_count_caustic_filter(separation,mass_ratio,probe);
                probe_counters().representative_count_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                if (group->rooted && group->count>=0 && cheap_count.image_count==group->count
                    && group->representative.x>=group->lower.x && group->representative.x<=group->upper.x
                    && group->representative.y>=group->lower.y && group->representative.y<=group->upper.y
                    && group->gradient.valid && disk_inside(group->representative) && disk_inside(probe)) {
                    using B=math::FilteredBall;
                    const B dx=B(probe.x)-B(group->representative.x),dy=B(probe.y)-B(group->representative.y);
                    const B derivative=group->gradient.x*dx+group->gradient.y*dy;
                    if ((dx.zero() && dy.zero()) || (derivative.finite() && !derivative.contains_zero())) {
                        ++probe_counters().representative_skipped;
                        return group->count;
                    }
                }
            } else if (monotone) {
                // Coarse probes still supply roots and topology witnesses.
            } else if (polyline_links) {
                const auto begin=std::chrono::steady_clock::now();
                bool skip=false;
                // Reuse representatives across directions/extrema only through
                // an enclosed caustic-free path inside the convex source disk.
                // Bound search cost: try the nearest rooted representative.
                if (pooled_links && disk_inside(probe)) {
                    const auto probe_count=binary_image_count_discriminant(separation,mass_ratio,probe);
                    ProbeGroup* nearest=nullptr;
                    double nearest_distance=std::numeric_limits<double>::infinity();
                    for (auto& other:groups) {
                        if (!other.rooted || other.count!=probe_count.image_count || other.count<0) continue;
                        const double distance=distance_squared(other.representative,probe);
                        if (distance<nearest_distance) { nearest=&other; nearest_distance=distance; }
                    }
                    if (nearest && disk_inside(nearest->representative)) {
                        int boxes=0;
                        ++probe_counters().representative_link_checks;
                        skip=binary_image_link_verified(separation,mass_ratio,nearest->representative,probe,nearest->count,&boxes);
                        probe_counters().representative_link_boxes+=boxes;
                        if (skip) {
                            ++probe_counters().representative_verified_skipped;
                            group->rooted=true; group->count=nearest->count;
                            group->representative=probe;
                        }
                    }
                }
                if (!skip && group->rooted && links) {
                    auto& c=probe_counters();
                    ++c.representative_link_checks;
                    if (disk_inside(group->representative) && disk_inside(probe)
                        && links->proposed_clear(group->representative,probe)) {
                        ++c.representative_polyline_clear;
                        if (verified_links) {
                            int boxes=0;
                            skip=binary_image_link_verified(separation,mass_ratio,group->representative,probe,group->count,&boxes);
                            c.representative_link_boxes+=boxes;
                            if (skip) ++c.representative_verified_skipped;
                        } else {
                            const auto count=binary_image_count_discriminant(separation,mass_ratio,probe);
                            skip=count.image_count>=0 && count.image_count==group->count;
                            if (skip) ++c.representative_conditional_skipped;
                        }
                    }
                }
                probe_counters().representative_link_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                if (skip) { ++probe_counters().representative_skipped; return group->count; }
            } else if (count_ablation) {
                const auto begin=std::chrono::steady_clock::now();
                const auto count=binary_image_count_discriminant(separation,mass_ratio,probe);
                probe_counters().representative_count_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                if (group->rooted && count.image_count>=0 && count.image_count==group->count) {
                    ++probe_counters().representative_skipped;
                    return count.image_count;
                }
                // No claim about caustics between points; this diagnostic
                // ablation only measures whether seed removal changes fill.
                group->count=count.image_count;
            } else if (!group->checked) {
                const auto begin=std::chrono::steady_clock::now();
                const auto box=binary_image_count_box(separation,mass_ratio,group->lower,group->upper);
                group->checked=true;
                group->count=box.image_count;
                auto& c=probe_counters();
                ++c.representative_groups;
                c.representative_boxes_resolved+=box.image_count>=0;
                c.representative_box_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            }
            if (!source_boxes && !ray_polynomial && !monotone && !fine_boxes && !polyline_links && !count_ablation && group->rooted && group->count>=0) {
                ++probe_counters().representative_skipped;
                return group->count;
            }
        }
        const auto started = stats ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point {};
        std::vector<SourcePosition> probe_images;
        bool continued=false;
        if (continuation) {
            const auto begin=std::chrono::steady_clock::now();
            ++probe_counters().continuation_attempts;
            for (const auto& previous:continuation_cache) {
                if (previous.certified && previous.source.x==probe.x && previous.source.y==probe.y) {
                    probe_images=previous.images; continued=true; break;
                }
            }
            const auto count=continued ? BinaryImageCount{} : binary_image_count_discriminant(separation,mass_ratio,probe);
            const ProbeContinuationImages* nearest=nullptr;
            double distance=std::numeric_limits<double>::infinity();
            if (count.image_count>=0) {
                for (const auto& previous:continuation_cache) {
                    if (static_cast<int>(previous.images.size())!=count.image_count) continue;
                    const double d=distance_squared(previous.source,probe);
                    if (d<distance) { distance=d; nearest=&previous; }
                }
                if (nearest) continued=continue_probe_images(mapper,separation,mass_ratio,probe,nearest->images,probe_images);
            }
            probe_counters().continuation_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            probe_counters().continuation_accepted+=continued;
        }
        const auto root_started=stats ? std::chrono::steady_clock::now() : started;
        if (!continued) probe_images=selected_point_images(point_magnifier,separation,mass_ratio,probe);
        if (continuation) {
            if (continuation_cache.size()==64) continuation_cache.erase(continuation_cache.begin());
            continuation_cache.push_back({probe,probe_images,continued});
        }
        if (stats && !continued) {
            auto& counters = probe_counters();
            counters.certified_solves += 1;
            counters.certified_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - root_started).count();
        }
        if (group!=nullptr) {
            group->rooted=true;
            if (!source_boxes && !ray_polynomial && !monotone && !polyline_links && group->count>=0 && group->count!=static_cast<int>(probe_images.size())) {
                ++probe_counters().representative_mismatches;
                group->count=-1;
            }
            if (monotone) {
                group->count=cheap_count.image_count==static_cast<int>(probe_images.size()) ? cheap_count.image_count : -1;
                if (cheap_count.image_count>=0 && group->count<0) ++probe_counters().representative_mismatches;
            }
            if (count_ablation || polyline_links) group->count=static_cast<int>(probe_images.size());
            if (ray_polynomial) group->count=static_cast<int>(probe_images.size());
            group->representative=probe;
            group->representative_offset=candidate.offset;
        }
        if (source_boxes) {
            const int count=static_cast<int>(probe_images.size());
            if (solved_sources.size()<256) solved_sources.push_back({probe,count});
            if (source_box.image_count>=0 && source_box.image_count==count && disk_inside(probe)) retain_box(source_box);
            else if (source_box.image_count>=0 && source_box.image_count!=count) ++probe_counters().representative_mismatches;
        }
        if (discriminant_diagnostics_enabled()) {
            const auto count_started = std::chrono::steady_clock::now();
            const auto count = binary_image_count_discriminant(separation, mass_ratio, probe);
            auto& counters = probe_counters();
            ++counters.discriminant_calls;
            counters.discriminant_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now()-count_started).count();
            if (count.status == DiscriminantStatus::resolved) {
                ++counters.discriminant_resolved;
                if (count.filtered) ++counters.discriminant_filtered;
                if (count.image_count != static_cast<int>(probe_images.size()))
                    ++counters.discriminant_mismatches;
            } else {
                ++counters.discriminant_unresolved;
            }
        }
        append_probe_images_as_seeds(
            mapper, source, source_radius, probe_images, seeds);
        return static_cast<int>(probe_images.size());
    });
    if (stats && !proven) {
        probe_counters().unproven += 1;
    }
    return proven;
}

std::vector<SourcePosition> augmented_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const BinaryLensMapper& mapper,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double hint_caustic_dist = std::numeric_limits<double>::infinity(),
    const std::vector<SourcePosition>* seed_hints = nullptr,
    const std::vector<std::vector<SourcePosition>>* caustic_branches = nullptr,
    bool* support_proven = nullptr,
    std::size_t* initial_seed_count = nullptr,
    const std::vector<std::vector<SourcePosition>>* critical_branches = nullptr,
    int known_centre_count=-1,
    const std::vector<std::vector<std::size_t>>* selected_vertices=nullptr)
{
    if (support_proven != nullptr) {
        *support_proven = true;
    }
    // Whole-curve Rouché coverage and centre Krawczyk checks establish a
    // stronger contract than replacing the historical probe image counts.
    // Keep them explicitly diagnostic; the reduced runtime route below uses
    // the existing support traversal and only local seed/count checks.
    // Retain the historical flag as an audit alias for saved experiments.
    const bool full_seed_audit=std::getenv("LCBININT_SEED_FULL_COVERAGE_AUDIT")!=nullptr
        || std::getenv("LCBININT_CRITICAL_SUPPORT_CERTIFICATE")!=nullptr;
    std::vector<SourcePosition> centre_positions;
    std::vector<SourcePosition> seeds;
    seeds.reserve(seed_hints == nullptr ? 5 : std::max<std::size_t>(5, seed_hints->size()));
    if (seed_hints != nullptr) {
        for (const auto& seed : *seed_hints) {
            append_valid_seed(mapper, source, source_radius, seed, seeds);
        }
    }
    int centre_image_count=-1;
    // Only the caller with genuine centre-image metadata may expose this
    // count. Arbitrary seed hints must never masquerade as centre images.
    if (seed_hints && known_centre_count>0
        && static_cast<int>(seed_hints->size())==known_centre_count
        && seeds.size()==seed_hints->size()) {
        centre_image_count=known_centre_count;if(full_seed_audit) centre_positions=seeds;
    }
    if (seeds.empty()) {
        const auto point_images = point_magnifier.binary_images(separation, mass_ratio, source);
        centre_image_count=static_cast<int>(point_images.size());
        for (const auto& image : point_images) {
            seeds.push_back(image.position);if(full_seed_audit) centre_positions.push_back(image.position);
        }
    }
    if(auto* work=adaptive_flux_request();work && centre_image_count>0) {
        for(int i=0;i<centre_image_count;++i)work->centres.push_back({seeds[i].x,seeds[i].y});
    }
    const char* requested_mode=std::getenv("LCBININT_REPRESENTATIVE_PROBES");
    if (requested_mode && std::string(requested_mode)=="critical-seeds"
        && critical_branches && centre_image_count<0) {
        // Obtain only the missing centre witness, retaining all earlier hints.
        const auto images=selected_point_images(point_magnifier,separation,mass_ratio,source);
        append_probe_images_as_seeds(mapper,source,source_radius,images,seeds);
        centre_image_count=static_cast<int>(images.size());
        if(full_seed_audit) for(const auto& image:images) centre_positions.push_back(image);
        ++probe_counters().critical_centre_solves;
    }
    if (initial_seed_count != nullptr) {
        // With no seed hints this is exactly the centre-image count.  Exposing
        // it lets callers that only need that count reuse the root solve which
        // built the seed set instead of solving the same lens equation again.
        *initial_seed_count = seeds.size();
    }
    if (source_radius <= 0.0) {
        return seeds;
    }
    if (full_seed_audit && requested_mode && std::string(requested_mode)=="critical-seeds") {
        auto& counters=probe_counters();++counters.critical_support_calls;
        const auto started=std::chrono::steady_clock::now();
        const auto proof=cached_critical_support_certificate(separation,mass_ratio,source,source_radius);
        bool centre_proven=certify_centre_seed_positions(separation,mass_ratio,source,source_radius,centre_positions);
        if (!centre_proven) {
            // Any regular source point strictly inside D witnesses every
            // critical-free component. On-axis/near-caustic centre ambiguity
            // therefore needs only one new representative, never a replay
            // of the legacy probe rings. Retain all previous roots/seeds.
            for (const auto offset:std::array<SourcePosition,4>{{{0,.25},{0,-.25},{.25,.125},{-.25,-.125}}}) {
                const SourcePosition probe{source.x+offset.x*source_radius,source.y+offset.y*source_radius};
                if (probe.x==source.x && probe.y==source.y) continue;
                using I=math::OutwardInterval;
                const I d=math::square(I(probe.x)-I(source.x))+math::square(I(probe.y)-I(source.y));
                if (!(d.hi<math::square(I(source_radius)).lo)) continue;
                if (binary_image_count_discriminant(separation,mass_ratio,probe).image_count<0) continue;
                const auto rooted_at=std::chrono::steady_clock::now();
                const auto positions=selected_point_images(point_magnifier,separation,mass_ratio,probe);
                ++counters.critical_centre_solves;++counters.certified_solves;
                counters.certified_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-rooted_at).count();
                append_probe_images_as_seeds(mapper,source,source_radius,positions,seeds);
                if (certify_centre_seed_positions(separation,mass_ratio,probe,source_radius,positions,&source)) {
                    centre_proven=true;break;
                }
            }
        }
        counters.critical_support_phase_queries+=proof.phase_queries;
        counters.critical_support_proof_queries+=proof.proof_queries;
        for (const auto& witness:proof.witnesses) {
            seeds.push_back(witness.position);++counters.critical_arc_seeds;
        }
        const bool proven=proof.resolved && centre_proven;
        certified_seed_support={std::abs(separation),std::abs(mass_ratio),source_radius,source,proven,seeds};
        if(proven) ++counters.critical_support_proven;else {++counters.critical_support_unresolved;++counters.unproven;}
        counters.critical_support_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        if(support_proven) *support_proven=proven;
        // Unresolved intervals retain every earlier witness and cache entry.
        // They are not silently replaced by sampled support acceptance.
        return seeds;
    }
    if (caustic_branches != nullptr) {
        // Cache-driven seeding: one pass over the cached caustic polylines
        // replaces the per-epoch critical-curve phase scans (up to ~2800
        // quartic root solves) with pure distance queries, and gates the
        // probe rings on the actual caustic geometry.
        //
        // The legacy walker's claimed-cell registry does NOT make its filled
        // area independent of the seed set.  In that diagnostic path a seed
        // whose |J| is below kFoldJacobianThreshold has its fill pinned to one
        // side of the critical curve, and a pinned fill still claims every cell
        // it counts, so whichever seed reaches a component first decides how
        // much of it is ever counted.  Adding a seed can therefore lower the
        // answer.  This restriction is specific to the legacy walker: the
        // production Cartesian multi-run fill is parity free and exhausts
        // each reached lattice component.  Seed reduction there must retain
        // a witness for every component, rather than every historical probe.
        // This scan uses the nearest point on every polyline segment, not just
        // its vertices.  That distinction is required for coarse configurable
        // caustic caches: a segment can enter the source disk while both ends
        // remain outside.  Keep it as the authoritative distance and seeding
        // pass until the support certificate carries equivalent segment data.
        auto scan = scan_caustic_branches(
            *caustic_branches, source, source_radius, true, selected_vertices);
        // Preserve the historical sampled-support traversal. Solving every
        // selected probe does not certify coverage between sampled caustic
        // segments; substituting a definite image count does not introduce
        // that new obligation either. Whole-curve coverage is a separate audit.
        const auto support = finalize_disk_support(
            std::move(scan.support), source, source_radius);
        const char* mode=std::getenv("LCBININT_REPRESENTATIVE_PROBES");
        const bool use_critical=mode && std::string(mode)=="critical-seeds" && critical_branches && centre_image_count>0;
        const bool proven = use_critical
            ? append_critical_arc_seeds(point_magnifier,mapper,separation,mass_ratio,source,source_radius,
                support,*caustic_branches,*critical_branches,seeds,centre_image_count)
            : append_certified_component_seeds(
                point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                support, seeds, caustic_branches, centre_image_count);
        if (support_proven != nullptr) {
            *support_proven = proven;
        }
        return seeds;
    }

    // Fallback (no cached branches supplied): per-epoch critical-curve phase
    // scans. Do not skip the caustic scan based on hint_caustic_dist alone:
    // sampled distances can be slightly over-estimated, causing a false early
    // exit when the source disk just straddles the caustic.

    const double source_radius2 = source_radius * source_radius;
    const int samples = 1400;
    const double phase_step = 2.0 * kPi / static_cast<double>(samples);
    double best_distance2 = std::numeric_limits<double>::infinity();
    double best_phase = 0.0;
    constexpr int nskip = 40;

    // Phase 1: find the first caustic crossing and add fold-image seeds.
    // Merge into existing seeds rather than replacing them so that the Phase 0
    // standard-image seed (high-J, non-fold image) is preserved.  Without this
    // the standard image is absent at low bins, where fold-image flood-fills do
    // not expand far enough to cover it.
    //
    // IMPORTANT: the duplicate check uses the Phase-0 seed count snapshot, not
    // the live seeds vector.  This prevents F+ from blocking F- when both fold
    // images land within rho of each other (which happens when the probe source
    // is only slightly inside the caustic, so F+/F- separation << rho).
    bool found_first_crossing = false;
    for (int kphi = 0; kphi < nskip && !found_first_crossing; ++kphi) {
        for (int jphi = 0; jphi < samples / nskip && !found_first_crossing; ++jphi) {
            const int sample = jphi * nskip + kphi;
            const double phi = phase_step * static_cast<double>(sample);
            const auto critical_sources =
                critical_sources_at_phase(mapper, separation, mass_ratio, phi);
            for (const auto& critical_source : critical_sources) {
                const double distance2 = distance_squared(critical_source, source);
                if (distance2 < best_distance2) {
                    best_distance2 = distance2;
                    best_phase = phi;
                }
                if (distance2 >= source_radius2 || distance2 <= 0.0) {
                    continue;
                }
                const std::size_t before = seeds.size();
                append_caustic_probe_image_seeds(
                    point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                    critical_source, seeds);
                found_first_crossing = seeds.size() > before;
            }
        }
    }
    if (!found_first_crossing && best_distance2 < source_radius2 && best_distance2 > 0.0) {
        const SourcePosition critical_source = refine_nearest_critical_source(
            mapper, separation, mass_ratio, source, best_phase, phase_step);
        append_caustic_probe_image_seeds(
            point_magnifier, mapper, separation, mass_ratio, source, source_radius,
            critical_source, seeds);
    }

    // Phase 2: detect additional arc crossings not covered by Phase 1 seeds.
    // Scan in sequential phase order so that contiguous "inside" segments on
    // each caustic branch are identified.  Seeds are added only when a branch
    // transitions from outside to inside the source disk; this prevents the
    // O(samples) seed explosion that occurs when the source disk engulfs a
    // large arc of the caustic (each sample would otherwise add new fold seeds
    // because fold-image positions vary rapidly along the arc).
    //
    // Branch continuity across samples is maintained by greedy nearest-neighbour
    // matching of each sample's critical-curve roots to the previous sample's
    // root positions.  The polynomial has at most 4 roots so the matching is O(1).
    if (seeds.size() >= 5) {
        constexpr int kMaxBranches = 4;
        // prev_pos[i]: image-plane position of branch i at the previous sample.
        // Initialised far away so that the first sample establishes branch order.
        std::array<SourcePosition, kMaxBranches> prev_pos;
        prev_pos.fill({1.0e30, 1.0e30});
        std::array<bool, kMaxBranches> branch_inside;
        branch_inside.fill(false);

        for (int sample = 0; sample < samples; ++sample) {
            const double phi = phase_step * static_cast<double>(sample);
            const auto critical_sources =
                critical_sources_at_phase(mapper, separation, mass_ratio, phi);
            const int ncur = static_cast<int>(critical_sources.size());
            if (ncur == 0) {
                continue;
            }

            // Match current roots to branches by greedy nearest-neighbour in image space.
            std::array<int, kMaxBranches> assignment;
            assignment.fill(-1);
            std::array<bool, kMaxBranches> used;
            used.fill(false);
            for (int bi = 0; bi < kMaxBranches; ++bi) {
                double best_d2 = 1.0e60;
                int best_j = -1;
                for (int j = 0; j < ncur; ++j) {
                    if (used[j]) {
                        continue;
                    }
                    const double d2 = distance_squared(critical_sources[j], prev_pos[bi]);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_j = j;
                    }
                }
                if (best_j >= 0) {
                    assignment[bi] = best_j;
                    used[best_j] = true;
                }
            }

            // Update branch states and fire on outside→inside transitions.
            for (int bi = 0; bi < kMaxBranches; ++bi) {
                const int j = assignment[bi];
                if (j < 0) {
                    branch_inside[bi] = false;
                    continue;
                }
                const auto& cs = critical_sources[j];
                const double d2 = distance_squared(cs, source);
                const bool now_inside = (d2 < source_radius2 && d2 > 0.0);

                const bool add_arc_seed =
                    source_radius >= 2.0e-2 && now_inside && (sample % 20 == 0);
                if ((now_inside && !branch_inside[bi]) || add_arc_seed) {
                    // This branch just entered the source disk: add fold seeds.
                    append_caustic_probe_image_seeds(
                        point_magnifier, mapper, separation, mass_ratio, source, source_radius,
                        cs, seeds);
                }
                branch_inside[bi] = now_inside;
                prev_pos[bi] = cs;
            }
        }
    }
    append_boundary_probe_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius, seeds);
    append_interior_probe_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius, seeds);
    return seeds;
}

std::vector<SourcePosition> selected_point_images(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source)
{
    auto* phase=adaptive_flux_request();
    AdaptiveStageTimer root_timer(phase && phase->cost_trace?&phase->seed_root_seconds:nullptr);
    if(phase && phase->cost_trace)++phase->seed_root_calls;

    std::vector<SourcePosition> images;
    const auto candidates =
        point_magnifier.binary_image_candidates(separation, mass_ratio, source);
    images.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (candidate.residual * candidate.residual < 1.0e-12) {
            images.push_back(candidate.position);
        }
    }
    return images;
}

template <bool UseLimbDarkening, bool GuardJacobian, typename ImageMap>
double cartesian_image_area_impl(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    SourcePosition seed,
    double dy,
    int& yi,
    LegacyImageAreaScratch& scratch,
    int jacobian_sign,
    ClaimedCellRuns& claimed,
    double magnification_hint = 0.0,
    bool* foreign_contact = nullptr,
    std::int64_t* step_budget = nullptr)
{
    double countx = 0.0;
    double countall = 0.0;
    double dz2 = 99999999.9;
    const double incr = std::abs(dy);
    const double inv_incr = 1.0 / incr;
    double dx = incr;
    SourcePosition image = seed;
    double x0 = seed.x;
    // Integer lattice coordinates of the current sample, maintained
    // incrementally alongside image.x/image.y (per-cell llround calls were a
    // measurable fraction of the scan cost).  Seeds are lattice-snapped by
    // the caller, so the increments stay exact.
    std::int64_t cell_ix = 0;
    std::int64_t cell_iy = 0;
    if (!rounded_lattice_index(seed.x * inv_incr, cell_ix) ||
        !rounded_lattice_index(seed.y * inv_incr, cell_iy)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::int64_t row_start_ix = cell_ix;
    int ix_step = 1;
    const int iy_step = dy > 0.0 ? 1 : -1;
    const ClaimedCellRuns::Row* claimed_row = claimed.find_row(cell_iy);
    // Pending contiguous stretch of counted cells, flushed to the claimed
    // registry whenever adjacency breaks.  Flushing continuously (rather
    // than at function exit) also terminates self-wrapping fills: a fill
    // that loops around a ring-shaped image runs into its own claims.
    bool claim_active = false;
    std::int64_t claim_iy = 0;
    std::int64_t claim_lo = 0;
    std::int64_t claim_hi = 0;
    const auto flush_claim = [&]() {
        if (claim_active) {
            claimed.claim(claim_iy, claim_lo, claim_hi);
            if (claim_iy == cell_iy) {
                claimed_row = claimed.find_row(cell_iy);
            }
            claim_active = false;
        }
    };
    const auto add_claim_cell = [&](std::int64_t ix, std::int64_t iy) {
        if (claim_active && claim_iy == iy &&
            (ix == claim_hi + 1 || ix == claim_lo - 1 ||
             (ix >= claim_lo && ix <= claim_hi))) {
            claim_lo = std::min(claim_lo, ix);
            claim_hi = std::max(claim_hi, ix);
            return;
        }
        flush_claim();
        claim_active = true;
        claim_iy = iy;
        claim_lo = ix;
        claim_hi = ix;
    };
    const double source_radius2 = source_radius * source_radius;
    const double inv_source_radius2 = 1.0 / source_radius2;
    // The crossing location is purely geometric: use the same accurately
    // bracketed source-limb root for uniform, linear, and two-coefficient
    // brightness profiles.  Only the strip weight below depends on the
    // profile.
    const bool bennett_root_enabled =
        std::getenv("LCBININT_DIAGNOSTIC_DISABLE_BENNETT_LIMB") == nullptr;
    // Surface brightness at the source edge, weighting the sub-cell boundary
    // correction strips (which always sit adjacent to the limb).
    double edge_brightness = 1.0;
    const double* brightness_table = nullptr;
    if constexpr (UseLimbDarkening) {
        edge_brightness = finite_magnifier != nullptr ?
            finite_magnifier->limb_darkening_table_brightness(1.0) :
            source_surface_brightness(1.0, settings);
        if (finite_magnifier != nullptr) {
            brightness_table = finite_magnifier->limb_darkening_table_data();
        }
    }
    const auto brightness_at_distance2 = [&](double distance2) {
        if constexpr (!UseLimbDarkening) {
            (void) distance2;
            return 1.0;
        }
        const double normalized2 = std::clamp(
            distance2 * inv_source_radius2, 0.0, 1.0);
        if (brightness_table != nullptr && settings.limb_darkening_d != 0.0) {
            const int index = static_cast<int>(
                normalized2 * static_cast<double>(kLimbDarkeningTableSize) + 0.5);
            return brightness_table[std::clamp(
                index, 0, kLimbDarkeningTableSize)];
        }
        return source_surface_brightness(normalized2, settings);
    };
    // Second-order boundary correction state.  Each row is scanned rightward
    // from x0 and then leftward from x0 - incr, so the sample spatially
    // adjacent to a boundary crossing is usually the previous iteration
    // (dz2_last).  The one exception is the pair (x0, x0 - incr) straddling
    // the turnaround: the row-start sample is remembered separately so the
    // crossing between it and the first leftward sample can still be
    // corrected (thin rows start within one cell of their own edge often).
    double dz2_row_start = -1.0;
    bool at_run_start = true;
    bool first_left_pending = false;
    bool jac_ok_prev = true;
    // Rightmost/leftmost counted sample of the current row.  When a run exits
    // right after a claim jump, dz2_last belongs to the suppressed landing
    // sample, so the shipped xmin/xmax bookkeeping would leave the row extent
    // stale; anchoring on the last counted cell keeps the next row's start
    // (x0 = xmax of this row) on the image instead of at the far side of a
    // foreign claimed span.
    double last_right_inside_x = std::numeric_limits<double>::quiet_NaN();
    double last_left_inside_x = std::numeric_limits<double>::quiet_NaN();
    // Fraction of the crossing cell that lies inside the source, measured
    // from the inside sample toward the outside one.  r_in <= rho < r_out
    // confines t to [0, 1); the clamp guards floating-point edge cases only.
    auto crossing_fraction = [source_radius](double inside_dz2, double outside_dz2) {
        const double r_in = std::sqrt(inside_dz2);
        const double r_out = std::sqrt(outside_dz2);
        const double dr_mapped = r_out - r_in;
        return dr_mapped > 0.0
            ? std::clamp((source_radius - r_in) / dr_mapped, 0.0, 1.0)
            : 0.5;
    };
    const auto refined_crossing_fraction = [&]
        (double inside_x, double outside_x, double y, double initial) {
        if (!bennett_root_enabled) {
            return initial;
        }
        return refine_boundary_crossing_fraction(
            mapper, source, source_radius2,
            inside_x, outside_x, y, initial, incr);
    };
    std::int64_t guard = 0;
    const auto seed_evaluation =
        evaluate_lens_cell(mapper, seed.x, seed.y, source);
    const double seed_magnification =
        std::isfinite(seed_evaluation.jacobian) &&
            std::abs(seed_evaluation.jacobian) > 1.0e-15
        ? 1.0 / std::abs(seed_evaluation.jacobian)
        : 1.0e15;
    const double bins2 =
        static_cast<double>(std::max(settings.source_bins, 1)) *
        static_cast<double>(std::max(settings.source_bins, 1));
    // A high-magnification image occupies O(A * bins^2) lattice cells.  The
    // old fixed 2000*bins^2 guard silently truncated otherwise healthy image
    // walks once A reached a few thousand.  Scale only the safety guard from
    // the seed Jacobian; the walk still terminates naturally at the image
    // boundary, so ordinary cases perform exactly the same amount of work.
    const double component_magnification_bound = std::max(
        seed_magnification,
        std::isfinite(magnification_hint) ? std::abs(magnification_hint) : 0.0);
    const double estimated_step_limit = bins2 * std::max(
        2000.0, 4.0 * kPi * component_magnification_bound);
    const std::int64_t max_steps = std::max<std::int64_t>(
        100000,
        estimated_step_limit < static_cast<double>(std::numeric_limits<std::int64_t>::max())
            ? static_cast<std::int64_t>(std::ceil(estimated_step_limit))
            : std::numeric_limits<std::int64_t>::max());
    // A caller that already knows how many cells the walk is supposed to cover
    // -- the per-component refinement below does -- caps it here.  Unlike
    // `max_steps`, which is a safety net for an otherwise healthy walk,
    // exhausting the budget is an expected answer ("this is not the component
    // you asked for"), so it is not reported as a numerical failure.
    const std::int64_t effective_max_steps = step_budget != nullptr
        ? std::min(max_steps, std::max<std::int64_t>(*step_budget, 1))
        : max_steps;
    scratch.ensure(static_cast<std::size_t>(yi));
    while (++guard < effective_max_steps) {
        const double dz2_last = dz2;
        const bool jac_ok_last = jac_ok_prev;
        const bool is_run_start = at_run_start;
        const bool is_first_left = first_left_pending;
        at_run_start = false;
        first_left_pending = false;
        const auto* interval = ClaimedCellRuns::find_in_row(
            claimed_row, cell_ix);
        if (interval != nullptr) {
                // Copy the bounds out before flush_claim(): if the pending
                // claim lands in the same row (claim_iy == cell_iy), its
                // push_back into that row's interval vector can reallocate
                // the very vector `interval` points into, dangling it.
                const std::int64_t interval_lo = interval->lo;
                const std::int64_t interval_hi = interval->hi;
                if (foreign_contact != nullptr && interval->owner != claimed.owner) {
                    *foreign_contact = true;
                }
                // The interval was counted (inside the disk) by an earlier
                // fill; skip it and resume on the far side.  The seam is an
                // interior junction, not a source boundary, so crossing
                // corrections are suppressed on the landing sample.  Guarded
                // fills jump too: the parity guard is evaluated per sample,
                // so the landing cell decides whether the run continues.
                flush_claim();
                cell_ix = ix_step > 0 ? interval_hi : interval_lo;
                image.x = static_cast<double>(cell_ix) * incr;
                dz2 = source_radius2 + 1.0;
                jac_ok_prev = false;
                image.x += dx;
                cell_ix += ix_step;
                continue;
        }
        double mapped_distance2 = 0.0;
        bool jac_ok = true;
        if constexpr (!GuardJacobian) {
            mapped_distance2 = mapped_lens_distance2(
                mapper, image.x, image.y, source);
        } else {
            // When a Jacobian-sign guard is active, treat pixels on the wrong side of the
            // critical curve as outside even if the mapped source is inside the disk.  This
            // prevents a fold-image flood-fill from bleeding across the critical curve into
            // the adjacent fold image on the opposite parity, which would otherwise cause
            // wildly wrong (sometimes negative) magnifications.  The mapped source and
            // Jacobian share the same lens denominators, so compute them together.
            const auto eval = evaluate_lens_cell(
                mapper, image.x, image.y, source);
            mapped_distance2 = eval.mapped_distance2;
            const int eval_sign = eval.jacobian > 0.0 ? 1 : eval.jacobian < 0.0 ? -1 : 0;
            jac_ok = eval_sign != -jacobian_sign;
        }
        dz2 = (jac_ok) ? mapped_distance2 : source_radius2 + 1.0;
        jac_ok_prev = jac_ok;

        if (dz2 <= source_radius2) {
            // Entering the disk while scanning left with nothing counted yet:
            // the previous (adjacent) sample is the outside side of the row's
            // right edge, so correct that crossing here.
            const bool entering_from_right = dx == -incr && countx == 0.0;
            if (entering_from_right) {
                scratch.xmax[static_cast<std::size_t>(yi)] = image.x - dx;
            }
            const double normalized_radius2 = mapped_distance2 * inv_source_radius2;
            double brightness = 1.0;
            if constexpr (UseLimbDarkening) {
                if (brightness_table != nullptr && settings.limb_darkening_d != 0.0) {
                    const int index = static_cast<int>(
                        normalized_radius2 *
                        static_cast<double>(kLimbDarkeningTableSize) + 0.5);
                    brightness = brightness_table[std::clamp(
                        index, 0, kLimbDarkeningTableSize)];
                } else {
                    brightness = source_surface_brightness(
                        normalized_radius2, settings);
                }
            }
            if (entering_from_right && jac_ok_last && dz2_last > source_radius2) {
                const double initial_delta =
                    crossing_fraction(mapped_distance2, dz2_last);
                const double delta = refined_crossing_fraction(
                    image.x, image.x + incr, image.y, initial_delta);
                countx += detail::bennett_boundary_residual(
                    delta, brightness, edge_brightness,
                    settings.bennett_delta_c);
            }
            countx += brightness;
            if (dx > 0.0) {
                last_right_inside_x = image.x;
            } else {
                last_left_inside_x = image.x;
            }
            add_claim_cell(cell_ix, cell_iy);
            if (is_run_start && dx == incr) {
                dz2_row_start = mapped_distance2;
            }
        } else {
            // Second-order boundary correction.  The scan only learns the edge
            // position once a sample maps outside the disk; linearly
            // interpolating the mapped radial distance between the adjacent
            // inside sample and this outside sample locates the crossing at
            // fraction t of the step.  Midpoint counting implicitly covered
            // 0.5 cells past the inside sample, so the residual strip is
            // (t - 0.5) cells.
            double edge_correction = 0.0;
            if (jac_ok && dz2_last <= source_radius2) {
                const double initial_delta =
                    crossing_fraction(dz2_last, mapped_distance2);
                const double inside_x = image.x - dx;
                const double delta = refined_crossing_fraction(
                    inside_x, image.x, image.y, initial_delta);
                edge_correction = detail::bennett_boundary_residual(
                    delta, brightness_at_distance2(dz2_last), edge_brightness,
                    settings.bennett_delta_c);
                countx += edge_correction;
            } else if (jac_ok && is_first_left && dz2_row_start >= 0.0) {
                // First sample left of the turnaround is outside while the
                // row-start sample was inside: the crossing between them is
                // not visible through dz2_last (which holds the right-edge
                // exit sample), so use the remembered row-start distance.
                const double initial_delta =
                    crossing_fraction(dz2_row_start, mapped_distance2);
                const double delta = refined_crossing_fraction(
                    x0, image.x, image.y, initial_delta);
                edge_correction = detail::bennett_boundary_residual(
                    delta, brightness_at_distance2(dz2_row_start), edge_brightness,
                    settings.bennett_delta_c);
                countx += edge_correction;
            }
            if (dx == incr) {
                if (dz2_last <= source_radius2) {
                    scratch.xmax[static_cast<std::size_t>(yi)] = image.x;
                } else if (!std::isnan(last_right_inside_x)) {
                    scratch.xmax[static_cast<std::size_t>(yi)] =
                        last_right_inside_x + incr;
                }
                dx = -incr;
                ix_step = -1;
                image.x = x0;
                cell_ix = row_start_ix;
                scratch.xmin[static_cast<std::size_t>(yi)] = image.x + dx;
                first_left_pending = true;
            } else {
                if (dz2_last <= source_radius2) {
                    scratch.xmin[static_cast<std::size_t>(yi)] = image.x;
                } else if (!std::isnan(last_left_inside_x)) {
                    scratch.xmin[static_cast<std::size_t>(yi)] =
                        last_left_inside_x - incr;
                }
                if (yi != 0 && countx == 0.0) {
                    scratch.ensure(static_cast<std::size_t>(yi - 1));
                    if (image.x >= scratch.xmin[static_cast<std::size_t>(yi - 1)] - dx) {
                        image.x += dx;
                        cell_ix += ix_step;
                        continue;
                    }
                }

                countall += countx;
                scratch.ax[static_cast<std::size_t>(yi)] = countx;
                scratch.y[static_cast<std::size_t>(yi)] = image.y;
                scratch.dys[static_cast<std::size_t>(yi)] = dy;
                if (countx == 0.0) {
                    scratch.dys[static_cast<std::size_t>(yi)] = -dy;
                    break;
                }

                ++yi;
                scratch.ensure(static_cast<std::size_t>(yi));
                dx = incr;
                ix_step = 1;
                x0 = scratch.xmax[static_cast<std::size_t>(yi - 1)];
                if (!rounded_lattice_index(x0 * inv_incr, row_start_ix)) {
                    flush_claim();
                    return std::numeric_limits<double>::quiet_NaN();
                }
                image.x = x0 - dx;
                cell_ix = row_start_ix - 1;
                image.y += dy;
                cell_iy += iy_step;
                claimed_row = claimed.find_row(cell_iy);
                countx = 0.0;
                at_run_start = true;
                dz2_row_start = -1.0;
                last_right_inside_x = std::numeric_limits<double>::quiet_NaN();
                last_left_inside_x = std::numeric_limits<double>::quiet_NaN();
            }
        }
        image.x += dx;
        cell_ix += ix_step;
    }

    flush_claim();
    if (step_budget != nullptr) {
        *step_budget = std::max<std::int64_t>(*step_budget - guard, 0);
    }
    if (guard >= effective_max_steps) {
        if (step_budget == nullptr && std::getenv("LCBININT_AREA_DIAGNOSTICS")) {
            std::fprintf(
                stderr,
                "CARTESIAN_WALK_EXHAUSTED bins=%d guard=%lld max_steps=%lld "
                "seed=(%.17g,%.17g) seed_mag=%.17g mag_hint=%.17g yi=%d count=%.17g\n",
                settings.source_bins,
                static_cast<long long>(guard),
                static_cast<long long>(max_steps),
                seed.x,
                seed.y,
                seed_magnification,
                magnification_hint,
                yi,
                countall);
        }
        // Never turn an exhausted image walk into a plausible partial area.
        // The Jacobian-scaled guard above should make this exceptional; NaN
        // keeps the numerical failure explicit if its estimate is insufficient.
        return std::numeric_limits<double>::quiet_NaN();
    }
    return countall;
}

template <typename ImageMap>
double cartesian_image_area(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    SourcePosition seed,
    double dy,
    int& yi,
    LegacyImageAreaScratch& scratch,
    int jacobian_sign,
    ClaimedCellRuns& claimed,
    double magnification_hint = 0.0,
    bool* foreign_contact = nullptr,
    std::int64_t* step_budget = nullptr)
{
    const bool use_limb_darkening =
        settings.limb_darkening_c != 0.0 || settings.limb_darkening_d != 0.0;
    if (jacobian_sign == 0) {
        if (!use_limb_darkening) {
            return cartesian_image_area_impl<false, false>(
                mapper, source, source_radius, settings, finite_magnifier, seed, dy, yi,
                scratch, jacobian_sign, claimed, magnification_hint, foreign_contact,
                step_budget);
        }
        return cartesian_image_area_impl<true, false>(
            mapper, source, source_radius, settings, finite_magnifier, seed, dy, yi,
            scratch, jacobian_sign, claimed, magnification_hint, foreign_contact,
            step_budget);
    }
    if (!use_limb_darkening) {
        return cartesian_image_area_impl<false, true>(
            mapper, source, source_radius, settings, finite_magnifier, seed, dy, yi,
            scratch, jacobian_sign, claimed, magnification_hint, foreign_contact,
            step_budget);
    }
    return cartesian_image_area_impl<true, true>(
        mapper, source, source_radius, settings, finite_magnifier, seed, dy, yi,
        scratch, jacobian_sign, claimed, magnification_hint, foreign_contact,
        step_budget);
}

const std::vector<TripleImageCandidate>& selected_triple_point_candidates(
    const PointSourceMagnifier& point_magnifier,
    const model::TripleLensGeometry& geometry,
    SourcePosition source)
{
    return point_magnifier.triple_image_candidates_cached(geometry, source);
}

std::vector<SourcePosition> augmented_triple_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    const TripleCausticBranches& caustics,
    bool* support_proven = nullptr);

double inverse_ray_polar_triple_mag(
    const PointSourceMagnifier& point_magnifier,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<SourcePosition>* precomputed_seeds = nullptr,
    PolarAreaDiagnostics* diagnostics = nullptr)
{
    // Centre images alone are not sufficient for a triple lens.  A tiny fold
    // component can intersect the finite source without producing a physical
    // image at its centre, and the polar flood fill then converges cleanly to
    // an incomplete area at every resolution.  Reuse the same caustic and
    // boundary probe seeds as the Cartesian integrator so every finite-source
    // image component has an interior starting cell.
    std::vector<SourcePosition> computed_images;
    if (precomputed_seeds == nullptr) {
        const auto& caustics = cached_triple_caustic_branches(
            geometry, settings.caustic_bins);
        computed_images = augmented_triple_image_seeds(
            point_magnifier, geometry, source, source_radius, caustics);
    }
    const auto& image_positions =
        precomputed_seeds == nullptr ? computed_images : *precomputed_seeds;
    const auto mapper = make_triple_lens_mapper(geometry);
    return inverse_ray_polar_core(
        mapper, image_positions, source, source_radius, settings, finite_magnifier,
        diagnostics);
}

// Seed-set half of a triple probe, split out so the certified stage can read
// the image count off the same solve it seeds from.
void append_triple_probe_images_as_seeds(
    const TripleLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const std::vector<TripleImageCandidate>& probe_images,
    std::vector<SourcePosition>& seeds)
{
    const double source_radius2 = source_radius * source_radius;
    const std::size_t n_seeds_before = seeds.size();
    for (const auto& candidate : probe_images) {
        if (!candidate.physical) {
            continue;
        }
        const SourcePosition image = candidate.position;
        const double mapped_distance2 =
            mapped_triple_lens_distance2(mapper, image.x, image.y, source);
        if (mapped_distance2 > source_radius2 * (1.0 + 1.0e-8)) {
            continue;
        }
        bool duplicate = false;
        for (std::size_t si = 0; si < n_seeds_before; ++si) {
            if (distance_squared(image, seeds[si]) < 0.0625 * source_radius2) {
                const double existing_distance2 =
                    mapped_triple_lens_distance2(mapper, seeds[si].x, seeds[si].y, source);
                if (mapped_distance2 + 1.0e-16 < existing_distance2) {
                    seeds[si] = image;
                }
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            seeds.push_back(image);
        }
    }
}

void append_valid_triple_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const TripleLensMapper& mapper,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    SourcePosition probe_source,
    std::vector<SourcePosition>& seeds)
{
    const double source_radius2 = source_radius * source_radius;
    if (source_radius <= 0.0 ||
        distance_squared(probe_source, source) >= source_radius2 * (1.0 + 1.0e-10)) {
        return;
    }
    append_triple_probe_images_as_seeds(
        mapper, source, source_radius,
        selected_triple_point_candidates(point_magnifier, geometry, probe_source),
        seeds);
}

// Triple counterpart of append_certified_component_seeds.  The certificate is
// lens-agnostic, so the only thing that differs is which root solver reports
// the image count.
bool append_certified_triple_component_seeds(
    const PointSourceMagnifier& point_magnifier,
    const TripleLensMapper& mapper,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    const DiskSupport& support,
    std::vector<SourcePosition>& seeds)
{
    const double source_radius2 = source_radius * source_radius;
    return resolve_certified_probes(support, [&](SourcePosition probe) {
        if (distance_squared(probe, source) >= source_radius2 * (1.0 + 1.0e-10)) {
            return -1;
        }
        const auto& probe_images =
            selected_triple_point_candidates(point_magnifier, geometry, probe);
        append_triple_probe_images_as_seeds(
            mapper, source, source_radius, probe_images, seeds);
        return static_cast<int>(std::count_if(
            probe_images.begin(), probe_images.end(),
            [](const auto& candidate) { return candidate.physical; }));
    });
}

void append_triple_caustic_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const TripleLensMapper& mapper,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    SourcePosition caustic_source,
    std::vector<SourcePosition>& seeds)
{
    const double dx = caustic_source.x - source.x;
    const double dy = caustic_source.y - source.y;
    const double distance = std::hypot(dx, dy);
    if (distance <= 0.0 || distance >= source_radius * 1.15) {
        return;
    }

    const double ux = dx / distance;
    const double uy = dy / distance;
    const double steps[] = {
        0.02 * source_radius,
        0.05 * source_radius,
        0.15 * source_radius,
        0.35 * source_radius,
    };
    for (const double step : steps) {
        const SourcePosition probes[2] = {
            {caustic_source.x + ux * step, caustic_source.y + uy * step},
            {caustic_source.x - ux * step, caustic_source.y - uy * step},
        };
        for (const auto& probe_source : probes) {
            append_valid_triple_probe_image_seeds(
                point_magnifier, mapper, geometry, source, source_radius,
                probe_source, seeds);
        }
    }
}

void append_triple_boundary_probe_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const TripleLensMapper& mapper,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    std::vector<SourcePosition>& seeds)
{
    if (source_radius <= 0.0) {
        return;
    }
    // Always probe all boundary positions regardless of current seed count.
    // The boundary ring is the only guaranteed way to find fold images whose
    // preimage lies near the disk edge but far from any sampled caustic segment.
    // Caustic probes may push past the old cap=64 gate and skip boundary coverage,
    // causing systematic V-notch artifacts where a fold image component is missed.
    //
    // For fold images (|J| << 1), the image is stretched in the image plane by 1/|J|.
    // Adjacent probe positions at angular step dφ sample the same fold component at
    // image-plane positions separated by ~(dφ * rho) / |J|, which can vastly exceed
    // the standard 0.25*rho dedup radius.  We use an adaptive dedup radius that scales
    // with 1/|J| among seeds added by the boundary phase itself, while keeping the
    // tight 0.25*rho dedup against pre-existing (center + caustic) seeds.
    constexpr int samples = 64;
    constexpr double inward_fraction = 0.02;
    // Angular step between adjacent probes on the boundary circle.
    constexpr double kProbeAngStep = 2.0 * kPi / static_cast<double>(samples);
    // Safety multiplier: ensure we catch the full span between adjacent probes.
    constexpr double kAdaptiveSafety = 24.0;
    const double probe_radius = source_radius * (1.0 - inward_fraction);
    const double source_radius2 = source_radius * source_radius;
    // Pre-boundary seeds are deduped with the standard tight radius.
    const double dedup_pre2 = 0.0625 * source_radius2;  // (0.25*rho)^2
    // Track where boundary-added seeds start so we can use the adaptive radius
    // only for same-phase comparisons.
    const std::size_t n_pre_boundary = seeds.size();
    for (int i = 0; i < samples; ++i) {
        const double phi = kProbeAngStep * static_cast<double>(i);
        const SourcePosition probe_source {
            source.x + probe_radius * std::cos(phi),
            source.y + probe_radius * std::sin(phi),
        };
        const auto& probe_images =
            selected_triple_point_candidates(point_magnifier, geometry, probe_source);
        for (const auto& candidate : probe_images) {
            if (!candidate.physical) {
                continue;
            }
            const SourcePosition image = candidate.position;
            const auto evaluation =
                evaluate_triple_lens_cell_strict(mapper, image.x, image.y, source);
            const double mapped2 = evaluation.mapped_distance2;
            if (mapped2 > source_radius2 * (1.0 + 1.0e-8)) {
                continue;
            }
            // Adaptive dedup radius for the boundary phase.
            // For a fold image with |J|, adjacent probe positions sample the same
            // component at image-plane separation ~(kProbeAngStep * rho) / |J|.
            // Using kAdaptiveSafety times that as the boundary dedup radius ensures
            // they are merged into one seed rather than spawning many redundant ones.
            const double abs_J = std::abs(evaluation.jacobian);
            double dedup_boundary2;
            if (abs_J > 1.0e-9) {
                const double d = kAdaptiveSafety * kProbeAngStep * source_radius / abs_J;
                dedup_boundary2 = d * d;
            } else {
                dedup_boundary2 = dedup_pre2;
            }
            // Never let the boundary radius go below the pre-boundary radius.
            if (dedup_boundary2 < dedup_pre2) {
                dedup_boundary2 = dedup_pre2;
            }

            bool duplicate = false;
            // Check against pre-existing (center + caustic) seeds: tight radius.
            for (std::size_t si = 0; si < n_pre_boundary && !duplicate; ++si) {
                if (distance_squared(image, seeds[si]) < dedup_pre2) {
                    const double existing_mapped2 =
                        mapped_triple_lens_distance2(mapper, seeds[si].x, seeds[si].y, source);
                    if (mapped2 + 1.0e-16 < existing_mapped2) {
                        seeds[si] = image;
                    }
                    duplicate = true;
                }
            }
            // Check against other boundary-added seeds: adaptive (larger) radius.
            for (std::size_t si = n_pre_boundary; si < seeds.size() && !duplicate; ++si) {
                if (distance_squared(image, seeds[si]) < dedup_boundary2) {
                    duplicate = true;
                }
            }
            if (!duplicate) {
                seeds.push_back(image);
            }
        }
    }
}

SourcePosition closest_point_on_segment(
    SourcePosition point,
    SourcePosition start,
    SourcePosition end)
{
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double length2 = dx * dx + dy * dy;
    if (length2 == 0.0) {
        return start;
    }
    const double t = std::clamp(
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / length2,
        0.0,
        1.0);
    return {start.x + t * dx, start.y + t * dy};
}

std::vector<SourcePosition> augmented_triple_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    const TripleCausticBranches& caustics,
    bool* support_proven)
{
    if (support_proven != nullptr) {
        *support_proven = true;
    }
    const TripleLensMapper mapper = make_triple_lens_mapper(geometry);
    const auto& centre_candidates =
        selected_triple_point_candidates(point_magnifier, geometry, source);
    std::vector<SourcePosition> seeds;
    seeds.reserve(centre_candidates.size());
    for (const auto& candidate : centre_candidates) {
        if (candidate.physical) {
            seeds.push_back(candidate.position);
        }
    }
    if (source_radius <= 0.0) {
        return seeds;
    }

    const double source_radius2 = source_radius * source_radius;
    const double seed_distance2 = 1.35 * source_radius2;
    const double caustic_probe_distance2 =
        (1.15 * source_radius) * (1.15 * source_radius);
    struct CausticSeedCandidate {
        SourcePosition point;
        double distance2;
    };
    std::vector<CausticSeedCandidate> caustic_candidates;
    DiskSupport raw_support;
    std::vector<double> vertex_radius;
    const bool certificate_stats = probe_stats_enabled();
    for (const auto& branch : caustics.branches) {
        if (branch.size() < 2) {
            continue;
        }
        // Gather the certificate's raw radial extrema during the caustic-seed
        // walk.  The former certify_disk_support call repeated this complete
        // O(Ncaustic) traversal immediately below; finalize_disk_support keeps
        // its exact sort/merge/probe and fingerprint rules unchanged.
        if (branch.size() >= 3) {
            const auto certificate_started = certificate_stats
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point {};
            const std::size_t count = branch.size();
            vertex_radius.assign(count, 0.0);
            for (std::size_t i = 0; i < count; ++i) {
                vertex_radius[i] = std::sqrt(distance_squared(branch[i], source));
                raw_support.min_caustic_distance = std::min(
                    raw_support.min_caustic_distance, vertex_radius[i]);
            }
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t previous = (i + count - 1) % count;
                const std::size_t next = (i + 1) % count;
                const bool minimum =
                    vertex_radius[i] <= vertex_radius[previous] &&
                    vertex_radius[i] < vertex_radius[next];
                const bool maximum =
                    vertex_radius[i] >= vertex_radius[previous] &&
                    vertex_radius[i] > vertex_radius[next];
                if (!minimum && !maximum) {
                    continue;
                }
                const SourcePosition curvature {
                    branch[previous].x - 2.0 * branch[i].x + branch[next].x,
                    branch[previous].y - 2.0 * branch[i].y + branch[next].y,
                };
                const double margin = std::hypot(curvature.x, curvature.y);
                if (vertex_radius[i] >= source_radius + margin) {
                    continue;
                }
                const SourcePosition chord {
                    branch[next].x - branch[previous].x,
                    branch[next].y - branch[previous].y,
                };
                const double chord_length = std::hypot(chord.x, chord.y);
                if (!(chord_length > 0.0)) {
                    continue;
                }
                DiskExtremum extremum;
                extremum.position = branch[i];
                extremum.tangent = {
                    chord.x / chord_length, chord.y / chord_length};
                extremum.normal = {-extremum.tangent.y, extremum.tangent.x};
                extremum.distance = vertex_radius[i];
                extremum.polyline_margin = margin;
                extremum.inside_disk = vertex_radius[i] < source_radius;
                raw_support.extrema.push_back(extremum);
            }
            if (certificate_stats) {
                probe_counters().certify_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - certificate_started)
                        .count();
            }
        }
        for (std::size_t i = 1; i <= branch.size(); ++i) {
            const SourcePosition start = branch[i - 1];
            const SourcePosition end = (i == branch.size()) ? branch.front() : branch[i];
            const SourcePosition caustic_point =
                closest_point_on_segment(source, start, end);
            // Reuse one squared distance for both the threshold and candidate
            // ordering.  This walk covers every cached caustic segment per
            // finite-source epoch; taking a square root and then recomputing
            // the squared value for accepted points was pure fixed overhead.
            const double caustic_distance2 =
                distance_squared(caustic_point, source);
            if (caustic_distance2 <= caustic_probe_distance2) {
                caustic_candidates.push_back({
                    caustic_point,
                    caustic_distance2,
                });
            }
            const double start_distance2 = distance_squared(start, source);
            if (start_distance2 <= seed_distance2) {
                caustic_candidates.push_back({start, start_distance2});
            }
        }
    }

    std::sort(
        caustic_candidates.begin(),
        caustic_candidates.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.distance2 < rhs.distance2;
        });
    constexpr std::size_t max_caustic_seed_candidates = 12;
    constexpr std::size_t max_triple_seeds = 64;
    std::size_t used_caustic_candidates = 0;
    for (const auto& candidate : caustic_candidates) {
        bool duplicate_candidate = false;
        for (std::size_t i = 0; i < used_caustic_candidates; ++i) {
            if (distance_squared(candidate.point, caustic_candidates[i].point) <
                0.0025 * source_radius2) {
                duplicate_candidate = true;
                break;
            }
        }
        if (duplicate_candidate) {
            continue;
        }
        append_triple_caustic_probe_image_seeds(
            point_magnifier, mapper, geometry, source, source_radius,
            candidate.point, seeds);
        caustic_candidates[used_caustic_candidates++] = candidate;
        if (used_caustic_candidates >= max_caustic_seed_candidates ||
            seeds.size() >= max_triple_seeds) {
            break;
        }
    }

    append_triple_boundary_probe_image_seeds(
        point_magnifier, mapper, geometry, source, source_radius, seeds);

    // Completeness stage; see augmented_image_seeds for why it runs last.
    {
        const auto support = finalize_disk_support(
            std::move(raw_support), source, source_radius);
        const bool proven = append_certified_triple_component_seeds(
            point_magnifier, mapper, geometry, source, source_radius, support, seeds);
        if (support_proven != nullptr) {
            *support_proven = proven;
        }
    }
    return seeds;
}


// Snap seeds onto the shared integration lattice (x = ix*incr, y = iy*incr)
// so that every fill of one epoch samples identical cell positions and the
// claimed-cell registry is exact.  A snapped seed must still map inside the
// source disk; otherwise the 8 lattice neighbours are tried and the seed is
// dropped if none qualifies (it marked a sub-cell image the lattice cannot
// resolve).  Seeds landing on the same cell are deduplicated.
//
// The legacy walker can optionally keep each snapped seed on its own side of
// the critical curve.  The run fill does not need that topological shortcut:
// its membership predicate is only the mapped source disk and its shared-cell
// registry handles duplicate coverage independently.  Keeping the switch here
// lets the legacy path retain its old behaviour while the run path removes the
// parity constraint entirely.
template <typename ImageMap>
std::vector<SourcePosition> lattice_snapped_seeds(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    double incr,
    const std::vector<SourcePosition>& seeds,
    bool preserve_jacobian_side = true)
{
    const double source_radius2 = source_radius * source_radius;
    std::vector<SourcePosition> snapped;
    snapped.reserve(seeds.size());
    std::unordered_set<LatticeCellIndex, LatticeCellIndexHash> taken;
    taken.reserve(seeds.size());
    const auto jacobian_sign_at = [&](double x, double y) {
        const double jacobian = lens_jacobian(mapper, x, y);
        return jacobian > 0.0 ? 1 : jacobian < 0.0 ? -1 : 0;
    };
    // `required_sign` of 0 accepts either side.  The sign check is only
    // active for the legacy walker; the run fill always passes zero here.
    const auto try_cell = [&]
        (std::int64_t ix, std::int64_t iy, int required_sign) {
        const SourcePosition cell {
            static_cast<double>(ix) * incr,
            static_cast<double>(iy) * incr};
        if (!preserve_jacobian_side || required_sign == 0) {
            if (mapped_lens_distance2(mapper, cell.x, cell.y, source) >
                source_radius2) {
                return false;
            }
        } else {
            const auto evaluation = evaluate_lens_cell(
                mapper, cell.x, cell.y, source);
            if (evaluation.mapped_distance2 > source_radius2 ||
                (evaluation.jacobian > 0.0 ? 1 :
                 evaluation.jacobian < 0.0 ? -1 : 0) != required_sign) {
                return false;
            }
        }
        if (taken.insert({ix, iy}).second) {
            snapped.push_back(cell);
        }
        return true;
    };
    const auto place = [&]
        (std::int64_t ix, std::int64_t iy, int required_sign) {
        if (try_cell(ix, iy, required_sign)) {
            return true;
        }
        for (std::int64_t dyc = -1; dyc <= 1; ++dyc) {
            for (std::int64_t dxc = -1; dxc <= 1; ++dxc) {
                if ((dxc != 0 || dyc != 0) &&
                    try_cell(ix + dxc, iy + dyc, required_sign)) {
                    return true;
                }
            }
        }
        return false;
    };
    for (const auto& seed : seeds) {
        std::int64_t ix = 0;
        std::int64_t iy = 0;
        if (!rounded_lattice_index(seed.x / incr, ix) ||
            !rounded_lattice_index(seed.y / incr, iy)) {
            return {};
        }
        const int seed_sign = preserve_jacobian_side
            ? jacobian_sign_at(seed.x, seed.y)
            : 0;
        if (place(ix, iy, seed_sign)) {
            continue;
        }
        // No neighbour reproduces the side; keep the seed rather than lose it.
        place(ix, iy, 0);
    }
    return snapped;
}

// |J| below which a seed counts as sitting on a fold and its fill is confined
// to one side of the critical curve.
constexpr double kFoldJacobianThreshold = 0.02;

// What one flood-filled image component contributed, and how well the lattice
// resolved it.
struct ComponentFill {
    double area = 0.0;      // cells, weighted by surface brightness
    int rows = 0;           // rows the scan visited, gap repairs included
    int boundary_rows = 0;  // rows carrying a sub-cell edge correction
    int rows_span = 0;      // distinct lattice rows the component spans
    double width = 0.0;     // mean row extent, in cells
    // True when the fill was stopped by cells another fill had already
    // counted, i.e. its extent was decided by a neighbour rather than by its
    // own boundary.
    bool foreign_contact = false;

    // Cells across the component's narrow direction.  The row scan resolves x
    // by sub-cell edge corrections and y by the midpoint rule over row widths,
    // so both degrade once one of the two directions is a few samples wide;
    // whichever is smaller is what limits this component.
    double narrow_cells() const
    {
        return std::min(static_cast<double>(rows_span), width);
    }
};

// Flood-fills the image component containing `seed` on the lattice of spacing
// `incr`, including the row-gap repairs that reconnect a component the vertical
// scan would otherwise leave in pieces.
//
// Split out of inverse_ray_cartesian_core so a component can be re-filled on a
// finer lattice of its own.  Everything here is a function of `incr`: the scan
// steps by +-incr and `cartesian_image_area` reads the walk guard off
// settings.source_bins, so a caller refining by k passes incr/k together with
// source_bins*k.
template <typename ImageMap>
ComponentFill fill_image_component(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    SourcePosition seed,
    double incr,
    LegacyImageAreaScratch& scratch,
    int jac_sign,
    ClaimedCellRuns& claimed,
    double magnification_hint,
    LegacyAreaDiagnostics* diagnostics,
    std::int64_t* step_budget = nullptr)
{
    scratch.reset();
    scratch.ensure(1);
    bool foreign_contact = false;
    double area0 = 0.0;
    double dy = incr;
    int yi = 0;

    scratch.xmin[0] = seed.x;
    scratch.xmax[0] = seed.x;
    double areai = cartesian_image_area(
        mapper, source, source_radius, settings, finite_magnifier, seed, dy, yi, scratch,
        jac_sign, claimed, magnification_hint, &foreign_contact, step_budget);

    dy = -incr;
    scratch.ensure(static_cast<std::size_t>(yi));
    const SourcePosition lower_seed {scratch.xmax[0], seed.y + dy};
    scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[0];
    scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[0];
    scratch.y[static_cast<std::size_t>(yi)] = scratch.y[0];
    scratch.dys[static_cast<std::size_t>(yi)] = dy;
    ++yi;
    areai += cartesian_image_area(
        mapper, source, source_radius, settings, finite_magnifier, lower_seed, dy, yi, scratch,
        jac_sign, claimed, magnification_hint, &foreign_contact, step_budget);

    int nyi = yi;
    double areabound = 0.0;
    for (int row = 0; row < nyi; ++row) {
        scratch.ensure(static_cast<std::size_t>(row + 1));
        const double dxmax =
            scratch.xmax[static_cast<std::size_t>(row + 1)] -
            scratch.xmax[static_cast<std::size_t>(row)];
        const double dxmin =
            scratch.xmin[static_cast<std::size_t>(row + 1)] -
            scratch.xmin[static_cast<std::size_t>(row)];
        if (diagnostics != nullptr && scratch.ax[static_cast<std::size_t>(row + 1)] > 0.0) {
            diagnostics->max_jump_cells = std::max(
                diagnostics->max_jump_cells,
                std::max(std::abs(dxmax), std::abs(dxmin)) / incr);
        }
        if (scratch.ax[static_cast<std::size_t>(row + 1)] > 0.0) {
            if (dxmax > 1.1 * incr) {
                if (diagnostics != nullptr) {
                    ++diagnostics->gap_repairs;
                }
                const SourcePosition extra_seed {
                    scratch.xmax[static_cast<std::size_t>(row + 1)],
                    scratch.y[static_cast<std::size_t>(row)]};
                scratch.ensure(static_cast<std::size_t>(yi));
                scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row)];
                scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row + 1)];
                dy = -scratch.dys[static_cast<std::size_t>(row)];
                scratch.dys[static_cast<std::size_t>(yi)] = dy;
                ++yi;
                area0 = cartesian_image_area(
                    mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy,
                    yi, scratch, jac_sign, claimed, magnification_hint, &foreign_contact,
                    step_budget);
                areai += area0;
                areabound += area0;
                if (area0 <= 0.0) {
                    --yi;
                }
            }
            if (dxmin > 1.1 * incr) {
                if (diagnostics != nullptr) {
                    ++diagnostics->gap_repairs;
                }
                const SourcePosition extra_seed {
                    scratch.xmin[static_cast<std::size_t>(row + 1)] - incr,
                    scratch.y[static_cast<std::size_t>(row + 1)]};
                scratch.ensure(static_cast<std::size_t>(yi));
                scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row)];
                scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row + 1)];
                dy = scratch.dys[static_cast<std::size_t>(row)];
                scratch.dys[static_cast<std::size_t>(yi)] = dy;
                ++yi;
                area0 = cartesian_image_area(
                    mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy,
                    yi, scratch, jac_sign, claimed, magnification_hint, &foreign_contact,
                    step_budget);
                areai += area0;
                areabound += area0;
                if (area0 <= 0.0) {
                    --yi;
                }
            }
            if (dxmin < -1.1 * incr) {
                if (diagnostics != nullptr) {
                    ++diagnostics->gap_repairs;
                }
                const SourcePosition extra_seed {
                    scratch.xmin[static_cast<std::size_t>(row)] - incr,
                    scratch.y[static_cast<std::size_t>(row)]};
                scratch.ensure(static_cast<std::size_t>(yi));
                scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row + 1)];
                scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmin[static_cast<std::size_t>(row)];
                dy = -scratch.dys[static_cast<std::size_t>(row)];
                scratch.dys[static_cast<std::size_t>(yi)] = dy;
                ++yi;
                area0 = cartesian_image_area(
                    mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy,
                    yi, scratch, jac_sign, claimed, magnification_hint, &foreign_contact,
                    step_budget);
                areai += area0;
                areabound += area0;
                if (area0 <= 0.0) {
                    --yi;
                }
            }
            if (dxmax < -1.1 * incr) {
                if (diagnostics != nullptr) {
                    ++diagnostics->gap_repairs;
                }
                const SourcePosition extra_seed {
                    scratch.xmax[static_cast<std::size_t>(row + 1)] + incr,
                    scratch.y[static_cast<std::size_t>(row + 1)]};
                scratch.ensure(static_cast<std::size_t>(yi));
                scratch.xmin[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row + 1)];
                scratch.xmax[static_cast<std::size_t>(yi)] = scratch.xmax[static_cast<std::size_t>(row)];
                dy = scratch.dys[static_cast<std::size_t>(row)];
                scratch.dys[static_cast<std::size_t>(yi)] = dy;
                ++yi;
                area0 = cartesian_image_area(
                    mapper, source, source_radius, settings, finite_magnifier, extra_seed, dy,
                    yi, scratch, jac_sign, claimed, magnification_hint, &foreign_contact,
                    step_budget);
                areai += area0;
                areabound += area0;
                if (area0 <= 0.0) {
                    --yi;
                }
            }
        }
        if (row == nyi - 1 && areabound > 0.0 && yi > nyi) {
            nyi = yi;
        }
    }

    ComponentFill fill;
    fill.area = areai;
    fill.rows = nyi;
    fill.foreign_contact = foreign_contact;
    double extent_cells = 0.0;
    int measured = 0;
    std::int64_t iy_min = std::numeric_limits<std::int64_t>::max();
    std::int64_t iy_max = std::numeric_limits<std::int64_t>::min();
    for (int row = 0; row < nyi; ++row) {
        scratch.ensure(static_cast<std::size_t>(row));
        if (scratch.ax[static_cast<std::size_t>(row)] > 0.0) {
            ++fill.boundary_rows;
        }
        const double extent =
            scratch.xmax[static_cast<std::size_t>(row)] -
            scratch.xmin[static_cast<std::size_t>(row)];
        if (!std::isfinite(extent) || extent < 0.0) {
            continue;
        }
        extent_cells += extent / incr + 1.0;
        ++measured;
        const double row_y = scratch.y[static_cast<std::size_t>(row)];
        if (std::isfinite(row_y)) {
            std::int64_t iy = 0;
            if (rounded_lattice_index(row_y / incr, iy)) {
                iy_min = std::min(iy_min, iy);
                iy_max = std::max(iy_max, iy);
            }
        }
    }
    fill.width = measured > 0 ? extent_cells / static_cast<double>(measured) : 0.0;
    // Gap repairs revisit rows, so `rows` over-counts the vertical extent; the
    // lattice row indices do not.
    const long double row_span = iy_max >= iy_min
        ? static_cast<long double>(iy_max) -
            static_cast<long double>(iy_min) + 1.0L
        : 0.0L;
    fill.rows_span = static_cast<int>(std::min<long double>(
        row_span, static_cast<long double>(1 << 24)));
    return fill;
}

// Per-component refinement.
//
// The uniform image-plane lattice is sized from the *source* (`rho / bins`),
// which says nothing about the images it has to resolve.  A fold pair near a
// tangency is a sliver — measured at 55:1 on the reference cusp geometry — so
// at `bins = 64` it is 2.1 cells across and 115 along, and the row scan is
// integrating a width it barely samples.  That is the whole of the residual
// h^1.7 order at the tangency: not a missing component, a component the grid
// cannot see across.
//
// Refining globally to fix it costs k^2 over the whole disk.  Refining the one
// component costs k^2 over the sliver, which is a fraction of a percent of the
// grid.  `narrow_cells` is the measurement that makes the choice, and it comes
// out of the coarse fill itself.
constexpr double kComponentRefineTrigger = 16.0;  // cells across
constexpr double kComponentRefineTarget = 32.0;   // cells across, after refining
constexpr int kComponentRefineMaxFactor = 32;
// Refined cells are capped at this many `bins^2`, so a component that is thin
// *and* long cannot take over the epoch it was meant to make cheaper.
constexpr double kComponentRefineCellBudget = 8.0;

int component_refinement_factor(const ComponentFill& fill, int source_bins)
{
    const double narrow = fill.narrow_cells();
    if (!(narrow > 0.0) || narrow >= kComponentRefineTrigger) {
        return 1;
    }
    int factor = std::min(
        kComponentRefineMaxFactor,
        static_cast<int>(std::ceil(kComponentRefineTarget / narrow)));
    // Odd only: a coarse cell then covers exactly the fine cells within
    // (k-1)/2 of its centre, so the two lattices partition the plane the same
    // way and a claim can be carried from one to the other without slack.
    factor += 1 - (factor & 1);
    const double coarse_cells = fill.width * static_cast<double>(fill.rows_span);
    const double budget = kComponentRefineCellBudget *
        static_cast<double>(source_bins) * static_cast<double>(source_bins);
    while (factor > 1 &&
           coarse_cells * static_cast<double>(factor) * static_cast<double>(factor) > budget) {
        factor -= 2;
    }
    return std::max(factor, 1);
}

// Carries a refined fill's footprint back to the coarse registry, and reports
// whether it was admissible to do so.
//
// Two things are needed of the projection.  The first is that a component
// thinner than a cell arrives at the coarse lattice in pieces -- at 32 bins the
// reference sliver breaks into four -- and each piece is seeded separately.
// Refined, every piece recovers the *whole* component, so without carrying the
// footprint back the component is counted once per piece.  With `factor` odd
// the coarse cell `i` covers exactly the fine cells within `(k-1)/2` of `i*k`,
// so the coarse cell is claimed iff its own centre cell was filled: no slack,
// and in particular nothing claimed across a critical curve on a fold
// partner's side.
//
// The second is the check that makes the first safe.  Components of f^-1(D)
// are disjoint, so a refined footprint that lands on cells another fill has
// already counted is not a refinement of anything: the coarse fill's extent was
// decided by cells it could not enter, and the refined walk, alone on an empty
// lattice, left through that seam and re-traced its neighbour.  Its area is
// already in the sum.  Rejecting it costs the refinement and never an
// over-count; accepting it doubled a 9000x magnification.
bool claim_refined_footprint(
    const ClaimedCellRuns& refined, int factor, ClaimedCellRuns& claimed)
{
    const auto divide_floor = [factor](std::int64_t value) {
        const std::int64_t quotient = value / factor;
        const std::int64_t remainder = value % factor;
        return quotient - (remainder < 0 ? 1 : 0);
    };
    const auto divide_ceil = [factor](std::int64_t value) {
        const std::int64_t quotient = value / factor;
        const std::int64_t remainder = value % factor;
        return quotient + (remainder > 0 ? 1 : 0);
    };
    for (const auto& row : refined.rows) {
        if (row.first % factor != 0) {
            continue;
        }
        const auto existing = claimed.rows.find(row.first / factor);
        if (existing == claimed.rows.end()) {
            continue;
        }
        for (const auto& run : row.second) {
            const std::int64_t lo = divide_ceil(run.lo);
            const std::int64_t hi = divide_floor(run.hi);
            for (const auto& owned : existing->second) {
                if (owned.owner != claimed.owner && owned.hi >= lo && owned.lo <= hi) {
                    return false;
                }
            }
        }
    }
    for (const auto& row : refined.rows) {
        if (row.first % factor != 0) {
            continue;
        }
        const std::int64_t coarse_iy = row.first / factor;
        for (const auto& run : row.second) {
            claimed.claim(coarse_iy, divide_ceil(run.lo), divide_floor(run.hi));
        }
    }
    return true;
}

struct EvaluatedCartesianSeed {
    SourcePosition position;
    double jacobian = 0.0;
};

struct RunCellState {
    double mapped_distance2 = std::numeric_limits<double>::quiet_NaN();
};

template <typename ImageMap>
detail::CartesianRunFillResult fill_cartesian_run_union(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<detail::CartesianLatticeSeed>& seeds,
    double incr,
    std::int64_t maximum_evaluations,
    std::size_t expected_rows,
    detail::CartesianRunFillTrace* trace = nullptr,
    const std::vector<detail::CartesianLatticeSeed>* coverage_probes = nullptr,
    bool certify_coverage=false,
    const std::vector<SourcePosition>* physical_support_seeds=nullptr,
    double row_spacing=0,
    const detail::CartesianRunFillResult* retained_rows=nullptr,int retained_row_factor=1,int retained_column_factor=1,
    const std::array<std::int64_t,2>* row_clip=nullptr,unsigned physical_tip_mask=3,int cap_weight_mode=0,
    const detail::CartesianRunFillResult* retained_fine_rows=nullptr)
{
    if(row_spacing==0)row_spacing=incr;
    if(certify_coverage && row_spacing!=incr)throw std::invalid_argument("rectangular adaptive grid does not support the square-grid coverage audit");
    auto* retained_work=adaptive_flux_request();
    const bool light_error=lightweight_flux_error_mode();
    const bool boundary_error=boundary_diagnostics_enabled();
    const double source_radius2 = source_radius * source_radius;
    const double inverse_source_radius2 = 1.0 / source_radius2;
    if (trace != nullptr) {
        trace->cell_spacing = incr;
    }
    const bool use_limb_darkening =
        settings.limb_darkening_c != 0.0 || settings.limb_darkening_d != 0.0;
    const double edge_brightness = use_limb_darkening
        ? (finite_magnifier != nullptr
              ? finite_magnifier->limb_darkening_table_brightness(1.0)
              : source_surface_brightness(1.0, settings))
        : 1.0;
    const bool bennett_root_enabled =
        std::getenv("LCBININT_DIAGNOSTIC_DISABLE_BENNETT_LIMB") == nullptr;
    const auto brightness_at_distance2 = [&](double distance2) {
        if (!use_limb_darkening) {
            return 1.0;
        }
        const double normalized2 = std::clamp(
            distance2 * inverse_source_radius2, 0.0, 1.0);
        if (settings.limb_darkening_d == 0.0) {
            return source_surface_brightness(normalized2, settings);
        }
        return finite_magnifier != nullptr
            ? finite_magnifier->limb_darkening_table_brightness(normalized2)
            : source_surface_brightness(normalized2, settings);
    };
    // A run contains at least one evaluated inside cell, so the evaluation
    // budget is also a strict upper bound on useful run storage.  Retain the
    // independent ceiling as the memory safety envelope for very large maps.
    constexpr std::size_t kRunMemorySafetyCeiling = 1U << 21;
    const std::size_t evaluation_run_ceiling = maximum_evaluations > 0
        ? static_cast<std::size_t>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(maximum_evaluations),
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())))
        : 0;

    std::optional<detail::CartesianCoverageDomain> coverage_domain;
    EnclosedBinaryGeometry coverage_geometry;
    if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
        if (certify_coverage && std::numeric_limits<double>::is_iec559 && std::fegetround()==FE_TONEAREST) {
            using I=math::OutwardInterval;
            coverage_geometry=enclosed_binary_geometry(mapper);
            const I cx=I(source.x)+coverage_geometry.shift,cy(source.y);
            const double b=math::up(std::sqrt((math::square(cx)+math::square(cy)).hi));
            const double radius=(I(2)*(I(std::abs(mapper.separation.real()))+I(b)+I(source_radius)+coverage_geometry.m1+coverage_geometry.m2+I(2))).hi;
            const double extent=std::ceil((I(radius)/I(incr)).hi);
            if (std::isfinite(extent) && extent>=0 && extent<1e9) {
                coverage_domain.emplace();
                std::int64_t aligned_extent=1;
                while (aligned_extent<extent) aligned_extent*=2;
                coverage_domain->extent=aligned_extent;
                struct Hash {
                    std::size_t operator()(const std::array<std::int64_t,4>& key) const {
                        std::size_t h=0;
                        for (auto v:key) h^=std::hash<std::int64_t>{}(v)+0x9e3779b9+(h<<6)+(h>>2);
                        return h;
                    }
                };
                struct MappingCache {
                    double a=0,m1=0,m2=0,shift=0,h=0,q=0;
                    std::unordered_map<std::array<std::int64_t,4>,std::array<double,4>,Hash> boxes;
                };
                static thread_local MappingCache mapping_cache;
                if (mapping_cache.a!=mapper.separation.real() || mapping_cache.m1!=mapper.m1 ||
                    mapping_cache.m2!=mapper.m2 || mapping_cache.shift!=mapper.source_origin_shift || mapping_cache.h!=incr || mapping_cache.q!=mapper.input_mass_ratio) {
                    mapping_cache.boxes.clear();mapping_cache.a=mapper.separation.real();
                    mapping_cache.m1=mapper.m1;mapping_cache.m2=mapper.m2;
                    mapping_cache.shift=mapper.source_origin_shift;mapping_cache.h=incr;mapping_cache.q=mapper.input_mass_ratio;
                }
                coverage_domain->outside=[&](std::int64_t lo,std::int64_t hi,std::int64_t bottom,std::int64_t top) {
                    const I x(static_cast<double>(lo)*incr,static_cast<double>(hi)*incr);
                    const I y(static_cast<double>(bottom)*incr,static_cast<double>(top)*incr);
                    const double a=mapper.separation.real();
                    if (lo==hi && bottom==top && y.zero() && (x.zero() || (x.lo==a && x.hi==a))) return true;
                    const std::array<std::int64_t,4> key{lo,hi,bottom,top};
                    std::array<double,4> bounds;
                    const auto found=mapping_cache.boxes.find(key);
                    if (found!=mapping_cache.boxes.end()) bounds=found->second;
                    else {
                        const I xa=x-I(a),y2=math::square(y);
                        const I inv1=I(1)/(math::square(xa)+y2),inv2=I(1)/(math::square(x)+y2);
                        const I fx=x-coverage_geometry.m1*xa*inv1-coverage_geometry.m2*x*inv2-coverage_geometry.shift;
                        const I fy=y-coverage_geometry.m1*y*inv1-coverage_geometry.m2*y*inv2;
                        bounds={fx.lo,fx.hi,fy.lo,fy.hi};
                        if (mapping_cache.boxes.size()<262144) mapping_cache.boxes.emplace(key,bounds);
                    }
                    if (!(std::isfinite(bounds[0]) && std::isfinite(bounds[1]) && std::isfinite(bounds[2]) && std::isfinite(bounds[3]))) return false;
                    const I d=math::square(I(bounds[0],bounds[1])-I(source.x))+math::square(I(bounds[2],bounds[3])-I(source.y));
                    return d.finite() && d.lo>math::square(I(source_radius)).hi;
                };
                if (physical_support_seeds && matching_seed_support(mapper,source,source_radius)) {
                    coverage_domain->fenced_regions=[&](const std::vector<std::array<std::int64_t,4>>& components) {
                        std::vector<ImageFenceBox> bounds;
                        for (const auto& b:components) bounds.push_back({b[0]*incr,b[1]*incr,b[2]*incr,b[3]*incr});
                        const auto fences=certified_image_fences(mapper,source,source_radius,incr,std::move(bounds),*physical_support_seeds);
                        std::vector<std::array<std::int64_t,4>> regions;
                        for (const auto& b:fences) regions.push_back({static_cast<std::int64_t>(std::floor((I(b[0])/I(incr)).lo)),
                            static_cast<std::int64_t>(std::ceil((I(b[1])/I(incr)).hi)),
                            static_cast<std::int64_t>(std::floor((I(b[2])/I(incr)).lo)),
                            static_cast<std::int64_t>(std::ceil((I(b[3])/I(incr)).hi))});
                        return regions;
                    };
                }
            }
        }
    }
    const auto run_fill = [&](auto& trace_sink) {
        const auto evaluate_cell = [&](std::int64_t ix, std::int64_t iy) {
            if(row_clip && (iy<(*row_clip)[0] || iy>(*row_clip)[1]))return RunCellState{std::numeric_limits<double>::infinity()};
            const double x = static_cast<double>(ix) * incr;
            const double y = static_cast<double>(iy) * row_spacing;
            if(auto* work=retained_work) {
                if (work->depth > 0) work->mapped.materialize();
                const std::array<double,2> key{x,y};const auto [saved,inserted]=work->mapped.access(key,work->new_maps<work->budget);
                if(!saved)throw AdaptiveFluxBudgetExhausted{};
                if(!inserted){++work->reused_maps;return RunCellState{*saved};}
                *saved=mapped_lens_distance2(mapper,x,y,source);++work->new_maps;return RunCellState{*saved};
            }
            return RunCellState {
                mapped_lens_distance2(mapper, x, y, source),
            };
        };
        const auto inside_cell = [&](const RunCellState& state) {
            return state.mapped_distance2 <= source_radius2;
        };
        const auto weight_cell = [&](std::int64_t, std::int64_t, const RunCellState& state) {
            if (!use_limb_darkening) {
                return 1.0;
            }
            const double normalized2 =
                state.mapped_distance2 * inverse_source_radius2;
            if (settings.limb_darkening_d == 0.0) {
                // Only accepted disk cells reach this hot path: normalized2
                // is nonnegative. Avoid a generic profile call and its
                // redundant upper clamp / two-coefficient dispatch.
                const double depth = std::max(0.0, 1.0 - normalized2);
                return 1.0 - settings.limb_darkening_c * (1.0 - std::sqrt(depth));
            }
            return finite_magnifier != nullptr
                ? finite_magnifier->limb_darkening_table_brightness(normalized2)
                : source_surface_brightness(normalized2, settings);
        };
        const auto diagnostic_boundary = [&](const detail::CartesianRun& run,
            const RunCellState& left_inside,
            const RunCellState& left_outside,
            const RunCellState& right_inside,
            const RunCellState& right_outside,
            const std::optional<RunCellState>& left_second,
            const std::optional<RunCellState>& right_second) {
#ifdef LCBININT_TRACE_ROW_BOUNDARY
            AdaptiveStageTimer boundary_timer(retained_work && retained_work->cost_trace?&retained_work->row_boundary_seconds:nullptr);
#endif

            detail::CartesianBoundaryContribution contribution;
            contribution.error_recorded=light_error;
            const bool estimate_edge=boundary_error&&(!light_error||use_limb_darkening);
            if (boundary_error && !light_error) contribution.diagnostic.emplace();
            const auto edge = [&](std::int64_t inside_ix,
                                  const RunCellState& inside,
                                  std::int64_t outside_ix,
                                  const RunCellState& outside,
                                  const std::optional<RunCellState>& second) {
                const double inside_x = static_cast<double>(inside_ix) * incr;
                const double outside_x = static_cast<double>(outside_ix) * incr;
                const double y = static_cast<double>(run.iy) * row_spacing;
                // The raw mapper gives NaN at an exact pole (0*infinity).
                // Only that known geometric pole is an infinite outside witness.
                const bool outside_is_pole = [&]() {
                    if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
                        return y == 0.0 && (outside_x == 0.0 || outside_x == mapper.separation.real());
                    } else {
                        for (std::size_t j=0;j<mapper.mass.size();++j)
                            if (mapper.mass[j]>0.0 && outside_x==mapper.lens_x[j] && y==mapper.lens_y[j]) return true;
                        return false;
                    }
                }();
                const double outside_distance2 = outside_is_pole
                    ? std::numeric_limits<double>::infinity() : outside.mapped_distance2;
                // A lens pole has +infinite mapped distance and is a valid
                // outside witness. Keep refining the finite interior segment.
                if (!std::isfinite(inside.mapped_distance2) ||
                    !(outside_distance2 > source_radius2) ||
                    !(inside.mapped_distance2 <= source_radius2) ||
                    !(outside_distance2 > source_radius2)) {
                    return 0.0;
                }
                const double inside_radius =
                    std::sqrt(std::max(inside.mapped_distance2, 0.0));
                const double outside_radius =
                    std::sqrt(std::max(outside_distance2, 0.0));
                const double mapped_step = outside_radius - inside_radius;
                const double initial = std::isfinite(mapped_step) && mapped_step > 0.0
                    ? std::clamp(
                        (source_radius - inside_radius) / mapped_step,
                        0.0, 1.0)
                    : 0.5;
                BoundaryBracketDiagnostic bracket;
                double limb_inverse_gradient=0;
                const double fraction = bennett_root_enabled
                    ? refine_boundary_crossing_fraction(
                        mapper, source, source_radius2,
                        inside_x, outside_x, y, initial, incr,
                        contribution.diagnostic ? &bracket : nullptr,
                        contribution.error_recorded&&use_limb_darkening&&settings.limb_darkening_d!=0?&limb_inverse_gradient:nullptr)
                    : initial;
                contribution.error_limb_density+=incr*limb_inverse_gradient/kPi;
                if (contribution.diagnostic) {
                    if (outside_ix<inside_ix) {
                        contribution.diagnostic->left_fraction=fraction;
                        contribution.diagnostic->left_bracket_lower=bracket.lower;
                        contribution.diagnostic->left_bracket_upper=bracket.upper;
                    } else {
                        contribution.diagnostic->right_fraction=fraction;
                        contribution.diagnostic->right_bracket_lower=bracket.lower;
                        contribution.diagnostic->right_bracket_upper=bracket.upper;
                    }
                    contribution.diagnostic->rounded_zero_edges+=bracket.rounded_zero;
                }
                if(contribution.error_recorded)contribution.error_fractions[outside_ix<inside_ix?0:1]=fraction;
                ++contribution.edges;
                const double node_value = brightness_at_distance2(inside.mapped_distance2);
                const auto boundary_weights=detail::bennett_boundary_weights(fraction,settings.bennett_delta_c);
                if (estimate_edge) {
                    auto& counters = probe_counters();
                    if (second && bennett_root_enabled &&
                        settings.limb_darkening_d == 0.0) {
                        const double second_dz2 = second->mapped_distance2;
                        ++counters.boundary_model_reused_evaluations;
                        auto model = second_dz2 <= source_radius2
                            ? (light_error
                                ? detail::bennett_linear_ld_endpoint_flux_residual(fraction,
                                    1-std::clamp(inside.mapped_distance2*inverse_source_radius2,0.,1.),
                                    1-std::clamp(second_dz2*inverse_source_radius2,0.,1.),
                                    settings.limb_darkening_c,boundary_weights)
                                : detail::bennett_boundary_model_residual(fraction,node_value,
                                    brightness_at_distance2(second_dz2),edge_brightness,settings.bennett_delta_c))
                            : detail::BennettBoundaryModelResidual {};
                        if(model.resolved && retained_work && retained_work->defect_correction) {
                            // Held-out third interior value validates the two-node
                            // Puiseux fit on this same lattice. Its interpolating
                            // t^(5/2) defect supplies model uncertainty, not a
                            // convergence comparison between different grids.
                            if(run.hi-run.lo<2)model.model_remainder+=std::abs(model.signed_residual);
                            else {
                                const auto third=evaluate_cell(inside_ix+(outside_ix<inside_ix?2:-2),run.iy);
                                if(!(std::isfinite(third.mapped_distance2)&&third.mapped_distance2<=source_radius2))model.model_remainder+=std::abs(model.signed_residual);
                                else {
                                    const double t=fraction+2;
                                    const double actual=settings.limb_darkening_c*std::sqrt(std::max(0.,1-third.mapped_distance2*inverse_source_radius2)/t);
                                    const double quadratic=(actual-model.sqrt_coefficient-model.three_half_coefficient*t)/2;
                                    const auto moments=detail::power_endpoint_quadrature_defects(fraction,boundary_weights.node);
                                    const double held_defect=quadratic*(moments[2]-(2*fraction+1)*moments[1]+fraction*(fraction+1)*moments[0]);
                                    model.signed_residual+=held_defect;
                                    model.model_remainder+=std::abs(held_defect);
                                }
                            }
                        }
                        if (model.resolved) {
                            ++counters.boundary_model_edges;
                            const double flux_residual = model.signed_residual*incr*row_spacing;
                            counters.boundary_model_signed_flux_residual += flux_residual;
                            counters.boundary_model_absolute_flux_residual += std::abs(flux_residual);
                            contribution.error_signed+=flux_residual;
                            contribution.error_absolute+=std::abs(flux_residual);
                            contribution.error_remainder+=model.model_remainder*incr*row_spacing;
                            if(contribution.diagnostic) {
                                contribution.diagnostic->signed_model_flux_residual+=flux_residual;
                                contribution.diagnostic->absolute_model_flux_residual+=std::abs(flux_residual);
                            }
                        } else {
                            ++counters.boundary_model_unresolved;
                            ++contribution.error_unresolved;
                            if(contribution.diagnostic) ++contribution.diagnostic->unresolved_edges;
                        }
                    } else {
                        ++counters.boundary_model_unresolved;
                        ++contribution.error_unresolved;
                        if(contribution.diagnostic) ++contribution.diagnostic->unresolved_edges;
                    }
                }
                return boundary_weights.limb*edge_brightness+(boundary_weights.node-1)*node_value;
            };
            if (run.lo == std::numeric_limits<std::int64_t>::min() ||
                run.hi == std::numeric_limits<std::int64_t>::max()) {
                contribution.valid = false;
                return contribution;
            }
            contribution.area += edge(
                run.lo, left_inside, run.lo - 1, left_outside,left_second);
            contribution.area += edge(
                run.hi, right_inside, run.hi + 1, right_outside,right_second);
            if(retained_work && retained_work->defect_correction && use_limb_darkening && settings.limb_darkening_d==0 && run.lo==run.hi && contribution.error_recorded) {
                // No second sample exists at either endpoint of a one-node run.
                // Fit finite parabolic depth between the two actual roots.
                const double dl=contribution.error_fractions[0],dr=contribution.error_fractions[1],width=dl+dr;
                const double raw=weight_cell(run.lo,run.iy,left_inside)+contribution.area;
                double correction=0,remainder=std::max(std::abs(raw),std::abs(width-raw));
                if constexpr(std::is_same_v<ImageMap,BinaryLensMapper>) {
                    if(dl*dr>64*std::numeric_limits<double>::epsilon()) {
                        const double x=run.lo*incr,y=run.iy*row_spacing;
                        const double depth=std::max(0.,1-left_inside.mapped_distance2*inverse_source_radius2);
                        double f1=1,f2=0,f3=0;
                        const auto pole=[&](double pos,double mass) {
                            const double dx=std::max({x-dl*incr-pos,pos-x-dr*incr,0.});
                            const double radius2=dx*dx+y*y;
                            f1+=mass/radius2;f2+=2*mass/(radius2*std::sqrt(radius2));f3+=6*mass/(radius2*radius2);
                        };
                        pole(mapper.separation.real(),mapper.m1);pole(0,mapper.m2);
                        const double extent=std::max(dl,dr);
                        const double f0=std::sqrt(std::max(0.,left_inside.mapped_distance2))+f1*incr*extent;
                        const double depth3=2*(f0*f3+3*f1*f2)*inverse_source_radius2*incr*incr*incr;
                        // D-P <= sup|D'''| |(t+dl)t(t-dr)|/6. The leading
                        // parabola exceeding this envelope also proves D>0.
                        const auto model=detail::finite_one_node_ld_model(dl,dr,depth,settings.limb_darkening_c,depth3);
                        if(model.resolved) {
                            correction=model.integral-raw;
                            remainder=model.uncertainty+
                                128*std::numeric_limits<double>::epsilon()*(1+std::abs(model.integral)+std::abs(raw));
                        }
                    }
                }
                auto& counters=probe_counters();counters.boundary_model_unresolved-=contribution.error_unresolved;
                counters.boundary_model_signed_flux_residual-=contribution.error_signed;
                counters.boundary_model_absolute_flux_residual-=contribution.error_absolute;
                contribution.error_unresolved=0;contribution.error_signed=correction*incr*row_spacing;
                contribution.error_absolute=std::abs(contribution.error_signed);contribution.error_remainder=remainder*incr*row_spacing;
                counters.boundary_model_signed_flux_residual+=contribution.error_signed;
                counters.boundary_model_absolute_flux_residual+=contribution.error_absolute;
            }
            return contribution;
        };
        // Diagnostic exact interior grouping. A failed interval enclosure simply
        // retains the original per-cell scan. No sampled endpoint can certify a block.
        std::function<std::optional<std::array<double,2>>(std::int64_t,std::int64_t,std::int64_t)> interior_block;
        if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
            if (retained_work && settings.source_bins>=64 && std::getenv("LCBININT_DIAGNOSTIC_INTERIOR_BLOCKS")) {
                const auto geometry=enclosed_binary_geometry(mapper);
                interior_block=[&,geometry](std::int64_t lo,std::int64_t hi,std::int64_t iy)->std::optional<std::array<double,2>> {
                    using I=math::OutwardInterval;
                    const I x(math::down(double(lo)*incr),math::up(double(hi)*incr));
                    const I y(math::down(double(iy)*row_spacing),math::up(double(iy)*row_spacing));
                    const I xa=x-I(mapper.separation.real());
                    if(!use_limb_darkening) {
                    // Mean-value enclosure avoids the dependency inflation of
                    // subtracting broad lens-map intervals near a cancelling image.
                    const I r1=math::square(xa)+math::square(y),r2=math::square(x)+math::square(y);
                    const I dx=I(1)+geometry.m1*(math::square(xa)-math::square(y))/math::square(r1)
                                  +geometry.m2*(math::square(x)-math::square(y))/math::square(r2);
                    const I dy=I(2)*y*(geometry.m1*xa/math::square(r1)+geometry.m2*x/math::square(r2));
                    if(!dx.finite() || !dy.finite())return std::nullopt;
                    const I cx(double(lo+16)*incr),ca=cx-I(mapper.separation.real());
                    const I ci1=I(1)/(math::square(ca)+math::square(y)),ci2=I(1)/(math::square(cx)+math::square(y));
                    const I fx=cx-geometry.m1*ca*ci1-geometry.m2*cx*ci2-geometry.shift-I(source.x);
                    const I fy=y-geometry.m1*y*ci1-geometry.m2*y*ci2-I(source.y);
                    const I centre_distance=math::square(fx)+math::square(fy);
                    const I derivative_norm=math::square(dx)+math::square(dy);
                    if(!centre_distance.finite() || !derivative_norm.finite())return std::nullopt;
                    const I bound=I(math::up(std::sqrt(centre_distance.hi)))+I(math::up(std::sqrt(derivative_norm.hi)))*I((x-cx).magnitude());
                    if (!bound.finite() || bound.hi>source_radius) return std::nullopt;
                    }
                    if (!use_limb_darkening) return std::array<double,2>{double(hi-lo+1),0};
                    if (settings.limb_darkening_d!=0) return std::nullopt;
                    // Refine only the missing odd samples. The even samples
                    // were mapped on the preceding dyadic lattice; a cache miss
                    // keeps the ordinary scan, rather than starting another probe.
                    // A cheap Lipschitz enclosure certifies the segment interior.
                    // |d lens_map/dx| <= 1 + sum m_j / min_segment |z-z_j|^2.
                    const double xl=double(lo)*incr,xr=double(hi)*incr,yy=double(iy)*row_spacing;
                    const auto centre=retained_work->mapped.access({double(lo+8)*incr,yy},false).first;
                    if(!centre || !std::isfinite(*centre) || *centre>=source_radius2)return std::nullopt;
                    const auto min_distance2=[&](double lens_x){const double dx=std::max({xl-lens_x,lens_x-xr,0.});return dx*dx+yy*yy;};
                    const double d1=min_distance2(mapper.separation.real()),d2=min_distance2(0);
                    if(!(d1>0 && d2>0))return std::nullopt;
                    const double lipschitz=1+mapper.m1/d1+mapper.m2/d2;
                    const double centre_radius=std::sqrt(*centre);
                    const double round_margin=64*std::numeric_limits<double>::epsilon()*(1+std::abs(xl)+std::abs(xr)+mapper.m1/std::sqrt(d1)+mapper.m2/std::sqrt(d2));
                    if(centre_radius+8*incr*lipschitz+round_margin>=source_radius)return std::nullopt;
                    std::array<double,12> brightness{},depth{};
                    for(int j=0;j<12;++j) {
                        const auto ix=lo-2+2*j;
                        const auto cached=retained_work->mapped.access({double(ix)*incr,double(iy)*row_spacing},false).first;
                        if(!cached || !std::isfinite(*cached) || *cached>=source_radius2)return std::nullopt;
                        const RunCellState state{*cached};
                        depth[j]=1-*cached*inverse_source_radius2;
                        brightness[j]=weight_cell(ix,iy,state);
                    }
                    double sum=0,error=0;
                    for(int j=1;j<=8;++j) {
                        sum+=brightness[j]; // retained even point
                        sum+=(-brightness[j-1]+9*brightness[j]+9*brightness[j+1]-brightness[j+2])/16;
                        // Leading cubic interpolation defect from the already
                        // available fourth difference; no additional map calls.
                        const double fourth=brightness[j-1]-4*brightness[j]+6*brightness[j+1]-4*brightness[j+2]+brightness[j+3];
                        error+=std::abs(fourth)*3/128;
                    }
                    const double allowance=retained_work->target/source_radius2*16/64;
                    if(!std::isfinite(sum) || !std::isfinite(error) || error>allowance)return std::nullopt;
                    return std::array<double,2>{sum,error*incr*row_spacing};
                };
            }
        }
        std::unordered_map<std::int64_t,std::vector<std::size_t>> retained_index;
        std::function<std::optional<detail::CartesianRetainedRun>(std::int64_t,std::int64_t)> retained_row;
        if(retained_rows && retained_work && retained_rows->run_flux_errors.size()==retained_rows->runs.size()) {
            for(std::size_t j=0;j<retained_rows->runs.size();++j)retained_index[retained_rows->runs[j].run.iy].push_back(j);
            std::int64_t retained_min_y=std::numeric_limits<std::int64_t>::max(),retained_max_y=std::numeric_limits<std::int64_t>::min();
            for(const auto& entry:retained_rows->runs){retained_min_y=std::min(retained_min_y,entry.run.iy);retained_max_y=std::max(retained_max_y,entry.run.iy);}
            const auto boundary_strip_row=[&](std::size_t j,std::int64_t ix,std::int64_t iy)->std::optional<detail::CartesianRetainedRun> {
                if(settings.limb_darkening_d!=0 || retained_rows->run_flux_errors[j][2]!=0)return std::nullopt;
                const auto& old=retained_rows->runs[j].run;
                const auto& fractions=retained_rows->run_error_fractions[j];
                const double coarse_h=incr*retained_column_factor;
                const double left=(old.lo-fractions[0])*retained_column_factor,right=(old.hi+fractions[1])*retained_column_factor;
                const auto lo=static_cast<std::int64_t>(std::ceil(left)),hi=static_cast<std::int64_t>(std::floor(right));
                if(lo>ix || hi<ix || lo>hi)return std::nullopt;
                const auto ls=evaluate_cell(lo,iy),rs=evaluate_cell(hi,iy),outside_l=evaluate_cell(lo-1,iy),outside_r=evaluate_cell(hi+1,iy);
                if(!inside_cell(ls)||!inside_cell(rs)||inside_cell(outside_l)||inside_cell(outside_r))return std::nullopt;
                detail::RetainedRowCore core;
                const auto value=[&](std::int64_t x)->std::optional<double> {
                    const auto state=evaluate_cell(x,iy);if(!inside_cell(state))return std::nullopt;
                    const double v=weight_cell(x,iy,state);if(!std::isfinite(v))return std::nullopt;return v;
                };
                if(retained_rows->run_cores.size()==retained_rows->runs.size() && retained_rows->run_cores[j]) {
                    core=*retained_rows->run_cores[j];
                    if(core.spacing<=0)return std::nullopt; // Interpolated row has no exact coarse core.
                } else {
                    constexpr int strip=4;
                    if(old.hi-old.lo<2*strip+8)return std::nullopt;
                    // Setup reads the coarse core once, two seven-point
                    // derivative stencils and both coarse/fine edge strips.
                    // If this requests at least as many samples as a complete
                    // fine row, retain the ordinary scan instead.
                    const auto span=old.hi-old.lo;
                    const auto setup_samples=(span-2*strip+1)+14+2*(strip+1)+2*(strip*retained_column_factor+1);
                    if(std::getenv("LCBININT_DIAGNOSTIC_STRIP_SETUP_GATE") && setup_samples>=span*retained_column_factor+1)return std::nullopt;
                    core.left=(old.lo+strip)*coarse_h;core.right=(old.hi-strip)*coarse_h;core.spacing=coarse_h;
                    // Both endpoints of every coarse core interval map inside
                    // the convex source disk. The linear-interpolation defect
                    // is at most h^2/8 sup|lens_map''|. Bound the derivative
                    // over the complete core once; subsequent refinements
                    // retain this same checked core and incur no new scan.
                    ++retained_work->strip_support_checks;
                    double map_defect=std::numeric_limits<double>::infinity();
                    if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
                        const double y=iy*row_spacing;
                        const auto pole_bound=[&](double lens,double mass) {
                            const double dx=lens<core.left?core.left-lens:(lens>core.right?lens-core.right:0);
                            const double d2=dx*dx+y*y;return mass/(d2*std::sqrt(d2));
                        };
                        map_defect=coarse_h*coarse_h/4*(pole_bound(mapper.separation.real(),mapper.m1)+pole_bound(0,mapper.m2));
                    }
                    map_defect*=1+128*std::numeric_limits<double>::epsilon();
                    const double margin=2*source_radius*map_defect+map_defect*map_defect+
                        128*std::numeric_limits<double>::epsilon()*(1+source_radius2);
                    if(!std::isfinite(margin) || margin>=source_radius2){++retained_work->strip_support_rejects;return std::nullopt;}
                    for(auto x=old.lo+strip;x<=old.hi-strip;++x) {
                        if(evaluate_cell(x*retained_column_factor,iy).mapped_distance2>=source_radius2-margin) {
                            ++retained_work->strip_support_rejects;return std::nullopt;
                        }
                    }
                    std::array<double,5> derivatives[2];double removed=0;
                    for(int side=0;side<2;++side) {
                        const auto anchor=side?old.hi-strip:old.lo+strip;
                        std::array<double,7> v;
                        for(int k=0;k<7;++k){const auto g=value((anchor+k-3)*retained_column_factor);if(!g)return std::nullopt;v[k]=*g;}
                        derivatives[side]=detail::retained_core_derivatives(v,coarse_h);
                        double trap=0;
                        for(int k=0;k<=strip;++k) {
                            const auto x=(side?old.hi-k:old.lo+k)*retained_column_factor;
                            const auto g=value(x);if(!g)return std::nullopt;
                            trap+=(k==0||k==strip?.5:1)*(*g);
                        }
                        const auto first=value((side?old.hi:old.lo)*retained_column_factor);if(!first)return std::nullopt;
                        const auto w=detail::bennett_boundary_weights(fractions[side],settings.bennett_delta_c);
                        removed+=coarse_h*(trap+(w.node-.5)*(*first)+w.limb*edge_brightness);
                    }
                    core.trapezoid_area=retained_rows->run_areas[j]*coarse_h-removed;
                    core.derivative1=derivatives[1][0]-derivatives[0][0];
                    core.derivative3=derivatives[1][1]-derivatives[0][1];
                    const double h2=coarse_h*coarse_h;
                    core.uncertainty=h2*(derivatives[0][2]+derivatives[1][2])/12+
                        h2*h2*(derivatives[0][3]+derivatives[1][3])/720+
                        h2*h2*h2*std::abs(derivatives[1][4]-derivatives[0][4])/30240;
                }
                if(!std::isfinite(core.uncertainty) || core.uncertainty*row_spacing>retained_work->target/(4*retained_rows->runs.size()))return std::nullopt;
                const auto a=static_cast<std::int64_t>(std::llround(core.left/incr)),b=static_cast<std::int64_t>(std::llround(core.right/incr));
                if(a<lo+3 || b>hi-3 || a>=b)return std::nullopt;
                double strip_sum=0;
                for(int side=0;side<2;++side) {
                    const auto begin=side?b:lo,end=side?hi:a;
                    for(auto x=begin;x<=end;++x){const auto g=value(x);if(!g)return std::nullopt;strip_sum+=(x==begin||x==end?.5:1)*(*g);}
                }
                const auto second_l=evaluate_cell(lo+1,iy),second_r=evaluate_cell(hi-1,iy);
                const auto boundary=diagnostic_boundary({iy,lo,hi},ls,outside_l,rs,outside_r,
                    std::optional<RunCellState>(second_l),std::optional<RunCellState>(second_r));
                if(!boundary.valid || boundary.error_unresolved)return std::nullopt;
                const auto modeled=detail::retained_core_fine_equivalent(core,incr);
                detail::CartesianRetainedRun saved;saved.run={iy,lo,hi};saved.core=core;
                saved.area=strip_sum+.5*(weight_cell(lo,iy,ls)+weight_cell(hi,iy,rs))+boundary.area+modeled[0]/incr;
                saved.errors={boundary.error_signed,boundary.error_absolute,double(boundary.error_unresolved),
                    boundary.error_remainder+modeled[1]*row_spacing,0};saved.fractions=boundary.error_fractions;
                if(!(std::isfinite(saved.area)&&saved.area>0))return std::nullopt;
                if(std::getenv("LCBININT_DIAGNOSTIC_AUDIT_BOUNDARY_STRIPS")) {
                    double exact=boundary.area;bool gap=false;
                    for(auto x=lo;x<=hi;++x){const auto state=evaluate_cell(x,iy);if(!inside_cell(state)){gap=true;continue;}exact+=weight_cell(x,iy,state);}
                    const double difference=std::abs(saved.area-exact)*incr*row_spacing;
                    ++retained_work->strip_audit_rows;retained_work->strip_audit_gap_rows+=gap;
                    retained_work->strip_audit_absolute_difference+=difference;
                    retained_work->strip_audit_max_difference=std::max(retained_work->strip_audit_max_difference,difference);
                }
                ++retained_work->retained_boundary_strip_rows;return saved;
            };
            retained_row=[&,retained_min_y,retained_max_y,boundary_strip_row](std::int64_t ix,std::int64_t iy)->std::optional<detail::CartesianRetainedRun>{
                if(iy%retained_row_factor) {
                    if(retained_row_factor!=2 || retained_column_factor!=1 || settings.limb_darkening_d!=0 ||
                        !retained_work->local_row_interpolation)return std::nullopt;
                    ++retained_work->row_interpolation_attempts;
                    const auto base=(iy-1)/2;
                    const auto reject=[&](int reason)->std::optional<detail::CartesianRetainedRun>{++retained_work->row_interpolation_outcomes[reason];return std::nullopt;};
                    std::array<std::size_t,6> indices;
                    std::array<double,6> areas,left,right;
                    for(int k=0;k<6;++k) {
                        const auto found=retained_index.find(base-2+k);
                        // Tip, gap and junction rows keep ordinary exact scans.
                        if(found==retained_index.end() || found->second.size()!=1)return reject(0);
                        const auto j=found->second.front();indices[k]=j;
                        const auto& r=retained_rows->runs[j].run;
                        if(retained_rows->run_flux_errors[j][2]!=0)return reject(1);
                        if(k>0) {
                            const auto& prev=retained_rows->runs[indices[k-1]].run;
                            if(r.lo>prev.hi || prev.lo>r.hi)return reject(1);
                        }
                        areas[k]=retained_rows->run_areas[j];
                        left[k]=r.lo-retained_rows->run_error_fractions[j][0];
                        right[k]=r.hi+retained_rows->run_error_fractions[j][1];
                        if(use_limb_darkening) {
                            // Interpolate the smooth LD moment, with the old
                            // Bennett grid-phase defect removed. The constant
                            // limb brightness times geometric width is exact
                            // once the new row's endpoints have been evaluated.
                            areas[k]+=retained_rows->run_flux_errors[j][0]/(incr*row_spacing*retained_row_factor);
                            areas[k]-=edge_brightness*(right[k]-left[k]);
                        }
                    }
                    const auto area=detail::retained_row_midpoint(areas);
                    const auto l=detail::retained_row_midpoint(left),r=detail::retained_row_midpoint(right);
                    if(!(std::isfinite(area[0]) && area[0]>=0 && std::isfinite(l[0]) && std::isfinite(r[0]) && r[0]>l[0]))return reject(2);
                    const double defect=std::max(area[1],(1-settings.limb_darkening_c)*(l[1]+r[1]))*incr*row_spacing;
                    // Reserve only a fraction of the global error budget for
                    // omitted scans; carried errors remain in the same sum.
                    if(!std::isfinite(defect) || defect>retained_work->target/(4*retained_rows->runs.size()))return reject(3);
                    const auto lo=static_cast<std::int64_t>(std::ceil(l[0]));
                    const auto hi=static_cast<std::int64_t>(std::floor(r[0]));
                    if(lo>ix || hi<ix || lo>hi)return reject(4);
                    // Actual endpoint brackets must agree with the predicted
                    // run. Four cheap classifications replace a whole scan.
                    if(!inside_cell(evaluate_cell(lo,iy)) || !inside_cell(evaluate_cell(hi,iy)) ||
                        inside_cell(evaluate_cell(lo-1,iy)) || inside_cell(evaluate_cell(hi+1,iy)))return reject(5);
                    detail::CartesianRetainedRun saved;saved.run={iy,lo,hi};saved.area=area[0];
                    saved.fractions={lo-l[0],r[0]-hi};
                    constexpr std::array<double,6> coefficients{0,-1./16,9./16,9./16,-1./16,0};
                    for(int k=0;k<6;++k) {
                        const auto& e=retained_rows->run_flux_errors[indices[k]];
                        saved.errors[0]+=coefficients[k]*e[0]/2;
                        for(int f:{1,3})saved.errors[f]+=std::abs(coefficients[k])*e[f]/2;
                    }
                    saved.errors[3]+=defect;
                    if(use_limb_darkening) {
                        const auto boundary=diagnostic_boundary({iy,lo,hi},evaluate_cell(lo,iy),evaluate_cell(lo-1,iy),
                            evaluate_cell(hi,iy),evaluate_cell(hi+1,iy),
                            std::optional<RunCellState>(evaluate_cell(lo+1,iy)),std::optional<RunCellState>(evaluate_cell(hi-1,iy)));
                        if(!boundary.valid || boundary.error_unresolved)return reject(2);
                        saved.fractions=boundary.error_fractions;
                        saved.area=area[0]+edge_brightness*(hi-lo+saved.fractions[0]+saved.fractions[1])-
                            boundary.error_signed/(incr*row_spacing);
                        saved.errors[0]=boundary.error_signed;saved.errors[1]=boundary.error_absolute;
                        saved.errors[3]+=boundary.error_remainder;
                        if(!(std::isfinite(saved.area)&&saved.area>0))return reject(2);
                    }
                    if(std::getenv("LCBININT_DIAGNOSTIC_AUDIT_INTERPOLATED_ROWS")) {
                        double exact=0;bool gap=false;
                        for(auto x=lo;x<=hi;++x){const auto state=evaluate_cell(x,iy);if(!inside_cell(state)){gap=true;continue;}exact+=weight_cell(x,iy,state);}
                        const auto boundary=diagnostic_boundary({iy,lo,hi},evaluate_cell(lo,iy),evaluate_cell(lo-1,iy),
                            evaluate_cell(hi,iy),evaluate_cell(hi+1,iy),
                            std::optional<RunCellState>(evaluate_cell(lo+1,iy)),std::optional<RunCellState>(evaluate_cell(hi-1,iy)));
                        exact+=boundary.area;
                        ++retained_work->interpolated_audit_rows;retained_work->interpolated_audit_gap_rows+=gap;
                        retained_work->interpolated_audit_max_difference=std::max(retained_work->interpolated_audit_max_difference,std::abs(saved.area-exact)*incr*row_spacing);
                    }
                    saved.core=detail::RetainedRowCore{}; // Approximate row: do not infer an exact interior integral from it.
                    ++retained_work->interpolated_rows;retained_work->row_interpolation_outcomes[6]+=hi-lo+1;return saved;
                }
                const auto found=retained_index.find(iy/retained_row_factor);if(found==retained_index.end())return std::nullopt;
                for(auto j:found->second){const auto& r=retained_rows->runs[j].run;
                    if(retained_column_factor>1) {
                        if(found->second.size()!=1 || settings.limb_darkening_d!=0 ||
                            iy/retained_row_factor-retained_min_y<4 || retained_max_y-iy/retained_row_factor<4)return std::nullopt;
                        if(retained_work->local_boundary_strips)return boundary_strip_row(j,ix,iy);
                        if(retained_row_factor!=1)return std::nullopt;
                        const auto& e=retained_rows->run_flux_errors[j];
                        if(e[2]!=0 || std::abs(e[0])+e[3]>retained_work->target/(4*retained_rows->runs.size()))return std::nullopt;
                        const auto& fractions=retained_rows->run_error_fractions[j];
                        const double left=(r.lo-fractions[0])*retained_column_factor,right=(r.hi+fractions[1])*retained_column_factor;
                        const auto lo=static_cast<std::int64_t>(std::ceil(left)),hi=static_cast<std::int64_t>(std::floor(right));
                        if(lo>ix || hi<ix || lo>hi)continue;
                        if(!inside_cell(evaluate_cell(lo,iy)) || !inside_cell(evaluate_cell(hi,iy)) ||
                            inside_cell(evaluate_cell(lo-1,iy)) || inside_cell(evaluate_cell(hi+1,iy)))return std::nullopt;
                        detail::CartesianRetainedRun saved;saved.run={iy,lo,hi};
                        saved.area=retained_rows->run_areas[j]*retained_column_factor;
                        saved.errors=e; // Same physical row integral and error.
                        saved.fractions={lo-left,right-hi};
                        ++retained_work->retained_horizontal_rows;return saved;
                    }
                    if(ix<r.lo || ix>r.hi)continue;
                    detail::CartesianRetainedRun saved;saved.run={iy,r.lo,r.hi};saved.area=retained_rows->run_areas[j];saved.errors=retained_rows->run_flux_errors[j];saved.fractions=retained_rows->run_error_fractions[j];
                    if(retained_rows->run_cores.size()==retained_rows->runs.size())saved.core=retained_rows->run_cores[j];
                    for(int k:{0,1,3})saved.errors[k]/=retained_row_factor;
                    ++retained_work->reused_rows;return saved;
                }return std::nullopt;
            };
        }
        std::unordered_map<std::int64_t,std::vector<std::size_t>> retained_fine_index;
        if(retained_fine_rows && retained_work) {
            for(std::size_t j=0;j<retained_fine_rows->runs.size();++j)retained_fine_index[retained_fine_rows->runs[j].run.iy].push_back(j);
            auto coarse_callback=std::move(retained_row);
            retained_row=[&,coarse_callback](std::int64_t ix,std::int64_t iy)->std::optional<detail::CartesianRetainedRun>{
                const auto found=retained_fine_index.find(iy);
                if(found!=retained_fine_index.end())for(auto j:found->second) {
                    const auto& r=retained_fine_rows->runs[j].run;if(ix<r.lo || ix>r.hi)continue;
                    detail::CartesianRetainedRun saved;saved.run=r;saved.area=retained_fine_rows->run_areas[j];
                    saved.errors=retained_fine_rows->run_flux_errors[j];saved.fractions=retained_fine_rows->run_error_fractions[j];
                    if(retained_fine_rows->run_cores.size()==retained_fine_rows->runs.size())saved.core=retained_fine_rows->run_cores[j];
                    ++retained_work->reused_cap_rows;return saved;
                }
                return coarse_callback?coarse_callback(ix,iy):std::nullopt;
            };
        }
        const auto execute = [&](const auto& boundary) {
            return detail::fill_cartesian_runs<RunCellState>(
                seeds,evaluate_cell,inside_cell,weight_cell,boundary,
        detail::CartesianRunFillLimits {
            maximum_evaluations,
            std::min(kRunMemorySafetyCeiling, evaluation_run_ceiling),
            expected_rows,
        },
        trace_sink,coverage_probes,coverage_domain ? &*coverage_domain : nullptr,
        interior_block ? &interior_block : nullptr,use_limb_darkening?16:32,retained_row?&retained_row:nullptr,
        retained_work ? retained_work->local_boundary_strips :
            std::getenv("LCBININT_DIAGNOSTIC_LOCAL_BOUNDARY_STRIPS")!=nullptr);
        };
        if (boundary_error && (!light_error || use_limb_darkening)) return execute(diagnostic_boundary);
        // A five-argument callback avoids retaining/updating two additional
        // optional neighbour states at every inside cell of the normal scan.
        const auto physical_boundary = [&](const detail::CartesianRun& run,
            const RunCellState& left_inside,const RunCellState& left_outside,
            const RunCellState& right_inside,const RunCellState& right_outside) {
            const std::optional<RunCellState> none;
            return diagnostic_boundary(run,left_inside,left_outside,right_inside,right_outside,none,none);
        };
        return execute(physical_boundary);
    };
    if (trace != nullptr) {
        return run_fill(*trace);
    }
    auto& no_trace = detail::default_cartesian_run_fill_trace();
    return run_fill(no_trace);
}

struct CartesianRunComponent {
    long double area = 0.0L;
    std::int64_t cells = 0;
    int boundary_edges = 0;
    std::int64_t min_iy = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_iy = std::numeric_limits<std::int64_t>::min();
    int row_count = 0;
    std::size_t first_run_index = std::numeric_limits<std::size_t>::max();
    ComponentFill measurement;
    int refinement_factor = 1;
};

bool refined_run_footprint_is_private(
    const detail::CartesianRunFillResult& refined,
    int factor,
    const detail::CartesianRunFillResult& coarse,
    std::size_t coarse_component,
    int row_factor=0)
{
    if(row_factor==0)row_factor=factor;
    const auto divide_floor = [factor](std::int64_t value) {
        const std::int64_t quotient = value / factor;
        const std::int64_t remainder = value % factor;
        return quotient - (remainder < 0 ? 1 : 0);
    };
    const auto divide_ceil = [factor](std::int64_t value) {
        const std::int64_t quotient = value / factor;
        const std::int64_t remainder = value % factor;
        return quotient + (remainder > 0 ? 1 : 0);
    };
    std::vector<const detail::CartesianRunRecord*> coarse_order;
    const bool indexed = !refined.runs.empty() &&
        coarse.runs.size() > 1000000U / refined.runs.size();
    if (indexed) {
        coarse_order.reserve(coarse.runs.size());
        for (const auto& record : coarse.runs) {
            coarse_order.push_back(&record);
        }
        std::sort(
            coarse_order.begin(), coarse_order.end(),
            [](const auto* lhs, const auto* rhs) {
                return lhs->run.iy != rhs->run.iy
                    ? lhs->run.iy < rhs->run.iy
                    : lhs->run.lo < rhs->run.lo;
            });
    }
    const auto overlaps_foreign = [&](std::int64_t iy,
                                      std::int64_t lo,
                                      std::int64_t hi) {
        const auto overlaps = [&](const auto& record) {
            return record.component != coarse_component &&
                record.run.hi >= lo && record.run.lo <= hi;
        };
        if (!indexed) {
            for (const auto& record : coarse.runs) {
                if (record.run.iy == iy && overlaps(record)) {
                    return true;
                }
            }
            return false;
        }
        const auto first = std::lower_bound(
            coarse_order.begin(), coarse_order.end(), iy,
            [](const auto* record, std::int64_t row) {
                return record->run.iy < row;
            });
        for (auto item = first;
             item != coarse_order.end() && (*item)->run.iy == iy; ++item) {
            if (overlaps(**item)) {
                return true;
            }
        }
        return false;
    };
    for (const auto& fine_record : refined.runs) {
        if (fine_record.run.iy % row_factor != 0) {
            continue;
        }
        const std::int64_t coarse_iy = fine_record.run.iy / row_factor;
        const std::int64_t coarse_lo = divide_ceil(fine_record.run.lo);
        const std::int64_t coarse_hi = divide_floor(fine_record.run.hi);
        if (coarse_lo > coarse_hi) {
            continue;
        }
        if (overlaps_foreign(coarse_iy, coarse_lo, coarse_hi)) {
            return false;
        }
    }
    return true;
}

struct RetainedAdaptiveComponent {
    detail::CartesianRunFillResult fill;
    double flux=0,signed_error=0,remainder=0;
    int bins=0,row_bins=0,order=2;
    int previous_axis=0;
    double work_estimate=0;
    double inner_signed=0,outer_signed=0,inner_remainder=0,outer_remainder=0,short_guard=0;
    std::size_t original_owner=0;
    bool blocked=false,missing_cap_rows=false;
    std::array<double,6> outer_parts{}; // cap signed/rem, smooth rem, junction signed/rem, resolved
    std::array<bool,2> cap_weights{};
    int retained_cap_bins=0;
    detail::CartesianRunFillResult cap_rows;
};

template <typename ImageMap>
double fill_all_cartesian_components_multirun(
    const ImageMap& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<EvaluatedCartesianSeed>& evaluated_seeds,
    double incr,
    double magnification_hint,
    LegacyAreaDiagnostics* diagnostics,
    detail::CartesianRunFillTrace* trace = nullptr,
    const std::vector<detail::CartesianLatticeSeed>* coverage_probes = nullptr,
    bool source_has_critical=false,
    std::vector<RetainedAdaptiveComponent>* retained_output=nullptr,
    double row_spacing=0,
    const detail::CartesianRunFillResult* retained_rows=nullptr,int retained_row_factor=1,int retained_column_factor=1,
    const std::array<std::int64_t,2>* row_clip=nullptr,unsigned physical_tip_mask=3,int cap_weight_mode=0,
    const detail::CartesianRunFillResult* retained_fine_rows=nullptr)
{
    if(row_spacing==0)row_spacing=incr;
    auto* adaptive=adaptive_flux_request();
    if(adaptive && adaptive->depth==0 &&
       (settings.limb_darkening_d!=0 || settings.limb_darkening_c<0 || settings.limb_darkening_c>1))
        adaptive->defect_correction=false;
    if(adaptive && adaptive->mapped.slots.empty() && adaptive->depth==0 &&
       std::isfinite(magnification_hint) && magnification_hint>0) {
        // Point magnification is only an allocation hint, never an error or
        // coverage criterion. Keep the ordinary growth path for underestimates.
        const double expected=2*kPi*magnification_hint*settings.source_bins*settings.source_bins;
        const std::size_t ceiling=262144;
        std::size_t capacity=adaptive->mapped.initial_capacity;
        while(capacity<ceiling && capacity<expected)capacity*=2;
        adaptive->mapped.initial_capacity=capacity;
    }
    // Small budgets retain eager deduplication; their stopping point must not
    // depend on repeated evaluations in the deferred initial pass.
    if(adaptive && adaptive->depth==0 && adaptive->mapped.pending.empty() &&
       (settings.source_bins!=16 || adaptive->budget<=adaptive->mapped.initial_capacity))
        adaptive->mapped.deferred=false;
    std::vector<RetainedAdaptiveComponent> retained;
    const double source_radius2 = source_radius * source_radius;
    std::vector<detail::CartesianLatticeSeed> seeds;
    seeds.reserve(evaluated_seeds.size());
    for (const auto& seed : evaluated_seeds) {
        std::int64_t ix = 0;
        std::int64_t iy = 0;
        if (!rounded_lattice_index(seed.position.x / incr, ix) ||
            !rounded_lattice_index(seed.position.y / row_spacing, iy)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        seeds.push_back({ix, iy});
    }

    const double bins = static_cast<double>(std::max(settings.source_bins, 1));
    const double evaluation_limit_estimate = bins * bins * std::max(
        2000.0,
        4.0 * kPi *
            (std::isfinite(magnification_hint)
                 ? std::abs(magnification_hint)
                 : 0.0));
    const std::int64_t evaluation_limit =
        evaluation_limit_estimate <
                static_cast<double>(std::numeric_limits<std::int64_t>::max())
            ? std::max<std::int64_t>(
                100000,
                static_cast<std::int64_t>(
                    std::ceil(evaluation_limit_estimate)))
            : std::numeric_limits<std::int64_t>::max();
    if (trace != nullptr) {
        trace->fill_level = 0;
    }
    const bool fill_stats=probe_stats_enabled();
    const bool phase_trace=adaptive && adaptive->cost_trace;
    const auto fill_started=(fill_stats || phase_trace) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<SourcePosition> support_positions;
    if (std::getenv("LCBININT_SEED_FULL_COVERAGE_AUDIT")
        || std::getenv("LCBININT_CRITICAL_SUPPORT_CERTIFICATE"))
        for (const auto& seed:evaluated_seeds) support_positions.push_back(seed.position);
    auto run_fill = fill_cartesian_run_union(
        mapper, source, source_radius, settings, finite_magnifier,
        seeds, incr, evaluation_limit,
        static_cast<std::size_t>(std::max(settings.source_bins, 1)) * 64U,
        trace,coverage_probes,(std::getenv("LCBININT_CARTESIAN_COVERAGE_CERTIFICATE")!=nullptr || std::getenv("LCBININT_SEED_LATTICE_CERTIFICATE")!=nullptr),&support_positions,row_spacing,retained_rows,retained_row_factor,retained_column_factor,row_clip,3,0,retained_fine_rows);
    const auto aggregation_started=phase_trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    if(phase_trace) {
        const double duration=std::chrono::duration<double>(aggregation_started-fill_started).count();
        adaptive->run_fill_seconds+=duration;
        adaptive->fill_queries+=run_fill.counters.mapped_cell_evaluations;
        adaptive->fill_rows+=run_fill.runs.size();
        adaptive->duplicate_seed_starts+=run_fill.counters.duplicate_seeds_avoided;
        for(auto cells:run_fill.component_cells)adaptive->fill_cells+=cells;
        if(adaptive->depth==0)adaptive->initial_fill_seconds+=duration;
    }
    if (fill_stats) {
        auto& c=probe_counters();++c.cartesian_fill_calls;
        c.cartesian_map_queries+=run_fill.counters.mapped_cell_evaluations;
        for (auto n:run_fill.component_cells) c.cartesian_fill_cells+=n;
        c.cartesian_fill_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-fill_started).count();
    }
    if (lightweight_flux_error_mode() && !fill_stats) {
        // Count existing work only. No timers, extra maps, event trace or solves.
        auto& c=probe_counters();++c.cartesian_fill_calls;
        c.cartesian_map_queries+=run_fill.counters.mapped_cell_evaluations;
    }
    if (trace && (std::getenv("LCBININT_CARTESIAN_COVERAGE_CERTIFICATE") || std::getenv("LCBININT_SEED_LATTICE_CERTIFICATE"))) {
        trace->coverage_requested=true;trace->coverage_certified=run_fill.coverage_certified;
        trace->coverage_boxes=run_fill.counters.coverage_boxes;
        trace->coverage_unresolved=run_fill.counters.coverage_unresolved;
        trace->coverage_components=run_fill.counters.coverage_components_added;
    }
    if (std::getenv("LCBININT_CARTESIAN_COVERAGE_CERTIFICATE") || std::getenv("LCBININT_SEED_LATTICE_CERTIFICATE")) {
        if constexpr (std::is_same_v<ImageMap,BinaryLensMapper>) {
            auto& c=probe_counters();++c.cartesian_coverage_calls;c.cartesian_coverage_certified+=run_fill.coverage_certified;
            c.cartesian_coverage_boxes+=run_fill.counters.coverage_boxes;c.cartesian_coverage_unresolved+=run_fill.counters.coverage_unresolved;
            c.cartesian_coverage_repairs+=run_fill.counters.coverage_components_added;c.cartesian_coverage_seconds+=run_fill.coverage_seconds;
            if(diagnostics) diagnostics->seed_coverage_proven=run_fill.coverage_certified;
        }
    }
    probe_counters().critical_grid_checks+=run_fill.counters.coverage_probe_cells;
    probe_counters().critical_grid_components+=run_fill.counters.coverage_components_added;
    if (!run_fill.ok()) {
        if (diagnostics != nullptr) {
            ++diagnostics->walk_budget_failures;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (boundary_diagnostics_enabled() && !lightweight_flux_error_mode()) {
        // Single-grid outer-coordinate shape indicator. It is not a bound:
        // sqrt endpoints, topology changes and quadrature remainder need their
        // own treatment. Reuse F(y); do not perform a second resolution fill.
        std::unordered_map<std::int64_t,double> row_areas;
        row_areas.reserve(run_fill.runs.size());
        for (std::size_t j=0;j<run_fill.runs.size();++j)
            row_areas[run_fill.runs[j].run.iy]+=run_fill.run_areas[j];
        long double curvature=0.0;
        std::unordered_set<std::int64_t> zero_halo;
        for (const auto& row : row_areas) {
            if (row.first==std::numeric_limits<std::int64_t>::min() ||
                row.first==std::numeric_limits<std::int64_t>::max()) continue;
            const auto below=row_areas.find(row.first-1),above=row_areas.find(row.first+1);
            const double low=below==row_areas.end() ? 0.0 : below->second;
            const double high=above==row_areas.end() ? 0.0 : above->second;
            curvature+=std::abs(low-2.0*row.second+high);
            if (below==row_areas.end()) zero_halo.insert(row.first-1);
            if (above==row_areas.end()) zero_halo.insert(row.first+1);
        }
        // The finite difference stencil extends one row beyond the support.
        // Omitting these zero-centred stencils misses endpoint curvature.
        for (const auto iy : zero_halo) {
            if (iy==std::numeric_limits<std::int64_t>::min() ||
                iy==std::numeric_limits<std::int64_t>::max()) continue;
            const auto below=row_areas.find(iy-1),above=row_areas.find(iy+1);
            const double low=below==row_areas.end() ? 0.0 : below->second;
            const double high=above==row_areas.end() ? 0.0 : above->second;
            curvature+=std::abs(low+high);
        }
        probe_counters().boundary_outer_curvature_flux_indicator+=
            static_cast<double>(curvature)*incr*row_spacing/24.0;
    }
    if (diagnostics != nullptr) {
        diagnostics->mapped_cell_evaluations +=
            run_fill.counters.mapped_cell_evaluations;
        diagnostics->frontier_intervals_popped +=
            run_fill.counters.frontier_intervals_popped;
        diagnostics->rows_with_multiple_runs +=
            run_fill.counters.rows_with_multiple_runs;
        diagnostics->maximum_runs_in_row = std::max(
            diagnostics->maximum_runs_in_row,
            run_fill.counters.maximum_runs_in_row);
        diagnostics->unique_seed_cells = run_fill.counters.unique_seed_cells;
        diagnostics->duplicate_seed_starts_avoided +=
            run_fill.counters.duplicate_seeds_avoided;
        diagnostics->provisional_components +=
            run_fill.counters.provisional_components;
        diagnostics->merged_components +=
            run_fill.counters.merged_components;
        for (std::size_t index = 0;
             index < run_fill.component_roots.size(); ++index) {
            if (run_fill.component_roots[index] == index) {
                diagnostics->integrated_cells +=
                    run_fill.component_cells[index];
            }
        }
    }

    constexpr std::size_t missing_component =
        std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> component_index(
        run_fill.component_roots.size(), missing_component);
    std::vector<CartesianRunComponent> components;
    for (std::size_t index = 0; index < run_fill.runs.size(); ++index) {
        const auto& record = run_fill.runs[index];
        if (component_index[record.component] == missing_component) {
            component_index[record.component] = components.size();
            components.emplace_back();
        }
        auto& component = components[component_index[record.component]];
        component.area = run_fill.component_areas[record.component];
        component.cells = run_fill.component_cells[record.component];
        component.boundary_edges =
            run_fill.component_boundary_edges[record.component];
        component.min_iy = std::min(component.min_iy, record.run.iy);
        component.max_iy = std::max(component.max_iy, record.run.iy);
        component.row_count = run_fill.component_rows[record.component];
        if (component.first_run_index == missing_component) {
            component.first_run_index = index;
        } else {
            const auto& current = run_fill.runs[component.first_run_index].run;
            if (record.run.iy < current.iy ||
                (record.run.iy == current.iy && record.run.lo < current.lo)) {
                component.first_run_index = index;
            }
        }
    }

    const bool bennett_outer_complex_diagnostic =
        std::getenv("LCBININT_ENABLE_BENNETT_OUTER_SEGMENTS") != nullptr;
    const char* seed_mode=std::getenv("LCBININT_REPRESENTATIVE_PROBES");
    const bool geometry_outer_guard=std::getenv("LCBININT_CARTESIAN_GEOMETRY_OUTER_GUARD")!=nullptr
        || (seed_mode && std::string(seed_mode)=="critical-seeds");
    const bool bennett_outer_enabled =cap_weight_mode>=0 &&
        std::getenv("LCBININT_DISABLE_BENNETT_OUTER") == nullptr &&
        !(geometry_outer_guard && source_has_critical) &&
        (std::getenv("LCBININT_ENABLE_BENNETT_OUTER") != nullptr ||
         bennett_outer_complex_diagnostic || components.size() <= 2U);

    struct CartesianOuterBoundaryEstimate {
        double eta = std::numeric_limits<double>::quiet_NaN();
    };
    const auto vertical_boundary_delta = [&](std::size_t run_index,
                                             int direction,
                                             bool permit_partial_crossings)
        -> std::optional<CartesianOuterBoundaryEstimate> {
        if (run_index >= run_fill.runs.size()) {
            return std::nullopt;
        }
        const auto& edge_run = run_fill.runs[run_index].run;
        if (edge_run.lo >= edge_run.hi ||
            edge_run.hi == std::numeric_limits<std::int64_t>::max()) {
            return std::nullopt;
        }
        const double inside_y = static_cast<double>(edge_run.iy) * row_spacing;
        const double outside_y = inside_y + static_cast<double>(direction) * row_spacing;
        const std::uint64_t width_u = static_cast<std::uint64_t>(
            edge_run.hi - edge_run.lo) + 1U;
        if (width_u > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return std::nullopt;
        }

        // Dave's row walker evaluates the boundary crossing for every radial
        // sample in the terminating row and uses the largest crossing as the
        // outer edge.  Keep that same per-segment operation here rather than
        // inferring a tangent from only a few raster samples.  A clean outer
        // endpoint must bracket the limb at every sample.  A backtrack
        // endpoint is allowed to have the neighbouring image branch in some
        // samples; it still needs at least one valid limb bracket.
        double eta = 0.0;
        bool found_crossing = false;
        for (std::int64_t ix = edge_run.lo;; ++ix) {
            const double x = static_cast<double>(ix) * incr;
            const double inside_residual = boundary_root_residual(
                mapper, x, inside_y, source, source_radius2);
            const double outside_residual = boundary_root_residual(
                mapper, x, outside_y, source, source_radius2);
            if (!(std::isfinite(inside_residual) &&
                  std::isfinite(outside_residual) &&
                  inside_residual <= 0.0)) {
                if (!permit_partial_crossings) {
                    return std::nullopt;
                }
                if (ix == edge_run.hi) {
                    break;
                }
                continue;
            }
            if (!(outside_residual > 0.0)) {
                if (!permit_partial_crossings) {
                    return std::nullopt;
                }
                if (ix == edge_run.hi) {
                    break;
                }
                continue;
            }
            const double denominator = outside_residual - inside_residual;
            const double initial = denominator > 0.0
                ? std::clamp(-inside_residual / denominator, 0.0, 1.0)
                : 0.5;
            const double candidate = refine_vertical_boundary_crossing_fraction(
                mapper, source, source_radius2, x,
                inside_y, outside_y, inside_residual, outside_residual,
                initial, row_spacing);
            eta = std::max(eta, candidate);
            found_crossing = true;
            if (ix == edge_run.hi) {
                break;
            }
        }
        return found_crossing
            ? std::optional<CartesianOuterBoundaryEstimate>(
                CartesianOuterBoundaryEstimate {eta})
            : std::nullopt;
    };
    // Apply equation (15) to the one-to-one outer segments returned by the
    // check_bktrack-style walker.  A component may contain several such
    // segments after a split or a merge.  The walker never lets the rule
    // cross a junction, and the physical boundary check below additionally
    // rejects an endpoint whose neighbouring row still contains image cells.
    // Diagnostic-only weights; ordinary integration allocates nothing here.
    std::vector<double> traced_outer_weights;
    std::vector<int> traced_outer_pairs;
    int traced_pair_count=0;
    if (trace != nullptr || lightweight_flux_error_mode()) {
        traced_outer_weights.assign(run_fill.runs.size(), 1.0);
        traced_outer_pairs.assign(run_fill.runs.size(), -1);
    }
    int outer_components_considered = 0;
    int outer_segments_considered = 0;
    int outer_segments_applied = 0;
    int outer_components_rejected_span = 0;
    int outer_components_rejected_boundary = 0;
    if (bennett_outer_enabled) {
        for (std::size_t component_id = 0;
             component_id < components.size(); ++component_id) {
            ++outer_components_considered;
            auto& component = components[component_id];
            const std::int64_t span = component.max_iy >= component.min_iy
                ? component.max_iy - component.min_iy + 1
                : 0;
            // Equation (15) is useful for the long, thin image arcs for which
            // Bennett designed it.  A short blob has no asymptotic outer
            // strip, and the multi-seed/fold regime remains fail-closed.
            if ((components.size() > 2U && !bennett_outer_complex_diagnostic) ||
                (!geometry_outer_guard && run_fill.counters.provisional_components > 8) ||
                span < 4 ||
                component.first_run_index == missing_component) {
                ++outer_components_rejected_span;
                continue;
            }
            const std::size_t root_component =
                run_fill.runs[component.first_run_index].component;
            const auto walk = detail::walk_cartesian_outer_segments(
                run_fill.runs, root_component);
            if (!walk.complete || walk.segments.empty()) {
                ++outer_components_rejected_span;
                continue;
            }
            // Provisional starts measure seed density, not image topology.
            // Use actual split/merge junctions for the experimental guard.
            // A clean-arc Eq. (15) rule must not become active merely because
            // the seed algorithm supplied fewer redundant starts.
            if (geometry_outer_guard && walk.junction_nodes!=0U) {
                ++outer_components_rejected_span;
                continue;
            }
            // The experimental multi-component switch is for the new
            // check_bktrack-style topology path, not a broad license to
            // change every clean image arc in a three/five-image event.  The
            // latter showed a small but repeatable bias in the planetary
            // control case, while a junction is the evidence that this
            // segment walker is doing work the old component rule could not.
            if (components.size() > 2U &&
                bennett_outer_complex_diagnostic &&
                walk.junction_nodes == 0U) {
                ++outer_components_rejected_span;
                continue;
            }
            if (std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr &&
                walk.junction_nodes > 0U) {
                std::fprintf(
                    stderr,
                    "BENNETT_OUTER_CART_WALK component=%zu segments=%zu edges=%zu junctions=%zu complete=%d\n",
                    component_id, walk.segments.size(), walk.adjacency_edges,
                    walk.junction_nodes, walk.complete ? 1 : 0);
            }

            for (std::size_t segment_id = 0;
                 segment_id < walk.segments.size(); ++segment_id) {
                const auto& segment = walk.segments[segment_id];
                ++outer_segments_considered;
                const std::int64_t segment_span =
                    segment.run_indices.size() >= 2U
                    ? run_fill.runs[segment.run_indices.back()].run.iy -
                        run_fill.runs[segment.run_indices.front()].run.iy + 1
                    : 0;
                std::int64_t segment_cells = 0;
                for (const std::size_t run_index : segment.run_indices) {
                    const auto& run = run_fill.runs[run_index].run;
                    if (run.hi < run.lo ||
                        run.hi - run.lo ==
                            std::numeric_limits<std::int64_t>::max()) {
                        segment_cells = 0;
                        break;
                    }
                    segment_cells += run.hi - run.lo + 1;
                }
                const double segment_mean_width =
                    segment.run_indices.empty() || segment_cells <= 0
                    ? 0.0
                    : static_cast<double>(segment_cells) /
                        static_cast<double>(segment.run_indices.size());
                const double segment_aspect = segment_mean_width > 0.0
                    ? static_cast<double>(segment_span) / segment_mean_width
                    : 0.0;
                if (std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr &&
                    walk.junction_nodes > 0U) {
                    std::fprintf(
                        stderr,
                        "BENNETT_OUTER_CART_SEGMENT component=%zu segment=%zu length=%zu aspect=%.6g lower=%d/%d upper=%d/%d\n",
                        component_id, segment_id, segment.run_indices.size(),
                        segment_aspect,
                        segment.lower_boundary ? 1 : 0,
                        segment.lower_junction ? 1 : 0,
                        segment.upper_boundary ? 1 : 0,
                        segment.upper_junction ? 1 : 0);
                }
                // Equation (15) needs two distinct interior samples at an
                // endpoint.  A short backtrack is retained in the midpoint
                // sum rather than being promoted to a one-point rule.
                if (segment.run_indices.size() < 4U ||
                    segment_span < 4 || (cap_weight_mode!=1 && segment_aspect < 8.0)) {
                    ++outer_components_rejected_span;
                    continue;
                }
                bool applied = false;
                std::optional<double> lower_eta;
                std::optional<double> upper_eta;
                if ((physical_tip_mask&1) && (segment.lower_boundary || segment.lower_junction)) {
                    const auto boundary = vertical_boundary_delta(
                        segment.run_indices.front(), -1,
                        segment.lower_junction);
                    if (boundary.has_value()) {
                        const auto weights =
                            detail::bennett_outer_boundary_weights(
                                boundary->eta);
                        const std::size_t first = segment.run_indices.front();
                        const std::size_t second = segment.run_indices[1];
                        component.area +=
                            (weights.first - 1.0) * run_fill.run_areas[first] +
                            (weights.second - 1.0) * run_fill.run_areas[second];
                        if (trace != nullptr || lightweight_flux_error_mode()) {
                            traced_outer_pairs[first]=traced_pair_count;
                            traced_outer_pairs[second]=traced_pair_count++;
                            traced_outer_weights[first] += weights.first - 1.0;
                            traced_outer_weights[second] += weights.second - 1.0;
                        }
                        lower_eta = boundary->eta;
                        applied = true;
                    }
                }
                if ((physical_tip_mask&2) && (segment.upper_boundary || segment.upper_junction)) {
                    const auto boundary = vertical_boundary_delta(
                        segment.run_indices.back(), +1,
                        segment.upper_junction);
                    if (boundary.has_value()) {
                        const auto weights =
                            detail::bennett_outer_boundary_weights(
                                boundary->eta);
                        const std::size_t last = segment.run_indices.back();
                        const std::size_t penultimate =
                            segment.run_indices[segment.run_indices.size() - 2];
                        component.area +=
                            (weights.first - 1.0) * run_fill.run_areas[last] +
                            (weights.second - 1.0) *
                                run_fill.run_areas[penultimate];
                        if (trace != nullptr || lightweight_flux_error_mode()) {
                            traced_outer_pairs[last]=traced_pair_count;
                            traced_outer_pairs[penultimate]=traced_pair_count++;
                            traced_outer_weights[last] += weights.first - 1.0;
                            traced_outer_weights[penultimate] += weights.second - 1.0;
                        }
                        upper_eta = boundary->eta;
                        applied = true;
                    }
                }
                if (!applied) {
                    ++outer_components_rejected_boundary;
                    continue;
                }
                ++outer_segments_applied;
                if (std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr) {
                    std::fprintf(
                        stderr,
                        "BENNETT_OUTER_CART_APPLY component=%zu segment=%zu length=%zu aspect=%.6g lower=%d/%d upper=%d/%d lower_eta=%.9g upper_eta=%.9g\n",
                        component_id, segment_id, segment.run_indices.size(),
                        segment_aspect,
                        segment.lower_boundary ? 1 : 0,
                        segment.lower_junction ? 1 : 0,
                        segment.upper_boundary ? 1 : 0,
                        segment.upper_junction ? 1 : 0,
                        lower_eta.has_value() ? *lower_eta : -1.0,
                        upper_eta.has_value() ? *upper_eta : -1.0);
                }
            }
        }
    }
    if (std::getenv("LCBININT_AREA_DIAGNOSTICS") != nullptr &&
        bennett_outer_enabled) {
        std::fprintf(
            stderr,
            "BENNETT_OUTER_CART components=%d segments=%d applied=%d span_reject=%d boundary_reject=%d\n",
            outer_components_considered, outer_segments_considered,
            outer_segments_applied,
            outer_components_rejected_span, outer_components_rejected_boundary);
    }

    if (trace != nullptr) {
        for (auto& event : trace->boundary_diagnostics) {
            if (event.fill_level == trace->fill_level &&
                event.run_index < traced_outer_weights.size()) {
                event.outer_weight = traced_outer_weights[event.run_index];
                event.outer_pair = traced_outer_pairs[event.run_index];
            }
        }
    }

    for (auto& component : components) {
        component.measurement.area = static_cast<double>(component.area);
        component.measurement.rows = component.row_count;
        component.measurement.boundary_rows = component.boundary_edges > 0
            ? component.row_count
            : 0;
        component.measurement.rows_span = component.max_iy >= component.min_iy
            ? static_cast<int>(std::min<long double>(
                static_cast<long double>(component.max_iy) -
                    static_cast<long double>(component.min_iy) + 1.0L,
                static_cast<long double>(1 << 24)))
            : 0;
        component.measurement.width = component.row_count == 0
            ? 0.0
            : static_cast<double>(component.cells) /
                static_cast<double>(component.row_count);
        component.refinement_factor =
            adaptive==nullptr && std::getenv("LCBININT_DIAGNOSTIC_DISABLE_COMPONENT_REFINEMENT") == nullptr
            ? component_refinement_factor(
                component.measurement, settings.source_bins)
            : 1;
    }

    // One index for this completed fill. Reused by component error sums,
    // adopted-grid bookkeeping, row aggregation and topology models.
    std::vector<std::vector<std::size_t>> component_error_runs;
    if(lightweight_flux_error_mode()) {
        component_error_runs.resize(run_fill.component_roots.size());
        for(std::size_t j=0;j<run_fill.runs.size();++j)
            component_error_runs[run_fill.component_roots[run_fill.runs[j].component]].push_back(j);
        for(auto& indices:component_error_runs)std::sort(indices.begin(),indices.end(),[&](std::size_t a,std::size_t b) {
            const auto& x=run_fill.runs[a].run;const auto& y=run_fill.runs[b].run;
            return x.iy!=y.iy?x.iy<y.iy:x.lo<y.lo;
        });
    }
    long double total_area = 0.0L;
    for (std::size_t component_id = 0;
        component_id < components.size(); ++component_id) {
        auto& component = components[component_id];
        double contribution = static_cast<double>(component.area);
        const int factor = component.refinement_factor;
        bool error_used_refined=false;
        double error_component_signed=0,error_component_remainder=0,error_component_limb_density=0;
        double error_limb_correction=0,error_limb_uncertainty=0;
        double adaptive_outer_signed=0,adaptive_outer_remainder=0;
        std::array<double,6> adaptive_outer_parts{};
        bool adaptive_missing_caps=false;
        if(lightweight_flux_error_mode() && component.first_run_index!=missing_component) {
            const auto owner=run_fill.runs[component.first_run_index].component;
            for(const auto j:component_error_runs[owner]) {
                    error_component_signed+=run_fill.run_flux_errors[j][0]*traced_outer_weights[j];
                    error_component_remainder+=run_fill.run_flux_errors[j][3]*std::abs(traced_outer_weights[j]);
                    error_component_limb_density+=run_fill.run_flux_errors[j][4];
                }
        }
        if (factor > 1 && component.first_run_index != missing_component) {
            const std::size_t coarse_component =
                run_fill.runs[component.first_run_index].component;
            const auto refined_seeds =
                detail::lift_cartesian_component_run_seeds(
                    run_fill, coarse_component, factor);
            if (refined_seeds.has_value() && !refined_seeds->empty()) {
                FiniteSourceSettings refined_settings = settings;
                refined_settings.source_bins = settings.source_bins * factor;
                const double refined_incr = incr / static_cast<double>(factor);
                const std::int64_t refined_budget =
                    static_cast<std::int64_t>(std::ceil(
                        8.0 * component.measurement.width *
                            static_cast<double>(component.measurement.rows_span) *
                            static_cast<double>(factor) *
                            static_cast<double>(factor) +
                        4096.0));
                if (trace != nullptr) {
                    trace->fill_level = 1;
                }
                const auto& error_before=probe_counters();
                const double prior_signed=error_before.boundary_model_signed_flux_residual;
                const double prior_absolute=error_before.boundary_model_absolute_flux_residual;
                const auto prior_unresolved=error_before.boundary_model_unresolved;
                auto refined = fill_cartesian_run_union(
                    mapper, source, source_radius, refined_settings,
                    finite_magnifier, *refined_seeds, refined_incr,
                    refined_budget,
                    static_cast<std::size_t>(std::max(
                        16,
                        2 * component.measurement.rows_span * factor + 16)),
                    trace);
                if(lightweight_flux_error_mode()) {
                    // A trial fill must not contribute until its flux is adopted.
                    auto& c=probe_counters();
                    c.boundary_model_signed_flux_residual=prior_signed;
                    c.boundary_model_absolute_flux_residual=prior_absolute;
                    c.boundary_model_unresolved=prior_unresolved;
                }
                if (refined.ok()) {
                    if (diagnostics != nullptr) {
                        diagnostics->fine_grid_cell_evaluations +=
                            refined.counters.mapped_cell_evaluations;
                    }
                    long double refined_area = 0.0L;
                    for (std::size_t index = 0;
                         index < refined.component_roots.size(); ++index) {
                        if (refined.component_roots[index] == index) {
                            refined_area += refined.component_areas[index];
                        }
                    }
                    const double factor2 =
                        static_cast<double>(factor) *
                        static_cast<double>(factor);
                    const double rescaled =
                        static_cast<double>(refined_area) / factor2;
                    if (std::isfinite(rescaled) && rescaled > 0.0 &&
                        refined_run_footprint_is_private(
                            refined, factor, run_fill, coarse_component)) {
                        contribution = rescaled;
                        if(lightweight_flux_error_mode()) {
                            auto& c=probe_counters();
                            for(const auto j:component_error_runs[coarse_component]) {
                                    const auto& e=run_fill.run_flux_errors[j];
                                    c.boundary_model_signed_flux_residual-=e[0];
                                    c.boundary_model_absolute_flux_residual-=e[1];
                                    c.boundary_model_unresolved-=static_cast<long long>(e[2]);
                                }
                            error_component_signed=0;error_component_remainder=0;error_component_limb_density=0;
                            for(const auto& e:refined.run_flux_errors) {
                                error_component_signed+=e[0];
                                error_component_remainder+=e[3];error_component_limb_density+=e[4];
                                c.boundary_model_signed_flux_residual+=e[0];
                                c.boundary_model_absolute_flux_residual+=e[1];
                                c.boundary_model_unresolved+=static_cast<long long>(e[2]);
                            }
                            const auto outer_error=detail::same_grid_outer_error(refined,std::numeric_limits<std::size_t>::max(),nullptr,1-settings.limb_darkening_c);
                            c.boundary_outer_curvature_flux_indicator+=(std::abs(outer_error.signed_residual)+outer_error.uncertainty)*refined_incr*refined_incr;
                            c.boundary_outer_signed_flux_residual+=outer_error.signed_residual*refined_incr*refined_incr;
                            c.boundary_outer_remainder_flux_indicator+=outer_error.uncertainty*refined_incr*refined_incr;
                            error_limb_correction=outer_error.limb_density_correction;error_limb_uncertainty=outer_error.limb_density_uncertainty;
                            c.error_outer_fallbacks+=!outer_error.resolved;
                            ++c.error_refined_components;
                            error_used_refined=true;
                        }
                        if (diagnostics != nullptr) {
                            ++diagnostics->refined_components;
                            diagnostics->refinement_factor = std::max(
                                diagnostics->refinement_factor, factor);
                        }
                    }
                }
                else if (diagnostics != nullptr) {
                    ++diagnostics->walk_budget_failures;
                }
            }
        }
        if(lightweight_flux_error_mode() && !error_used_refined && component.first_run_index!=missing_component) {
            const auto outer_error=detail::same_grid_outer_error(run_fill,
                run_fill.runs[component.first_run_index].component,&traced_outer_weights,1-settings.limb_darkening_c,
                &component_error_runs[run_fill.runs[component.first_run_index].component],settings.limb_darkening_d!=0,physical_tip_mask,row_clip==nullptr,
                adaptive && adaptive->defect_correction?1/(incr*row_spacing):0,
                adaptive && adaptive->guard_missing_caps);
            auto& c=probe_counters();
            c.boundary_outer_curvature_flux_indicator+=(std::abs(outer_error.signed_residual)+outer_error.uncertainty)*incr*row_spacing;
            c.boundary_outer_signed_flux_residual+=outer_error.signed_residual*incr*row_spacing;
            c.boundary_outer_remainder_flux_indicator+=outer_error.uncertainty*incr*row_spacing;
            adaptive_missing_caps=outer_error.missing_cap_rows;
            adaptive_outer_signed=outer_error.signed_residual*incr*row_spacing;
            adaptive_outer_remainder=outer_error.uncertainty*incr*row_spacing;
            if(adaptive && (adaptive->row_panel_trace || adaptive->local_vertical_caps)) {
                const double scale=incr*row_spacing;
                adaptive_outer_parts={outer_error.cap_signed*scale,outer_error.cap_remainder*scale,outer_error.smooth_remainder*scale,
                    (outer_error.signed_residual-outer_error.cap_signed)*scale,
                    (outer_error.uncertainty-outer_error.cap_remainder-outer_error.smooth_remainder)*scale,double(outer_error.resolved)};
            }
            error_limb_correction=outer_error.limb_density_correction;error_limb_uncertainty=outer_error.limb_density_uncertainty;
            c.error_outer_fallbacks+=!outer_error.resolved;
        }
        if(lightweight_flux_error_mode()) {
            auto& c=probe_counters();
            c.boundary_component_flux_indicator+=std::abs(error_component_signed)+error_component_remainder;
            c.boundary_component_signed_flux_residual+=error_component_signed;
            c.boundary_component_remainder_flux_indicator+=error_component_remainder;
            if(settings.limb_darkening_c!=0 && settings.limb_darkening_d!=0) {
                const double scale=kPi*source_radius*source_radius*settings.limb_darkening_c*detail::linear_ld_table_bias_moment(kLimbDarkeningTableSize);
                c.brightness_table_signed_flux_residual+=scale*std::max(0.,error_component_limb_density+error_limb_correction);
                c.brightness_table_remainder_flux_indicator+=scale*error_limb_uncertainty;
            }
        }
        if(adaptive && component.first_run_index!=missing_component) {
            RetainedAdaptiveComponent state;
            state.flux=contribution*incr*row_spacing;state.bins=settings.source_bins;state.row_bins=int(std::llround(source_radius/row_spacing));
            state.original_owner=run_fill.runs[component.first_run_index].component;state.outer_parts=adaptive_outer_parts;
            state.missing_cap_rows=adaptive_missing_caps;
            state.signed_error=error_component_signed+adaptive_outer_signed;
            state.remainder=error_component_remainder+adaptive_outer_remainder;
            state.inner_signed=error_component_signed;state.outer_signed=adaptive_outer_signed;
            state.inner_remainder=error_component_remainder;state.outer_remainder=adaptive_outer_remainder;
            // Fewer than eight retained rows cannot fit the endpoint model.
            // Use the represented component flux scale, rather than accepting
            // a tiny curvature fallback as a resolved short-component error.
            if(component.row_count<8){state.short_guard=std::max(0.,std::abs(state.flux)-state.remainder);state.remainder=std::max(state.remainder,std::abs(state.flux));}
            state.fill.component_roots={0};
            for(auto j:component_error_runs[state.original_owner]) {
                state.work_estimate+=double(run_fill.runs[j].run.hi-run_fill.runs[j].run.lo+1)+4;
                state.fill.runs.push_back({run_fill.runs[j].run,0});
                state.fill.run_areas.push_back(run_fill.run_areas[j]);
                state.fill.run_flux_errors.push_back(run_fill.run_flux_errors[j]);
                state.fill.run_error_fractions.push_back(run_fill.run_error_fractions[j]);
                if(state.fill.runs.size()<=2 && traced_outer_weights[j]!=1)state.cap_weights[0]=true;
                if(traced_outer_weights[j]!=1)state.cap_weights[1]=true;
                if(run_fill.run_cores.size()==run_fill.runs.size())state.fill.run_cores.push_back(run_fill.run_cores[j]);
            }
            if(state.fill.runs.size()>=2){const auto& js=component_error_runs[state.original_owner];state.cap_weights[1]=traced_outer_weights[js[js.size()-1]]!=1 || traced_outer_weights[js[js.size()-2]]!=1;}
            retained.push_back(std::move(state));
        }
        total_area += static_cast<long double>(contribution);
        if (diagnostics != nullptr) {
            diagnostics->boundary_rows += component.measurement.boundary_rows;
        }
    }

    if (diagnostics != nullptr) {
        diagnostics->processed_images = static_cast<int>(components.size());
        diagnostics->interval_count = static_cast<int>(run_fill.runs.size());
    }
    if(phase_trace)adaptive->error_aggregation_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-aggregation_started).count();
    if(retained_output)*retained_output=std::move(retained);
    else if(adaptive && adaptive->depth==0) {
        const auto control_started=phase_trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        const double children_seconds_before=adaptive->run_fill_seconds+adaptive->error_aggregation_seconds;
        adaptive->executed=true;adaptive->stop_reason="resolution_limit";
        // Centre-image witnesses were already solved by the seed stage.
        // A subcell image may have no inside point at the coarse lattice.
        // Find only its missing lattice witness, without another lens solve.
        for(const auto& witness:adaptive->centres) {
            bool represented=false;
            for(const auto& c:retained) {
                const double h=source_radius/c.bins;
                for(const auto& r:c.fill.runs)if(std::abs(witness[1]-r.run.iy*h)<=h &&
                    witness[0]>=(r.run.lo-1)*h && witness[0]<=(r.run.hi+1)*h){represented=true;break;}
                if(represented)break;
            }
            if(represented)continue;
            bool settled=false;
            try {
                for(int bins=settings.source_bins*2;bins<=adaptive->max_bins;bins*=2) {
                    const double h=source_radius/bins;
                    const auto ix=std::llround(witness[0]/h),iy=std::llround(witness[1]/h);
                    std::vector<EvaluatedCartesianSeed> local;
                    for(int dy=-1;dy<=1 && local.empty();++dy)for(int dx=-1;dx<=1;++dx) {
                        const std::array<double,2> key{(ix+dx)*h,(iy+dy)*h};double distance;
                        const auto [saved,inserted]=adaptive->mapped.access(key,adaptive->new_maps<adaptive->budget);
                        if(!saved)throw AdaptiveFluxBudgetExhausted{};
                        if(!inserted){distance=*saved;++adaptive->reused_maps;}
                        else {*saved=distance=mapped_lens_distance2(mapper,key[0],key[1],source);++adaptive->new_maps;}
                        if(distance<=source_radius2){local.push_back({{key[0],key[1]},0});break;}
                    }
                    if(local.empty())continue;
                    auto fine_settings=settings;fine_settings.source_bins=bins;
                    std::vector<RetainedAdaptiveComponent> found;const auto before=probe_counters();++adaptive->depth;
                    try {fill_all_cartesian_components_multirun(mapper,source,source_radius,fine_settings,finite_magnifier,
                        local,h,magnification_hint,nullptr,nullptr,nullptr,source_has_critical,&found);}
                    catch(...){--adaptive->depth;probe_counters()=before;throw;}
                    --adaptive->depth;probe_counters()=before;
                    for(auto& c:found) {
                        c.original_owner=run_fill.component_roots.size()+retained.size();
                        if(refined_run_footprint_is_private(c.fill,bins/settings.source_bins,run_fill,c.original_owner)) {
                            retained.push_back(std::move(c));++adaptive->small_image_repairs;
                        }
                    }
                    settled=!found.empty();break;
                }
            } catch(const AdaptiveFluxBudgetExhausted&){++adaptive->unresolved_centres;adaptive->stop_reason="evaluation_budget";break;}
            if(!settled)++adaptive->unresolved_centres;
        }
        const auto correction_for=[&](const RetainedAdaptiveComponent& c) {
            // Critical/junction models remain scheduling estimates. Apply
            // only regular endpoint defects, retaining the junction defect
            // in admission until its local refinement resolves it.
            const double correction=adaptive->defect_correction?c.signed_error-c.outer_parts[3]:0;
            const double flux=c.flux+correction;
            return std::isfinite(flux)&&flux>0?correction:0.;
        };
        const auto totals=[&]() {
            long double flux=0,signed_sum=0,remainder_sum=0,absolute_sum=0;
            long double applied=0;
            for(const auto& c:retained){const auto correction=correction_for(c);flux+=c.flux+correction;applied+=correction;signed_sum+=c.signed_error-correction;remainder_sum+=c.remainder;absolute_sum+=std::abs(c.signed_error-correction)+c.remainder;}
            adaptive->applied_correction=double(applied);
            adaptive->signed_error=double(signed_sum);adaptive->remainder=double(remainder_sum);
            adaptive->acceptance_indicator=double(absolute_sum);
            return std::array<double,2>{double(flux),double(absolute_sum)};
        };
        const auto record_cost=[&](double flux) {
            if(adaptive->row_panel_trace)for(std::size_t i=0;i<retained.size();++i) {
                const auto& c=retained[i];const double hx=source_radius/c.bins;
                adaptive->row_panel_errors.push_back({double(adaptive->history.size()-1),double(i),c.inner_signed,c.inner_remainder,
                    c.outer_parts[0],c.outer_parts[1],c.outer_parts[2],c.outer_parts[3],c.outer_parts[4],c.short_guard,c.outer_parts[5]});
                for(std::size_t j=0;j<c.fill.runs.size();++j) {
                    const auto& r=c.fill.runs[j].run;const auto& e=c.fill.run_flux_errors[j];
                    adaptive->row_panel_rows.push_back({double(adaptive->history.size()-1),double(i),double(c.bins),double(c.row_bins),double(r.iy),double(r.lo),double(r.hi),hx*c.fill.run_areas[j],e[0],e[3],e[2]});
                }
            }
            if(!adaptive->cost_trace)return;
            adaptive->history_flux.push_back(flux);
            for(std::size_t i=0;i<retained.size();++i){const auto& c=retained[i];double cells=0,edges=0;
                for(const auto& r:c.fill.runs){const double width=double(r.run.hi-r.run.lo+1);cells+=width;edges+=std::min(4.,width);}
                adaptive->cost_components.push_back({double(adaptive->history.size()-1),double(i),double(c.bins),c.flux,c.inner_signed,c.outer_signed,c.inner_remainder,c.outer_remainder,c.short_guard,double(c.fill.runs.size()),cells,edges});
            }
        };
        auto current=totals();adaptive->initial_flux=current[0];
        if(adaptive->relative_target>0 && std::isfinite(adaptive->relative_target) &&
           std::isfinite(current[0]) && current[0]!=0)
            adaptive->target=adaptive->relative_target*std::abs(current[0]);
        adaptive->largest_bins=settings.source_bins;
        for(const auto& c:retained)adaptive->largest_bins=std::max({adaptive->largest_bins,c.bins,c.row_bins});
        adaptive->history.push_back({double(settings.source_bins),current[1],double(adaptive->new_maps),double(adaptive->reused_maps),0});
        record_cost(current[0]);
        while(std::isfinite(current[1]) && current[1]>adaptive->target) {
            adaptive->mapped.materialize();
            std::size_t selected=retained.size();double priority=-1,total_priority=0,best_efficiency=-1;
            for(std::size_t i=0;i<retained.size();++i) {
                const auto& c=retained[i];const double score=std::abs(c.signed_error-correction_for(c))+c.remainder;
                total_priority+=score;
                const double efficiency=score*(1-std::pow(.5,c.order))/std::max(1.,c.work_estimate);
                if(!c.blocked && (c.bins<=adaptive->max_bins/2 || c.row_bins<=adaptive->max_bins/2) && efficiency>best_efficiency){selected=i;priority=score;best_efficiency=efficiency;}
            }
            if(selected==retained.size())break;
            auto& previous=retained[selected];
            const auto trial_start=adaptive->cost_trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            const auto trial_maps=adaptive->new_maps,trial_hits=adaptive->reused_maps;
            // Preserve the predicted remaining work from the local error
            // order, but add the minimum dyadic layer before re-evaluation.
            // A 4x jump is diagnostic only: grid phase can change its coefficient.
            const double local_target=adaptive->target*priority/std::max(total_priority,priority);
            const int factor=std::getenv("LCBININT_DIAGNOSTIC_ADAPTIVE_JUMP4") && previous.bins<=adaptive->max_bins/4 && priority>local_target*std::pow(2.,previous.order)?4:2;
            int factor_x=factor,factor_y=factor,axis=0;
            if(adaptive->directional_adaptive) {
                const bool corrected=correction_for(previous)!=0;
                const double horizontal=(corrected?0:std::abs(previous.inner_signed))+previous.inner_remainder;
                const double vertical=std::abs(corrected?previous.outer_parts[3]:previous.outer_signed)+previous.outer_remainder;
                if(previous.short_guard>horizontal+vertical){factor_x=1;axis=2;}
                else if(horizontal>2*vertical){factor_y=1;axis=1;}
                else if(vertical>2*horizontal){factor_x=1;axis=2;}
                // Keep an unresolved second direction from starving indefinitely.
                if(axis==previous.previous_axis && previous.order==1){factor_x=factor;factor_y=factor;axis=0;}
                if(previous.missing_cap_rows){factor_x=factor;factor_y=1;axis=1;}
                if(previous.bins*factor_x>adaptive->max_axis_ratio*previous.row_bins*factor_y){factor_y=factor;axis=0;}
                if(previous.row_bins*factor_y>adaptive->max_axis_ratio*previous.bins*factor_x){factor_x=factor;axis=0;}
            }
            if(previous.bins*factor_x>adaptive->max_bins)factor_x=1;
            if(previous.row_bins*factor_y>adaptive->max_bins)factor_y=1;
            if(factor_x==1 && factor_y==1){previous.blocked=true;continue;}
            const auto record_trial=[&](bool adopted){if(adaptive->cost_trace)adaptive->cost_trials.push_back({double(adaptive->history.size()-1),double(selected),double(previous.bins),double(previous.bins*factor_x),std::chrono::duration<double>(std::chrono::steady_clock::now()-trial_start).count(),double(adaptive->new_maps-trial_maps),double(adaptive->reused_maps-trial_hits),double(adopted)});};
            detail::CartesianRunFillResult failed_cap_rows;
            if(previous.retained_cap_bins && factor_x==1 && factor_y==2)failed_cap_rows=previous.cap_rows;
            if(adaptive->local_vertical_caps)failed_cap_rows.component_roots={0};
            const auto append_cap_rows=[&](const detail::CartesianRunFillResult& f) {
                for(std::size_t j=0;j<f.runs.size();++j) {
                    failed_cap_rows.runs.push_back({f.runs[j].run,0});failed_cap_rows.run_areas.push_back(f.run_areas[j]);
                    failed_cap_rows.run_flux_errors.push_back(f.run_flux_errors[j]);failed_cap_rows.run_error_fractions.push_back(f.run_error_fractions[j]);
                    failed_cap_rows.run_cores.push_back(f.run_cores.size()==f.runs.size()?f.run_cores[j]:std::nullopt);
                }
            };
            // Local cap candidate: keep the coarse interior integral,
            // refine only its two physical caps, and replace the cap error.
            // Failed candidates retain mapped work for ordinary refinement.
            if(adaptive->local_vertical_caps && previous.retained_cap_bins==0 && factor_x==1 && factor_y==2 &&
               settings.limb_darkening_d==0 && previous.fill.runs.size()>=25 &&
               (correction_for(previous)!=0?0:std::abs(previous.inner_signed))+previous.inner_remainder<=local_target/2) {
                const auto try_caps=[&]()->bool {
                    const auto& old=previous.fill;const auto n=old.runs.size();
                    for(std::size_t j=0;j<n;++j) {
                        if(old.run_flux_errors[j][2]!=0)return false;
                        if(j && (old.runs[j].run.iy!=old.runs[j-1].run.iy+1 ||
                            old.runs[j].run.lo>old.runs[j-1].run.hi || old.runs[j-1].run.lo>old.runs[j].run.hi))return false;
                    }
                    if(old.runs.front().run.iy<(std::numeric_limits<std::int64_t>::min()+4)/2 ||
                       old.runs.back().run.iy>(std::numeric_limits<std::int64_t>::max()-4)/2)return false;
                    const std::size_t a=4,b=n-5;
                    const double hx=source_radius/previous.bins,hy=source_radius/previous.row_bins,hf=hy/2;
                    std::array<double,7> vl,vu;double derivative_uncertainty1=0,derivative_uncertainty3=0;
                    constexpr std::array<double,7> d1weights{1,9,45,0,45,9,1},d3weights{1,8,13,0,13,8,1};
                    for(int k=0;k<7;++k) {
                        const auto l=a+k-3,u=b+k-3;
                        vl[k]=hx*old.run_areas[l]+old.run_flux_errors[l][0]/hy;
                        vu[k]=hx*old.run_areas[u]+old.run_flux_errors[u][0]/hy;
                        const double row_uncertainty=(old.run_flux_errors[l][3]+old.run_flux_errors[u][3])/hy;
                        derivative_uncertainty1+=d1weights[k]*row_uncertainty/(60*hy);
                        derivative_uncertainty3+=d3weights[k]*row_uncertainty/(8*hy*hy*hy);
                    }
                    const auto dl=detail::retained_core_derivatives(vl,hy),du=detail::retained_core_derivatives(vu,hy);
                    const double h2=hy*hy,f2=hf*hf;
                    const double interface_signed=(h2-f2)*(dl[0]-du[0])/12-(h2*h2-f2*f2)*(dl[1]-du[1])/720;
                    const double interface_remainder=(h2-f2)*(dl[2]+du[2]+derivative_uncertainty1)/12+
                        (h2*h2-f2*f2)*(dl[3]+du[3]+derivative_uncertainty3)/720+h2*h2*h2*(std::abs(dl[4])+std::abs(du[4]))/30240;
                    if(!std::isfinite(interface_remainder)||interface_remainder>local_target/4)return false;
                    ++adaptive->cap_trials;
                    double flux=0,inner=0,inner_rem=0,outer=interface_signed,outer_rem=interface_remainder;
                    for(auto j=a;j<=b;++j) {
                        const double w=j==a||j==b?.5:1;
                        flux+=w*hx*hy*old.run_areas[j];inner+=w*old.run_flux_errors[j][0];inner_rem+=w*old.run_flux_errors[j][3];
                    }
                    const auto saved_counters=probe_counters();
                    for(int side=0;side<2;++side) {
                        const auto anchor=side?b:a,seed_index=side?n-5:4;
                        const auto& sr=old.runs[seed_index].run;
                        const std::int64_t pad=adaptive->cap_initial_pad;
                        std::array<std::int64_t,2> clip=side?
                            std::array<std::int64_t,2>{old.runs[b].run.iy*2,old.runs.back().run.iy*2+pad}:
                            std::array<std::int64_t,2>{old.runs.front().run.iy*2-pad,old.runs[a].run.iy*2};
                        std::vector<EvaluatedCartesianSeed> local{{{(sr.lo+(sr.hi-sr.lo)/2)*hx,sr.iy*hy},0}};
                        auto fs=settings;fs.source_bins=previous.bins;
                        std::vector<RetainedAdaptiveComponent> band;
                        while(true) {
                            std::vector<RetainedAdaptiveComponent> next_band;
                            const auto* known_band=band.empty()?nullptr:&band.front().fill;
                            ++adaptive->depth;
                            try {fill_all_cartesian_components_multirun(mapper,source,source_radius,fs,finite_magnifier,
                                local,hx,magnification_hint,nullptr,nullptr,nullptr,source_has_critical,&next_band,hf,
                                &old,2,1,&clip,1U<<side,previous.cap_weights[side]?1:-1,known_band);}
                            catch(...){--adaptive->depth;probe_counters()=saved_counters;throw;}
                            --adaptive->depth;probe_counters()=saved_counters;
                            band=std::move(next_band);
                            if(band.size()!=1){for(const auto& c:band)append_cap_rows(c.fill);return false;}
                            bool touches_outer_clip=false;
                            for(const auto& r:band.front().fill.runs)
                                if(r.run.iy==clip[side?1:0])touches_outer_clip=true;
                            if(!touches_outer_clip)break;
                            auto& edge=clip[side?1:0];
                            if((side && edge>std::numeric_limits<std::int64_t>::max()-4) ||
                               (!side && edge<std::numeric_limits<std::int64_t>::min()+4))return false;
                            edge+=side?4:-4;++adaptive->cap_extensions;
                        }
                        const auto& c=band.front();
                        append_cap_rows(c.fill);
                        if(!refined_run_footprint_is_private(c.fill,previous.bins/settings.source_bins,run_fill,
                            previous.original_owner,previous.row_bins*2/settings.source_bins))return false;
                        flux+=c.flux;inner+=c.inner_signed;inner_rem+=c.inner_remainder;outer+=c.outer_signed;outer_rem+=c.outer_remainder;
                        bool found=false;
                        for(std::size_t j=0;j<c.fill.runs.size();++j)if(c.fill.runs[j].run.iy==old.runs[anchor].run.iy*2) {
                            if(found)return false;found=true;
                            flux-=.5*hx*hf*c.fill.run_areas[j];inner-=.5*c.fill.run_flux_errors[j][0];inner_rem-=.5*c.fill.run_flux_errors[j][3];
                        }
                        if(!found)return false;
                    }
                    const double indicator=std::abs(adaptive->defect_correction?interface_signed:inner+outer)+inner_rem+outer_rem;
                    if(!(std::isfinite(flux)&&flux>0&&std::isfinite(indicator)&&inner_rem>=0&&outer_rem>=0&&indicator<=local_target))return false;
                    previous.flux=flux;previous.inner_signed=inner;previous.inner_remainder=inner_rem;
                    previous.outer_signed=outer;previous.outer_remainder=outer_rem;
                    previous.signed_error=inner+outer;previous.remainder=inner_rem+outer_rem;
                    previous.outer_parts={outer-interface_signed,outer_rem-interface_remainder,interface_remainder,interface_signed,0,1};
                    previous.retained_cap_bins=previous.row_bins*2;previous.cap_rows=std::move(failed_cap_rows);
                    previous.blocked=false;previous.short_guard=0;
                    return true;
                };
                const auto trials_before=adaptive->cap_trials;
                bool adopted=false;
                try {adopted=try_caps();}catch(const AdaptiveFluxBudgetExhausted&){adaptive->stop_reason="evaluation_budget";record_trial(false);break;}
                if(adopted) {
                    record_trial(true);
                    ++adaptive->cap_adopted;++adaptive->refinements;
                    adaptive->largest_bins=std::max(adaptive->largest_bins,previous.retained_cap_bins);
                    adaptive->refinement_grid_bins.push_back({double(selected),double(previous.bins),double(previous.row_bins),double(previous.bins),double(previous.retained_cap_bins),2});
                    current=totals();adaptive->history.push_back({double(previous.bins),current[1],double(adaptive->new_maps),double(adaptive->reused_maps),double(selected)});
                    record_cost(current[0]);continue;
                }
                if(adaptive->cap_trials>trials_before)++adaptive->cap_rejected;
            }
            const auto lifted=detail::lift_cartesian_component_run_seeds(previous.fill,0,factor_x,factor_y);
            if(!lifted || lifted->empty()){previous.blocked=true;continue;}
            const int next_bins=previous.bins*factor_x,next_row_bins=previous.row_bins*factor_y;
            const double next_h=source_radius/next_bins,next_row_h=source_radius/next_row_bins;
            std::vector<EvaluatedCartesianSeed> next_seeds;next_seeds.reserve(lifted->size());
            for(const auto& seed:*lifted)next_seeds.push_back({{seed.ix*next_h,seed.iy*next_row_h},0});
            auto next_settings=settings;next_settings.source_bins=next_bins;
            std::vector<RetainedAdaptiveComponent> children;
            const auto counters_before=probe_counters();
            ++adaptive->depth;
            try {
                fill_all_cartesian_components_multirun(mapper,source,source_radius,next_settings,finite_magnifier,
                    next_seeds,next_h,magnification_hint,nullptr,nullptr,nullptr,source_has_critical,&children,next_row_h,
                    !std::getenv("LCBININT_DIAGNOSTIC_DISABLE_ROW_REUSE") &&
                        (factor_x==1 || adaptive->local_boundary_strips ||
                            (factor_y==1 && std::getenv("LCBININT_DIAGNOSTIC_LOCAL_HORIZONTAL_ROWS")))?&previous.fill:nullptr,factor_y,factor_x,nullptr,3,0,
                    failed_cap_rows.runs.empty()?nullptr:&failed_cap_rows);
            } catch(const AdaptiveFluxBudgetExhausted&) {
                --adaptive->depth;probe_counters()=counters_before;
                adaptive->stop_reason="evaluation_budget";record_trial(false);break;
            } catch(...) {--adaptive->depth;throw;}
            --adaptive->depth;
            // Trial error counters must not be mistaken for adopted flux.
            // Work counts survive in the explicit adaptive context.
            probe_counters()=counters_before;
            RetainedAdaptiveComponent next;next.bins=next_bins;next.row_bins=next_row_bins;next.previous_axis=axis;next.original_owner=previous.original_owner;next.fill.component_roots={0};
            if(children.size()==1 && !std::getenv("LCBININT_DIAGNOSTIC_DISABLE_SINGLE_CHILD_MOVE")) {
                // The ordinary refinement of one connected component normally
                // has one child. Transfer its retained rows instead of growing
                // and copying five vectors that were just constructed.
                next=std::move(children.front());
                next.bins=next_bins;next.row_bins=next_row_bins;next.previous_axis=axis;
                next.original_owner=previous.original_owner;
            } else for(const auto& child:children) {
                for(int k=0;k<2;++k)next.cap_weights[k]=next.cap_weights[k]||child.cap_weights[k];
                next.flux+=child.flux;next.signed_error+=child.signed_error;next.remainder+=child.remainder;next.work_estimate+=child.work_estimate;
                for(int k=0;k<6;++k)next.outer_parts[k]+=child.outer_parts[k];
                next.inner_signed+=child.inner_signed;next.outer_signed+=child.outer_signed;next.inner_remainder+=child.inner_remainder;next.outer_remainder+=child.outer_remainder;next.short_guard+=child.short_guard;
                for(std::size_t j=0;j<child.fill.runs.size();++j) {
                    next.fill.runs.push_back({child.fill.runs[j].run,0});next.fill.run_areas.push_back(child.fill.run_areas[j]);
                    next.fill.run_flux_errors.push_back(child.fill.run_flux_errors[j]);next.fill.run_error_fractions.push_back(child.fill.run_error_fractions[j]);
                    if(child.fill.run_cores.size()==child.fill.runs.size())next.fill.run_cores.push_back(child.fill.run_cores[j]);
                }
            }
            if(children.empty() || !(std::isfinite(next.flux)&&next.flux>0&&std::isfinite(next.signed_error)&&std::isfinite(next.remainder)&&next.remainder>=0)
                || !refined_run_footprint_is_private(next.fill,next_bins/settings.source_bins,run_fill,previous.original_owner,next_row_bins/settings.source_bins)) {
                previous.blocked=true;adaptive->stop_reason="unresolved_component";record_trial(false);continue;
            }
            const double next_priority=std::abs(next.signed_error-correction_for(next))+next.remainder;
            next.order=next_priority>priority/factor?1:2;
            record_trial(true);
            adaptive->refinement_grid_bins.push_back({double(selected),double(previous.bins),double(previous.row_bins),double(next_bins),double(next_row_bins),double(axis)});
            previous=std::move(next);++adaptive->refinements;
            adaptive->largest_bins=std::max({adaptive->largest_bins,next_bins,next_row_bins});current=totals();
            adaptive->history.push_back({double(next_bins),current[1],double(adaptive->new_maps),double(adaptive->reused_maps),double(selected)});
            record_cost(current[0]);
        }
        adaptive->met=adaptive->unresolved_centres==0 && std::isfinite(current[1])&&current[1]<=adaptive->target;
        if(adaptive->unresolved_centres)adaptive->stop_reason="unresolved_centre_image";
        if(adaptive->met)adaptive->stop_reason="estimated_target_met";
        for(const auto& c:retained){const double correction=correction_for(c);if(adaptive->local_vertical_caps)adaptive->mixed_component_grids.push_back({double(c.bins),double(c.row_bins),double(c.retained_cap_bins)});adaptive->components.push_back({double(std::max({c.bins,c.row_bins,c.retained_cap_bins})),c.flux+correction,c.signed_error-correction,c.remainder,double(c.fill.runs.size())});adaptive->component_grid_bins.push_back({double(c.bins),double(c.row_bins)});}
        total_area=current[0]/(incr*row_spacing);
        if(phase_trace)adaptive->adaptive_control_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-control_started).count()-(adaptive->run_fill_seconds+adaptive->error_aggregation_seconds-children_seconds_before);
    }
    return static_cast<double>(total_area);
}

template <typename ImageMap>
double inverse_ray_cartesian_core(
    const ImageMap& mapper,
    const std::vector<SourcePosition>& raw_images,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    double point_source_magnification_hint,
    LegacyAreaDiagnostics* diagnostics,
    const char* diagnostics_label,
    detail::CartesianRunFillTrace* trace = nullptr,
    const std::vector<std::vector<SourcePosition>>* critical_coverage = nullptr,
    const std::vector<std::vector<SourcePosition>>* caustic_coverage = nullptr)
{
    if (raw_images.empty() || source_radius <= 0.0) {
        return std::nan("");
    }
    const double nbin = static_cast<double>(std::max(settings.source_bins, 1));
    const double incr = source_radius / nbin;
    const char* fill_selector = std::getenv("LCBININT_CARTESIAN_FILL");
    // The deterministic run fill is the production default.  `legacy` keeps
    // the former walker available for diagnostics and rollback comparisons;
    // the diagnostic unsorted-seed mode also deliberately selects that path.
    const bool use_multirun = trace != nullptr ||
        ((fill_selector == nullptr || std::string(fill_selector) == "run") &&
        std::getenv("LCBININT_DIAGNOSTIC_UNSORTED_SEEDS") == nullptr);
    auto images = lattice_snapped_seeds(
        mapper, source, source_radius, incr, raw_images, !use_multirun);
    struct EvaluatedSeed {
        SourcePosition position;
        double jacobian;
    };
    std::vector<EvaluatedSeed> evaluated_images;
    evaluated_images.reserve(images.size());
    for (const auto& image : images) {
        evaluated_images.push_back({
            image, lens_jacobian(mapper, image.x, image.y)});
    }
    // Diagnostic-only perturbation used by the tangency correctness corpus.
    // Keeping this below lattice snapping ensures the two runs have exactly
    // the same seed cells and differ only in flood-fill order.  Production
    // canonicalises the result below; the legacy diagnostic deliberately does
    // not, so it can reproduce the former order dependence.
    if (const char* order = std::getenv("LCBININT_DIAGNOSTIC_SEED_ORDER")) {
        const std::string order_text(order);
        if (order_text == "fold-first" || order_text == "fold-last") {
            const bool fold_first = order_text == "fold-first";
            std::stable_sort(
                evaluated_images.begin(), evaluated_images.end(),
                [&](const auto& lhs, const auto& rhs) {
                const bool lhs_fold =
                    std::abs(lhs.jacobian) < kFoldJacobianThreshold;
                const bool rhs_fold =
                    std::abs(rhs.jacobian) < kFoldJacobianThreshold;
                return lhs_fold != rhs_fold && lhs_fold == fold_first;
            });
        } else if (order_text == "jacobian-ascending" ||
                   order_text == "jacobian-descending") {
            const bool ascending = order_text == "jacobian-ascending";
            std::stable_sort(
                evaluated_images.begin(), evaluated_images.end(),
                [&](const auto& lhs, const auto& rhs) {
                const double lhs_abs = std::abs(lhs.jacobian);
                const double rhs_abs = std::abs(rhs.jacobian);
                return ascending ? lhs_abs < rhs_abs : lhs_abs > rhs_abs;
            });
        } else if (order_text == "reverse") {
            std::reverse(evaluated_images.begin(), evaluated_images.end());
        }
    }
    if (std::getenv("LCBININT_DIAGNOSTIC_UNSORTED_SEEDS") == nullptr) {
        std::stable_sort(
            evaluated_images.begin(), evaluated_images.end(),
            [&](const auto& lhs, const auto& rhs) {
            const double lhs_abs = std::abs(lhs.jacobian);
            const double rhs_abs = std::abs(rhs.jacobian);
            if (lhs_abs != rhs_abs) {
                return lhs_abs > rhs_abs;
            }
            return lhs.position.y != rhs.position.y
                ? lhs.position.y < rhs.position.y
                : lhs.position.x < rhs.position.x;
        });
    }
    if (evaluated_images.empty()) {
        return std::nan("");
    }
    double walk_magnification_hint = point_source_magnification_hint;
    for (const auto& image : evaluated_images) {
        const double jacobian = image.jacobian;
        if (std::isfinite(jacobian) && std::abs(jacobian) > 1.0e-15) {
            walk_magnification_hint = std::max(
                walk_magnification_hint, 1.0 / std::abs(jacobian));
        }
    }
    if (diagnostics != nullptr) {
        *diagnostics = {};
        diagnostics->seed_count = static_cast<int>(evaluated_images.size());
    }
    double area = 0.0;
    if (use_multirun) {
        if (diagnostics != nullptr) {
            diagnostics->parity_free_topology = true;
        }
        std::vector<EvaluatedCartesianSeed> multirun_seeds;
        multirun_seeds.reserve(evaluated_images.size());
        for (const auto& seed : evaluated_images) {
            multirun_seeds.push_back({seed.position, seed.jacobian});
            if (diagnostics != nullptr &&
                std::abs(seed.jacobian) < kFoldJacobianThreshold) {
                ++diagnostics->fold_seed_count;
            }
        }
        std::vector<detail::CartesianLatticeSeed> coverage_probes;
        const char* seed_mode=std::getenv("LCBININT_REPRESENTATIVE_PROBES");
        const bool geometry_guard=std::getenv("LCBININT_CARTESIAN_GEOMETRY_OUTER_GUARD")!=nullptr
            || (seed_mode && std::string(seed_mode)=="critical-seeds");
        bool source_has_critical=false;
        if (geometry_guard && caustic_coverage) {
            const double r2=source_radius*source_radius;
            for (const auto& branch:*caustic_coverage) {
                for (const auto& point:branch) {
                    if (distance_squared(point,source)<r2) {source_has_critical=true;break;}
                }
                if (source_has_critical) break;
            }
        }
        if (seed_mode && std::string(seed_mode)=="critical-seeds" && critical_coverage && caustic_coverage
            && std::getenv("LCBININT_CRITICAL_GRID_REPAIR")) {
            const auto& critical=*critical_coverage;
            const auto& caustics=*caustic_coverage;
            const double r2=source_radius*source_radius;
            for (std::size_t branch=0;branch<critical.size() && branch<caustics.size();++branch)
                for (std::size_t i=0;i<critical[branch].size() && i<caustics[branch].size();++i) {
                    if (distance_squared(caustics[branch][i],source)>=r2) continue;
                    std::int64_t ix=0,iy=0;
                    if (!rounded_lattice_index(critical[branch][i].x/incr,ix)
                        || !rounded_lattice_index(critical[branch][i].y/incr,iy)) continue;
                    if (!coverage_probes.empty() && coverage_probes.back().ix==ix && coverage_probes.back().iy==iy) continue;
                    coverage_probes.push_back({ix,iy});
                }
        }
        area = fill_all_cartesian_components_multirun(
            mapper, source, source_radius, settings, finite_magnifier,
            multirun_seeds, incr, walk_magnification_hint, diagnostics, trace,
            coverage_probes.empty() ? nullptr : &coverage_probes,source_has_critical);
        if (!std::isfinite(area)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    } else {
    ClaimedCellRuns claimed;
    // Components are filled serially, and every entry read by a fill is first
    // written by that fill.  Retaining the largest allocation across fills
    // avoids rebuilding five growable row arrays for every seed/refinement.
    LegacyImageAreaScratch scratch;

    for (std::size_t image_index = 0;
         image_index < evaluated_images.size(); ++image_index) {
        const SourcePosition seed = evaluated_images[image_index].position;
        std::int64_t seed_ix = 0;
        std::int64_t seed_iy = 0;
        if (!rounded_lattice_index(seed.x / incr, seed_ix) ||
            !rounded_lattice_index(seed.y / incr, seed_iy)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // A seed on an already-counted cell lies inside a component another
        // fill has traced; its fill could only wander already-claimed rows.
        if (claimed.find(seed_iy, seed_ix) != nullptr) {
            continue;
        }
        // Guard fold-image flood-fills against crossing the critical curve.
        // When the source disk straddles the caustic, both fold images (F+ and F-)
        // map into the source disk; without this guard the x-scan bleeds across the
        // critical curve from one fold image into the other, giving wrong (sometimes
        // negative) magnifications.
        //
        // Only seeds sufficiently near a fold need this per-cell parity guard.
        // The canonical high-|J| seed ordering above makes the unguarded
        // component ownership independent of the order in which probes were
        // discovered.
        const double J_seed = evaluated_images[image_index].jacobian;
        const int jac_sign = std::abs(J_seed) < kFoldJacobianThreshold
            ? (J_seed > 0.0 ? 1 : J_seed < 0.0 ? -1 : 0)
            : 0;

        if (diagnostics != nullptr) {
            ++diagnostics->processed_images;
            if (std::abs(J_seed) < kFoldJacobianThreshold) {
                ++diagnostics->fold_seed_count;
            }
        }

        // Stamp this fill's claims so a later fill can tell whose territory it
        // walked into; the refinement below is only sound for a component
        // whose extent nobody else decided.
        claimed.owner = static_cast<int>(image_index) + 1;
        const ComponentFill fill = fill_image_component(
            mapper, source, source_radius, settings, finite_magnifier, seed, incr,
            scratch, jac_sign, claimed, walk_magnification_hint, diagnostics);

        // The claimed-cell registry guarantees each cell is counted at most
        // once across fills, so no cross-seed overlap correction is needed:
        // redundant fills simply contribute zero.
        double contribution = fill.area;

        const int factor = std::isfinite(fill.area)
            ? component_refinement_factor(fill, settings.source_bins)
            : 1;
        if (factor > 1) {
            // Components of f^-1(D) are disjoint, and the lattice only puts two
            // of them in contact where their images merge -- on a critical
            // curve.  So a coarse fill that ended against another component's
            // cells is one member of a fold pair, and the parity guard is the
            // boundary condition that holds it to its own side without the
            // neighbour's claims to lean on.  `kFoldJacobianThreshold` does not
            // apply: the evidence here is the contact, not the size of |J|.
            const int refined_sign = fill.foreign_contact && jac_sign == 0
                ? (J_seed > 0.0 ? 1 : J_seed < 0.0 ? -1 : 0)
                : jac_sign;
            ClaimedCellRuns refined_claimed;
            FiniteSourceSettings refined_settings = settings;
            refined_settings.source_bins = settings.source_bins * factor;
            const double refined_incr = incr / static_cast<double>(factor);
            const auto refined_seeds = lattice_snapped_seeds(
                mapper, source, source_radius, refined_incr, {seed}, true);
            // The refined walk is supposed to cover the coarse footprint, k^2
            // finer, plus the few cells per row the scan overshoots by; the
            // margin is generous because the cost of the budget being wrong is
            // a refinement declined, not an answer.  It matters because the
            // walk is alone on an empty lattice: without the neighbours' claims
            // to stop it, a component whose coarse extent was decided by those
            // claims is free to run away into the rest of the image.
            std::int64_t refined_budget = static_cast<std::int64_t>(std::ceil(
                8.0 * fill.width * static_cast<double>(fill.rows_span) *
                    static_cast<double>(factor) * static_cast<double>(factor) +
                4096.0));
            if (!refined_seeds.empty()) {
                const ComponentFill refined = fill_image_component(
                    mapper, source, source_radius, refined_settings, finite_magnifier,
                    refined_seeds.front(), refined_incr, scratch, refined_sign, refined_claimed,
                    walk_magnification_hint, nullptr, &refined_budget);
                // Cell counts scale with the lattice, the area they measure
                // does not: k^-2 puts the refined count back in coarse cells.
                const double rescaled =
                    refined.area / (static_cast<double>(factor) * static_cast<double>(factor));
                if (std::getenv("LCBININT_AREA_DIAGNOSTICS")) {
                    std::fprintf(stderr,
                        "  COMPONENT bins=%d k=%d narrow=%.3g rows_span=%d width=%.3g "
                        "coarse=%.10g refined=%.10g ratio=%.6g fine_rows_span=%d "
                        "fine_width=%.4g seed=(%.17g,%.17g) jac=%d\n",
                        settings.source_bins, factor, fill.narrow_cells(), fill.rows_span,
                        fill.width, fill.area, rescaled, rescaled / fill.area,
                        refined.rows_span, refined.width, seed.x, seed.y, jac_sign);
                }
                if (std::isfinite(rescaled) && rescaled > 0.0 &&
                    claim_refined_footprint(refined_claimed, factor, claimed)) {
                    contribution = rescaled;
                    if (diagnostics != nullptr) {
                        ++diagnostics->refined_components;
                        diagnostics->refinement_factor =
                            std::max(diagnostics->refinement_factor, factor);
                    }
                }
            }
        }

        area += contribution;

        if (diagnostics != nullptr) {
            diagnostics->boundary_rows += fill.boundary_rows;
        }
    }
    if (diagnostics != nullptr && std::getenv("LCBININT_AREA_DIAGNOSTICS")) {
        for (const auto& row : claimed.rows) {
            for (const auto& run : row.second) {
                diagnostics->integrated_cells += run.hi - run.lo + 1;
            }
        }
    }
    }

    const double scale =
        source_flux(source_radius, settings) / (source_radius * source_radius) * nbin * nbin;
    const double magnification = area / scale;
    if (diagnostics != nullptr) {
        diagnostics->estimated_error =
            cartesian_area_error_indicator(*diagnostics, source_radius, settings);
        const double floor_coefficient =
            high_magnification_floor_coefficient(*diagnostics, magnification, source_radius);
        if (floor_coefficient > 0.0) {
            const double high_magnification_floor =
                std::abs(magnification) *
                (floor_coefficient / static_cast<double>(std::max(settings.source_bins, 1)));
            diagnostics->estimated_error =
                std::max(diagnostics->estimated_error, high_magnification_floor);
        }
        if (std::getenv("LCBININT_AREA_DIAGNOSTICS")) {
            std::fprintf(stderr,
                "%s bins=%d seeds=%d processed=%d fold=%d rows=%d intervals=%d gaps=%d overlaps=%d "
                "refined=%d/%d maxjump=%.3g mag=%.8g err=%.8g "
                "mapped=%lld frontier=%lld unique_seeds=%lld duplicate_starts=%lld "
                "components=%lld merged=%lld multi_rows=%lld max_row_runs=%zu "
                "fine_mapped=%lld budget_failures=%lld cells=%lld\n",
                diagnostics_label, settings.source_bins, diagnostics->seed_count,
                diagnostics->processed_images, diagnostics->fold_seed_count,
                diagnostics->boundary_rows, diagnostics->interval_count,
                diagnostics->gap_repairs, diagnostics->overlaps,
                diagnostics->refined_components, diagnostics->refinement_factor,
                diagnostics->max_jump_cells, magnification, diagnostics->estimated_error,
                static_cast<long long>(diagnostics->mapped_cell_evaluations),
                static_cast<long long>(diagnostics->frontier_intervals_popped),
                static_cast<long long>(diagnostics->unique_seed_cells),
                static_cast<long long>(diagnostics->duplicate_seed_starts_avoided),
                static_cast<long long>(diagnostics->provisional_components),
                static_cast<long long>(diagnostics->merged_components),
                static_cast<long long>(diagnostics->rows_with_multiple_runs),
                diagnostics->maximum_runs_in_row,
                static_cast<long long>(diagnostics->fine_grid_cell_evaluations),
                static_cast<long long>(diagnostics->walk_budget_failures),
                static_cast<long long>(diagnostics->integrated_cells));
        }
    }
    return magnification;
}

double inverse_ray_cartesian_binary_mag(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<SourcePosition>* precomputed_seeds = nullptr,
    LegacyAreaDiagnostics* diagnostics = nullptr,
    double point_source_magnification_hint =
        std::numeric_limits<double>::quiet_NaN(),
    detail::CartesianRunFillTrace* trace = nullptr)
{
    if ((settings.limb_darkening_c != 0.0 || settings.limb_darkening_d != 0.0) &&
        finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }

    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    auto computed_images = precomputed_seeds == nullptr ?
        augmented_image_seeds(
            point_magnifier, mapper, separation, mass_ratio, source, source_radius,
            std::numeric_limits<double>::infinity(), nullptr,
            finite_magnifier != nullptr
                ? &finite_magnifier->binary_caustic_branches(separation, mass_ratio)
                : nullptr,nullptr,nullptr,
            finite_magnifier != nullptr
                ? &finite_magnifier->binary_critical_curve_branches_cached(separation,mass_ratio)
                : nullptr) :
        std::vector<SourcePosition> {};
    const auto& raw_images = precomputed_seeds == nullptr ? computed_images : *precomputed_seeds;
    if(auto* work=adaptive_flux_request();work && work->centres.empty()) {
        // A cached augmented seed set may bypass its original centre metadata.
        // Recover only centre witnesses from those existing positions; no solve.
        for(const auto& p:raw_images)if(mapped_lens_distance2(mapper,p.x,p.y,source)<=source_radius*source_radius*1e-12)
            work->centres.push_back({p.x,p.y});
    }
    // The normal automatic path has no point-image result available and keeps
    // the historical fallback solve.  A preplanned epoch has already solved
    // the point images to build its seeds and point-source magnification; in
    // that case use the supplied value instead of solving the same point
    // lens a second time just to size the walk guard.
    const double point_source_hint = std::isfinite(point_source_magnification_hint)
        ? std::abs(point_source_magnification_hint)
        : std::abs(
              point_magnifier.binary_mag0(separation, mass_ratio, source).magnification);
    return inverse_ray_cartesian_core(
        mapper, raw_images, source, source_radius, settings, finite_magnifier,
        point_source_hint, diagnostics, "AREA_DIAGNOSTICS", trace,
        finite_magnifier ? &finite_magnifier->binary_critical_curve_branches_cached(separation,mass_ratio) : nullptr,
        finite_magnifier ? &finite_magnifier->binary_caustic_branches(separation,mass_ratio) : nullptr);
}

double inverse_ray_cartesian_triple_mag(
    const PointSourceMagnifier& point_magnifier,
    const model::TripleLensGeometry& geometry,
    const TripleCausticBranches& caustics,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<SourcePosition>* precomputed_seeds = nullptr,
    LegacyAreaDiagnostics* diagnostics = nullptr,
    bool* support_proven = nullptr,
    double point_source_magnification_hint =
        std::numeric_limits<double>::quiet_NaN())
{
    if ((settings.limb_darkening_c != 0.0 || settings.limb_darkening_d != 0.0) &&
        finite_magnifier != nullptr) {
        finite_magnifier->ensure_limb_darkening_table();
    }

    const TripleLensMapper mapper = make_triple_lens_mapper(geometry);
    auto computed_images = precomputed_seeds == nullptr ?
        augmented_triple_image_seeds(
            point_magnifier,
            geometry,
            source,
            source_radius,
            caustics,
            support_proven) :
        std::vector<SourcePosition> {};
    const auto& raw_images =
        precomputed_seeds == nullptr ? computed_images : *precomputed_seeds;
    const double point_source_hint = std::isfinite(point_source_magnification_hint)
        ? std::abs(point_source_magnification_hint)
        : std::abs(point_magnifier.triple_mag0(geometry, source).magnification);
    return inverse_ray_cartesian_core(
        mapper, raw_images, source, source_radius, settings, finite_magnifier,
        point_source_hint, diagnostics, "TRIPLE_AREA_DIAGNOSTICS");
}

FiniteSourceResult fixed_inverse_ray_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    FiniteSourceDecision decision,
    double caustic_distance = std::numeric_limits<double>::infinity(),
    double consistency_reference = std::numeric_limits<double>::quiet_NaN(),
    const std::vector<SourcePosition>* seed_hints = nullptr,
    bool prevalidated_accuracy = false,
    double point_source_magnification_hint =
        std::numeric_limits<double>::quiet_NaN())
{
    bool support_proven = true;
    auto seeds = finite_magnifier != nullptr
        ? finite_magnifier->cached_binary_image_seeds(
            point_magnifier,
            separation,
            mass_ratio,
            source,
            source_radius,
            caustic_distance,
            seed_hints,
            &support_proven)
        : augmented_image_seeds(
            point_magnifier,
            make_binary_lens_mapper(separation, mass_ratio),
            separation,
            mass_ratio,
            source,
            source_radius,
            caustic_distance,
            seed_hints,
            nullptr,
            &support_proven);
    if (decision.method == FiniteSourceMethod::inverse_ray_polar) {
        const auto evaluation = evaluate_polar_to_tolerance(
            settings,
            support_proven,
            [&](const FiniteSourceSettings& active, PolarAreaDiagnostics* diagnostics) {
                return inverse_ray_polar_boundary_binary(
                    point_magnifier, separation, mass_ratio, source, source_radius,
                    active, finite_magnifier, &seeds, diagnostics);
            },
            false);
        if (!std::isfinite(evaluation.magnification)) {
            return {evaluation.magnification, 0, decision, std::nan(""), 0, false};
        }
        return {
            evaluation.magnification,
            0,
            decision,
            evaluation.error_estimate,
            0,
            settings.automatic_source_bins
                ? support_proven && evaluation.seed_coverage_proven
                : evaluation.converged,
        };
    }

    FiniteSourceSettings active_settings = settings;
    LegacyAreaDiagnostics diagnostics;
    double magnification = inverse_ray_cartesian_binary_mag(
        point_magnifier, separation, mass_ratio, source, source_radius,
        active_settings, finite_magnifier, &seeds, &diagnostics,
        point_source_magnification_hint);
    support_proven=support_proven && diagnostics.seed_coverage_proven;
    if (!std::isfinite(magnification)) {
        return {magnification, 0, decision, std::nan(""), 0, false};
    }
    if (prevalidated_accuracy) {
        // warmup() already compared this exact route and resolution with an
        // independently witnessed reference.  Retain the support certificate,
        // but do not pay for an indicator-driven retry or a half-grid rerun.
        const double error_estimate = support_proven
            ? diagnostics.estimated_error
            : std::numeric_limits<double>::infinity();
        return {
            magnification, 0, decision, error_estimate, 0, support_proven,
        };
    }

    if (settings.automatic_source_bins) {
        // The calibrated empirical law is the complete automatic-resolution
        // decision.  Keep the cheap area indicator as a diagnostic, but do not
        // let it trigger another grid or veto a supported one-shot result.
        // The run union exposes every interval in a multi-run component.  Its
        // boundary-only indicator is consequently less pessimistic than the
        // legacy walk's gap/jump indicator.  For large disks, where many
        // separate image intervals share the source footprint, retain a
        // conservative factor so a supported result does not under-report its
        // error.  Small-source calibration remains on the established scale.
        const bool large_source_run_union = diagnostics.parity_free_topology &&
            source_radius >= 5.0e-2;
        const double automatic_area_indicator_scale =
            large_source_run_union ? 2.0 : 0.25;
        const double error_estimate = support_proven
            ? automatic_area_indicator_scale * diagnostics.estimated_error
            : std::numeric_limits<double>::infinity();
        return {
            magnification, 0, decision, error_estimate, 0, support_proven,
        };
    }

    // Fixed low nbin is diagnostic-only.  With an explicit tolerance it keeps
    // the independent half-resolution compatibility check below, but it never
    // changes the caller's requested resolution.
    auto diagnose = [&](double value, const LegacyAreaDiagnostics& current_diagnostics) {
        double error = current_diagnostics.estimated_error;
        // The support certificate is resolution independent, so a finer grid
        // cannot repair an unproven one.  Refusing convergence here is what
        // keeps a silently missing component from being reported as an
        // accurate value: the area indicator only measures the boundary of the
        // components that were found.
        return assess_finite_source_error(
            settings, value, error, support_proven);
    };

    const auto assessment = diagnose(magnification, diagnostics);
    double error_estimate = assessment.estimate;
    bool converged = assessment.converged;
    auto evaluate_at = [&](const FiniteSourceSettings& grid) {
        return inverse_ray_cartesian_binary_mag(
            point_magnifier, separation, mass_ratio, source, source_radius,
            grid, finite_magnifier, &seeds, nullptr,
            point_source_magnification_hint);
    };
    // Fixed-grid explicit-tolerance calls retain one independent
    // half-resolution check.  Automatic calls use the same calibrated area
    // estimate and common budget regardless of whether tol/reltol was
    // explicitly supplied.
    if (support_proven && std::isfinite(error_estimate) &&
        has_explicit_finite_source_tolerance(settings)) {
        reconcile_with_half_resolution(
            settings, active_settings, magnification, evaluate_at,
            error_estimate, converged,
            cartesian_error_convergence_order(diagnostics));
    }
    return {magnification, 0, decision, error_estimate, 0, converged};
}

// ---------- image-spine kernel (finite_mode = 3) ----------
// Ported from lcbinint-idea-stable commit eb2c08e.
// Timing instrumentation omitted; timing pointers in stable are replaced by
// passing nullptr at all call sites.

constexpr double kLocal7LambdaMin = 1.0e-5;
constexpr double kLocal7SpineAreaJacMin = 100.0;
constexpr double kLocal7SpineDetMax = 1.0e-2;
constexpr double kLocal7SpineMaxStepCells = 4.0;
constexpr double kLocal7SpineMinStepCells = 0.125;
constexpr int kLocal7SpineMaxPoints = 200000;
constexpr int kLocal7SpineMaxNormalSamples = 20000000;
constexpr int kLocal7SpineOutsideStop = 3;
constexpr double kLocal7SpineCurvatureMax = 0.55;
constexpr double kLocal7SpineNormalSubstep = 1.0;
constexpr double kLocal7SpineTargetTolCells = 2.0;
constexpr double kLocal7SpineFrameDetMin = 1.0e-9;
constexpr double kLocal7SpineFrameLambdaMin = 1.0e-9;
constexpr double kLocal7SpineFrameAreaJacMax = 1.0e12;
constexpr double kLocal7SpinePairDistanceCells = 50000.0;
constexpr double kLocal7SpineMaxRelativeArea = 1.0e8;
// var_ratio = 2*beta*rho / (lambda_s * |lambda_l|) measures how much lambda_l
// changes across the source disk.  When this exceeds ~2 the linear fold model
// breaks down and the spine gives errors of several percent.  Skip to cartesian.
constexpr double kLocal7SpineMaxVarRatio = 2.0;

struct Local7Frame {
    Complex za;
    SourcePosition wa;
    SourcePosition sa;
    double gamma_r = 0.0;
    double gamma_i = 0.0;
    double beta_r = 0.0;
    double beta_i = 0.0;
    double kappa_r = 0.0;
    double kappa_i = 0.0;
    double lambda_l = 0.0;
    double lambda_s = 0.0;
    double det_j = 0.0;
    double area_jac = 0.0;
    double e_lx = 0.0;
    double e_ly = 0.0;
    double e_sx = 0.0;
    double e_sy = 0.0;
    bool ok = false;
};

bool local7_derivatives_binary(
    Complex z,
    const BinaryLensMapper& mapper,
    double* gr,
    double* gi,
    double* br,
    double* bi,
    double* kr,
    double* ki)
{
    *gr = *gi = *br = *bi = *kr = *ki = 0.0;
    const Complex lenses[2] = {mapper.separation, Complex(0.0, 0.0)};
    const double masses[2] = {mapper.m1, mapper.m2};
    for (int i = 0; i < 2; ++i) {
        const double dx = lenses[i].real() - z.real();
        const double dy = z.imag() - lenses[i].imag();
        const double r2 = dx * dx + dy * dy;
        if (r2 <= 1.0e-30 || !std::isfinite(r2)) return false;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double dx2 = dx * dx, dy2 = dy * dy;
        const double dx3 = dx2 * dx, dy3 = dy2 * dy;
        const double dx4 = dx2 * dx2, dy4 = dy2 * dy2;
        const double mass = masses[i];
        *gr += mass * (dx2 - dy2) / r4;
        *gi += mass * (-2.0 * dx * dy) / r4;
        *br += 2.0 * mass * dx * (dx2 - 3.0 * dy2) / r6;
        *bi += 2.0 * mass * (dy3 - 3.0 * dx2 * dy) / r6;
        *kr += 6.0 * mass * (dx4 - 6.0 * dx2 * dy2 + dy4) / r8;
        *ki += 6.0 * mass * (-4.0 * dx3 * dy + 4.0 * dx * dy3) / r8;
    }
    return std::isfinite(*gr) && std::isfinite(*gi) && std::isfinite(*br) &&
           std::isfinite(*bi) && std::isfinite(*kr) && std::isfinite(*ki);
}

bool local7_make_frame(
    Complex za,
    SourcePosition source,
    const BinaryLensMapper& mapper,
    Local7Frame* frame)
{
    double gr, gi, br, bi, kr, ki;
    if (!local7_derivatives_binary(za, mapper, &gr, &gi, &br, &bi, &kr, &ki)) return false;
    const double g = std::hypot(gr, gi);
    const double lambda_s = 1.0 + g;
    const double lambda_l = 1.0 - g;
    const double det_j = lambda_l * lambda_s;
    const double abs_det = std::abs(det_j);
    if (abs_det <= 0.0 || !std::isfinite(abs_det)) return false;
    const double phi = 0.5 * std::atan2(gi, gr);
    const SourcePosition wa = map_binary_lens_real(mapper, za.real(), za.imag());
    *frame = {};
    frame->za = za;
    frame->wa = wa;
    frame->sa = {wa.x - source.x, wa.y - source.y};
    frame->gamma_r = gr; frame->gamma_i = gi;
    frame->beta_r = br;  frame->beta_i = bi;
    frame->kappa_r = kr; frame->kappa_i = ki;
    frame->lambda_l = lambda_l;
    frame->lambda_s = lambda_s;
    frame->det_j = det_j;
    frame->area_jac = 1.0 / abs_det;
    frame->e_lx = -std::sin(phi);
    frame->e_ly =  std::cos(phi);
    frame->e_sx =  std::cos(phi);
    frame->e_sy =  std::sin(phi);
    frame->ok = std::isfinite(frame->area_jac);
    return frame->ok;
}

Complex local7_apply_inverse_linear(const Local7Frame& frame, double sx, double sy)
{
    const double dx = sx - frame.sa.x;
    const double dy = sy - frame.sa.y;
    const double xi  = dx * frame.e_lx + dy * frame.e_ly;
    const double eta = dx * frame.e_sx + dy * frame.e_sy;
    return {
        (xi / frame.lambda_l) * frame.e_lx + (eta / frame.lambda_s) * frame.e_sx,
        (xi / frame.lambda_l) * frame.e_ly + (eta / frame.lambda_s) * frame.e_sy,
    };
}

Complex local7_apply_inverse_jacobian(const Local7Frame& frame, Complex residual)
{
    const double xi  = residual.real() * frame.e_lx + residual.imag() * frame.e_ly;
    const double eta = residual.real() * frame.e_sx + residual.imag() * frame.e_sy;
    return {
        (xi / frame.lambda_l) * frame.e_lx + (eta / frame.lambda_s) * frame.e_sx,
        (xi / frame.lambda_l) * frame.e_ly + (eta / frame.lambda_s) * frame.e_sy,
    };
}

Complex local7_correct_quadratic(const Local7Frame& frame, Complex dz0)
{
    const Complex beta(frame.beta_r, frame.beta_i);
    const Complex residual = 0.5 * beta * std::conj(dz0) * std::conj(dz0);
    return dz0 - local7_apply_inverse_jacobian(frame, residual);
}

Complex local7_approx_image(const Local7Frame& frame, double sx, double sy)
{
    const Complex dz0 = local7_apply_inverse_linear(frame, sx, sy);
    return frame.za + local7_correct_quadratic(frame, dz0);
}

struct Local7SpinePoint {
    SourcePosition image;
    SourcePosition source_offset;
    Local7Frame frame;
    double half_weight = 0.0;
};

struct Local7SpineEligibility {
    bool ok = false;
    std::size_t pair_index = 0;
    int reason = 0;
};

bool local7_is_spine_candidate(const Local7Frame& frame)
{
    return frame.ok &&
           std::isfinite(frame.area_jac) &&
           frame.area_jac >= kLocal7SpineAreaJacMin &&
           std::abs(frame.det_j) <= kLocal7SpineDetMax;
}

Local7SpineEligibility local7_spine_eligibility(
    const std::vector<SourcePosition>& seeds,
    const std::vector<int>& overlap,
    std::size_t image_index,
    const Local7Frame& frame,
    SourcePosition source,
    double source_step,
    const BinaryLensMapper& mapper,
    int caustic_born_branches)
{
    if (caustic_born_branches <= 0) return {false, 0, 30};
    if (!local7_is_spine_candidate(frame)) return {false, 0, 31};

    const auto nearest_partner = [&](std::size_t from_index, const Local7Frame& from_frame) {
        double best_distance = std::numeric_limits<double>::infinity();
        std::size_t best_index = seeds.size();
        int candidate_count = 0;
        for (std::size_t other = 0; other < seeds.size(); ++other) {
            if (other == from_index || overlap[other] == 1) continue;
            Local7Frame other_frame;
            if (!local7_make_frame(Complex(seeds[other].x, seeds[other].y), source, mapper, &other_frame) ||
                !local7_is_spine_candidate(other_frame) ||
                std::signbit(other_frame.det_j) == std::signbit(from_frame.det_j)) {
                continue;
            }
            const double image_distance =
                std::hypot(seeds[other].x - seeds[from_index].x, seeds[other].y - seeds[from_index].y);
            if (image_distance > kLocal7SpinePairDistanceCells * source_step) continue;
            ++candidate_count;
            if (image_distance < best_distance) {
                best_distance = image_distance;
                best_index = other;
            }
        }
        return std::pair<std::size_t, int> {best_index, candidate_count};
    };

    const auto [partner_index, partner_count] = nearest_partner(image_index, frame);
    if (partner_count < 1 || partner_index >= seeds.size()) return {false, 0, 33};
    if (partner_count > 1) {
        Local7Frame partner_frame;
        if (!local7_make_frame(Complex(seeds[partner_index].x, seeds[partner_index].y), source, mapper, &partner_frame))
            return {false, 0, 34};
        const auto [mutual_index, mutual_count] = nearest_partner(partner_index, partner_frame);
        if (mutual_count < 1 || mutual_index != image_index) return {false, 0, 35};
    }
    if (partner_index < image_index) return {false, 0, 36};
    return {true, partner_index, 0};
}

double local7_spine_step(const Local7Frame& frame, double source_step, double source_radius)
{
    const double abs_lambda = std::max(std::abs(frame.lambda_l), kLocal7LambdaMin);
    double step = source_step / abs_lambda;
    const double beta_abs = std::hypot(frame.beta_r, frame.beta_i);
    if (beta_abs > 0.0 && std::isfinite(beta_abs)) {
        const double nonlinear_cap =
            2.0 * source_radius / (abs_lambda + std::sqrt(abs_lambda * abs_lambda + 2.0 * beta_abs * source_radius));
        if (std::isfinite(nonlinear_cap) && nonlinear_cap > 0.0) step = std::min(step, nonlinear_cap);
    }
    step = std::min(step, kLocal7SpineMaxStepCells * source_step);
    step = std::max(step, kLocal7SpineMinStepCells * source_step);
    return step;
}

bool local7_spine_frame_safe(const Local7Frame& frame)
{
    return frame.ok && std::isfinite(frame.area_jac) &&
           std::abs(frame.lambda_l) >= kLocal7SpineFrameLambdaMin &&
           std::abs(frame.lambda_s) >= kLocal7SpineFrameLambdaMin &&
           std::abs(frame.det_j) >= kLocal7SpineFrameDetMin &&
           frame.area_jac <= kLocal7SpineFrameAreaJacMax;
}

bool local7_spine_try_step(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    double source_step,
    SourcePosition current,
    SourcePosition current_source_offset,
    const Local7Frame& current_frame,
    double signed_step,
    Local7SpinePoint* output,
    int* fail_reason)
{
    double step = signed_step;
    const double min_abs_step = kLocal7SpineMinStepCells * source_step;
    int last_reason = 25;
    for (int attempt = 0; attempt < 12; ++attempt) {
        const SourcePosition candidate {
            current.x + step * current_frame.e_lx,
            current.y + step * current_frame.e_ly,
        };
        SourcePosition target_offset {
            current_source_offset.x + current_frame.lambda_l * step * current_frame.e_lx,
            current_source_offset.y + current_frame.lambda_l * step * current_frame.e_ly,
        };
        if (target_offset.x * target_offset.x + target_offset.y * target_offset.y >
            source_radius * source_radius) {
            last_reason = 20;
            step *= 0.5;
            if (std::abs(step) < min_abs_step) break;
            continue;
        }
        SourcePosition corrected = candidate;
        Local7Frame candidate_frame;
        bool frame_ok = false;
        double residual_norm = std::numeric_limits<double>::infinity();
        for (int newton = 0; newton < 5; ++newton) {
            const SourcePosition mapped = map_binary_lens_real(mapper, corrected.x, corrected.y);
            frame_ok = local7_make_frame(Complex(corrected.x, corrected.y), source, mapper, &candidate_frame) &&
                       local7_spine_frame_safe(candidate_frame);
            if (!frame_ok) {
                last_reason = newton == 0 ? 21 : 22;
                break;
            }
            const Complex residual(
                mapped.x - (source.x + target_offset.x),
                mapped.y - (source.y + target_offset.y));
            residual_norm = std::abs(residual);
            if (residual_norm <= kLocal7SpineTargetTolCells * source_step) break;
            const Complex dz = local7_apply_inverse_jacobian(candidate_frame, residual);
            double damping = 1.0;
            const double dz_abs = std::abs(dz);
            const double max_dz = 4.0 * std::max(std::abs(step), source_step);
            if (dz_abs > max_dz && dz_abs > 0.0) damping = max_dz / dz_abs;
            corrected.x -= damping * dz.real();
            corrected.y -= damping * dz.imag();
        }
        if (frame_ok) {
            const double dot =
                current_frame.e_lx * candidate_frame.e_lx + current_frame.e_ly * candidate_frame.e_ly;
            if (residual_norm <= kLocal7SpineTargetTolCells * source_step &&
                std::abs(dot) >= std::cos(kLocal7SpineCurvatureMax)) {
                *output = {corrected, target_offset, candidate_frame, 0.0};
                return true;
            }
            last_reason = residual_norm > kLocal7SpineTargetTolCells * source_step ? 23 : 24;
        }
        step *= 0.5;
        if (std::abs(step) < min_abs_step) break;
    }
    if (fail_reason != nullptr) *fail_reason = last_reason;
    return false;
}

bool local7_build_spine_direction(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    double source_step,
    const Local7SpinePoint& seed,
    double sign,
    std::vector<Local7SpinePoint>* points,
    int* fail_reason)
{
    SourcePosition current = seed.image;
    SourcePosition current_source_offset = seed.source_offset;
    Local7Frame frame = seed.frame;
    int guard = 0;
    while (++guard < kLocal7SpineMaxPoints) {
        const double step = sign * local7_spine_step(frame, source_step, source_radius);
        Local7SpinePoint next;
        if (!local7_spine_try_step(
                mapper, source, source_radius, source_step,
                current, current_source_offset, frame, step, &next, fail_reason)) {
            return true;
        }
        points->push_back(next);
        current = next.image;
        current_source_offset = next.source_offset;
        frame = next.frame;
        if (static_cast<int>(points->size()) > kLocal7SpineMaxPoints) return false;
    }
    return false;
}

double local7_spine_integrate_normals(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const std::vector<Local7SpinePoint>& spine,
    double source_step,
    int* fallback_reason)
{
    const bool spine_debug = std::getenv("LCBININT_SPINE_DEBUG") != nullptr;
    const double radius2 = source_radius * source_radius;
    const double inv_radius2 = 1.0 / radius2;
    double area = 0.0;
    long long normal_samples = 0;
    for (std::size_t i = 0; i < spine.size(); ++i) {
        const auto& point = spine[i];
        const double tangent_weight = 2.0 * point.half_weight;
        if (!std::isfinite(tangent_weight) || tangent_weight <= 0.0) {
            *fallback_reason = 10;
            return std::nan("");
        }
        const double normal_step = std::min(
            std::max(source_step / std::max(std::abs(point.frame.lambda_s), kLocal7LambdaMin),
                kLocal7SpineMinStepCells * source_step),
            kLocal7SpineMaxStepCells * source_step) * kLocal7SpineNormalSubstep;
        const double cell_area = tangent_weight * normal_step;
        if (!std::isfinite(cell_area) || cell_area <= 0.0) {
            *fallback_reason = 11;
            return std::nan("");
        }
        for (int direction = -1; direction <= 1; direction += 2) {
            int outside = 0;
            for (int n = direction == -1 ? -1 : 0; ; n += direction) {
                const double offset = static_cast<double>(n) * normal_step;
                const SourcePosition image {
                    point.image.x + offset * point.frame.e_sx,
                    point.image.y + offset * point.frame.e_sy,
                };
                const SourcePosition mapped = map_binary_lens_real(mapper, image.x, image.y);
                ++normal_samples;
                if (normal_samples > kLocal7SpineMaxNormalSamples) {
                    *fallback_reason = 12;
                    return std::nan("");
                }
                const double dx = mapped.x - source.x;
                const double dy = mapped.y - source.y;
                const double q = (dx * dx + dy * dy) * inv_radius2;
                if (q <= 1.0) {
                    outside = 0;
                    area += source_limb_brightness(q, settings, finite_magnifier) * cell_area;
                } else {
                    ++outside;
                    if (outside >= kLocal7SpineOutsideStop) break;
                }
                if (std::abs(offset) >
                    4.0 * source_radius / std::max(std::abs(point.frame.lambda_s), kLocal7LambdaMin)) {
                    *fallback_reason = 13;
                    return std::nan("");
                }
            }
        }
    }
    if (spine_debug && !spine.empty()) {
        const double tw0 = 2.0 * spine[0].half_weight;
        const double tw_mid = 2.0 * spine[spine.size()/2].half_weight;
        const double ll0 = spine[0].frame.lambda_l;
        const double ll_mid = spine[spine.size()/2].frame.lambda_l;
        std::fprintf(stderr, "  normals: spine_pts=%zu normal_samples=%lld "
            "tw[0]=%.3e tw[mid]=%.3e ll[0]=%.3e ll[mid]=%.3e source_step=%.3e\n",
            spine.size(), normal_samples, tw0, tw_mid, ll0, ll_mid, source_step);
    }
    return area;
}

double local7_spine_area_binary(
    const BinaryLensMapper& mapper,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier,
    const Local7Frame& seed_frame,
    int* fallback_reason)
{
    *fallback_reason = 0;
    if (!local7_spine_frame_safe(seed_frame) || !local7_is_spine_candidate(seed_frame)) {
        *fallback_reason = 1;
        return std::nan("");
    }
    const int bins = std::max(settings.source_bins, 1);
    const double source_step = source_radius / static_cast<double>(bins);
    std::vector<Local7SpinePoint> minus_points, plus_points;
    minus_points.reserve(1024);
    plus_points.reserve(1024);
    const Local7SpinePoint seed {
        {seed_frame.za.real(), seed_frame.za.imag()},
        seed_frame.sa,
        seed_frame,
        0.0,
    };
    if (!local7_build_spine_direction(
            mapper, source, source_radius, source_step, seed, -1.0, &minus_points, fallback_reason) ||
        !local7_build_spine_direction(
            mapper, source, source_radius, source_step, seed, 1.0, &plus_points, fallback_reason)) {
        *fallback_reason = 2;
        return std::nan("");
    }
    std::vector<Local7SpinePoint> spine;
    spine.reserve(minus_points.size() + plus_points.size() + 1);
    for (auto it = minus_points.rbegin(); it != minus_points.rend(); ++it) spine.push_back(*it);
    spine.push_back(seed);
    for (const auto& pt : plus_points) spine.push_back(pt);
    if (spine.size() < 3) {
        if (*fallback_reason == 0) *fallback_reason = 3;
        return std::nan("");
    }
    for (std::size_t i = 0; i < spine.size(); ++i) {
        double left_cross = 0.0, right_cross = 0.0;
        double left_source = 0.0, right_source = 0.0;
        if (i > 0) {
            const double dx = spine[i].image.x - spine[i - 1].image.x;
            const double dy = spine[i].image.y - spine[i - 1].image.y;
            left_cross = std::abs(dx * spine[i].frame.e_sy - dy * spine[i].frame.e_sx);
            left_source = std::hypot(
                spine[i].source_offset.x - spine[i - 1].source_offset.x,
                spine[i].source_offset.y - spine[i - 1].source_offset.y);
        }
        if (i + 1 < spine.size()) {
            const double dx = spine[i + 1].image.x - spine[i].image.x;
            const double dy = spine[i + 1].image.y - spine[i].image.y;
            right_cross = std::abs(dx * spine[i].frame.e_sy - dy * spine[i].frame.e_sx);
            right_source = std::hypot(
                spine[i + 1].source_offset.x - spine[i].source_offset.x,
                spine[i + 1].source_offset.y - spine[i].source_offset.y);
        }
        if (i == 0) { left_cross = right_cross; left_source = right_source; }
        else if (i + 1 == spine.size()) { right_cross = left_cross; right_source = left_source; }
        spine[i].half_weight = 0.25 * (left_cross + right_cross);
        if (!std::isfinite(spine[i].half_weight) || spine[i].half_weight <= 0.0 ||
            left_source > kLocal7SpineMaxStepCells * source_step * 2.0 ||
            right_source > kLocal7SpineMaxStepCells * source_step * 2.0) {
            *fallback_reason = 4;
            return std::nan("");
        }
    }
    const double area = local7_spine_integrate_normals(
        mapper, source, source_radius, settings, finite_magnifier, spine, source_step, fallback_reason);
    const double total_source = source_flux(source_radius, settings);
    if (!std::isfinite(area) || area <= 0.0 ||
        !std::isfinite(total_source) || total_source <= 0.0 ||
        area / total_source > kLocal7SpineMaxRelativeArea) {
        if (*fallback_reason == 0) *fallback_reason = 5;
        return std::nan("");
    }
    return area;
}

FiniteSourceResult fixed_inverse_ray_spine_binary(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings,
    const FiniteSourceMagnifier* finite_magnifier)
{
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_spine,
        estimate_cartesian_cost(settings),
        "experimental image-spine guided scan",
    };

    // Generate the same centre-plus-certificate seed set as every other
    // public binary route.  All fallback paths reuse it.
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    auto seeds = finite_magnifier != nullptr
        ? finite_magnifier->cached_binary_image_seeds(
            point_magnifier, separation, mass_ratio, source, source_radius,
            std::numeric_limits<double>::infinity(), nullptr, nullptr)
        : augmented_image_seeds(
            point_magnifier, mapper, separation, mass_ratio, source, source_radius);

    const int n_ps = static_cast<int>(
        point_magnifier.binary_images(separation, mass_ratio, source).size());
    const int caustic_born = std::max(0, static_cast<int>(seeds.size()) - n_ps);
    const int bins = std::max(settings.source_bins, 1);
    const double source_step = source_radius / static_cast<double>(bins);

    // Boundary seeding may add many samples along the same extended fold arc.
    // The local spine kernel assumes isolated fold pairs; applying it to many
    // same-arc seeds double-counts the image area.  Use the robust cartesian
    // kernel for these multi-component / oversampled caustic crossings.
    if (caustic_born > 4) {
        FiniteSourceDecision fb {
            FiniteSourceMethod::inverse_ray_cartesian,
            estimate_cartesian_cost(settings),
            "spine skipped for multi-seed caustic crossing",
        };
        const double mag = inverse_ray_cartesian_binary_mag(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings, finite_magnifier, &seeds);
        if (!std::isfinite(mag)) return {mag, 0, fb, std::nan(""), 0, false};
        return {mag, 0, fb, 0.0, 0, true};
    }

    if (const char* dbg = std::getenv("LCBININT_SPINE_DEBUG")) {
        (void)dbg;
        std::fprintf(stderr, "SPINE_DEBUG seeds=%zu n_ps=%d caustic_born=%d rho=%.4e bins=%d\n",
            seeds.size(), n_ps, caustic_born, source_radius, bins);
        for (std::size_t di = 0; di < seeds.size(); ++di) {
            Local7Frame df;
            bool dok = local7_make_frame(Complex(seeds[di].x, seeds[di].y), source, mapper, &df);
            const double beta_abs = dok ? std::hypot(df.beta_r, df.beta_i) : 0.0;
        const double var_ratio = (dok && std::abs(df.lambda_l) > 1e-30 && std::abs(df.lambda_s) > 0.0)
            ? 2.0 * beta_abs * source_radius / (std::abs(df.lambda_s) * std::abs(df.lambda_l)) : 0.0;
        std::fprintf(stderr, "  seed[%zu] (%.4f,%.4f) frame_ok=%d area_jac=%.2f det_j=%.4f beta=%.4f var_ratio=%.2f candidate=%d\n",
                di, seeds[di].x, seeds[di].y, (int)dok,
                dok ? df.area_jac : 0.0, dok ? df.det_j : 0.0,
                beta_abs, var_ratio, dok && local7_is_spine_candidate(df));
        }
    }

    // Pass 1: try spine for caustic-born fold pairs.
    // Non-eligible seeds (PS images) are left untouched; spine-covered seeds
    // are marked so they are excluded from the cartesian pass.
    std::vector<bool> spine_covered(seeds.size(), false);
    std::vector<int> elig_overlap(seeds.size(), 0);
    double spine_area = 0.0;
    bool any_spine_tried = false;

    for (std::size_t i = 0; i < seeds.size(); ++i) {
        if (elig_overlap[i] == 1) continue;

        Local7Frame frame;
        if (!local7_make_frame(Complex(seeds[i].x, seeds[i].y), source, mapper, &frame) ||
            std::abs(frame.lambda_l) < kLocal7LambdaMin ||
            std::abs(frame.lambda_s) < kLocal7LambdaMin ||
            !std::isfinite(frame.area_jac)) {
            continue; // not a usable frame; handle this seed via cartesian
        }

        const auto elig = local7_spine_eligibility(
            seeds, elig_overlap, i, frame, source, source_step, mapper, caustic_born);
        if (!elig.ok) continue; // not eligible; handle this seed via cartesian

        // Skip when var_ratio is too large: the linear fold model breaks down
        // and spine error exceeds a few percent.
        {
            const double beta_abs = std::hypot(frame.beta_r, frame.beta_i);
            const double abs_ll = std::abs(frame.lambda_l);
            const double abs_ls = std::abs(frame.lambda_s);
            if (abs_ll > 0.0 && abs_ls > 0.0) {
                const double var_ratio = 2.0 * beta_abs * source_radius / (abs_ls * abs_ll);
                if (var_ratio > kLocal7SpineMaxVarRatio) continue;
            }
        }

        any_spine_tried = true;
        int fallback_reason = 0;
        const double area = local7_spine_area_binary(
            mapper, source, source_radius, settings, finite_magnifier, frame, &fallback_reason);
        if (!std::isfinite(area) || area <= 0.0) {
            // Spine failed for an eligible pair. Fall back to full cartesian
            // using the already-computed seeds (no second seed generation).
            FiniteSourceDecision fb {
                FiniteSourceMethod::inverse_ray_cartesian,
                estimate_cartesian_cost(settings),
                "spine integration failed; cartesian fallback",
            };
            const double mag = inverse_ray_cartesian_binary_mag(
                point_magnifier, separation, mass_ratio, source, source_radius,
                settings, finite_magnifier, &seeds);
            if (!std::isfinite(mag)) return {mag, 0, fb, std::nan(""), 0, false};
            return {mag, 0, fb, 0.0, 0, true};
        }

        spine_area += area;
        spine_covered[i] = true;
        spine_covered[elig.pair_index] = true;
        elig_overlap[i] = 1;
        elig_overlap[elig.pair_index] = 1;
        if (std::getenv("LCBININT_SPINE_DEBUG")) {
            std::fprintf(stderr, "  spine pair (%zu,%zu) area=%.6f\n", i, elig.pair_index, area);
        }
    }

    // Pass 2: cartesian for seeds not handled by spine.
    std::vector<SourcePosition> cartesian_seeds;
    cartesian_seeds.reserve(seeds.size());
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        if (!spine_covered[i]) cartesian_seeds.push_back(seeds[i]);
    }

    const double total_source = source_flux(source_radius, settings);
    if (!std::isfinite(total_source) || total_source <= 0.0) {
        return {std::nan(""), 0, decision, std::nan(""), 0, false};
    }
    double total_mag = spine_area / total_source;

    if (!cartesian_seeds.empty()) {
        const double cart_mag = inverse_ray_cartesian_binary_mag(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings, finite_magnifier, &cartesian_seeds);
        if (!std::isfinite(cart_mag)) {
            return {cart_mag, 0, decision, std::nan(""), 0, false};
        }
        if (std::getenv("LCBININT_SPINE_DEBUG")) {
            std::fprintf(stderr, "  spine_mag=%.4f cart_mag=%.4f total=%.4f (source_flux=%.6f)\n",
                spine_area/total_source, cart_mag, spine_area/total_source+cart_mag, total_source);
        }
        total_mag += cart_mag;
    }

    if (!any_spine_tried) {
        // No seed was eligible for spine; result is pure cartesian.
        decision.method = FiniteSourceMethod::inverse_ray_cartesian;
        decision.reason = "spine not eligible; cartesian";
    }

    return {total_mag, 0, decision, 0.0, 0, true};
}

} // namespace

bool binary_centre_seed_certificate(double separation,double mass_ratio,SourcePosition source,double rho) {
    PointSourceMagnifier point;std::vector<SourcePosition> proposals;
    for (const auto& image:point.binary_images(separation,mass_ratio,source)) proposals.push_back(image.position);
    return certify_centre_seed_positions(separation,mass_ratio,source,rho,proposals);
}

std::vector<SourcePosition> binary_critical_phase_proposals(
    double separation,double mass_ratio,double phase_angle)
{
    return critical_curve_points_at_phase(separation,mass_ratio,phase_angle);
}

bool binary_auto_tolerance_supported(
    double requested_absolute_tolerance,
    double requested_relative_tolerance)
{
    double absolute_tolerance =
        std::isfinite(requested_absolute_tolerance)
        ? std::max(requested_absolute_tolerance, 0.0)
        : 0.0;
    double relative_tolerance =
        std::isfinite(requested_relative_tolerance)
        ? std::max(requested_relative_tolerance, 0.0)
        : 0.0;
    if (absolute_tolerance <= 0.0 && relative_tolerance <= 0.0) {
        absolute_tolerance = kDefaultFiniteSourceAbsoluteTolerance;
        relative_tolerance = kDefaultFiniteSourceRelativeTolerance;
    }
    const bool absolute_supported =
        absolute_tolerance > kEmpiricalResolutionMinimumAbsoluteTolerance &&
        absolute_tolerance <= kEmpiricalResolutionMaximumTolerance;
    const bool relative_supported =
        relative_tolerance >= kEmpiricalResolutionMinimumRelativeTolerance &&
        relative_tolerance <= kEmpiricalResolutionMaximumTolerance;
    return absolute_supported || relative_supported;
}

BinaryResolutionSelection calibrated_binary_resolution(
    double mass_ratio,
    double source_radius,
    double caustic_distance,
    double point_source_magnification,
    double limb_darkening_c,
    int finite_mode,
    double requested_absolute_tolerance,
    double requested_relative_tolerance,
    int maximum_bins)
{
    const int cap = std::max(maximum_bins, 1);
    // These arguments remain part of the selector API for callers and the
    // diagnostic mirror; the calibrated runtime rule intentionally does not
    // spend work extracting extra geometry or profile features.
    (void) mass_ratio;
    (void) source_radius;
    (void) caustic_distance;
    (void) limb_darkening_c;
    const double a_point = std::abs(point_source_magnification);
    if (!std::isfinite(a_point)) {
        return {std::min(100, cap), true};
    }

    // Point-source magnification is already computed before this selector, so
    // the switch adds no root solves or probes.  A=200 is the stable joint
    // minimum in the speed holdout; keeping it as a single threshold also makes
    // the routing rule easy to state and reproduce in a paper.
    const bool prefer_polar = a_point >= 200.0;
    const bool use_polar_law = finite_mode == 2 ||
        (finite_mode == 4 && prefer_polar);

    double absolute_tolerance =
        std::isfinite(requested_absolute_tolerance)
        ? std::max(requested_absolute_tolerance, 0.0)
        : 0.0;
    double relative_tolerance =
        std::isfinite(requested_relative_tolerance)
        ? std::max(requested_relative_tolerance, 0.0)
        : 0.0;
    if (absolute_tolerance <= 0.0 && relative_tolerance <= 0.0) {
        absolute_tolerance = kDefaultFiniteSourceAbsoluteTolerance;
        relative_tolerance = kDefaultFiniteSourceRelativeTolerance;
    }
    const bool absolute_supported =
        absolute_tolerance > kEmpiricalResolutionMinimumAbsoluteTolerance &&
        absolute_tolerance <= kEmpiricalResolutionMaximumTolerance;
    const bool relative_supported =
        relative_tolerance >= kEmpiricalResolutionMinimumRelativeTolerance &&
        relative_tolerance <= kEmpiricalResolutionMaximumTolerance;

    const double relative_c = use_polar_law
        ? kPolarRelativeResolutionC : kCartesianRelativeResolutionC;
    const double relative_beta = use_polar_law
        ? kPolarRelativeResolutionBeta : kCartesianRelativeResolutionBeta;
    const double absolute_c = use_polar_law
        ? kPolarAbsoluteResolutionC : kCartesianAbsoluteResolutionC;
    const double absolute_beta = use_polar_law
        ? kPolarAbsoluteResolutionBeta : kCartesianAbsoluteResolutionBeta;
    const double absolute_gamma = use_polar_law
        ? kPolarAbsoluteResolutionGamma : kCartesianAbsoluteResolutionGamma;

    double required_bins = std::numeric_limits<double>::infinity();
    if (relative_supported) {
        required_bins = relative_c * std::pow(
            relative_tolerance / kEmpiricalResolutionBaselineTolerance,
            -relative_beta);
    }
    if (absolute_supported) {
        const double absolute_bins = absolute_c * std::pow(
            absolute_tolerance / kEmpiricalResolutionBaselineTolerance,
            -absolute_beta) * std::pow(std::max(a_point, 1.0), absolute_gamma);
        required_bins = std::min(required_bins, absolute_bins);
    }

    // Absolute and relative tolerances are alternative allowances, so the
    // less demanding branch sets the initial resolution.  The calibration is
    // continuous; use any positive integer rather than snapping to the
    // measurement ladder, and preserve max_source_bins as the hard cap.
    const int selected_bins = !std::isfinite(required_bins) || required_bins >= cap
        ? cap
        : std::max(1, static_cast<int>(std::ceil(required_bins)));
    return {selected_bins, prefer_polar};
}

BinaryResolutionSelection calibrated_triple_resolution(
    const model::TripleLensGeometry& geometry,
    double source_radius,
    double caustic_distance,
    double point_source_magnification,
    double limb_darkening_c,
    double requested_relative_tolerance,
    int maximum_bins)
{
    // Triple Cartesian boundary error follows one continuous resolution law.
    // The earlier selector discarded the requested tolerance almost entirely:
    // 1e-2 and 1e-3 both ran 256 bins, while every tighter request jumped to
    // the cap.  That made tolerance inert and over-integrated the loose cases
    // by roughly an order of magnitude in cells.  Use the same general form as
    // the binary calibration, but without a geometry/regime branch: the
    // measured boundary-corrected triple grid follows one continuous power
    // law.  A 32-bin floor is retained because the loose holdout contains
    // local structure that a 16-bin grid can alias.  The caller still verifies
    // Cartesian grids against a nested half-resolution result, and either grid
    // can grow to the hard cap from its embedded error estimate.
    const int cap = std::max(maximum_bins, 1);
    const double tolerance = requested_relative_tolerance > 0.0
        ? requested_relative_tolerance
        : kDefaultFiniteSourceRelativeTolerance;
    // This selects 32/128/400 bins at 1e-3/1e-4/1e-5 without snapping to a
    // case-specific ladder.  The former 80/319 law over-resolved the measured
    // 1e-3/1e-4 population while almost every nested check stopped at level
    // zero.  The measured 0.6 exponent preserves the tighter-tolerance tail.
    constexpr double kTripleResolutionAtBaseline = 32.0;
    constexpr double kTripleResolutionExponent = 0.6;
    const double required = kTripleResolutionAtBaseline * std::pow(
        tolerance / kEmpiricalResolutionBaselineTolerance,
        -kTripleResolutionExponent);
    const double required_with_floor = std::max(required, 32.0);
    const int bins = !std::isfinite(required_with_floor) || required_with_floor >= cap
        ? cap
        : std::max(1, static_cast<int>(std::ceil(required_with_floor)));
    return {bins, false};

}

FiniteSourceMagnifier::FiniteSourceMagnifier(FiniteSourceSettings settings)
    : settings_(settings)
{
}

std::vector<SourcePosition> FiniteSourceMagnifier::cached_binary_image_seeds(
    const PointSourceMagnifier& point_magnifier,
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double caustic_distance,
    const std::vector<SourcePosition>* center_image_seeds,
    bool* support_proven) const
{
    auto* phase=adaptive_flux_request();
    AdaptiveStageTimer seed_timer(phase && phase->cost_trace?&phase->seed_seconds:nullptr);

    for (const auto& entry : binary_seed_cache_) {
        const bool center_seeds_match =
            entry.has_center_image_seeds == (center_image_seeds != nullptr);
        if (entry.separation == separation &&
            entry.mass_ratio == mass_ratio &&
            entry.source.x == source.x && entry.source.y == source.y &&
            entry.source_radius == source_radius &&
            entry.caustic_distance == caustic_distance &&
            center_seeds_match) {
            if (support_proven != nullptr) {
                *support_proven = entry.support_proven;
            }
            return entry.seeds;
        }
    }

    const auto& branches = binary_caustic_branches(separation, mass_ratio);
    // Reuse the geometric segment index already built with the caustics.
    // This is geometry metadata, not a previous source/epoch result.
    // A retained vertex satisfies d < rho + |v_prev-2*v+v_next|.
    // The curvature term is at most 2*max_segment_length, plus roundoff.
    // A blind segment crossing also has an endpoint within rho+max_length.
    // Thus querying this padded source box retains both kinds of candidate.
    std::vector<std::vector<std::size_t>> selected(branches.size());
    const std::vector<std::vector<std::size_t>>* selection = nullptr;
    const double coord_scale = std::max({1.0,std::fabs(caustic_cache_min_x_),
        std::fabs(caustic_cache_max_x_),std::fabs(caustic_cache_min_y_),std::fabs(caustic_cache_max_y_),std::fabs(source.x),std::fabs(source.y),source_radius});
    const double padding = std::nextafter(2*caustic_cache_max_seg_len_ +
        32*std::numeric_limits<double>::epsilon()*coord_scale,
        std::numeric_limits<double>::infinity());
    const double reach = std::nextafter(source_radius+padding,
        std::numeric_limits<double>::infinity());
    if (std::isfinite(source.x) && std::isfinite(source.y) &&
        std::isfinite(reach) && reach > 0 && caustic_cache_grid_step_x_ > 0 &&
        caustic_cache_grid_step_y_ > 0) {
        const double xlo = std::nextafter(source.x-reach,-INFINITY);
        const double xhi = std::nextafter(source.x+reach,INFINITY);
        const double ylo = std::nextafter(source.y-reach,-INFINITY);
        const double yhi = std::nextafter(source.y+reach,INFINITY);
        if (xlo > caustic_cache_max_x_ || xhi < caustic_cache_min_x_ ||
            ylo > caustic_cache_max_y_ || yhi < caustic_cache_min_y_) {
            selection = &selected;
        } else {
        auto cell = [&](double x,double origin,double step,int guard) {
            const double value = std::floor((x-origin)/step)+guard;
            return static_cast<int>(std::clamp(value,0.0,double(caustic_cache_grid_size_-1)));
        };
        const int xl = cell(std::nextafter(source.x-reach,-INFINITY),caustic_cache_min_x_,caustic_cache_grid_step_x_,-1);
        const int xh = cell(std::nextafter(source.x+reach,INFINITY),caustic_cache_min_x_,caustic_cache_grid_step_x_,1);
        const int yl = cell(std::nextafter(source.y-reach,-INFINITY),caustic_cache_min_y_,caustic_cache_grid_step_y_,-1);
        const int yh = cell(std::nextafter(source.y+reach,INFINITY),caustic_cache_min_y_,caustic_cache_grid_step_y_,1);
        if (xl != 0 || xh != caustic_cache_grid_size_-1 || yl != 0 || yh != caustic_cache_grid_size_-1) {
            std::size_t count = 0,total = 0;
            for (const auto& branch : branches) total += branch.size();
            const double bound2 = std::nextafter(reach*reach*(1+16*std::numeric_limits<double>::epsilon()),INFINITY);
            for (int iy=yl;iy<=yh;++iy) for(int ix=xl;ix<=xh;++ix)
                for(const auto& ref:caustic_cache_branch_grid_[iy*caustic_cache_grid_size_+ix]) {
                    const auto& vertex = branches[ref.branch][ref.pos];
                    if (vertex.x < xlo || vertex.x > xhi || vertex.y < ylo || vertex.y > yhi) continue;
                    if (distance_squared(vertex,source) > bound2) continue;
                    selected[ref.branch].push_back(ref.pos); ++count;
                }
            // Dense queries use the original linear traversal. Only its
            // execution cost changes; the retained candidate set is identical.
            if (count < total/2) {
                for(auto& vertices:selected) std::sort(vertices.begin(),vertices.end());
                selection = &selected;
            }
        }
        }
    }
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    bool proven = true;
    auto seeds = augmented_image_seeds(
        point_magnifier,
        mapper,
        separation,
        mass_ratio,
        source,
        source_radius,
        caustic_distance,
        center_image_seeds,
        &branches,
        &proven,nullptr,&binary_critical_curve_branches_cached(separation,mass_ratio),
        center_image_seeds ? static_cast<int>(center_image_seeds->size()) : -1, selection);

    constexpr std::size_t maximum_entries = 1024;
    if (binary_seed_cache_.size() >= maximum_entries) {
        binary_seed_cache_.erase(binary_seed_cache_.begin());
    }
    binary_seed_cache_.push_back({
        separation,
        mass_ratio,
        source,
        source_radius,
        caustic_distance,
        center_image_seeds != nullptr,
        seeds,
        proven,
    });
    if (support_proven != nullptr) {
        *support_proven = proven;
    }
    return seeds;
}

void FiniteSourceMagnifier::ensure_limb_darkening_table() const
{
    const bool cache_matches = limb_darkening_table_valid_ &&
                               limb_darkening_table_c_ == settings_.limb_darkening_c &&
                               limb_darkening_table_d_ == settings_.limb_darkening_d &&
                               static_cast<int>(limb_darkening_table_.size()) == kLimbDarkeningTableSize + 1;
    if (cache_matches) {
        return;
    }

    limb_darkening_table_.resize(kLimbDarkeningTableSize + 1);
    for (int i = 0; i <= kLimbDarkeningTableSize; ++i) {
        const double normalized_radius2 =
            static_cast<double>(i) / static_cast<double>(kLimbDarkeningTableSize);
        limb_darkening_table_[static_cast<std::size_t>(i)] =
            source_surface_brightness(normalized_radius2, settings_);
    }
    limb_darkening_table_valid_ = true;
    limb_darkening_table_c_ = settings_.limb_darkening_c;
    limb_darkening_table_d_ = settings_.limb_darkening_d;
}

double FiniteSourceMagnifier::limb_darkening_table_brightness(double normalized_radius2) const
{
    // Linear LD is analytic and cheap: avoid a discontinuous nearest-bin
    // profile that the smooth quadrature residual model cannot represent.
    if (settings_.limb_darkening_d == 0.0) {
        return source_surface_brightness(normalized_radius2, settings_);
    }
    if (normalized_radius2 <= 0.0) {
        return limb_darkening_table_[0];
    }
    if (normalized_radius2 >= 1.0) {
        return limb_darkening_table_[static_cast<std::size_t>(kLimbDarkeningTableSize)];
    }
    const int index = static_cast<int>(
        normalized_radius2 * static_cast<double>(kLimbDarkeningTableSize) + 0.5);
    return limb_darkening_table_[static_cast<std::size_t>(index)];
}

void FiniteSourceMagnifier::ensure_binary_caustic_cache(
    double separation,
    double mass_ratio,
    double separation_tolerance) const
{
    const int bins = std::max(settings_.caustic_bins, 32);
    const bool cache_matches = caustic_cache_valid_ &&
                               caustic_cache_bins_ == bins &&
                               caustic_cache_mass_ratio_ == mass_ratio &&
                               std::signbit(caustic_cache_separation_) ==
                                   std::signbit(separation) &&
                               std::abs(caustic_cache_separation_ - separation) <=
                                   std::max(separation_tolerance, 0.0);
    if (!cache_matches) {
        caustic_cache_critical_branches_ =
            build_binary_critical_curves(separation, mass_ratio, bins);
        caustic_cache_branches_ = map_binary_critical_curves_to_caustics(
            caustic_cache_critical_branches_, separation, mass_ratio);
        caustic_cache_points_.clear();
        caustic_cache_points_.reserve(static_cast<std::size_t>(bins) * 4);
        caustic_cache_min_x_ = std::numeric_limits<double>::infinity();
        caustic_cache_max_x_ = -std::numeric_limits<double>::infinity();
        caustic_cache_min_y_ = std::numeric_limits<double>::infinity();
        caustic_cache_max_y_ = -std::numeric_limits<double>::infinity();
        for (const auto& branch : caustic_cache_branches_) {
            for (const auto& point : branch) {
                caustic_cache_points_.push_back(point);
                caustic_cache_min_x_ = std::min(caustic_cache_min_x_, point.x);
                caustic_cache_max_x_ = std::max(caustic_cache_max_x_, point.x);
                caustic_cache_min_y_ = std::min(caustic_cache_min_y_, point.y);
                caustic_cache_max_y_ = std::max(caustic_cache_max_y_, point.y);
            }
        }
        const double width = std::max(caustic_cache_max_x_ - caustic_cache_min_x_, 1.0e-12);
        const double height = std::max(caustic_cache_max_y_ - caustic_cache_min_y_, 1.0e-12);
        caustic_cache_grid_step_x_ = width / static_cast<double>(caustic_cache_grid_size_);
        caustic_cache_grid_step_y_ = height / static_cast<double>(caustic_cache_grid_size_);
        caustic_cache_grid_.assign(
            static_cast<std::size_t>(caustic_cache_grid_size_ * caustic_cache_grid_size_), {});
        for (std::size_t i = 0; i < caustic_cache_points_.size(); ++i) {
            const auto& point = caustic_cache_points_[i];
            const int ix = std::clamp(static_cast<int>((point.x - caustic_cache_min_x_) /
                                           caustic_cache_grid_step_x_),
                0,
                caustic_cache_grid_size_ - 1);
            const int iy = std::clamp(static_cast<int>((point.y - caustic_cache_min_y_) /
                                           caustic_cache_grid_step_y_),
                0,
                caustic_cache_grid_size_ - 1);
            caustic_cache_grid_[static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]
                .push_back(static_cast<int>(i));
        }
        // Build branch-position grid for fast per-segment distance queries.
        caustic_cache_branch_grid_.assign(
            static_cast<std::size_t>(caustic_cache_grid_size_ * caustic_cache_grid_size_), {});
        // max_seg_len is used as a safety margin when bounding the grid search
        // radius in binary_caustic_distance. The monodromy reconstruction makes
        // every branch a physical closed curve, including its last-to-first
        // segment.
        double max_seg2 = 0.0;
        for (int b = 0; b < static_cast<int>(caustic_cache_branches_.size()); ++b) {
            const auto& br = caustic_cache_branches_[static_cast<std::size_t>(b)];
            const int n = static_cast<int>(br.size());
            for (int j = 0; j < n; ++j) {
                const auto& pt = br[static_cast<std::size_t>(j)];
                const auto& next_pt = br[static_cast<std::size_t>((j + 1) % n)];
                const double seg2 = distance_squared(pt, next_pt);
                if (seg2 > max_seg2) {
                    max_seg2 = seg2;
                }
                const int ix = std::clamp(
                    static_cast<int>((pt.x - caustic_cache_min_x_) / caustic_cache_grid_step_x_),
                    0, caustic_cache_grid_size_ - 1);
                const int iy = std::clamp(
                    static_cast<int>((pt.y - caustic_cache_min_y_) / caustic_cache_grid_step_y_),
                    0, caustic_cache_grid_size_ - 1);
                caustic_cache_branch_grid_[
                    static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]
                    .push_back({b, j});
            }
        }
        caustic_cache_max_seg_len_ = std::sqrt(max_seg2);

        caustic_cache_valid_ = true;
        caustic_cache_separation_ = separation;
        caustic_cache_mass_ratio_ = mass_ratio;
        caustic_cache_bins_ = bins;
    }
}

const std::vector<std::vector<SourcePosition>>&
FiniteSourceMagnifier::binary_caustic_branches(
    double separation,
    double mass_ratio) const
{
    ensure_binary_caustic_cache(separation, mass_ratio);
    return caustic_cache_branches_;
}

std::vector<std::vector<SourcePosition>>
FiniteSourceMagnifier::binary_critical_curve_branches(
    double separation,
    double mass_ratio) const
{
    return binary_critical_curve_branches_cached(separation,mass_ratio);
}

const std::vector<std::vector<SourcePosition>>&
FiniteSourceMagnifier::binary_critical_curve_branches_cached(double separation,double mass_ratio) const
{
    ensure_binary_caustic_cache(separation,mass_ratio);
    return caustic_cache_critical_branches_;
}

std::vector<std::vector<SourcePosition>>
FiniteSourceMagnifier::triple_caustic_branches(
    const model::TripleLensGeometry& geometry) const
{
    return triple_caustic_branches_cached(geometry);
}

const std::vector<std::vector<SourcePosition>>&
FiniteSourceMagnifier::triple_caustic_branches_cached(
    const model::TripleLensGeometry& geometry) const
{
    return cached_triple_caustic_branches(
        geometry, settings_.caustic_bins).branches;
}

double FiniteSourceMagnifier::triple_caustic_distance_for_source(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double refine_within) const
{
    const auto& caustics = cached_triple_caustic_branches(
        geometry, settings_.caustic_bins);
    return magnification::triple_caustic_distance(
        geometry, caustics, source, refine_within);
}

std::vector<std::vector<SourcePosition>>
FiniteSourceMagnifier::triple_critical_curve_branches(
    const model::TripleLensGeometry& geometry) const
{
    const int bins = std::max(settings_.caustic_bins, 32);
    std::vector<std::vector<SourcePosition>> branches(6);
    for (int i = 0; i < bins; ++i) {
        const double phase_angle = 2.0 * kPi * static_cast<double>(i) /
                                   static_cast<double>(bins);
        append_tracked_caustic_points(
            branches, triple_critical_curve_points_at_phase(geometry, phase_angle));
    }
    return merge_tracked_phase_branches(std::move(branches));
}

double FiniteSourceMagnifier::binary_caustic_distance(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double hint_nearest_point_dist,
    double separation_tolerance) const
{
    ensure_binary_caustic_cache(separation, mass_ratio, separation_tolerance);

    // Obtain nearest caustic POINT distance as an upper bound on segment distance.
    // The caller often already has this from binary_sampled_caustic_distance;
    // if so, the hint skips this O(N) scan.
    double nearest_dist = hint_nearest_point_dist;
    if (!std::isfinite(nearest_dist)) {
        double nearest2 = std::numeric_limits<double>::infinity();
        for (const auto& pt : caustic_cache_points_) {
            const double dx = source.x - pt.x;
            const double dy = source.y - pt.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 < nearest2) {
                nearest2 = d2;
            }
        }
        nearest_dist = std::sqrt(nearest2);
    }

    if (caustic_cache_branch_grid_.empty()) {
        return nearest_dist;
    }

    // Search the branch grid within nearest_dist + max_seg_len to catch all segments
    // whose distance to source could be less than nearest_dist.
    const double seg_radius = nearest_dist + caustic_cache_max_seg_len_;
    const int ix0 = std::clamp(
        static_cast<int>((source.x - seg_radius - caustic_cache_min_x_) /
                         caustic_cache_grid_step_x_),
        0, caustic_cache_grid_size_ - 1);
    const int ix1 = std::clamp(
        static_cast<int>((source.x + seg_radius - caustic_cache_min_x_) /
                         caustic_cache_grid_step_x_),
        0, caustic_cache_grid_size_ - 1);
    const int iy0 = std::clamp(
        static_cast<int>((source.y - seg_radius - caustic_cache_min_y_) /
                         caustic_cache_grid_step_y_),
        0, caustic_cache_grid_size_ - 1);
    const int iy1 = std::clamp(
        static_cast<int>((source.y + seg_radius - caustic_cache_min_y_) /
                         caustic_cache_grid_step_y_),
        0, caustic_cache_grid_size_ - 1);

    double distance = nearest_dist;
    for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (const auto& ref : caustic_cache_branch_grid_[
                    static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)]) {
                const auto& branch =
                    caustic_cache_branches_[static_cast<std::size_t>(ref.branch)];
                const int n = static_cast<int>(branch.size());
                if (n < 2) {
                    continue;
                }
                const int prev = (ref.pos > 0) ? ref.pos - 1 : n - 1;
                const int next = (ref.pos + 1) % n;
                distance = std::min(distance,
                    point_segment_distance(source,
                        branch[static_cast<std::size_t>(prev)],
                        branch[static_cast<std::size_t>(ref.pos)]));
                distance = std::min(distance,
                    point_segment_distance(source,
                        branch[static_cast<std::size_t>(ref.pos)],
                        branch[static_cast<std::size_t>(next)]));
            }
        }
    }
    return distance;
}

double FiniteSourceMagnifier::binary_caustic_distance_for_source(
    double separation,
    double mass_ratio,
    SourcePosition source) const
{
    return binary_caustic_distance(separation, mass_ratio, source);
}

BinaryRoutingDiagnostics
FiniteSourceMagnifier::binary_routing_diagnostics_for_source(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double point_source_magnification,
    const PointSourceMagnifier* point_magnifier_hint) const
{
    BinaryRoutingDiagnostics out;
    out.point_magnification = point_source_magnification;
    if (source_radius <= 0.0) {
        out.point_error_estimate = 0.0;
        out.point_preflight_safe = true;
        out.point_safe = true;
        return out;
    }

    PointSourceMagnifier local_point_magnifier;
    const PointSourceMagnifier& point_magnifier =
        point_magnifier_hint != nullptr ? *point_magnifier_hint : local_point_magnifier;
    const auto safety = evaluate_point_source_safety(
        point_magnifier,
        separation,
        mass_ratio,
        source,
        source_radius,
        point_source_magnification,
        settings_);
    out.point_magnification = safety.diagnostic.magnification;
    out.point_absolute_tolerance = safety.absolute_tolerance;
    out.quadrupole_indicator = safety.diagnostic.quadrupole_indicator;
    out.cusp_indicator = safety.diagnostic.cusp_indicator;
    out.ghost_indicator = safety.diagnostic.ghost_indicator;
    out.planetary_distance2 = safety.planetary_distance2;
    out.image_count = safety.diagnostic.image_count;
    out.ghost_count = safety.diagnostic.ghost_count;
    out.safety_flags =
        (safety.quadrupole_cusp_safe ? 1 : 0) |
        (safety.ghost_safe ? 2 : 0) |
        (safety.planetary_safe ? 4 : 0);
    out.point_error_estimate =
        (safety.diagnostic.quadrupole_indicator + safety.diagnostic.cusp_indicator) *
        source_radius * source_radius;

    double requested_relative = settings_.adaptive_hex_threshold > 0.0
        ? settings_.adaptive_hex_threshold
        : kDefaultFiniteSourceRelativeTolerance;
    if (has_explicit_finite_source_tolerance(settings_)) {
        requested_relative = explicit_finite_source_relative_budget(
            settings_, point_source_magnification);
    }
    constexpr double kPreflightPointSafety = 30.0;
    const double derivative_relative_error =
        out.point_error_estimate /
        std::max(std::abs(point_source_magnification), 1.0e-10);
    out.point_preflight_safe =
        settings_.hex_threshold > 0.0 &&
        safety.point_source_safe() &&
        kPreflightPointSafety * derivative_relative_error <= requested_relative;
    if (out.point_preflight_safe) {
        out.point_safe = finite_source_error_within_budget(
            settings_, point_source_magnification, out.point_error_estimate);
        return out;
    }

    const double bbox_margin = settings_.kinji_threshold * source_radius;
    const double topology_margin =
        binary_topology_boundary_margin(separation, mass_ratio);
    const double caustic_reuse_tolerance = std::min(
        0.25 * source_radius, 0.02 * topology_margin);
    const double sampled_distance = binary_sampled_caustic_distance(
        separation,
        mass_ratio,
        source,
        bbox_margin,
        caustic_reuse_tolerance);
    const bool exact_caustic_geometry = caustic_cache_separation_ == separation;
    if (safety.point_source_safe() &&
        exact_caustic_geometry &&
        (source.x < caustic_cache_min_x_ - bbox_margin ||
         source.x > caustic_cache_max_x_ + bbox_margin ||
         source.y < caustic_cache_min_y_ - bbox_margin ||
         source.y > caustic_cache_max_y_ + bbox_margin)) {
        out.point_safe = finite_source_error_within_budget(
            settings_, point_source_magnification, out.point_error_estimate);
        return out;
    }

    out.caustic_distance = binary_caustic_distance(
        separation,
        mass_ratio,
        source,
        sampled_distance,
        caustic_reuse_tolerance);
    constexpr double kMeasuredTopologyReleaseDistance = 10.0;
    const bool measured_topology_safe =
        std::isfinite(out.caustic_distance) &&
        out.caustic_distance >=
            kMeasuredTopologyReleaseDistance * source_radius;
    const bool effective_topology_safe =
        safety.topology_safe() || measured_topology_safe;
    const bool effective_point_safe =
        safety.quadrupole_cusp_safe && effective_topology_safe;
    const bool near_caustic =
        out.caustic_distance < settings_.hex_threshold * source_radius;
    if (!near_caustic && effective_point_safe) {
        double distance_safety = 1.0;
        if (source_radius >= 1.0e-3 &&
            std::isfinite(out.caustic_distance)) {
            const double distance_ratio =
                out.caustic_distance / source_radius;
            const double t =
                settings_.hex_threshold /
                std::max(distance_ratio, settings_.hex_threshold);
            distance_safety = std::max(1.0, 30.0 * t * t * t);
        }
        out.point_safe =
            derivative_relative_error <=
                requested_relative / distance_safety &&
            finite_source_error_within_budget(
                settings_, point_source_magnification, out.point_error_estimate);
    }

    constexpr double kGrazeQuadratureDistanceFactor = 2.0;
    if (std::isfinite(out.caustic_distance) &&
        out.caustic_distance <
            kGrazeQuadratureDistanceFactor * source_radius) {
        const auto scan = scan_caustic_branches(
            binary_caustic_branches(separation, mass_ratio),
            source,
            source_radius);
        out.scan_performed = true;
        out.scan_min_distance = scan.min_distance;
        out.any_vertex_inside = scan.any_vertex_inside;
        out.has_crossing_probes = !scan.crossing_probes.empty();
        // The closest point may be inside the disk along a segment even when
        // every sampled vertex and crossing probe is outside it.
        const bool caustic_enters_disk =
            scan.min_distance < source_radius ||
            out.any_vertex_inside || out.has_crossing_probes;
        out.chord_band =
            !caustic_enters_disk &&
            scan.min_distance >= 0.95 * source_radius &&
            scan.min_distance < 1.35 * source_radius;
        out.tangent_band =
            !caustic_enters_disk &&
            !out.chord_band &&
            std::abs(scan.min_distance - source_radius) <
                0.35 * source_radius;
        out.grazing_ring_band =
            !caustic_enters_disk &&
            scan.min_distance >= source_radius &&
            !out.tangent_band &&
            !out.chord_band;
    }
    return out;
}

double FiniteSourceMagnifier::binary_sampled_caustic_distance(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double search_radius,
    double separation_tolerance) const
{
    ensure_binary_caustic_cache(separation, mass_ratio, separation_tolerance);
    double distance2 = std::numeric_limits<double>::infinity();

    if (search_radius > 0.0 &&
        (source.x < caustic_cache_min_x_ - search_radius ||
            source.x > caustic_cache_max_x_ + search_radius ||
            source.y < caustic_cache_min_y_ - search_radius ||
            source.y > caustic_cache_max_y_ + search_radius)) {
        return std::numeric_limits<double>::infinity();
    }

    if (search_radius > 0.0 && !caustic_cache_grid_.empty()) {
        const int ix0 = std::clamp(static_cast<int>((source.x - search_radius - caustic_cache_min_x_) /
                                       caustic_cache_grid_step_x_),
            0,
            caustic_cache_grid_size_ - 1);
        const int ix1 = std::clamp(static_cast<int>((source.x + search_radius - caustic_cache_min_x_) /
                                       caustic_cache_grid_step_x_),
            0,
            caustic_cache_grid_size_ - 1);
        const int iy0 = std::clamp(static_cast<int>((source.y - search_radius - caustic_cache_min_y_) /
                                       caustic_cache_grid_step_y_),
            0,
            caustic_cache_grid_size_ - 1);
        const int iy1 = std::clamp(static_cast<int>((source.y + search_radius - caustic_cache_min_y_) /
                                       caustic_cache_grid_step_y_),
            0,
            caustic_cache_grid_size_ - 1);
        for (int iy = iy0; iy <= iy1; ++iy) {
            for (int ix = ix0; ix <= ix1; ++ix) {
                const auto& cell =
                    caustic_cache_grid_[static_cast<std::size_t>(iy * caustic_cache_grid_size_ + ix)];
                for (const int index : cell) {
                    const auto& point = caustic_cache_points_[static_cast<std::size_t>(index)];
                    const double dx = source.x - point.x;
                    const double dy = source.y - point.y;
                    const double candidate = dx * dx + dy * dy;
                    if (candidate < distance2) {
                        distance2 = candidate;
                    }
                }
            }
        }
        if (distance2 == std::numeric_limits<double>::infinity()) {
            return distance2;
        }
    } else {
        for (const auto& point : caustic_cache_points_) {
            const double dx = source.x - point.x;
            const double dy = source.y - point.y;
            const double candidate = dx * dx + dy * dy;
            if (candidate < distance2) {
                distance2 = candidate;
            }
        }
    }
    return std::sqrt(distance2);
}

FiniteSourceResult FiniteSourceMagnifier::inverse_ray_polar_binary_mag(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double caustic_distance) const
{
    const PointSourceMagnifier point_magnifier;
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_polar,
        estimate_polar_cost(settings_),
        "polar cached inverse-ray",
    };
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    bool support_proven = true;
    std::size_t centre_image_count = 0;
    auto seeds = augmented_image_seeds(
        point_magnifier, mapper, separation, mass_ratio, source, source_radius,
        std::numeric_limits<double>::infinity(), nullptr,
        &binary_caustic_branches(separation, mass_ratio), &support_proven,
        &centre_image_count,&binary_critical_curve_branches_cached(separation,mass_ratio));
    const double sampled_caustic_distance = binary_sampled_caustic_distance(
        separation, mass_ratio, source, source_radius);
    const double polar_fallback_distance =
        std::max(settings_.hex_threshold, 1.0) * source_radius;
    if (seeds.size() > centre_image_count ||
        (std::isfinite(sampled_caustic_distance) && sampled_caustic_distance < polar_fallback_distance) ||
        (std::isfinite(caustic_distance) && caustic_distance < polar_fallback_distance)) {
        LegacyAreaDiagnostics diagnostics;
        const double magnification = inverse_ray_cartesian_binary_mag(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings_, this, &seeds, &diagnostics);
        decision.method = FiniteSourceMethod::inverse_ray_cartesian;
        decision.reason = "polar mode used cartesian fallback for caustic-crossing";
        if (!std::isfinite(magnification)) {
            return {magnification, 0, decision, std::nan(""), 0, false};
        }
        return {
            magnification,
            0,
            decision,
            diagnostics.estimated_error,
            0,
            support_proven && diagnostics.seed_coverage_proven && finite_source_error_within_budget(
                settings_, magnification, diagnostics.estimated_error),
        };
    }
    const auto evaluation = evaluate_polar_to_tolerance(
        settings_,
        support_proven,
        [&](const FiniteSourceSettings& active, PolarAreaDiagnostics* diagnostics) {
            return inverse_ray_polar_boundary_binary(
                point_magnifier, separation, mass_ratio, source, source_radius,
                active, this, &seeds, diagnostics);
        });
    if (!std::isfinite(evaluation.magnification)) {
        return {evaluation.magnification, 0, decision, std::nan(""), 0, false};
    }
    return {
        evaluation.magnification,
        0,
        decision,
        evaluation.error_estimate,
        evaluation.refinement_level,
        evaluation.converged,
    };
}

const char* finite_source_method_name(FiniteSourceMethod method)
{
    switch (method) {
    case FiniteSourceMethod::point_source:
        return "point_source";
    case FiniteSourceMethod::hexadecapole:
        return "hexadecapole";
    case FiniteSourceMethod::inverse_ray_cartesian:
        return "inverse_ray_cartesian";
    case FiniteSourceMethod::inverse_ray_polar:
        return "inverse_ray_polar";
    case FiniteSourceMethod::inverse_ray_spine:
        return "inverse_ray_spine";
    case FiniteSourceMethod::source_plane_quadrature:
        return "source_plane_quadrature";
    default:
        return "unknown";
    }
}

FiniteSourceResult FiniteSourceMagnifier::triple_mag(
    const model::TripleLensGeometry& geometry,
    SourcePosition source,
    double source_radius,
    double point_source_magnification,
    const PointSourceMagnifier* point_magnifier_hint) const
{
    PointSourceMagnifier local_point_magnifier;
    const PointSourceMagnifier& point_magnifier =
        point_magnifier_hint != nullptr ? *point_magnifier_hint : local_point_magnifier;
    if (source_radius <= 0.0) {
        const auto point = point_magnifier.triple_mag0(geometry, source);
        FiniteSourceDecision decision {FiniteSourceMethod::point_source, 0, "zero source radius"};
        return {point.magnification, point.image_count, decision, 0.0, 0, true};
    }
    if (settings_.finite_mode <= 0) {
        FiniteSourceDecision decision {FiniteSourceMethod::point_source, 0, "point-source mode"};
        return {point_source_magnification, 0, decision, 0.0, 0, true};
    }

    const auto& caustics = cached_triple_caustic_branches(
        geometry,
        settings_.caustic_bins);
    const double point_threshold = settings_.kinji_threshold * source_radius;
    const double caustic_distance = triple_caustic_distance(
        geometry,
        caustics,
        source,
        1.5 * point_threshold);
    if (!has_explicit_finite_source_tolerance(settings_) &&
        std::isfinite(caustic_distance) && caustic_distance > point_threshold) {
        FiniteSourceDecision decision {
            FiniteSourceMethod::point_source,
            settings_.caustic_bins * 6,
            "triple source outside caustic point-source threshold",
        };
        return {point_source_magnification, 0, decision, 0.0, 0, true};
    }

    // Triple calibration: hex self-consistency underestimates error within
    // five source radii of a caustic.  The binary threshold is not reusable
    // here because triple topology adds unresolved local caustic structure.
    const double hex_dist_threshold = std::max(settings_.hex_threshold, 5.0) * source_radius;
    const bool near_caustic =
        std::isfinite(caustic_distance) && caustic_distance < hex_dist_threshold;
    FiniteSourceSettings runtime_settings = settings_;
    if (settings_.automatic_source_bins) {
        const auto calibrated_resolution = calibrated_triple_resolution(
            geometry,
            source_radius,
            caustic_distance,
            point_source_magnification,
            settings_.limb_darkening_c,
            explicit_finite_source_relative_budget(
                settings_, point_source_magnification),
            settings_.max_source_bins);
        runtime_settings.source_bins = calibrated_resolution.source_bins;
        if (runtime_settings.polar_source_bins <= 0) {
            runtime_settings.polar_source_bins = calibrated_resolution.source_bins;
        }
    }
    // Seed-complete polar integration is both accurate and substantially
    // cheaper than Cartesian for the calibrated high-magnification population.
    // Keep the inner three-rho band on the topology-aware Cartesian/source-
    // plane path; outside it the augmented centre/caustic/boundary seed set
    // removes the former missing-fold-component failures.  This is independent
    // of the five-rho hex guard: polar has a complete image-component search,
    // while hex is still a local Taylor approximation.
    constexpr double kTriplePolarPointMagnificationThreshold = 100.0;
    constexpr double kTriplePolarCausticDistanceFactor = 3.0;
    const bool auto_polar =
        settings_.finite_mode == 4 &&
        std::isfinite(caustic_distance) &&
        caustic_distance >= kTriplePolarCausticDistanceFactor * source_radius &&
        std::abs(point_source_magnification) >=
            kTriplePolarPointMagnificationThreshold;

    // For explicit polar (finite_mode==2), retain the close-caustic Cartesian
    // fallback.  Auto polar is already excluded inside three source radii.
    const double polar_fallback_distance =
        std::max(settings_.hex_threshold, 1.0) * source_radius;
    const bool polar_needs_cartesian_fallback =
        settings_.finite_mode == 2 &&
        std::isfinite(caustic_distance) &&
        caustic_distance < polar_fallback_distance;

    // The seed set is a property of the epoch, not of the grid: it depends on
    // the geometry, the source centre and rho, and augmented_triple_image_seeds
    // never sees a resolution.  Every resolution an inverse-ray path tries must
    // therefore share one computation -- otherwise the certified probes, which
    // are degree-ten root solves, are repeated at each tolerance retry and each
    // coarse comparison for an answer that cannot change.  The binary paths
    // already thread their seeds this way.  Built on first use so an epoch that
    // routes to the hexadecapole or to source-plane quadrature never pays for
    // it.
    std::vector<SourcePosition> epoch_seeds;
    bool epoch_support_proven = true;
    bool epoch_seeds_ready = false;
    const auto seeds_for_epoch = [&]() -> const std::vector<SourcePosition>& {
        if (!epoch_seeds_ready) {
            epoch_seeds = augmented_triple_image_seeds(
                point_magnifier, geometry, source, source_radius, caustics,
                &epoch_support_proven);
            epoch_seeds_ready = true;
        }
        return epoch_seeds;
    };

    // Explicit polar and calibrated high-magnification auto polar take
    // precedence over the hexadecapole approximation.
    if ((settings_.finite_mode == 2 || auto_polar) && !polar_needs_cartesian_fallback) {
        FiniteSourceSettings polar_settings = runtime_settings;
        if (auto_polar) {
            polar_settings.polar_grid_ratio =
                std::max(active_polar_grid_ratio(polar_settings), 12.0);
        }
        const auto& shared_seeds = seeds_for_epoch();
        const auto evaluation = evaluate_polar_to_tolerance(
            polar_settings,
            epoch_support_proven,
            [&](const FiniteSourceSettings& active, PolarAreaDiagnostics* diagnostics) {
                return inverse_ray_polar_triple_mag(
                    point_magnifier, geometry, source, source_radius, active, this,
                    &shared_seeds, diagnostics);
            });
        FiniteSourceDecision decision {
            FiniteSourceMethod::inverse_ray_polar,
            estimate_polar_cost(polar_settings),
            auto_polar ? "auto polar triple inverse-ray for high magnification"
                       : "triple polar inverse-ray",
        };
        if (!std::isfinite(evaluation.magnification)) {
            return {evaluation.magnification, 0, decision, std::nan(""), 0, false};
        }
        return {
            evaluation.magnification,
            0,
            decision,
            evaluation.error_estimate,
            evaluation.refinement_level,
            evaluation.converged,
        };
    }

    if (!near_caustic && settings_.adaptive_hex_threshold > 0.0) {
        const auto hex = hexadecapole_triple(
            point_magnifier,
            geometry,
            source,
            source_radius,
            runtime_settings,
            &point_source_magnification);
        double hex_threshold = settings_.adaptive_hex_threshold;
        if (settings_.finite_source_tol > 0.0 || settings_.finite_source_reltol > 0.0) {
            hex_threshold =
                std::max(
                    settings_.finite_source_reltol,
                    settings_.finite_source_tol /
                        std::max(std::abs(hex.magnification), 1.0));
        }
        if (std::isfinite(hex.magnification) && hex.relative_error <= hex_threshold) {
            FiniteSourceDecision decision {
                FiniteSourceMethod::hexadecapole,
                kHexadecapoleEvaluations,
                "triple hexadecapole self-consistency check passed",
            };
            return {
                hex.magnification,
                0,
                decision,
                hex.relative_error * std::max(std::abs(hex.magnification), 1.0),
                0,
                true,
            };
        }
    }

    // Distance alone cannot distinguish a genuine crossing from a grazing or
    // tangent source limb.  Pay for the branch walk only in the narrow
    // topology-sensitive band; ordinary point/hex rows incur no extra scan.
    constexpr double kTripleTopologyScanDistanceFactor = 2.0;
    bool tangent_band = false;
    if (settings_.finite_mode == 4 && std::isfinite(caustic_distance) &&
        caustic_distance < kTripleTopologyScanDistanceFactor * source_radius) {
        const auto scan = scan_caustic_branches(caustics.branches, source, source_radius);
        const bool caustic_enters_disk =
            caustic_distance < source_radius ||
            scan.any_vertex_inside || !scan.crossing_probes.empty();
        const bool chord_band = !caustic_enters_disk &&
            scan.min_distance >= 0.95 * source_radius &&
            scan.min_distance < 1.35 * source_radius;
        tangent_band = !caustic_enters_disk && !chord_band &&
            std::abs(scan.min_distance - source_radius) < 0.35 * source_radius;
        // The topology scan is retained only for the inverse-ray fail-closed
        // error floor.  Automatic triple evaluation deliberately does not
        // consume a source-plane value; source-plane quadrature is reserved
        // for explicit preplanned diagnostic calls.
    }

    const auto apply_triple_tangent_floor = [&](FiniteSourceResult result) {
        if (tangent_band) {
            const double error_floor =
                5.0e-3 * std::max(std::abs(result.magnification), 1.0);
            result.error_estimate = std::max(result.error_estimate, error_floor);
            result.converged = false;
        }
        return result;
    };

    {
        LegacyAreaDiagnostics diagnostics;
        const auto& shared_seeds = seeds_for_epoch();
        const bool support_proven = epoch_support_proven;
        FiniteSourceSettings active_settings = runtime_settings;
        // Fixed-grid calls keep their requested resolution.  Automatic calls
        // start no lower than the conservative baseline grid; the previous
        // 1e-2 experiment at 16 bins exposed unresolved local triple structure
        // that a same-grid error indicator cannot see, whereas 32 bins met the
        // loose holdout budget.
        if (settings_.automatic_source_bins) {
            const int automatic_cap = std::max(settings_.max_source_bins, 1);
            active_settings.source_bins = std::min(
                automatic_cap, std::max(active_settings.source_bins, 32));
            if (active_settings.polar_source_bins > 0) {
                active_settings.polar_source_bins = active_settings.source_bins;
            }
        }
        const auto evaluate_cartesian = [&]
            (const FiniteSourceSettings& grid, LegacyAreaDiagnostics* active_diagnostics) {
            return inverse_ray_cartesian_triple_mag(
                point_magnifier,
                geometry,
                caustics,
                source,
                source_radius,
                grid,
                this,
                &shared_seeds,
                active_diagnostics,
                nullptr,
                point_source_magnification);
        };
        double cartesian_magnification =
            evaluate_cartesian(active_settings, &diagnostics);
        if (std::isfinite(cartesian_magnification) && cartesian_magnification > 0.0) {
            // Fail closed: an unproven support means a component of the source
            // disk was never entered, so the area is short by an unknown
            // amount and no error estimate derived from it can be trusted.
            double error_estimate = support_proven
                ? diagnostics.estimated_error
                : std::numeric_limits<double>::infinity();
            bool converged = support_proven && finite_source_error_within_budget(
                settings_, cartesian_magnification, error_estimate);
            int refinement_level = 0;
            const int maximum_bins = std::max(settings_.max_source_bins, 1);
            const bool needs_pair_check = settings_.automatic_source_bins ||
                has_explicit_finite_source_tolerance(settings_);
            int reusable_source_bins = 0;
            int reusable_polar_source_bins = 0;
            double reusable_magnification =
                std::numeric_limits<double>::quiet_NaN();
            while (support_proven && needs_pair_check) {
                // The geometric area indicator and an independent nested grid
                // guard different failure modes.  Let the measured pair rule
                // in both directions, then grow only when the common error
                // budget still is not met.  Fixed-nbin calls retain their one
                // diagnostic comparison and never change the requested grid.
                reconcile_with_half_resolution(
                    settings_,
                    active_settings,
                    cartesian_magnification,
                    [&](const FiniteSourceSettings& grid) {
                        if (grid.source_bins == reusable_source_bins &&
                            grid.polar_source_bins == reusable_polar_source_bins) {
                            return reusable_magnification;
                        }
                        return evaluate_cartesian(grid, nullptr);
                    },
                    error_estimate,
                    converged);
                if (active_settings.source_bins <= 16 &&
                    std::abs(cartesian_magnification) > 1000.0 &&
                    diagnostics.rows_with_multiple_runs > 1000) {
                    // A coincident coarse/fine estimate cannot certify a
                    // topology that is represented by thousands of split
                    // rows at the hard resolution cap.  Preserve fail-closed
                    // behavior; an uncapped automatic call will take another
                    // rung immediately below.
                    converged = false;
                }
                if (converged || !settings_.automatic_source_bins ||
                    active_settings.source_bins >= maximum_bins) {
                    break;
                }
                const auto assessment = assess_finite_source_error(
                    settings_, cartesian_magnification, error_estimate,
                    support_proven);
                const int retry_bins = next_finite_source_resolution(
                    active_settings.source_bins,
                    maximum_bins,
                    assessment.shortfall,
                    cartesian_error_convergence_order(diagnostics));
                if (retry_bins <= active_settings.source_bins) {
                    break;
                }
                const int previous_source_bins = active_settings.source_bins;
                const int previous_polar_source_bins =
                    active_settings.polar_source_bins;
                const double previous_magnification = cartesian_magnification;
                active_settings.source_bins = retry_bins;
                if (active_settings.polar_source_bins > 0) {
                    active_settings.polar_source_bins = retry_bins;
                }
                LegacyAreaDiagnostics retry_diagnostics;
                const double retry_magnification =
                    evaluate_cartesian(active_settings, &retry_diagnostics);
                if (!std::isfinite(retry_magnification) ||
                    retry_magnification <= 0.0) {
                    active_settings.source_bins = previous_source_bins;
                    active_settings.polar_source_bins =
                        previous_polar_source_bins;
                    break;
                }
                cartesian_magnification = retry_magnification;
                diagnostics = retry_diagnostics;
                // A doubling retry asks for this previous fine grid as the
                // next half-resolution comparison.  Carry its already-computed
                // value forward; non-doubling retries simply miss this exact
                // key and keep the historical evaluation path.
                reusable_source_bins = previous_source_bins;
                reusable_polar_source_bins = previous_polar_source_bins;
                reusable_magnification = previous_magnification;
                error_estimate = diagnostics.estimated_error;
                converged = finite_source_error_within_budget(
                    settings_, cartesian_magnification, error_estimate);
                ++refinement_level;
            }
            FiniteSourceDecision decision {
                FiniteSourceMethod::inverse_ray_cartesian,
                estimate_cartesian_cost(active_settings),
                polar_needs_cartesian_fallback
                    ? "triple polar used cartesian fallback for caustic-crossing"
                    : "triple finite-source cartesian image-plane inverse ray",
            };
            return apply_triple_tangent_floor({
                cartesian_magnification,
                diagnostics.seed_count,
                decision,
                error_estimate,
                refinement_level,
                converged,
            });
        }
    }

    // All automatic image-plane attempts failed to return a finite result.
    // Do not silently switch integrators here: report the inverse-ray failure
    // and let the caller handle the non-converged result explicitly.
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_cartesian,
        estimate_cartesian_cost(runtime_settings),
        "triple inverse-ray failed; source-plane fallback disabled",
    };
    return apply_triple_tangent_floor({
        std::numeric_limits<double>::quiet_NaN(),
        0,
        decision,
        std::numeric_limits<double>::infinity(),
        0,
        false,
    });
}

HexadecapoleDiagnosticResult diagnostic_hexadecapole_binary(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    const FiniteSourceSettings& settings)
{
    const PointSourceMagnifier point_magnifier;
    const auto result = hexadecapole_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, settings);
    const auto derivatives =
        point_magnifier.binary_mag0_with_derivatives(separation, mass_ratio, source);
    const double derivative_relative_error =
        derivatives.derivative_error_indicator * source_radius * source_radius /
        std::max(std::abs(result.magnification), 1.0e-10);
    return {result.magnification, result.relative_error, derivative_relative_error};
}


FiniteSourceResult FiniteSourceMagnifier::binary_mag_preplanned(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double point_source_magnification,
    FiniteSourceMethod method,
    int resolution,
    const std::vector<SourcePosition>* center_image_seeds,
    const PointSourceMagnifier* point_magnifier_hint) const
{
    PointSourceMagnifier local_point_magnifier;
    const PointSourceMagnifier& point_magnifier = point_magnifier_hint != nullptr
        ? *point_magnifier_hint : local_point_magnifier;

    if (source_radius <= 0.0 || method == FiniteSourceMethod::point_source) {
        return {
            point_source_magnification,
            0,
            {FiniteSourceMethod::point_source, 0, "preplanned point-source"},
            0.0,
            0,
            std::isfinite(point_source_magnification),
        };
    }

    if (method == FiniteSourceMethod::hexadecapole) {
        const auto hex = hexadecapole_binary(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings_, &point_source_magnification);
        return {
            hex.magnification,
            0,
            {FiniteSourceMethod::hexadecapole, kHexadecapoleEvaluations,
             "preplanned hexadecapole"},
            hex.relative_error * std::max(std::abs(hex.magnification), 1.0),
            0,
            std::isfinite(hex.magnification),
        };
    }

    if (method == FiniteSourceMethod::source_plane_quadrature) {
        const int panels = std::max(resolution, 1);
        const double value = binary_source_plane_chord_quadrature(
            point_magnifier, separation, mass_ratio, source, source_radius,
            settings_, panels);
        return {
            value,
            0,
            {FiniteSourceMethod::source_plane_quadrature,
             (panels * 8) * (panels * 8),
             "preplanned composite chord quadrature"},
            0.0,
            0,
            std::isfinite(value),
        };
    }


    if (method == FiniteSourceMethod::inverse_ray_cartesian ||
        method == FiniteSourceMethod::inverse_ray_polar) {
        FiniteSourceSettings planned = settings_;
        const int bins = std::max(resolution, 1);
        planned.source_bins = bins;
        planned.polar_source_bins = bins;
        planned.max_source_bins = bins;
        planned.automatic_source_bins = false;
        // Accuracy was established against A_ref during warmup.  Clearing the
        // runtime budget prevents fixed-grid compatibility probes; support is
        // still certified inside fixed_inverse_ray_binary.
        planned.finite_source_tol = 0.0;
        planned.finite_source_reltol = 0.0;
        FiniteSourceDecision decision {
            method,
            method == FiniteSourceMethod::inverse_ray_polar
                ? estimate_polar_cost(planned)
                : estimate_cartesian_cost(planned),
            method == FiniteSourceMethod::inverse_ray_polar
                ? "preplanned polar inverse-ray"
                : "preplanned cartesian inverse-ray",
        };
        return fixed_inverse_ray_binary(
            point_magnifier,
            separation,
            mass_ratio,
            source,
            source_radius,
            planned,
            this,
            std::move(decision),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN(),
            center_image_seeds,
            true,
            point_source_magnification);
    }

    // The experimental spine is not calibrated by the binary warm-up ladder.
    // Preserve existing behaviour if a future caller supplies such a plan.
    return binary_mag(
        separation,
        mass_ratio,
        source,
        source_radius,
        point_source_magnification,
        center_image_seeds,
        true,
        &point_magnifier);
}

CartesianTraceResult FiniteSourceMagnifier::binary_cartesian_trace(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    int source_bins) const
{
    CartesianTraceResult result;
    result.complete = false;
    constexpr int kMaximumTraceSourceBins = 256;
    if (!(std::isfinite(separation) && std::isfinite(mass_ratio) &&
          std::isfinite(source.x) && std::isfinite(source.y) &&
          std::isfinite(source_radius)) ||
        separation <= 0.0 || mass_ratio <= 0.0 || source_radius <= 0.0 ||
        (source_bins > kMaximumTraceSourceBins) ||
        (source_bins <= 0 && settings_.source_bins > kMaximumTraceSourceBins)) {
        return result;
    }

    FiniteSourceSettings trace_settings = settings_;
    trace_settings.finite_mode = 1;
    trace_settings.automatic_source_bins = false;
    trace_settings.source_bins = std::max(
        source_bins > 0 ? source_bins : settings_.source_bins, 1);
    trace_settings.finite_source_tol = 0.0;
    trace_settings.finite_source_reltol = 0.0;
    result.source_bins = trace_settings.source_bins;
    result.lattice_spacing = source_radius /
        static_cast<double>(trace_settings.source_bins);

    const PointSourceMagnifier point_magnifier;
    const auto mapper = make_binary_lens_mapper(separation, mass_ratio);
    const auto& caustics = binary_caustic_branches(separation, mass_ratio);
    std::size_t centre_seed_count=0;
    const auto seeds = augmented_image_seeds(
        point_magnifier,
        mapper,
        separation,
        mass_ratio,
        source,
        source_radius,
        std::numeric_limits<double>::infinity(),
        nullptr,
        &caustics,nullptr,&centre_seed_count,&binary_critical_curve_branches_cached(separation,mass_ratio));
    result.centre_image_seeds.assign(seeds.begin(),seeds.begin()+centre_seed_count);
    const double point_source_magnification = std::abs(
        point_magnifier.binary_mag0(separation, mass_ratio, source).magnification);
    detail::CartesianRunFillTrace trace;
    result.magnification = inverse_ray_cartesian_binary_mag(
        point_magnifier,
        separation,
        mass_ratio,
        source,
        source_radius,
        trace_settings,
        this,
        &seeds,
        nullptr,
        point_source_magnification,
        &trace);
    result.complete = !trace.truncated;
    result.events = std::move(trace.events);
    result.boundary_diagnostics = std::move(trace.boundary_diagnostics);
    result.coverage_requested=trace.coverage_requested;result.coverage_certified=trace.coverage_certified;
    result.coverage_boxes=trace.coverage_boxes;result.coverage_unresolved=trace.coverage_unresolved;result.coverage_components=trace.coverage_components;
    return result;
}

FiniteSourceResult FiniteSourceMagnifier::binary_mag(
    double separation,
    double mass_ratio,
    SourcePosition source,
    double source_radius,
    double point_source_magnification,
    const std::vector<SourcePosition>* center_image_seeds,
    bool point_source_magnification_is_exact,
    const PointSourceMagnifier* point_magnifier_hint) const
{
    if (!lightweight_flux_error_mode() && result_cache_valid_ && result_cache_separation_ == separation &&
        result_cache_mass_ratio_ == mass_ratio && result_cache_source_x_ == source.x &&
        result_cache_source_y_ == source.y && result_cache_source_radius_ == source_radius &&
        result_cache_point_magnification_ == point_source_magnification) {
        return result_cache_;
    }

    PointSourceSafetyEvaluation point_safety;
    bool point_safety_available = false;
    double caustic_distance_out = std::numeric_limits<double>::infinity();
    const auto cache_and_return = [&](FiniteSourceResult result) {
        if (has_explicit_finite_source_tolerance(settings_) &&
            !finite_source_error_within_budget(
                settings_, result.magnification, result.error_estimate)) {
            result.converged = false;
        }
        if (point_safety_available) {
            result.point_source_quadrupole_indicator =
                point_safety.diagnostic.quadrupole_indicator;
            result.point_source_cusp_indicator = point_safety.diagnostic.cusp_indicator;
            result.point_source_ghost_indicator = point_safety.diagnostic.ghost_indicator;
            result.point_source_planetary_distance2 = point_safety.planetary_distance2;
            result.point_source_safety_tolerance = point_safety.absolute_tolerance;
            result.point_source_ghost_count = point_safety.diagnostic.ghost_count;
            result.point_source_safety_flags =
                (point_safety.quadrupole_cusp_safe ? 1 : 0) |
                (point_safety.ghost_safe ? 2 : 0) |
                (point_safety.planetary_safe ? 4 : 0);
        }
        result.caustic_distance = caustic_distance_out;
        // Request-local adaptive results must not populate the fixed-result cache.
        if(lightweight_flux_error_mode())return result;
        result_cache_valid_ = true;
        result_cache_separation_ = separation;
        result_cache_mass_ratio_ = mass_ratio;
        result_cache_source_x_ = source.x;
        result_cache_source_y_ = source.y;
        result_cache_source_radius_ = source_radius;
        result_cache_point_magnification_ = point_source_magnification;
        result_cache_ = result;
        return result;
    };

    PointSourceMagnifier local_point_magnifier;
    const PointSourceMagnifier& point_magnifier =
        point_magnifier_hint != nullptr ? *point_magnifier_hint : local_point_magnifier;
    if (source_radius <= 0.0) {
        const auto point = point_magnifier.binary_mag0(separation, mass_ratio, source);
        FiniteSourceDecision decision {FiniteSourceMethod::point_source, 0, "zero source radius"};
        return cache_and_return({point.magnification, point.image_count, decision, 0.0, 0, true});
    }

    if (settings_.finite_mode <= 0) {
        FiniteSourceDecision decision {FiniteSourceMethod::point_source, 0, "point-source mode"};
        return cache_and_return({point_source_magnification, 0, decision, 0.0, 0, true});
    }

    point_safety = evaluate_point_source_safety(
        point_magnifier,
        separation,
        mass_ratio,
        source,
        source_radius,
        point_source_magnification,
        settings_);
    point_safety_available = true;

    // A dynamic binary changes separation at every epoch, which invalidates
    // the exact caustic cache.  Do not build a 1400-phase caustic merely to
    // rediscover that an obviously smooth point is safe.  This preflight uses
    // only the already-computed local derivative/ghost/planetary diagnostics
    // and the maximum (near-caustic) factor of the later distance-dependent
    // point safety rule.  Anything that does not pass this stricter test falls
    // through unchanged to measured caustic geometry.
    if (settings_.hex_threshold > 0.0 && point_safety.point_source_safe()) {
        double requested_relative = settings_.adaptive_hex_threshold > 0.0
            ? settings_.adaptive_hex_threshold : 1.0e-3;
        if (settings_.finite_source_tol > 0.0 || settings_.finite_source_reltol > 0.0) {
            requested_relative =
                std::max(
                    settings_.finite_source_reltol,
                    settings_.finite_source_tol /
                        std::max(std::abs(point_source_magnification), 1.0));
        }
        const double estimated_error =
            (point_safety.diagnostic.quadrupole_indicator +
                point_safety.diagnostic.cusp_indicator) *
            source_radius * source_radius;
        constexpr double kPreflightPointSafety = 30.0;
        const double derivative_relative_error = estimated_error /
            std::max(std::abs(point_source_magnification), 1.0e-10);
        if (kPreflightPointSafety * derivative_relative_error <= requested_relative) {
            FiniteSourceDecision decision {
                FiniteSourceMethod::point_source,
                1,
                "strict local point-source preflight passed",
            };
            return cache_and_return({
                point_source_magnification,
                point_safety.diagnostic.image_count,
                decision,
                estimated_error,
                0,
                true});
        }
    }

    // Fast PS exit for sources outside the caustic bounding box by kinji_threshold·ρ.
    // The caustic cache is built (or validated) inside binary_sampled_caustic_distance,
    // making the bbox members available immediately after the call.
    const double bbox_margin = settings_.kinji_threshold * source_radius;
    constexpr bool adaptive_ir_requested = false;
    const double adaptive_bbox_margin = adaptive_ir_requested
        ? std::max(bbox_margin, 60.0 * source_radius)
        : bbox_margin;
    const double topology_margin = binary_topology_boundary_margin(
        separation, mass_ratio);
    const double caustic_reuse_tolerance = std::min(
        0.25 * source_radius, 0.02 * topology_margin);
    double sampled_dist = binary_sampled_caustic_distance(
        separation, mass_ratio, source, adaptive_bbox_margin,
        caustic_reuse_tolerance);
    bool reused_caustic_geometry = caustic_cache_separation_ != separation;
    if (point_safety.point_source_safe() &&
        !reused_caustic_geometry &&
        (source.x < caustic_cache_min_x_ - adaptive_bbox_margin ||
        source.x > caustic_cache_max_x_ + adaptive_bbox_margin ||
        source.y < caustic_cache_min_y_ - adaptive_bbox_margin ||
        source.y > caustic_cache_max_y_ + adaptive_bbox_margin)) {
        FiniteSourceDecision decision {
            FiniteSourceMethod::point_source,
            settings_.caustic_bins * 4,
            "source outside caustic bounding box",
        };
        const double estimated_error =
            (point_safety.diagnostic.quadrupole_indicator +
                point_safety.diagnostic.cusp_indicator) *
            source_radius * source_radius;
        return cache_and_return({
            point_source_magnification,
            point_safety.diagnostic.image_count,
            decision,
            estimated_error,
            0,
            finite_source_error_within_budget(
                settings_, point_source_magnification, estimated_error),
        });
    }

    // VBM-style adaptive mode selection.  When the source center is far enough
    // from the caustic the hexadecapole approximation is tried first; its
    // self-consistency error estimate (|a4 correction| / |magnification|) then
    // determines whether hex is accurate enough or IR is needed.
    //
    // When the source is close to the caustic (sampled_dist < hex_threshold·ρ)
    // the hex Taylor expansion can give a misleadingly small a4 term even when
    // the result is wrong (e.g. a4≈0 when a1_plus and a1_cross both happen to
    // be large and cancel through a2rho2).  Skip hex and go straight to IR.
    // Use segment-based (refined) caustic distance for accurate near-caustic detection.
    // The sampled distance uses only discrete caustic points and can badly overestimate
    // the true distance (e.g. 7x rho) when the caustic is sparsely sampled; the segment
    // distance queries the actual line segments between consecutive points and returns the
    // correct distance.  We pass sampled_dist as a hint to skip the O(N) point scan.
    double refined_dist = binary_caustic_distance(
        separation, mass_ratio, source, sampled_dist,
        caustic_reuse_tolerance);
    // Approximate anchor geometry is only a far-field routing accelerator.
    // Rebuild at the exact epoch near the caustic and around the measured
    // 20-rho topology-release boundary.  This also guarantees that any later
    // inverse-ray seed request, which asks for exact branches, cannot inherit
    // an approximate caustic.
    if (reused_caustic_geometry && std::isfinite(refined_dist)) {
        const double distance_ratio = refined_dist / source_radius;
        if (distance_ratio < 6.0 ||
            std::abs(distance_ratio - 20.0) < 2.0) {
            sampled_dist = binary_sampled_caustic_distance(
                separation, mass_ratio, source, adaptive_bbox_margin, 0.0);
            refined_dist = binary_caustic_distance(
                separation, mass_ratio, source, sampled_dist, 0.0);
            reused_caustic_geometry = false;
        }
    }
    caustic_distance_out = refined_dist;

    // The ghost and planetary checks are inexpensive local-topology proxies.
    // They are deliberately conservative close to a caustic, but disconnected
    // binary caustics can make them veto a broad region even when the source is
    // far from every actual caustic segment.  Once the refined segment distance
    // reaches the same calibrated 20-rho point-source boundary used by the
    // triple-lens selector, the measured geometry supersedes those proxies.
    // Point-source acceptance still has to pass the tolerance-aware derivative
    // check below; otherwise the independently checked hexadecapole route is
    // tried before inverse rays.
    // The local ghost/planetary topology proxies are intentionally
    // conservative, but keeping their veto all the way to the point-source
    // boundary sends a broad, demonstrably smooth 10--20 rho annulus to
    // inverse rays without even trying the independently error-checked
    // hexadecapole approximation.  Once the nearest caustic segment has been
    // measured at ten source radii, that geometry is a stronger topology
    // discriminator than the local proxy.  Hexadecapole still has to pass its
    // own tolerance-aware self-consistency check before it is accepted.
    constexpr double kMeasuredTopologyReleaseDistance = 10.0;
    const bool measured_topology_safe =
        std::isfinite(refined_dist) &&
        refined_dist >= kMeasuredTopologyReleaseDistance * source_radius;
    const bool effective_topology_safe =
        point_safety.topology_safe() || measured_topology_safe;
    const bool effective_point_source_safe =
        point_safety.quadrupole_cusp_safe && effective_topology_safe;
    const auto calibrated_resolution = calibrated_binary_resolution(
        mass_ratio,
        source_radius,
        refined_dist,
        point_source_magnification,
        settings_.limb_darkening_c,
        settings_.finite_mode,
        settings_.finite_source_tol,
        settings_.finite_source_reltol,
        settings_.max_source_bins);
    FiniteSourceSettings runtime_settings = settings_;
    if (settings_.automatic_source_bins) {
        runtime_settings.source_bins = calibrated_resolution.source_bins;
        if (runtime_settings.polar_source_bins <= 0) {
            runtime_settings.polar_source_bins = calibrated_resolution.source_bins;
        }
    }
    const double hex_dist_threshold = settings_.hex_threshold * source_radius;
    const bool near_caustic = refined_dist < hex_dist_threshold;
    double rejected_hex_magnification = std::numeric_limits<double>::quiet_NaN();
    if (!near_caustic && effective_topology_safe) {
        auto caustic_distance_safety = [&]() {
            double safety = 1.0;
            if (source_radius >= 1.0e-3 && std::isfinite(refined_dist) &&
                source_radius > 0.0) {
                const double dist_ratio = refined_dist / source_radius;
                const double t =
                    settings_.hex_threshold / std::max(dist_ratio, settings_.hex_threshold);
                safety = std::max(1.0, 30.0 * t * t * t);
            }
            return safety;
        };
        double requested_relative =
            settings_.adaptive_hex_threshold > 0.0 ? settings_.adaptive_hex_threshold : 1.0e-3;
        if (settings_.finite_source_tol > 0.0 || settings_.finite_source_reltol > 0.0) {
            requested_relative =
                std::max(
                    settings_.finite_source_reltol,
                    settings_.finite_source_tol /
                        std::max(std::abs(point_source_magnification), 1.0));
        }
        if (settings_.hex_threshold > 0.0 && effective_point_source_safe) {
            const double estimated_error =
                (point_safety.diagnostic.quadrupole_indicator +
                    point_safety.diagnostic.cusp_indicator) *
                source_radius * source_radius;
            const double derivative_threshold = requested_relative / caustic_distance_safety();
            const double derivative_relative_error = estimated_error /
                std::max(std::abs(point_source_magnification), 1.0e-10);
            if (derivative_relative_error <= derivative_threshold) {
                FiniteSourceDecision decision {
                    FiniteSourceMethod::point_source,
                    1,
                    "point-source safety checks passed",
                };
                return cache_and_return({
                    point_source_magnification,
                    point_safety.diagnostic.image_count,
                    decision,
                    estimated_error,
                    0,
                    true});
            }
        }

        const double* known_point = point_source_magnification_is_exact ?
            &point_source_magnification :
            nullptr;
        const auto hex = hexadecapole_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, settings_,
            known_point);
        double hex_threshold = settings_.adaptive_hex_threshold;
        if (settings_.finite_source_tol > 0.0 || settings_.finite_source_reltol > 0.0) {
            requested_relative =
                std::max(
                    settings_.finite_source_reltol,
                    settings_.finite_source_tol /
                        std::max(std::abs(hex.magnification), 1.0));
            // Graduate hex_safety by caustic distance.  The hex self-consistency
            // check underestimates the actual error most severely when the source
            // boundary is near a caustic fold (dist_ratio ~ hex_threshold).  For
            // sources far from the caustic the Taylor expansion is reliable and
            // needs little or no safety margin.
            //
            // Power-law: safety = 30 * (hex_threshold / dist_ratio)^3
            // clamped to [1, 30].  This gives safety≈30 at dist_ratio=hex_threshold
            // and safety≈1 at dist_ratio≈3*hex_threshold (~9 source radii away).
            // For small sources (rho < 1e-3) hex is always reliable: safety=1.
            const double hex_safety = caustic_distance_safety();
            hex_threshold = std::min(hex_threshold, requested_relative / hex_safety);
        }
        if (hex.relative_error <= hex_threshold) {
            FiniteSourceDecision decision {
                FiniteSourceMethod::hexadecapole,
                kHexadecapoleEvaluations,
                "hexadecapole self-consistency check passed",
            };
            return cache_and_return({
                hex.magnification,
                0,
                decision,
                hex.relative_error * std::max(std::abs(hex.magnification), 1.0),
                0,
                true});
        }
        rejected_hex_magnification = hex.magnification;
    }

    // Grazing-caustic diagnostics.  Binary automatic evaluation deliberately
    // stays on the same calibrated inverse-ray path as the explicit Cartesian
    // and polar modes.  The scan is retained only to mark a genuinely tangent
    // limb, where the inverse-ray result must fail closed rather than silently
    // claim a precision that the caustic polyline cannot certify.  The
    // source-plane routine remains available to explicit preplanned callers
    // and to the separate triple-lens implementation, but it is not an
    // automatic binary fallback.
    constexpr double kGrazeQuadratureDistanceFactor = 2.0;
    bool tangent_band = false;
    if (std::isfinite(refined_dist) &&
        refined_dist < kGrazeQuadratureDistanceFactor * source_radius) {
        const auto scan = scan_caustic_branches(
            binary_caustic_branches(separation, mass_ratio), source, source_radius);
        // Segment-interior minima prove a caustic entry that vertex-only
        // flags cannot see when both endpoints remain outside the disk.
        const bool caustic_enters_disk =
            scan.min_distance < source_radius ||
            scan.any_vertex_inside || !scan.crossing_probes.empty();
        // Split the near-limb regimes by how deep the nearest polyline chord
        // dips into the disk.  The shallow band stays on the ordinary seeded
        // inverse-ray path; a deeper dip is a genuine crossing hidden by the
        // polyline sag and is handled conservatively by the error floor below.
        const bool chord_band = !caustic_enters_disk &&
            scan.min_distance >= 0.95 * source_radius &&
            scan.min_distance < 1.35 * source_radius;
        tangent_band = !caustic_enters_disk && !chord_band &&
            std::abs(scan.min_distance - source_radius) < 0.35 * source_radius;
        // The scan above supplies the fail-closed tangent flag for the
        // inverse-ray result.  Binary automatic mode never consumes a
        // source-plane value here.
    }

    // A caustic tangent to the source limb can hide a crossing sliver below
    // both the grid and the caustic-polyline resolution; inverse rays then
    // miss the corresponding image flux.  Report the risk instead of silently
    // claiming convergence.
    const auto apply_tangent_band_floor = [&](FiniteSourceResult result) {
        if (tangent_band) {
            const double error_floor =
                5.0e-3 * std::max(std::abs(result.magnification), 1.0);
            if (result.error_estimate < error_floor) {
                result.error_estimate = error_floor;
                result.converged = false;
            }
        }
        return result;
    };

    const bool auto_polar =
        settings_.finite_mode == 4 &&
        calibrated_resolution.prefer_polar;
    if (settings_.finite_mode == 2 || auto_polar) {
        FiniteSourceSettings inverse_ray_settings = runtime_settings;
        if (auto_polar) {
            // Auto mode uses polar only for high magnification, where the polar
            // topology is valuable but the low-magnification radius-aware
            // angular resolution is unnecessarily expensive.  Keep the explicit
            // mode=2 path exact to the user-provided polar grid settings.
            inverse_ray_settings.polar_grid_ratio =
                std::max(active_polar_grid_ratio(inverse_ray_settings), 12.0);
        }
        FiniteSourceDecision decision {
            FiniteSourceMethod::inverse_ray_polar,
            estimate_polar_cost(inverse_ray_settings),
            auto_polar ? "auto polar inverse-ray for high magnification" : "polar inverse-ray",
        };
        return cache_and_return(apply_tangent_band_floor(fixed_inverse_ray_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, inverse_ray_settings, this,
            decision, refined_dist, rejected_hex_magnification, center_image_seeds)));
    }
    if (settings_.finite_mode == 3) {
        if (has_explicit_finite_source_tolerance(settings_)) {
            FiniteSourceDecision explicit_tolerance_decision {
                FiniteSourceMethod::inverse_ray_cartesian,
                estimate_cartesian_cost(runtime_settings),
                "experimental spine bypassed for explicit tolerance",
            };
            return cache_and_return(apply_tangent_band_floor(fixed_inverse_ray_binary(
                point_magnifier,
                separation,
                mass_ratio,
                source,
                source_radius,
                runtime_settings,
                this,
                explicit_tolerance_decision,
                refined_dist,
                rejected_hex_magnification,
                center_image_seeds)));
        }
        return cache_and_return(fixed_inverse_ray_spine_binary(
            point_magnifier, separation, mass_ratio, source, source_radius, runtime_settings, this));
    }
    FiniteSourceDecision decision {
        FiniteSourceMethod::inverse_ray_cartesian,
        estimate_cartesian_cost(runtime_settings),
        "cartesian inverse-ray",
    };
    auto result = fixed_inverse_ray_binary(
        point_magnifier, separation, mass_ratio, source, source_radius, runtime_settings, this, decision,
        refined_dist, rejected_hex_magnification, center_image_seeds);
    return cache_and_return(apply_tangent_band_floor(std::move(result)));
}

} // namespace lcbinint::magnification

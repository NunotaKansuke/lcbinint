#pragma once

#include "lcbinint/magnification/algebraic_boundary.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace lcbinint::magnification::critical_events {

// The event polynomial has one deliberately frozen binary64 geometry boundary.
// Keep it separate from the legacy band search until the new policy is wired.
struct FrozenGeometry {
    double a = 0.0;
    double m1 = 0.0;
    double m2 = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    double rho = 0.0;
};

struct CriticalEventInput {
    SourcePosition source;
    double rho = 0.0;
    double q = 0.0;
    double separation = 0.0;
};

struct PhysicalResidual {
    bool finite = false;
    bool non_pole = false;
    double value = 0.0;
    double scale = 0.0;
};

struct CertifiedInterval {
    double lo = 0.0;
    double hi = 0.0;
};

struct CriticalEvent {
    std::size_t event_id = 0;
    CertifiedInterval u_box;
    CertifiedInterval radius_box;
    CertifiedInterval chart_box;
    bool cot_chart = false;
    double radius = 0.0;
    double theta = 0.0;
};

// Result of the bounded numerical kernel only.  `success` certifies exact
// discriminant construction, interval isolation of every positive root, and
// one strict Krawczyk lift per root.  It does not certify the separate
// physical-pole/residual, scaled-IFT, or topology gates required by the full
// production plan.
struct CriticalEventNumericalResult {
    bool success = false;
    std::string unsafe_reason;
    std::vector<CriticalEvent> events;
    int isolation_calls = 0;
    int root_bisections = 0;
    int lift_candidates = 0;
};

struct CriticalEventPlan {
    bool success = false;
    std::string unsafe_reason;
    std::vector<CriticalEvent> events;
    // Consecutive cells are deliberately not merged across an event.
    std::vector<CertifiedInterval> active_bands;
    std::vector<std::pair<std::size_t, std::size_t>> band_event_ids;
    int root_work = 0;
    int lift_work = 0;
};

// New-policy precondition.  It intentionally rejects, rather than canonicalizes,
// non-positive q/separation/rho; legacy paths retain their historical behavior.
bool freeze_positive_geometry(
    const CriticalEventInput& input,
    FrozenGeometry* frozen,
    std::string* reason);

// Uncleared lens-map residual at an image coordinate.  This is the physical
// gate paired with the algebraic P(R,t) lift; it is not used by legacy
// band discovery.
PhysicalResidual physical_boundary_residual(
    const FrozenGeometry& frozen,
    SourcePosition raw_source,
    double image_x,
    double image_y);

// Exact-Dyadic / outward-interval enumeration and exact-center Krawczyk lift.
// This is exposed separately so tests cannot mistake a numerical-kernel PASS
// for acceptance of the not-yet-connected production plan.
CriticalEventNumericalResult enumerate_and_lift_critical_events(
    const CriticalEventInput& input);

// Focused arithmetic/isolation checks retained from the accepted probe.
bool critical_event_numerical_unit_checks();

// Opt-in generic finder.  It is intentionally independent of the legacy
// integrator while E04 connects event-tagged bands in a later change.
CriticalEventPlan find_critical_events(const CriticalEventInput& input);

}  // namespace lcbinint::magnification::critical_events

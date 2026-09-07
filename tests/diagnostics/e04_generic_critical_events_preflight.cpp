#include "lcbinint/magnification/algebraic_critical_event_core.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace ce = lcbinint::magnification::critical_events;

namespace {

int failures = 0;

void check(bool condition, const char* label)
{
    std::printf("%s %s\n", condition ? "ok" : "FAIL", label);
    if (!condition) ++failures;
}

bool full_positive_radius_square_subset(const ce::CriticalEvent& event)
{
    if (!std::isfinite(event.u_box.lo) || !std::isfinite(event.u_box.hi) ||
        !std::isfinite(event.radius_box.lo) ||
        !std::isfinite(event.radius_box.hi) ||
        !(event.radius_box.lo > 0.0) ||
        !(event.radius_box.lo <= event.radius_box.hi)) {
        return false;
    }
    const double lower = std::nextafter(
        event.radius_box.lo * event.radius_box.lo,
        -std::numeric_limits<double>::infinity());
    const double upper = std::nextafter(
        event.radius_box.hi * event.radius_box.hi,
        std::numeric_limits<double>::infinity());
    return std::isfinite(lower) && std::isfinite(upper) &&
        lower >= event.u_box.lo && upper <= event.u_box.hi;
}

void check_expected_certified_events(
    const ce::CriticalEventInput& input,
    std::size_t expected_events,
    const char* label)
{
    const auto numerical = ce::enumerate_and_lift_critical_events(input);
    std::printf(
        "REGRESSION label=%s success=%d reason=%s events=%zu\n",
        label, numerical.success ? 1 : 0, numerical.unsafe_reason.c_str(),
        numerical.events.size());
    check(numerical.success, label);
    check(numerical.events.size() == expected_events,
          "regression has its expected certified event count");
    for (const auto& event : numerical.events) {
        check(full_positive_radius_square_subset(event),
              "full positive radius box squared is outward-contained in U");
    }
}

}  // namespace

int main()
{
    check(ce::critical_event_numerical_unit_checks(),
          "exact-dyadic and interval-isolation units");

    ce::FrozenGeometry frozen;
    std::string reason;
    const ce::CriticalEventInput regular5{
        {0.0555363373628048, 0.22652972262448196}, 0.01, 10.0, 1.2};
    check(ce::freeze_positive_geometry(regular5, &frozen, &reason),
          "regular5 positive geometry freezes");
    check(frozen.sx == 1.1464454282718957,
          "regular5 uses the rounded binary64 sx boundary");

    const auto numerical = ce::enumerate_and_lift_critical_events(regular5);
    std::printf(
        "NUMERICAL success=%d reason=%s events=%zu calls=%d bisections=%d "
        "lift_candidates=%d\n",
        numerical.success ? 1 : 0, numerical.unsafe_reason.c_str(),
        numerical.events.size(), numerical.isolation_calls,
        numerical.root_bisections, numerical.lift_candidates);
    check(numerical.success, "regular5 numerical module succeeds");
    check(numerical.events.size() == 6,
          "regular5 has six certified positive-root lifts");
    for (std::size_t i = 0; i < numerical.events.size(); ++i) {
        const auto& event = numerical.events[i];
        std::printf(
            "EVENT id=%zu U=[%.17g,%.17g] R=[%.17g,%.17g] "
            "chart=%s Z=[%.17g,%.17g] centerR=%.17g theta=%.17g\n",
            event.event_id, event.u_box.lo, event.u_box.hi,
            event.radius_box.lo, event.radius_box.hi,
            event.cot_chart ? "cot" : "tan", event.chart_box.lo,
            event.chart_box.hi, event.radius, event.theta);
        check(event.event_id == i, "event IDs are sorted and typed");
        check(event.u_box.lo <= event.radius * event.radius &&
                  event.radius * event.radius <= event.u_box.hi,
              "returned Krawczyk center remains in its U box");
        check(event.radius_box.lo <= event.radius &&
                  event.radius <= event.radius_box.hi,
              "event representative is inside returned Krawczyk box");
        check(full_positive_radius_square_subset(event),
              "regular5 full positive radius box squared is outward-contained in U");
    }

    // These inputs exercised the former noncertifying seed drift.  They are
    // numerical-kernel regressions only; limb-darkening coefficients do not
    // enter this geometry-only API.
    check_expected_certified_events(
        ce::CriticalEventInput{{0.037, -0.021}, 0.007, 0.02, 0.9}, 10,
        "q_exchange_low geometry has ten certified events");
    check_expected_certified_events(
        ce::CriticalEventInput{{-0.037, -0.021}, 0.007, 50.0, 0.9}, 10,
        "q_exchange_high geometry has ten certified events");
    check_expected_certified_events(
        ce::CriticalEventInput{{0.08, 0.03}, 0.01, 0.001, 1.1}, 6,
        "LD geometry has six certified events");

    const auto plan = ce::find_critical_events(regular5);
    check_expected_certified_events(
        ce::CriticalEventInput{{0.002, 0.0007}, 0.001, 0.001, 1.1}, 12,
        "low-q geometry has twelve certified events");
    check(plan.success, "regular5 physical and topology plan succeeds");
    check(plan.events.size() == 6,
          "physical plan retains the numerical module evidence");
    check(plan.unsafe_reason.empty() && plan.active_bands.size() == 3 &&
              plan.band_event_ids.size() == plan.active_bands.size(),
          "regular5 has three event-tagged bands with the empty gap preserved");

    for (const auto invalid : {
             ce::CriticalEventInput{{0.0, 0.0}, 0.01, 0.1, 0.0},
             ce::CriticalEventInput{{0.0, 0.0}, 0.01, 0.0, 1.0},
             ce::CriticalEventInput{{0.0, 0.0}, 0.0, 0.1, 1.0},
             ce::CriticalEventInput{
                 {std::numeric_limits<double>::quiet_NaN(), 0.0},
                 0.01, 0.1, 1.0}}) {
        reason.clear();
        check(!ce::freeze_positive_geometry(invalid, &frozen, &reason),
              "invalid/nonpositive geometry fails closed");
        check(!reason.empty(), "invalid geometry supplies a reason");
    }

    std::printf("SUMMARY failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}

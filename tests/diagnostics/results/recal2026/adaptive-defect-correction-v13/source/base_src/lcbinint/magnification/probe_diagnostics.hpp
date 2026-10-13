#pragma once

#include <string>

namespace lcbinint::magnification {

// Calibration instrumentation for the certificate seeding stage.
//
// The production cached-binary path uses the source-centre images followed by
// the component certificate.  Each certificate probe is a full quintic solve.
//
// Whether that is affordable or wasteful is a measurement, not an opinion, and
// the shipping build reports neither of the two numbers it needs: what share of
// an evaluation the probes are, and whether the answer moves when a stage is
// removed.  Both are provided here, gated on environment variables, and both
// default to exactly the shipping behaviour when unset.
//
// This is deliberately not a public option.  A knob that can silently weaken
// the completeness argument does not belong in the API; it belongs in the
// calibration harness, which has to run the weakened variants in order to
// report what they cost.  ``probe_policy()`` is read once per process.

// Per-thread tallies.  Zero-initialised, and only written while
// ``probe_stats_enabled()``; the counters are thread-local because the row loop
// is an OpenMP parallel for, and a harness that wants a total either runs the
// library single-threaded or sums per thread.
struct ProbeCounters {
    // Point-image solves, which is the unit of probe cost: everything else in
    // the seeding stage is arithmetic on the results.
    long long certified_solves = 0;
    // What the certificate proposed against what was actually spent.  The
    // traversal stops a direction as soon as it departs from its baseline, so
    // ``certified_offered`` above ``certified_solves`` is the early-out working
    // rather than probes going missing.
    long long certified_offered = 0;
    long long certified_extrema = 0;
    // Support descriptors built, and disks whose support the ladder failed to
    // prove.  The second is the accuracy cost of an ablation: a support that
    // stops being proven is refused, not silently returned wrong.
    long long certifications = 0;
    long long unproven = 0;
    // Wall seconds, split so the polyline scan that builds the descriptor is
    // separable from the solves that consume it -- they are reduced by
    // different means.
    double certified_seconds = 0.0;
    double certify_seconds = 0.0;
};

ProbeCounters& probe_counters();
bool probe_stats_enabled();
void reset_probe_counters();

// Diagnostic controls for the certificate probe ladder.  There is deliberately
// no switch for restoring the retired ring or branch-grid heuristic.
struct ProbePolicy {
    // How deep the certified offset ladder goes, counted from the coarse end,
    // and whether the tangent directions are probed at all.  Dropping the
    // tangents is expected to lose cusps; it is kept as a knob so the corpus
    // can be shown to notice.
    int offsets = 8;
    bool tangents = true;
    bool normals = true;
};

const ProbePolicy& probe_policy();

// Parsed form of LCBININT_PROBE_POLICY, for the harness to echo back.
std::string probe_policy_description();

} // namespace lcbinint::magnification

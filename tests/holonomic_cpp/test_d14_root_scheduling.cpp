#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "lcbinint/magnification/holonomic/radial_events.hpp"

using namespace lcbinint::holonomic;
using namespace lcbinint::holonomic::re_detail;
using qf = __float128;

static void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static std::vector<Cplx<qf>> to_qf_roots(
    const std::array<Cplx<D14Real>, 14>& roots) {
    std::vector<Cplx<qf>> out;
    out.reserve(roots.size());
    for (const auto& root : roots)
        out.emplace_back(d14_to_qf(root.re), d14_to_qf(root.im));
    return out;
}

int main() {
    check(positive_detail::environment_ok(),
          "D14 root schedule arithmetic environment");
    const PrimaryFrame pf = PrimaryFrame::from(
        LensParams{0.3, 0.2, 0.01, 2.0, 1.0, true});
    const D14StructQf struct_qf = d14_struct_build(
        static_cast<qf>(pf.a), static_cast<qf>(pf.m0),
        static_cast<qf>(pf.X), static_cast<qf>(pf.Y),
        static_cast<qf>(pf.rho));
    const auto ascending = d14_expanded_from_struct(struct_qf);
    check(ascending.size() == 15, "D14 schedule fixture degree 14");
    std::vector<qf> desc(ascending.rbegin(), ascending.rend());
    const auto reference = aberth_d14_struct<qf>(
        struct_qf, desc.data(), 400, nullptr, static_cast<qf>(1e-25));
    check(reference.size() == 14, "D14 qf reference roots");

    const D14StructC<D14Real> struct_real = d14_struct_cast<D14Real>(struct_qf);
    std::array<Cplx<D14Real>, 14> initial{};
    for (int i = 0; i < 14; ++i)
        initial[i] = Cplx<D14Real>(d14_from_qf(reference[i].re),
                                   d14_from_qf(reference[i].im));

    // With scheduling disabled, the incumbent full-root path remains active
    // and no diagnostic root counters are populated.
    const auto incumbent = aberth_d14_real_mixed(
        struct_real, initial, 25, D14Real(1e-14), true);
    check(incumbent.finite && incumbent.converged,
          "unscheduled D14Real polish converged");
    check(std::all_of(incumbent.root_updates.begin(), incumbent.root_updates.end(),
                      [](int n) { return n == 0; }),
          "disabled scheduler does not collect per-root work");

    D14RealScheduleConfig tight;
    tight.enabled = true;
    tight.absolute_correction_tolerance = 1e-12;
    tight.relative_correction_separation_tolerance = 1e-12;
    tight.patience = 1;
    tight.cluster_relative_separation = 1e-10;
    tight.reactivate_step_separation = 0.25;
    const auto frozen = aberth_d14_real_mixed(
        struct_real, initial, 25, D14Real(1e-14), true, &tight);
    check(frozen.finite && frozen.converged,
          "scheduled D14Real polish converged through incumbent tolerance");
    check(std::accumulate(frozen.root_freezes.begin(), frozen.root_freezes.end(), 0) > 0,
          "scheduled D14Real root freezes observed");
    const auto frozen_qf = to_qf_roots(frozen.roots);
    qf scale = 0;
    for (qf coefficient : ascending)
        scale = std::max(scale, fabsq(coefficient));
    check(d14_worst_res(desc.data(), 14, frozen_qf, scale) < qf(1e-12),
          "scheduled root set preserves scalar residual");

    // Force one root to remain active while the other roots can freeze.  A
    // single large cluster threshold checks that any active member wakes its
    // frozen peers; all roots still participate in the Aberth interaction.
    auto perturbed = initial;
    perturbed[0].re = perturbed[0].re + D14Real(1e-4);
    std::array<int, 14> roles{};
    roles.fill(static_cast<int>(D14RootUse::PositiveRealCandidate));
    roles[0] = static_cast<int>(D14RootUse::OtherWarmCompleteness);
    D14RealScheduleConfig cluster;
    cluster.enabled = true;
    cluster.consumer_precision = true;
    cluster.physical_position_rtol = 1e6;
    cluster.soft_cut_position_rtol = 1e6;
    cluster.other_correction_separation_rtol = 1e-40;
    cluster.patience = 1;
    cluster.cluster_relative_separation = 1e6;
    cluster.reactivate_step_separation = 0.0;
    const auto woken = aberth_d14_real_mixed(
        struct_real, perturbed, 25, D14Real(1e-14), true, &cluster, &roles);
    check(woken.finite, "cluster wake D14Real candidate finite");
    check(woken.cluster_wakeups > 0 && woken.reactivations > 0,
          "active cluster member reactivates frozen peers");
    const auto woken_qf = to_qf_roots(woken.roots);
    check(d14_worst_res(desc.data(), 14, woken_qf, scale) < qf(1e-10),
          "reactivated root set remains a valid D14 candidate");

    std::cout << "D14 root-wise freeze/cluster wake checks passed; freezes="
              << std::accumulate(frozen.root_freezes.begin(), frozen.root_freezes.end(), 0)
              << " wakeups=" << woken.cluster_wakeups << '\n';
}

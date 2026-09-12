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

    // A failed in-place double sweep must expose the previous complete root
    // set, never a partially updated interaction state.  Exact coincident
    // seeds provide a deterministic early failure without relaxing any
    // downstream D14Real/qf certificate.
    const double quadratic[] = {1.0, 0.0, -1.0};
    const Cplx<double> coincident[] = {
        Cplx<double>(0.25, -0.5), Cplx<double>(0.25, -0.5)};
    const auto failed_presearch = d14_active_presearch(
        quadratic, 2, 5, coincident, 1e-12, 2, true);
    check(!failed_presearch.finite && failed_presearch.failure_code == 4,
          "double presearch identifies a coincident-root interaction failure");
    check(failed_presearch.failure_iteration == 1 &&
              failed_presearch.failure_root == 0 &&
              failed_presearch.failure_partner == 1,
          "double presearch reports the first failed root pair");
    check(failed_presearch.has_last_finite_roots &&
              failed_presearch.last_finite_roots.size() == 2 &&
              failed_presearch.roots.size() == 2,
          "double presearch retains a last-finite root checkpoint");
    for (int i = 0; i < 2; ++i) {
        check(std::isfinite(failed_presearch.roots[i].re) &&
                  std::isfinite(failed_presearch.roots[i].im),
              "failed double presearch returns only finite checkpoint roots");
        check(failed_presearch.roots[i].re == coincident[i].re &&
                  failed_presearch.roots[i].im == coincident[i].im,
              "failed sweep does not leak a partial in-place update");
    }

    // The double Aberth correction can have a perfectly finite quotient even
    // when the unscaled numerator products overflow binary64.
    const Cplx<double> large_quotient =
        Cplx<double>(8e159, 2e159) / Cplx<double>(1e149, 2e148);
    check(std::isfinite(large_quotient.re) &&
              std::isfinite(large_quotient.im),
          "scale-safe complex division handles finite large quotient");
    check(std::fabs(large_quotient.re / 8.076923076923077e10 - 1.0) < 1e-14 &&
              std::fabs(large_quotient.im / 3.846153846153846e9 - 1.0) < 1e-14,
          "scale-safe complex quotient agrees with analytic result");

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

    // Regression fixture from case 92 / d_bin 0.  Its initial balanced roots
    // make |P| and |P'-P*S| large enough that the unscaled double complex
    // quotient overflowed before producing an otherwise ordinary correction.
    // The normal production presearch must now reach D14Real without a qf
    // cold restart, and the unchanged qf residual gate remains satisfied.
    const PrimaryFrame case92 = PrimaryFrame::from(LensParams{
        0.016817436700993973, -1.7129651189458623e-05,
        3.1430126951461966e-05, 1.0 / 0.0045820894088154547,
        3.8691964055932191, true});
    const D14StructQf case92_struct = d14_struct_build(
        static_cast<qf>(case92.a), static_cast<qf>(case92.m0),
        static_cast<qf>(case92.X), static_cast<qf>(case92.Y),
        static_cast<qf>(case92.rho));
    const auto case92_ascending = d14_expanded_from_struct(case92_struct);
    check(case92_ascending.size() == 15, "case 92 D14 fixture degree");
    qf case92_scale = 0;
    for (qf coefficient : case92_ascending)
        case92_scale = std::max(case92_scale, fabsq(coefficient));
    std::vector<qf> case92_descending(case92_ascending.rbegin(),
                                      case92_ascending.rend());
    V2Profile case92_profile;
    D14Solve case92_solution;
    {
        V2ProfileScope scope(case92_profile);
        case92_solution = solve_d14(case92_descending, 14, true, nullptr,
                                    &case92_struct);
    }
    check(case92_solution.roots.size() == 14,
          "case 92 D14 returns all fourteen roots");
    check(case92_profile.d14_qf_cold_calls == 0 &&
              case92_profile.d14_real_finite_calls == 1,
          "case 92 avoids qf cold restart through finite D14Real polish");
    check(d14_worst_res(case92_descending.data(), 14,
                        case92_solution.roots, case92_scale) < qf(1e-13),
          "case 92 retains the existing qf residual certificate");
    check(case92_profile.d14_real_nonconverged == 0,
          "case 92 D14Real handoff converges before acceptance");

    // This input exposed a separate fail-closed gap: the finite D14Real
    // iterate had a small backward residual but had not converged, and its
    // near-zero roots changed the positive-real topology.  It must therefore
    // continue through the existing qf cold path instead of being accepted
    // on residual alone.
    const PrimaryFrame case6 = PrimaryFrame::from(LensParams{
        0.11778654951701377, -0.29572518963135319,
        0.32822578487891141, 1.0 / 0.00034997931344948061,
        2.9086634667611073, true});
    const D14StructQf case6_struct = d14_struct_build(
        static_cast<qf>(case6.a), static_cast<qf>(case6.m0),
        static_cast<qf>(case6.X), static_cast<qf>(case6.Y),
        static_cast<qf>(case6.rho));
    const auto case6_ascending = d14_expanded_from_struct(case6_struct);
    std::vector<qf> case6_descending(case6_ascending.rbegin(),
                                     case6_ascending.rend());
    V2Profile case6_profile;
    D14Solve case6_solution;
    {
        V2ProfileScope scope(case6_profile);
        case6_solution = solve_d14(case6_descending, 14, true, nullptr,
                                   &case6_struct);
    }
    check(case6_solution.roots.size() == 14,
          "case 6 returns all fourteen D14 roots after nonconverged polish");
    int case6_positive_real_roots = 0;
    for (const auto& root : case6_solution.roots) {
        if (root.re > 0 && fabsq(root.im) <=
                (qf)1e-8 * ((qf)1 + fabsq(root.re)))
            ++case6_positive_real_roots;
    }
    check(case6_profile.d14_real_nonconverged == 1 &&
              case6_profile.d14_qf_warm_calls == 1 &&
              case6_profile.d14_qf_cold_calls == 0,
          "case 6 recovers a finite nonconverged D14Real seed with qf warm polish");
    check(case6_positive_real_roots == 6,
          "case 6 qf warm polish restores the six-root physical classification");

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

    // The residual reduction must preserve the legacy value, including
    // roots with unequal norms. Test actual solved and perturbed sets.
    for (int exponent : {-80, -20, 0, 20, 80}) {
        auto probe = woken_qf;
        for (int i=0; i<14; ++i) {
            probe[i].re += scalbnq(qf(i+1), exponent);
            probe[i].im -= scalbnq(qf(14-i), exponent-2);
        }
        qf expected = 0;
        for (const auto& root : probe) {
            qf residual = cabs(poly_eval_c(desc.data(),14,root)) /
                          (scale + qf(1e-300));
            if (residual > expected) expected = residual;
        }
        check(d14_worst_res(desc.data(),14,probe,scale)==expected,
              "max-norm reduction preserves original qf residual");
    }

    std::cout << "D14 root-wise freeze/cluster wake checks passed; freezes="
              << std::accumulate(frozen.root_freezes.begin(), frozen.root_freezes.end(), 0)
              << " wakeups=" << woken.cluster_wakeups << '\n';
}

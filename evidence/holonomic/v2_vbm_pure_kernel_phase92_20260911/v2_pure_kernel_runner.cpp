#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/holonomic_ode_transport.hpp"
#include "lcbinint/magnification/holonomic/holonomic_transport.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

using namespace lcbinint::holonomic;
using Clock = std::chrono::steady_clock;

static double median3(double a, double b, double c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: v2_pure_kernel_runner INPUT OUTPUT\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);
    if (!in || !out) {
        std::cerr << "cannot open input/output\n";
        return 2;
    }

    // Force the shipped V2 value path: root-pair transport + K-rule/rescue,
    // with the experimental ODE lane disabled.  No production router is
    // modified by this benchmark.
    holo_mv_transport_override() = 1;
    holo_holonomic_transport_override() = 1;
    holo_ode_transport_override() = 0;

    out << "# fair V2 pure finite-source kernel; HEAD=8a9f701; n_r=64; "
           "timer=flux_value_integrate only; LensParams/PrimaryFrame/"
           "classify_cells(D14+topology)/I-O outside timer; no epoch_value; "
           "mapping=LensParams{time,y,rho,1/q,s,true}; repeats=3 median\n";
    out << "# case_id configuration_id profile d_bin_index epoch_index s q rho x y time u X reference v2_mu F0 F_half status_code status ms ratio_v2_over_reference relative_error\n";
    out << std::setprecision(17);

    std::string line;
    std::size_t count = 0;
    std::size_t topo_nonok = 0;
    std::size_t value_nonok = 0;
    volatile double sink = 0.0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int case_id = 0, configuration_id = 0, d_bin_index = 0, epoch_index = 0;
        std::string profile;
        double s = 0, q = 0, rho = 0, x = 0, y = 0, time = 0, u = 0,
               X = 0, reference = 0;
        if (!(ss >> case_id >> configuration_id >> profile >> d_bin_index >>
              epoch_index >> s >> q >> rho >> x >> y >> time >> u >> X >>
              reference)) {
            std::cerr << "malformed input row: " << line << "\n";
            return 3;
        }

        // The corpus VBM call is BinaryMag(..., -x, y, ...).  The verified
        // V2 equivalent uses the swapped mass convention and reflected
        // source coordinate represented by this parameter pack.
        const LensParams p{time, y, rho, 1.0 / q, s, true};
        const ScopedFlushDenormals fp_guard;
        const PrimaryFrame pf = PrimaryFrame::from(p);
        const TopologyResult topo = classify_cells(pf);
        if (topo.status != Status::OK) ++topo_nonok;

        // Match the existing VBM protocol's warmed direct-call timing while
        // keeping all object/geometry construction out of the hot interval.
        const FluxValue warm = flux_value_integrate(64, u, pf, topo);
        sink += warm.F0 + warm.F_half;
        double elapsed[3]{};
        FluxValue measured;
        for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = Clock::now();
            measured = flux_value_integrate(64, u, pf, topo);
            const auto t1 = Clock::now();
            elapsed[rep] =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            sink += measured.F0 + measured.F_half;
        }
        const double ms = median3(elapsed[0], elapsed[1], elapsed[2]);
        const EpochValue ev = epoch_value_blend(measured, p, u, pf);
        if (ev.status != Status::OK) ++value_nonok;
        const double ratio = ev.mu / reference;
        const double relerr = std::fabs(ev.mu - reference) /
                              std::max(std::fabs(reference), 1e-300);
        out << case_id << ' ' << configuration_id << ' ' << profile << ' '
            << d_bin_index << ' ' << epoch_index << ' ' << s << ' ' << q << ' '
            << rho << ' ' << x << ' ' << y << ' ' << time << ' ' << u << ' '
            << X << ' ' << reference << ' ' << ev.mu << ' ' << ev.F0 << ' '
            << ev.F_half << ' ' << static_cast<int>(ev.status) << ' '
            << to_string(ev.status) << ' ' << ms << ' ' << ratio << ' '
            << relerr << '\n';
        ++count;
        if ((count % 256) == 0)
            std::cerr << "processed " << count << " rows\n";
    }
    std::cerr << "wrote " << count << " rows; topology_nonok=" << topo_nonok
              << " value_nonok=" << value_nonok << " sink=" << sink << '\n';
    return 0;
}

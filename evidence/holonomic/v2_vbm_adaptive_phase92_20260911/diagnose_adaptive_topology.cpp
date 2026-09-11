#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: diagnose_adaptive_topology INPUT OUTPUT\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);
    if (!in || !out) return 2;
    out << "case_id configuration_id profile d_bin_index epoch_index "
           "topology_status topology_cells empty_cells arc_cells full_cells "
           "degenerate_cells physical_events\n";
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int case_id = 0, configuration_id = 0, d_bin_index = 0, epoch_index = 0;
        std::string profile;
        double s = 0, q = 0, rho = 0, x = 0, y = 0, time = 0, u = 0,
               X = 0, reference = 0;
        if (!(ss >> case_id >> configuration_id >> profile >> d_bin_index >>
              epoch_index >> s >> q >> rho >> x >> y >> time >> u >> X >>
              reference)) return 3;
        (void)x;
        (void)u;
        (void)X;
        (void)reference;
        const LensParams p{time, y, rho, 1.0 / q, s, true};
        const TopologyResult topo = classify_cells(
            PrimaryFrame::from(p), nullptr, nullptr,
            /*retain_adaptive_metadata=*/true);
        int empty = 0, arcs = 0, full = 0, degenerate = 0;
        for (const auto& cell : topo.cells) {
            if (cell.kind == ArcKind::kEmpty) ++empty;
            else if (cell.kind == ArcKind::kArcs) ++arcs;
            else if (cell.kind == ArcKind::kFull) ++full;
            else if (cell.kind == ArcKind::kDegenerate) ++degenerate;
        }
        int physical = 0;
        for (const auto& event : topo.events)
            if (event.physically_real && event.kind == "physical_real")
                ++physical;
        out << case_id << ' ' << configuration_id << ' ' << profile << ' '
            << d_bin_index << ' ' << epoch_index << ' '
            << to_string(topo.status) << ' ' << topo.cells.size() << ' '
            << empty << ' ' << arcs << ' ' << full << ' ' << degenerate << ' '
            << physical << '\n';
    }
    return 0;
}

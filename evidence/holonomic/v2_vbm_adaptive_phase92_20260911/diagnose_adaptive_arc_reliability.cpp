#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"

using namespace lcbinint::holonomic;

struct Counts {
    long probes = 0, arcs = 0, mismatch = 0, full = 0, degenerate = 0;
    long endpoint = 0, width = 0, finite = 0;
};

static void inspect(const PrimaryFrame& pf, const TopologyResult& topo,
                    Counts& c) {
    for (const auto& cell : topo.cells) {
        if (cell.kind != ArcKind::kArcs) continue;
        for (int k = 1; k <= 128; ++k) {
            const double R = cell.r_lo + (cell.r_hi - cell.r_lo) *
                             (double(k) / 129.0);
            ++c.probes;
            QuarticWarm qw;
            RootPairWarm rpw;
            const auto as = arc_intervals(R, pf, &qw, &rpw);
            if (as.kind == ArcKind::kFull) { ++c.full; continue; }
            if (as.kind == ArcKind::kDegenerate) { ++c.degenerate; continue; }
            if (as.kind != cell.kind ||
                (as.kind == ArcKind::kArcs &&
                 int(as.arcs.size() * 2) != cell.n_crossings)) {
                ++c.mismatch;
                continue;
            }
            for (const auto& arc : as.arcs) {
                auto pe = polish_endpoint(R, arc[0], pf);
                auto pl = polish_endpoint(R, arc[1], pf);
                if (!pe.reliable || !pl.reliable) { ++c.endpoint; continue; }
                double te = pe.theta, tl = pl.theta;
                if (tl <= te) tl += kTwoPi;
                const auto ge = phi_val_dtheta(R, te, pf);
                const auto gl = phi_val_dtheta(R, tl, pf);
                if (!std::isfinite(ge.dphi_dtheta) ||
                    !std::isfinite(gl.dphi_dtheta) ||
                    ge.dphi_dtheta == 0.0 || gl.dphi_dtheta == 0.0) {
                    ++c.endpoint;
                    continue;
                }
                double de = std::fabs(ge.phi / ge.dphi_dtheta);
                double dl = std::fabs(gl.phi / gl.dphi_dtheta);
                double map_noise = std::numeric_limits<double>::epsilon() *
                    (std::fabs(R) + std::fabs(pf.X) + std::fabs(pf.Y) + 1) /
                    pf.rho;
                de += (map_noise + std::numeric_limits<double>::epsilon() *
                       std::fabs(te) * std::fabs(ge.dphi_dtheta)) /
                      std::fabs(ge.dphi_dtheta);
                dl += (map_noise + std::numeric_limits<double>::epsilon() *
                       std::fabs(tl) * std::fabs(gl.dphi_dtheta)) /
                      std::fabs(gl.dphi_dtheta);
                if (!(tl - te > de + dl) || !std::isfinite(de + dl))
                    ++c.width;
                else
                    ++c.finite;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream in(argv[1]);
    if (!in) return 2;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int case_id = 0, configuration_id = 0, d_bin_index = 0, epoch_index = 0;
        std::string profile;
        double s=0,q=0,rho=0,x=0,y=0,time=0,u=0,X=0,reference=0;
        if (!(ss >> case_id >> configuration_id >> profile >> d_bin_index >>
              epoch_index >> s >> q >> rho >> x >> y >> time >> u >> X >>
              reference)) return 3;
        if (!((case_id == 9 && d_bin_index == 2 && epoch_index == 7) ||
              (case_id == 92 && d_bin_index == 0))) continue;
        const LensParams p{time, y, rho, 1.0 / q, s, true};
        const auto topo = classify_cells(PrimaryFrame::from(p), nullptr, nullptr,
                                         /*retain_adaptive_metadata=*/true);
        Counts c;
        inspect(PrimaryFrame::from(p), topo, c);
        std::cout << case_id << ' ' << profile << ' ' << d_bin_index << ' '
                  << epoch_index << ' ' << std::setprecision(17) << rho << ' '
                  << c.probes << ' ' << c.mismatch << ' ' << c.full << ' '
                  << c.degenerate << ' ' << c.endpoint << ' ' << c.width << ' '
                  << c.finite << '\n';
    }
    return 0;
}

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "lcbinint/magnification/finite_source_magnifier.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/magnification/holonomic/seeded_radial_frontier.hpp"

using namespace lcbinint::holonomic;
using namespace lcbinint::magnification;

struct Row {
    int case_id = 0;
    std::string profile;
    double target = 0.0;
    int d_bin = 0;
    int epoch = 0;
    double s = 0.0, q = 0.0, rho = 0.0, y = 0.0, time = 0.0;
    double u = 0.0, reference = 0.0, previous = 0.0;
};

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: diagnose_outliers INPUT.tsv OUTPUT.tsv [RAY_MAX_BINS=512]\n";
        return 2;
    }
    const int ray_max_bins = argc == 4 ? std::atoi(argv[3]) : 512;
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output) return 2;

    output << std::setprecision(17)
           << "case_id\tprofile\ttarget\td_bin\tepoch\ts\tq\trho\txs_input\tys\t"
              "reference\tprevious_seeded\tseeded_repeat\tseed_count\tcenter_count\t"
              "support_proven\tfrontier_reason\tfrontier_bands\tfrontier_nodes\t"
              "frontier_panels\td14_mu\td14_converged\td14_stop\td14_nodes\t"
              "d14_cells\td14_events\td14_cell_ranges\td14_arc_seed_audit\t"
              "seeded_cells\tseeded_events\t"
              "cart_128\tcart_256\tcart_512\tcart_1024\t"
              "polar_128\tpolar_256\tpolar_512\tpolar_1024\n";

    std::string line;
    std::getline(input, line);
    int count = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        Row r;
        std::istringstream is(line);
        if (!(is >> r.case_id >> r.profile >> r.target >> r.d_bin >> r.epoch
                 >> r.s >> r.q >> r.rho >> r.y >> r.time >> r.u
                 >> r.reference >> r.previous)) {
            std::cerr << "bad row: " << line << '\n';
            return 3;
        }

        // Match the paper benchmark's mass-swap and x-reflection mapping.
        LensParams p{r.time, r.y, r.rho, 1.0 / r.q, r.s, true};
        const lcbinint::SourcePosition source{p.xs, p.ys};
        PointSourceMagnifier point;
        FiniteSourceSettings settings;
        settings.caustic_bins = 1400;
        settings.limb_darkening_c = r.u;
        settings.automatic_source_bins = false;
        settings.finite_source_tol = 0.0;
        settings.finite_source_reltol = 0.0;
        FiniteSourceMagnifier finite(settings);
        const auto center_images = point.binary_images(p.a, p.q, source);
        std::vector<lcbinint::SourcePosition> center;
        center.reserve(center_images.size());
        for (const auto& image : center_images) center.push_back(image.position);
        bool support = false;
        const auto seeds = finite.cached_binary_image_seeds(
            point, p.a, p.q, source, p.rho,
            std::numeric_limits<double>::infinity(), &center, &support);
        std::vector<Cplx<double>> primary_seeds;
        primary_seeds.reserve(seeds.size());
        for (const auto& z : seeds)
            primary_seeds.push_back({z.x + (p.q < 1.0 ? p.a : 0.0), z.y});

        AdaptiveConfig cfg;
        cfg.tol.mu_rtol = r.target;
        cfg.gradient_policy = GradientPolicy::None;
        AdaptiveWorkspace frontier_workspace;
        const auto frontier = epoch_seeded_radial_frontier(
            p, r.u, primary_seeds, cfg, frontier_workspace, false);

        AdaptiveWorkspace d14_workspace;
        const auto d14 = epoch_adaptive(p, r.u, cfg, d14_workspace);
        const auto d14_topology = classify_cells(PrimaryFrame::from(p), nullptr,
                                                  nullptr, true);

        std::vector<double> cart, polar;
        const double point_mu = point.binary_mag0(p.a, p.q, source).magnification;
        for (int bins : {128, 256, 512, 1024}) {
            if (bins > ray_max_bins) {
                cart.push_back(std::numeric_limits<double>::quiet_NaN());
                polar.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            const auto c = finite.binary_mag_preplanned(
                p.a, p.q, source, p.rho, point_mu,
                FiniteSourceMethod::inverse_ray_cartesian, bins, &center, &point);
            cart.push_back(c.magnification);
            const auto pol = finite.binary_mag_preplanned(
                p.a, p.q, source, p.rho, point_mu,
                FiniteSourceMethod::inverse_ray_polar, bins, &center, &point);
            polar.push_back(pol.magnification);
        }

        const auto cells = [](const TopologyResult& t) {
            std::ostringstream text;
            for (std::size_t i = 0; i < t.cells.size(); ++i) {
                if (i) text << ';';
                const auto& c = t.cells[i];
                text << c.r_lo << ':' << c.r_hi << ':' << int(c.kind)
                     << ':' << c.n_crossings;
            }
            return text.str();
        };
        std::ostringstream arc_seed_audit;
        bool first_arc = true;
        for (const auto& c : d14_topology.cells) {
            if (c.kind != ArcKind::kArcs) continue;
            if (!first_arc) arc_seed_audit << ';';
            first_arc = false;
            const double mid = .5 * (c.r_lo + c.r_hi);
            const auto cert = certify_quartic(boundary_quartic(mid,
                PrimaryFrame::from(p)).p);
            std::size_t seeds_in_cell = 0;
            double nearest_seed = std::numeric_limits<double>::infinity();
            for (const auto& seed : primary_seeds) {
                const double radius = std::hypot(seed.re, seed.im);
                if (radius >= c.r_lo && radius <= c.r_hi) ++seeds_in_cell;
                nearest_seed = std::min(nearest_seed, std::abs(radius - mid));
            }
            arc_seed_audit << c.r_lo << ':' << c.r_hi << ':' << c.n_crossings
                           << ':' << cert.certified << ':' << cert.root_count
                           << ':' << seeds_in_cell << ':' << nearest_seed;
        }
        output << r.case_id << '\t' << r.profile << '\t' << r.target << '\t'
               << r.d_bin << '\t' << r.epoch << '\t' << r.s << '\t' << r.q
               << '\t' << r.rho << '\t' << p.xs << '\t' << p.ys << '\t'
               << r.reference << '\t' << r.previous << '\t' << frontier.integral.mu
               << '\t' << primary_seeds.size() << '\t' << center.size() << '\t'
               << support << '\t' << frontier.frontier.reason << '\t'
               << frontier.frontier.bands << '\t'
               << frontier.integral.stats.unique_nodes << '\t'
               << frontier.integral.stats.panels << '\t' << d14.mu << '\t'
               << d14.value_converged << '\t' << adaptive_stop_name(d14.stop) << '\t'
               << d14.stats.unique_nodes << '\t' << d14_topology.cells.size() << '\t'
               << d14_topology.events.size() << '\t' << cells(d14_topology) << '\t'
               << arc_seed_audit.str() << '\t'
               << cells(frontier.topology) << '\t'
               << frontier.topology.events.size() << '\t';
        for (std::size_t i = 0; i < cart.size(); ++i)
            output << cart[i] << (i + 1 == cart.size() ? '\t' : '\t');
        for (std::size_t i = 0; i < polar.size(); ++i)
            output << polar[i] << (i + 1 == polar.size() ? '\n' : '\t');
        output.flush();
        std::cerr << "done case=" << r.case_id << " profile=" << r.profile
                  << " D14=" << d14.mu << " cart1024=" << cart.back()
                  << " polar1024=" << polar.back() << '\n';
        ++count;
    }
    return count ? 0 : 4;
}

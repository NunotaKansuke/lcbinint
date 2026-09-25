// Independent direct-value FD reference for the remaining caustic LD X case.
// Reuses the Phase 72 QF/Sturm/direct-angular value integrator, but promotes
// only independently certified projective contacts in a copied topology so
// the reference radial map resolves the known theta=pi folds.
#define main phase72_reference_main
#include "../moving_map_gradient_phase72/rand035_reference/reference.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: projective_fd CASES_TSV OUTPUT_TSV\n";
        return 2;
    }
    LensParams base;
    bool found = false;
    std::ifstream input(argv[1]);
    std::string line, name;
    int row = 0, barycentric = 0;
    double u = 0.0, ignored_mu = 0.0;
    while (std::getline(input, line)) {
        ++row;
        std::istringstream ss(line);
        if (!(ss >> base.xs >> base.ys >> base.rho >> base.q >> base.a >>
              barycentric >> u >> ignored_mu >> name)) continue;
        if (row == 17 && name == "caustic-cross") {
            found = true;
            break;
        }
    }
    if (!found) return 3;
    base.barycentric = barycentric != 0;
    constexpr double flux_u = 0.5;

    const std::array<Resolution, 2> resolutions{{
        {64, 1, 128}, {128, 2, 256}
    }};
    const std::array<double, 2> hscales{{1.0e-3, 2.5e-4}};
    const std::array<int, 4> offsets{{-2, -1, 1, 2}};
    std::ofstream out(argv[2], std::ios::trunc);
    if (!out) return 4;
    out << std::setprecision(17)
        << "mode\tresolution\thscale\toffset\ttopology_status\t"
           "projective_folds\tmap_used\tfail\tfail_segment\tfail_node\t"
           "fail_R\tmu0\tmuhalf\tfd_dX\n";

    for (std::size_t ir = 0; ir < resolutions.size(); ++ir) {
        for (double hscale : hscales) {
            const double h = base.rho * hscale;
            std::array<double, 4> mu_off{}, mu_on{};
            bool all_off = true, all_on = true;
            for (int oi = 0; oi < 4; ++oi) {
                const int offset = offsets[oi];
                LensParams p = base;
                p.xs += offset * h;
                const TopologyResult raw = classify_cells(
                    PrimaryFrame::from(p), nullptr, nullptr, true);
                TopologyResult mapped = raw;
                adaptive_detail::certify_projective_p4_events(
                    mapped, PrimaryFrame::from(p));
                int projective_count = 0;
                for (auto& e : mapped.events) {
                    if (!e.projective_fold_certified) continue;
                    ++projective_count;
                    // The Phase72 independent integrator accepts only the
                    // generic physical-fold label. Promote on this local copy
                    // solely to select the equivalent squared radial map.
                    e.physically_real = true;
                    e.kind = "physical_real";
                }
                const auto ref_off = integrate_direct(p, raw, resolutions[ir]);
                const auto ref_on = integrate_direct(p, mapped, resolutions[ir]);
                mu_off[oi] = ref_off.mu_half;
                mu_on[oi] = ref_on.mu_half;
                all_off = all_off && ref_off.fail == 0 && std::isfinite(ref_off.mu_half);
                all_on = all_on && ref_on.fail == 0 && std::isfinite(ref_on.mu_half);
                const auto emit = [&](const char* mode, const RefValue& r,
                                      bool finite) {
                    out << "VALUE\tGL" << resolutions[ir].nr << "x"
                        << resolutions[ir].subdivisions << "x"
                        << resolutions[ir].angular << '\t' << hscale << '\t'
                        << offset << '\t' << static_cast<int>(raw.status) << '\t'
                        << projective_count << '\t' << mode << '\t' << r.fail
                        << '\t' << r.segment << '\t' << r.node << '\t'
                        << r.fail_R << '\t' << r.mu0 << '\t' << r.mu_half
                        << "\tNaN\n";
                    (void)finite;
                };
                emit("off", ref_off, std::isfinite(ref_off.mu_half));
                emit("on", ref_on, std::isfinite(ref_on.mu_half));
                out.flush();
            }
            double fd_off = NAN, fd_on = NAN;
            if (all_off)
                fd_off = (mu_off[0] - 8.0 * mu_off[1] +
                          8.0 * mu_off[2] - mu_off[3]) / (12.0 * h);
            if (all_on)
                fd_on = (mu_on[0] - 8.0 * mu_on[1] +
                         8.0 * mu_on[2] - mu_on[3]) / (12.0 * h);
            out << "FD5\tGL" << resolutions[ir].nr << "x"
                << resolutions[ir].subdivisions << "x"
                << resolutions[ir].angular << '\t' << hscale
                << "\t0\t0\t2\toff\t0\t-1\t-1\tNaN\tNaN\tNaN\t"
                << fd_off << '\n';
            out << "FD5\tGL" << resolutions[ir].nr << "x"
                << resolutions[ir].subdivisions << "x"
                << resolutions[ir].angular << '\t' << hscale
                << "\t0\t0\t2\ton\t0\t-1\t-1\tNaN\tNaN\tNaN\t"
                << fd_on << '\n';
        }
    }
    return 0;
}

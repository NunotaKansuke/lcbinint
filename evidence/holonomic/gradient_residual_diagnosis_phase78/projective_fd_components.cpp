#define main phase78_single_u_reference_main
#include "projective_fd.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: projective_fd_components CASES_TSV OUTPUT_TSV\n";
        return 2;
    }
    LensParams base;
    bool found = false;
    std::ifstream in(argv[1]);
    std::string line;
    int row = 0, barycentric = 0;
    double input_u = 0.0, reference_mu = 0.0;
    std::string name;
    while (std::getline(in, line)) {
        ++row;
        std::istringstream ss(line);
        if (!(ss >> base.xs >> base.ys >> base.rho >> base.q >> base.a >>
              barycentric >> input_u >> reference_mu >> name)) continue;
        if (row == 17 && name == "caustic-cross") {
            found = true;
            break;
        }
    }
    if (!found) return 3;
    base.barycentric = barycentric != 0;

    std::ofstream out(argv[2], std::ios::trunc);
    if (!out) return 4;
    out << std::setprecision(17)
        << "u\tresolution\thscale\tprojective_folds\tfd_dx\tstatus\t"
           "minus2\tminus1\tplus1\tplus2\n";
    for (double u : {0.0, 0.5, 1.0}) {
        const auto base_topology = cold_topology(base);
        auto marked = base_topology;
        adaptive_detail::certify_projective_p4_events(
            marked, PrimaryFrame::from(base));
        int folds = 0;
        for (const auto& e : marked.events)
            folds += e.projective_fold_certified ? 1 : 0;
        for (const auto [n, subdivisions] :
             {std::pair<int,int>{128,2}, {256,4}}) {
            for (double hscale : {1e-3, 2.5e-4}) {
                const double h = base.rho * hscale;
                std::array<double, 4> values{};
                bool ok = true;
                for (int i = 0; i < 4; ++i) {
                    const int offset = std::array<int,4>{{-2,-1,1,2}}[i];
                    LensParams p = base;
                    p.xs += offset * h;
                    const auto topo = cold_topology(p);
                    values[i] = direct_reference_projective(
                        p, u, topo, n, subdivisions);
                    ok = ok && std::isfinite(values[i]);
                }
                const double fd = ok ? (values[0] - 8.0 * values[1] +
                    8.0 * values[2] - values[3]) / (12.0 * h) : NAN;
                out << u << '\t' << "GL" << n << 'x' << subdivisions << '\t'
                    << hscale << '\t' << folds << '\t' << fd << '\t'
                    << (ok ? "finite" : "reference_failure");
                for (double value : values) out << '\t' << value;
                out << '\n';
            }
        }
    }
    return 0;
}

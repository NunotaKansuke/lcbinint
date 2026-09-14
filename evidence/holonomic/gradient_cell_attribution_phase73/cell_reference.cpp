// Independent cell-restricted value finite differences for Phase 73.
// Reuses the Phase72 QF/Sturm/direct-angular reference implementation, but
// integrates each base-cell's fixed physical-R interval separately while
// splitting at the perturbed topology events.
#define main phase72_reference_main
#include "../moving_map_gradient_phase72/rand035_reference/reference.cpp"
#undef main

#include <map>
#include <set>

namespace {

struct BaseCell { int index; double a, b; };
struct CaseRow {
    int row = 0;
    std::string name, parameter;
    double u = 0.0;
    LensParams p;
};
struct CellValue { double mu0 = NAN, muhalf = NAN; int fail = 0; };

double& parameter(LensParams& p, const std::string& name) {
    if (name == "X") return p.xs;
    if (name == "Y") return p.ys;
    if (name == "a") return p.a;
    throw std::runtime_error("unsupported parameter " + name);
}

CellValue integrate_fixed_interval(const LensParams& p,
                                   const TopologyResult& topo,
                                   const Resolution& cfg,
                                   double outer_a, double outer_b,
                                   bool need_ld) {
    CellValue out;
    if (topo.status != Status::OK || !(outer_b > outer_a)) {
        out.fail = 1;
        return out;
    }
    const PrimaryFrame pf = PrimaryFrame::from(p);
    Rule radial(cfg.nr);
    std::unique_ptr<Cheb1Dyn> angular;
    if (need_ld) angular.reset(new Cheb1Dyn(cfg.angular));

    // Keep the integration domain fixed at the base cell, but resolve every
    // event of the perturbed physical problem inside it.  This gives the FD
    // of the fixed-R cell integral, matching the old J*F_p quantity.
    std::vector<double> cuts{outer_a, outer_b};
    for (const auto& c : topo.cells) {
        if (c.r_lo > outer_a && c.r_lo < outer_b) cuts.push_back(c.r_lo);
        if (c.r_hi > outer_a && c.r_hi < outer_b) cuts.push_back(c.r_hi);
    }
    for (const auto& e : topo.events)
        if (std::isfinite(e.radius) && e.radius > outer_a && e.radius < outer_b)
            cuts.push_back(e.radius);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    long double sum0 = 0.0L, sumh = 0.0L;
    for (std::size_t seg = 1; seg < cuts.size(); ++seg) {
        const double a = cuts[seg - 1], b = cuts[seg];
        if (!(b > a)) continue;
        const FoldRadialMap map{a, b, is_physical_fold(topo, a),
                                is_physical_fold(topo, b)};
        for (int sub = 0; sub < cfg.subdivisions; ++sub) {
            const double xl = -1.0 + 2.0 * sub / cfg.subdivisions;
            const double xr = -1.0 + 2.0 * (sub + 1) / cfg.subdivisions;
            const double xc = 0.5 * (xl + xr), xh = 0.5 * (xr - xl);
            for (int k = 0; k < cfg.nr; ++k) {
                const auto rr = map(xc + xh * radial.x[k]);
                const double R = rr[0];
                const double rw = xh * radial.w[k] * rr[1];
                RefValue local;
                const RefArcs arcs = certified_arcs(R, pf, local);
                if (local.fail || arcs.kind == ArcKind::kDegenerate) {
                    out.fail = local.fail ? local.fail : 2;
                    return out;
                }
                double f0 = 0.0, fh = 0.0;
                if (arcs.kind == ArcKind::kFull) {
                    f0 = kTwoPi * R;
                    if (need_ld) {
                        for (int j = 0; j < cfg.angular; ++j) {
                            const double th = kTwoPi * (j + 0.5) / cfg.angular;
                            const Q ph = qphi(Q(R), Q(th), pf);
                            if (!(ph > Q(0)) || !finiteq(ph)) {
                                out.fail = 3;
                                return out;
                            }
                            fh += kTwoPi * R * static_cast<double>(sqrtq(ph)) /
                                  cfg.angular;
                        }
                    }
                } else if (arcs.kind == ArcKind::kArcs) {
                    for (const auto& arc : arcs.arcs) {
                        const Q hq = (arc.hi - arc.lo) / Q(2);
                        const Q mq = (arc.hi + arc.lo) / Q(2);
                        const double h = static_cast<double>(hq);
                        f0 += R * static_cast<double>(arc.hi - arc.lo);
                        if (need_ld) {
                            for (int j = 0; j < cfg.angular; ++j) {
                                const Q ph = qphi(Q(R), mq + hq * Q(angular->x[j]), pf);
                                if (!(ph > Q(0)) || !finiteq(ph)) {
                                    out.fail = 4;
                                    return out;
                                }
                                fh += R * h * angular->w[j] *
                                      static_cast<double>(sqrtq(ph));
                            }
                        }
                    }
                }
                sum0 += static_cast<long double>(rw * f0);
                sumh += static_cast<long double>(rw * fh);
            }
        }
    }
    const double den = kPi * p.rho * p.rho;
    out.mu0 = static_cast<double>(sum0 / den);
    out.muhalf = static_cast<double>((0.5L * sum0 + 0.5L * sumh) /
                                     (den * (1.0 - 0.5 / 3.0)));
    if (!std::isfinite(out.mu0) || (need_ld && !std::isfinite(out.muhalf)))
        out.fail = 5;
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: cell_reference CASES_TSV CELLS_TSV OUTPUT_TSV\n";
        return 2;
    }
    std::map<std::pair<int, std::string>, std::vector<BaseCell>> cells;
    {
        std::ifstream in(argv[2]);
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            int row, cell, level;
            std::string name, u, parameter_name, kind;
            double crossings, left_fold, right_fold, a, b;
            double alo, blo, width, left_rp, right_rp;
            if (!(ss >> row >> name >> u >> parameter_name >> cell >> kind >> crossings >>
                  left_fold >> right_fold >> a >> b >> alo >> blo >> width >> left_rp >>
                  right_rp >> level)) continue;
            // Only one row per base geometry is required: u=0 rows 99/100
            // carry the same lens geometry used for both u=0 and u=.5 FD.
            if (level != 8) continue;
            if (name == "caustic-cross" || (name == "rand035" && u == "0"))
                cells[{row, parameter_name}].push_back({cell, a, b});
        }
    }
    for (auto& kv : cells) {
        auto& v = kv.second;
        std::sort(v.begin(), v.end(), [](const BaseCell& a, const BaseCell& b) {
            return a.index < b.index;
        });
        v.erase(std::unique(v.begin(), v.end(), [](const BaseCell& a, const BaseCell& b) {
            return a.index == b.index;
        }), v.end());
    }

    std::ofstream out(argv[3], std::ios::out | std::ios::trunc);
    out << std::setprecision(17)
        << "row\tcase\tparameter\tcell\ta\tb\th_scale\tnr\tsubdivisions\tangular\t"
           "fd_mu0\tfd_muhalf\tspread_mu0_by_offset\tspread_muhalf_by_offset\t"
           "fail_count\tfd_mu0_m2\tfd_mu0_m1\tfd_mu0_p1\tfd_mu0_p2\t"
           "fd_muhalf_m2\tfd_muhalf_m1\tfd_muhalf_p1\tfd_muhalf_p2\n";

    std::ifstream in(argv[1]);
    std::string line;
    int row = 0;
    while (std::getline(in, line)) {
        ++row;
        std::istringstream ss(line);
        LensParams base; int bary = 0; double u = 0.0, mu = 0.0;
        std::string name;
        if (!(ss >> base.xs >> base.ys >> base.rho >> base.q >> base.a >> bary >> u >> mu >> name))
            continue;
        base.barycentric = bary != 0;
        if (name != "caustic-cross" && name != "rand035") continue;
        if (name == "rand035" && u != 0.0) continue;
        for (const std::string parameter_name : name == "caustic-cross"
                 ? std::vector<std::string>{"a"} : std::vector<std::string>{"X", "Y"}) {
            const auto key = std::make_pair(row, parameter_name);
            const auto found = cells.find(key);
            if (found == cells.end()) continue;
            std::vector<BaseCell> selected;
            for (const auto& c : found->second) {
                const bool use = name == "caustic-cross"
                    ? (c.index == 5 || c.index == 6 || c.index == 7 || c.index == 12)
                    : (c.index == 2 || c.index == 3 || c.index == 8 || c.index == 11);
                if (use) selected.push_back(c);
            }
            const std::vector<double> hscales = name == "caustic-cross"
                ? std::vector<double>{3e-3, 1e-3, 3e-4, 1e-4}
                : std::vector<double>{1e-3, 5e-4, 2.5e-4};
            // Screening sequence.  If a cell remains resolution-sensitive,
            // rerun just that cell with the independent-reference resolutions.
            const std::vector<Resolution> resolutions = name == "caustic-cross"
                ? std::vector<Resolution>{{64,1,32},{128,2,32},{256,4,32}}
                : std::vector<Resolution>{{32,1,64},{48,2,128},{64,4,256}};
            const bool need_ld = name == "rand035";
            const bool do_muhalf = name == "rand035";
            for (double hscale : hscales) for (const auto& cfg : resolutions) {
                const double h = base.rho * hscale;
                std::map<int, std::array<CellValue, 4>> values;
                int fail_count = 0;
                for (int oi = 0; oi < 4; ++oi) {
                    const int offset = std::array<int,4>{{-2,-1,1,2}}[oi];
                    LensParams p = base;
                    parameter(p, parameter_name) += offset * h;
                    const TopologyResult topo = classify_cells(
                        PrimaryFrame::from(p), nullptr, nullptr, true);
                    if (topo.status != Status::OK) {
                        ++fail_count;
                        for (const auto& c : selected) values[c.index][oi].fail = 99;
                        continue;
                    }
                    for (const auto& c : selected) {
                        values[c.index][oi] = integrate_fixed_interval(
                            p, topo, cfg, c.a, c.b, need_ld);
                        if (values[c.index][oi].fail) ++fail_count;
                    }
                }
                for (const auto& c : selected) {
                    const auto& v = values[c.index];
                    if (std::any_of(v.begin(), v.end(), [](const CellValue& z){return z.fail!=0;})) {
                        out << row << '\t' << name << '\t' << parameter_name << '\t'
                            << c.index << '\t' << c.a << '\t' << c.b << '\t' << hscale
                            << '\t' << cfg.nr << '\t' << cfg.subdivisions << '\t' << cfg.angular
                            << "\tNaN\tNaN\tNaN\tNaN\t" << fail_count
                            << "\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\n";
                        continue;
                    }
                    const double den = 12.0 * h;
                    const double fd0 = (v[0].mu0 - 8.0*v[1].mu0 + 8.0*v[2].mu0 - v[3].mu0)/den;
                    const double fdh = (v[0].muhalf - 8.0*v[1].muhalf + 8.0*v[2].muhalf - v[3].muhalf)/den;
                    const double spread0 = std::max({v[0].mu0,v[1].mu0,v[2].mu0,v[3].mu0}) -
                                           std::min({v[0].mu0,v[1].mu0,v[2].mu0,v[3].mu0});
                    const double spreadh = std::max({v[0].muhalf,v[1].muhalf,v[2].muhalf,v[3].muhalf}) -
                                           std::min({v[0].muhalf,v[1].muhalf,v[2].muhalf,v[3].muhalf});
                    out << row << '\t' << name << '\t' << parameter_name << '\t'
                        << c.index << '\t' << c.a << '\t' << c.b << '\t' << hscale
                        << '\t' << cfg.nr << '\t' << cfg.subdivisions << '\t' << cfg.angular
                        << '\t' << fd0 << '\t' << (do_muhalf ? fdh : NAN)
                        << '\t' << spread0 << '\t' << (do_muhalf ? spreadh : NAN)
                        << '\t' << fail_count
                        << '\t' << v[0].mu0 << '\t' << v[1].mu0 << '\t' << v[2].mu0 << '\t' << v[3].mu0
                        << '\t' << (do_muhalf ? v[0].muhalf : NAN)
                        << '\t' << (do_muhalf ? v[1].muhalf : NAN)
                        << '\t' << (do_muhalf ? v[2].muhalf : NAN)
                        << '\t' << (do_muhalf ? v[3].muhalf : NAN) << '\n';
                }
                out.flush();
            }
        }
    }
}

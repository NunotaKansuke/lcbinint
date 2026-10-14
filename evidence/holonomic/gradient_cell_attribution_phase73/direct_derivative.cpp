// Fixed-R, cell-by-cell derivative reference for Phase73.
// Uses binary128 lens equations, quartic Sturm isolation, implicit angular
// endpoint derivatives (uniform flux), and a direct endpoint-weighted
// angular derivative integral (LD).  It does not call adaptive mapped_radius,
// root-pair IFT, vK/K-rule, or the adaptive estimator.
#define main phase72_reference_main
#include "../moving_map_gradient_phase72/rand035_reference/reference.cpp"
#undef main

#include <map>
#include <memory>
#include <cstdlib>

namespace {

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> out;
    std::size_t at = 0;
    for (;;) {
        const std::size_t end = line.find('\t', at);
        out.push_back(line.substr(at, end == std::string::npos ? end : end-at));
        if (end == std::string::npos) break;
        at = end + 1;
    }
    return out;
}

std::map<std::string, std::size_t> header_map(const std::string& line) {
    const auto names = split_tsv(line);
    std::map<std::string, std::size_t> out;
    for (std::size_t i = 0; i < names.size(); ++i) out[names[i]] = i;
    return out;
}

struct LensCase {
    int row = 0;
    std::string name, parameter;
    double u = 0, rmax = 0;
    LensParams p;
};
struct RawCell {
    int index = 0;
    std::string kind;
    bool left_fold = false, right_fold = false;
    double a = 0, b = 0;
};
struct DerivativePair {
    long double f0 = 0.0L, fh = 0.0L;
    bool ok = true;
    int rejected = 0;
    int accepted_nodes = 0, rejected_nodes = 0;
};

Q qphi_dp(Q R, Q theta, const PrimaryFrame& pf,
          const std::string& parameter, bool barycentric) {
    const Q ct = cosq(theta), st = sinq(theta);
    const QC z{R*ct, R*st}, zb{R*ct, -R*st};
    const Q m0 = Q(pf.m0), m1 = Q(1)-m0, a = Q(pf.a);
    const QC iz = qdiv({Q(1),Q(0)},zb);
    const QC delta = qsub(zb,{a,Q(0)});
    const QC iza = qdiv({Q(1),Q(0)},delta);
    const QC f = qsub(qsub(z,{m0*iz.re,m0*iz.im}),
                      {m1*iza.re,m1*iza.im});
    const QC g = qsub(f,{Q(pf.X),Q(pf.Y)});
    QC gp{Q(0),Q(0)};
    if (parameter == "X") gp = {Q(-1),Q(0)};
    else if (parameter == "Y") gp = {Q(0),Q(-1)};
    else if (parameter == "a") {
        const QC delta2 = qmul(delta,delta);
        gp = qdiv({-m1,Q(0)},delta2);
        if (barycentric) gp.re -= m1;
    } else {
        return std::numeric_limits<Q>::quiet_NaN();
    }
    const Q rho = Q(pf.rho);
    return -Q(2)*(g.re*gp.re+g.im*gp.im)/(rho*rho);
}

Q qphi_dp_over_endpoint(Q R, Q theta, const PrimaryFrame& pf,
                        const std::string& parameter, bool barycentric) {
    const Q phi_p = qphi_dp(R,theta,pf,parameter,barycentric);
    const Q phi_t = qphi_dtheta(R,theta,pf);
    if (!finiteq(phi_p) || !finiteq(phi_t) || phi_t == Q(0))
        return std::numeric_limits<Q>::quiet_NaN();
    return -phi_p/phi_t;
}

DerivativePair fixed_r_derivative(double R, const PrimaryFrame& pf,
                                  const std::string& parameter,
                                  bool barycentric, int angular_nodes) {
    DerivativePair out;
    RefValue local;
    const RefArcs arcs = certified_arcs(R,pf,local);
    if (local.fail || arcs.kind == ArcKind::kDegenerate) {
        out.ok = false; out.rejected = local.fail ? local.fail : 1; return out;
    }
    if (arcs.kind == ArcKind::kEmpty) return out;
    if (arcs.kind == ArcKind::kFull) {
        if (parameter == "X" || parameter == "Y" || parameter == "a") {
            const Q pi = acosq(Q(-1));
            Q sum = Q(0);
            for (int j = 0; j < angular_nodes; ++j) {
                const Q th = Q(2)*pi*(Q(j)+Q(0.5))/Q(angular_nodes);
                const Q ph = qphi(Q(R),th,pf);
                const Q php = qphi_dp(Q(R),th,pf,parameter,barycentric);
                if (!(ph > Q(0)) || !finiteq(php)) {
                    out.ok = false; out.rejected = 2; return out;
                }
                sum += php/(Q(2)*sqrtq(ph));
            }
            out.fh = static_cast<long double>(Q(R)*Q(2)*pi*sum/Q(angular_nodes));
        }
        return out;
    }
    if (arcs.kind != ArcKind::kArcs) {
        out.ok = false; out.rejected = 3; return out;
    }
    const Q pi = acosq(Q(-1));
    Q f0p = Q(0), fhp = Q(0);
    for (const auto& arc : arcs.arcs) {
        const Q lo = arc.lo, hi = arc.hi;
        const Q dlo = qphi_dp_over_endpoint(Q(R),lo,pf,parameter,barycentric);
        const Q dhi = qphi_dp_over_endpoint(Q(R),hi,pf,parameter,barycentric);
        if (!finiteq(dlo) || !finiteq(dhi)) {
            out.ok = false; out.rejected = 4; return out;
        }
        f0p += Q(R)*(dhi-dlo);

        // For a simple angular endpoint phi=(1-x^2) H(x), H is smooth.
        // Chebyshev-I integrates phi_p/(2 sqrt(phi)) without endpoint samples.
        const Q mid = (lo+hi)/Q(2), half = (hi-lo)/Q(2);
        for (int j = 0; j < angular_nodes; ++j) {
            const Q x = cosq(pi*(Q(2*j+1))/Q(2*angular_nodes));
            const Q omx2 = Q(1)-x*x;
            const Q th = mid+half*x;
            const Q ph = qphi(Q(R),th,pf);
            const Q php = qphi_dp(Q(R),th,pf,parameter,barycentric);
            if (!(omx2 > Q(0)) || !(ph > Q(0)) || !finiteq(php)) {
                out.ok = false; out.rejected = 5; return out;
            }
            const Q H = ph/omx2;
            if (!(H > Q(0)) || !finiteq(H)) {
                out.ok = false; out.rejected = 6; return out;
            }
            fhp += Q(R)*half*(pi/Q(angular_nodes))*php/(Q(2)*sqrtq(H));
        }
    }
    out.f0 = static_cast<long double>(f0p);
    out.fh = static_cast<long double>(fhp);
    return out;
}

std::vector<LensCase> read_cases(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::vector<LensCase> out;
    int row = 0;
    while (std::getline(in,line)) {
        ++row;
        std::istringstream ss(line);
        LensCase c; int bary = 0; double mu = 0;
        if (!(ss >> c.p.xs >> c.p.ys >> c.p.rho >> c.p.q >> c.p.a >> bary >>
              c.u >> mu >> c.name)) continue;
        c.p.barycentric = bary != 0;
        c.row = row;
        if (c.name == "caustic-cross" && c.u == 0.0) {
            c.parameter = "a"; out.push_back(c);
        }
        if (c.name == "rand035" && (row == 99 || row == 100)) {
            c.parameter = "X"; out.push_back(c);
            c.parameter = "Y"; out.push_back(c);
        }
    }
    return out;
}

std::vector<RawCell> read_cells(const std::string& path, int row,
                                const std::string& case_name,
                                const std::string& parameter) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in,line)) return {};
    const auto h = header_map(line);
    auto get = [&](const std::vector<std::string>& f,const std::string& key)->std::string {
        return h.at(key) < f.size() ? f[h.at(key)] : std::string{};
    };
    std::vector<RawCell> out;
    while (std::getline(in,line)) {
        const auto f = split_tsv(line);
        if (std::stoi(get(f,"row")) != row || get(f,"case") != case_name ||
            get(f,"parameter") != parameter || std::stoi(get(f,"level")) != 8)
            continue;
        RawCell c;
        c.index = std::stoi(get(f,"cell")); c.kind = get(f,"kind");
        c.left_fold = std::stoi(get(f,"left_fold")) != 0;
        c.right_fold = std::stoi(get(f,"right_fold")) != 0;
        c.a = std::stod(get(f,"a")); c.b = std::stod(get(f,"b"));
        out.push_back(c);
    }
    std::sort(out.begin(),out.end(),[](const RawCell& a,const RawCell& b){return a.index<b.index;});
    return out;
}

} // namespace

int main(int argc,char** argv) {
    if (argc != 4) {
        std::cerr << "usage: direct_derivative CASES_TSV CELLS_TSV OUTPUT_TSV\n";
        return 2;
    }
    std::ofstream out(argv[3],std::ios::out|std::ios::trunc);
    out << std::setprecision(17)
        << "row\tcase\tu\tparameter\tcell\tkind\tleft_fold_source\tright_fold_source\t"
           "left_fold_used\tright_fold_used\ta\tb\tnr\tsubdivisions\tangular\t"
           "mu0_grad\tmuhalf_grad\taccepted_nodes\trejected_nodes\tcell_ok\t"
           "mu0_partial_sum\tmuhalf_partial_sum\n";
    const char* only_name_env = std::getenv("PHASE73_ONLY_CASE");
    const std::string only_name = only_name_env ? only_name_env : "";
    const bool extend_caustic = std::getenv("PHASE73_CAUSTIC_EXTENDED") != nullptr;
    const bool p4_fold_probe = std::getenv("PHASE73_P4_FOLD_PROBE") != nullptr;
    const bool extend_rand = std::getenv("PHASE73_RAND_EXTENDED") != nullptr;
    const char* only_cell_env = std::getenv("PHASE73_ONLY_CELL");
    const int only_cell = only_cell_env ? std::atoi(only_cell_env) : -1;
    for (const LensCase& c : read_cases(argv[1])) {
        if (!only_name.empty() && c.name != only_name) continue;
        const auto raw_cells = read_cells(argv[2],c.row,c.name,c.parameter);
        if (raw_cells.empty()) continue;
        const PrimaryFrame pf = PrimaryFrame::from(c.p);
        std::vector<Resolution> radial = c.name == "caustic-cross"
            ? std::vector<Resolution>{{128,1,128},{256,1,128},{512,1,128},{1024,1,128}}
            : std::vector<Resolution>{{48,1,128},{80,4,256},{112,8,512},{112,8,1024}};
        if (extend_caustic && c.name == "caustic-cross") {
            radial.push_back({2048,1,128});
            radial.push_back({4096,1,128});
        }
        if (extend_rand && c.name == "rand035") {
            radial.push_back({112,8,1024});
            radial.push_back({128,16,1024});
            radial.push_back({160,32,1024});
        }
        for (const auto& cfg : radial) {
            Rule gl(cfg.nr);
            std::vector<std::pair<int,DerivativePair>> cell_values;
            long double total0=0.0L,totalhalf=0.0L;
            for (const auto& cell : raw_cells) {
                if (only_cell >= 0 && cell.index != only_cell) continue;
                DerivativePair cell0,cellh;
                if (cell.kind == "empty") {
                    cell_values.push_back({cell.index,{}});
                    continue;
                }
                const bool left_fold = cell.left_fold ||
                    (p4_fold_probe && c.name=="caustic-cross" &&
                     (cell.index==5 || cell.index==7));
                const FoldRadialMap map{cell.a,cell.b,left_fold,cell.right_fold};
                long double s0=0.0L,sh=0.0L;
                bool ok=true; int cell_accepted=0,cell_rejected=0;
                for (int sub=0;sub<cfg.subdivisions;++sub) {
                    const double xl=-1.0+2.0*sub/cfg.subdivisions;
                    const double xr=-1.0+2.0*(sub+1)/cfg.subdivisions;
                    const double xc=0.5*(xl+xr),xh=0.5*(xr-xl);
                    for (int k=0;k<cfg.nr;++k) {
                        const auto rr=map(xc+xh*gl.x[k]);
                        const DerivativePair point=fixed_r_derivative(
                            rr[0],pf,c.parameter,c.p.barycentric,cfg.angular);
                        if (!point.ok) { ok=false; ++cell_rejected; continue; }
                        ++cell_accepted;
                        const long double w=static_cast<long double>(xh*gl.w[k]*rr[1]);
                        s0 += w*point.f0;
                        sh += w*point.fh;
                    }
                }
                cell0.accepted_nodes=cell_accepted;
                cell0.rejected_nodes=cell_rejected;
                if (!ok) {
                    cell0.ok=false; cell0.rejected=cell_rejected;
                    cell_values.push_back({cell.index,cell0});
                    continue;
                }
                const long double den=static_cast<long double>(kPi*c.p.rho*c.p.rho);
                DerivativePair normalized;
                normalized.f0=s0/den;
                normalized.fh=(0.5L*s0+0.5L*sh)/(den*(1.0L-0.5L/3.0L));
                normalized.accepted_nodes=cell_accepted;
                normalized.rejected_nodes=cell_rejected;
                cell_values.push_back({cell.index,normalized});
                total0 += normalized.f0; totalhalf += normalized.fh;
            }
            for (const auto& cv : cell_values) {
                const auto it=std::find_if(raw_cells.begin(),raw_cells.end(),
                    [&](const RawCell& rc){return rc.index==cv.first;});
                if (it==raw_cells.end()) continue;
                const bool left_fold_used=it->left_fold ||
                    (p4_fold_probe && c.name=="caustic-cross" &&
                     (cv.first==5 || cv.first==7));
                out << c.row << '\t' << c.name << '\t' << c.u << '\t' << c.parameter << '\t'
                    << cv.first << '\t' << it->kind << '\t' << it->left_fold << '\t'
                    << it->right_fold << '\t' << left_fold_used << '\t' << it->right_fold
                    << '\t' << it->a << '\t' << it->b << '\t'
                    << cfg.nr << '\t' << cfg.subdivisions << '\t' << cfg.angular << '\t'
                    << static_cast<double>(cv.second.f0) << '\t'
                    << static_cast<double>(cv.second.fh) << '\t' << cv.second.accepted_nodes << '\t'
                    << cv.second.rejected_nodes << '\t' << cv.second.ok << '\t'
                    << static_cast<double>(total0) << '\t'
                    << static_cast<double>(totalhalf) << '\n';
            }
            out.flush();
        }
    }
}

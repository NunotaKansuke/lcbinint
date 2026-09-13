#include "../../../../tests/holonomic_cpp/adaptive_reference.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tuple>

using namespace lcbinint::holonomic;

namespace {

struct Rule {
    std::vector<double> x, w;
    explicit Rule(int n) : x(n), w(n) {
        for (int i = 0; i < (n + 1) / 2; ++i) {
            double z = std::cos(kPi * (i + .75) / (n + .5)), dp = 0.0;
            for (int it = 0; it < 32; ++it) {
                double p = 1.0, q = 0.0;
                for (int k = 1; k <= n; ++k) {
                    const double r = ((2.0 * k - 1.0) * z * p - (k - 1.0) * q) / k;
                    q = p; p = r;
                }
                dp = n * (z * p - q) / (z * z - 1.0);
                const double step = p / dp;
                z -= step;
                if (std::fabs(step) <= 2.0 * std::numeric_limits<double>::epsilon()) break;
            }
            double p = 1.0, q = 0.0;
            for (int k = 1; k <= n; ++k) {
                const double r = ((2.0 * k - 1.0) * z * p - (k - 1.0) * q) / k;
                q = p; p = r;
            }
            dp = n * (z * p - q) / (z * z - 1.0);
            const double wt = 2.0 / ((1.0 - z * z) * dp * dp);
            x[i] = -z; x[n - 1 - i] = z;
            w[i] = wt; w[n - 1 - i] = wt;
        }
    }
};

struct Resolution { int nr, subdivisions, angular; };
using Q = __float128;
struct QArc { Q lo, hi; };
struct RefArcs { ArcKind kind = ArcKind::kDegenerate; std::vector<QArc> arcs; int roots = -1; int fail = 0; };
struct RefValue {
    double mu0 = NAN, mu_half = NAN, min_phi = INFINITY;
    int fail = 0, segment = -1, node = -1, kind = -1, sturm_roots = -1;
    double fail_R = NAN;
    int radial_nodes = 0, arc_nodes = 0, full_nodes = 0, empty_nodes = 0;
};

struct QC { Q re, im; };
QC qadd(QC a, QC b) { return {a.re+b.re,a.im+b.im}; }
QC qsub(QC a, QC b) { return {a.re-b.re,a.im-b.im}; }
QC qmul(QC a, QC b) { return {a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re}; }
QC qdiv(QC a, QC b) { const Q d=b.re*b.re+b.im*b.im; return {(a.re*b.re+a.im*b.im)/d,(a.im*b.re-a.re*b.im)/d}; }
QC qi(QC a) { return {-a.im,a.re}; }

Q qphi(Q R, Q theta, const PrimaryFrame& pf) {
    const Q ct = cosq(theta), st = sinq(theta);
    const QC z{R*ct,R*st}, zb{R*ct,-R*st};
    const Q a=Q(pf.a), m0=Q(pf.m0), m1=Q(1)-m0;
    const QC iz=qdiv({Q(1),Q(0)},zb), iza=qdiv({Q(1),Q(0)},qsub(zb,{a,Q(0)}));
    const QC f=qsub(qsub(z,{m0*iz.re,m0*iz.im}),{m1*iza.re,m1*iza.im});
    const QC g=qsub(f,{Q(pf.X),Q(pf.Y)});
    const Q rho=Q(pf.rho);
    return Q(1)-(g.re*g.re+g.im*g.im)/(rho*rho);
}

Q qphi_dtheta(Q R, Q theta, const PrimaryFrame& pf) {
    const Q ct = cosq(theta), st = sinq(theta);
    const QC z{R*ct,R*st}, zb{R*ct,-R*st};
    const Q a=Q(pf.a), m0=Q(pf.m0), m1=Q(1)-m0;
    const QC iz=qdiv({Q(1),Q(0)},zb), iza=qdiv({Q(1),Q(0)},qsub(zb,{a,Q(0)}));
    const QC iz2=qmul(iz,iz), iza2=qmul(iza,iza);
    const QC f=qsub(qsub(z,{m0*iz.re,m0*iz.im}),{m1*iza.re,m1*iza.im});
    const QC g=qsub(f,{Q(pf.X),Q(pf.Y)});
    const QC fzb{m0*iz2.re+m1*iza2.re,m0*iz2.im+m1*iza2.im};
    const QC dz=qsub(qi(z),qi(qmul(zb,fzb)));
    const Q rho=Q(pf.rho);
    return -Q(2)*(g.re*dz.re+g.im*dz.im)/(rho*rho);
}

Q normalize_angle(Q x) {
    const Q tp=Q(2)*acosq(Q(-1));
    while (x < Q(0)) x += tp;
    while (x >= tp) x -= tp;
    return x;
}

std::array<Q,5> qf_boundary_coeffs(Q R, const PrimaryFrame& pf) {
    const Q a=Q(pf.a), m0=Q(pf.m0), X=Q(pf.X), Y=Q(pf.Y), rho=Q(pf.rho);
    const Q R2=R*R;
    const QC n0{-X*R2,-Y*R2};
    const QC n1{R*(R2-Q(1)+a*X),R*a*Y};
    const QC n2{a*(m0-R2),Q(0)};
    const QC c0=qadd(qadd(n0,n1),n2);
    const QC c1{-Q(2)*(n2.im-n0.im),Q(2)*(n2.re-n0.re)};
    const QC c2=qsub(qsub(n1,n0),n2);
    const Q rho2=rho*rho, bm=(R-a)*(R-a), bp=(R+a)*(R+a), k=rho2*R2;
    const Q lin0=k*bm, lin2=k*(bm+bp), lin4=k*bp;
    const auto norm=[](QC x){return x.re*x.re+x.im*x.im;};
    const auto dot=[](QC x,QC y){return x.re*y.re+x.im*y.im;};
    const Q a0=norm(c0), a1=Q(2)*dot(c0,c1);
    const Q a2=norm(c1)+Q(2)*dot(c0,c2), a3=Q(2)*dot(c1,c2), a4=norm(c2);
    return {lin0-a0,-a1,lin2-a2,-a3,lin4-a4};
}

std::vector<Q> isolate_qf_roots(const std::array<Q,5>& coeffs, int expected) {
    using namespace quartic_sturm_detail;
    Q scale=Q(0);
    for(Q c:coeffs)if(fabsq(c)>scale)scale=fabsq(c);
    if(!(scale>Q(0)))return {};
    std::array<Q,5> normalized_coeffs{};
    for(int i=0;i<5;++i)normalized_coeffs[i]=coeffs[i]/scale;
    const auto counted=count_chain(normalized_coeffs);
    if (!stable(counted,2) || counted.count != expected) return {};
    Q max_ratio=Q(0), lead=abs_value(Q(coeffs[4]));
    if (!(lead>Q(0))) return {};
    for (int k=0;k<4;++k) {
        const Q ratio=abs_value(Q(coeffs[k]))/lead;
        if (ratio>max_ratio) max_ratio=ratio;
    }
    const Q bound=Q(2)*(Q(1)+max_ratio)+Q(1), lo=-bound, hi=bound;
    if (variation_at(counted.chain,lo)-variation_at(counted.chain,hi)!=expected) return {};
    std::vector<Interval<Q>> pending{{lo,hi,expected,0}};
    std::vector<Q> roots; roots.reserve(expected);
    while (!pending.empty()) {
        const auto cur=pending.back(); pending.pop_back();
        if (cur.count<=0) continue;
        if (cur.count==1) {
            Q a=cur.lo,b=cur.hi;
            for (int it=0;it<240;++it) {
                const Q m=(a+b)/Q(2);
                if (m==a || m==b) break;
                const int left=variation_at(counted.chain,a)-variation_at(counted.chain,m);
                if (left>0) b=m; else a=m;
            }
            roots.push_back((a+b)/Q(2));
            continue;
        }
        if (cur.depth>=160) return {};
        const Q m=(cur.lo+cur.hi)/Q(2);
        const int left=variation_at(counted.chain,cur.lo)-variation_at(counted.chain,m);
        const int right=cur.count-left;
        if (left<0 || right<0 || left+right!=cur.count) return {};
        pending.push_back({m,cur.hi,right,cur.depth+1});
        pending.push_back({cur.lo,m,left,cur.depth+1});
    }
    if (static_cast<int>(roots.size())!=expected) return {};
    std::sort(roots.begin(),roots.end());
    return roots;
}

// Independent angular-boundary reconstruction: isolate all real roots in
// binary128 Sturm arithmetic and refine each against the original lens
// equation. This deliberately bypasses the production complex-root filter
// and its double endpoint polish; radial/angular quadrature is direct.
RefArcs certified_arcs(double R, const PrimaryFrame& pf, RefValue& out) {
    const auto direct=qf_boundary_coeffs(Q(R),pf);
    auto cauchy=[](const std::array<Q,5>& c) {
        if(c[4]==Q(0))return HUGE_VALQ;
        Q ratio=Q(0);
        for(int i=0;i<4;++i){const Q x=fabsq(c[i]/c[4]);if(x>ratio)ratio=x;}
        return Q(2)*(Q(1)+ratio)+Q(1);
    };
    const std::array<Q,5> reciprocal{{direct[4],-direct[3],direct[2],-direct[1],direct[0]}};
    const bool use_recip=cauchy(reciprocal)<cauchy(direct);
    const auto chart=use_recip?reciprocal:direct;
    using namespace quartic_sturm_detail;
    Q scale=Q(0);for(Q c:chart)if(fabsq(c)>scale)scale=fabsq(c);
    if(!(scale>Q(0))){out.fail=1;return {};}
    std::array<Q,5> normalized_chart{};for(int i=0;i<5;++i)normalized_chart[i]=chart[i]/scale;
    const auto counted=count_chain(normalized_chart);
    if(counted.degree!=4||!valid_count(counted.degree,counted.count)||!counted.square_free){out.fail=1;out.sturm_roots=counted.count;return {};}
    const int root_count=counted.count;
    out.sturm_roots=root_count;
    if (root_count == 0) {
        const Q ph = qphi(Q(R), Q(0), pf);
        if (!finiteq(ph) || ph == Q(0)) { out.fail = 2; return {}; }
        RefArcs result; result.roots=0; result.kind=ph>Q(0)?ArcKind::kFull:ArcKind::kEmpty; return result;
    }
    const auto roots = isolate_qf_roots(chart, root_count);
    if (static_cast<int>(roots.size()) != root_count) { out.fail = 3; return {}; }
    const Q pi=acosq(Q(-1)), tp=Q(2)*pi;
    std::vector<Q> theta;
    theta.reserve(roots.size());
    for (Q r : roots) theta.push_back(normalize_angle((use_recip?pi:Q(0))+Q(2)*atanq(r)));
    std::sort(theta.begin(), theta.end());
    // Correct coefficient-rounding drift with safeguarded Newton on phi=0.
    for (std::size_t i=0;i<theta.size();++i) {
        Q gap=tp;
        const Q prev=theta[(i+theta.size()-1)%theta.size()];
        const Q next=theta[(i+1)%theta.size()];
        Q gp=theta[i]-prev; if(gp<=Q(0))gp+=tp;
        Q gn=next-theta[i]; if(gn<=Q(0))gn+=tp;
        if(gp<gap)gap=gp; if(gn<gap)gap=gn;
        Q th=theta[i];
        for(int it=0;it<24;++it) {
            const Q ph=qphi(Q(R),th,pf), dph=qphi_dtheta(Q(R),th,pf);
            if(!finiteq(ph)||!finiteq(dph)||dph==Q(0))break;
            if(fabsq(ph)<ldexpq(Q(1),-95))break;
            Q step=ph/dph, cap=gap/Q(4);
            if(fabsq(step)>cap)step=(step>Q(0)?cap:-cap);
            Q alpha=Q(1), old=fabsq(ph), trial=th;
            bool accepted=false;
            for(int ls=0;ls<20;++ls) {
                trial=normalize_angle(th-alpha*step);
                if(fabsq(qphi(Q(R),trial,pf))<old){accepted=true;break;}
                alpha/=Q(2);
            }
            if(!accepted)break;
            th=trial;
        }
        theta[i]=th;
    }
    std::sort(theta.begin(),theta.end());
    RefArcs result; result.kind=ArcKind::kArcs; result.roots=root_count;
    for(std::size_t i=0;i<theta.size();++i) {
        const Q lo=theta[i]; Q hi=theta[(i+1)%theta.size()]; if(hi<=lo)hi+=tp;
        if(qphi(Q(R),(lo+hi)/Q(2),pf)>Q(0))result.arcs.push_back({lo,hi});
    }
    if(result.arcs.empty()){result.fail=4;out.fail=4;}
    return result;
}

bool is_physical_fold(const TopologyResult& topo, double R) {
    for (const auto& e : topo.events)
        if (e.radius == R && e.physically_real && e.kind == "physical_real") return true;
    return false;
}

RefValue integrate_direct(const LensParams& p, const TopologyResult& topo,
                          const Resolution& cfg) {
    RefValue out;
    if (topo.status != Status::OK) { out.fail = 10; return out; }
    const PrimaryFrame pf = PrimaryFrame::from(p);
    const Rule radial(cfg.nr);
    const Cheb1Dyn angular(cfg.angular);
    std::vector<double> cuts{0.0, topo.r_max};
    for (const auto& c : topo.cells) { cuts.push_back(c.r_lo); cuts.push_back(c.r_hi); }
    for (const auto& e : topo.events)
        if (e.physically_real && e.kind == "physical_real" && e.radius > 0.0 && e.radius < topo.r_max)
            cuts.push_back(e.radius);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    long double sum0 = 0.0L, sumh = 0.0L;
    for (std::size_t ci = 1; ci < cuts.size(); ++ci) {
        const double a = cuts[ci - 1], b = cuts[ci];
        if (!(b > a)) continue;
        const auto mid = quartic_topology(a + 0.5 * (b - a), pf);
        if (!mid.certified) { out.fail = 11; out.segment = static_cast<int>(ci - 1); return out; }
        if (mid.kind == ArcKind::kEmpty) continue;
        FoldRadialMap map{a, b, is_physical_fold(topo, a), is_physical_fold(topo, b)};
        for (int sub = 0; sub < cfg.subdivisions; ++sub) {
            const double xl = -1.0 + 2.0 * sub / cfg.subdivisions;
            const double xr = -1.0 + 2.0 * (sub + 1) / cfg.subdivisions;
            const double xc = 0.5 * (xl + xr), xh = 0.5 * (xr - xl);
            for (int k = 0; k < cfg.nr; ++k) {
                const double x = xc + xh * radial.x[k];
                const auto rr = map(x);
                const double R = rr[0];
                const double rw = xh * radial.w[k] * rr[1];
                ++out.radial_nodes;
                RefValue local;
                const RefArcs arcs = certified_arcs(R, pf, local);
                if (local.fail || arcs.kind == ArcKind::kDegenerate) {
                    out.fail = local.fail ? local.fail : 5;
                    out.segment = static_cast<int>(ci - 1); out.node = out.radial_nodes - 1;
                    out.kind = static_cast<int>(arcs.kind); out.sturm_roots = local.sturm_roots; out.fail_R = R;
                    return out;
                }
                double f0 = 0.0, fh = 0.0;
                if (arcs.kind == ArcKind::kEmpty) {
                    ++out.empty_nodes;
                } else if (arcs.kind == ArcKind::kFull) {
                    ++out.full_nodes;
                    f0 = kTwoPi * R;
                    // Periodic trapezoid on a smooth full-circle integrand.
                    for (int j = 0; j < cfg.angular; ++j) {
                        const double th = kTwoPi * (j + 0.5) / cfg.angular;
                        const double ph = static_cast<double>(qphi(Q(R),Q(th),pf));
                        out.min_phi = std::min(out.min_phi, ph);
                        if (!(ph > 0.0) || !std::isfinite(ph)) {
                            out.fail = 6; out.segment = static_cast<int>(ci - 1); out.node = out.radial_nodes - 1;
                            out.kind = static_cast<int>(arcs.kind); out.fail_R = R; return out;
                        }
                        fh += kTwoPi * R * std::sqrt(ph) / cfg.angular;
                    }
                } else {
                    ++out.arc_nodes;
                    for (const auto& arc : arcs.arcs) {
                        const Q hq = (arc.hi-arc.lo)/Q(2), mq=(arc.lo+arc.hi)/Q(2);
                        const double h = static_cast<double>(hq);
                        f0 += R * static_cast<double>(arc.hi-arc.lo);
                        for (int j = 0; j < cfg.angular; ++j) {
                            const Q phq=qphi(Q(R),mq+hq*Q(angular.x[j]),pf);
                            const double ph=static_cast<double>(phq);
                            out.min_phi = std::min(out.min_phi, ph);
                            if (!(phq > Q(0)) || !finiteq(phq)) {
                                out.fail = 8; out.segment = static_cast<int>(ci - 1); out.node = out.radial_nodes - 1;
                                out.kind = static_cast<int>(arcs.kind); out.fail_R = R; return out;
                            }
                            fh += R * h * angular.w[j] * static_cast<double>(sqrtq(phq));
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
    out.mu_half = static_cast<double>((0.5L * sum0 + 0.5L * sumh) / (den * (1.0 - 0.5 / 3.0)));
    if (!std::isfinite(out.mu0) || !std::isfinite(out.mu_half)) out.fail = 9;
    return out;
}

double& coord(LensParams& p, int axis) { return axis == 0 ? p.xs : p.ys; }

struct PointRecord { TopologyResult topo; LensParams p; int topology_status = -1; };
using Key = std::tuple<int, int, int>; // axis, offset in {-2,-1,+1,+2}, h index

void emit_reference(int row, const std::string& name, const LensParams& base,
                    bool x_high_resolution_only) {
    // Coarse-to-fine grids independently raise radial subdivision, GL order,
    // and direct angular resolution. H is scaled by source radius.
    const std::array<Resolution, 7> resolutions{{
        {48, 1, 96},    // initial independent reference
        {48, 1, 384},   // angular-only increase
        {80, 1, 384},   // radial order increase
        {80, 4, 384},   // radial subdivision increase
        {112, 8, 768},  // combined high-resolution reference
        {128, 16, 1024}, // final convergence check for the two smaller FD steps
        {160, 32, 1024}  // focused X-gradient radial convergence check
    }};
    const std::array<double, 3> hscales{{1.0e-3, 5.0e-4, 2.5e-4}};
    const std::array<int, 4> offsets{{-2, -1, +1, +2}};
    std::cout << std::setprecision(17);
    std::map<std::tuple<int,int,int,int>, std::array<double,2>> vals; // axis,h,offset,res -> mu0,muhalf
    std::map<std::tuple<int,int,int>, int> topo_status;
    for (int axis = 0; axis < 2; ++axis) {
        if (x_high_resolution_only && axis != 0) continue;
        for (int ih = 0; ih < static_cast<int>(hscales.size()); ++ih) {
            if (x_high_resolution_only && ih == 0) continue;
            const double h = base.rho * hscales[ih];
            for (int off : offsets) {
                LensParams p = base;
                coord(p, axis) += off * h;
                auto roots = std::vector<Cplx<__float128>>{};
                TopologyResult topo = classify_cells(PrimaryFrame::from(p), nullptr, &roots, true);
                topo_status[{axis, ih, off}] = static_cast<int>(topo.status);
                std::cout << "TOPO\t" << row << '\t' << name << '\t' << (axis == 0 ? "X" : "Y")
                          << '\t' << ih << '\t' << off << '\t' << h << '\t' << static_cast<int>(topo.status)
                          << '\t' << topo.cells.size() << '\t' << topo.events.size() << '\t' << topo.r_max << '\n';
                for (int ir = 0; ir < static_cast<int>(resolutions.size()); ++ir) {
                    if (x_high_resolution_only && ir != 6) continue;
                    if (!x_high_resolution_only && ir == 6) continue;
                    if (ir == 5 && ih == 0) continue; // large step already converged at resolution 4
                    const auto r = integrate_direct(p, topo, resolutions[ir]);
                    vals[{axis, ih, off, ir}] = {r.mu0, r.mu_half};
                    std::cout << "REF\t" << row << '\t' << name << '\t' << (axis == 0 ? "X" : "Y")
                              << '\t' << ih << '\t' << off << '\t' << ir << '\t' << resolutions[ir].nr
                              << '\t' << resolutions[ir].subdivisions << '\t' << resolutions[ir].angular
                              << '\t' << r.mu0 << '\t' << r.mu_half << '\t' << r.fail << '\t' << r.segment
                              << '\t' << r.node << '\t' << r.kind << '\t' << r.sturm_roots << '\t' << r.fail_R
                              << '\t' << r.radial_nodes << '\t' << r.arc_nodes << '\t' << r.full_nodes
                              << '\t' << r.empty_nodes << '\t' << r.min_phi << '\n';
                }
            }
        }
    }
    // 5-point central derivative; compare resolution sequences and h-sequence.
    for (int axis = 0; axis < 2; ++axis) for (int ih = 0; ih < static_cast<int>(hscales.size()); ++ih) {
        if (x_high_resolution_only && (axis != 0 || ih == 0)) continue;
        for (int ir = 0; ir < static_cast<int>(resolutions.size()); ++ir) {
            if (x_high_resolution_only && ir != 6) continue;
            if (!x_high_resolution_only && ir == 6) continue;
            if (ir == 5 && ih == 0) continue;
            std::array<double,2> fd{};
            for (int quantity = 0; quantity < 2; ++quantity) {
                const double fm2 = vals[{axis,ih,-2,ir}][quantity];
                const double fm1 = vals[{axis,ih,-1,ir}][quantity];
                const double fp1 = vals[{axis,ih,+1,ir}][quantity];
                const double fp2 = vals[{axis,ih,+2,ir}][quantity];
                const double h = base.rho * hscales[ih];
                fd[quantity] = (fm2 - 8.0*fm1 + 8.0*fp1 - fp2) / (12.0*h);
            }
            std::cout << "FD5\t" << row << '\t' << name << '\t' << (axis == 0 ? "X" : "Y")
                      << '\t' << ih << '\t' << hscales[ih] << '\t' << ir << '\t' << fd[0] << '\t' << fd[1] << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return 2;
    const std::string mode = argc == 3 ? argv[2] : "all";
    if (mode != "all" && mode != "x-high") return 2;
    const bool x_high_resolution_only = mode == "x-high";
    std::ifstream in(argv[1]);
    if (!in) return 2;
    std::cout << "type\trow\tname\taxis\th_index\toffset\tresolution\tnr\tsubdivisions\tangular\tmu0\tmu_half\tfail\tsegment\tnode\tkind\tsturm_roots\tfail_R\tradial_nodes\tarc_nodes\tfull_nodes\tempty_nodes\tmin_phi\n";
    std::string line;
    int row = 0;
    while (std::getline(in, line)) {
        ++row;
        std::istringstream ss(line);
        LensParams p; int bary; double u, dummy; std::string name;
        if (!(ss >> p.xs >> p.ys >> p.rho >> p.q >> p.a >> bary >> u >> dummy >> name)) continue;
        // Rows 99 (u=0) and 100 (u=.5) have exactly the same LensParams.
        // Each reference integration returns both u=0 and u=.5 observables.
        if (name != "rand035" || row != 99) continue;
        p.barycentric = bary;
        emit_reference(row, name, p, x_high_resolution_only);
    }
}

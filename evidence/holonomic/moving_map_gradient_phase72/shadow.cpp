#include "../../../tests/holonomic_cpp/adaptive_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace lcbinint::holonomic;

namespace {
constexpr double kPiShadow = 3.141592653589793238462643383279502884;
constexpr double kTwoPiShadow = 2.0 * kPiShadow;

struct Direction {
    std::array<double, 5> internal{}; // (X,Y,rho,m0,a)
};

Direction user_direction(const LensParams& p, int j) {
    Direction d;
    if (j == 0) d.internal[0] = 1.0;
    else if (j == 1) d.internal[1] = 1.0;
    else if (j == 4) {
        d.internal[4] = 1.0;
        if (p.barycentric) d.internal[0] = p.m1();
    }
    return d;
}

double& user_parameter(LensParams& p,int j) {
    switch(j) {
    case 0:return p.xs;
    case 1:return p.ys;
    case 2:return p.rho;
    case 3:return p.q;
    default:return p.a;
    }
}

double horner(const std::array<double, 5>& c, double t) {
    return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])));
}
double horner_d1(const std::array<double, 5>& c, double t) {
    return c[1] + t * (2.0 * c[2] + t * (3.0 * c[3] + t * 4.0 * c[4]));
}

double phi_dR(double R, double theta, const PrimaryFrame& pf) {
    using C = std::complex<double>;
    const C ei = std::polar(1.0, theta);
    const C z = R * ei, zb = std::conj(z), zba = zb - pf.a;
    const C g = z - pf.m0 / zb - (1.0 - pf.m0) / zba - C(pf.X, pf.Y);
    const C gR = ei + pf.m0 * std::polar(1.0, theta) / (R * R) +
                 (1.0 - pf.m0) * std::conj(ei) / (zba * zba);
    return -2.0 * std::real(std::conj(g) * gR) / (pf.rho * pf.rho);
}

struct EventSensitivity {
    double R = 0.0, Rlo = 0.0, t = 0.0, Rp = 0.0, tp = 0.0;
    double P = 0.0, Pt = 0.0, PR = 0.0, Ptt = 0.0;
    double Pp = 0.0, Ptr = 0.0, Ptp = 0.0;
    double uncertainty = 0.0;
    int tier = -1;
    bool valid = false;
    bool ordinary = false;
};

EventSensitivity event_sensitivity(const RadialEvent& e, const PrimaryFrame& pf,
                                   const Direction& dir,
                                   double value_budget) {
    EventSensitivity out;
    if (!e.physically_real || e.kind != "physical_real" ||
        !e.fold_t_seed_valid || !std::isfinite(e.fold_t_seed)) return out;

    DD rr = DD(e.radius) + DD(e.radius_lo);
    DD tt(e.fold_t_seed);
    int iterations = 0;
    double residual = std::numeric_limits<double>::infinity();
    double last_step = std::numeric_limits<double>::infinity();
    const bool solved = adaptive_detail::coupled_fold_newton(
        rr, tt, pf, 12, 1e-24, 1e-24, &iterations, &residual, &last_step);
    (void)iterations;
    if (!solved) return out;

    out.R = static_cast<double>(rr);
    out.Rlo = static_cast<double>(rr-DD(out.R));
    out.t = static_cast<double>(tt);
    out.uncertainty = e.radius_uncertainty;
    out.tier = e.precision_tier;
    const auto g = local_fold_quantities<DD>(rr, tt, pf);
    out.P = static_cast<double>(g.P);
    out.Pt = static_cast<double>(g.Pt);
    out.PR = static_cast<double>(g.PR);
    out.Ptt = static_cast<double>(g.Ptt);
    out.Ptr = static_cast<double>(g.Ptr);

    const auto dpc = boundary_quartic_dp(out.R, pf);
    std::array<double, 5> dcoeff{};
    for (int j = 0; j < 5; ++j)
        for (int k = 0; k < 5; ++k)
            dcoeff[k] += dir.internal[j] * dpc.dp[j][k];
    out.Pp = horner(dcoeff, out.t);
    out.Ptp = horner_d1(dcoeff, out.t);
    out.Rp = -out.Pp / out.PR;
    out.tp = -(out.Ptr * out.Rp + out.Ptp) / out.Ptt;

    const auto pc = boundary_quartic(out.R, pf).p;
    const auto pcR = boundary_quartic_dR(out.R, pf).p;
    const auto pd = p_derivs(pc, out.t);
    const double absPp = [&] {
        double v = std::fabs(dcoeff[4]), a = std::fabs(out.t);
        for (int k = 3; k >= 0; --k) v = v * a + std::fabs(dcoeff[k]);
        return v;
    }();
    const double absPt = [&] {
        double v = 4.0 * std::fabs(dcoeff[4]), a = std::fabs(out.t);
        v = v * a + 3.0 * std::fabs(dcoeff[3]);
        v = v * a + 2.0 * std::fabs(dcoeff[2]);
        v = v * a + std::fabs(dcoeff[1]);
        return v;
    }();
    double absPR = std::fabs(pcR[4]), absPtt = 12.0 * std::fabs(pc[4]);
    for (int k = 3; k >= 0; --k) absPR = absPR * std::fabs(out.t) + std::fabs(pcR[k]);
    absPtt = absPtt * std::fabs(out.t) + 6.0 * std::fabs(pc[3]);
    absPtt = absPtt * std::fabs(out.t) + 2.0 * std::fabs(pc[2]);
    const double eps = std::numeric_limits<double>::epsilon();
    out.ordinary = std::isfinite(out.Rp) && std::isfinite(out.tp) &&
        std::fabs(out.PR) > 128.0 * eps * std::max(1.0, absPR) &&
        std::fabs(out.Ptt) > 128.0 * eps * std::max(1.0, absPtt);
    out.valid = out.ordinary && std::isfinite(out.Pp) &&
        std::isfinite(out.Ptp) && std::isfinite(pd.P) &&
        std::isfinite(absPp) && std::isfinite(absPt);
    return out;
}

struct MapEndpoint {
    double hi = 0.0, lo = 0.0, uncertainty = 0.0;
    double Rp = 0.0;
    bool fold = false, sensitivity_valid = true;
};
struct CellMap {
    int cell = -1;
    CellPlan plan{};
    FoldRadialMap map{};
    MapEndpoint left, right;

    std::array<double, 2> at(double x) const {
        const long double a = static_cast<long double>(left.hi) + left.lo;
        const long double b = static_cast<long double>(right.hi) + right.lo;
        const long double t = (1.0L + static_cast<long double>(x)) * 0.5L;
        const long double w = b - a;
        const long double pi = static_cast<long double>(kPiShadow);
        long double R = 0.0L, J = 0.0L;
        if (map.left && map.right) {
            const long double z = std::sin(pi * 0.5L * (t <= 0.5L ? t : 1.0L - t));
            R = t <= 0.5L ? a + w * z * z : b - w * z * z;
            J = pi * w * 0.25L * std::sin(pi * t);
        } else if (map.left) {
            R = a + w * t * t; J = w * t;
        } else if (map.right) {
            const long double s = 1.0L - t;
            R = b - w * s * s; J = w * s;
        } else {
            R = a + w * t; J = w * 0.5L;
        }
        return {static_cast<double>(R), static_cast<double>(J)};
    }

    std::array<double, 2> motion(double x) const {
        const double t = 0.5 * (1.0 + x), wdot = right.Rp - left.Rp;
        const double pi = kPiShadow;
        double Rp = 0.0, Jp = 0.0;
        if (map.left && map.right) {
            const double z = std::sin(0.5 * pi * (t <= 0.5 ? t : 1.0 - t));
            const double A = z * z;
            Rp = t <= 0.5 ? left.Rp + wdot * A : right.Rp - wdot * A;
            Jp = pi * 0.25 * std::sin(pi * t) * wdot;
        } else if (map.left) {
            Rp = left.Rp + wdot * t * t; Jp = t * wdot;
        } else if (map.right) {
            const double s = 1.0 - t;
            Rp = right.Rp - wdot * s * s; Jp = s * wdot;
        } else {
            Rp = left.Rp + wdot * t; Jp = 0.5 * wdot;
        }
        return {Rp, Jp};
    }
};

bool ordinary_event(const RadialEvent& e) {
    return e.physically_real && e.kind == "physical_real";
}

bool make_cell_maps(const TopologyResult& topo, const PrimaryFrame& pf,
                    const LensParams& p, int user_j, std::vector<CellPlan>& plans,
                    std::vector<CellMap>& maps,
                    std::vector<EventSensitivity>& sensitivities,
                    double value_budget) {
    (void)value_budget;
    if (!adaptive_detail::restore_physical_cuts(topo, pf, plans)) return false;
    const Direction dir = user_direction(p, user_j);
    sensitivities.resize(topo.events.size());
    for (std::size_t i = 0; i < topo.events.size(); ++i) {
        const auto& e = topo.events[i];
        if (!ordinary_event(e)) continue;
        sensitivities[i] = event_sensitivity(e, pf, dir, value_budget);
        if (!sensitivities[i].valid) return false;
    }
    auto event_index = [&](double R) -> int {
        for (std::size_t i = 0; i < topo.events.size(); ++i)
            if (topo.events[i].radius == R && ordinary_event(topo.events[i]))
                return static_cast<int>(i);
        return -1;
    };
    maps.reserve(plans.size());
    for (std::size_t ci = 0; ci < plans.size(); ++ci) {
        const auto& c = plans[ci];
        const int li = event_index(c.r_lo), ri = event_index(c.r_hi);
        CellMap cm; cm.cell = static_cast<int>(ci); cm.plan = c;
        cm.left.hi = c.r_lo; cm.right.hi = c.r_hi;
        cm.left.fold = li >= 0; cm.right.fold = ri >= 0;
        cm.map = FoldRadialMap{c.r_lo, c.r_hi, cm.left.fold, cm.right.fold};
        if (li >= 0) {
            cm.left.hi = sensitivities[li].R;
            cm.left.lo = sensitivities[li].Rlo;
            cm.left.uncertainty = topo.events[li].radius_uncertainty;
            cm.left.Rp = sensitivities[li].Rp;
            cm.left.sensitivity_valid = sensitivities[li].valid;
        }
        if (ri >= 0) {
            cm.right.hi = sensitivities[ri].R;
            cm.right.lo = sensitivities[ri].Rlo;
            cm.right.uncertainty = topo.events[ri].radius_uncertainty;
            cm.right.Rp = sensitivities[ri].Rp;
            cm.right.sensitivity_valid = sensitivities[ri].valid;
        }
        if (!(cm.left.hi + cm.left.lo < cm.right.hi + cm.right.lo)) return false;
        maps.push_back(cm);
    }
    return true;
}

struct KRadialJet { double value = 0.0, derivative = 0.0; bool ok = false; };

KRadialJet k_radial_jet(double m, double v, double R,
                        const PrimaryFrame& pf,
                        const std::array<double, 5>& pc,
                        const std::array<double, 5>& pcR,
                        const RootPairDR& dr, bool reciprocal) {
    KRadialJet out;
    if (!dr.ok || !(v > kHoloVFloor) || !std::isfinite(m) ||
        !std::isfinite(v)) return out;
    const double s = std::sqrt(v), rho2 = 2.0 / pf.rho;
    const DeflatedQuad q = deflate_mv(pc, m, v);
    const double b0 = reciprocal ? R + pf.a : R - pf.a;
    const double b2 = reciprocal ? (R - pf.a) * (R - pf.a)
                                 : (R + pf.a) * (R + pf.a);
    double checked = 0.0; KRejectReason why = KRejectReason::none;
    if (!k_rule_converged(m, v, s, q, b0, b2, rho2, checked, &why)) return out;

    const double p4 = q.p4, d1 = q.d1, d0 = q.d0;
    const double dp4 = pcR[4];
    const double dd1 = pcR[3] + 2.0 * (dr.dm_dR * p4 + m * dp4);
    const double dd0 = pcR[2] + 2.0 * (dr.dm_dR * d1 + m * dd1) -
        (2.0 * m * dr.dm_dR - dr.dv_dR) * p4 - (m * m - v) * dp4;
    const auto& gc = gc2_rule16();
    double G = 0.0, dG = 0.0;
    for (int i = 0; i < kHoloNK; ++i) {
        const double xi = gc.x[i], wi = gc.w[i];
        const double t = m + s * xi;
        const double S2 = -(d0 + t * (d1 + t * p4));
        if (!(S2 > 0.0)) return out;
        const double A = 1.0 + t * t;
        const double B = b0 * b0 + b2 * t * t;
        if (!(B > 0.0)) return out;
        const double wv = std::sqrt(S2) / (A * std::sqrt(A) * std::sqrt(B));
        G += wi * wv;
        const double dt = dr.dm_dR + xi * (0.5 / s) * dr.dv_dR;
        const double dS2dt = -(d1 + 2.0 * t * p4);
        const double dS2 = -(dd0 + t * (dd1 + t * dp4)) + dS2dt * dt;
        const double dA = 2.0 * t * dt;
        const double db2 = 2.0 * (reciprocal ? R - pf.a : R + pf.a);
        const double dB = 2.0 * b0 + db2 * t * t + 2.0 * b2 * t * dt;
        const double dw = wv * (0.5 * dS2 / S2 - 1.5 * dA / A - 0.5 * dB / B);
        dG += wi * dw;
    }
    out.value = rho2 * v * G;
    out.derivative = rho2 * (dr.dv_dR * G + v * dG);
    out.ok = std::isfinite(out.value) && std::isfinite(out.derivative) &&
             std::fabs(out.value - checked) <=
                 2e-11 * std::max(1.0, std::fabs(checked));
    return out;
}

struct FluxRJet { double FR = 0.0; int method_mask = 0; bool ok = false; };

FluxRJet radial_derivative(double R, double u, const PrimaryFrame& pf,
                           const CellPlan& cell,
                           const AdaptiveSample* same_node=nullptr) {
    FluxRJet out;
    QuarticWarm qw=same_node?same_node->quartic:QuarticWarm{};
    RootPairWarm rp=same_node?same_node->roots:RootPairWarm{};
    QuarticCoeffs pc{};
    const ArcSet arcs = adaptive_detail::adaptive_arc_intervals(
        R, pf, &qw, &rp, &pc, cell.n_crossings);
    if (arcs.kind == ArcKind::kEmpty) { out.ok = true; return out; }
    if (arcs.kind == ArcKind::kFull) {
        double sum = 0.0, sumR = 0.0;
        constexpr int n = 256;
        const double dth = kTwoPiShadow / n;
        for (int k = 0; k < n; ++k) {
            const double th = kTwoPiShadow * k / n;
            const double ph = phi_val(R, th, pf);
            if (!(ph > 0.0)) return out;
            const double sq = std::sqrt(ph);
            sum += sq; sumR += phi_dR(R, th, pf) / (2.0 * sq);
        }
        out.FR = (1.0 - u) * kTwoPiShadow +
                 u * dth * (sum + R * sumR);
        out.method_mask = 8; out.ok = std::isfinite(out.FR); return out;
    }
    if (arcs.kind != ArcKind::kArcs) return out;

    const QuarticCoeffs pcR = boundary_quartic_dR(R, pf);
    double f0R = 0.0, fhR = 0.0;
    for (const auto& arc : arcs.arcs) {
        const auto pe = polish_endpoint(R, arc[0], pf);
        const auto pl = polish_endpoint(R, arc[1], pf);
        if (!pe.reliable || !pl.reliable) return out;
        const double te = pe.theta;
        double tl = pl.theta;
        if (tl <= te) tl += kTwoPiShadow;
        const double width = tl - te;
        double widthR = std::numeric_limits<double>::quiet_NaN();
        const std::array<double, 5> zero{};
        for (bool reciprocal : {false, true}) {
            const ArcPairJac ap = reciprocal
                ? arc_pair_jac_reciprocal(te, tl, zero, zero)
                : arc_pair_jac(te, tl, zero, zero);
            if (!ap.ok) continue;
            const auto cpc = reciprocal ? boundary_quartic_reciprocal(pc)
                                        : pc;
            const auto cpcR = reciprocal ? boundary_quartic_reciprocal(pcR)
                                          : pcR;
            const RootPairDR dr = root_pair_dR(RootPair{ap.m, ap.v},
                                               cpc.p, cpcR.p);
            if (!dr.ok) continue;
            const double den = (1.0 + ap.m * ap.m - ap.v) *
                               (1.0 + ap.m * ap.m - ap.v) + 4.0 * ap.v;
            const double num = 2.0 * ((1.0 + ap.m * ap.m + ap.v) *
                                      dr.dv_dR - 4.0 * ap.m * ap.v * dr.dm_dR);
            widthR = num / (std::sqrt(ap.v) * den);
            if (std::isfinite(widthR)) break;
        }
        if (!std::isfinite(widthR)) {
            const auto theta_R = [&](double th) {
                const double phR = phi_dR(R, th, pf);
                const double phT = phi_val_dtheta(R, th, pf).dphi_dtheta;
                return -phR / phT;
            };
            widthR = theta_R(tl) - theta_R(te);
        }
        if (!std::isfinite(widthR)) return out;
        f0R += width + R * widthR;

        if (u == 0.0) continue;
        bool got_k = false;
        for (bool reciprocal : {false, true}) {
            const ArcPairJac ap = reciprocal
                ? arc_pair_jac_reciprocal(te, tl, zero, zero)
                : arc_pair_jac(te, tl, zero, zero);
            if (!ap.ok) continue;
            const auto cpc = reciprocal ? boundary_quartic_reciprocal(pc)
                                        : pc;
            const auto cpcR = reciprocal ? boundary_quartic_reciprocal(pcR)
                                          : pcR;
            const RootPairDR dr = root_pair_dR(RootPair{ap.m, ap.v},
                                               cpc.p, cpcR.p);
            const KRadialJet kj = k_radial_jet(ap.m, ap.v, R, pf, cpc.p,
                                               cpcR.p, dr, reciprocal);
            if (kj.ok) {
                fhR += kj.derivative; out.method_mask |= reciprocal ? 2 : 1;
                got_k = true; break;
            }
        }
        if (!got_k) {
            const auto& rule = ang_rule();
            const double half = 0.5 * width, mid = te + half;
            double I = 0.0, IR = 0.0;
            for (int k = 0; k < 64; ++k) {
                const double th = mid + half * rule.x[k];
                const double ph = phi_val(R, th, pf);
                if (!(ph > 0.0)) return out;
                const double sq = std::sqrt(ph);
                I += rule.w[k] * sq;
                IR += rule.w[k] * phi_dR(R, th, pf) / (2.0 * sq);
            }
            fhR += half * (I + R * IR);
            out.method_mask |= 4;
        }
    }
    out.FR = (1.0 - u) * f0R + u * fhR;
    out.ok = std::isfinite(out.FR);
    return out;
}

struct Node {
    double x=0,R=0,J=0,Rp=0,Jp=0,weight=0,F=0,Fp=0,FR=0;
    double value_integrand=0,old=0,radial_term=0,jac_term=0,moving=0;
    double inner=0,geometry=0,roundoff=0;
    int fr_method=0;
    int reject_reason=0;
    bool sample_ok=false,fr_ok=false,ok=false;
};
struct CellSamples {
    CellMap cell;
    std::array<Node, 256> n{}; // finest-grid slots [1,255]
};

struct CompensatedLongDouble {
    long double s=0.0L,c=0.0L;
    void add(long double x) {
        const long double t=s+x;
        c += std::fabs(s)>=std::fabs(x) ? (s-t)+x : (x-t)+s;
        s=t;
    }
    long double get() const { return s+c; }
};

struct QSummary {
    long double value=0,old=0,moving=0,old_abs=0,moving_abs=0;
    double old_detail=0,moving_detail=0;
};

QSummary integrate_cell(const CellSamples& cs, int level) {
    QSummary q;
    const int m=1<<level, stride=256/m;
    const auto& rule=fejer_rule(level);
    CompensatedLongDouble sv,so,sm,soa,sma;
    for (int k=1;k<m;++k) {
        const Node& n=cs.n[k*stride];
        if (!n.ok) continue;
        const long double w=rule.w[k-1];
        sv.add(w*n.value_integrand); so.add(w*n.old); sm.add(w*n.moving);
        soa.add(w*std::fabs(n.old)); sma.add(w*std::fabs(n.moving));
    }
    q.value=sv.get(); q.old=so.get(); q.moving=sm.get();
    q.old_abs=soa.get(); q.moving_abs=sma.get();
    if (level >= 2) {
        auto detail=[&](bool moving) {
            const int nc=m/2-1;
            if(nc<=0)return 0.0;
            long double norm=0.0L;
            for(int k=1;k<m;k+=2) {
                const Node& fine=cs.n[k*stride];
                if(!fine.ok)continue;
                long double pred=0.0L;
                bool valid=true;
                for(int h=1;h<=nc;++h) {
                    const Node& coarse=cs.n[2*h*stride];
                    if(!coarse.ok){valid=false;break;}
                    pred += rule.interp[(k/2)*nc+h-1] *
                            (moving?coarse.moving:coarse.old);
                }
                if(!valid)continue;
                const double val=moving?fine.moving:fine.old;
                const long double d=static_cast<long double>(val)-pred;
                norm += rule.norm_w[k-1]*d*d;
            }
            return std::sqrt(static_cast<double>(kPiShadow*norm));
        };
        q.old_detail=detail(false); q.moving_detail=detail(true);
    }
    return q;
}

std::string user_name(int j) { return j==0?"X":(j==1?"Y":"a"); }
std::string kind_name(ArcKind k) {
    switch(k) { case ArcKind::kArcs:return "arcs"; case ArcKind::kFull:return "full";
        case ArcKind::kEmpty:return "empty"; default:return "degenerate"; }
}

void run_case(const LensParams& p, double u, const std::string& name, int row,
              int user_j, const std::string& outdir) {
    const PrimaryFrame pf=PrimaryFrame::from(p);
    const auto topo=classify_cells(pf,nullptr,nullptr,true);
    std::vector<CellPlan> plans;
    std::vector<CellMap> maps;
    std::vector<EventSensitivity> sens;
    AdaptiveConfig cfg; cfg.gradient_policy=GradientPolicy::ValueFirst;
    cfg.tol.mu_rtol=1e-3;
    const double value_budget=cfg.tol.budget(0,1.0);
    const bool map_ok=topo.status==Status::OK && make_cell_maps(
        topo,pf,p,user_j,plans,maps,sens,value_budget);

    static bool header_written=false;
    static std::ofstream nodes, panels, events, all_events, totals, cases, pointwise;
    if(!header_written) {
        nodes.open(outdir+"/nodes.tsv",std::ios::out|std::ios::trunc);
        panels.open(outdir+"/cells.tsv",std::ios::out|std::ios::trunc);
        events.open(outdir+"/events.tsv",std::ios::out|std::ios::trunc);
        all_events.open(outdir+"/all_events.tsv",std::ios::out|std::ios::trunc);
        totals.open(outdir+"/totals.tsv",std::ios::out|std::ios::trunc);
        cases.open(outdir+"/cases.tsv",std::ios::out|std::ios::trunc);
        pointwise.open(outdir+"/pointwise_fd.tsv",std::ios::out|std::ios::trunc);
        nodes<<std::setprecision(17);
        panels<<std::setprecision(17);
        events<<std::setprecision(17);
        all_events<<std::setprecision(17);
        totals<<std::setprecision(17);
        cases<<std::setprecision(17);
        pointwise<<std::setprecision(17);
        nodes<<"row\tcase\tu\tparameter\tcell\tkind\tlevel\tk\tfejer_weight\tx\tR\tJ\tRp\tJp\tF\tFp\tFR\told\tJFRRp\tFJp\tmoving\tvalue\tinner\tgeometry\troundoff\tfr_method\tsample_ok\treject_code\treject_reason\tfr_ok\tok\n";
        panels<<"row\tcase\tu\tparameter\tcell\tkind\tcrossings\tleft_fold\tright_fold\ta\tb\ta_lo\tb_lo\twidth\tleft_Rp\tright_Rp\tlevel\tnodes\tvalid_nodes\tcomplete\tvalue_q\told_q\tmoving_q\told_detail\tmoving_detail\told_abs_q\tmoving_abs_q\n";
        events<<"row\tcase\tu\tparameter\tevent_index\tinput_R\tinput_R_lo\tR\tR_lo\tt\tP\tPt\tPR\tPtt\tPp\tPtr\tPtp\tR_p\tt_p\tuncertainty\ttier\tordinary\tvalid\n";
        all_events<<"row\tcase\tu\tparameter\tevent_index\tkind\tphysically_real\tdetail\tR\tR_lo\tuncertainty\tfold_t_seed\tfold_t_seed_valid\ttier\tpositive_certified\tpositive_root_id\td14_condition\n";
        totals<<"row\tcase\tu\tparameter\tlevel\tcomplete_cells\tall_cells_complete\tvalid_node_count\tmu_q\told_gradient_valid_only\tmoving_gradient_valid_only\tmoving_minus_old_valid_only\told_abs_contribution\tmoving_abs_contribution\told_cancellation\tmoving_cancellation\n";
        cases<<"row\tcase\tu\tparameter\ttopology_status\tr_max\tevents\tphysical_events\ttopology_cells\tadaptive_cells\tmap_ok\tbarycentric\txs\tys\trho\tq\ta\n";
        pointwise<<"row\tcase\tu\tparameter\tcell\tR\tstep\tanalytic_Fp\tfd_Fp\trelative_difference\tbase_ok\tplus_ok\tminus_ok\n";
        header_written=true;
    }
    std::cout<<std::setprecision(17);
    int nphysical=0;
    for(const auto& e:topo.events)if(ordinary_event(e))++nphysical;
    for(std::size_t i=0;i<topo.events.size();++i) {
        const auto& e=topo.events[i];
        all_events<<row<<'\t'<<name<<'\t'<<u<<'\t'<<user_name(user_j)<<'\t'<<i<<'\t'<<e.kind<<'\t'
                  <<e.physically_real<<'\t'<<e.detail<<'\t'<<e.radius<<'\t'
                  <<e.radius_lo<<'\t'<<e.radius_uncertainty<<'\t'<<e.fold_t_seed<<'\t'
                  <<e.fold_t_seed_valid<<'\t'<<e.precision_tier<<'\t'
                  <<e.positive_certified<<'\t'<<e.positive_root_id<<'\t'
                  <<e.d14_condition<<'\n';
    }
    cases<<row<<'\t'<<name<<'\t'<<u<<'\t'<<user_name(user_j)<<'\t'
         <<int(topo.status)<<'\t'<<topo.r_max<<'\t'<<topo.events.size()<<'\t'
         <<nphysical<<'\t'<<topo.cells.size()<<'\t'<<plans.size()<<'\t'<<map_ok<<'\t'
         <<p.barycentric<<'\t'<<p.xs<<'\t'<<p.ys<<'\t'<<p.rho<<'\t'<<p.q<<'\t'<<p.a<<'\n';
    if(!map_ok) {
        std::cerr<<"mapping failed: "<<name<<" u="<<u<<" parameter="<<user_name(user_j)
                 <<" topology="<<int(topo.status)<<" events="<<topo.events.size()<<"\n";
        return;
    }

    for(std::size_t i=0;i<topo.events.size();++i) {
        if(!ordinary_event(topo.events[i]))continue;
        const auto& e=topo.events[i]; const auto& s=sens[i];
        events<<row<<'\t'<<name<<'\t'<<u<<'\t'<<user_name(user_j)<<'\t'<<i<<'\t'
              <<e.radius<<'\t'<<e.radius_lo<<'\t'<<s.R<<'\t'<<s.Rlo<<'\t'
              <<s.t<<'\t'<<s.P<<'\t'<<s.Pt<<'\t'<<s.PR<<'\t'
              <<s.Ptt<<'\t'<<s.Pp<<'\t'<<s.Ptr<<'\t'<<s.Ptp<<'\t'
              <<s.Rp<<'\t'<<s.tp<<'\t'<<s.uncertainty<<'\t'<<s.tier<<'\t'
              <<s.ordinary<<'\t'<<s.valid<<'\n';
    }

    // Pointwise fixed-R sensitivity check. This is independent of the radial
    // integration and tests that J*F_p itself matches a perturbation of the
    // same angular observable at the same R and same cell topology.
    const PrimaryFrame base_pf=pf;
    for(const auto& cm:maps) {
        if(cm.plan.kind!=ArcKind::kArcs)continue;
        const double R=cm.at(0.0)[0];
        const auto base=adaptive_detail::mapped_radius(R,1.0,p,u,base_pf,
            cm.plan,true,false,nullptr,nullptr,0.0,false,nullptr);
        for(double h:{1e-6,1e-7}) {
            LensParams pp=p,pm=p;
            user_parameter(pp,user_j)+=h; user_parameter(pm,user_j)-=h;
            const auto plus=adaptive_detail::mapped_radius(R,1.0,pp,u,
                PrimaryFrame::from(pp),cm.plan,false,false,nullptr,nullptr,
                0.0,false,nullptr);
            const auto minus=adaptive_detail::mapped_radius(R,1.0,pm,u,
                PrimaryFrame::from(pm),cm.plan,false,false,nullptr,nullptr,
                0.0,false,nullptr);
            const double fd=(plus.value[0]-minus.value[0])/(2.0*h);
            const double an=base.value[user_j+1];
            const double rel=std::fabs(an-fd)/std::max({1.0,std::fabs(an),std::fabs(fd)});
            pointwise<<row<<'\t'<<name<<'\t'<<u<<'\t'<<user_name(user_j)<<'\t'
                     <<cm.cell<<'\t'<<R<<'\t'<<h<<'\t'<<an<<'\t'<<fd<<'\t'
                     <<rel<<'\t'<<base.reliable<<'\t'<<plus.reliable<<'\t'
                     <<minus.reliable<<'\n';
        }
    }

    std::vector<CellSamples> all;
    all.reserve(maps.size());
    const double D=kPiShadow*p.rho*p.rho*(1.0-u/3.0);
    for(const auto& cm:maps) {
        CellSamples cs; cs.cell=cm;
        AdaptiveSample previous; bool have_previous=false;
        for(int k=1;k<256;++k) {
            const double x=fejer_rule(8).x[k-1];
            const auto rj=cm.at(x); const auto motion=cm.motion(x);
            if(!cm.left.sensitivity_valid||!cm.right.sensitivity_valid)continue;
            AdaptiveSample sample;
            FluxRJet fr;
            if(cm.plan.kind==ArcKind::kEmpty) {
                // The certified radial cell is empty for its whole interior.
                // Re-solving angular roots here can be ill-conditioned near
                // a neighboring pair of physical events and is unnecessary.
                sample.reliable=true;
                fr.ok=true;
            } else {
                sample=adaptive_detail::mapped_radius(
                    rj[0],1.0,p,u,pf,cm.plan,true,topo.from_warm_d14,
                    have_previous?&previous:nullptr,nullptr,0.0,false,nullptr);
                if(!sample.reliable && have_previous)
                    sample=adaptive_detail::mapped_radius(rj[0],1.0,p,u,pf,
                        cm.plan,true,false,nullptr,nullptr,0.0,false,nullptr);
                fr=radial_derivative(rj[0],u,pf,cm.plan,&sample);
            }
            Node n; n.x=x; n.R=rj[0]; n.J=rj[1]; n.Rp=motion[0]; n.Jp=motion[1];
            n.weight=fejer_rule(8).w[k-1];
            n.fr_method=fr.method_mask; n.FR=fr.FR;
            n.sample_ok=sample.reliable;
            n.reject_reason=static_cast<int>(sample.reject_reason);
            n.fr_ok=fr.ok;
            n.ok=sample.reliable && fr.ok && std::isfinite(D) && D>0.0;
            if(n.ok) {
                n.F=sample.value[0]*D;
                n.Fp=sample.value[user_j+1]*D;
                n.value_integrand=n.J*n.F/D;
                n.old=n.J*n.Fp/D;
                n.radial_term=n.J*n.FR*n.Rp/D;
                n.jac_term=n.F*n.Jp/D;
                n.moving=n.old+n.radial_term+n.jac_term;
                n.inner=sample.inner[user_j+1];
                n.geometry=sample.geometry[user_j+1];
                n.roundoff=sample.roundoff[user_j+1];
                n.ok=std::isfinite(n.moving)&&std::isfinite(n.old)&&
                     std::isfinite(n.value_integrand);
            }
            if(n.ok)previous=sample,have_previous=true;
            cs.n[k]=n;
            nodes<<row<<'\t'<<name<<'\t'<<u<<'\t'<<user_name(user_j)<<'\t'
                 <<cm.cell<<'\t'<<kind_name(cm.plan.kind)<<'\t'<<8<<'\t'<<k<<'\t'
                 <<n.weight<<'\t'<<x<<'\t'<<n.R<<'\t'<<n.J<<'\t'<<n.Rp<<'\t'<<n.Jp<<'\t'
                 <<n.F<<'\t'<<n.Fp<<'\t'<<n.FR<<'\t'<<n.old<<'\t'
                 <<n.radial_term<<'\t'<<n.jac_term<<'\t'<<n.moving<<'\t'
                 <<n.value_integrand<<'\t'<<n.inner<<'\t'<<n.geometry<<'\t'
                 <<n.roundoff<<'\t'<<n.fr_method<<'\t'<<n.sample_ok<<'\t'
                 <<n.reject_reason<<'\t'
                 <<adaptive_sample_reject_reason_name(sample.reject_reason)<<'\t'
                 <<n.fr_ok<<'\t'<<n.ok<<'\n';
        }
        all.push_back(std::move(cs));
    }

    for(int level=3;level<=8;++level) {
        CompensatedLongDouble value,old,moving,oldabs,movingabs;
        int valid_cells=0,valid_node_count=0;
        for(const auto& cs:all) {
            const auto q=integrate_cell(cs,level);
            bool valid=true; int valid_nodes=0;
            const int stride=256/(1<<level),m=1<<level;
            for(int k=1;k<m;++k) {
                if(cs.n[k*stride].ok)++valid_nodes;
                else valid=false;
            }
            if(valid)++valid_cells;
            valid_node_count+=valid_nodes;
            value.add(q.value);old.add(q.old);moving.add(q.moving);
            oldabs.add(q.old_abs);movingabs.add(q.moving_abs);
            const long double a=static_cast<long double>(cs.cell.left.hi)+cs.cell.left.lo;
            const long double b=static_cast<long double>(cs.cell.right.hi)+cs.cell.right.lo;
            panels<<row<<'\t'<<name<<'\t'<<u<<'\t'<<user_name(user_j)<<'\t'
                  <<cs.cell.cell<<'\t'<<kind_name(cs.cell.plan.kind)<<'\t'
                  <<cs.cell.plan.n_crossings<<'\t'<<cs.cell.map.left<<'\t'
                  <<cs.cell.map.right<<'\t'<<static_cast<double>(a)<<'\t'
                  <<static_cast<double>(b)<<'\t'<<cs.cell.left.lo<<'\t'
                  <<cs.cell.right.lo<<'\t'<<static_cast<double>(b-a)<<'\t'
                  <<cs.cell.left.Rp<<'\t'<<cs.cell.right.Rp<<'\t'<<level<<'\t'
                  <<m-1<<'\t'<<valid_nodes<<'\t'<<valid<<'\t'
                  <<static_cast<double>(q.value)<<'\t'<<static_cast<double>(q.old)<<'\t'
                  <<static_cast<double>(q.moving)<<'\t'<<q.old_detail<<'\t'
                  <<q.moving_detail<<'\t'<<static_cast<double>(q.old_abs)<<'\t'
                  <<static_cast<double>(q.moving_abs)<<'\n';
        }
        const long double ov=old.get(),mv=moving.get();
        totals<<row<<'\t'<<name<<'\t'<<u<<'\t'<<user_name(user_j)<<'\t'
              <<level<<'\t'<<valid_cells<<'\t'<<(valid_cells==int(all.size()))<<'\t'
              <<valid_node_count<<'\t'<<static_cast<double>(value.get())<<'\t'
              <<static_cast<double>(ov)<<'\t'<<static_cast<double>(mv)<<'\t'
              <<static_cast<double>(mv-ov)<<'\t'<<static_cast<double>(oldabs.get())<<'\t'
              <<static_cast<double>(movingabs.get())<<'\t'
              <<static_cast<double>(oldabs.get()/std::max(std::fabs(ov),1e-300L))<<'\t'
              <<static_cast<double>(movingabs.get()/std::max(std::fabs(mv),1e-300L))<<'\n';
    }
}

} // namespace

int main(int argc,char** argv) {
    if(argc!=3) {
        std::cerr<<"usage: shadow INPUT_TSV OUTPUT_DIR\n"; return 2;
    }
    std::ifstream in(argv[1]);
    std::string line; int row=0;
    while(std::getline(in,line)) {
        ++row; std::istringstream ss(line);
        LensParams p; int bary=0; double u=0,mu=0; std::string name;
        if(!(ss>>p.xs>>p.ys>>p.rho>>p.q>>p.a>>bary>>u>>mu>>name))continue;
        p.barycentric=bary!=0;
        if(name=="caustic-cross" && u==0.0)
            run_case(p,u,name,row,4,argv[2]);
        if(name=="rand035" && u==0.0) {
            run_case(p,u,name,row,0,argv[2]);
            run_case(p,u,name,row,1,argv[2]);
        }
        if(name=="rand035" && u==0.5) {
            run_case(p,u,name,row,0,argv[2]);
            run_case(p,u,name,row,1,argv[2]);
        }
    }
    return 0;
}

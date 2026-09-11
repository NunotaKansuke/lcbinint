#pragma once
#include "lcbinint/magnification/holonomic/adaptive_radial.hpp"
#include <cstdlib>
#include <type_traits>

namespace lcbinint::holonomic::adaptive_detail {
inline bool topology_event_reuse_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("HOLO_ADAPTIVE_EVENT_REUSE");
        return !(value && value[0] == '0');
    }();
    return enabled;
}

// Estimate/refine a physical event from P=Pt=0.  The topology/D14 path can
// supply both the event radius and a stationary-root seed; the local kernels
// below then use explicit quartic formulas at DD/qf precision.  They do not
// construct the generic R-polynomial family.
inline bool valid_epoch_config(const LensParams& p,double u,const AdaptiveConfig& cfg) {
    return valid_config(cfg)&&std::isfinite(u)&&u>=0&&u<=1&&
        std::isfinite(p.xs)&&std::isfinite(p.ys)&&std::isfinite(p.rho)&&p.rho>0&&
        std::isfinite(p.q)&&p.q>0&&std::isfinite(p.a)&&p.a>0;
}
struct EventLocation {
    double radius=0,uncertainty=std::numeric_limits<double>::infinity(),radius_lo=0;
    double t_seed=0;
    bool t_seed_valid=false;
    bool needs_qf=true;
    bool qf_refined=false;
    int precision_tier=0; // 0=double, 1=DD, 2=__float128
    EventDecisionReason double_reason=EventDecisionReason::NotAttempted;
    EventDecisionReason dd_reason=EventDecisionReason::NotAttempted;
    EventDecisionReason qf_reason=EventDecisionReason::NotAttempted;
    double double_residual=std::numeric_limits<double>::infinity();
    double dd_residual=std::numeric_limits<double>::infinity();
};
inline double eval_poly5(const std::array<double,5>& c,double x) {
    double value=c[4];
    for(int k=3;k>=0;--k)value=value*x+c[k];
    return value;
}
inline double eval_poly5_derivative(const std::array<double,5>& c,double x) {
    return c[1]+x*(2*c[2]+x*(3*c[3]+x*4*c[4]));
}

template <class T>
inline double event_abs(T x) {
    return std::fabs(static_cast<double>(x));
}

template <class T>
inline bool event_finite(T x) {
    return std::isfinite(static_cast<double>(x));
}

// The reflected projective chart u=-1/t is used only when the supplied local
// stationary seed is large.  It keeps the coupled fold solve conditioned near
// theta=pi without changing the topology or physical routing policy.
template <class T>
inline LocalFoldQuantities<T> reciprocal_local_fold_quantities(
    T R, T u, const PrimaryFrame& pf) {
    const T t = -T(1.0) / u;
    const auto g = local_fold_quantities<T>(R, t, pf);
    const T u2 = u * u, u3 = u2 * u, u4 = u2 * u2;
    LocalFoldQuantities<T> out;
    out.P = u4 * g.P;
    out.Pt = T(4.0) * u3 * g.P + u2 * g.Pt;
    out.PR = u4 * g.PR;
    out.Ptt = T(12.0) * u2 * g.P + T(6.0) * u * g.Pt + g.Ptt;
    out.Ptr = T(4.0) * u3 * g.PR + u2 * g.Ptr;
    return out;
}

// Newton on the exact local system P(R,t)=P_t(R,t)=0.  The equations are
// evaluated in T; only the stopping/certification screen is reduced to a
// double magnitude.  `last_step` is retained as part of the local event
// uncertainty, so a successful Newton correction cannot be mistaken for a
// mere higher-precision re-evaluation at the old seed.
template <class T, bool Reciprocal = false>
inline bool coupled_fold_newton(T& r, T& t, const PrimaryFrame& pf,
                                int max_iter, double step_tol,
                                double residual_tol, int* iterations = nullptr,
                                double* residual_out = nullptr,
                                double* last_step_out = nullptr) {
    double last_step = std::numeric_limits<double>::infinity();
    bool step_converged = false;
    int used = 0;
    for (; used < max_iter; ++used) {
        const auto g = [&] {
            if constexpr (Reciprocal)
                return reciprocal_local_fold_quantities<T>(r, t, pf);
            else
                return local_fold_quantities<T>(r, t, pf);
        }();
        const T det = g.PR * g.Ptt - g.Pt * g.Ptr;
        const double det_scale = event_abs(g.PR * g.Ptt) + event_abs(g.Pt * g.Ptr);
        const double det_rel_floor =
            std::is_same_v<T, __float128>
                ? 64.0 * (double)FLT128_EPSILON
                : 64.0 * std::numeric_limits<double>::epsilon();
        if (!event_finite(det) || event_abs(det) <=
                det_rel_floor *
                    std::max(1.0, det_scale))
            break;
        const T dr = (g.P * g.Ptt - g.Pt * g.Pt) / det;
        const T dt = (g.PR * g.Pt - g.Ptr * g.P) / det;
        if (!event_finite(dr) || !event_finite(dt)) break;
        r = r - dr;
        t = t - dt;
        last_step = event_abs(dr) + event_abs(dt);
        if (last_step <= step_tol *
                (1.0 + event_abs(r) + event_abs(t))) {
            step_converged = true;
            ++used;
            break;
        }
    }
    const auto final = [&] {
        if constexpr (Reciprocal)
            return reciprocal_local_fold_quantities<T>(r, t, pf);
        else
            return local_fold_quantities<T>(r, t, pf);
    }();
    const double pscale = 1.0 + event_abs(final.P) + event_abs(final.PR) *
                                    (1.0 + event_abs(r));
    const double tscale = 1.0 + event_abs(final.Pt) + event_abs(final.Ptt) *
                                    (1.0 + event_abs(t));
    const double residual = std::max(event_abs(final.P) / pscale,
                                     event_abs(final.Pt) / tscale);
    if (iterations) *iterations = used;
    if (residual_out) *residual_out = residual;
    if (last_step_out) *last_step_out = last_step;
    return step_converged && event_finite(r) && event_finite(t) &&
           residual <= residual_tol;
}

template <class T>
inline void store_refined_radius(EventLocation& out, T r, double last_step,
                                 int tier) {
    const double rounded = static_cast<double>(r);
    const T hi(rounded);
    const double lo = static_cast<double>(r - hi);
    const double ulp = std::fabs(std::nextafter(
        rounded, std::numeric_limits<double>::infinity()) - rounded);
    // `radius_lo` is retained and added back in the long-double map.  The
    // uncertainty must therefore describe the error of that split, rather
    // than the binary64 spacing of the unsplit radius.  A blanket
    // 64*eps*(1+R) floor is much larger than the actual DD/qf correction and
    // rejects valid fold-adjacent gradient nodes in high-magnification cases.
    const double lo_ulp = std::fabs(std::nextafter(
        lo, std::numeric_limits<double>::infinity()) - lo);
    out.radius = rounded;
    out.radius_lo = lo;
    out.uncertainty = std::max({ulp, 0.5 * lo_ulp,
                                std::fabs(lo) +
                                    (std::isfinite(last_step) ? last_step : 0.0)});
    out.precision_tier = tier;
    out.needs_qf = tier < 2;
    out.qf_refined = tier >= 2;
}
inline EventLocation double_event_estimate(double R,const PrimaryFrame& pf,double value_budget) {
    const auto q=boundary_quartic(R,pf);
    const auto qr=boundary_quartic_dR(R,pf);
    // boundary_quartic stores ascending coefficients; Aberth consumes
    // descending coefficients.  Passing q.p directly here solved the
    // reciprocal polynomial and could select a spurious huge stationary root.
    double qdesc[5]={q.p[4],q.p[3],q.p[2],q.p[1],q.p[0]};
    auto roots=aberth<double>(qdesc,4,32);
    EventLocation best;
    best.radius = R;
    best.double_reason = EventDecisionReason::NoRealCandidate;
    double best_res=std::numeric_limits<double>::infinity();
    double best_t=0;
    auto consider=[&](double t) {
        double ap=0,at=0,ar=0,att=0,pow=1,dpow=1;
        for(int k=0;k<5;++k) {
            ap+=std::fabs(q.p[k])*pow;
            ar+=std::fabs(qr.p[k])*pow;
            if(k>0)at+=k*std::fabs(q.p[k])*dpow;
            if(k>1)att+=k*(k-1)*std::fabs(q.p[k])*((k==2)?1:std::pow(std::fabs(t),k-2));
            pow*=std::fabs(t);
            if(k>0)dpow*=std::fabs(t);
        }
        ap=std::max(1.0,ap);at=std::max(1.0,at);ar=std::max(1.0,ar);att=std::max(1.0,att);
        const double P=eval_poly5(q.p,t),Pt=eval_poly5_derivative(q.p,t),PR=eval_poly5(qr.p,t);
        const double p_res=std::fabs(P)/ap,t_res=std::fabs(Pt)/at,res=std::max(p_res,t_res);
        if(res>=best_res)return;
        best_res=res;
        best_t=t;
        best.t_seed=t;
        best.t_seed_valid=std::isfinite(t);
        best.double_residual=res;
        const double ulp=std::fabs(std::nextafter(R,std::numeric_limits<double>::infinity())-R);
        const double radius_shift=std::fabs(PR)>64*eps*ar?std::fabs(P)/std::fabs(PR):std::numeric_limits<double>::infinity();
        best.uncertainty=std::max(ulp,radius_shift);
        const bool numerically_clean=res<=4096*eps&&std::fabs(PR)>64*eps*ar&&
            std::fabs(eval_poly5(q.p,t+std::sqrt(eps))-2*P+eval_poly5(q.p,t-std::sqrt(eps)))>
            64*eps*att;
        // A physical event is used as the endpoint of a squared fold map.
        // Its binary64 anchor therefore needs a representation-level
        // certificate in addition to the value ledger: allowing a coarse
        // event merely because a loose value tolerance can absorb it makes
        // the controller non-monotone (the first mapped Fejer node can lie
        // inside the reported event uncertainty).  The minimum is a local
        // ULP/conditioning test, not a physical-parameter route.
        const double budget_radius=0.05*value_budget*std::max(1.0,std::fabs(R));
        const double ulp_budget=std::max(256*ulp,ulp);
        const double anchor_budget=std::min(ulp_budget,budget_radius);
        if(numerically_clean && best.uncertainty<=anchor_budget) {
            best.double_reason=EventDecisionReason::CleanDouble;
            best.qf_reason=EventDecisionReason::NotAttempted;
            best.needs_qf=false;
        } else if(best.uncertainty<=anchor_budget) {
            best.double_reason=EventDecisionReason::DoubleBudgetAccepted;
            best.qf_reason=EventDecisionReason::NotAttempted;
            best.needs_qf=false;
        } else {
            best.double_reason=EventDecisionReason::DoubleAmbiguous;
            best.qf_reason=EventDecisionReason::QfRequired;
            best.needs_qf=true;
        }
    };
    for(const auto& z:roots) {
        if(std::fabs(z.im)>1e-7*(1+std::fabs(z.re)))continue;
        consider(z.re);
    }
    // At an ordinary fold the quartic has a double real root, which a
    // binary64 all-root solve may report as a complex pair. The stationary
    // root of P' is simple and is a cheaper, branch-independent local probe.
    if(!std::isfinite(best_res)) {
        double dc[4]={4*q.p[4],3*q.p[3],2*q.p[2],q.p[1]};
        int deg=3;while(deg&&dc[0]==0){for(int j=0;j<deg;++j)dc[j]=dc[j+1];--deg;}
        if(deg>0)for(const auto& z:aberth<double>(dc,deg,32))
            if(std::fabs(z.im)<=1e-7*(1+std::fabs(z.re)))consider(z.re);
    }
    if(std::isfinite(best_res)&&best.needs_qf) {
        DD rr(R), tt(best_t);
        int dd_iterations=0;
        double dd_res=std::numeric_limits<double>::infinity();
        double dd_step=std::numeric_limits<double>::infinity();
        const bool dd_converged = coupled_fold_newton(
            rr, tt, pf, 8, 1e-24, 1e-14, &dd_iterations, &dd_res,
            &dd_step);
        best.dd_residual=dd_res;
        const auto dd_final=local_fold_quantities<DD>(rr,tt,pf);
        const double ulp=std::fabs(std::nextafter(R,std::numeric_limits<double>::infinity())-R);
        const double dd_shift=event_abs(rr-DD(R));
        const double dd_radius_res=event_abs(dd_final.P)/
            std::max(event_abs(dd_final.PR),64*eps);
        const double budget_radius=0.05*value_budget*std::max(1.0,std::fabs(R));
        const double anchor_budget=std::min(256*ulp,budget_radius);
        const bool dd_derivative_ok=event_abs(dd_final.PR)>64*eps*std::max(1.0,event_abs(dd_final.P));
        const bool dd_budget_ok=std::max({ulp,dd_shift+dd_radius_res})<=anchor_budget;
        if(dd_converged&&dd_derivative_ok&&dd_budget_ok) {
            store_refined_radius(best,rr,dd_step,1);
            best.needs_qf=false;best.qf_refined=false;
            best.dd_reason=EventDecisionReason::DDAccepted;
            best.qf_reason=EventDecisionReason::NotAttempted;
            best.uncertainty=std::max(best.uncertainty,dd_radius_res);
        } else {
            best.dd_reason=!dd_converged ? EventDecisionReason::DDResidualRejected :
                (!dd_derivative_ok ? EventDecisionReason::DDDerivativeRejected :
                 EventDecisionReason::DDBudgetRejected);
        }
    }
    if(best.needs_qf && best.double_reason==EventDecisionReason::NoRealCandidate)
        best.qf_reason=EventDecisionReason::QfFailed;
    return best;
}
inline EventLocation refine_event(double R, double radius_lo, double t_seed,
                                  const PrimaryFrame& pf, double value_budget,
                                  EventLocation best = EventLocation{}) {
    using Q = __float128;
    Q r = (Q)R + (Q)radius_lo;
    const bool reciprocal = std::fabs(t_seed) > 64.0;
    Q t = reciprocal ? -Q(1.0) / (Q)t_seed : (Q)t_seed;
    // Re-polish the supplied stationary seed at the retained radius before
    // entering the coupled solve.  This is a one-dimensional local Newton,
    // not a second global derivative-cubic search; it is useful when the
    // binary64 event anchor has lost the qf remainder and |t| is large.
    for (int it = 0; !reciprocal && it < 12; ++it) {
        const auto g = reciprocal
                           ? reciprocal_local_fold_quantities<Q>(r, t, pf)
                           : local_fold_quantities<Q>(r, t, pf);
        const double ptt = event_abs(g.Ptt);
        if (!event_finite(g.Ptt) || ptt <= 64.0 * (double)FLT128_EPSILON)
            break;
        const Q dt = g.Pt / g.Ptt;
        if (!event_finite(dt)) break;
        t = t - dt;
        if (event_abs(dt) <= 1e-30 * (1.0 + event_abs(t))) break;
    }
    int iterations = 0;
    double residual = std::numeric_limits<double>::infinity();
    double last_step = std::numeric_limits<double>::infinity();
    const bool ok = reciprocal
        ? coupled_fold_newton<Q, true>(r, t, pf, 20, 1e-28, 1e-24,
                                       &iterations, &residual, &last_step)
        : coupled_fold_newton<Q, false>(r, t, pf, 20, 1e-28, 1e-24,
                                        &iterations, &residual, &last_step);
    best.radius = R;
    best.radius_lo = radius_lo;
    best.needs_qf = true;
    best.qf_refined = false;
    best.precision_tier = 2;
    best.qf_reason = ok ? EventDecisionReason::QfRefined
                        : EventDecisionReason::QfFailed;
    if (!ok) {
        // A failed local high-precision solve must not leave the finite
        // binary64 probe uncertainty in place.  The caller's panel guard then
        // fails closed instead of integrating across an uncertified event.
        best.uncertainty = std::numeric_limits<double>::infinity();
        best.needs_qf = true;
        return best;
    }
    store_refined_radius(best, r, last_step, 2);
    best.needs_qf = false;
    best.qf_refined = true;
    const auto final = reciprocal
                           ? reciprocal_local_fold_quantities<Q>(r, t, pf)
                           : local_fold_quantities<Q>(r, t, pf);
    const double radius_res = event_abs(final.P) /
        std::max(event_abs(final.PR), 64.0 * (double)FLT128_EPSILON);
    // Keep the qf root as a certified local result, while exposing the local
    // residual to the event ledger.  Whether this fits the value/gradient
    // budget is decided by the caller, never hidden inside the refinement.
    best.uncertainty = std::max(best.uncertainty, radius_res);
    return best;
}

// Consume the qf D14 radius and the stationary-root seed retained by
// radial_events().  The first local correction is DD; qf is entered only if
// the DD coupled solve cannot certify the event.  This is the cheap path that
// removes the duplicate quartic/cubic search in adaptive setup.
inline EventLocation topology_event_estimate(const RadialEvent& event,
                                             const PrimaryFrame& pf,
                                             double value_budget) {
    EventLocation out;
    out.radius = event.radius;
    out.radius_lo = event.radius_lo;
    out.uncertainty = event.radius_uncertainty;
    out.t_seed = event.fold_t_seed;
    out.t_seed_valid = event.fold_t_seed_valid;
    out.precision_tier = event.precision_tier;
    out.double_reason = event.fold_t_seed_valid
                            ? EventDecisionReason::TopologySeedReused
                            : EventDecisionReason::NoRealCandidate;
    out.qf_reason = EventDecisionReason::NotAttempted;
    if (!event.fold_t_seed_valid || !std::isfinite(event.fold_t_seed)) {
        auto fallback = double_event_estimate(event.radius, pf, value_budget);
        if (fallback.t_seed_valid) {
            // Preserve the D14 root remainder even if topology did not retain
            // a usable stationary seed.  The direct qf fallback can then
            // start from the same event anchor.
            fallback.radius_lo = event.radius_lo;
            fallback.uncertainty = std::max(
                fallback.uncertainty, std::fabs(event.radius_lo));
        }
        return fallback;
    }

    DD r = DD(event.radius) + DD(event.radius_lo);
    DD t(event.fold_t_seed);
    int iterations = 0;
    double residual = std::numeric_limits<double>::infinity();
    double last_step = std::numeric_limits<double>::infinity();
    const bool dd_ok = coupled_fold_newton(
        r, t, pf, 8, 1e-24, 1e-14, &iterations, &residual, &last_step);
    out.dd_residual = residual;
    const auto dd_final = local_fold_quantities<DD>(r, t, pf);
    const double radius_res = event_abs(dd_final.P) /
        std::max(event_abs(dd_final.PR), 64.0 * eps);
    const double ulp = std::fabs(std::nextafter(
        event.radius, std::numeric_limits<double>::infinity()) - event.radius);
    const double budget_radius = 0.05 * value_budget *
                                 std::max(1.0, std::fabs(event.radius));
    const double anchor_budget = std::min(256.0 * ulp, budget_radius);
    const bool derivative_ok = event_abs(dd_final.PR) >
        64.0 * eps * std::max(1.0, event_abs(dd_final.P));
    const double uncertainty = std::max({
        ulp, std::fabs(event.radius_lo) + radius_res});
    if (dd_ok && derivative_ok && uncertainty <= anchor_budget) {
        store_refined_radius(out, r, last_step, 1);
        out.needs_qf = false;
        out.qf_refined = false;
        out.dd_reason = EventDecisionReason::DDAccepted;
        out.qf_reason = EventDecisionReason::NotAttempted;
        out.uncertainty = std::max(out.uncertainty, radius_res);
        return out;
    }
    out.dd_reason = !dd_ok ? EventDecisionReason::DDResidualRejected
                           : (!derivative_ok
                                  ? EventDecisionReason::DDDerivativeRejected
                                  : EventDecisionReason::DDBudgetRejected);
    // qf starts from the same high/low radius and the corrected DD stationary
    // seed.  No polynomial-family reconstruction or derivative-cubic search.
    return refine_event(event.radius, event.radius_lo,
                        static_cast<double>(t), pf, value_budget, out);
}
// The fixed-resolution planner merges close representation / physical
// events. Restore retained physical cuts for adaptive integration; otherwise
// a fold lies inside a nominally smooth panel. No angular sampling authority.
inline bool restore_physical_cuts(const TopologyResult& topo,const PrimaryFrame& pf,
                                  std::vector<CellPlan>& cells) {
    double end=0;
    for(const auto& c:topo.cells) {
        // Skipped intervals have no certified geometry here. Do not silently
        // assign zero contribution or call it a radial tolerance success.
        if(c.r_lo!=end)return false;
        end=c.r_hi;
        std::vector<double> cuts{c.r_lo,c.r_hi};
        for(const auto& e:topo.events)if(e.physically_real&&e.kind=="physical_real"&&e.radius>c.r_lo&&e.radius<c.r_hi)cuts.push_back(e.radius);
        std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
        if(cuts.size()==2){cells.push_back(c);continue;}
        for(size_t j=1;j<cuts.size();++j) {
            double mid=cuts[j-1]+.5*(cuts[j]-cuts[j-1]);
            if(!(mid>cuts[j-1]&&mid<cuts[j]))return false;
            auto g=quartic_topology(mid,pf);
            if(!g.certified||g.n_crossings<0||(g.n_crossings&1))return false;
            cells.push_back(CellPlan{int(cells.size()),cuts[j-1],cuts[j],mid,g.kind,g.n_crossings,Status::OK});
        }
    }
    return end==topo.r_max;
}
struct FullCircleLevel {
    double integral=0;
    std::array<double,5> derivative{};
};
struct FullCircleNode {
    double value=0;
    std::array<double,5> derivative{};
    bool valid=false;
};

// Nested periodic trapezoid levels for a certified full-circle cell.  The
// finest lattice is fixed at 256 slots, so M=64/128/256 reuses the samples
// already evaluated at the coarser level.  The cell topology is supplied by
// D14/quartic classification; this routine only checks the physical phi>0
// integrand and estimates its inner quadrature error.
inline bool full_circle_periodic_level(
    int M,double R,const PrimaryFrame& pf,bool with_jac,
    std::array<FullCircleNode,256>& nodes,FullCircleLevel& level) {
    const int stride=256/M;
    Sum value_sum;
    std::array<Sum,5> derivative_sum;
    for(int i=0;i<M;++i) {
        const int slot=i*stride;
        auto& node=nodes[slot];
        if(!node.valid) {
            const double theta=kTwoPi*static_cast<double>(slot)/256.0;
            if(with_jac) {
                const PhiValDP g=phi_val_dP(R,theta,pf);
                if(!(g.phi>0.0)||!std::isfinite(g.phi))return false;
                node.value=std::sqrt(g.phi);
                if(!std::isfinite(node.value))return false;
                for(int j=0;j<5;++j) {
                    node.derivative[j]=g.dP[j]/(2.0*node.value);
                    if(!std::isfinite(node.derivative[j]))return false;
                }
            } else {
                const double phi=phi_val(R,theta,pf);
                if(!(phi>0.0)||!std::isfinite(phi))return false;
                node.value=std::sqrt(phi);
                if(!std::isfinite(node.value))return false;
            }
            node.valid=true;
        }
        value_sum.add(node.value);
        if(with_jac)for(int j=0;j<5;++j)derivative_sum[j].add(node.derivative[j]);
    }
    const double weight=kTwoPi/static_cast<double>(M);
    level.integral=weight*value_sum.get();
    if(with_jac)for(int j=0;j<5;++j)level.derivative[j]=weight*derivative_sum[j].get();
    if(!std::isfinite(level.integral))return false;
    if(with_jac)for(double d:level.derivative)if(!std::isfinite(d))return false;
    return true;
}

inline AdaptiveSample mapped_full_circle(
    double R,double jac,const LensParams& p,double u,const PrimaryFrame& pf,
    bool with_jac,const AdaptiveConfig* adaptive_cfg=nullptr) {
    AdaptiveSample out;
    auto reject=[&](AdaptiveSampleRejectReason reason) {
        out.reliable=false;out.reject_reason=reason;return out;
    };
    const double D=kPi*p.rho*p.rho*(1.0-u/3.0);
    if(!(R>0.0)||!std::isfinite(R)||!std::isfinite(jac)||!(D>0.0)||
       !std::isfinite(D))return reject(AdaptiveSampleRejectReason::Nonfinite);
    const double scale=jac/D;
    const double f0=R*kTwoPi;
    double fh=0,efh=0;
    std::array<double,5> dfh{},edfh{};
    if(u!=0.0) {
        std::array<FullCircleNode,256> nodes{};
        FullCircleLevel l32{},l64{},l128{},l256{};
        if(!full_circle_periodic_level(32,R,pf,with_jac,nodes,l32) ||
           !full_circle_periodic_level(64,R,pf,with_jac,nodes,l64))
            return reject(AdaptiveSampleRejectReason::InnerPhiNonpositive);
        const double atol=adaptive_cfg?adaptive_cfg->tol.mu_atol:1e-8;
        const double rtol=adaptive_cfg?adaptive_cfg->tol.mu_rtol:1e-4;
        const bool strict=adaptive_cfg &&
            adaptive_cfg->effective_gradient_policy()==GradientPolicy::Strict;
        const auto abs_map=[&](const std::array<double,5>& v) {
            std::array<double,5> e{};
            for(int k=0;k<5;++k) {
                std::array<double,5> unit{};unit[k]=v[k];
                const auto col=internal_to_user_jac(unit,p);
                for(int j=0;j<5;++j)e[j]+=std::fabs(col[j]);
            }
            return e;
        };
        const auto level_meets=[&](const FullCircleLevel& low,
                                   const FullCircleLevel& high) {
            const double high_fh=R*high.integral;
            const double estimate=scale*((1.0-u)*f0+u*high_fh);
            const double target=std::max(atol,rtol*std::fabs(estimate));
            const double value_error=std::fabs(scale*u)*R*
                                     std::fabs(high.integral-low.integral);
            if(!(value_error<=0.25*target))return false;
            if(!strict||!with_jac)return true;
            for(int j=0;j<5;++j) {
                std::array<double,5> raw{},raw_error{};
                raw[j]=jac*R*high.derivative[j];
                raw_error[j]=std::fabs(jac*R)*
                    std::fabs(high.derivative[j]-low.derivative[j]);
                const auto user=internal_to_user_jac(raw,p);
                const auto user_error=abs_map(raw_error);
                double gradient= u*user[j]/D;
                if(j==2)gradient-=2.0*estimate/p.rho;
                double gradient_error=std::fabs(u/D)*user_error[j];
                gradient_error+=2.0*value_error/p.rho*(j==2);
                const double gt=std::max(adaptive_cfg->tol.grad_atol[j],
                    adaptive_cfg->tol.grad_rtol[j]*std::fabs(gradient));
                if(!(gradient_error<=0.25*gt))return false;
            }
            return true;
        };
        const FullCircleLevel*low=&l32,*high=&l64;
        if(!level_meets(*low,*high)) {
            if(!full_circle_periodic_level(128,R,pf,with_jac,nodes,l128))
                return reject(AdaptiveSampleRejectReason::InnerPhiNonpositive);
            low=high;high=&l128;
            if(!level_meets(*low,*high)) {
                if(!full_circle_periodic_level(256,R,pf,with_jac,nodes,l256))
                    return reject(AdaptiveSampleRejectReason::InnerPhiNonpositive);
                low=high;high=&l256;
            }
        }
        fh=R*high->integral;
        efh=R*std::fabs(high->integral-low->integral);
        if(with_jac)for(int j=0;j<5;++j) {
            dfh[j]=jac*R*high->derivative[j];
            edfh[j]=std::fabs(jac*R)*
                std::fabs(high->derivative[j]-low->derivative[j]);
        }
    }
    out.value[0]=scale*((1.0-u)*f0+u*fh);
    out.inner[0]=std::fabs(scale*u)*efh;
    out.geometry[0]=0.0;
    out.roundoff[0]=eps*std::fabs(scale)*
        (std::fabs((1.0-u)*f0)+std::fabs(u*fh));
    if(with_jac) {
        const auto dh=internal_to_user_jac(dfh,p);
        const auto abs_map=[&](const std::array<double,5>& v) {
            std::array<double,5> e{};
            for(int k=0;k<5;++k) {
                std::array<double,5> unit{};unit[k]=v[k];
                const auto col=internal_to_user_jac(unit,p);
                for(int j=0;j<5;++j)e[j]+=std::fabs(col[j]);
            }
            return e;
        };
        const auto eh=abs_map(edfh);
        for(int j=0;j<5;++j) {
            out.value[j+1]=(u*dh[j])/D;
            out.inner[j+1]=std::fabs(u/D)*eh[j];
            out.geometry[j+1]=0.0;
            out.roundoff[j+1]=eps/std::fabs(D)*std::fabs(u*dh[j]);
        }
        out.value[3]-=2.0*out.value[0]/p.rho;
        out.inner[3]+=2.0*out.inner[0]/p.rho;
        out.roundoff[3]+=2.0*(out.roundoff[0]+eps*std::fabs(out.value[0]))/p.rho;
    }
    for(int j=0;j<(with_jac?6:1);++j) {
        if(!std::isfinite(out.value[j])||!std::isfinite(out.inner[j])||
           !std::isfinite(out.geometry[j])||!std::isfinite(out.roundoff[j])||
           out.inner[j]<0.0||out.geometry[j]<0.0||out.roundoff[j]<0.0)
            return reject(AdaptiveSampleRejectReason::Nonfinite);
    }
    return out;
}

// The incumbent arc_intervals() intentionally keeps its historical complex
// Aberth path for the fixed V2 route.  Adaptive samples have a stronger
// local contract: when that path returns an odd/incorrect real-root set,
// repair only this sample from the already available degree-4 Sturm
// certificate.  No angular grid is involved, and failure remains a hard
// rejection.
inline std::vector<double> adaptive_sturm_thetas(
    const QuarticSturmCertificate& cert) {
    if(!cert.certified || cert.root_count<0 || cert.root_count>4 ||
       (cert.root_count&1)) return {};
    const auto roots=sturm_isolate_real_roots(cert.chart_coeffs,cert.root_count);
    if(static_cast<int>(roots.size())!=cert.root_count)return {};
    std::vector<double> out;
    out.reserve(roots.size());
    for(double x:roots) {
        double theta=(cert.reciprocal?kPi:0.0)+2.0*std::atan(x);
        theta=std::fmod(theta,kTwoPi);
        if(theta<0.0)theta+=kTwoPi;
        out.push_back(theta);
    }
    std::sort(out.begin(),out.end());
    for(size_t i=1;i<out.size();++i)
        if(!(out[i]>out[i-1]) ||
           out[i]-out[i-1]<=1e-11*(1.0+std::fabs(out[i]))) return {};
    return out;
}

inline ArcSet adaptive_arc_intervals(
    double R,const PrimaryFrame& pf,QuarticWarm* qw,RootPairWarm* rpw,
    QuarticCoeffs* out_pc,int expected_crossings) {
    ArcSet arcs=arc_intervals(R,pf,qw,rpw,out_pc);
    const bool mismatch = arcs.kind==ArcKind::kDegenerate ||
        (expected_crossings>0 && arcs.kind!=ArcKind::kArcs) ||
        (arcs.kind==ArcKind::kArcs &&
         static_cast<int>(arcs.arcs.size()*2)!=expected_crossings);
    if(!mismatch)return arcs;
    QuarticCoeffs q=out_pc?*out_pc:boundary_quartic(R,pf);
    const QuarticSturmCertificate cert=certify_quartic(q.p);
    if(!cert.certified || cert.root_count!=expected_crossings)return arcs;
    const auto theta=adaptive_sturm_thetas(cert);
    if(static_cast<int>(theta.size())!=expected_crossings)return arcs;
    const ArcSet repaired=arc_set_from_root_thetas(R,pf,theta);
    if(repaired.kind==ArcKind::kArcs &&
       static_cast<int>(repaired.arcs.size()*2)==expected_crossings)
        return repaired;
    return arcs;
}

// Reuses V2 arc/root continuation, endpoint IFT and vK. The adapter carries
// numerical estimates separately from the radial interpolation detail.
inline AdaptiveSample mapped_radius(double R,double jac,const LensParams& p,
    double u,const PrimaryFrame& pf,const CellPlan& cell,bool with_jac,
    bool warm,const AdaptiveSample* seed,
    const AdaptiveConfig* adaptive_cfg=nullptr) {
    AdaptiveSample out;
    auto reject=[&](AdaptiveSampleRejectReason reason) {
        out.reliable=false;out.reject_reason=reason;return out;
    };
    // D14/cell classification certifies that a full-circle cell has no
    // boundary crossing throughout its open radial interval.  Do not redo a
    // quartic solve at every Fejer node; the periodic integrand is smooth and
    // receives its own nested error estimate below.
    if(cell.kind==ArcKind::kFull)
        return mapped_full_circle(R,jac,p,u,pf,with_jac,adaptive_cfg);
    if(cell.kind==ArcKind::kEmpty)return out;
    if(cell.kind==ArcKind::kDegenerate)
        return reject(AdaptiveSampleRejectReason::DegenerateChart);
    if(seed){out.quartic=seed->quartic;out.roots=seed->roots;}
    out.roots.certify=warm;
    QuarticCoeffs arc_pc{};
    auto arcs=adaptive_arc_intervals(R,pf,&out.quartic,&out.roots,&arc_pc,
                                     cell.n_crossings);
    if(arcs.kind==ArcKind::kDegenerate)
        return reject(AdaptiveSampleRejectReason::DegenerateChart);
    if(arcs.kind!=cell.kind)
        return reject(AdaptiveSampleRejectReason::ArcKindMismatch);
    if(arcs.kind==ArcKind::kArcs &&
       int(arcs.arcs.size()*2)!=cell.n_crossings)
        return reject(AdaptiveSampleRejectReason::RootContinuationMismatch);
    const double D=kPi*p.rho*p.rho*(1-u/3),scale=jac/D;
    double f0=0,fh=0,ef0=0,efh=0;
    std::array<double,5> df0{},dfh{},edf0{},edfh{};
    const auto pc=u!=0?arc_pc:QuarticCoeffs{};
    QuarticParamJac dpc{};if(with_jac)dpc=boundary_quartic_dp(R,pf);
    for(auto a:arcs.arcs) {
        auto pe=polish_endpoint(R,a[0],pf),pl=polish_endpoint(R,a[1],pf);
        double te=pe.theta,tl=pl.theta;if(tl<=te)tl+=kTwoPi;
        // No flag override: transformed arithmetic still requires resolvable endpoints.
        if(!pe.reliable||!pl.reliable)
            return reject(AdaptiveSampleRejectReason::EndpointUnreliable);
        PhiGrad ge{},gl{};
        if(with_jac){ge=phi_grad(R,te,pf);gl=phi_grad(R,tl,pf);}
        else{auto e=phi_val_dtheta(R,te,pf),l=phi_val_dtheta(R,tl,pf);ge.phi=e.phi;ge.dphi_dtheta=e.dphi_dtheta;gl.phi=l.phi;gl.dphi_dtheta=l.dphi_dtheta;}
        double de=std::fabs(ge.phi/ge.dphi_dtheta),dl=std::fabs(gl.phi/gl.dphi_dtheta);
        // Floating point lens-map cancellation grows as 1/rho. This is an
        // arithmetic sensitivity estimate, NOT an interval enclosure.
        double map_noise=eps*(std::fabs(R)+std::fabs(pf.X)+std::fabs(pf.Y)+1)/pf.rho;
        de+=(map_noise+eps*std::fabs(te)*std::fabs(ge.dphi_dtheta))/std::fabs(ge.dphi_dtheta);
        dl+=(map_noise+eps*std::fabs(tl)*std::fabs(gl.dphi_dtheta))/std::fabs(gl.dphi_dtheta);
        const double width=tl-te;
        if(!(width>de+dl) || !std::isfinite(de+dl))
            return reject(AdaptiveSampleRejectReason::ArcWidthUnresolved);
        f0+=R*width;ef0+=R*(de+dl);
        std::array<double,5> dte{},dtl{};
        if(with_jac)for(int j=0;j<5;++j){
            // Form the mapped derivative before division by the small fold
            // derivative: never construct the divergent fixed-R dtheta first.
            dte[j]=-(jac*ge.dP[j])/ge.dphi_dtheta;
            dtl[j]=-(jac*gl.dP[j])/gl.dphi_dtheta;
            df0[j]+=R*(dtl[j]-dte[j]);
            edf0[j]+=R*(std::fabs(dte[j])+std::fabs(dtl[j]))*((de+dl)/width+eps);
        }
        if(u==0)continue;
        bool success=false;
        // Same exact t / reciprocal choices and gates as V2. Parameter
        // derivatives come from regular E=O=0, not singular endpoint IFTs.
        const std::array<double,5> zero{};
        for(bool reciprocal:{false,true}) {
            auto ap=reciprocal?arc_pair_jac_reciprocal(te,tl,zero,zero):arc_pair_jac(te,tl,zero,zero);
            if(!ap.ok)continue;
            auto cp=reciprocal?boundary_quartic_reciprocal(pc):pc;
            auto dp=reciprocal?boundary_quartic_reciprocal_dp(dpc):dpc;
            if(with_jac) {
                bool regular=true;
                for(int j=0;j<5;++j) {
                    auto d=root_pair_dR(RootPair{ap.m,ap.v},cp.p,dp.dp[j]);
                    regular=regular&&d.ok&&std::isfinite(d.dm_dR)&&std::isfinite(d.dv_dR);
                    ap.dm[j]=d.dm_dR;ap.dv[j]=d.dv_dR;
                }
                if(!regular)continue;
                auto hi=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal);
                if(hi.ok){auto lo=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal,8);
                    if(lo.ok){fh+=hi.vK;efh+=std::fabs(hi.vK-lo.vK);for(int j=0;j<5;++j){dfh[j]+=jac*hi.dvK[j];edfh[j]+=jac*std::fabs(hi.dvK[j]-lo.dvK[j]);}success=true;}}
            }else{auto vk=v_times_K(ap.m,ap.v,R,pf,cp.p,reciprocal);if(vk.ok){fh+=vk.vK;efh+=vk.estimated_error;success=true;}}
            if(success)break;
        }
        if(!success){
            // Retain V2's angular rescue. A second, cheap 32-point rule supplies
            // the missing inner-error estimate; radial refinement cannot fix it.
            static const Cheb1<32> low;
            const auto& high=ang_rule();
            double vh=0,vl=0;std::array<double,5> dh{},dd{};
            double half=0.5*width,mid=te+half;
            for(int k=0;k<64;++k){double th=mid+half*high.x[k];
                if(with_jac){auto g=phi_val_dP(R,th,pf);if(!(g.phi>0))return reject(AdaptiveSampleRejectReason::InnerPhiNonpositive);double sq=std::sqrt(g.phi);vh+=high.w[k]*sq;for(int j=0;j<5;++j)dh[j]+=high.w[k]*(jac*g.dP[j])/(2*sq);}
                else{double ph=phi_val(R,th,pf);if(!(ph>0))return reject(AdaptiveSampleRejectReason::InnerPhiNonpositive);vh+=high.w[k]*std::sqrt(ph);}}
            for(int k=0;k<32;++k){double th=mid+half*low.x[k];
                if(with_jac){auto g=phi_val_dP(R,th,pf);if(!(g.phi>0))return reject(AdaptiveSampleRejectReason::InnerPhiNonpositive);double sq=std::sqrt(g.phi);vl+=low.w[k]*sq;for(int j=0;j<5;++j)dd[j]+=low.w[k]*(jac*g.dP[j])/(2*sq);}
                else{double ph=phi_val(R,th,pf);if(!(ph>0))return reject(AdaptiveSampleRejectReason::InnerPhiNonpositive);vl+=low.w[k]*std::sqrt(ph);}}
            fh+=R*half*vh;efh+=R*half*std::fabs(vh-vl);
            if(with_jac)for(int j=0;j<5;++j){dfh[j]+=R*half*dh[j];edfh[j]+=R*half*std::fabs(dh[j]-dd[j]);}
        }
    }
    out.value[0]=scale*((1-u)*f0+u*fh);
    out.inner[0]=std::fabs(scale*u)*efh;
    out.geometry[0]=std::fabs(scale*(1-u))*ef0+std::fabs(scale*u)*ef0;
    out.roundoff[0]=eps*std::fabs(scale)*(std::fabs((1-u)*f0)+std::fabs(u*fh));
    if(with_jac){
        auto d0=internal_to_user_jac(df0,p),dh=internal_to_user_jac(dfh,p);
        // Propagate absolute errors through the same linear map without cancellation.
        auto abs_map=[&](const std::array<double,5>& v){std::array<double,5> e{};
            for(int k=0;k<5;++k){std::array<double,5> unit{};unit[k]=1;auto col=internal_to_user_jac(unit,p);for(int j=0;j<5;++j)e[j]+=std::fabs(col[j])*v[k];}return e;};
        auto e0=abs_map(edf0),eh=abs_map(edfh);
        for(int j=0;j<5;++j){out.value[j+1]=((1-u)*d0[j]+u*dh[j])/D;out.inner[j+1]=std::fabs(u/D)*eh[j];out.geometry[j+1]=std::fabs((1-u)/D)*e0[j];
            out.roundoff[j+1]=eps/std::fabs(D)*(std::fabs((1-u)*d0[j])+std::fabs(u*dh[j]));}
        out.value[3]-=2*out.value[0]/p.rho;
        out.inner[3]+=2*out.inner[0]/p.rho;out.geometry[3]+=2*out.geometry[0]/p.rho;out.roundoff[3]+=2*(out.roundoff[0]+eps*std::fabs(out.value[0]))/p.rho;
    }
    return out;
}
}
namespace lcbinint::holonomic {
inline AdaptiveResult flux_adaptive_integrate(const LensParams& p,double u,
    const TopologyResult& topo,const AdaptiveConfig& cfg,AdaptiveWorkspace& workspace) {
    const ScopedFlushDenormals fp_guard;
    workspace.reset();
    const auto policy=cfg.effective_gradient_policy();
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)) return adaptive_detail::failure_result(AdaptiveStop::InvalidConfig,policy);
    if(cfg.require_bound) return adaptive_detail::failure_result(AdaptiveStop::BoundUnavailable,policy);
    if(topo.status!=Status::OK) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    auto setup_start=adaptive_detail::Clock::now();
    auto frame_start=adaptive_detail::Clock::now();
    const auto pf=PrimaryFrame::from(p);
    const double setup_frame_ms=adaptive_detail::ms(frame_start);
    if(near_origin_source(pf)) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    auto& cells=workspace.cells;
    auto cuts_start=adaptive_detail::Clock::now();
    if(!adaptive_detail::restore_physical_cuts(topo,pf,cells)) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    const double setup_cuts_ms=adaptive_detail::ms(cuts_start);
    // A squared map is a valid substitution even if a physical event is a
    // higher contact: we make no smooth-fold guarantee from its label alone.
    // The exact boundary values are copied from topology, so return the
    // metadata record as well as the boolean.  This avoids reconstructing the
    // same D14/stationary-root event in adaptive setup.
    auto physical_event=[&](double R)->const RadialEvent* {
        for(const auto& e:topo.events)
            if(e.radius==R && e.physically_real && e.kind=="physical_real")
                return &e;
        return nullptr;
    };
    auto physical=[&](double R){return physical_event(R)!=nullptr;};
    std::vector<std::pair<double,adaptive_detail::EventLocation>> event_errors;
    size_t event_double_checks=0,event_dd_checks=0,event_dd_accepts=0,
           event_qf_refinements=0,qf_family_constructions=0,
           event_topology_reuses=0,event_radius_reuses=0,
           event_direct_qf_failures=0;
    std::vector<AdaptiveEventDiagnostic> event_diagnostics;
    double setup_event_ms=0;
    double event_budget=cfg.tol.budget(0,1.0);
    if(policy==GradientPolicy::Strict)for(int j=1;j<6;++j)event_budget=std::min(event_budget,cfg.tol.budget(j,1.0));
    auto uncertainty=[&](double R){
        auto event_call_start=adaptive_detail::Clock::now();
        for(const auto& e:event_errors)if(e.first==R){setup_event_ms+=adaptive_detail::ms(event_call_start);return e.second;}
        const RadialEvent* topology_event=physical_event(R);
        adaptive_detail::EventLocation d;
        if(topology_event && topology_event->fold_t_seed_valid &&
           adaptive_detail::topology_event_reuse_enabled()) {
            ++event_topology_reuses;
            ++event_radius_reuses;
            d=adaptive_detail::topology_event_estimate(*topology_event,pf,event_budget);
            if(d.dd_reason!=EventDecisionReason::NotAttempted)++event_dd_checks;
            if(d.dd_reason==EventDecisionReason::DDAccepted)++event_dd_accepts;
        } else {
            ++event_double_checks;
            d=adaptive_detail::double_event_estimate(R,pf,event_budget);
            if(d.precision_tier>=1)++event_dd_checks;
            if(d.precision_tier==1)++event_dd_accepts;
        }
        if(d.needs_qf) {
            const auto double_reason=d.double_reason;
            const auto dd_reason=d.dd_reason;
            const double double_residual=d.double_residual;
            const double dd_residual=d.dd_residual;
            if(d.t_seed_valid) {
                d=adaptive_detail::refine_event(
                    R,d.radius_lo,d.t_seed,pf,event_budget,d);
                ++event_qf_refinements;
                if(d.qf_reason==EventDecisionReason::QfFailed)
                    ++event_direct_qf_failures;
            }
            d.double_reason=double_reason;d.dd_reason=dd_reason;
            d.double_residual=double_residual;d.dd_residual=dd_residual;
            if(!d.t_seed_valid)d.qf_reason=EventDecisionReason::QfFailed;
        }
        // A local stationary-point solve must not replace a certified D14
        // event by another root. Keep the incumbent path unchanged.
        if(topology_event && topology_event->positive_certified) {
            const __float128 corrected=(__float128)d.radius+d.radius_lo;
            if(!(corrected>=topology_event->certified_radius_lo &&
                 corrected<=topology_event->certified_radius_hi)) {
                d.uncertainty=std::numeric_limits<double>::infinity();
                d.t_seed_valid=false;d.qf_reason=EventDecisionReason::QfFailed;
            }
        }
        if(cfg.collect_diagnostics) {
            AdaptiveEventDiagnostic record;
            record.input_radius=R;record.selected_radius=d.radius;
            record.radius_lo=d.radius_lo;record.uncertainty=d.uncertainty;
            record.t_seed=d.t_seed;record.t_seed_valid=d.t_seed_valid;
            record.d14_condition=topology_event
                                    ? topology_event->d14_condition
                                    : std::numeric_limits<double>::infinity();
            record.topology_reused=topology_event &&
                                   topology_event->fold_t_seed_valid &&
                                   adaptive_detail::topology_event_reuse_enabled();
            record.precision_tier=d.precision_tier;
            record.double_reason=d.double_reason;record.dd_reason=d.dd_reason;
            record.qf_reason=d.qf_reason;
            record.double_residual=d.double_residual;record.dd_residual=d.dd_residual;
            event_diagnostics.push_back(record);
        }
        event_errors.emplace_back(R,d);
        setup_event_ms+=adaptive_detail::ms(event_call_start);
        return d;};
    auto panel_start=adaptive_detail::Clock::now();
    for(size_t i=0;i<cells.size();++i){const auto& c=cells[i];if(c.kind==ArcKind::kEmpty)continue;
        if(workspace.panels.size()>=cfg.max_panels) return adaptive_detail::failure_result(AdaptiveStop::BudgetExceeded,policy);
        AdaptivePanel panel;panel.cell=i;panel.map={c.r_lo,c.r_hi,cfg.fold_maps&&physical(c.r_lo),cfg.fold_maps&&physical(c.r_hi)};if(physical(c.r_lo)){auto e=uncertainty(c.r_lo);panel.map.a=e.radius;panel.left_radius_lo=e.radius_lo;panel.left_uncertainty=e.uncertainty;}
        if(physical(c.r_hi)){auto e=uncertainty(c.r_hi);panel.map.b=e.radius;panel.right_radius_lo=e.radius_lo;panel.right_uncertainty=e.uncertainty;}
        // A local correction may not jump across another cell/event.
        if(std::fabs(panel.map.a-c.r_lo)>.25*(c.r_hi-c.r_lo)||std::fabs(panel.map.b-c.r_hi)>.25*(c.r_hi-c.r_lo)||!(panel.map.a<panel.map.b)) return adaptive_detail::failure_result(AdaptiveStop::EventLocationLimited,policy);
        if(!std::isfinite(panel.left_uncertainty+panel.right_uncertainty)) return adaptive_detail::failure_result(AdaptiveStop::EventLocationLimited,policy);
        workspace.panels.push_back(panel);}
    const bool with_jacobian=policy!=GradientPolicy::None;
    const double panel_total_ms=adaptive_detail::ms(panel_start);
    const double setup_panel_ms=std::max(0.0,panel_total_ms-setup_event_ms);
    // Stop the setup clock before adaptive quadrature starts.  Previously the
    // assignment happened after integrate(), so setup_ms double-counted the
    // entire adaptive physics/estimator phase.
    const double setup_total_ms=adaptive_detail::ms(setup_start);
    auto result=adaptive_detail::integrate(workspace,cfg,[&](double R,double jac,int i,const AdaptiveSample* seed,bool force_cold){
        return adaptive_detail::mapped_radius(R,jac,p,u,pf,cells[i],with_jacobian,
                                              force_cold?false:topo.from_warm_d14,
                                              force_cold?nullptr:seed,&cfg);
    });
    result.stats.setup_ms=setup_total_ms;
    result.stats.setup_frame_ms=setup_frame_ms;
    result.stats.setup_cuts_ms=setup_cuts_ms;
    result.stats.setup_event_ms=setup_event_ms;
    result.stats.setup_panel_ms=setup_panel_ms;
    result.stats.event_double_checks=event_double_checks;
    result.stats.event_dd_checks=event_dd_checks;
    result.stats.event_dd_accepts=event_dd_accepts;
    result.stats.event_qf_refinements=event_qf_refinements;
    result.stats.qf_family_constructions=qf_family_constructions;
    result.stats.event_topology_reuses=event_topology_reuses;
    result.stats.event_radius_reuses=event_radius_reuses;
    result.stats.event_direct_qf_failures=event_direct_qf_failures;
    if(cfg.collect_diagnostics) result.stats.event_diagnostics=std::move(event_diagnostics);
    return result;
}
inline AdaptiveResult epoch_adaptive(const LensParams& p,double u,const AdaptiveConfig& cfg,AdaptiveWorkspace& w) {
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)){w.reset();return adaptive_detail::failure_result(AdaptiveStop::InvalidConfig,cfg.effective_gradient_policy());}
    if(cfg.require_bound){w.reset();return adaptive_detail::failure_result(AdaptiveStop::BoundUnavailable,cfg.effective_gradient_policy());}
    auto start=adaptive_detail::Clock::now();
    auto topo=classify_cells(PrimaryFrame::from(p),nullptr,nullptr,
                             /*retain_adaptive_metadata=*/true);
    double t=adaptive_detail::ms(start);
    auto out=flux_adaptive_integrate(p,u,topo,cfg,w);out.stats.topology_ms=t;return out;
}
inline AdaptiveResult epoch_value_adaptive(const LensParams& p,double u,AdaptiveConfig cfg,AdaptiveWorkspace& w) {cfg.gradient_policy=GradientPolicy::None;cfg.with_jacobian=false;return epoch_adaptive(p,u,cfg,w);}
inline AdaptiveResult epoch_jacobian_adaptive(const LensParams& p,double u,AdaptiveConfig cfg,AdaptiveWorkspace& w) {cfg.with_jacobian=true;return epoch_adaptive(p,u,cfg,w);}
inline AdaptiveResult epoch_adaptive_prepared(const LensParams& p,double u,const AdaptiveConfig& cfg,AdaptiveWorkspace& w,PreparedEpochGeometry& state,const PreparedReuseConfig& reuse=PreparedReuseConfig{}) {
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)){w.reset();return adaptive_detail::failure_result(AdaptiveStop::InvalidConfig,cfg.effective_gradient_policy());}
    if(cfg.require_bound){w.reset();return adaptive_detail::failure_result(AdaptiveStop::BoundUnavailable,cfg.effective_gradient_policy());}
    auto start=adaptive_detail::Clock::now();auto topo=prepared_topology(PrimaryFrame::from(p),state,reuse,nullptr);double t=adaptive_detail::ms(start);
    auto out=flux_adaptive_integrate(p,u,topo,cfg,w);out.stats.topology_ms=t;return out;
}
}

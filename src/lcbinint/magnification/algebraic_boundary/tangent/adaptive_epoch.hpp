#pragma once
#include "lcbinint/magnification/algebraic_boundary/tangent/adaptive_radial.hpp"
#include "lcbinint/magnification/algebraic_boundary/tangent/projective_fold_screen.hpp"
#include "atlas_near_fold.hpp"
#include <cstdlib>
#include <type_traits>
#include <optional>
#include "fused_k.hpp"
#include "fused_gaussian.hpp"
#include <cstdio>
#include "lcbinint/magnification/algebraic_boundary/tangent/quartic_fast_count.hpp"

namespace lcbinint::algebraic_boundary::tangent::adaptive_detail {
inline bool topology_event_reuse_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("BOUNDARY_ADAPTIVE_EVENT_REUSE");
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
    double u_seed=0;
    bool u_seed_valid=false;
    bool seed_reciprocal=false;
    bool needs_qf=true;
    bool qf_refined=false;
    int precision_tier=0; // 0=double, 1=DD, 2=__float128
    EventDecisionReason double_reason=EventDecisionReason::NotAttempted;
    EventDecisionReason dd_reason=EventDecisionReason::NotAttempted;
    EventDecisionReason qf_reason=EventDecisionReason::NotAttempted;
    double double_residual=std::numeric_limits<double>::infinity();
    double dd_residual=std::numeric_limits<double>::infinity();
    double pair_m=0,pair_dm=0,pair_dv=0;
    bool pair_motion_valid=false,pair_reciprocal=false;
};

enum class ProjectiveFoldReject {
    None,
    NotChartP4,
    P4RootRefinementFailed,
    NoCoincidentD14Event,
    NoReciprocalContact,
    ProjectiveContactUnresolved,
    DegenerateAngularContact,
    NoRadialCrossing
};

inline const char* projective_fold_reject_name(ProjectiveFoldReject r) {
    switch (r) {
        case ProjectiveFoldReject::None: return "accepted";
        case ProjectiveFoldReject::NotChartP4: return "not_chart_p4";
        case ProjectiveFoldReject::P4RootRefinementFailed: return "p4_refinement_failed";
        case ProjectiveFoldReject::NoCoincidentD14Event: return "no_coincident_d14_event";
        case ProjectiveFoldReject::NoReciprocalContact: return "no_reciprocal_contact";
        case ProjectiveFoldReject::ProjectiveContactUnresolved: return "projective_contact_unresolved";
        case ProjectiveFoldReject::DegenerateAngularContact: return "degenerate_angular_contact";
        case ProjectiveFoldReject::NoRadialCrossing: return "no_radial_crossing";
    }
    return "unknown";
}

struct ProjectiveFoldProbe {
    ProjectiveFoldReject reject=ProjectiveFoldReject::NotChartP4;
    double radius=0.0, radius_lo=0.0, uncertainty=std::numeric_limits<double>::infinity();
    double p4_relative=std::numeric_limits<double>::infinity();
    double contact_relative=std::numeric_limits<double>::infinity();
    double angular_curvature_relative=std::numeric_limits<double>::infinity();
    double radial_crossing_relative=std::numeric_limits<double>::infinity();
    double d14_delta=std::numeric_limits<double>::infinity();
    double d14_match_budget=0.0;
    int d14_event_index=-1;
    int p4_newton_steps=0;
    ProjectiveFastScreen fast_screen=ProjectiveFastScreen::Disabled;
    double fast_p4_radius_lo=std::numeric_limits<double>::quiet_NaN();
    double fast_p4_radius_hi=std::numeric_limits<double>::quiet_NaN();
    double fast_p3_lo=std::numeric_limits<double>::quiet_NaN();
    double fast_p3_hi=std::numeric_limits<double>::quiet_NaN();
    double fast_p3_gate_hi=std::numeric_limits<double>::quiet_NaN();
    double fast_d14_gap=std::numeric_limits<double>::infinity();
    double fast_d14_budget=0.0;
    int fast_p4_factor_roots=0;
    int fast_expansions=0;
    bool accepted() const { return reject==ProjectiveFoldReject::None; }
};

inline ProjectiveFoldReject projective_fold_contact_gate(
    __float128 contact_relative,__float128 angular_curvature_relative,
    __float128 radial_crossing_relative) {
    const __float128 eps=(__float128)FLT128_EPSILON;
    if(!finiteq(contact_relative)||!finiteq(angular_curvature_relative)||
       !finiteq(radial_crossing_relative))
        return ProjectiveFoldReject::ProjectiveContactUnresolved;
    if(contact_relative>(__float128)8192*eps)
        return ProjectiveFoldReject::ProjectiveContactUnresolved;
    const __float128 nondegenerate_floor=(__float128)4096*sqrtq(eps);
    if(angular_curvature_relative<=nondegenerate_floor)
        return ProjectiveFoldReject::DegenerateAngularContact;
    if(radial_crossing_relative<=nondegenerate_floor)
        return ProjectiveFoldReject::NoRadialCrossing;
    return ProjectiveFoldReject::None;
}

// Test whether a chart_p4 event is also a true projective fold.  In the
// reciprocal polynomial Q(u)=u^4 P(-1/u), u=0 is a double angular root iff
// Q(0)=Q_u(0)=0.  The nonzero Q_uu and Q_R tests exclude higher contact and a
// tangential radial touch.  Finally, the existing positive-real D14 event
// must have a high/low radius overlapping this locally polished p4 root;
// that independent discriminant event prevents a small Q_u residual alone
// from promoting a near-fold chart crossing.
inline ProjectiveFoldProbe probe_projective_p4_fold(
    const RadialEvent& chart_event, const std::vector<RadialEvent>& events,
    const PrimaryFrame& pf) {
    using Q=__float128;
    ProjectiveFoldProbe out;
    out.radius=chart_event.radius;
    if(chart_event.kind!="chart_p4")return out;

#if defined(BOUNDARY_ADAPTIVE_PROJECTIVE_P4_FAST_SCREEN)
    const auto screen=projective_p4_fast_screen(chart_event,events,pf);
    out.fast_screen=screen.result;
    out.fast_p4_radius_lo=screen.radius_lo;
    out.fast_p4_radius_hi=screen.radius_hi;
    out.fast_p3_lo=screen.p3_lo;
    out.fast_p3_hi=screen.p3_hi;
    out.fast_p3_gate_hi=screen.p3_gate_hi;
    out.fast_d14_gap=screen.nearest_d14_gap;
    out.fast_d14_budget=screen.overlap_budget;
    out.fast_p4_factor_roots=screen.factor_roots;
    out.fast_expansions=screen.expansions;
    if(screen.result==ProjectiveFastScreen::NoD14Overlap) {
        out.reject=ProjectiveFoldReject::NoCoincidentD14Event;
        out.d14_delta=screen.nearest_d14_gap;
        out.d14_match_budget=screen.overlap_budget;
        return out;
    }
    if(screen.result==ProjectiveFastScreen::NoReciprocalContact) {
        out.reject=ProjectiveFoldReject::NoReciprocalContact;
        return out;
    }
#endif

    const Q qeps=(Q)FLT128_EPSILON;
    const Q start=(Q)chart_event.radius;
    Q r=start, last_step=Q(0);
    bool converged=false;
    for(int k=0;k<12;++k) {
        const auto g=local_fold_quantities<Q>(r,Q(0),pf,true);
        if(!finiteq(g.P)||!finiteq(g.PR)||g.PR==Q(0))break;
        const Q step=g.P/g.PR;
        if(!finiteq(step)||fabsq(step)>Q(1e-8)*(Q(1)+fabsq(start)))break;
        r-=step;
        last_step=fabsq(step);
        ++out.p4_newton_steps;
        if(last_step<=Q(16)*qeps*(Q(1)+fabsq(r))) {
            converged=true;
            break;
        }
    }
    const auto g=local_fold_quantities<Q>(r,Q(0),pf,true);
    const Q q0=g.P, q1=g.Pt, q2=g.Ptt/Q(2);
    const Q q3=g.Psss/Q(6), q4=g.Pssss/Q(24);
    const Q qscale=fabsq(q0)+fabsq(q1)+fabsq(q2)+fabsq(q3)+fabsq(q4);
    if(!(converged&&finiteq(qscale)&&qscale>Q(0)&&finiteq(g.PR)&&g.PR!=Q(0))) {
        out.reject=ProjectiveFoldReject::P4RootRefinementFailed;
        return out;
    }
    const Q p4_rel=fabsq(q0)/qscale;
    const Q contact_rel=fabsq(q1)/qscale;
    const Q angular_rel=fabsq(q2)/qscale;
    const Q radial_rel=fabsq(g.PR)*fmaxq(Q(1),fabsq(r))/qscale;
    out.p4_relative=(double)p4_rel;
    out.contact_relative=(double)contact_rel;
    out.angular_curvature_relative=(double)angular_rel;
    out.radial_crossing_relative=(double)radial_rel;
    const Q p4_limit=Q(8192)*qeps;
    if(p4_rel>p4_limit) {
        out.reject=ProjectiveFoldReject::P4RootRefinementFailed;
        return out;
    }

    const Q p4_radius_error=fabsq(g.P/g.PR)+last_step+Q(128)*qeps*(Q(1)+fabsq(r));
    if(!finiteq(p4_radius_error)) {
        out.reject=ProjectiveFoldReject::P4RootRefinementFailed;
        return out;
    }
    Q best_delta=Q(1e100), best_budget=Q(0);
    for(std::size_t i=0;i<events.size();++i) {
        const auto& candidate=events[i];
        if((candidate.kind!="physical_real"&&candidate.kind!="physical_complex")||
           candidate.detail!="D14 real root"||candidate.precision_tier<2||
           !std::isfinite(candidate.radius)||candidate.radius_uncertainty<0.0||
           !std::isfinite(candidate.radius_lo)||
           !std::isfinite(candidate.radius_uncertainty))continue;
        const Q d14_r=(Q)candidate.radius+(Q)candidate.radius_lo;
        const Q delta=fabsq(r-d14_r);
        const Q allowed=Q(8)*((Q)candidate.radius_uncertainty+p4_radius_error)+
                        Q(512)*qeps*(Q(1)+fabsq(r));
        if(delta<best_delta) {
            best_delta=delta;
            best_budget=allowed;
            out.d14_event_index=(int)i;
        }
    }
    out.d14_delta=(double)best_delta;
    out.d14_match_budget=(double)best_budget;
    if(out.d14_event_index<0||best_delta>best_budget) {
        out.reject=ProjectiveFoldReject::NoCoincidentD14Event;
        return out;
    }

    // This is intentionally much tighter than a geometry-quality heuristic:
    // the D14 coincidence above identifies a repeated root, while Q_u(0)
    // verifies that this is the projective root at infinity.  Anything not
    // resolved at qf scale stays a chart event (fail closed).
    out.reject=projective_fold_contact_gate(contact_rel,angular_rel,radial_rel);
    if(out.reject!=ProjectiveFoldReject::None) {
        return out;
    }

    const double hi=chart_event.radius;
    const Q loq=r-(Q)hi;
    const double lo=(double)loq;
    const double ulp=std::fabs(std::nextafter(
        hi,std::numeric_limits<double>::infinity())-hi);
    out.radius=hi;
    out.radius_lo=lo;
    out.uncertainty=std::max(ulp,std::fabs(lo)+(double)p4_radius_error);
    if(!std::isfinite(lo)||!std::isfinite(out.uncertainty)) {
        out.reject=ProjectiveFoldReject::P4RootRefinementFailed;
        return out;
    }
    out.reject=ProjectiveFoldReject::None;
    return out;
}

inline void certify_projective_p4_events(TopologyResult& topo,
                                         const PrimaryFrame& pf) {
    for(auto& event:topo.events) {
        if(event.kind!="chart_p4")continue;
        const auto probe=probe_projective_p4_fold(event,topo.events,pf);
        if(!probe.accepted())continue;
        event.radius_lo=probe.radius_lo;
        event.radius_uncertainty=probe.uncertainty;
        event.precision_tier=2;
        event.physically_real=true;
        event.projective_fold_certified=true;
        event.fold_u_seed=0.0;
        event.fold_u_seed_valid=true;
        event.detail="p4 chart boundary; certified ordinary projective fold at u=0";
    }
}

inline bool adaptive_physical_fold_event(const RadialEvent& event) {
    return event.physically_real &&
           (event.kind=="physical_real" || event.projective_fold_certified);
}
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
    // Evaluate the homogeneous reversed polynomial directly. In particular
    // u=0 (theta=pi) is finite; reconstructing t=-1/u loses that contact.
    return local_fold_quantities<T>(R,u,pf,true);
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
                                double* last_step_out = nullptr,
                                LocalFoldQuantities<T>* final_out = nullptr) {
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
    // The acceptance caller may reuse this same final jet.
    if(final_out)*final_out=final;
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
inline EventLocation double_event_estimate(double R,const PrimaryFrame& pf,double value_budget,
                                           bool coordinate_only=false) {
    const auto q=boundary_quartic(R,pf);
    const auto qr=coordinate_only?QuarticCoeffs{}:boundary_quartic_dR(R,pf);
    // boundary_quartic stores ascending coefficients; Aberth consumes
    // descending coefficients.  Passing q.p directly here solved the
    // reciprocal polynomial and could select a spurious huge stationary root.
    double qdesc[5]={q.p[4],q.p[3],q.p[2],q.p[1],q.p[0]};
    std::vector<Cplx<double>> roots;
#ifndef BOUNDARY_FRONTIER_LEGACY_EVENT_SEED
    if(coordinate_only) {
        const auto certificate=certify_quartic(q.p);
        if(certificate.certified && (certificate.root_count==2 || certificate.root_count==4)) {
            QuarticWarm warm;
            (void)real_root_thetas_warm(q.p,warm,certificate.root_count);
            if(warm.valid && warm.deg==4)roots.assign(warm.z,warm.z+4);
        }
    }
#endif
    if(roots.empty())roots=aberth<double>(qdesc,4,32);
    EventLocation best;
    best.radius = R;
    best.double_reason = EventDecisionReason::NoRealCandidate;
    double best_res=std::numeric_limits<double>::infinity();
    double best_t=0;
    auto consider=[&](double t) {
        double ap=0,at=0,ar=0,att=0,pow=1,dpow=1;
        for(int k=0;k<5;++k) {
            ap+=std::fabs(q.p[k])*pow;
            if(!coordinate_only)ar+=std::fabs(qr.p[k])*pow;
            if(k>0)at+=k*std::fabs(q.p[k])*dpow;
            if(!coordinate_only && k>1)att+=k*(k-1)*std::fabs(q.p[k])*((k==2)?1:std::pow(std::fabs(t),k-2));
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
        if(coordinate_only)return;
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
    // Coordinate-only callers discard the precision ledger and refine their
    // own physical event. The DD ledger pass never changes t_seed.
    if(coordinate_only)return best;
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

// Retain the final fold jet's local pair motion rather than rebuilding
// the boundary polynomials when the first quadrature node needs a seed.
template<class T>
inline void retain_fold_pair_motion(EventLocation& out,double m,const LocalFoldQuantities<T>& g,bool reciprocal) {
    const double pt=(double)g.Pt,pr=(double)g.PR,ptt=(double)g.Ptt,ptr=(double)g.Ptr,ov=(double)g.Psss/6;
    const double ev=ptt/2,det=pt*ov-ev*ptt;
    if(det==0 || !std::isfinite(det))return;
    out.pair_m=m;out.pair_dm=-(ov*pr-ev*ptr)/det;
    out.pair_dv=-(-ptt*pr+pt*ptr)/det;
    out.pair_reciprocal=reciprocal;
    out.pair_motion_valid=std::isfinite(out.pair_dm)&&std::isfinite(out.pair_dv)&&std::isfinite(m);
}

// Consume the qf D14 radius and the stationary-root seed retained by
// radial_events().  The first local correction is DD; qf is entered only if
// the DD coupled solve cannot certify the event.  This is the cheap path that
// removes the duplicate quartic/cubic search in adaptive setup.
inline EventLocation topology_event_estimate(const RadialEvent& event,
                                             const PrimaryFrame& pf,
                                             double value_budget,bool retain_pair_motion=false) {
    EventLocation out;
    if(event.projective_fold_certified && event.fold_u_seed_valid) {
        out.radius=event.radius;
        out.radius_lo=event.radius_lo;
        out.uncertainty=event.radius_uncertainty;
        out.u_seed=event.fold_u_seed;
        out.u_seed_valid=true;
        out.seed_reciprocal=true;
        out.precision_tier=event.precision_tier;
        out.needs_qf=false;
        out.qf_refined=true;
        out.double_reason=EventDecisionReason::TopologySeedReused;
        out.qf_reason=EventDecisionReason::NotAttempted;
        return out;
    }
    // Seeded topology can retain a finite reciprocal stationary coordinate
    // without a D14 anchor. Reuse the existing local coupled fold solve.
    if(event.fold_u_seed_valid && std::isfinite(event.fold_u_seed)) {
#ifndef BOUNDARY_RECIPROCAL_EVENT_QF_ONLY
        // Use the same DD acceptance contract as the ordinary t chart.
        // A reciprocal seed does not by itself require software quad precision.
        DD rd=DD(event.radius)+DD(event.radius_lo), ud(event.fold_u_seed);
        double dd_residual=std::numeric_limits<double>::infinity();
        double dd_last_step=std::numeric_limits<double>::infinity();
        LocalFoldQuantities<DD> f;
        const bool dd_ok=coupled_fold_newton<DD,true>(rd,ud,pf,8,1e-24,1e-14,
                                                     nullptr,&dd_residual,&dd_last_step,&f);
        const double radius_res=event_abs(f.P)/std::max(event_abs(f.PR),64.0*eps);
        const double ulp=std::fabs(std::nextafter(event.radius,
            std::numeric_limits<double>::infinity())-event.radius);
        const double anchor_budget=std::min(256.0*ulp,
            0.05*value_budget*std::max(1.0,std::fabs(event.radius)));
        const bool derivative_ok=event_abs(f.PR)>64.0*eps*std::max(1.0,event_abs(f.P));
        const double uncertainty=std::max(ulp,std::fabs(event.radius_lo)+radius_res);
        if(dd_ok && derivative_ok && uncertainty<=anchor_budget) {
            store_refined_radius(out,rd,dd_last_step,1);
            if(retain_pair_motion)retain_fold_pair_motion(out,(double)ud,f,true);
            out.uncertainty=std::max(out.uncertainty,radius_res);
            out.u_seed=static_cast<double>(ud);out.u_seed_valid=true;
            out.seed_reciprocal=true;out.needs_qf=false;out.qf_refined=false;
            out.double_reason=EventDecisionReason::TopologySeedReused;
            out.dd_reason=EventDecisionReason::DDAccepted;
            out.dd_residual=dd_residual;
            out.qf_reason=EventDecisionReason::NotAttempted;
            return out;
        }
#endif
        using Q=__float128;
        Q r=(Q)event.radius+(Q)event.radius_lo, u=(Q)event.fold_u_seed;
        double residual=0,last_step=0;
        const bool ok=coupled_fold_newton<Q,true>(r,u,pf,20,1e-28,1e-24,
                                                 nullptr,&residual,&last_step);
        out.radius=event.radius;out.radius_lo=event.radius_lo;
        out.u_seed=static_cast<double>(u);out.u_seed_valid=ok;
        out.seed_reciprocal=true;out.needs_qf=!ok;out.qf_refined=ok;
        out.qf_reason=ok?EventDecisionReason::QfRefined:EventDecisionReason::QfFailed;
        out.double_reason=EventDecisionReason::TopologySeedReused;
        if(ok) {
            store_refined_radius(out,r,last_step,2);
            const auto f=reciprocal_local_fold_quantities<Q>(r,u,pf);
            out.uncertainty=std::max(out.uncertainty,event_abs(f.P)/
                std::max(event_abs(f.PR),64.0*(double)FLT128_EPSILON));
        }
        return out;
    }
    if(event.atlas_anchor){
        out.radius=event.radius;out.radius_lo=event.radius_lo;out.uncertainty=event.radius_uncertainty;
        out.t_seed=event.fold_t_seed;out.t_seed_valid=event.fold_t_seed_valid;out.precision_tier=2;
        out.needs_qf=false;out.double_reason=EventDecisionReason::TopologySeedReused;
        return out;
    }
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
    LocalFoldQuantities<DD> dd_final;
    const bool dd_ok = coupled_fold_newton(
        r, t, pf, 8, 1e-24, 1e-14, &iterations, &residual, &last_step,&dd_final);
    out.dd_residual = residual;
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
        if(retain_pair_motion)retain_fold_pair_motion(out,(double)t,dd_final,false);
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
        for(const auto& e:topo.events)if(adaptive_physical_fold_event(e)&&e.radius>c.r_lo&&e.radius<c.r_hi)cuts.push_back(e.radius);
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

// Keep D14/topology's complete cell partition intact, but allow one narrowly
// certified integration span across a soft complex-root cut.  This addresses
// the case where a nonphysical cut lies between two physical folds: splitting
// there leaves the second panel starting just beyond the first fold, so its
// fixed-R fold derivative is still singular at the panel's nearby left edge.
// Only an adjacent, same-topology arcs pair bracketed by physical folds can
// be coalesced, and only when the configured initial rule cannot sample the
// fold-to-cut gap. Other soft cuts and all physical/chart events remain cuts.
struct AdaptiveIntegrationSpan {
    CellPlan cell{};
    int first_source_cell=0,last_source_cell=0;
};
struct AdaptiveIntegrationPlan {
    std::vector<AdaptiveIntegrationSpan> spans;
    std::size_t softcut_fold_merges=0;
};
inline bool physical_fold_at_boundary(const TopologyResult& topo,double R) {
    for(const auto& event:topo.events)
        if(event.radius==R && adaptive_physical_fold_event(event))return true;
    return false;
}
inline bool soft_complex_cut_only_at(const TopologyResult& topo,double R) {
    bool found=false;
    for(const auto& event:topo.events)if(event.radius==R) {
        if(event.kind!="physical_complex" || event.physically_real)return false;
        found=true;
    }
    return found;
}
inline bool softcut_inside_fold_map_resolution(const CellPlan& left,
                                              const CellPlan& right,
                                              int initial_level,
                                              bool fold_maps_enabled) {
    if(!fold_maps_enabled)return false;
    if(initial_level<1 || initial_level>8)return false;
    // On the cell next to the cut, the omitted physical fold is an exterior
    // square-root singularity of the fixed-R derivative.  Its distance from
    // the artificial edge matters only when it lies inside the first
    // Fejer-II radial sample stride of that cell.  Compute that stride from
    // the actual configured nested rule and the one-fold map used by the
    // adjacent panel; this avoids a lens-parameter or fixed-radius heuristic.
    const auto& rule=fejer_rule(initial_level);
    const double t_near=0.5*(1.0+rule.x.back());
    const double mapped_stride_fraction=2.0*t_near-t_near*t_near;
    const double left_width=left.r_hi-left.r_lo;
    const double right_width=right.r_hi-right.r_lo;
    if(!(left_width>0.0 && right_width>0.0 &&
         std::isfinite(left_width+right_width)))return false;
    return left_width<=mapped_stride_fraction*right_width ||
           right_width<=mapped_stride_fraction*left_width;
}
inline AdaptiveIntegrationPlan make_adaptive_integration_plan(
    const TopologyResult& topo,const std::vector<CellPlan>& source_cells,
    int initial_level=3,bool fold_maps_enabled=true) {
    AdaptiveIntegrationPlan out;
    out.spans.reserve(source_cells.size());
    for(std::size_t i=0;i<source_cells.size();) {
        const auto& left=source_cells[i];
        if(left.kind==ArcKind::kEmpty) {++i;continue;}
        const bool pair=i+1<source_cells.size();
        if(pair) {
            const auto& right=source_cells[i+1];
            const bool same_certified_arcs=
                left.status==Status::OK && right.status==Status::OK &&
                left.kind==ArcKind::kArcs && right.kind==ArcKind::kArcs &&
                left.n_crossings>0 && left.n_crossings==right.n_crossings &&
                !(left.n_crossings&1);
            const bool exact_shared_boundary=left.r_hi==right.r_lo;
            if(same_certified_arcs && exact_shared_boundary &&
               soft_complex_cut_only_at(topo,left.r_hi) &&
               physical_fold_at_boundary(topo,left.r_lo) &&
               physical_fold_at_boundary(topo,right.r_hi) &&
               softcut_inside_fold_map_resolution(left,right,initial_level,
                                                  fold_maps_enabled)) {
                CellPlan joined=left;
                joined.r_hi=right.r_hi;
                joined.r_mid=left.r_lo+0.5*(right.r_hi-left.r_lo);
                out.spans.push_back({joined,static_cast<int>(i),
                                     static_cast<int>(i+1)});
                ++out.softcut_fold_merges;
                i+=2;
                continue;
            }
        }
        out.spans.push_back({left,static_cast<int>(i),static_cast<int>(i)});
        ++i;
    }
    return out;
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
        std::array<double,5> df0{};
        if(const auto* motion=fused_gradient::active_motion)for(int j=0;j<5;++j) {
            df0[j]=jac*motion->radius[j]*kTwoPi+motion->jacobian[j]*f0;
            dfh[j]+=jac*motion->radius[j]*(fh/R)+motion->jacobian[j]*fh;
        }
        const auto d0=internal_to_user_jac(df0,p);
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
            out.value[j+1]=((1-u)*d0[j]+u*dh[j])/D;
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
    QuarticCoeffs* out_pc,int expected_crossings,
    const QuarticCoeffs* prepared_pc=nullptr) {
    #ifdef BOUNDARY_DISABLE_COUNT_REUSE
    ArcSet arcs=arc_intervals(R,pf,qw,rpw,out_pc,prepared_pc);
#else
    // prepared_pc originates from this node's square-free Sturm certificate.
    ArcSet arcs=arc_intervals(R,pf,qw,rpw,out_pc,prepared_pc,
        prepared_pc?expected_crossings:-1);
#endif
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
template <bool PlainValue=false, bool Uniform=false>
inline AdaptiveSample mapped_radius(double R,double jac,const LensParams& p,
    double input_u,const PrimaryFrame& pf,const CellPlan& cell,bool input_with_jac,
    bool warm,const AdaptiveSample* seed,
    const AdaptiveConfig* adaptive_cfg=nullptr,double radius_lo=0,
    bool request_radial_hermite_jet=false,
    AdaptiveHermiteJet* hermite_jet_out=nullptr,
    const QuarticCoeffs* prepared_pc=nullptr) {
    const bool with_jac=PlainValue?false:input_with_jac;
    const double u=Uniform?0.0:input_u;
#ifdef BOUNDARY_FRONTIER_COST_PROFILE
    const auto mapped_header_start=V2Clock::now();
#endif
    AdaptiveSample out;
    out.jac=jac;
    if(hermite_jet_out)*hermite_jet_out=AdaptiveHermiteJet{};
    const bool request_jet=!PlainValue && request_radial_hermite_jet && hermite_jet_out && adaptive_cfg &&
        adaptive_cfg->same_node_hermite_shadow && !with_jac && u==0.0;
    const bool angular_research_requested=!PlainValue && adaptive_cfg &&
        adaptive_cfg->angular_adaptive && u!=0.0;
    // Gradient-adaptive mode starts from the incumbent 8/16 K-rule result.
    // It only enters the higher-order research path when the scheduler asks
    // to upgrade this exact cached sample.  This keeps the common initial
    // sample cost unchanged while still allowing a same-R angular correction.
    const bool angular_deferred_requested=!PlainValue && adaptive_cfg &&
        adaptive_cfg->gradient_adaptive &&
        adaptive_cfg->effective_gradient_policy()!=GradientPolicy::None &&
        u!=0.0;
    const bool angular_mode_requested=angular_research_requested ||
        angular_deferred_requested;
    const bool angular_nested_requested=angular_mode_requested &&
        adaptive_cfg->angular_nested;
    bool nested_angular_attempted=false;
    if(!PlainValue && !with_jac && !angular_mode_requested &&
       atlas_detail::try_pair_sample(R,radius_lo,jac,p,u,pf,cell,out))return out;
    // Without atlas hooks or angular upgrades the entry state is pristine:
    // its default construction already cleared all sample fields.
#ifdef BOUNDARY_ADAPTIVE_LEGACY_SAMPLE_INIT
    out=AdaptiveSample{};
#else
    if(with_jac || (!PlainValue && active_atlas_samples) || angular_mode_requested)
        out=AdaptiveSample{};
#endif
    out.jac=jac;
    out.angular_nested_attempted=nested_angular_attempted;
    // Preserve the same atlas-contact guard for a same-node angular upgrade
    // as for a newly reconstructed radial sample.
    if(angular_mode_requested && active_atlas_samples &&
       active_atlas_samples->contacts) {
        for(const auto& anchor:*active_atlas_samples->contacts) {
            const __float128 r=(__float128)R;
            if(r>=anchor.R.lo-fabsq((__float128)radius_lo) &&
               r<=anchor.R.hi+fabsq((__float128)radius_lo)) {
                out.reliable=false;
                out.reject_reason=AdaptiveSampleRejectReason::EndpointUnreliable;
                return out;
            }
        }
    }

#if !defined(BOUNDARY_ADAPTIVE_DISABLE_ANGULAR_STATE_REUSE)
    // An angular upgrade is a same-node operation.  The previous sample has
    // already passed all radial topology, endpoint, and root-pair checks at
    // this exact R.  Reuse that immutable state and evaluate only the next
    // fixed GC2 rule; rebuilding the quartic/root pair here would erase the
    // main benefit of the two-dimensional controller.
    if(angular_mode_requested && seed && seed->angular_valid &&
       seed->angular_state_valid && seed->angular_order>0 &&
       ((!angular_nested_requested && seed->angular_order<adaptive_cfg->angular_max_order) ||
        (angular_nested_requested && seed->angular_nested &&
         seed->angular_nested_cache &&
         seed->angular_order<adaptive_cfg->angular_nested_max_order)) &&
       seed->R==R && seed->jac==jac) {
        const int next=angular_nested_requested
            ? adaptive_detail::next_nested_angular_order(seed->angular_order)
            : (angular_deferred_requested && seed->angular_order==16
                ? 32 : adaptive_detail::next_angular_order(seed->angular_order));
        const int max_order=angular_nested_requested
            ? adaptive_cfg->angular_nested_max_order
            : adaptive_cfg->angular_max_order;
        if(next>seed->angular_order &&
           next<=max_order) {
            const double D=kPi*p.rho*p.rho*(1.0-u/3.0);
            const double scale=jac/D;
            auto abs_map=[&](const std::array<double,5>& v) {
                std::array<double,5> e{};
                for(int k=0;k<5;++k) {
                    std::array<double,5> unit{};unit[k]=v[k];
                    const auto col=internal_to_user_jac(unit,p);
                    for(int j=0;j<5;++j)e[j]+=std::fabs(col[j]);
                }
                return e;
            };
            bool ok=std::isfinite(D)&&std::isfinite(scale)&&
                    std::isfinite(seed->angular_m)&&
                    std::isfinite(seed->angular_v)&&
                    std::isfinite(seed->R);
            VKValue vh{};
            VKJacobian jh{};
            std::array<double,5> high_derivative{};
            std::shared_ptr<NestedAngularCache> nested_cache;
            std::size_t nested_point_evaluations=0;
            std::size_t nested_cached_point_reuses=0;
            if(ok && angular_nested_requested) {
                // Same-R upgrades mutate the existing common-lattice cache
                // in place.  No point kernel is evaluated for an old node.
                nested_cache=seed->angular_nested_cache;
                nested_angular_attempted=true;
                if(with_jac) {
                    const auto nr=v_times_K_nested_jac(
                        seed->angular_m,seed->angular_v,seed->angular_data().dm,
                        seed->angular_data().dv,R,pf,seed->angular_data().pc,
                        seed->angular_data().dpc,seed->angular_order,*nested_cache,
                        seed->angular_reciprocal,next);
                    ok=nr.ok;
                    if(ok)for(int j=0;j<5;++j)high_derivative[j]=nr.high.dvK[j];
                    if(ok) {
                        nested_point_evaluations=nr.point_evaluations;
                        nested_cached_point_reuses=nr.cached_point_reuses;
                    }
                } else {
                    const auto nr=v_times_K_nested_value(
                        seed->angular_m,seed->angular_v,R,pf,seed->angular_data().pc,
                        seed->angular_order,*nested_cache,
                        seed->angular_reciprocal,next);
                    ok=nr.ok;
                    if(ok) {
                        nested_point_evaluations=nr.point_evaluations;
                        nested_cached_point_reuses=nr.cached_point_reuses;
                    }
                }
            } else if(ok && with_jac) {
                jh=v_times_K_jac_order(seed->angular_m,seed->angular_v,
                    seed->angular_data().dm,seed->angular_data().dv,R,pf,seed->angular_data().pc,
                    seed->angular_data().dpc,seed->angular_reciprocal,next);
                ok=jh.ok;
                if(ok)for(int j=0;j<5;++j)high_derivative[j]=jh.dvK[j];
            } else if(ok) {
                vh=v_times_K_order(seed->angular_m,seed->angular_v,R,pf,
                    seed->angular_data().pc,seed->angular_reciprocal,next);
                ok=vh.ok;
            }
            if(ok) {
                AdaptiveSample fast=*seed;
                const double high0=angular_nested_requested
                    ? (nested_cache->aggregate[0])
                    : (with_jac?jh.vK:vh.vK);
                const double delta_value=scale*u*(high0-
                    seed->angular_data().result[0]);
                fast.value[0]+=delta_value;
                fast.inner[0]=std::fabs(scale*u)*
                    std::fabs(high0-seed->angular_data().result[0]);
                fast.roundoff[0]+=eps*std::fabs(delta_value);
                if(with_jac) {
                    std::array<double,5> delta_internal{},mapped_abs{};
                    for(int j=0;j<5;++j)
                        delta_internal[j]=high_derivative[j]*jac-
                            seed->angular_data().result[j+1];
                    mapped_abs=abs_map(delta_internal);
                    const auto mapped=internal_to_user_jac(delta_internal,p);
                    for(int j=0;j<5;++j) {
                        fast.value[j+1]+=u* mapped[j]/D;
                        fast.inner[j+1]=std::fabs(u/D)*mapped_abs[j];
                        fast.roundoff[j+1]+=eps*std::fabs(u/D)*mapped_abs[j];
                    }
                    // The rho derivative contains the normalization term
                    // -2*value/rho.  Apply the same delta to that term and
                    // keep its cancellation error in the local ledger.
                    fast.value[3]-=2.0*delta_value/p.rho;
                    fast.inner[3]+=2.0*fast.inner[0]/p.rho;
                    fast.roundoff[3]+=2.0*(fast.roundoff[0]+
                        eps*std::fabs(delta_value))/p.rho;
                    for(int j=0;j<5;++j)
                        ok=ok&&std::isfinite(fast.value[j+1])&&
                           std::isfinite(fast.inner[j+1])&&
                           std::isfinite(fast.roundoff[j+1]);
                }
                ok=ok&&std::isfinite(fast.value[0])&&
                    std::isfinite(fast.inner[0])&&
                    std::isfinite(fast.roundoff[0]);
                if(ok) {
                    fast.angular_data().result.fill(0.0);
                    fast.angular_data().result[0]=high0;
                    fast.angular_data().error.fill(0.0);
                    fast.angular_data().error[0]=std::fabs(high0-
                        seed->angular_data().result[0]);
                    if(with_jac)for(int j=0;j<5;++j) {
                        fast.angular_data().result[j+1]=high_derivative[j]*jac;
                        fast.angular_data().error[j+1]=std::fabs(
                            fast.angular_data().result[j+1]-
                            seed->angular_data().result[j+1]);
                    }
                    fast.angular_order=next;
                    fast.angular_work=angular_nested_requested
                        ? nested_point_evaluations : static_cast<size_t>(next);
                    fast.angular_point_evaluations=angular_nested_requested
                        ? nested_point_evaluations : 0;
                    fast.angular_cached_point_reuses=angular_nested_requested
                        ? nested_cached_point_reuses : 0;
                    fast.angular_valid=true;
                    fast.angular_state_valid=true;
                    fast.angular_state_reused=true;
                    if(angular_nested_requested) {
                        fast.angular_nested_cache=nested_cache;
                        fast.angular_nested=true;
                    }
                    fast.angular_nested_attempted=nested_angular_attempted;
                    fast.reject_reason=AdaptiveSampleRejectReason::None;
                    return fast;
                }
            }
        }
    }
#endif
    // Without atlas hooks or angular upgrades the entry state is pristine:
    // its default construction already cleared all sample fields.
#ifdef BOUNDARY_ADAPTIVE_LEGACY_SAMPLE_INIT
    out=AdaptiveSample{};
#else
    if(with_jac || (!PlainValue && active_atlas_samples) || angular_mode_requested)
        out=AdaptiveSample{};
#endif
    out.jac=jac;
    out.angular_nested_attempted=nested_angular_attempted;
    // If rounding the abscissa can change its side of an atlas fold, the
    // ordinary double endpoint path is not a substitute for the pair proof.
    if(!PlainValue && active_atlas_samples && active_atlas_samples->contacts){
        for(const auto& anchor:*active_atlas_samples->contacts){
            const __float128 r=(__float128)R;
            if(r>=anchor.R.lo-fabsq((__float128)radius_lo) &&
               r<=anchor.R.hi+fabsq((__float128)radius_lo)){
                out.reliable=false;
                out.reject_reason=AdaptiveSampleRejectReason::EndpointUnreliable;
                return out;
            }
        }
    }
    auto reject=[&](AdaptiveSampleRejectReason reason) {
        out.reliable=false;out.reject_reason=reason;return out;
    };
    // A negative crossing count is reserved for the certified-component
    // research path.  Its radial support is complete, but it deliberately
    // has no D14 interior-event partition, so topology is certified locally
    // at every adaptive node.
#ifdef BOUNDARY_FRONTIER_COST_PROFILE
    v2_profile_add_ms(&V2Profile::mapped_header_ms,mapped_header_start,V2Clock::now());
#endif
    QuarticCoeffs node_pc{};
    int node_crossings=cell.n_crossings;
    ArcKind node_kind=cell.kind;
    if(cell.n_crossings<0) {
#ifdef BOUNDARY_FRONTIER_COST_PROFILE
        const QuarticCoeffs pc=[&] {
            V2ProfileTimer coeff_timer(&V2Profile::mapped_coeff_ms);
            return boundary_quartic(R,pf);
        }();
#else
        const QuarticCoeffs pc=boundary_quartic(R,pf);
#endif
#ifdef BOUNDARY_FRONTIER_COST_PROFILE
        const QuarticSturmCertificate cert=[&] {
            V2ProfileTimer timer(&V2Profile::dynamic_count_ms);
            if constexpr(PlainValue) {
                if(adaptive_cfg && adaptive_cfg->reuse_frontier_count_kernel) {
                    QuarticSturmCertificate result;
                    result.root_count=boundary_quartic_count(pc.p);
                    result.certified=result.root_count>=0;
                    return result;
                }
            }
            return certify_quartic(pc.p);
        }();
#else
        const QuarticSturmCertificate cert=[&] {
            if constexpr(PlainValue) {
                if(adaptive_cfg && adaptive_cfg->reuse_frontier_count_kernel) {
                    QuarticSturmCertificate result;
                    result.root_count=boundary_quartic_count(pc.p);
                    result.certified=result.root_count>=0;
                    return result;
                }
            }
            return certify_quartic(pc.p);
        }();
#endif
        if(!cert.certified || cert.root_count<0 || (cert.root_count&1))
            return reject(AdaptiveSampleRejectReason::DegenerateChart);
        if(cert.root_count==0) {
            const double theta=0.371;
            if(phi_val(R,theta,pf)>0.0) {
                auto sample=mapped_full_circle(R,jac,p,u,pf,with_jac,adaptive_cfg);
                sample.local_crossings=-2;return sample;
            }
            out.local_crossings=0;
            return out;
        }
        CellPlan local=cell;
        local.kind=ArcKind::kArcs;
        local.n_crossings=cert.root_count;
        // Atlas and same-node angular upgrades have cell-dependent entry
        // hooks; retain their existing dispatch. Ordinary frontier nodes can
        // continue with the certified local cell without reinitializing a
        // second sample or repeating the common entry checks.
#ifdef BOUNDARY_ADAPTIVE_LEGACY_NODE_DISPATCH
        const bool retain_recursive_dispatch=true;
#else
        const bool retain_recursive_dispatch=(!PlainValue && active_atlas_samples) || angular_mode_requested;
#endif
        if(retain_recursive_dispatch) {
            auto sample=mapped_radius<PlainValue,Uniform>(R,jac,p,u,pf,local,with_jac,warm,seed,
                                 adaptive_cfg,radius_lo,request_radial_hermite_jet,
                                 hermite_jet_out,&pc);
            sample.local_crossings=cert.root_count;
            return sample;
        }
        node_pc=pc;prepared_pc=&node_pc;
        node_crossings=cert.root_count;node_kind=ArcKind::kArcs;
        out.local_crossings=cert.root_count;
    }
    // D14/cell classification certifies that a full-circle cell has no
    // boundary crossing throughout its open radial interval.  Do not redo a
    // quartic solve at every Fejer node; the periodic integrand is smooth and
    // receives its own nested error estimate below.
    if(node_kind==ArcKind::kFull)
        return mapped_full_circle(R,jac,p,u,pf,with_jac,adaptive_cfg);
    if(node_kind==ArcKind::kEmpty)return out;
    if(node_kind==ArcKind::kDegenerate)
        return reject(AdaptiveSampleRejectReason::DegenerateChart);
    if(seed){out.quartic=seed->quartic;out.roots=seed->roots;}
    out.roots.certify=warm;
    if(adaptive_cfg)out.quartic.seed_tolerance=adaptive_cfg->warm_root_seed_tolerance;
    QuarticCoeffs arc_pc{};
    auto arcs=adaptive_arc_intervals(R,pf,&out.quartic,&out.roots,&arc_pc,
                                     node_crossings,prepared_pc);
    if(arcs.kind==ArcKind::kDegenerate)
        return reject(AdaptiveSampleRejectReason::DegenerateChart);
    if(arcs.kind!=node_kind)
        return reject(AdaptiveSampleRejectReason::ArcKindMismatch);
    if(arcs.kind==ArcKind::kArcs &&
       int(arcs.arcs.size()*2)!=node_crossings)
        return reject(AdaptiveSampleRejectReason::RootContinuationMismatch);
    const double D=kPi*p.rho*p.rho*(1-u/3),scale=jac/D;
    double f0=0,fh=0,ef0=0,efh=0;
    const bool radial_hermite_requested=request_jet;
    if(radial_hermite_requested)hermite_jet_out->attempted=true;
    double radial_jet_elapsed=0;
    adaptive_detail::Sum width_R_sum,weighted_width_R_sum;
    bool width_R_finite=true,weighted_width_R_finite=true;
    bool radial_jet_ok=radial_hermite_requested && arcs.kind==ArcKind::kArcs;
    QuarticCoeffs pc_R{};
    if(radial_hermite_requested) {
#ifdef BOUNDARY_ADAPTIVE_HERMITE_MICROTIMING
        const auto radial_jet_start=adaptive_detail::Clock::now();
#endif
        pc_R=boundary_quartic_dR(R,pf);
#ifdef BOUNDARY_ADAPTIVE_HERMITE_MICROTIMING
        radial_jet_elapsed+=adaptive_detail::ms(radial_jet_start);
#endif
    }
    std::array<double,5> df0{},dfh{},edf0{},edfh{};
    const auto pc=u!=0?arc_pc:QuarticCoeffs{};
    QuarticParamJac dpc{};if(with_jac) {
        dpc=boundary_quartic_dp(R,pf);
        if(const auto* motion=fused_gradient::active_motion) {
            const auto radial=boundary_quartic_dR(R,pf);
            for(int j=0;j<5;++j)for(int k=0;k<5;++k)dpc.dp[j][k]+=radial.p[k]*motion->radius[j];
        }
    }
    // The K-rule is a one-arc representation.  Keep the existing path for
    // multi-arc samples; this prevents a failed experimental chart/order on
    // one arc from forcing a second reconstruction of earlier arcs.  The
    // radial sample, root pair, endpoint data and the aggregate from the
    // previous order are still reused for the common one-arc case.
    const bool seed_angular_upgrade=angular_deferred_requested && seed &&
        seed->angular_valid && seed->angular_state_valid && seed->R==R &&
        seed->jac==jac &&
        ((!angular_nested_requested && !seed->angular_nested) ||
         (angular_nested_requested && seed->angular_nested &&
          seed->angular_nested_cache));
    bool use_angular_research=(angular_research_requested ||
                               seed_angular_upgrade) &&
                              arcs.arcs.size()==1;
    const bool use_nested_angular=use_angular_research &&
                                  adaptive_cfg->angular_nested;
    const bool angular_upgrade=use_angular_research && seed &&
        seed->angular_valid && seed->angular_order>0 && seed->R==R &&
        seed->jac==jac &&
        ((!use_nested_angular && !seed->angular_nested) ||
         (use_nested_angular && seed->angular_nested &&
          seed->angular_nested_cache));
    int angular_order=use_angular_research
        ? (angular_upgrade
            ? (use_nested_angular
                ? adaptive_detail::next_nested_angular_order(seed->angular_order)
                : (angular_deferred_requested && seed->angular_order==16
                    ? 32 : adaptive_detail::next_angular_order(seed->angular_order)))
            : (use_nested_angular
                ? adaptive_cfg->angular_nested_initial_order
                : adaptive_cfg->angular_initial_order))
        : 0;
    std::array<double,6> angular_low{},angular_high{};
    double angular_m=0,angular_v=0;
    std::array<double,5> angular_dm{},angular_dv{};
    std::array<double,5> angular_pc{};
    std::array<std::array<double,5>,5> angular_dpc{};
    bool angular_reciprocal=false,angular_state_valid=false;
    bool angular_path_used=false;
    bool angular_error_available=true;
    bool angular_nested_used=false;
    std::shared_ptr<NestedAngularCache> angular_nested_cache;
    size_t angular_point_evaluations=0,angular_cached_point_reuses=0;
    size_t angular_work=0;
    for(auto a:arcs.arcs) {
#ifdef BOUNDARY_FRONTIER_COST_PROFILE
        V2ProfileTimer arc_body_timer(&V2Profile::mapped_arc_body_ms);
#endif
#if !defined(BOUNDARY_ADAPTIVE_DISABLE_ENDPOINT_CACHE)
        PhiValDtheta endpoint_e{},endpoint_l{};
        // For nonnegative normalized LD value-only integration, angular endpoint
        // boundary-strip sensitivity is at most the uniform-brightness term.
        // Keep residuals in the existing geometry ledger; gradients retain
        // the original endpoint solver. This is a residual estimate, not a bound.
        const double endpoint_tolerance=adaptive_cfg && u>=0.0 && u<=1.0 && (!with_jac || adaptive_cfg->fused_value_mesh)
            ? adaptive_cfg->uniform_endpoint_budget_fraction*adaptive_cfg->tol.mu_rtol*(a[1]-a[0]) : 0.0;
        auto pe=polish_endpoint(R,a[0],pf,6,&endpoint_e,endpoint_tolerance),
             pl=polish_endpoint(R,a[1],pf,6,&endpoint_l,endpoint_tolerance);
#else
        auto pe=polish_endpoint(R,a[0],pf),pl=polish_endpoint(R,a[1],pf);
#endif
        double te=pe.theta,tl=pl.theta;if(tl<=te)tl+=kTwoPi;
        // No flag override: transformed arithmetic still requires resolvable endpoints.
        if(!pe.reliable||!pl.reliable)
            return reject(AdaptiveSampleRejectReason::EndpointUnreliable);
        PhiGrad ge{},gl{};
        if(with_jac){ge=phi_grad(R,te,pf);gl=phi_grad(R,tl,pf);}
        else{
#if !defined(BOUNDARY_ADAPTIVE_DISABLE_ENDPOINT_CACHE)
            auto e=endpoint_e,l=tl==pl.theta?endpoint_l:phi_val_dtheta(R,tl,pf);
#else
            auto e=phi_val_dtheta(R,te,pf),l=phi_val_dtheta(R,tl,pf);
#endif
            ge.phi=e.phi;ge.dphi_dtheta=e.dphi_dtheta;gl.phi=l.phi;gl.dphi_dtheta=l.dphi_dtheta;}
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
        if(radial_hermite_requested) {
#ifdef BOUNDARY_ADAPTIVE_HERMITE_MICROTIMING
            const auto radial_jet_start=adaptive_detail::Clock::now();
#endif
            bool pair_ok=false;
            const std::array<double,5> zero{};
            for(bool reciprocal:{false,true}) {
                const auto ap=reciprocal
                    ? arc_pair_jac_reciprocal(te,tl,zero,zero)
                    : arc_pair_jac(te,tl,zero,zero);
                if(!ap.ok || !(ap.v>0) || !std::isfinite(ap.v))continue;
                const RootPair rp{ap.m,ap.v};
                const double pair_width=rp.delta_theta();
                if(std::fabs(pair_width-width)>2e-9*(1+width))continue;
                const QuarticCoeffs coeff=reciprocal
                    ? boundary_quartic_reciprocal(arc_pc) : arc_pc;
                const QuarticCoeffs coeff_R=reciprocal
                    ? boundary_quartic_reciprocal(pc_R) : pc_R;
                const auto dr=root_pair_dR(rp,coeff.p,coeff_R.p);
                if(!dr.ok || !std::isfinite(dr.dm_dR) ||
                   !std::isfinite(dr.dv_dR))continue;
                const double sv=std::sqrt(ap.v);
                const double den=(1+ap.m*ap.m-ap.v)*(1+ap.m*ap.m-ap.v)+4*ap.v;
                if(!(sv>0) || !(den>0) || !std::isfinite(den))continue;
                const double numerator=2*((1+ap.m*ap.m+ap.v)*dr.dv_dR-
                                          4*ap.m*ap.v*dr.dm_dR);
                const double wr=numerator/(sv*den);
                if(std::isfinite(wr))width_R_sum.add(wr);
                else width_R_finite=false;
                const double wr_j2=hermite_scaled_product_ratio(
                    std::array<double,4>{{R,numerator,jac,jac}},
                    std::array<double,3>{{D,sv,den}});
                if(std::isfinite(wr_j2))weighted_width_R_sum.add(wr_j2);
                else weighted_width_R_finite=false;
                pair_ok=std::isfinite(wr_j2);if(pair_ok)break;
            }
            radial_jet_ok=radial_jet_ok&&pair_ok;
#ifdef BOUNDARY_ADAPTIVE_HERMITE_MICROTIMING
            radial_jet_elapsed+=adaptive_detail::ms(radial_jet_start);
#endif
        }
        std::array<double,5> dte{},dtl{};
        if(with_jac)for(int j=0;j<5;++j){
            // Form the mapped derivative before division by the small fold
            // derivative: never construct the divergent fixed-R dtheta first.
            dte[j]=-(jac*ge.dP[j])/ge.dphi_dtheta;
            dtl[j]=-(jac*gl.dP[j])/gl.dphi_dtheta;
            df0[j]+=R*(dtl[j]-dte[j]);
            if(const auto* motion=fused_gradient::active_motion)
                df0[j]+=jac*motion->radius[j]*width;
            edf0[j]+=R*(std::fabs(dte[j])+std::fabs(dtl[j]))*((de+dl)/width+eps);
        }
        if(u==0)continue;
        if(use_angular_research) {
            // Try the existing t chart and then the reciprocal chart.  A
            // requested angular order is evaluated exactly once.  On an
            // upgrade, the lower order is read from the cached aggregate;
            // it is never recomputed here.  The nested branch additionally
            // reuses every old angular point at the same R.
            bool arc_ok=false;
            for(bool reciprocal:{false,true}) {
                // A same-R upgrade must stay in the representation that
                // produced the cached point values.  A chart switch starts a
                // fresh cache; reinterpreting t-chart values as reciprocal
                // values would be a silent numerical corruption.
                if(angular_upgrade && reciprocal!=seed->angular_reciprocal)
                    continue;
                const auto ap=reciprocal
                    ? arc_pair_jac_reciprocal(te,tl,
                                              std::array<double,5>{},
                                              std::array<double,5>{})
                    : arc_pair_jac(te,tl,std::array<double,5>{},
                                   std::array<double,5>{});
                if(!ap.ok)continue;
                auto cp=reciprocal?boundary_quartic_reciprocal(pc):pc;
                    if(use_nested_angular) {
                    std::shared_ptr<NestedAngularCache> candidate_cache;
                    size_t candidate_points=0,candidate_cached=0;
                    if(angular_upgrade)
                        // The cache is attached to the sample that is about
                        // to be replaced.  Reuse it in place: its points
                        // live on the common denominator-512 lattice, so an
                        // upgrade only fills previously unused slots and
                        // cannot invalidate the old aggregate on failure.
                        candidate_cache=seed->angular_nested_cache;
                    else
                        candidate_cache=std::make_shared<NestedAngularCache>();
                    bool nested_ok=false;
                    if(with_jac) {
                        auto aj=ap;
                        bool regular=true;
                        const auto dp=reciprocal
                            ? boundary_quartic_reciprocal_dp(dpc) : dpc;
                        for(int j=0;j<5;++j) {
                            const auto d=root_pair_dR(RootPair{aj.m,aj.v},
                                                      cp.p,dp.dp[j]);
                            regular=regular&&d.ok&&std::isfinite(d.dm_dR)&&
                                     std::isfinite(d.dv_dR);
                            aj.dm[j]=d.dm_dR;aj.dv[j]=d.dv_dR;
                        }
                        if(!regular)continue;
                        nested_angular_attempted=true;
                        const auto nr=v_times_K_nested_jac(
                            aj.m,aj.v,aj.dm,aj.dv,R,pf,cp.p,dp.dp,
                            angular_upgrade?seed->angular_order:0,
                            *candidate_cache,reciprocal,angular_order);
                        if(!nr.ok)continue;
                        angular_high[0]=nr.high.vK;
                        for(int j=0;j<5;++j)
                            angular_high[j+1]=jac*nr.high.dvK[j];
                        if(angular_upgrade) {
                            angular_low=seed->angular_data().result;
                        } else {
                            angular_error_available=nr.low_available;
                            if(angular_error_available) {
                                angular_low[0]=nr.low.vK;
                                for(int j=0;j<5;++j)
                                    angular_low[j+1]=jac*nr.low.dvK[j];
                            }
                        }
                        angular_m=aj.m;angular_v=aj.v;
                        angular_dm=aj.dm;angular_dv=aj.dv;
                        angular_pc=cp.p;angular_dpc=dp.dp;
                        angular_reciprocal=reciprocal;
                        angular_state_valid=true;
                        candidate_points=nr.point_evaluations;
                        candidate_cached=nr.cached_point_reuses;
                        nested_ok=true;
                    } else {
                        nested_angular_attempted=true;
                        const auto nr=v_times_K_nested_value(
                            ap.m,ap.v,R,pf,cp.p,
                            angular_upgrade?seed->angular_order:0,
                            *candidate_cache,reciprocal,angular_order);
                        if(!nr.ok)continue;
                        angular_high[0]=nr.high.vK;
                        if(angular_upgrade)angular_low=seed->angular_data().result;
                        else if(nr.low_available)angular_low[0]=nr.low.vK;
                        else angular_error_available=false;
                        angular_m=ap.m;angular_v=ap.v;
                        angular_pc=cp.p;
                        angular_reciprocal=reciprocal;
                        angular_state_valid=true;
                        candidate_points=nr.point_evaluations;
                        candidate_cached=nr.cached_point_reuses;
                        nested_ok=true;
                    }
                    if(nested_ok) {
                        angular_nested_cache=std::move(candidate_cache);
                        angular_point_evaluations=candidate_points;
                        angular_cached_point_reuses=candidate_cached;
                        angular_work+=candidate_points;
                        angular_nested_used=true;
                        arc_ok=true;
                        break;
                    }
                    continue;
                }
                if(with_jac) {
                    auto aj=ap;
                    bool regular=true;
                    const auto dp=reciprocal
                        ? boundary_quartic_reciprocal_dp(dpc) : dpc;
                    for(int j=0;j<5;++j) {
                        const auto d=root_pair_dR(RootPair{aj.m,aj.v},
                                                  cp.p,dp.dp[j]);
                        regular=regular&&d.ok&&std::isfinite(d.dm_dR)&&
                                 std::isfinite(d.dv_dR);
                        aj.dm[j]=d.dm_dR;aj.dv[j]=d.dv_dR;
                    }
                    if(!regular)continue;
                    const auto hi=v_times_K_jac_order(
                        aj.m,aj.v,aj.dm,aj.dv,R,pf,cp.p,dp.dp,
                        reciprocal,angular_order);
                    if(!hi.ok)continue;
                    angular_high[0]=hi.vK;
                    for(int j=0;j<5;++j)angular_high[j+1]=jac*hi.dvK[j];
                    if(!angular_upgrade) {
                        const auto lo=v_times_K_jac_order(
                            aj.m,aj.v,aj.dm,aj.dv,R,pf,cp.p,dp.dp,
                            reciprocal,8);
                        if(!lo.ok)continue;
                        angular_low[0]=lo.vK;
                        for(int j=0;j<5;++j)angular_low[j+1]=jac*lo.dvK[j];
                        angular_work+=static_cast<size_t>(angular_order)+8;
                    } else {
                        angular_low=seed->angular_data().result;
                        angular_work+=static_cast<size_t>(angular_order);
                    }
                    angular_m=aj.m;angular_v=aj.v;
                    angular_dm=aj.dm;angular_dv=aj.dv;
                    angular_pc=cp.p;angular_dpc=dp.dp;
                    angular_reciprocal=reciprocal;
                    angular_state_valid=true;
                } else {
                    const auto hi=v_times_K_order(ap.m,ap.v,R,pf,cp.p,
                                                   reciprocal,angular_order);
                    if(!hi.ok)continue;
                    angular_high[0]=hi.vK;
                    if(!angular_upgrade) {
                        const auto lo=v_times_K_order(ap.m,ap.v,R,pf,cp.p,
                                                      reciprocal,8);
                        if(!lo.ok)continue;
                        angular_low[0]=lo.vK;
                        angular_work+=static_cast<size_t>(angular_order)+8;
                    } else {
                        angular_low=seed->angular_data().result;
                        angular_work+=static_cast<size_t>(angular_order);
                    }
                    angular_m=ap.m;angular_v=ap.v;
                    angular_pc=cp.p;angular_reciprocal=reciprocal;
                    angular_state_valid=true;
                }
                arc_ok=true;break;
            }
            if(arc_ok) {
                angular_path_used=true;
                fh=angular_high[0];
                if(angular_error_available) {
                    efh=std::fabs(angular_high[0]-angular_low[0]);
                    if(with_jac)for(int j=0;j<5;++j) {
                        dfh[j]=angular_high[j+1];
                        edfh[j]=std::fabs(angular_high[j+1]-angular_low[j+1]);
                    }
                } else {
                    efh=kAdaptiveUnresolvedError;
                    edfh.fill(kAdaptiveUnresolvedError);
                    if(with_jac)for(int j=0;j<5;++j)
                        dfh[j]=angular_high[j+1];
                }
                continue;
            }
            // The ordinary V2 path remains the fail-closed backstop.  Since
            // this branch is restricted to one arc, no partial angular
            // aggregate has to be rolled back when the experimental order
            // cannot be evaluated.
            use_angular_research=false;
        }
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
                const auto tangents=fused_gradient::root_pair_tangents(RootPair{ap.m,ap.v},cp.p,dp.dp);
                for(int j=0;j<5;++j) {
                    const auto& d=tangents[j];
                    regular=regular&&d.ok&&std::isfinite(d.dm_dR)&&std::isfinite(d.dv_dR);
                    ap.dm[j]=d.dm_dR;ap.dv[j]=d.dv_dR;
                }
                if(!regular)continue;
#ifdef BOUNDARY_ADAPTIVE_KJAC_24_DIAGNOSTIC
                // Shadow-only derivative quadrature A/B.  Keep the incumbent
                // 16-node value exactly; use 24 nodes only for the analytic
                // derivative and compare its local estimate against 16.
                auto hi=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal);
                auto dhi=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal,24);
                auto lo=v_times_K_jac(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal,16);
                if(hi.ok&&dhi.ok&&lo.ok){
                    fh+=hi.vK;efh+=std::fabs(hi.vK-lo.vK);
                    for(int j=0;j<5;++j){dfh[j]+=jac*dhi.dvK[j];edfh[j]+=jac*std::fabs(dhi.dvK[j]-hi.dvK[j]);}
                    success=true;
                }
#else
                const double vg=adaptive_cfg?0.05*adaptive_cfg->tol.mu_rtol:kAngularKRelTol;
                double gg=vg;
                if(adaptive_cfg)gg=0.05*(*std::min_element(adaptive_cfg->tol.grad_rtol.begin(),adaptive_cfg->tol.grad_rtol.end()));
                if(adaptive_cfg && adaptive_cfg->fused_value_mesh)gg=std::numeric_limits<double>::infinity();
                auto coupled=fused_gradient::angular_pair(ap.m,ap.v,ap.dm,ap.dv,R,pf,cp.p,dp.dp,reciprocal,vg,gg);
                auto hi=coupled.high,lo=coupled.low;
                if(coupled.ok){
                    if(lo.ok){
                        fh+=hi.vK;efh+=2*std::fabs(hi.vK-lo.vK);
                        for(int j=0;j<5;++j){
                            dfh[j]+=jac*hi.dvK[j];
                            edfh[j]+=2*jac*std::fabs(hi.dvK[j]-lo.dvK[j]);
                        }
                        // Deferred gradient mode exposes the incumbent
                        // 8/16 state so a later same-R angular upgrade can
                        // reuse the already computed geometry and K-rule
                        // aggregate.  No extra kernel work is introduced on
                        // this initial path.
                        if(angular_deferred_requested && arcs.arcs.size()==1) {
                            angular_path_used=true;
                            angular_order=16;
                            angular_high[0]=hi.vK;
                            angular_low[0]=lo.vK;
                            for(int j=0;j<5;++j) {
                                angular_high[j+1]=jac*hi.dvK[j];
                                angular_low[j+1]=jac*lo.dvK[j];
                            }
                            angular_m=ap.m;angular_v=ap.v;
                            angular_dm=ap.dm;angular_dv=ap.dv;
                            angular_pc=cp.p;angular_dpc=dp.dp;
                            angular_reciprocal=reciprocal;
                            angular_state_valid=true;
                            angular_work+=24;
                        }
                        success=true;
                    }}
#endif
            }else{
                // All linear-LD contributions are nonnegative for 0<=u<=1.
                // Reserve a fraction of the requested relative flux budget
                // for the angular difference; keep its actual error in the
                // global ledger. Geometry/chart gates remain unchanged.
                const double gate=adaptive_cfg && u>=0 && u<=1
                    ? std::max(kAngularKRelTol,adaptive_cfg->angular_value_budget_fraction*
                               adaptive_cfg->tol.mu_rtol) : kAngularKRelTol;
                auto vk=gate>kAngularKRelTol
                    ? v_times_K_budgeted(ap.m,ap.v,R,pf,cp.p,reciprocal,gate)
                    : v_times_K(ap.m,ap.v,R,pf,cp.p,reciprocal);if(vk.ok){fh+=vk.vK;efh+=vk.estimated_error;success=true;}}
            if(success)break;
        }
        if(!success){
#ifdef BOUNDARY_FRONTIER_COST_PROFILE
            V2ProfileTimer rescue_timer(&V2Profile::angular_rescue_ms);
            if(auto* prof=v2_profile_current()) {
                ++prof->angular_rescue_calls;prof->angular_rescue_nodes+=96;
            }
#endif
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
            if(with_jac)if(const auto* motion=fused_gradient::active_motion)
                for(int j=0;j<5;++j)dfh[j]+=jac*motion->radius[j]*half*vh;
            if(with_jac)for(int j=0;j<5;++j){dfh[j]+=R*half*dh[j];edfh[j]+=R*half*std::fabs(dh[j]-dd[j]);}
        }
    }
#ifdef BOUNDARY_FRONTIER_COST_PROFILE
    V2ProfileTimer mapped_tail_timer(&V2Profile::mapped_tail_ms);
#endif
    if(angular_path_used) {
        out.angular_data().result=angular_high;
        if(angular_error_available) {
            for(int j=0;j<6;++j)
                out.angular_data().error[j]=std::fabs(angular_high[j]-angular_low[j]);
        } else {
            out.angular_data().error.fill(kAdaptiveUnresolvedError);
        }
        out.angular_order=angular_order;
        out.angular_work=angular_work;
        out.angular_valid=true;
        out.angular_m=angular_m;out.angular_v=angular_v;
        out.angular_data().dm=angular_dm;out.angular_data().dv=angular_dv;
        out.angular_data().pc=angular_pc;out.angular_data().dpc=angular_dpc;
        out.angular_reciprocal=angular_reciprocal;
        out.angular_state_valid=angular_state_valid;
        out.angular_nested_cache=std::move(angular_nested_cache);
        out.angular_nested=angular_nested_used;
        out.angular_error_available=angular_error_available;
        out.angular_point_evaluations=angular_point_evaluations;
        out.angular_cached_point_reuses=angular_cached_point_reuses;
    }
    out.angular_nested_attempted=nested_angular_attempted;
    out.value[0]=scale*((1-u)*f0+u*fh);
    if(adaptive_cfg && adaptive_cfg->split_ld_radial) {
        out.ld_radial_value=scale*u*fh;out.uniform_radial_value=scale*(1-u)*f0;
    }
    if(radial_hermite_requested) {
        hermite_jet_out->elapsed_ms=radial_jet_elapsed;
        if(radial_jet_ok) {
            hermite_jet_out->F_over_norm=f0/D;
            const double base_width_j2=hermite_scaled_product_ratio(
                std::array<double,3>{{f0/R,jac,jac}},std::array<double,1>{{D}});
            const double radial_width_j2=base_width_j2+weighted_width_R_sum.get();
            hermite_jet_out->FR_J2_over_norm=radial_width_j2;
            hermite_jet_out->radial_jet=std::isfinite(hermite_jet_out->F_over_norm)&&
                                         weighted_width_R_finite&&
                                         std::isfinite(base_width_j2)&&
                                         std::isfinite(radial_width_j2);
            if(width_R_finite) {
                const double fr=(f0/R+R*width_R_sum.get())/D;
                if(std::isfinite(fr)) {
                    hermite_jet_out->FR_over_norm=fr;
                    hermite_jet_out->fixed_r_derivative_finite=true;
                }
            }
        }
    }
    if(angular_error_available) {
        const double inner=std::fabs(scale*u)*efh;
        out.inner[0]=std::isfinite(inner)?inner:kAdaptiveUnresolvedError;
    } else {
        out.inner[0]=kAdaptiveUnresolvedError;
    }
    out.geometry[0]=std::fabs(scale*(1-u))*ef0+std::fabs(scale*u)*ef0;
    out.roundoff[0]=eps*std::fabs(scale)*(std::fabs((1-u)*f0)+std::fabs(u*fh));
    if(with_jac){
        if(const auto* motion=fused_gradient::active_motion)for(int j=0;j<5;++j) {
            df0[j]+=motion->jacobian[j]*f0;
            dfh[j]+=motion->jacobian[j]*fh;
        }
        auto d0=internal_to_user_jac(df0,p),dh=internal_to_user_jac(dfh,p);
        if(adaptive_cfg && adaptive_cfg->split_ld_radial)for(int j=0;j<5;++j) {
            out.uniform_radial_grad[j]=(1-u)*d0[j]/D;
            out.ld_radial_grad[j]=u*dh[j]/D;
        }
        if(adaptive_cfg && adaptive_cfg->split_ld_radial) {
            out.uniform_radial_grad[2]-=2*out.uniform_radial_value/p.rho;
            out.ld_radial_grad[2]-=2*out.ld_radial_value/p.rho;
        }
        if(adaptive_cfg && adaptive_cfg->fused_value_mesh) {
            for(int j=0;j<5;++j)out.value[j+1]=((1-u)*d0[j]+u*dh[j])/D;
            out.value[3]-=2*out.value[0]/p.rho;
            return out;
        }
        // Propagate absolute errors through the same linear map without cancellation.
        auto abs_map=[&](const std::array<double,5>& v){std::array<double,5> e{};
            for(int k=0;k<5;++k){std::array<double,5> unit{};unit[k]=1;auto col=internal_to_user_jac(unit,p);for(int j=0;j<5;++j)e[j]+=std::fabs(col[j])*v[k];}return e;};
        auto e0=abs_map(edf0),eh=abs_map(edfh);
        for(int j=0;j<5;++j){out.value[j+1]=((1-u)*d0[j]+u*dh[j])/D;out.inner[j+1]=angular_error_available?std::fabs(u/D)*eh[j]:kAdaptiveUnresolvedError;out.geometry[j+1]=std::fabs((1-u)/D)*e0[j];
            out.roundoff[j+1]=eps/std::fabs(D)*(std::fabs((1-u)*d0[j])+std::fabs(u*dh[j]));}
        out.value[3]-=2*out.value[0]/p.rho;
        if(angular_error_available)out.inner[3]+=2*out.inner[0]/p.rho;
        out.geometry[3]+=2*out.geometry[0]/p.rho;out.roundoff[3]+=2*(out.roundoff[0]+eps*std::fabs(out.value[0]))/p.rho;
    }
    return out;
}
}
namespace lcbinint::algebraic_boundary::tangent {
inline AdaptiveResult flux_adaptive_integrate(const LensParams& p,double u,
    const TopologyResult& topo,const AdaptiveConfig& cfg,AdaptiveWorkspace& workspace,
    bool retain_matching_panels=false,
    std::vector<std::pair<double,adaptive_detail::EventLocation>>* epoch_event_errors=nullptr) {
    // Research resume mode requires identical p, u and cfg. Only the radial
    // partition may change. The default always starts a fresh workspace.
    const ScopedFlushDenormals fp_guard;
    if(!retain_matching_panels)workspace.reset();
    else workspace.cells.clear();
    const auto policy=cfg.effective_gradient_policy();
    if(!adaptive_detail::valid_epoch_config(p,u,cfg)) return adaptive_detail::failure_result(AdaptiveStop::InvalidConfig,policy);
    if(cfg.require_bound) return adaptive_detail::failure_result(AdaptiveStop::BoundUnavailable,policy);
    if(topo.status!=Status::OK) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    auto setup_start=adaptive_detail::Clock::now();
    auto frame_start=adaptive_detail::Clock::now();
    const auto pf=PrimaryFrame::from(p);
#if defined(BOUNDARY_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH)
    TopologyResult projective_topology=topo;
    adaptive_detail::certify_projective_p4_events(projective_topology,pf);
    const TopologyResult& active_topology=projective_topology;
#else
    const TopologyResult& active_topology=topo;
#endif
    const double setup_frame_ms=adaptive_detail::ms(frame_start);
    if(near_origin_source(pf)) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    auto& cells=workspace.cells;
    auto cuts_start=adaptive_detail::Clock::now();
    if(!adaptive_detail::restore_physical_cuts(active_topology,pf,cells)) return adaptive_detail::failure_result(AdaptiveStop::TopologyUnresolved,policy);
    const double setup_cuts_ms=adaptive_detail::ms(cuts_start);
    // A squared map is a valid substitution even if a physical event is a
    // higher contact: we make no smooth-fold guarantee from its label alone.
    // The exact boundary values are copied from topology, so return the
    // metadata record as well as the boolean.  This avoids reconstructing the
    // same D14/stationary-root event in adaptive setup.
    auto physical_event=[&](double R)->const RadialEvent* {
        for(const auto& e:active_topology.events)
            if(e.radius==R && adaptive_detail::adaptive_physical_fold_event(e))
                return &e;
        return nullptr;
    };
    auto physical=[&](double R){return physical_event(R)!=nullptr;};
    std::vector<std::pair<double,adaptive_detail::EventLocation>> local_event_errors;
    auto& event_errors=epoch_event_errors?*epoch_event_errors:local_event_errors;
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
        if(topology_event && (topology_event->projective_fold_certified || topology_event->atlas_anchor || ((topology_event->fold_t_seed_valid || topology_event->fold_u_seed_valid) &&
           adaptive_detail::topology_event_reuse_enabled()))) {
            ++event_topology_reuses;
            ++event_radius_reuses;
            d=adaptive_detail::topology_event_estimate(*topology_event,pf,event_budget,cfg.prepared_fold_pair_seed);
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
            record.u_seed=d.u_seed;record.u_seed_valid=d.u_seed_valid;
            record.seed_reciprocal=d.seed_reciprocal;
            record.projective_fold_certified=topology_event &&
                topology_event->projective_fold_certified;
            record.d14_condition=topology_event
                                    ? topology_event->d14_condition
                                    : std::numeric_limits<double>::infinity();
            record.topology_reused=topology_event &&
                                   (topology_event->projective_fold_certified ||
                                    ((topology_event->fold_t_seed_valid || topology_event->fold_u_seed_valid) &&
                                     adaptive_detail::topology_event_reuse_enabled()));
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
    const auto integration_plan=
        adaptive_detail::make_adaptive_integration_plan(
            active_topology,cells,cfg.initial_level,cfg.fold_maps);
    std::vector<fused_gradient::MaterialMap> material_maps(integration_plan.spans.size());
    const std::size_t previous_panels=retain_matching_panels?workspace.panels.size():0;
    std::vector<bool> previously_active(previous_panels);
    for(std::size_t j=0;j<previous_panels;++j) {
        previously_active[j]=workspace.panels[j].active;
        workspace.panels[j].active=false;
    }
    for(size_t i=0;i<integration_plan.spans.size();++i){
        const auto& span=integration_plan.spans[i];const auto& c=span.cell;
        AdaptivePanel panel;panel.cell=span.first_source_cell;
        panel.cell_end=span.last_source_cell;panel.evaluator_cell=static_cast<int>(i);
        const bool physical_left=cfg.fold_maps&&physical(c.r_lo);
        const bool physical_right=cfg.fold_maps&&physical(c.r_hi);
        panel.map={c.r_lo,c.r_hi,physical_left,physical_right};
        // The ordinary radial map still benefits from the best event estimate
        // as a cell split, but unlike the fold substitution it does not need a
        // certified endpoint uncertainty. Retain the raw topology anchor if
        // refinement is unresolved, and fail closed only when a fold map needs
        // that uncertainty to place its singular endpoint.
        if(physical(c.r_lo)) {
            auto e=uncertainty(c.r_lo);
            if(cfg.fold_maps || std::isfinite(e.radius)) {
                panel.map.a=e.radius;panel.left_radius_lo=e.radius_lo;
                if(cfg.fold_maps)panel.left_uncertainty=e.uncertainty;
            }
        }
        if(physical(c.r_hi)) {
            auto e=uncertainty(c.r_hi);
            if(cfg.fold_maps || std::isfinite(e.radius)) {
                panel.map.b=e.radius;panel.right_radius_lo=e.radius_lo;
                if(cfg.fold_maps)panel.right_uncertainty=e.uncertainty;
            }
        }
        // A local correction may not jump across another cell/event.
        if(std::fabs(panel.map.a-c.r_lo)>.25*(c.r_hi-c.r_lo)||std::fabs(panel.map.b-c.r_hi)>.25*(c.r_hi-c.r_lo)||!(panel.map.a<panel.map.b)) return adaptive_detail::failure_result(AdaptiveStop::EventLocationLimited,policy);
        if(!std::isfinite(panel.left_uncertainty+panel.right_uncertainty)) return adaptive_detail::failure_result(AdaptiveStop::EventLocationLimited,policy);
        const bool empty_left=span.first_source_cell==0 ||
            cells[span.first_source_cell-1].kind==ArcKind::kEmpty;
        const bool empty_right=span.last_source_cell+1==static_cast<int>(cells.size()) ||
            cells[span.last_source_cell+1].kind==ArcKind::kEmpty;
        // Differentiate the radial coordinate map on every span. Artificial
        // partition endpoints are fixed; physical folds follow the implicit phi=0.
        if(policy!=GradientPolicy::None) {
            adaptive_detail::EventLocation left{},right{};
            left.radius=panel.map.a;left.radius_lo=panel.left_radius_lo;
            right.radius=panel.map.b;right.radius_lo=panel.right_radius_lo;
            if(physical(c.r_lo))left=uncertainty(c.r_lo);
            if(physical(c.r_hi))right=uncertainty(c.r_hi);
            material_maps[i]=fused_gradient::make_material_map(
                left,right,pf,!physical(c.r_lo),!physical(c.r_hi));
        }
        bool weighted_candidate=cfg.weighted_radial_spans==nullptr;
        if(cfg.weighted_radial_spans)for(const auto& interval:*cfg.weighted_radial_spans)
            if(interval.first==c.r_lo && interval.second==c.r_hi)weighted_candidate=true;
        if(cfg.weighted_radial && (u==0 || cfg.split_ld_radial) && weighted_candidate &&
           panel.map.left && panel.map.right && empty_left && empty_right) {
            bool material_ok=true;
            if(policy!=GradientPolicy::None) {
                material_ok=material_maps[i].valid;
            }
            if(material_ok) {
                panel.split_ld=u!=0 && cfg.split_ld_radial;
                panel.weighted=true;panel.map.left=panel.map.right=false;
            }
        }
        bool reused=false;
        for(std::size_t j=0;j<previous_panels;++j) {
            auto& old=workspace.panels[j];
            std::size_t root=j;
            while(workspace.panels[root].parent>=0)
                root=static_cast<std::size_t>(workspace.panels[root].parent);
            const auto& ancestor=workspace.panels[root];
            // Match the entire refinement tree. Child panels retain only
            // their relevant endpoint remainder; matching children separately
            // can accidentally keep one half and discard the other.
            if(ancestor.map.a!=panel.map.a || ancestor.map.b!=panel.map.b ||
               ancestor.map.left!=panel.map.left || ancestor.map.right!=panel.map.right ||
               ancestor.weighted!=panel.weighted || ancestor.split_ld!=panel.split_ld ||
               ancestor.left_radius_lo!=panel.left_radius_lo ||
               ancestor.right_radius_lo!=panel.right_radius_lo)continue;
            old.cell=panel.cell;old.cell_end=panel.cell_end;old.evaluator_cell=panel.evaluator_cell;
            for(auto& index:old.samples)
                if(index>=0 && !workspace.samples[static_cast<std::size_t>(index)].reliable)
                    index=-1;
            old.active=previously_active[j];reused=reused||previously_active[j];
        }
        if(!reused) {
            if(workspace.panels.size()>=cfg.max_panels)
                return adaptive_detail::failure_result(AdaptiveStop::BudgetExceeded,policy);
            workspace.panels.push_back(panel);
        }}
    const bool with_jacobian=policy!=GradientPolicy::None;
    // Choose once per integration call. The specialised body retains the
    // identical numerical kernels while compiling away derivative/upgrade
    // machinery and, for uniform sources, all limb-darkening work.
    const bool plain_value_lane=!with_jacobian && !cfg.angular_adaptive &&
        !cfg.gradient_adaptive && !cfg.same_node_hermite_shadow && !active_atlas_samples;
    // Event storage is complete and immutable during this integration call.
    // Index its two boundary states once per span instead of searching the
    // event cache again at every node that may need a fold initializer.
    std::vector<std::array<const adaptive_detail::EventLocation*,2>> span_fold_states;
    if(cfg.prepared_fold_pair_seed) {
        span_fold_states.resize(integration_plan.spans.size(),{nullptr,nullptr});
        for(std::size_t i=0;i<integration_plan.spans.size();++i) {
            const auto& cell=integration_plan.spans[i].cell;
            for(const auto& entry:event_errors)if(entry.second.pair_motion_valid) {
                if(entry.first==cell.r_lo)span_fold_states[i][0]=&entry.second;
                if(entry.first==cell.r_hi)span_fold_states[i][1]=&entry.second;
            }
        }
    }
    const double panel_total_ms=adaptive_detail::ms(panel_start);
    const double setup_panel_ms=std::max(0.0,panel_total_ms-setup_event_ms);
    // Stop the setup clock before adaptive quadrature starts.  Previously the
    // assignment happened after integrate(), so setup_ms double-counted the
    // entire adaptive physics/estimator phase.
    const double setup_total_ms=adaptive_detail::ms(setup_start);
    AdaptiveConfig radial_cfg=cfg;
    if(cfg.weighted_radial) {
        bool weighted_tree=false;
        for(std::size_t i=0;i<workspace.panels.size();++i)if(workspace.panels[i].active) {
            int ancestor=static_cast<int>(i);
            while(ancestor>=0) {
                if(workspace.panels[ancestor].weighted){weighted_tree=true;break;}
                ancestor=workspace.panels[ancestor].parent;
            }
            if(weighted_tree)break;
        }
        // No eligible weighted interval: retain the incumbent controller.
        radial_cfg.weighted_radial=weighted_tree;
    }
    auto evaluate=[&](double R,double jac,int i,const AdaptiveSample* seed,bool force_cold,double radius_lo,int target_level,AdaptiveHermiteJet* hermite_jet){
        if(i<0||static_cast<std::size_t>(i)>=integration_plan.spans.size()) {
            AdaptiveSample invalid;invalid.reliable=false;
            invalid.reject_reason=AdaptiveSampleRejectReason::Nonfinite;
            return invalid;
        }
        const bool node_with_jacobian=with_jacobian && !fused_gradient::defer_sample_tangents;
        const bool node_plain_value_lane=!node_with_jacobian && !cfg.angular_adaptive &&
            !cfg.gradient_adaptive && !cfg.same_node_hermite_shadow && !active_atlas_samples;
        const CellPlan& cell=integration_plan.spans[static_cast<std::size_t>(i)].cell;
        fused_gradient::Motion material_motion;
        const bool material_active=cfg.material_coordinate_motion&&node_with_jacobian&&material_maps[i].valid;
        if(material_active)material_motion=material_maps[i].at(R,radius_lo,jac);
        fused_gradient::MotionScope motion_scope(material_active?&material_motion:nullptr);
        fused_gradient::RadiusOffsetScope radius_scope(radius_lo);
        // The event location was already refined during setup. Use its
        // stationary coordinate to initialize the first cold pair, retaining
        // the node-local root count and all transport/endpoint gates.
#ifdef BOUNDARY_PREPARED_HANDOFF_PROFILE
        V2Profile* handoff_profile=v2_profile_current();
        const auto warm_before=handoff_profile?handoff_profile->rootpair_warm_success:0;
        const auto cold_before=handoff_profile?handoff_profile->rootpair_cold_falls:0;
        const auto predictor_before=handoff_profile?handoff_profile->rootpair_predictor_reject:0;
        const auto newton_before=handoff_profile?handoff_profile->rootpair_newton_reject:0;
        const auto floor_before=handoff_profile?handoff_profile->rootpair_vfloor_reject:0;
        const auto chart_before=handoff_profile?handoff_profile->rootpair_tmax_reject:0;
        const auto branch_before=handoff_profile?handoff_profile->rootpair_branch_reject:0;
        const auto disc_before=handoff_profile?handoff_profile->rootpair_disc_mismatch:0;
        double handoff_dr=0;
#endif
        std::optional<AdaptiveSample> prepared_fold_seed;
        if(cfg.prepared_fold_pair_seed && !force_cold &&
           cell.kind==ArcKind::kArcs &&
           (!seed || (u>0 && seed->roots.valid && seed->roots.pairs.size()==1))) {
            const adaptive_detail::EventLocation* fold=nullptr;
            for(const auto* state:span_fold_states[i])
                if(state && (!fold || std::fabs(R-state->radius)<std::fabs(R-fold->radius)))fold=state;
            bool use_fold=fold && !seed;
            if(fold && seed) {
                const auto& previous=seed->roots;
                const double step=R-previous.R_prev;
                const double previous_distance=std::fabs((previous.R_prev-fold->radius)-fold->radius_lo);
                // Check the costly predictor only for steps whose radial
                // expansion can exceed the incumbent relative-width gate.
                if(std::fabs(step)>kTransportVJumpRel*previous_distance) {
                    const auto& pair=previous.pairs[0];
                    const auto motion=root_pair_dR(pair,previous.pc_prev,previous.pcR_prev);
                    const double predicted_v=pair.v+motion.dv_dR*step;
                    use_fold=motion.ok && std::isfinite(predicted_v) &&
                        (predicted_v<=0 || std::fabs(predicted_v-pair.v)>
                         kTransportVJumpRel*pair.v+kTransportVFloor);
                }
            }
            if(use_fold) {
                const double dr=(R-fold->radius)+(radius_lo-fold->radius_lo);
                const RootPair predicted{fold->pair_m+fold->pair_dm*dr,fold->pair_dv*dr};
#ifdef BOUNDARY_PREPARED_HANDOFF_PROFILE
                handoff_dr=dr/(cell.r_hi-cell.r_lo);
#endif
                if(std::isfinite(predicted.m) && std::isfinite(predicted.v) && predicted.v>0) {
                    auto& roots=prepared_fold_seed.emplace().roots;
                    // Keep the previous full quartic state for the ordinary
                    // fallback if this new pair proposal is not accepted.
                    if(seed)prepared_fold_seed->quartic=seed->quartic;
                    roots.pairs.push_back(predicted);roots.R_prev=R;
                    roots.pc_prev.fill(0.0);roots.pcR_prev.fill(0.0);
                    roots.reciprocal=fold->pair_reciprocal;roots.disc_sign=-1;roots.valid=true;
                    seed=&*prepared_fold_seed;
                }
            }
        }
#ifdef BOUNDARY_ADAPTIVE_FOLD_QUARTIC_SEED
        // Approximate initialization only. The ordinary quartic warm solve,
        // crossing-count check and physical endpoint gates still decide use.
        // A forced cold retry must not repeat this same speculative seed.
        AdaptiveSample fold_seed;
        if(!seed && !force_cold && cell.kind==ArcKind::kArcs) {
            const RadialEvent* event=nullptr;
            for(const auto& e:active_topology.events) {
                if(!adaptive_detail::adaptive_physical_fold_event(e) ||
                   e.projective_fold_certified || !e.fold_t_seed_valid)continue;
                if(e.radius!=cell.r_lo && e.radius!=cell.r_hi)continue;
                if(!event || std::fabs(R-e.radius)<std::fabs(R-event->radius))event=&e;
            }
            if(event) {
                const double m=event->fold_t_seed;
                const auto pc=boundary_quartic(event->radius,pf).p;
                const auto dp=boundary_quartic_dR(event->radius,pf).p;
                const auto motion=root_pair_dR(RootPair{m,0.0},pc,dp);
                const double dr=(R-event->radius)-event->radius_lo;
                const double mp=m+motion.dm_dR*dr, v=motion.dv_dR*dr;
                const double a=pc[4],b=pc[3]+2*m*a,c=pc[2]+2*m*pc[3]+3*m*m*a;
                const double disc=b*b-4*a*c;
                if(motion.ok && v!=0.0 && a!=0.0 && std::isfinite(disc)) {
                    auto& q=fold_seed.quartic;
                    const double gap=std::sqrt(std::fabs(v));
                    q.z[0]={mp+(v>0?gap:0),v<0?gap:0};
                    q.z[1]={mp-(v>0?gap:0),v<0?-gap:0};
                    if(disc>=0) {
                        const double z=-0.5*(b+std::copysign(std::sqrt(disc),b));
                        q.z[2]={z/a,0};q.z[3]={c/z,0};
                    } else {
                        q.z[2]={-b/(2*a),std::sqrt(-disc)/(2*a)};
                        q.z[3]={q.z[2].re,-q.z[2].im};
                    }
                    bool finite=true;
                    for(const auto& z:q.z)finite=finite&&std::isfinite(z.re)&&std::isfinite(z.im);
                    for(int j=0;j<4;++j)for(int k=0;k<j;++k)
                        finite=finite&&(q.z[j].re!=q.z[k].re || q.z[j].im!=q.z[k].im);
                    if(finite) {q.deg=4;q.n_real=cell.n_crossings;q.valid=true;seed=&fold_seed;}
                }
            }
        }
#endif
#ifndef BOUNDARY_ADAPTIVE_GENERIC_VALUE_BODY
        if(node_plain_value_lane) {
            auto fast=u==0.0
                ? adaptive_detail::mapped_radius<true,true>(R,jac,p,u,pf,cell,false,
                    force_cold?false:active_topology.from_warm_d14,
                    force_cold?nullptr:seed,&cfg,radius_lo,false,hermite_jet)
                : adaptive_detail::mapped_radius<true,false>(R,jac,p,u,pf,cell,false,
                    force_cold?false:active_topology.from_warm_d14,
                    force_cold?nullptr:seed,&cfg,radius_lo,false,hermite_jet);
#ifdef BOUNDARY_PREPARED_HANDOFF_PROFILE
            if(!prepared_fold_seed && seed && u>0 && handoff_profile)std::fprintf(stderr,
                "TRANSPORT %.17g %.17g %.17g %d %d %llu %llu %llu %llu %llu %llu %llu %llu\n",
                R,seed->R,(seed->roots.valid && !seed->roots.pairs.empty())?seed->roots.pairs[0].v:0,
                fast.local_crossings,(int)fast.reject_reason,
                (unsigned long long)(handoff_profile->rootpair_warm_success-warm_before),
                (unsigned long long)(handoff_profile->rootpair_cold_falls-cold_before),
                (unsigned long long)(handoff_profile->rootpair_predictor_reject-predictor_before),
                (unsigned long long)(handoff_profile->rootpair_newton_reject-newton_before),
                (unsigned long long)(handoff_profile->rootpair_vfloor_reject-floor_before),
                (unsigned long long)(handoff_profile->rootpair_tmax_reject-chart_before),
                (unsigned long long)(handoff_profile->rootpair_branch_reject-branch_before),
                (unsigned long long)(handoff_profile->rootpair_disc_mismatch-disc_before));
            if(prepared_fold_seed && handoff_profile)std::fprintf(stderr,
                "HANDOFF %.17g %.17g %.17g %.17g %d %d %d %llu %llu %llu %llu %llu %llu %llu %llu\n",
                u,R,handoff_dr,prepared_fold_seed->roots.pairs[0].v,
                target_level,fast.local_crossings,(int)fast.reject_reason,
                (unsigned long long)(handoff_profile->rootpair_warm_success-warm_before),
                (unsigned long long)(handoff_profile->rootpair_cold_falls-cold_before),
                (unsigned long long)(handoff_profile->rootpair_predictor_reject-predictor_before),
                (unsigned long long)(handoff_profile->rootpair_newton_reject-newton_before),
                (unsigned long long)(handoff_profile->rootpair_vfloor_reject-floor_before),
                (unsigned long long)(handoff_profile->rootpair_tmax_reject-chart_before),
                (unsigned long long)(handoff_profile->rootpair_branch_reject-branch_before),
                (unsigned long long)(handoff_profile->rootpair_disc_mismatch-disc_before));
#endif
#ifdef BOUNDARY_ADAPTIVE_VALUE_BODY_AUDIT
            const auto generic=adaptive_detail::mapped_radius<>(R,jac,p,u,pf,cell,false,
                force_cold?false:active_topology.from_warm_d14,
                force_cold?nullptr:seed,&cfg,radius_lo,false,hermite_jet);
            if(fast.value!=generic.value || fast.inner!=generic.inner ||
               fast.geometry!=generic.geometry || fast.roundoff!=generic.roundoff ||
               fast.local_crossings!=generic.local_crossings ||
               fast.reliable!=generic.reliable || fast.reject_reason!=generic.reject_reason)
                std::abort();
#endif
            return fast;
        }
#endif
        return adaptive_detail::mapped_radius(R,jac,p,u,pf,cell,node_with_jacobian,
                                              force_cold?false:active_topology.from_warm_d14,
                                              force_cold?nullptr:seed,&cfg,radius_lo,
                                              cfg.same_node_hermite_shadow &&
                                              cfg.effective_gradient_policy()==GradientPolicy::None &&
                                              !cfg.with_jacobian && u==0.0 && target_level==3,
                                              hermite_jet);
    };
    auto result=adaptive_detail::integrate(workspace,radial_cfg,evaluate);
    bool deferred_children=false;
    for(const auto& panel:workspace.panels)deferred_children|=panel.active && panel.depth>0;
    const bool geometry_special=geometry_experiment::active &&
        !geometry_experiment::active->empty();
    // A value-converged mesh may have no deferred children even when a
    // close-root geometry screen found a gradient-sensitive epoch. Replay
    // its tangents on the existing fixed Gaussian rules in that case too;
    // the trigger is geometric and does not estimate derivative error.
    if(cfg.fused_value_mesh && with_jacobian && result.value_converged &&
       (deferred_children || geometry_special)) {
        const auto tangent=fused_gradient::gaussian_replay(workspace,radial_cfg,evaluate,result.stats.node_evaluations);
        result.grad_mu=tangent.grad_mu;result.grad_quality=tangent.grad_quality;result.grad_reason=tangent.grad_reason;
        result.grad_converged.fill(false);result.gradient_stop_reason=tangent.stop==AdaptiveStop::Converged?AdaptiveStop::BudgetExceeded:tangent.stop;
        result.estimated_abs_error_grad.fill(kAdaptiveUnresolvedError);
        result.stats.node_evaluations+=tangent.stats.node_evaluations;
    }
    result.stats.setup_ms=setup_total_ms;
    result.stats.setup_frame_ms=setup_frame_ms;
    result.stats.setup_cuts_ms=setup_cuts_ms;
    result.stats.setup_event_ms=setup_event_ms;
    result.stats.setup_panel_ms=setup_panel_ms;
    result.stats.integration_spans=integration_plan.spans.size();
    result.stats.softcut_fold_merges=integration_plan.softcut_fold_merges;
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
}

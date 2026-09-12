#pragma once
#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/nested_fejer2.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace lcbinint::holonomic {
using AdaptiveVector = std::array<double,6>;
enum class AccuracyAssurance { None, Estimated, Bounded };
enum class AdaptiveStop { Converged, BudgetExceeded, RoundoffLimited,
    InnerAccuracyLimited, TopologyUnresolved, EventLocationLimited,
    Nonfinite, BoundUnavailable, InvalidConfig };
inline const char* adaptive_stop_name(AdaptiveStop s) {
    switch(s) {
#define S(x) case AdaptiveStop::x:return #x;
    S(Converged) S(BudgetExceeded) S(RoundoffLimited) S(InnerAccuracyLimited)
    S(TopologyUnresolved) S(EventLocationLimited) S(Nonfinite) S(BoundUnavailable) S(InvalidConfig)
#undef S
    } return "Unknown";
}
enum class GradientPolicy { None, Strict, ValueFirst };
enum class RadialErrorEstimator { WeightedDetail, HybridEmbedded };
inline const char* gradient_policy_name(GradientPolicy p) {
    switch (p) {
    case GradientPolicy::None: return "None";
    case GradientPolicy::Strict: return "Strict";
    case GradientPolicy::ValueFirst: return "ValueFirst";
    }
    return "Unknown";
}
enum class GradientQuality { NotRequested, ToleranceMet,
                             FiniteUncertified, Invalid };
inline const char* gradient_quality_name(GradientQuality q) {
    switch (q) {
    case GradientQuality::NotRequested: return "NotRequested";
    case GradientQuality::ToleranceMet: return "ToleranceMet";
    case GradientQuality::FiniteUncertified: return "FiniteUncertified";
    case GradientQuality::Invalid: return "Invalid";
    }
    return "Unknown";
}
enum class GradientReason { NotRequested, ToleranceMet, BudgetExceeded,
    InnerAccuracyLimited, EventLocationLimited, RoundoffLimited,
    TopologyUnresolved, Nonfinite, InvalidConfig, BoundUnavailable };
inline const char* gradient_reason_name(GradientReason r) {
    switch (r) {
    case GradientReason::NotRequested: return "NotRequested";
    case GradientReason::ToleranceMet: return "ToleranceMet";
    case GradientReason::BudgetExceeded: return "BudgetExceeded";
    case GradientReason::InnerAccuracyLimited: return "InnerAccuracyLimited";
    case GradientReason::EventLocationLimited: return "EventLocationLimited";
    case GradientReason::RoundoffLimited: return "RoundoffLimited";
    case GradientReason::TopologyUnresolved: return "TopologyUnresolved";
    case GradientReason::Nonfinite: return "Nonfinite";
    case GradientReason::InvalidConfig: return "InvalidConfig";
    case GradientReason::BoundUnavailable: return "BoundUnavailable";
    }
    return "Unknown";
}
enum class EventDecisionReason { NotAttempted, NoRealCandidate, CleanDouble,
    DoubleBudgetAccepted, DoubleAmbiguous, DDAccepted, DDResidualRejected,
    DDDerivativeRejected, DDBudgetRejected, TopologySeedReused, QfRequired,
    QfRefined, QfFailed };
inline const char* event_decision_reason_name(EventDecisionReason r) {
    switch (r) {
    case EventDecisionReason::NotAttempted: return "NotAttempted";
    case EventDecisionReason::NoRealCandidate: return "NoRealCandidate";
    case EventDecisionReason::CleanDouble: return "CleanDouble";
    case EventDecisionReason::DoubleBudgetAccepted: return "DoubleBudgetAccepted";
    case EventDecisionReason::DoubleAmbiguous: return "DoubleAmbiguous";
    case EventDecisionReason::DDAccepted: return "DDAccepted";
    case EventDecisionReason::DDResidualRejected: return "DDResidualRejected";
    case EventDecisionReason::DDDerivativeRejected: return "DDDerivativeRejected";
    case EventDecisionReason::DDBudgetRejected: return "DDBudgetRejected";
    case EventDecisionReason::TopologySeedReused: return "TopologySeedReused";
    case EventDecisionReason::QfRequired: return "QfRequired";
    case EventDecisionReason::QfRefined: return "QfRefined";
    case EventDecisionReason::QfFailed: return "QfFailed";
    }
    return "Unknown";
}
// Reasons returned by one adaptive mapped-radius sample.  These are kept
// separate from AdaptiveStop::TopologyUnresolved: the latter is the stable
// external stop label, while this enum identifies the local representation or
// continuation failure that caused it.
enum class AdaptiveSampleRejectReason {
    None,
    ArcKindMismatch,
    DegenerateChart,
    EndpointUnreliable,
    ArcWidthUnresolved,
    InnerPhiNonpositive,
    Nonfinite,
    RootContinuationMismatch,
};
inline const char* adaptive_sample_reject_reason_name(
    AdaptiveSampleRejectReason r) {
    switch (r) {
    case AdaptiveSampleRejectReason::None: return "None";
    case AdaptiveSampleRejectReason::ArcKindMismatch: return "ArcKindMismatch";
    case AdaptiveSampleRejectReason::DegenerateChart: return "DegenerateChart";
    case AdaptiveSampleRejectReason::EndpointUnreliable: return "EndpointUnreliable";
    case AdaptiveSampleRejectReason::ArcWidthUnresolved: return "ArcWidthUnresolved";
    case AdaptiveSampleRejectReason::InnerPhiNonpositive: return "InnerPhiNonpositive";
    case AdaptiveSampleRejectReason::Nonfinite: return "Nonfinite";
    case AdaptiveSampleRejectReason::RootContinuationMismatch: return "RootContinuationMismatch";
    }
    return "Unknown";
}
inline constexpr std::size_t adaptive_sample_reject_reason_count = 8;
inline constexpr std::size_t adaptive_sample_reject_reason_index(
    AdaptiveSampleRejectReason r) {
    return static_cast<std::size_t>(r);
}
inline bool adaptive_sample_reject_retryable(AdaptiveSampleRejectReason r) {
    return r == AdaptiveSampleRejectReason::ArcKindMismatch ||
           r == AdaptiveSampleRejectReason::DegenerateChart ||
           r == AdaptiveSampleRejectReason::RootContinuationMismatch;
}
struct AdaptiveTolerance {
    double mu_atol=1e-8,mu_rtol=1e-4;
    std::array<double,5> grad_atol{{1e-6,1e-6,1e-6,1e-6,1e-6}};
    // Gradient tolerances are intentionally a separate contract. The
    // defaults are looser than the value relative tolerance; callers that
    // need a strict derivative contract must set them explicitly.
    std::array<double,5> grad_rtol{{1e-3,1e-3,1e-3,1e-3,1e-3}};
    double budget(int j,double Q) const {
        return std::max(j ? grad_atol[j-1] : mu_atol,
                        (j ? grad_rtol[j-1] : mu_rtol)*std::fabs(Q));
    }
};
struct AdaptiveConfig {
    AdaptiveTolerance tol;
    GradientPolicy gradient_policy=GradientPolicy::None;
    bool with_jacobian=false,require_bound=false,fold_maps=true;
    bool reuse_samples=true; // isolated A/B control; default keeps nested samples
    int initial_level=3,max_level=8,max_depth=12;
    size_t max_node_evals=32768,max_panels=512,max_bytes=64*1024*1024;
    size_t value_first_gradient_node_budget=4096;
    int value_first_gradient_round_budget=4;
    bool collect_diagnostics=false;
    // HybridEmbedded retains a fixed fraction of the weighted interpolation
    // detail while also using the actual embedded Fejer integral difference.
    // The floor prevents an accidentally cancelling embedded difference from
    // hiding unresolved radial structure.
    RadialErrorEstimator radial_error_estimator=RadialErrorEstimator::HybridEmbedded;
    double nested_difference_safety=2.0;
    double weighted_detail_floor_fraction=0.125;
    bool preserve_radial_offset=false; // explicit atlas experiment only
    GradientPolicy effective_gradient_policy() const {
        // Keep the old bool source-compatible for isolated callers. New code
        // should select the policy explicitly.
        return gradient_policy != GradientPolicy::None ? gradient_policy
                                                        : (with_jacobian ? GradientPolicy::Strict
                                                                         : GradientPolicy::None);
    }
};
struct AdaptiveEventDiagnostic {
    double input_radius=0,selected_radius=0,radius_lo=0,uncertainty=std::numeric_limits<double>::infinity();
    double t_seed=0;
    double d14_condition=std::numeric_limits<double>::infinity();
    bool t_seed_valid=false,topology_reused=false;
    int precision_tier=0;
    EventDecisionReason double_reason=EventDecisionReason::NotAttempted;
    EventDecisionReason dd_reason=EventDecisionReason::NotAttempted;
    EventDecisionReason qf_reason=EventDecisionReason::NotAttempted;
    double double_residual=std::numeric_limits<double>::infinity();
    double dd_residual=std::numeric_limits<double>::infinity();
};
struct AdaptiveRefinementRecord {
    size_t panel=0,node_evaluations=0;
    int cell=0,parent=-1,depth=0,level=0;
    double value_error=std::numeric_limits<double>::infinity();
    std::array<double,5> gradient_error{};
    bool gradient_phase=false,value_resolved=false;
    std::array<bool,5> gradient_resolved{};
};
struct AdaptiveSampleDiagnostic {
    std::size_t panel=0,node_slot=0;
    int cell=0,level=0;
    double R=0;
    AdaptiveSampleRejectReason reason=AdaptiveSampleRejectReason::None;
    bool cold_retry=false,cold_retry_succeeded=false;
};
struct AdaptiveStats {
    size_t unique_nodes=0,node_evaluations=0,reused_nodes=0,split_discarded_nodes=0,splits=0;
    size_t root_anchors=0,panels=0;
    size_t event_double_checks=0,event_dd_checks=0,event_dd_accepts=0,event_qf_refinements=0,qf_family_constructions=0;
    size_t event_topology_reuses=0,event_radius_reuses=0,event_direct_qf_failures=0;
    size_t first_value_pass_nodes=0,first_gradient_error_pass_nodes=0,first_gradient_contract_nodes=0;
    unsigned gradient_error_pass_mask=0,gradient_contract_mask=0;
    std::array<size_t,9> level_histogram{};
    std::array<size_t,adaptive_sample_reject_reason_count> sample_reject_counts{};
    std::array<size_t,adaptive_sample_reject_reason_count> sample_cold_reject_counts{};
    size_t sample_cold_retries=0,sample_cold_retry_successes=0;
    double physical_ms=0,estimator_ms=0,scheduler_ms=0,topology_ms=0,setup_ms=0;
    double setup_frame_ms=0,setup_cuts_ms=0,setup_event_ms=0,setup_panel_ms=0;
    std::vector<AdaptiveEventDiagnostic> event_diagnostics;
    std::vector<AdaptiveSampleDiagnostic> sample_diagnostics;
    std::vector<AdaptiveRefinementRecord> refinement_history;
};
struct AdaptiveResult {
    double mu=0,estimated_abs_error_mu=std::numeric_limits<double>::infinity();
    std::array<double,5> grad_mu{},estimated_abs_error_grad{};
    bool value_converged=false;
    std::array<bool,5> grad_converged{};
    AdaptiveStop value_stop_reason=AdaptiveStop::BudgetExceeded;
    AdaptiveStop gradient_stop_reason=AdaptiveStop::BudgetExceeded;
    std::array<GradientQuality,5> grad_quality{{GradientQuality::NotRequested,
                                                 GradientQuality::NotRequested,
                                                 GradientQuality::NotRequested,
                                                 GradientQuality::NotRequested,
                                                 GradientQuality::NotRequested}};
    std::array<GradientReason,5> grad_reason{{GradientReason::NotRequested,
                                               GradientReason::NotRequested,
                                               GradientReason::NotRequested,
                                               GradientReason::NotRequested,
                                               GradientReason::NotRequested}};
    AccuracyAssurance assurance=AccuracyAssurance::None;
    AdaptiveStop stop=AdaptiveStop::BudgetExceeded;
    Status numerical_status=Status::GRADIENT_UNRELIABLE;
    AdaptiveVector radial_error{},inner_error{},geometry_error{},event_error{},roundoff_error{};
    AdaptiveStats stats;
};
struct AdaptiveSample {
    double R=0,R_lo=0;
    bool precise_R=false;
    AdaptiveVector value{},inner{},geometry{},roundoff{};
    QuarticWarm quartic{};
    RootPairWarm roots{};
    std::array<bool,6> component_finite{{true,true,true,true,true,true}};
    bool reliable=true;
    AdaptiveSampleRejectReason reject_reason=AdaptiveSampleRejectReason::None;
};
struct AdaptivePanel {
    FoldRadialMap map;
    double xl=-1,xr=1;
    int cell=0,level=0,depth=0,parent=-1;
    bool active=true,resolved=false,value_resolved=false;
    std::array<bool,5> gradient_resolved{};
    std::array<bool,5> gradient_invalid{};
    double left_uncertainty=0,right_uncertainty=0;
    // Keep the qf event remainder separate from the binary64 event anchor.
    // The current evaluator still receives a double R, but all local map
    // arithmetic is formed from hi+lo before that final interface cast.
    double left_radius_lo=0,right_radius_lo=0;
    // Finest lattice IDs: inherited k doubles on refinement, so slot is fixed.
    std::array<int,256> samples;
    AdaptiveVector q{},error{},radial{},inner{},geometry{},event{},roundoff{};
    AdaptivePanel() { samples.fill(-1);gradient_resolved.fill(false);gradient_invalid.fill(false); }
};
struct AdaptiveWorkspace {
    std::vector<AdaptiveSample> samples;
    std::vector<AdaptivePanel> panels;
    std::vector<CellPlan> cells;
    std::uint64_t generation=0;
    void reset() { samples.clear();panels.clear();cells.clear();++generation; }
};
namespace adaptive_detail {
using Clock=std::chrono::steady_clock;
inline double ms(Clock::time_point a) { return std::chrono::duration<double,std::milli>(Clock::now()-a).count(); }
inline constexpr double eps=std::numeric_limits<double>::epsilon();
// Compensated summation for global and per-panel observables, including rho
// cancellation. The quadrature estimator is a contraction MODEL, not a bound.
struct Sum {
    double s=0,c=0;
    void add(double x) { double t=s+x; c+=std::fabs(s)>=std::fabs(x)?(s-t)+x:(x-t)+s; s=t; }
    double get() const {return s+c;}
};
inline bool valid_config(const AdaptiveConfig& c) {
    if(c.initial_level<3 || c.max_level<c.initial_level || c.max_level>8 || c.max_depth<0) return false;
    for(int j=0;j<6;++j) {
        double a=j?c.tol.grad_atol[j-1]:c.tol.mu_atol;
        double r=j?c.tol.grad_rtol[j-1]:c.tol.mu_rtol;
        if(!std::isfinite(a)||!std::isfinite(r)||a<0||r<0||!(a>0||r>0))return false;
    }
    if(c.effective_gradient_policy()==GradientPolicy::ValueFirst &&
       c.value_first_gradient_round_budget<0) return false;
    if(!std::isfinite(c.nested_difference_safety)||c.nested_difference_safety<1.0)
        return false;
    if(!std::isfinite(c.weighted_detail_floor_fraction)||
       c.weighted_detail_floor_fraction<0.0||c.weighted_detail_floor_fraction>1.0)
        return false;
    return true;
}
inline GradientReason gradient_reason(AdaptiveStop s) {
    switch (s) {
    case AdaptiveStop::Converged: return GradientReason::ToleranceMet;
    case AdaptiveStop::BudgetExceeded: return GradientReason::BudgetExceeded;
    case AdaptiveStop::InnerAccuracyLimited: return GradientReason::InnerAccuracyLimited;
    case AdaptiveStop::EventLocationLimited: return GradientReason::EventLocationLimited;
    case AdaptiveStop::RoundoffLimited: return GradientReason::RoundoffLimited;
    case AdaptiveStop::TopologyUnresolved: return GradientReason::TopologyUnresolved;
    case AdaptiveStop::Nonfinite: return GradientReason::Nonfinite;
    case AdaptiveStop::InvalidConfig: return GradientReason::InvalidConfig;
    case AdaptiveStop::BoundUnavailable: return GradientReason::BoundUnavailable;
    }
    return GradientReason::BudgetExceeded;
}
inline bool invalid_gradient_stop(AdaptiveStop s) {
    return s==AdaptiveStop::TopologyUnresolved || s==AdaptiveStop::Nonfinite ||
           s==AdaptiveStop::InvalidConfig;
}
inline AdaptiveResult failure_result(AdaptiveStop stop,GradientPolicy policy) {
    AdaptiveResult r;
    r.stop=r.value_stop_reason=r.gradient_stop_reason=stop;
    r.numerical_status=Status::GRADIENT_UNRELIABLE;
    if(policy==GradientPolicy::None) {
        for(int j=0;j<5;++j) {
            r.grad_quality[j]=GradientQuality::NotRequested;
            r.grad_reason[j]=GradientReason::NotRequested;
        }
    } else {
        for(int j=0;j<5;++j) {
            r.grad_quality[j]=GradientQuality::Invalid;
            r.grad_reason[j]=gradient_reason(stop);
        }
    }
    return r;
}
inline void estimate(AdaptivePanel& p,const AdaptiveWorkspace& w,int nc,
                     const AdaptiveConfig& cfg) {
    const int m=1<<p.level,step=256/m;
    const auto& rule=fejer_rule(p.level);
    std::array<Sum,6> q,inn,geo,rnd;
    for(int k=1;k<m;++k) {
        const auto& s=w.samples[p.samples[k*step]];
        for(int j=0;j<nc;++j) {
            if(!s.component_finite[j]) {
                if(j>0)p.gradient_invalid[j-1]=true;
                continue;
            }
            double wt=rule.w[k-1];
            q[j].add(wt*s.value[j]); inn[j].add(wt*s.inner[j]);
            geo[j].add(wt*s.geometry[j]);rnd[j].add(wt*s.roundoff[j]+eps*std::fabs(wt*s.value[j]));
        }
    }
    AdaptiveVector detail{},previous{},nested_difference{};
    for(int lev=p.level-1;lev<=p.level;++lev) {
        const int mm=1<<lev,st=256/mm,coarse=mm/2-1;
        const auto& r=fejer_rule(lev);
        AdaptiveVector norm{};
        for(int k=1;k<mm;k+=2) {
            for(int j=0;j<nc;++j) {
                const auto& fine=w.samples[p.samples[k*st]];
                if(!fine.component_finite[j]) {
                    if(j>0)p.gradient_invalid[j-1]=true;
                    continue;
                }
                Sum pred;
                bool valid=true;
                for(int h=1;h<=coarse;++h) {
                    const auto& coarse_sample=w.samples[p.samples[2*h*st]];
                    if(!coarse_sample.component_finite[j]) {valid=false;break;}
                    pred.add(r.interp[(k/2)*coarse+h-1]*coarse_sample.value[j]);
                }
                if(!valid) {
                    if(j>0)p.gradient_invalid[j-1]=true;
                    continue;
                }
                double d=fine.value[j]-pred.get();
                norm[j]+=r.norm_w[k-1]*d*d;
            }
        }
        for(int j=0;j<nc;++j) (lev==p.level?detail:previous)[j]=std::sqrt(3.14159265358979323846*norm[j]);
    }
    if(cfg.radial_error_estimator==RadialErrorEstimator::HybridEmbedded) {
        const int coarse_m=m/2,coarse_step=256/coarse_m;
        const auto& coarse_rule=fejer_rule(p.level-1);
        std::array<Sum,6> coarse_q;
        for(int k=1;k<coarse_m;++k) {
            const auto& s=w.samples[p.samples[k*coarse_step]];
            for(int j=0;j<nc;++j) if(s.component_finite[j])
                coarse_q[j].add(coarse_rule.w[k-1]*s.value[j]);
        }
        for(int j=0;j<nc;++j)
            nested_difference[j]=std::fabs(q[j].get()-coarse_q[j].get());
    }
    p.event.fill(0);
    for(int side=0;side<2;++side) {
        const double delta=side?p.right_uncertainty:p.left_uncertainty;
        if(!(delta>0))continue;
        int k=side?1:m-1;
        const auto& sample=w.samples[p.samples[k*step]];
        const double x=0.5*(p.xl+p.xr)+0.5*(p.xr-p.xl)*rule.x[k-1];
        auto mapped=[&] {
            const long double aa=(long double)p.map.a+p.left_radius_lo;
            const long double bb=(long double)p.map.b+p.right_radius_lo;
            const long double t=(1.0L+x)*0.5L,wid=bb-aa;
            if(p.left_radius_lo==0&&p.right_radius_lo==0)return p.map(x);
            if(p.map.left&&p.map.right) {
                const long double s=std::sin(3.1415926535897932384626433832795029L*0.5L*(t<=0.5L?t:1.0L-t));
                return std::array<double,2>{(double)(t<=0.5L?aa+wid*s*s:bb-wid*s*s),
                                             (double)(3.1415926535897932384626433832795029L*wid*0.25L*std::sin(3.1415926535897932384626433832795029L*t))};
            }
            if(p.map.left)return std::array<double,2>{(double)(aa+wid*t*t),(double)(wid*t)};
            if(p.map.right)return std::array<double,2>{(double)(bb-wid*(1.0L-t)*(1.0L-t)),(double)(wid*(1.0L-t))};
            return std::array<double,2>{(double)(aa+wid*t),(double)(wid*0.5L)};
        }();
        double distance=sample.precise_R?double(side?((__float128)p.map.b+p.right_radius_lo-((__float128)sample.R+sample.R_lo)):((__float128)sample.R+sample.R_lo-((__float128)p.map.a+p.left_radius_lo))):(side?p.map.b-sample.R:sample.R-p.map.a);
        double map_jac=mapped[1]*0.5*(p.xr-p.xl);
        // Local endpoint model, not a verified coefficient envelope. For
        // derivatives use the integrable 1/sqrt(distance) worst case.
        for(int j=0;j<nc;++j) {
            if(!sample.component_finite[j]) {
                if(j>0)p.gradient_invalid[j-1]=true;
                continue;
            }
            p.event[j]+=2*std::fabs(sample.value[j]/map_jac)*(j?std::sqrt(distance*delta):delta);
        }
    }
    p.value_resolved=true;
    p.gradient_resolved.fill(false);
    p.resolved=true;
    for(int j=0;j<nc;++j) {
        p.q[j]=q[j].get();p.inner[j]=inn[j].get();p.geometry[j]=geo[j].get();p.roundoff[j]=rnd[j].get();
        const double floor=p.roundoff[j]+p.inner[j]+p.geometry[j];
        const bool decays=detail[j]<=0.5*previous[j] || detail[j]<=floor;
        if(j==0) p.value_resolved=decays;
        else p.gradient_resolved[j-1]=!p.gradient_invalid[j-1]&&decays;
        // Unresolved panels must refine even if their integral difference cancels.
        // Gradient contracts retain the conservative estimator in this
        // phase.  Only the independently validated primal value uses the
        // embedded integral difference.
        const double radial=j==0&&cfg.radial_error_estimator==RadialErrorEstimator::HybridEmbedded
            ? std::max(cfg.weighted_detail_floor_fraction*detail[j],
                       std::min(detail[j],cfg.nested_difference_safety*nested_difference[j]))
            : detail[j];
        p.radial[j]=radial;p.error[j]=(j>0&&p.gradient_invalid[j-1])
            ? std::numeric_limits<double>::infinity()
            : radial+floor+p.event[j];
    }
    p.resolved=p.value_resolved;
    for(int j=0;j<nc-1;++j) p.resolved=p.resolved&&p.gradient_resolved[j];
}
inline void estimate(AdaptivePanel& p,const AdaptiveWorkspace& w,int nc) {
    estimate(p,w,nc,AdaptiveConfig{});
}
// Callback evaluates a NEW mapped node. Existing sample values never change.
// sample snapshots remain available after a split as same-cell root anchors.
template<class Evaluate>
AdaptiveResult integrate(AdaptiveWorkspace& w,const AdaptiveConfig& cfg,Evaluate eval) {
    AdaptiveResult out;
    const GradientPolicy policy=cfg.effective_gradient_policy();
    const bool gradients=policy!=GradientPolicy::None;
    if(cfg.require_bound){
        out.stop=out.value_stop_reason=out.gradient_stop_reason=AdaptiveStop::BoundUnavailable;
        if(gradients)for(int j=0;j<5;++j){out.grad_quality[j]=GradientQuality::Invalid;out.grad_reason[j]=GradientReason::BoundUnavailable;}
        return out;
    }
    if(!valid_config(cfg)){
        out.stop=out.value_stop_reason=out.gradient_stop_reason=AdaptiveStop::InvalidConfig;
        if(gradients)for(int j=0;j<5;++j){out.grad_quality[j]=GradientQuality::Invalid;out.grad_reason[j]=GradientReason::InvalidConfig;}
        return out;
    }
    const int nc=gradients?6:1;
    AdaptiveStats stats;
    bool gradient_phase=false,invalid=false;
    size_t gradient_start_nodes=0;
    int gradient_rounds=0;
    std::array<bool,5> last_gradient_pass{};
    std::array<bool,5> gradient_invalid_seen{};
    struct ValueSnapshot {
        bool valid=false;
        double q=0,error=std::numeric_limits<double>::infinity();
        std::array<double,5> ledger{};
    } value_snapshot;
    auto allocated_bytes=[&](){return w.samples.capacity()*sizeof(AdaptiveSample)+
        w.panels.capacity()*sizeof(AdaptivePanel);};
    auto invoke_eval=[&](double R,double jac,int cell,const AdaptiveSample* seed,
                         bool force_cold,double radius_lo)->AdaptiveSample {
        // The fifth argument is an opt-in extension for the production
        // adaptive epoch adapter.  Keep the four-argument callback source
        // compatible for the small radial-controller tests and research
        // callers that do not own a warm/cold path.
        if constexpr(std::is_invocable_v<Evaluate,double,double,int,const AdaptiveSample*,bool,double>)
            return eval(R,jac,cell,seed,force_cold,radius_lo);
        else if constexpr(std::is_invocable_v<Evaluate,double,double,int,
                                         const AdaptiveSample*,bool>)
            return eval(R,jac,cell,seed,force_cold);
        else
            return eval(R,jac,cell,seed);
    };
    auto record_sample_reject=[&](std::size_t panel,std::size_t slot,int cell,
                                  int level,double R,
                                  const AdaptiveSample& sample,bool cold_retry,
                                  bool cold_retry_succeeded) {
        const auto reason=sample.reject_reason;
        if(reason==AdaptiveSampleRejectReason::None)return;
        const std::size_t ri=adaptive_sample_reject_reason_index(reason);
        if(ri<stats.sample_reject_counts.size()) {
            if(cold_retry)++stats.sample_cold_reject_counts[ri];
            else ++stats.sample_reject_counts[ri];
        }
        if(cfg.collect_diagnostics) {
            stats.sample_diagnostics.push_back(AdaptiveSampleDiagnostic{
                panel,slot,cell,level,R,reason,cold_retry,cold_retry_succeeded});
        }
    };
    auto refine=[&](size_t ip,int level)->AdaptiveStop {
        auto& p=w.panels[ip];const int m=1<<level,step=256/m;
        size_t needed=0;for(int k=1;k<m;++k)needed+=!cfg.reuse_samples||p.samples[k*step]<0;
        if(stats.node_evaluations+needed>cfg.max_node_evals ||
           (w.samples.size()+needed)*sizeof(AdaptiveSample)+w.panels.size()*sizeof(AdaptivePanel)>cfg.max_bytes)
            return AdaptiveStop::BudgetExceeded;
        if(gradient_phase && policy==GradientPolicy::ValueFirst &&
           (stats.node_evaluations-gradient_start_nodes+needed>cfg.value_first_gradient_node_budget ||
            gradient_rounds>=cfg.value_first_gradient_round_budget))
            return AdaptiveStop::BudgetExceeded;
        const size_t target=w.samples.size()+needed;
        const size_t growth=target>w.samples.capacity()?(target-w.samples.capacity())*sizeof(AdaptiveSample):0;
        // RootPairWarm uses two inline slots, so there is no per-sample heap
        // growth to reserve here; the AdaptiveSample size already includes it.
        if(allocated_bytes()+growth>cfg.max_bytes)return AdaptiveStop::BudgetExceeded;
        if(target>w.samples.capacity())w.samples.reserve(target);
        for(int k=1;k<m;++k) {
            int slot=k*step;
            bool fresh=p.samples[slot]<0;
            if(!fresh&&cfg.reuse_samples){++stats.reused_nodes;continue;}
            const double x=0.5*(p.xl+p.xr)+0.5*(p.xr-p.xl)*fejer_rule(level).x[k-1];
            double radius_lo=0;
            auto mapped=[&] {
                if(cfg.preserve_radial_offset){
                    const long double t=(1.0L+x)*.5L;
                    const D14Real wid=D14Real(p.map.b,p.right_radius_lo)-D14Real(p.map.a,p.left_radius_lo);
                    long double factor=t,J=.5L;bool right=false;
                    constexpr long double pi=3.1415926535897932384626433832795029L;
                    if(p.map.left&&p.map.right){right=t>.5L;long double z=std::sin(pi*.5L*(right?1-t:t));factor=z*z;J=pi*.25L*std::sin(pi*t);}
                    else if(p.map.left){factor=t*t;J=t;}
                    else if(p.map.right){right=true;factor=(1-t)*(1-t);J=1-t;}
                    D14Real f(double(factor),double(factor-(long double)double(factor)));
                    D14Real rr=right?D14Real(p.map.b,p.right_radius_lo)-wid*f:D14Real(p.map.a,p.left_radius_lo)+wid*f;
                    radius_lo=rr.lo;
                    return std::array<double,2>{rr.hi,double((long double)double(wid)*J)};
                }
                const long double aa=(long double)p.map.a+p.left_radius_lo;
                const long double bb=(long double)p.map.b+p.right_radius_lo;
                const long double t=(1.0L+x)*0.5L,wid=bb-aa;
                if(p.left_radius_lo==0&&p.right_radius_lo==0)return p.map(x);
                if(p.map.left&&p.map.right) {
                    const long double s=std::sin(3.1415926535897932384626433832795029L*0.5L*(t<=0.5L?t:1.0L-t));
                    return std::array<double,2>{(double)(t<=0.5L?aa+wid*s*s:bb-wid*s*s),
                                                 (double)(3.1415926535897932384626433832795029L*wid*0.25L*std::sin(3.1415926535897932384626433832795029L*t))};
                }
                if(p.map.left)return std::array<double,2>{(double)(aa+wid*t*t),(double)(wid*t)};
                if(p.map.right)return std::array<double,2>{(double)(bb-wid*(1.0L-t)*(1.0L-t)),(double)(wid*(1.0L-t))};
                return std::array<double,2>{(double)(aa+wid*t),(double)(wid*0.5L)};
            }();
            mapped[1]*=0.5*(p.xr-p.xl);
            if(cfg.preserve_radial_offset){
                const __float128 rr=(__float128)mapped[0]+radius_lo;
                if(!(rr>(__float128)p.map.a+p.left_radius_lo+p.left_uncertainty && rr<(__float128)p.map.b+p.right_radius_lo-p.right_uncertainty))return AdaptiveStop::EventLocationLimited;
            }else if(!(mapped[0]>p.map.a+p.left_uncertainty && mapped[0]<p.map.b-p.right_uncertainty))return AdaptiveStop::EventLocationLimited;
            const AdaptiveSample* anchor=nullptr;double distance=std::numeric_limits<double>::infinity();
            for(int ancestor=int(ip);ancestor>=0;ancestor=w.panels[ancestor].parent)
            for(int idx:w.panels[ancestor].samples) if(idx>=0) {
                double d=std::fabs(w.samples[idx].R-mapped[0]);
                if(d<distance && w.samples[idx].reliable){distance=d;anchor=&w.samples[idx];}
            }
            auto start=Clock::now();
            AdaptiveSample s=invoke_eval(mapped[0],mapped[1],p.cell,anchor,false,radius_lo);
            stats.physical_ms+=ms(start); if(anchor)++stats.root_anchors;
            ++stats.node_evaluations;
            const AdaptiveSampleRejectReason initial_reason=s.reject_reason;
            bool cold_retry_succeeded=false;
            // A warm continuation is an optimization layer.  If it reports a
            // topology/continuation representation failure, retry this same R
            // from fresh quartic/root state once.  A failure from an already
            // cold evaluation is retained and remains fail-closed.
            if(!s.reliable && anchor &&
               adaptive_sample_reject_retryable(initial_reason) &&
               stats.node_evaluations<cfg.max_node_evals) {
                ++stats.sample_cold_retries;
                auto cold_start=Clock::now();
                AdaptiveSample cold=invoke_eval(mapped[0],mapped[1],p.cell,nullptr,true,radius_lo);
                stats.physical_ms+=ms(cold_start);
                ++stats.node_evaluations;
                cold.R=mapped[0];cold.R_lo=radius_lo;cold.precise_R=cfg.preserve_radial_offset;
                cold_retry_succeeded=cold.reliable;
                if(cold_retry_succeeded)++stats.sample_cold_retry_successes;
                s=std::move(cold);
                AdaptiveSample initial_failed;
                initial_failed.reject_reason=initial_reason;
                record_sample_reject(ip,slot,p.cell,level,mapped[0],
                                     initial_failed,false,cold_retry_succeeded);
                if(!s.reliable)record_sample_reject(ip,slot,p.cell,level,
                                                    mapped[0],s,true,false);
            } else if(!s.reliable) {
                record_sample_reject(ip,slot,p.cell,level,mapped[0],s,false,false);
            }
            s.R=mapped[0];s.R_lo=radius_lo;s.precise_R=cfg.preserve_radial_offset;
            p.samples[slot]=int(w.samples.size());w.samples.push_back(std::move(s));
            if(fresh)++stats.unique_nodes;
            auto& stored=w.samples.back();
            bool value_finite=true;
            for(int j=0;j<nc;++j) {
                const bool ok=std::isfinite(stored.value[j])&&std::isfinite(stored.inner[j])&&
                    std::isfinite(stored.geometry[j])&&std::isfinite(stored.roundoff[j])&&
                    stored.inner[j]>=0&&stored.geometry[j]>=0&&stored.roundoff[j]>=0;
                stored.component_finite[j]=ok;
                if(j==0)value_finite=value_finite&&ok;
                else if(!ok) gradient_invalid_seen[j-1]=true;
            }
            if(!value_finite){
                if(stored.reject_reason==AdaptiveSampleRejectReason::None)
                    stored.reject_reason=AdaptiveSampleRejectReason::Nonfinite;
                invalid=true;return AdaptiveStop::Nonfinite;
            }
            if(!stored.reliable){invalid=true;return AdaptiveStop::TopologyUnresolved;}
        }
        p.level=level;auto start=Clock::now();estimate(p,w,nc,cfg);stats.estimator_ms+=ms(start);
        if(cfg.collect_diagnostics) {
            AdaptiveRefinementRecord record;
            record.panel=ip;record.node_evaluations=stats.node_evaluations;
            record.cell=p.cell;record.parent=p.parent;record.depth=p.depth;record.level=p.level;
            record.value_error=p.error[0];record.gradient_phase=gradient_phase;
            record.value_resolved=p.value_resolved;record.gradient_resolved=p.gradient_resolved;
            for(int j=0;j<5;++j)record.gradient_error[j]=nc>j+1?p.error[j+1]:0;
            stats.refinement_history.push_back(record);
        }
        if(gradient_phase)++gradient_rounds;
        return AdaptiveStop::Converged;
    };
    AdaptiveStop failure=AdaptiveStop::Converged;
    if(w.panels.size()>cfg.max_panels){
        out= failure_result(AdaptiveStop::BudgetExceeded,policy);
        return out;
    }
    for(size_t i=0;i<w.panels.size();++i) {failure=refine(i,cfg.initial_level);if(failure!=AdaptiveStop::Converged)break;}
    auto gather=[&](bool* value_pass_out,bool* gradient_pass_out){
        auto start=Clock::now();std::array<Sum,6> total,errors,rad,inn,geo,evt,rnd;
        bool resolved_value=true;
        std::array<bool,5> resolved_gradient{};
        resolved_gradient.fill(true);
        std::array<bool,5> invalid_gradient_local{};
        for(const auto& p:w.panels)if(p.active){
            resolved_value=resolved_value&&p.value_resolved;
            for(int j=0;j<nc;++j){
                total[j].add(p.q[j]);errors[j].add(p.error[j]);rad[j].add(p.radial[j]);inn[j].add(p.inner[j]);geo[j].add(p.geometry[j]);evt[j].add(p.event[j]);rnd[j].add(p.roundoff[j]);
                if(j>0) {
                    resolved_gradient[j-1]=resolved_gradient[j-1]&&p.gradient_resolved[j-1];
                    invalid_gradient_local[j-1]=invalid_gradient_local[j-1]||p.gradient_invalid[j-1];
                }
            }
        }
        bool value_pass=resolved_value;
        std::array<bool,5> gradient_pass{},gradient_error_pass{};
        for(int j=0;j<nc;++j){double Q=total[j].get(),E=errors[j].get();bool pass=E<=cfg.tol.budget(j,Q);
            if(j){out.grad_mu[j-1]=Q;out.estimated_abs_error_grad[j-1]=E;gradient_error_pass[j-1]=pass;gradient_pass[j-1]=pass&&resolved_gradient[j-1]&&!invalid_gradient_local[j-1];}
            else{out.mu=Q;out.estimated_abs_error_mu=E;value_pass=value_pass&&pass;}
            out.radial_error[j]=rad[j].get();out.inner_error[j]=inn[j].get();out.geometry_error[j]=geo[j].get();out.event_error[j]=evt[j].get();out.roundoff_error[j]=rnd[j].get();
        }
        if(value_pass && !value_snapshot.valid){
            value_snapshot.valid=true;value_snapshot.q=out.mu;value_snapshot.error=out.estimated_abs_error_mu;
            value_snapshot.ledger[0]=out.radial_error[0];value_snapshot.ledger[1]=out.inner_error[0];value_snapshot.ledger[2]=out.geometry_error[0];value_snapshot.ledger[3]=out.event_error[0];value_snapshot.ledger[4]=out.roundoff_error[0];
            if(gradients){gradient_phase=true;gradient_start_nodes=stats.node_evaluations;gradient_rounds=0;}
        }
        bool all_gradient_pass=!gradients;
        if(gradients)for(bool x:gradient_pass)all_gradient_pass=all_gradient_pass&&x;
        bool all_gradient_error_pass=!gradients;
        if(gradients)for(bool x:gradient_error_pass)all_gradient_error_pass=all_gradient_error_pass&&x;
        if(value_pass&&stats.first_value_pass_nodes==0)stats.first_value_pass_nodes=stats.node_evaluations;
        if(all_gradient_error_pass&&stats.first_gradient_error_pass_nodes==0)stats.first_gradient_error_pass_nodes=stats.node_evaluations;
        if(all_gradient_pass&&stats.first_gradient_contract_nodes==0)stats.first_gradient_contract_nodes=stats.node_evaluations;
        stats.gradient_error_pass_mask=0;stats.gradient_contract_mask=0;
        for(int j=0;j<5;++j) {
            if(gradient_error_pass[j])stats.gradient_error_pass_mask|=1u<<j;
            if(gradient_pass[j])stats.gradient_contract_mask|=1u<<j;
        }
        last_gradient_pass=gradient_pass;
        for(int j=0;j<5;++j)gradient_invalid_seen[j]=gradient_invalid_seen[j]||invalid_gradient_local[j];
        if(value_pass_out)*value_pass_out=value_pass;
        if(gradient_pass_out)*gradient_pass_out=all_gradient_pass;
        stats.scheduler_ms+=ms(start);
    };
    while(failure==AdaptiveStop::Converged) {
        bool value_pass=false,gradient_pass=false;
        gather(&value_pass,&gradient_pass);
        if(value_pass && gradient_pass)break;
        const bool any_gradient_invalid=std::any_of(gradient_invalid_seen.begin(),gradient_invalid_seen.end(),[](bool x){return x;});
        if(value_snapshot.valid && any_gradient_invalid) {
            failure=AdaptiveStop::Nonfinite;
            break;
        }
        if(value_snapshot.valid && policy==GradientPolicy::ValueFirst &&
           (gradient_pass || stats.node_evaluations-gradient_start_nodes>=cfg.value_first_gradient_node_budget ||
            gradient_rounds>=cfg.value_first_gradient_round_budget)){
            failure=gradient_pass?AdaptiveStop::Converged:AdaptiveStop::BudgetExceeded;
            break;
        }
        size_t worst=0;double priority=-1;
        for(size_t i=0;i<w.panels.size();++i)if(w.panels[i].active){auto& p=w.panels[i];double score=0;
            if(!value_pass || !value_snapshot.valid){
                score=p.error[0]/cfg.tol.budget(0,out.mu);
                if(!p.value_resolved)score=std::max(score,1.0);
            } else {
                for(int j=1;j<nc;++j){
                    score=std::max(score,p.error[j]/cfg.tol.budget(j,out.grad_mu[j-1]));
                    if(!p.gradient_resolved[j-1])score=std::max(score,1.0);
                }
            }
            if(score>priority){priority=score;worst=i;}}
        auto& p=w.panels[worst];
        const int first_component=(!value_pass || !value_snapshot.valid)?0:1;
        bool inner_dominant=false,geometry_dominant=false,event_dominant=false;
        for(int j=first_component;j<nc;++j){double T=cfg.tol.budget(j,j?out.grad_mu[j-1]:out.mu);
            inner_dominant=inner_dominant||(out.inner_error[j]>T && p.inner[j]>p.radial[j]);
            geometry_dominant=geometry_dominant||(out.geometry_error[j]+out.roundoff_error[j]>T && p.geometry[j]+p.roundoff[j]>p.radial[j]);
            event_dominant=event_dominant||(out.event_error[j]>T && p.event[j]>p.radial[j]);}
        if(event_dominant){failure=AdaptiveStop::EventLocationLimited;break;}
        if(inner_dominant){failure=AdaptiveStop::InnerAccuracyLimited;break;}
        if(geometry_dominant){failure=AdaptiveStop::RoundoffLimited;break;}
        if(p.level<cfg.max_level){failure=refine(worst,p.level+1);continue;}
        if(p.depth>=cfg.max_depth||w.panels.size()+2>cfg.max_panels){failure=AdaptiveStop::BudgetExceeded;break;}
        if(allocated_bytes()+2*sizeof(AdaptivePanel)>cfg.max_bytes){failure=AdaptiveStop::BudgetExceeded;break;}
        AdaptivePanel l,r;l.parent=r.parent=int(worst);l.map=r.map=p.map;l.cell=r.cell=p.cell;l.depth=r.depth=p.depth+1;
        l.left_uncertainty=p.left_uncertainty;r.right_uncertainty=p.right_uncertainty;
        l.left_radius_lo=p.left_radius_lo;r.right_radius_lo=p.right_radius_lo;
        l.xl=p.xl;l.xr=r.xl=0.5*(p.xl+p.xr);r.xr=p.xr;p.active=false;
        stats.split_discarded_nodes+=(1<<p.level)-1;++stats.splits;
        if(w.panels.capacity()<w.panels.size()+2)w.panels.reserve(w.panels.size()+2);
        w.panels.push_back(l);w.panels.push_back(r);
        failure=refine(w.panels.size()-2,cfg.initial_level);if(failure==AdaptiveStop::Converged)failure=refine(w.panels.size()-1,cfg.initial_level);
    }
    bool final_value_pass=false,final_gradient_pass=false;
    gather(&final_value_pass,&final_gradient_pass);
    bool gradient_contract=final_gradient_pass;
    if(gradients) {
        gradient_contract=true;
        for(bool x:last_gradient_pass)gradient_contract=gradient_contract&&x;
    }
    const bool value_valid=value_snapshot.valid;
    out.value_converged=value_valid;
    out.value_stop_reason=value_valid?AdaptiveStop::Converged:failure;
    if(value_valid) {
        // Gradient refinement must never replace an already accepted primal
        // value. Keep the value and its ledger from the value stage exactly.
        out.mu=value_snapshot.q;
        out.estimated_abs_error_mu=value_snapshot.error;
        out.radial_error[0]=value_snapshot.ledger[0];
        out.inner_error[0]=value_snapshot.ledger[1];
        out.geometry_error[0]=value_snapshot.ledger[2];
        out.event_error[0]=value_snapshot.ledger[3];
        out.roundoff_error[0]=value_snapshot.ledger[4];
    }
    if(!gradients) {
        out.gradient_stop_reason=AdaptiveStop::Converged;
        for(int j=0;j<5;++j) {
            out.grad_quality[j]=GradientQuality::NotRequested;
            out.grad_reason[j]=GradientReason::NotRequested;
            out.grad_converged[j]=false;
        }
    } else {
        out.gradient_stop_reason=gradient_contract?AdaptiveStop::Converged:failure;
        for(int j=0;j<5;++j) {
            const bool finite=std::isfinite(out.grad_mu[j])&&std::isfinite(out.estimated_abs_error_grad[j]);
            if(invalid||gradient_invalid_seen[j]||!finite) {
                out.grad_quality[j]=GradientQuality::Invalid;
                out.grad_reason[j]=(!finite||gradient_invalid_seen[j])
                    ? GradientReason::Nonfinite : gradient_reason(failure);
                out.grad_converged[j]=false;
            } else if(last_gradient_pass[j]) {
                out.grad_quality[j]=GradientQuality::ToleranceMet;
                out.grad_reason[j]=GradientReason::ToleranceMet;
                out.grad_converged[j]=true;
            } else {
                out.grad_quality[j]=GradientQuality::FiniteUncertified;
                out.grad_reason[j]=gradient_reason(failure);
                out.grad_converged[j]=false;
            }
        }
    }
    // ValueFirst deliberately reports a usable value once its independent
    // contract is met. Strict retains the old whole-request stop semantics,
    // while value_converged remains available in both modes.
    const bool any_gradient_invalid=std::any_of(gradient_invalid_seen.begin(),gradient_invalid_seen.end(),[](bool x){return x;});
    out.stop=(value_valid&&!invalid&&!any_gradient_invalid&&
              (policy!=GradientPolicy::Strict||gradient_contract))
        ?AdaptiveStop::Converged:failure;
    out.assurance=value_valid?AccuracyAssurance::Estimated:AccuracyAssurance::None;
    out.numerical_status=(!value_valid||invalid||any_gradient_invalid)?Status::GRADIENT_UNRELIABLE:Status::OK;
    for(const auto& p:w.panels)if(p.active){++stats.panels;++stats.level_histogram[p.level];}
    out.stats=stats;return out;
}
} // adaptive_detail
} // lcbinint::holonomic

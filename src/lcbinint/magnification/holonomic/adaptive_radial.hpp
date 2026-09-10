#pragma once
#include "lcbinint/magnification/holonomic/epoch_jacobian.hpp"
#include "lcbinint/magnification/holonomic/nested_fejer2.hpp"
#include <chrono>
#include <cstdint>
#include <limits>

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
struct AdaptiveTolerance {
    double mu_atol=1e-8,mu_rtol=1e-4;
    std::array<double,5> grad_atol{{1e-6,1e-6,1e-6,1e-6,1e-6}};
    std::array<double,5> grad_rtol{{1e-4,1e-4,1e-4,1e-4,1e-4}};
    double budget(int j,double Q) const {
        return std::max(j ? grad_atol[j-1] : mu_atol,
                        (j ? grad_rtol[j-1] : mu_rtol)*std::fabs(Q));
    }
};
struct AdaptiveConfig {
    AdaptiveTolerance tol;
    bool with_jacobian=false,require_bound=false,fold_maps=true;
    bool reuse_samples=true; // isolated A/B control; default keeps nested samples
    int initial_level=3,max_level=8,max_depth=12;
    size_t max_node_evals=32768,max_panels=512,max_bytes=64*1024*1024;
};
struct AdaptiveStats {
    size_t unique_nodes=0,node_evaluations=0,reused_nodes=0,split_discarded_nodes=0,splits=0;
    size_t root_anchors=0,panels=0;
    std::array<size_t,9> level_histogram{};
    double physical_ms=0,estimator_ms=0,scheduler_ms=0,topology_ms=0,setup_ms=0;
};
struct AdaptiveResult {
    double mu=0,estimated_abs_error_mu=std::numeric_limits<double>::infinity();
    std::array<double,5> grad_mu{},estimated_abs_error_grad{};
    bool value_converged=false;
    std::array<bool,5> grad_converged{};
    AccuracyAssurance assurance=AccuracyAssurance::None;
    AdaptiveStop stop=AdaptiveStop::BudgetExceeded;
    Status numerical_status=Status::GRADIENT_UNRELIABLE;
    AdaptiveVector radial_error{},inner_error{},geometry_error{},event_error{},roundoff_error{};
    AdaptiveStats stats;
};
struct AdaptiveSample {
    double R=0;
    AdaptiveVector value{},inner{},geometry{},roundoff{};
    QuarticWarm quartic{};
    RootPairWarm roots{};
    bool reliable=true;
};
struct AdaptivePanel {
    FoldRadialMap map;
    double xl=-1,xr=1;
    int cell=0,level=0,depth=0,parent=-1;
    bool active=true,resolved=false;
    double left_uncertainty=0,right_uncertainty=0;
    // Finest lattice IDs: inherited k doubles on refinement, so slot is fixed.
    std::array<int,256> samples;
    AdaptiveVector q{},error{},radial{},inner{},geometry{},event{},roundoff{};
    AdaptivePanel() { samples.fill(-1); }
};
struct AdaptiveWorkspace {
    std::vector<AdaptiveSample> samples;
    std::vector<AdaptivePanel> panels;
    std::uint64_t generation=0;
    void reset() { samples.clear();panels.clear();++generation; }
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
    } return true;
}
inline void estimate(AdaptivePanel& p,const AdaptiveWorkspace& w,int nc) {
    const int m=1<<p.level,step=256/m;
    const auto& rule=fejer_rule(p.level);
    std::array<Sum,6> q,inn,geo,rnd;
    for(int k=1;k<m;++k) {
        const auto& s=w.samples[p.samples[k*step]];
        for(int j=0;j<nc;++j) {
            double wt=rule.w[k-1];
            q[j].add(wt*s.value[j]); inn[j].add(wt*s.inner[j]);
            geo[j].add(wt*s.geometry[j]);rnd[j].add(wt*s.roundoff[j]+eps*std::fabs(wt*s.value[j]));
        }
    }
    AdaptiveVector detail{},previous{};
    for(int lev=p.level-1;lev<=p.level;++lev) {
        const int mm=1<<lev,st=256/mm,coarse=mm/2-1;
        const auto& r=fejer_rule(lev);
        AdaptiveVector norm{};
        for(int k=1;k<mm;k+=2) {
            for(int j=0;j<nc;++j) {
                Sum pred;
                for(int h=1;h<=coarse;++h) pred.add(r.interp[(k/2)*coarse+h-1]*w.samples[p.samples[2*h*st]].value[j]);
                double d=w.samples[p.samples[k*st]].value[j]-pred.get();
                norm[j]+=r.norm_w[k-1]*d*d;
            }
        }
        for(int j=0;j<nc;++j) (lev==p.level?detail:previous)[j]=std::sqrt(3.14159265358979323846*norm[j]);
    }
    p.event.fill(0);
    for(int side=0;side<2;++side) {
        const double delta=side?p.right_uncertainty:p.left_uncertainty;
        if(!(delta>0))continue;
        int k=side?1:m-1;
        const auto& sample=w.samples[p.samples[k*step]];
        const double x=0.5*(p.xl+p.xr)+0.5*(p.xr-p.xl)*rule.x[k-1];
        auto mapped=p.map(x);
        double distance=side?p.map.b-sample.R:sample.R-p.map.a;
        double map_jac=mapped[1]*0.5*(p.xr-p.xl);
        // Local endpoint model, not a verified coefficient envelope. For
        // derivatives use the integrable 1/sqrt(distance) worst case.
        for(int j=0;j<nc;++j)p.event[j]+=2*std::fabs(sample.value[j]/map_jac)*(j?std::sqrt(distance*delta):delta);
    }
    p.resolved=true;
    for(int j=0;j<nc;++j) {
        p.q[j]=q[j].get();p.inner[j]=inn[j].get();p.geometry[j]=geo[j].get();p.roundoff[j]=rnd[j].get();
        const double floor=p.roundoff[j]+p.inner[j]+p.geometry[j];
        const bool decays=detail[j]<=0.5*previous[j] || detail[j]<=floor;
        p.resolved=p.resolved&&decays;
        // Unresolved panels must refine even if their integral difference cancels.
        p.radial[j]=detail[j];p.error[j]=detail[j]+floor+p.event[j];
    }
}
// Callback evaluates a NEW mapped node. Existing sample values never change.
// sample snapshots remain available after a split as same-cell root anchors.
template<class Evaluate>
AdaptiveResult integrate(AdaptiveWorkspace& w,const AdaptiveConfig& cfg,Evaluate eval) {
    AdaptiveResult out;
    if(cfg.require_bound){out.stop=AdaptiveStop::BoundUnavailable;return out;}
    if(!valid_config(cfg)){out.stop=AdaptiveStop::InvalidConfig;return out;}
    const int nc=cfg.with_jacobian?6:1;
    AdaptiveStats stats;
    auto allocated_bytes=[&](){size_t n=w.samples.capacity()*sizeof(AdaptiveSample)+w.panels.capacity()*sizeof(AdaptivePanel);
        for(const auto& s:w.samples)n+=s.roots.pairs.capacity()*sizeof(RootPair);return n;};
    auto refine=[&](size_t ip,int level)->AdaptiveStop {
        auto& p=w.panels[ip];const int m=1<<level,step=256/m;
        size_t needed=0;for(int k=1;k<m;++k)needed+=!cfg.reuse_samples||p.samples[k*step]<0;
        if(stats.node_evaluations+needed>cfg.max_node_evals ||
           (w.samples.size()+needed)*sizeof(AdaptiveSample)+w.panels.size()*sizeof(AdaptivePanel)>cfg.max_bytes)
            return AdaptiveStop::BudgetExceeded;
        const size_t target=w.samples.size()+needed;
        const size_t growth=target>w.samples.capacity()?(target-w.samples.capacity())*sizeof(AdaptiveSample):0;
        // Quartic has at most two pairs; allow four slots for the existing
        // tracker vector capacity. This budget covers retained sample storage.
        if(allocated_bytes()+growth+needed*4*sizeof(RootPair)>cfg.max_bytes)return AdaptiveStop::BudgetExceeded;
        if(target>w.samples.capacity())w.samples.reserve(target);
        for(int k=1;k<m;++k) {
            int slot=k*step;
            bool fresh=p.samples[slot]<0;
            if(!fresh&&cfg.reuse_samples){++stats.reused_nodes;continue;}
            const double x=0.5*(p.xl+p.xr)+0.5*(p.xr-p.xl)*fejer_rule(level).x[k-1];
            auto mapped=p.map(x); mapped[1]*=0.5*(p.xr-p.xl);
            if(!(mapped[0]>p.map.a+p.left_uncertainty && mapped[0]<p.map.b-p.right_uncertainty))return AdaptiveStop::EventLocationLimited;
            const AdaptiveSample* anchor=nullptr;double distance=std::numeric_limits<double>::infinity();
            for(int ancestor=int(ip);ancestor>=0;ancestor=w.panels[ancestor].parent)
            for(int idx:w.panels[ancestor].samples) if(idx>=0) {
                double d=std::fabs(w.samples[idx].R-mapped[0]);
                if(d<distance && w.samples[idx].reliable){distance=d;anchor=&w.samples[idx];}
            }
            auto start=Clock::now();AdaptiveSample s=eval(mapped[0],mapped[1],p.cell,anchor);
            stats.physical_ms+=ms(start); if(anchor)++stats.root_anchors;
            s.R=mapped[0];bool finite=true;
            for(int j=0;j<nc;++j)finite=finite&&std::isfinite(s.value[j])&&std::isfinite(s.inner[j])&&std::isfinite(s.geometry[j])&&std::isfinite(s.roundoff[j])&&s.inner[j]>=0&&s.geometry[j]>=0&&s.roundoff[j]>=0;
            p.samples[slot]=int(w.samples.size());w.samples.push_back(std::move(s));++stats.node_evaluations;if(fresh)++stats.unique_nodes;
            if(!finite)return AdaptiveStop::Nonfinite;
            if(!w.samples.back().reliable)return AdaptiveStop::TopologyUnresolved;
        }
        p.level=level;auto start=Clock::now();estimate(p,w,nc);stats.estimator_ms+=ms(start);
        return AdaptiveStop::Converged;
    };
    AdaptiveStop failure=AdaptiveStop::Converged;
    if(w.panels.size()>cfg.max_panels){out.stop=AdaptiveStop::BudgetExceeded;return out;}
    for(size_t i=0;i<w.panels.size();++i) {failure=refine(i,cfg.initial_level);if(failure!=AdaptiveStop::Converged)break;}
    while(failure==AdaptiveStop::Converged) {
        auto start=Clock::now();std::array<Sum,6> total,errors,rad,inn,geo,evt,rnd;
        bool resolved=true;
        for(const auto& p:w.panels)if(p.active){resolved=resolved&&p.resolved;for(int j=0;j<nc;++j){total[j].add(p.q[j]);errors[j].add(p.error[j]);rad[j].add(p.radial[j]);inn[j].add(p.inner[j]);geo[j].add(p.geometry[j]);evt[j].add(p.event[j]);rnd[j].add(p.roundoff[j]);}}
        bool good=resolved;
        for(int j=0;j<nc;++j){double Q=total[j].get(),E=errors[j].get();bool pass=E<=cfg.tol.budget(j,Q);
            if(j){out.grad_mu[j-1]=Q;out.estimated_abs_error_grad[j-1]=E;out.grad_converged[j-1]=pass;}else{out.mu=Q;out.estimated_abs_error_mu=E;out.value_converged=pass;}
            out.radial_error[j]=rad[j].get();out.inner_error[j]=inn[j].get();out.geometry_error[j]=geo[j].get();out.event_error[j]=evt[j].get();out.roundoff_error[j]=rnd[j].get();good=good&&pass;
        }
        if(good){stats.scheduler_ms+=ms(start);break;}
        size_t worst=0;double priority=-1;
        for(size_t i=0;i<w.panels.size();++i)if(w.panels[i].active){auto& p=w.panels[i];double score=0;
            for(int j=0;j<nc;++j)score=std::max(score,p.error[j]/cfg.tol.budget(j,j?out.grad_mu[j-1]:out.mu));
            if(!p.resolved)score=std::max(score,1.0);
            if(score>priority){priority=score;worst=i;}}
        stats.scheduler_ms+=ms(start);
        auto& p=w.panels[worst];
        bool inner_dominant=false,geometry_dominant=false;
        for(int j=0;j<nc;++j){double T=cfg.tol.budget(j,j?out.grad_mu[j-1]:out.mu);
            inner_dominant=inner_dominant||(out.inner_error[j]>T && p.inner[j]>p.radial[j]);
            geometry_dominant=geometry_dominant||(out.geometry_error[j]+out.roundoff_error[j]>T && p.geometry[j]+p.roundoff[j]>p.radial[j]);}
        bool event_dominant=false;
        for(int j=0;j<nc;++j)event_dominant=event_dominant||(out.event_error[j]>cfg.tol.budget(j,j?out.grad_mu[j-1]:out.mu) && p.event[j]>p.radial[j]);
        if(event_dominant){failure=AdaptiveStop::EventLocationLimited;break;}
        if(inner_dominant){failure=AdaptiveStop::InnerAccuracyLimited;break;}
        if(geometry_dominant){failure=AdaptiveStop::RoundoffLimited;break;}
        if(p.level<cfg.max_level){failure=refine(worst,p.level+1);continue;}
        if(p.depth>=cfg.max_depth||w.panels.size()+2>cfg.max_panels){failure=AdaptiveStop::BudgetExceeded;break;}
        if(allocated_bytes()+2*sizeof(AdaptivePanel)>cfg.max_bytes){failure=AdaptiveStop::BudgetExceeded;break;}
        AdaptivePanel l,r;l.parent=r.parent=int(worst);l.map=r.map=p.map;l.cell=r.cell=p.cell;l.depth=r.depth=p.depth+1;
        l.left_uncertainty=p.left_uncertainty;r.right_uncertainty=p.right_uncertainty;
        l.xl=p.xl;l.xr=r.xl=0.5*(p.xl+p.xr);r.xr=p.xr;p.active=false;
        stats.split_discarded_nodes+=(1<<p.level)-1;++stats.splits;
        if(w.panels.capacity()<w.panels.size()+2)w.panels.reserve(w.panels.size()+2);
        w.panels.push_back(l);w.panels.push_back(r);
        failure=refine(w.panels.size()-2,cfg.initial_level);if(failure==AdaptiveStop::Converged)failure=refine(w.panels.size()-1,cfg.initial_level);
    }
    out.stop=failure;out.assurance=AccuracyAssurance::Estimated;
    out.numerical_status=failure==AdaptiveStop::Converged?Status::OK:Status::GRADIENT_UNRELIABLE;
    if(failure!=AdaptiveStop::Converged){out.value_converged=false;out.grad_converged.fill(false);}
    for(const auto& p:w.panels)if(p.active){++stats.panels;++stats.level_histogram[p.level];}
    out.stats=stats;return out;
}
} // adaptive_detail
} // lcbinint::holonomic

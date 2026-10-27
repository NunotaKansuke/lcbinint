#pragma once
#include "support_resultant.hpp"
#include "radial_events.hpp"
#include "lcbinint/math/polynomial_roots.hpp"

namespace lcbinint::algebraic_boundary::triple::projected_event_detail {
using I=surface_detail::I;
using Coefficients=resultant_detail::Coefficients;
struct Guess {double radius,coordinate;bool chart;};
inline int variations(const Coefficients&b) {
    int previous=0,count=0;
    for(auto x:b) {
        const int sign=x.sign();if(sign==2)return -1;
        if(sign){if(previous && previous!=sign)++count;previous=sign;}
    }
    return count;
}
// A proposal never certifies an event. Keep failed Newton work for a later
// child interval; only the physical interval inclusion below is accepted.
inline std::optional<RadialEvent> locate(const Geometry&p,double slo,double shi,
        double tolerance,std::vector<Guess>&retained,RadialEventPlan&stats,
        const std::vector<Guess>* previous_events=nullptr) {
    const double rlo=std::nextafter(std::sqrt(slo),INFINITY);
    const double rhi=std::nextafter(std::sqrt(shi),-INFINITY);
    if(!(rhi>rlo && rlo>0))return {};
    auto try_guess=[&](Guess seed)->std::optional<RadialEvent> {
        double r=seed.radius,t=seed.coordinate;
        if(!(r>rlo && r<rhi))r=rlo+(rhi-rlo)*.5;
        for(int step=0;step<24;++step) {
            const auto f=event_detail::physical(r,t,p,seed.chart);if(!f)break;
            const double det=f->r*f->tt-f->t*f->rt;
            if(!std::isfinite(det) || det==0)break;
            double dr=(-f->v*f->tt+f->t*f->t)/det,dt=(-f->r*f->t+f->rt*f->v)/det;
            bool inside=false;
            for(int backtrack=0;backtrack<24;++backtrack) {
                if(r+dr>rlo && r+dr<rhi && std::abs(t+dt)<2){inside=true;break;}
                dr*=.5;dt*=.5;
            }
            if(!inside)break;
            r+=dr;t+=dt;
            if(std::abs(dr)+std::abs(dt)<8*std::numeric_limits<double>::epsilon()*(1+std::abs(r)+std::abs(t)))break;
        }
        if(!std::isfinite(r) || !std::isfinite(t))return {};
        retained.push_back({r,t,seed.chart});
        const double eps=std::numeric_limits<double>::epsilon();
        const double goal=tolerance*p.rho*p.rho/(1024*(1+r));
        for(double factor:{64.,256.,1024.}) {
            const double dr=std::max(goal,factor*eps*(1+std::abs(r)));
            const double dt=std::max(goal,factor*eps*(1+std::abs(t)));
            if(!(r-dr>rlo && r+dr<rhi))continue;
            const event_detail::Box box{r-dr,r+dr,t-dt,t+dt,seed.chart};
            const auto q=event_detail::prove_physical(box,p);if(q.kind!=2)continue;
            const I squared=q.r*q.r;
            if(!(squared.lo>slo && squared.hi<shi))continue;
            ++stats.double_refinements;
            return RadialEvent{q.r.lo+(q.r.hi-q.r.lo)*.5,q.t.lo+(q.t.hi-q.t.lo)*.5,
                q.r.lo,q.r.hi,seed.chart,q.t.lo,q.t.hi};
        }
        return {};
    };
    // Prior-query locations are proposals only. Pick at most two by radial
    // proximity; every success still passes the current physical inclusion.
    if(previous_events && !previous_events->empty()) {
        const double mid=rlo+(rhi-rlo)*.5;
        std::size_t first=previous_events->size(),second=first;
        double best=INFINITY,next=INFINITY;
        for(std::size_t j=0;j<previous_events->size();++j) {
            const double distance=std::abs((*previous_events)[j].radius-mid);
            if(distance<best){second=first;next=best;first=j;best=distance;}
            else if(distance<next){second=j;next=distance;}
        }
        for(auto index:{first,second})if(index<previous_events->size())
            if(auto event=try_guess((*previous_events)[index]))return event;
    }
    const std::size_t previous=retained.size();
    for(std::size_t j=0;j<previous;++j) {
        const auto seed=retained[j];
        if(seed.radius>rlo && seed.radius<rhi)if(auto event=try_guess(seed))return event;
    }
    const double radius=rlo+(rhi-rlo)*.5;
    const auto boundary=boundary_from_radius(radius,p).coefficients;
    // Use the second chart only if the first set of proposals did not yield
    // an inclusion. Large finite roots are mapped to the other local chart.
    for(bool chart:{false,true}) {
        const auto coefficients=chart?reciprocal(boundary):boundary;
        std::vector<Complex> derivative(6);
        for(int j=0;j<6;++j)derivative[j]=double(j+1)*coefficients[j+1];
        while(derivative.size()>1 && derivative.back()==Complex{})derivative.pop_back();
        if(derivative.size()<2)continue;
        math::PolynomialRootSolver solver;++stats.resultant_solves;
        const auto roots=solver.solve(derivative).roots;
        struct Candidate {Guess seed;double score;};std::vector<Candidate> candidates;
        for(auto root:roots)if(std::isfinite(root.real()) && std::isfinite(root.imag()) &&
                std::abs(root.imag())<1e-6*(1+std::abs(root.real()))) {
            double t=root.real();bool local_chart=chart;
            if(std::abs(t)>1){t=-1/t;local_chart=!local_chart;}
            const auto f=event_detail::physical(radius,t,p,local_chart);
            if(f)candidates.push_back({{radius,t,local_chart},std::abs(f->v)});
        }
        std::sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a.score<b.score;});
        for(auto c:candidates)if(auto event=try_guess(c.seed))return event;
    }
    return {};
}
}

namespace lcbinint::algebraic_boundary::triple {
namespace projected_event_detail {
struct NoComponents {PartialComponents operator()()const{return {};}};
}
// The degree-28 support resultant supplies completeness over the full angle
// circle. A one-variation positive cell has exactly one discriminant zero,
// counted with multiplicity. A simple discriminant zero can contain only one
// repeated projective sextic root. One local physical inclusion therefore
// assigns that entire radial cell, not merely a small angular neighbourhood.
template<class ComponentFactory=projected_event_detail::NoComponents>
inline RadialEventPlan projected_radial_event_plan(const Geometry&p,int max_boxes,
        double flux_tolerance,const PartialComponents* components=nullptr,
        const std::vector<projected_event_detail::Guess>* previous_events=nullptr,
        ComponentFactory make_components={}) {
    using namespace projected_event_detail;
    // The measured coefficient tier is the hardware extended significand.
    // Keep the incumbent planner on platforms with software quad or no
    // extended significand; the physical/integration work remains double.
    if constexpr(std::numeric_limits<long double>::digits!=64) {
        auto late=components?PartialComponents{}:make_components();
        return radial_event_plan(p,max_boxes,flux_tolerance,components?components:&late);
    }
    CertifiedRadialCover cover;auto&out=cover.prepared;p.validate();
    out.identity={p.X,p.Y,p.rho,p.m1,p.m2,p.a,p.bx,p.by};
    if(!interval_detail::environment_ok()){out.reason="floating_environment";return out;}
    if(max_boxes<1){out.reason="event_box_budget";return out;}
    if(!event_detail::radial_domain(p,out))return out;
    const bool partial=components && components->identity==out.identity && !components->images.empty();
    // Geometry-only projection may finish without any image seeds. Build
    // partial components only if unresolved coverage actually needs them.
    std::optional<PartialComponents> late_components;
    auto fallback_components=[&]() {
        if(!components){late_components.emplace(make_components());components=&*late_components;}
        return components;
    };
    const double lo=interval_detail::down(out.rmin*out.rmin),hi=interval_detail::up(out.rmax*out.rmax);
    if(!(std::isfinite(lo) && std::isfinite(hi) && lo>=0 && hi>lo))
        return radial_event_plan(p,max_boxes,flux_tolerance,fallback_components());
    auto coefficients=resultant_detail::bernstein(p,lo,hi);
    bool bounded=true;
    for(auto c:coefficients)if(!c.valid() || std::max(std::abs(c.lo),std::abs(c.hi))>
            std::numeric_limits<double>::max()/4)bounded=false;
    if(!bounded){cover.unresolved={{out.rmin,out.rmax}};return radial_event_plan(p,max_boxes,flux_tolerance,fallback_components(),&cover);}
    struct Node {unsigned index;int depth;Coefficients coefficients;};
    std::vector<Node> pending{{0,0,coefficients}};
    std::vector<std::pair<double,double>> unresolved_s;
    std::vector<Guess> guesses;
    // A few hundred cheap projected cells can exclude broad regions before
    // the two-dimensional complement solve. Keep the same total work budget;
    // the projection and every accepted physical inclusion remain certified.
    const int projection_budget=std::min(max_boxes,512);
    while(!pending.empty()) {
        auto n=std::move(pending.back());pending.pop_back();
        const I width=I(hi)-I(lo);
        const I left=I(lo)+width*I(std::ldexp(double(n.index),-n.depth));
        const I right=I(lo)+width*I(std::ldexp(double(n.index+1),-n.depth));
        if(out.resultant_nodes>=projection_budget){unresolved_s.emplace_back(left.lo,right.hi);continue;}
        ++out.resultant_nodes;
        if(resultant_detail::excludes(n.coefficients))continue;
        const int count=variations(n.coefficients);
        // One variation already certifies the whole-circle count. Try a
        // physical inclusion now; a fixed subdivision depth adds no proof.
        // Failed Newton proposals are retained for the child intervals.
        if(count==1 && n.coefficients.front().sign()!=0 && n.coefficients.back().sign()!=0) {
            bool assigned=false;
            if(partial)for(auto e:components->events) {
                const I square=I(e.radius_lo,e.radius_hi)*I(e.radius_lo,e.radius_hi);
                if(square.lo>left.hi && square.hi<right.lo){assigned=true;break;}
            }
            if(assigned)continue;
            if(auto event=locate(p,left.hi,right.lo,flux_tolerance,guesses,out,previous_events)) {
                out.events.push_back(*event);++out.resultant_events;continue;
            }
        }
        if(n.depth>=12){unresolved_s.emplace_back(left.lo,right.hi);continue;}
        auto halves=resultant_detail::split(n.coefficients);
        pending.push_back({2*n.index+1,n.depth+1,halves.second});
        pending.push_back({2*n.index,n.depth+1,halves.first});
    }
    std::sort(unresolved_s.begin(),unresolved_s.end());
    for(auto cell:unresolved_s) {
        if(cell.second<=0)continue;
        const double rlo=std::max(out.rmin,std::nextafter(std::sqrt(std::max(0.,cell.first)),-INFINITY));
        const double rhi=std::min(out.rmax,std::nextafter(std::sqrt(cell.second),INFINITY));
        if(!(rhi>rlo))continue;
        if(!cover.unresolved.empty() && rlo<=cover.unresolved.back().second)
            cover.unresolved.back().second=std::max(cover.unresolved.back().second,rhi);
        else cover.unresolved.emplace_back(rlo,rhi);
    }
    if(!cover.unresolved.empty() && max_boxes<=out.resultant_nodes) {
        out.reason="event_box_budget";return out;
    }
    if(cover.unresolved.empty() && out.events.empty() && (!partial || components->events.empty())) {
        out.reason="resultant_empty_events";return out;
    }
    return radial_event_plan(p,max_boxes-out.resultant_nodes,flux_tolerance,
        cover.unresolved.empty()?components:fallback_components(),&cover);
}
}

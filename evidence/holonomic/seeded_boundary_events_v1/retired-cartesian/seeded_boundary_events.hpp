#pragma once

// Research-only alternative to the global D14 solve. Trace actual source-limb
// preimages from supplied finite-source image seeds and refine radial extrema.
// Successful continuation is NOT a certificate that every boundary loop was
// seeded. Production dispatch does not call this header.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "lcbinint/magnification/holonomic/cells.hpp"

namespace lcbinint::holonomic {
struct SeededBoundaryConfig {
    double turn_limit = 0.20;
    double width_fraction = 0.25;
    bool width_control = true;
    int max_steps_per_loop = 4096;
};
struct SeededBoundaryStats {
    int quartics=0, evaluations=0, steps=0, retries=0, loops=0;
    int fold_solves=0, fold_failures=0, width_limits=0, fold_crossings=0;
};
struct SeededBoundaryResult {
    std::vector<RadialEvent> events;
    SeededBoundaryStats stats;
    bool completed=false;
    bool coverage_certified=false;
    const char* reason="";
    double r_max=0;
};
namespace seeded_boundary_detail {
struct V { double x=0,y=0; };
inline V operator+(V a,V b){return {a.x+b.x,a.y+b.y};}
inline V operator-(V a,V b){return {a.x-b.x,a.y-b.y};}
inline V operator*(V a,double s){return {a.x*s,a.y*s};}
inline double dot(V a,V b){return a.x*b.x+a.y*b.y;}
inline double norm(V a){return std::hypot(a.x,a.y);}
struct Geometry { double f=0,xx=0,xy=0,yy=0; V g; bool valid=false; };
inline Geometry geometry(V p,const PrimaryFrame& pf,SeededBoundaryStats& stats){
    ++stats.evaluations;
    using namespace phi_detail;
    C zb{p.x,-p.y}, zba{p.x-pf.a,-p.y};
    if(!(dot(p,p)>0) || !(zba.re*zba.re+zba.im*zba.im>0))return {};
    C a=C{1,0}/zb,b=C{1,0}/zba;
    C w=C{p.x-pf.X,p.y-pf.Y}-C{pf.m0,0}*a-C{1-pf.m0,0}*b;
    C k=C{pf.m0,0}*(a*a)+C{1-pf.m0,0}*(b*b);
    C h=C{-2*pf.m0,0}*(a*a*a)+C{-2*(1-pf.m0),0}*(b*b*b);
    C dx=C{1,0}+k,dy=muli(C{1,0}-k),hy=C{-h.re,-h.im};
    auto prod=[](C c,C d){return c.re*d.re+c.im*d.im;};
    Geometry o; o.f=prod(w,w)-pf.rho*pf.rho;
    o.g={2*prod(w,dx),2*prod(w,dy)};
    o.xx=2*(prod(dx,dx)+prod(w,h));
    o.xy=2*(prod(dx,dy)+prod(w,C{h.im,-h.re}));
    o.yy=2*(prod(dy,dy)+prod(w,hy));
    o.valid=std::isfinite(o.f+o.g.x+o.g.y+o.xx+o.xy+o.yy)&&norm(o.g)>0;
    return o;
}
inline V tangent(const Geometry& g){double n=norm(g.g);return {-g.g.y/n,g.g.x/n};}
inline bool correct(V pred,V normal,const PrimaryFrame& pf,V& p,Geometry& g,
                    SeededBoundaryStats& stats){
    p=pred;
    for(int i=0;i<10;++i){
        g=geometry(p,pf,stats);if(!g.valid)return false;
        double d=dot(g.g,normal);if(std::fabs(d)<1e-8*norm(g.g))return false;
        double s=g.f/d;p=p-normal*s;
        if(std::fabs(s)<8*std::numeric_limits<double>::epsilon()*(1+norm(p))){
            g=geometry(p,pf,stats);
            // A Cartesian endpoint cannot move below its binary64 ULP.
            // Near a lens pole the lens-map Jacobian amplifies this coordinate
            // rounding; include that amplification instead of a rho-only floor.
            return g.valid && std::fabs(g.f)<2e-10*pf.rho*pf.rho+
                32*std::numeric_limits<double>::epsilon()*(1+norm(p))*norm(g.g);
        }
    }
    return false;
}
inline bool fold(V p,const PrimaryFrame&pf,RadialEvent& e,SeededBoundaryStats&st){
    ++st.fold_solves;
    for(int i=0;i<14;++i){
        auto g=geometry(p,pf,st);if(!g.valid)return false;
        double c=-p.y*g.g.x+p.x*g.g.y;
        double cx=-p.y*g.xx+g.g.y+p.x*g.xy;
        double cy=-g.g.x-p.y*g.xy+p.x*g.yy;
        double det=g.g.x*cy-g.g.y*cx;
        if(!std::isfinite(det)||std::fabs(det)<1e-12*(norm(g.g)*std::hypot(cx,cy)))return false;
        V dp{(g.f*cy-g.g.y*c)/det,(g.g.x*c-g.f*cx)/det};p=p-dp;
        if(norm(dp)<1e-13*(1+norm(p))){
            g=geometry(p,pf,st);
            double R=norm(p),th=std::atan2(p.y,p.x);
            if(std::fabs(g.f)>2e-9*pf.rho*pf.rho+32*std::numeric_limits<double>::epsilon()*(1+R)*norm(g.g))return false;
            if(std::fabs(-p.y*g.g.x+p.x*g.g.y)>1e-9*R*norm(g.g))return false;
            e.radius=R;e.kind="physical_real";e.physically_real=true;
            e.detail="seeded boundary radial extremum (research)";
            e.fold_t_seed=std::tan(.5*th);e.fold_t_seed_valid=std::isfinite(e.fold_t_seed);
            // Double continuation does not establish a validated enclosure.
            e.radius_uncertainty=std::numeric_limits<double>::infinity();
            return true;
        }
    }return false;
}
struct Start {double R,th,partner; V p; bool covered=false;};
inline double angle_diff(double a,double b){return std::remainder(a-b,kTwoPi);}
inline double positive_width(double a,double b){double w=std::fmod(b-a,kTwoPi);if(w<0)w+=kTwoPi;return w;}
// Cached opposite endpoint, polished only when its implicit slope is regular.
// Near a radial fold the contour coordinate takes over, so width->0 cannot
// force infinitely many radial steps. Failed partner polish does not affect
// the accepted contour state.
inline double width_step(V p,const Geometry&g,double& partner,const PrimaryFrame&pf,
                         double ds,const SeededBoundaryConfig&cfg,SeededBoundaryStats&st){
    double R=norm(p),th=std::atan2(p.y,p.x),ft=-p.y*g.g.x+p.x*g.g.y;
    if(std::fabs(ft)<.1*R*norm(g.g)) {++st.fold_crossings;return ds;}
    auto pp=polish_endpoint(R,partner,pf,6);
    // Paired endpoints have opposite crossing orientation. Newton returning
    // the active endpoint is not evidence of a narrow physical interval.
    if(!pp.reliable || ft*pp.dphi_dtheta<=0){
        ++st.quartics;auto arcs=arc_intervals(R,pf);bool found=false;
        for(auto a:arcs.arcs)for(int j=0;j<2;++j)
            if(std::fabs(angle_diff(a[j],th))<1e-6){partner=a[1-j];found=true;}
        if(!found)return ds;
        pp=polish_endpoint(R,partner,pf,6);
        if(!pp.reliable || ft*pp.dphi_dtheta<=0)return ds;
    }
    partner=pp.theta;
    double w=std::fabs(angle_diff(pp.theta,th));
    if(w<1e-7)return ds;
    double fr=dot(p,g.g)/R,dth=-fr/ft;
    V t=tangent(g);double dr=std::fabs(dot(p,t)/R);
    if(dr<.1||std::fabs(dth)<1e-10)return ds;
    double limit=cfg.width_fraction*w/(std::fabs(dth)*dr);
    if(limit<ds){++st.width_limits;return limit;}return ds;
}
}

inline SeededBoundaryResult seeded_boundary_events(const PrimaryFrame&pf,
    const std::vector<Cplx<double>>&seeds,const SeededBoundaryConfig&cfg={}){
    using namespace seeded_boundary_detail;
    SeededBoundaryResult out;
    const double W=std::hypot(pf.X,pf.Y)+pf.rho;
    out.r_max=.5*(pf.a+W+std::hypot(pf.a-W,2.0))+1e-12;
    std::vector<Start> starts;
    for(auto z:seeds){
        double R=std::hypot(z.re,z.im);if(!(R>0))continue;
        bool duplicate=false;for(auto&s:starts)if(std::fabs(s.R-R)<1e-12*(1+R))duplicate=true;
        if(duplicate)continue;
        ++out.stats.quartics;auto arcs=arc_intervals(R,pf);
        if(arcs.kind==ArcKind::kDegenerate){out.reason="initial angular roots unresolved";return out;}
        for(auto a:arcs.arcs)for(int j=0;j<2;++j){
            auto e=polish_endpoint(R,a[j],pf);
            if(!e.reliable){out.reason="initial boundary polish failed";return out;}
            starts.push_back({R,e.theta,a[1-j],{R*std::cos(e.theta),R*std::sin(e.theta)},false});
        }
    }
    if(starts.empty()){out.reason="no boundary starts";return out;}
    for(std::size_t si=0;si<starts.size();++si){
        if(starts[si].covered)continue;
        auto start=starts[si];V p=start.p;auto g=geometry(p,pf,out.stats);
        if(!g.valid){out.reason="singular initial contour";return out;}
        V t0=tangent(g);double h=std::max(1e-10,pf.rho/norm(g.g)*pf.rho*.3);
        double traveled=0,partner=start.partner;bool closed=false;
        for(int k=0;k<cfg.max_steps_per_loop;++k){
            V t=tangent(g);double curvature=std::fabs(t.x*t.x*g.xx+2*t.x*t.y*g.xy+t.y*t.y*g.yy)/norm(g.g);
            double pole=std::min(norm(p),std::hypot(p.x-pf.a,p.y));
            double ds=std::min({h,cfg.turn_limit/std::max(curvature,1e-20),.15*pole});
            if(cfg.width_control)ds=width_step(p,g,partner,pf,ds,cfg,out.stats);
            V next;Geometry ng;bool accepted=false;
            for(int retry=0;retry<18;++retry){
                V pred=p+t*ds,normal{t.y,-t.x};
                if(correct(pred,normal,pf,next,ng,out.stats)&&
                   norm(next-pred)<.08*ds && dot(t,tangent(ng))>std::cos(cfg.turn_limit*1.5)){
                    accepted=true;break;
                }
                ds*=.5;++out.stats.retries;
            }
            if(!accepted || ds<32*std::numeric_limits<double>::epsilon()*(1+norm(p))){
                out.reason="local contour correction exhausted";return out;
            }
            ++out.stats.steps;traveled+=norm(next-p);
            // Every traversed seed-radius intersection reuses the local angular
            // root polish to identify starts already covered by this loop.
            double r0=norm(p),r1=norm(next);
            for(auto&s:starts)if(!s.covered && (s.R-r0)*(s.R-r1)<=0 && r0!=r1){
                double f=(s.R-r0)/(r1-r0);V q=p+(next-p)*f;
                auto ep=polish_endpoint(s.R,std::atan2(q.y,q.x),pf);
                if(ep.reliable && std::fabs(angle_diff(ep.theta,s.th))<1e-7)s.covered=true;
            }
            double c0=-p.y*g.g.x+p.x*g.g.y,c1=-next.y*ng.g.x+next.x*ng.g.y;
            if(c0*c1<0){
                V guess=p+(next-p)*(std::fabs(c0)/(std::fabs(c0)+std::fabs(c1)));
                RadialEvent ev;
                if(!fold(guess,pf,ev,out.stats)){++out.stats.fold_failures;out.reason="radial extremum refinement failed";return out;}
                bool exists=false;for(auto&e:out.events)if(std::fabs(e.radius-ev.radius)<2e-11*(1+ev.radius))exists=true;
                if(!exists)out.events.push_back(ev);
            }
            double q0=dot(p-start.p,t0),q1=dot(next-start.p,t0);
            if(k>8 && q0<=0 && q1>0 && norm(next-start.p)<2*ds && dot(t0,tangent(ng))>.8){
                V cross=p+(next-p)*(-q0/(q1-q0)),check;Geometry cg;
                if(correct(cross,{t0.y,-t0.x},pf,check,cg,out.stats)&&
                   norm(check-start.p)<1e-7*std::max(traveled,1e-8)){
                    closed=true;starts[si].covered=true;break;
                }
            }
            p=next;g=ng;h=ds*1.5;
        }
        if(!closed){out.reason="contour loop did not close";return out;}
        ++out.stats.loops;
    }
    std::sort(out.events.begin(),out.events.end(),[](auto&a,auto&b){return a.radius<b.radius;});
    out.completed=true;return out;
}

// Integration adapter for isolated A/B only. The caller must independently
// audit coverage; `coverage_certified` deliberately remains false above.
inline TopologyResult seeded_boundary_cell_plan(const PrimaryFrame&pf,
                                                const SeededBoundaryResult&r){
    if(!r.completed){TopologyResult t;t.status=Status::TOPOLOGY_UNCERTAIN;return t;}
    auto events=r.events;
    for(double R:{pf.a,std::sqrt(pf.m0)})if(R>0 && R<r.r_max){
        RadialEvent e;e.radius=R;e.kind="representation";events.push_back(e);
    }
    for(double R:re_detail::chart_p4_factor_roots(pf))if(R>0 && R<r.r_max){
        RadialEvent e;e.radius=R;e.kind="chart_p4";events.push_back(e);
    }
    return classify_event_cells(pf,std::move(events),r.r_max);
}
} // namespace lcbinint::holonomic

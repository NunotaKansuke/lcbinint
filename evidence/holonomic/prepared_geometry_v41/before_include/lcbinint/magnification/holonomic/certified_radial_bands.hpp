#pragma once

// Faithful radial-support search used by the algebraic boundary solver, fed
// by a complete certified image-component seed set.  Every connected image
// component has a seed, and the continuous image-radius projection of a
// connected component is an interval; no blind solidity scan is required.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "lcbinint/magnification/holonomic/boundary_polynomial.hpp"
#include "lcbinint/magnification/holonomic/radius_terms.hpp"

namespace lcbinint::holonomic {

struct CertifiedRadialBands {
    std::vector<std::pair<double,double>> bands;
    bool reliable=true;
    const char* reason="";
    int radius_probes=0;
    int tangency_accepts=0;
    int tangency_fallbacks=0;
};

namespace certified_band_detail {
enum class Image { No,Yes,Uncertain };
inline Image has_image(double R,const PrimaryFrame&pf,int&calls) {
    ++calls;if(!(R>0.0))return Image::No;
    const auto a=arc_intervals(R,pf);
    if(a.kind==ArcKind::kFull)return Image::Yes;
    if(a.kind==ArcKind::kArcs)return a.arcs.empty()?Image::No:Image::Yes;
    if(a.kind==ArcKind::kEmpty)return Image::No;
    return Image::Uncertain;
}
inline bool tangency(double lo,double hi,double R,double t,
                     const PrimaryFrame&pf,double&root) {
    R=std::clamp(R,lo,hi);
    for(int it=0;it<12;++it) {
        const auto g=local_fold_quantities<double>(R,t,pf);
        const double det=g.PR*g.Ptt-g.Pt*g.Ptr;
        const double scale=std::fabs(g.PR*g.Ptt)+std::fabs(g.Pt*g.Ptr)+1e-300;
        if(!std::isfinite(det)||std::fabs(det)<=
           64*std::numeric_limits<double>::epsilon()*scale)return false;
        double dr=(g.P*g.Ptt-g.Pt*g.Pt)/det;
        double dt=(g.PR*g.Pt-g.Ptr*g.P)/det;
        dr=std::clamp(dr,-.45*(hi-lo),.45*(hi-lo));dt=std::clamp(dt,-.75,.75);
        R-=dr;t-=dt;
        if(!(R>lo&&R<hi)||!std::isfinite(R+t))return false;
        if(std::fabs(dr)<=2e-12*(1+std::fabs(R))&&std::fabs(dt)<=2e-11) {
            const auto q=local_fold_quantities<double>(R,t,pf);
            if(std::fabs(q.P)<=2e-9*(1+std::fabs(q.PR))&&
               std::fabs(q.Pt)<=2e-8*(1+std::fabs(q.Ptt))){root=R;return true;}
        }
    }
    return false;
}
}

inline CertifiedRadialBands certified_radial_bands(
    const PrimaryFrame&pf,const std::vector<double>&seed_radii) {
    using namespace certified_band_detail;CertifiedRadialBands out;
    constexpr int bisect_total=16,bisect_prefilter=3;
    for(double seed:seed_radii) {
        if(!(seed>0.0))continue;
        if(std::any_of(out.bands.begin(),out.bands.end(),[&](auto b){return seed>=b.first&&seed<=b.second;}))continue;
        auto si=has_image(seed,pf,out.radius_probes);
        if(si!=Image::Yes){out.reliable=false;out.reason=si==Image::Uncertain?"seed topology uncertain":"certified seed inactive";return out;}
        auto side=[&](int dir,double&edge) {
            double active=seed,step=std::max(pf.rho,1e-10*(1+seed)),outside=active;bool found=false;
            for(int k=0;k<64;++k){double c=dir<0?std::max(0.0,active-step):active+step;if(c==active){outside=c;found=true;break;}auto s=has_image(c,pf,out.radius_probes);if(s==Image::Uncertain)return false;if(s==Image::No||c==0.0){outside=c;found=true;break;}active=c;step*=2;}
            if(!found)return false;if(outside==active){edge=outside;return true;}
            double lo=std::min(active,outside),hi=std::max(active,outside);int n=0;
            for(;n<bisect_prefilter;++n){double m=.5*(lo+hi);auto s=has_image(m,pf,out.radius_probes);if(s==Image::Uncertain)return false;bool in=s==Image::Yes;if(dir<0){if(in)hi=m;else lo=m;}else{if(in)lo=m;else hi=m;}}
            const double ar=dir<0?hi:lo,orr=dir<0?lo:hi;const auto arcs=arc_intervals(ar,pf);
            if(arcs.kind==ArcKind::kArcs)for(const auto&a:arcs.arcs)for(double theta:{a[0],a[1]}){
                double t=std::tan(.5*theta),candidate=0;if(!std::isfinite(t)||!tangency(lo,hi,ar,t,pf,candidate))continue;
                auto ia=has_image(.5*(candidate+ar),pf,out.radius_probes),io=has_image(.5*(candidate+orr),pf,out.radius_probes);
                if(ia==Image::Yes&&io==Image::No){edge=candidate;++out.tangency_accepts;return true;}
            }
            ++out.tangency_fallbacks;
            for(;n<bisect_total;++n){double m=.5*(lo+hi);auto s=has_image(m,pf,out.radius_probes);if(s==Image::Uncertain)return false;bool in=s==Image::Yes;if(dir<0){if(in)hi=m;else lo=m;}else{if(in)lo=m;else hi=m;}}
            edge=dir<0?hi:lo;return true;
        };
        double lo=seed,hi=seed;if(!side(-1,lo)||!side(1,hi)){out.reliable=false;out.reason="radial support search failed";return out;}if(hi>lo)out.bands.push_back({lo,hi});
    }
    std::sort(out.bands.begin(),out.bands.end());std::vector<std::pair<double,double>> merged;
    for(auto b:out.bands){double tol=1e-11*(1+std::fabs(b.second));if(merged.empty()||b.first>merged.back().second+tol)merged.push_back(b);else merged.back().second=std::max(merged.back().second,b.second);}out.bands=std::move(merged);
    if(out.bands.empty()){out.reliable=false;out.reason="no radial bands";}return out;
}
} // namespace lcbinint::holonomic

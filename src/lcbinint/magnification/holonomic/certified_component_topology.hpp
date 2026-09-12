#pragma once

// Research topology front end using the complete image-component seeds from
// the classical inverse-ray support certificate.  Unlike fast_topology.hpp,
// this does not assume that every finite-source component contains an image
// of the source centre.  The caller supplies all certified image seeds,
// including components born where the source disk intersects a caustic.

#include <cmath>
#include <vector>

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/fast_bands.hpp"

namespace lcbinint::holonomic {

struct CertifiedComponentTopologyStats {
    bool support_proven=false;
    bool bands_reliable=false;
    int seed_count=0;
    int component_seed_count=0;
    int band_count=0;
    int radius_probes=0;
    const char* reason="";
};

// A straight image-plane segment contained in phi>0 is a constructive proof
// that its endpoints belong to one connected finite-source image component.
// Failure to prove connection only leaves duplicate seeds; it never merges
// components.  This cheaply collapses the dense source-boundary probe ladder
// before radial support marching.
inline std::vector<PointImage> certified_component_representatives(
    const PrimaryFrame& pf,const std::vector<PointImage>& input) {
    const int n=static_cast<int>(input.size());
    std::vector<int> parent(n);for(int i=0;i<n;++i)parent[i]=i;
    auto root=[&](int x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
    auto join=[&](int a,int b){a=root(a);b=root(b);if(a!=b)parent[b]=a;};
    const double margin=256*std::numeric_limits<double>::epsilon()*
                        std::max(pf.rho*pf.rho,1e-300);
    for(int i=0;i<n;++i)for(int j=0;j<i;++j) {
        bool inside=true;
        for(int k=1;k<8;++k) {
            const double f=k/8.0;
            const double x=(1-f)*input[i].z.re+f*input[j].z.re;
            const double y=(1-f)*input[i].z.im+f*input[j].z.im;
            const double R=std::hypot(x,y);
            if(!(R>0)||phi_val(R,std::atan2(y,x),pf)<=margin){inside=false;break;}
        }
        if(inside)join(i,j);
    }
    std::vector<PointImage> out;
    for(int i=0;i<n;++i)if(root(i)==i)out.push_back(input[i]);
    return out;
}

inline TopologyResult classify_certified_component_bands(
    const PrimaryFrame& pf,const std::vector<Cplx<double>>& image_seeds,
    bool support_proven,CertifiedComponentTopologyStats* stats=nullptr) {
    CertifiedComponentTopologyStats local;
    auto& st=stats?*stats:local;
    st.support_proven=support_proven;
    st.seed_count=static_cast<int>(image_seeds.size());
    TopologyResult out;
    if(!support_proven || image_seeds.empty()) {
        out.status=Status::TOPOLOGY_UNCERTAIN;
        st.reason=!support_proven?"component support not proven":"no image seeds";
        return out;
    }
    PointImages seeds;
    seeds.reliable=true;
    for(const auto& z:image_seeds) {
        const double radius=std::hypot(z.re,z.im);
        if(!(radius>0.0) || !std::isfinite(radius))continue;
        bool duplicate=false;
        for(const auto& old:seeds.images)
            if(std::hypot(old.z.re-z.re,old.z.im-z.im)<=
               1e-10*(1.0+radius)){duplicate=true;break;}
        if(!duplicate)seeds.images.push_back({z,radius,0.0,0.0,false});
    }
    std::sort(seeds.images.begin(),seeds.images.end(),
              [](const PointImage& a,const PointImage& b){return a.radius<b.radius;});
    seeds.images=certified_component_representatives(pf,seeds.images);
    std::sort(seeds.images.begin(),seeds.images.end(),
              [](const PointImage& a,const PointImage& b){return a.radius<b.radius;});
    st.component_seed_count=static_cast<int>(seeds.images.size());
    // fast_bands normally carries only a point-image completeness assumption.
    // Here its seeds come from cached_binary_image_seeds(), whose caustic and
    // source-boundary probes certify every finite-source image component.
    // Its solidity pass is therefore the desired D14-free radial-support
    // planner, including image-free notches skipped by a doubling march.
    // The planner only needs a bracket fine enough for the local P=P_t event
    // Newton below.  Spending 26 boolean quartic solves per side duplicates
    // endpoint precision work; eight leaves a narrow certified bracket and
    // the adaptive event ladder performs the final DD/qf correction.
    const FastBands bands=fast_bands(pf,seeds,4,true);
    st.radius_probes=bands.has_image_calls;
    st.bands_reliable=bands.reliable;
    if(!bands.reliable) {
        out.status=Status::TOPOLOGY_UNCERTAIN;
        st.reason=bands.reason;
        return out;
    }
    std::vector<double> edges{0.0};
    for(const auto& b:bands.bands)if(b.r_hi>b.r_lo) {
        if(b.r_lo>0.0)edges.push_back(b.r_lo);
        edges.push_back(b.r_hi);
    }
    std::sort(edges.begin(),edges.end());
    edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    if(!edges.empty())out.r_max=edges.back();
    // The adaptive integrator requires a contiguous partition.  Classify each
    // open interval once at its midpoint: a gap is an exact kEmpty cell, not
    // an arcs cell whose runtime evaluation is then rejected as an ArcKind
    // mismatch.  The solidified band endpoints guarantee that no unsampled
    // image-free notch remains between these edges.
    for(std::size_t i=1;i<edges.size();++i)if(edges[i]>edges[i-1]) {
        const double mid=0.5*(edges[i-1]+edges[i]);
        const ArcSet arcs=arc_intervals(mid,pf);
        if(arcs.kind==ArcKind::kDegenerate) {
            out.cells.clear();out.status=Status::TOPOLOGY_UNCERTAIN;
            st.reason="midpoint chart/root ambiguity";return out;
        }
        const ArcKind kind=(arcs.kind==ArcKind::kArcs&&arcs.arcs.empty())
            ? ArcKind::kEmpty:arcs.kind;
        out.cells.push_back({static_cast<int>(out.cells.size()),edges[i-1],edges[i],
                             mid,kind,-1,Status::OK});
    }
    st.band_count=static_cast<int>(out.cells.size());
    st.reason="certified component bands";
    out.status=out.cells.empty()?Status::TOPOLOGY_UNCERTAIN:Status::OK;
    return out;
}

} // namespace lcbinint::holonomic

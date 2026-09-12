#pragma once

// Research topology front end using the complete image-component seeds from
// the classical inverse-ray support certificate.  Unlike fast_topology.hpp,
// this does not assume that every finite-source component contains an image
// of the source centre.  The caller supplies all certified image seeds,
// including components born where the source disk intersects a caustic.

#include <cmath>
#include <vector>

#include "lcbinint/magnification/holonomic/cells.hpp"
#include "lcbinint/magnification/holonomic/certified_radial_bands.hpp"

namespace lcbinint::holonomic {

struct CertifiedComponentTopologyStats {
    bool support_proven=false;
    bool bands_reliable=false;
    int seed_count=0;
    int band_count=0;
    int radius_probes=0;
    const char* reason="";
};

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
    std::vector<double> radii;for(const auto&s:seeds.images)radii.push_back(s.radius);
    CertifiedRadialBands bands=certified_radial_bands(pf,radii);
    st.radius_probes=bands.radius_probes;
    st.bands_reliable=bands.reliable;
    if(!bands.reliable) {
        out.status=Status::TOPOLOGY_UNCERTAIN;
        st.reason=bands.reason;
        return out;
    }
    std::vector<double> edges{0.0};
    for(const auto& b:bands.bands)if(b.second>b.first) {
        if(b.first>0.0)edges.push_back(b.first);
        edges.push_back(b.second);
    }
    std::sort(edges.begin(),edges.end());
    edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    if(!edges.empty())out.r_max=edges.back();
    // Include the empty gaps as cells. flux_adaptive_integrate requires a
    // contiguous partition; the dynamic-node classifier returns exact zero
    // there while the certified band endpoints prevent a narrow component
    // from being skipped by the radial estimator.
    for(std::size_t i=1;i<edges.size();++i)if(edges[i]>edges[i-1])
        out.cells.push_back({static_cast<int>(out.cells.size()),edges[i-1],edges[i],
                             0.5*(edges[i-1]+edges[i]),ArcKind::kArcs,-1,Status::OK});
    st.band_count=static_cast<int>(out.cells.size());
    st.reason="certified component bands";
    out.status=out.cells.empty()?Status::TOPOLOGY_UNCERTAIN:Status::OK;
    return out;
}

} // namespace lcbinint::holonomic

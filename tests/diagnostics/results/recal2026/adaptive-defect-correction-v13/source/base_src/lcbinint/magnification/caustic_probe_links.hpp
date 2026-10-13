#pragma once

#include "lcbinint/types.hpp"
#include "lcbinint/math/outward_interval.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace lcbinint::magnification::detail {

// Geometric proposal relative to inflated sampled caustics, NOT an enclosure
// of the exact caustic. Second differences provide a scale, not a proved
// interpolation remainder. Consumers requiring certification must also check
// the lens discriminant along the proposed source link.
class CausticProbeLinks {
    struct Box { double xmin,xmax,ymin,ymax; };
    struct Block { Box box; std::size_t first,last; };
    std::vector<Box> edges_;
    std::vector<Block> blocks_;
    bool valid_=true;
    static bool overlap(Box a,Box b) {
        return a.xmin<=b.xmax && b.xmin<=a.xmax && a.ymin<=b.ymax && b.ymin<=a.ymax;
    }
public:
    explicit CausticProbeLinks(const std::vector<std::vector<SourcePosition>>& branches) {
        for (const auto& branch:branches) {
            const auto n=branch.size();
            if (n<3) { valid_=false; continue; }
            std::vector<double> margins(n);
            for (std::size_t i=0;i<n;++i) {
                const auto& a=branch[(i+n-1)%n]; const auto& b=branch[i]; const auto& c=branch[(i+1)%n];
                margins[i]=std::hypot(a.x-2*b.x+c.x,a.y-2*b.y+c.y);
                if (!std::isfinite(b.x) || !std::isfinite(b.y) || !std::isfinite(margins[i])) valid_=false;
            }
            for (std::size_t i=0;i<n;++i) {
                const auto& a=branch[i]; const auto& b=branch[(i+1)%n];
                const double m=std::max(margins[i],margins[(i+1)%n]);
                edges_.push_back({math::down(std::min(a.x,b.x)-m),math::up(std::max(a.x,b.x)+m),
                                  math::down(std::min(a.y,b.y)-m),math::up(std::max(a.y,b.y)+m)});
            }
        }
        // Coarse boxes cheaply exclude whole stretches of the sampled curve.
        for (std::size_t first=0;first<edges_.size();first+=32) {
            const auto last=std::min(first+32,edges_.size()); Box b=edges_[first];
            for (auto i=first+1;i<last;++i) {
                b.xmin=std::min(b.xmin,edges_[i].xmin); b.xmax=std::max(b.xmax,edges_[i].xmax);
                b.ymin=std::min(b.ymin,edges_[i].ymin); b.ymax=std::max(b.ymax,edges_[i].ymax);
            }
            blocks_.push_back({b,first,last});
        }
    }
    bool proposed_clear(SourcePosition a,SourcePosition b) const {
        if (!valid_) return false;
        using I=math::OutwardInterval;
        const Box query{std::min(a.x,b.x),std::max(a.x,b.x),std::min(a.y,b.y),std::max(a.y,b.y)};
        const I dx=I(b.x)-I(a.x),dy=I(b.y)-I(a.y);
        for (const auto& block:blocks_) {
            if (!overlap(query,block.box)) continue;
            for (auto i=block.first;i<block.last;++i) {
                const auto& box=edges_[i];
                if (!overlap(query,box)) continue;
                const I cross=dx*(I(box.ymin,box.ymax)-I(a.y))-dy*(I(box.xmin,box.xmax)-I(a.x));
                if (!cross.finite() || cross.contains_zero()) return false;
            }
        }
        return true;
    }
};

} // namespace lcbinint::magnification::detail

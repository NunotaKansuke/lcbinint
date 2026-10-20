#pragma once
#include "lcbinint/magnification/binary_caustic_enclosure.hpp"

namespace lcbinint::magnification::detail {

// Each node covers a continuous phase interval and a separation interval.
// Refine only intervals whose source-plane disks cannot clear the query disk.
// Budget exhaustion refuses the shortcut and retains the full solver.
class BinaryCausticTree {
    struct Node {
        double center=0.0, half_width=0.0;
        int depth=0, left=-1, right=-1;
        unsigned long long generation=0;
        std::vector<CausticEnclosureDisk> disks;
        std::vector<Complex> root_centers;
        std::vector<double> image_radii;
    };
    double separation_=0.0,q_=0.0,width_=0.0;
    std::vector<Node> nodes_;
    unsigned long long generation_=0;
    SourcePosition witness_source_{};
    double witness_clearance_=0.0;

    double visit(int index,SourcePosition source,double required,int& budget) {
        if(nodes_[index].generation!=generation_) {
            if(budget--<=0)return 0.0;
            const auto center=nodes_[index].center,half=nodes_[index].half_width;
            nodes_[index].disks=cover_binary_caustic_phase(
                separation_,q_,center,half,width_,&nodes_[index].root_centers,true,true,&nodes_[index].image_radii);
            nodes_[index].generation=generation_;
        }
        if(!nodes_[index].disks.empty()) {
            const double lower=binary_caustic_enclosure_clearance(nodes_[index].disks,source);
            if(lower>required)return lower;
            // This is only a conservative refusal, never an acceptance test.
            // It avoids refining a phase whose anchor already enters the disk.
            for(const auto& disk:nodes_[index].disks)
                if(std::hypot(source.x-disk.first.x,source.y-disk.first.y)<required)
                    return 0.0;
        }
        if(nodes_[index].depth>=12)return 0.0;
        if(nodes_[index].left<0) {
            if(nodes_.size()+2>8192)return 0.0;
            const auto center=nodes_[index].center,half=nodes_[index].half_width/2.0;
            const int depth=nodes_[index].depth+1,left=static_cast<int>(nodes_.size());
            const auto centers=nodes_[index].root_centers;
            Node first{center-half,half,depth},second{center+half,half,depth};
            first.root_centers=centers;second.root_centers=centers;
            nodes_.push_back(std::move(first));nodes_.push_back(std::move(second));
            nodes_[index].left=left;nodes_[index].right=left+1;
        }
        const int left=nodes_[index].left,right=nodes_[index].right;
        const double first=visit(left,source,required,budget);
        if(!(first>required))return 0.0;
        return std::min(first,visit(right,source,required,budget));
    }
public:
    bool enclosure_at(double separation,double q,double phase,double half,
        std::vector<CausticEnclosureDisk>& disks,std::vector<Complex>& roots,
        std::vector<double>& radii) const {
        if(separation_!=separation || q_!=q || width_!=0 || nodes_.empty())return false;
        constexpr double pi=3.14159265358979323846;
        int index=std::min(15,std::max(0,int(phase*16/pi)));
        const double tolerance=32*std::numeric_limits<double>::epsilon();
        while(index>=0) {
            const auto& node=nodes_[index];
            if(std::abs(node.center-phase)<tolerance &&
               std::abs(node.half_width-half)<tolerance) {
                if(node.generation!=generation_ || node.disks.empty() ||
                   node.image_radii.size()!=4)return false;
                disks=node.disks;roots=node.root_centers;radii=node.image_radii;
                return true;
            }
            if(node.half_width<=half || node.left<0)return false;
            index=phase<node.center?node.left:node.right;
        }
        return false;
    }

    void clear() {nodes_.clear();witness_clearance_=0.0;}

    double clearance(double separation,double q,SourcePosition source,
                     double required,double width) {
        if(!(separation>0.0 && q>0.0 && required>=0.0 && width>=0.0 && width<separation) ||
           !std::isfinite(separation+q+required+width+source.x+source.y))return 0.0;
        constexpr double padding=512.0*std::numeric_limits<double>::epsilon();
        const double drift=std::abs(separation-separation_)+
            padding*(1.0+separation+separation_);
        if(q_!=q || nodes_.empty() || !(separation==separation_ || drift<=width_)) {
            if(q_!=q || nodes_.empty()) {
                nodes_.clear();
                constexpr double pi=3.14159265358979323846;
                for(int k=0;k<16;++k)nodes_.push_back({2.0*pi*(k+0.5)/32.0,pi/32.0,0});
            }
            separation_=separation;q_=q;width_=width;witness_clearance_=0.0;
            if(++generation_==0) {
                generation_=1;
                for(auto& node:nodes_)node.generation=0;
            }
        }
        const double moved=std::hypot(source.x-witness_source_.x,source.y-witness_source_.y);
        const double available=witness_clearance_-moved-
            padding*(1.0+witness_clearance_+moved+std::abs(source.x)+std::abs(source.y));
        if(available>required)return available;
        int budget=128;
        double lower=std::numeric_limits<double>::infinity();
        for(int k=0;k<16;++k) {
            lower=std::min(lower,visit(k,source,required,budget));
            if(!(lower>required))return 0.0;
        }
        witness_source_=source;witness_clearance_=lower;
        return lower;
    }
};

struct BinaryCausticTrees {BinaryCausticTree moving,exact;};
inline BinaryCausticTrees& binary_caustic_trees() {
    static thread_local BinaryCausticTrees cache;
    return cache;
}
inline void clear_binary_caustic_tree_cache() {
    auto& cache=binary_caustic_trees();cache.moving.clear();cache.exact.clear();
}

inline double binary_caustic_disk_clearance(
    double separation,double mass_ratio,SourcePosition source,double required,double source_radius)
{
    if(!(required>=0.0 && source_radius>=0.0) ||
       !std::isfinite(required+source_radius+source.x+source.y))return 0.0;
    double lower=std::max(binary_caustic_strip_clearance(separation,mass_ratio,source),
                         binary_caustic_radial_clearance(separation,mass_ratio,source));
    if(lower>required)return lower;
    auto& cache=binary_caustic_trees();
    const double width=std::min(0.125*source_radius,0.0005*std::abs(separation));
    lower=std::max(lower,cache.moving.clearance(separation,mass_ratio,source,required,width));
    if(lower>required)return lower;
    return std::max(lower,cache.exact.clearance(separation,mass_ratio,source,required,0.0));
}

} // namespace lcbinint::magnification::detail

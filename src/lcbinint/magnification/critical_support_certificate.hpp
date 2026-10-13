#pragma once
#include "lcbinint/magnification/critical_phase_enclosure_fast.hpp"
#include "lcbinint/magnification/critical_cluster_enclosure.hpp"
#include <map>
#include <vector>
#include <memory>
#include <complex>

namespace lcbinint::magnification {
std::vector<SourcePosition> binary_critical_phase_proposals(double,double,double);

struct CriticalSupportWitness {
    SourcePosition position;
    double inside_radius=0;
};
struct CriticalSupportCertificate {
    bool resolved=true;
    std::vector<CriticalSupportWitness> witnesses;
    std::vector<std::array<int,3>> unresolved; // phase index, bins, root mask
    int phase_queries=0,proof_queries=0,source_visits=0;
    int outside_roots=0,inside_roots=0,monotone_roots=0,point_failures=0,match_failures=0;
    std::vector<std::size_t> witness_parent;
    std::map<std::array<int,3>,std::size_t> vertex_witness;
};
struct CriticalSupportGeometryTube {
    int index=0,bins=0,root=0;
    SourcePosition centre,source_lower,source_upper;
    double radius=0;
};

// Geometry-only tubes survive source changes. Source classification retains
// accepted root intervals and refines only the unresolved root mask.
class CriticalSupportCache {
    using I=critical_enclosure_detail::I;
    using C=critical_enclosure_detail::C;
    struct Node {
        std::array<SourcePosition,4> roots{};
        CriticalPhaseEnclosure tube,point;
        std::array<bool,4> slope_ready{},curve_ready{};
        std::array<critical_enclosure_detail::C,4> half_phase{};
        std::array<I,4> amplitude{};
        Node* next=nullptr;
        std::array<int,4> next_roots{{-2,-2,-2,-2}};
        std::array<Node*,2> children{};
        std::array<std::array<int,16>,2> child_masks{};
        std::array<bool,2> child_masks_ready{};
        std::array<std::array<int,4>,2> child_parent{};
        std::array<SourcePosition,4> curve_lower{},curve_upper{};
        bool clusters_ready=false;
        std::vector<CriticalClusterTube> clusters;
    };
    struct CacheLimit {};
    double s_,q_;
    bool geometry_dirty_=false;
    std::map<std::pair<int,int>,Node> nodes_;
    std::map<std::pair<int,int>,std::array<SourcePosition,4>> anchors_;
    // Numerical proposals only. The same interval root-count proof accepts
    // them, so a failed continuation never weakens the certificate.
    bool propose_from_anchor(int index,int bins,std::array<SourcePosition,4>& roots) {
        if (bins<=64 || index%2==0) return false;
        auto parent=anchors_.find({index/2,bins/2});
        int ai=index/2,ab=bins/2;
        while (parent==anchors_.end() && ab>64 && ai%2==0) {
            ai/=2;ab/=2;parent=anchors_.find({ai,ab});
        }
        if (parent==anchors_.end()) return false;
        const double qi=std::abs(q_),q=qi<1 ? qi : 1/qi;
        const double m1=1/(1+q),m2=q*m1,a=qi<1 ? -std::abs(s_) : std::abs(s_);
        const std::complex<double> target=std::polar(1.0,2*3.141592653589793*index/bins);
        for (int j=0;j<4;++j) {
            std::complex<double> z(parent->second[j].x,parent->second[j].y);
            bool converged=false;
            for (int k=0;k<12;++k) {
                const auto u=1.0/(z-a),v=1.0/z;
                const auto p=m1*u*u,t=m2*v*v,derivative=-2.0*(p*u+t*v);
                const auto dz=(p+t-target)/derivative;z-=dz;
                if (!(std::isfinite(z.real()) && std::isfinite(z.imag()))) return false;
                if (std::abs(dz)<8*std::numeric_limits<double>::epsilon()*(1+std::abs(z))) {converged=true;break;}
            }
            if (!converged) return false;
            roots[j]={z.real(),z.imag()};
        }
        return true;
    }
    Node& node(int index,int bins,CriticalSupportCertificate& out) {
        index%=bins;
        // Dyadic anchors share proposals even when their interval lengths differ.
        auto key=std::make_pair(index,bins);
        auto found=nodes_.find(key);
        if (found!=nodes_.end()) return found->second;
        if (nodes_.size()>=32768) throw CacheLimit{};
        Node n;
        int ai=index,ab=bins;
        while (ab>64 && ai%2==0) { ai/=2;ab/=2; }
        auto anchor=anchors_.find({ai,ab});
        if (anchor!=anchors_.end()) n.roots=anchor->second;
        else {
            const bool continued=propose_from_anchor(ai,ab,n.roots);
            const auto continued_point=continued ? binary_critical_phase_enclosure_fast(s_,q_,ai,ab,n.roots,0.0) : CriticalPhaseEnclosure{};
            if (continued) ++out.proof_queries;
            if (continued_point.resolved && ai==index && ab==bins) n.point=continued_point;
            if (!continued_point.resolved) {
                auto roots=binary_critical_phase_proposals(s_,q_,2.0*3.141592653589793*ai/ab);
                ++out.phase_queries;
                if (roots.size()==4) std::copy(roots.begin(),roots.end(),n.roots.begin());
            }
            anchors_[{ai,ab}]=n.roots;
        }
        n.tube=binary_critical_phase_enclosure_fast(s_,q_,index,bins,n.roots);
        ++out.proof_queries;
        if (!n.point.resolved) {
            n.point=binary_critical_phase_enclosure_fast(s_,q_,index,bins,n.roots,0.0);
            ++out.proof_queries;
        }
        n.curve_lower=n.tube.source_lower;n.curve_upper=n.tube.source_upper;
        geometry_dirty_=true;
        return nodes_.emplace(key,n).first->second;
    }
    void enclose_curve(Node& n,int j,int index,int bins) {
        if (n.curve_ready[j] || !n.tube.resolved || !n.point.resolved) return;
        n.curve_ready[j]=true;
        // Only ambiguous source classifications need this geometric derivative.
        // A full physical disk already classified inside/outside is sufficient.
        radial_slope(n,j,index,bins,{0,0});
        const I step(0,(I(2)*critical_enclosure_detail::pi()/I(bins)).hi);
        const I rate=n.amplitude[j]/I(2);
        const I x=I(n.point.source_lower[j].x,n.point.source_upper[j].x)+step*rate*n.half_phase[j].x;
        const I y=I(n.point.source_lower[j].y,n.point.source_upper[j].y)-step*rate*n.half_phase[j].y;
        n.curve_lower[j].x=std::max(n.curve_lower[j].x,x.lo);
        n.curve_upper[j].x=std::min(n.curve_upper[j].x,x.hi);
        n.curve_lower[j].y=std::max(n.curve_lower[j].y,y.lo);
        n.curve_upper[j].y=std::min(n.curve_upper[j].y,y.hi);
        geometry_dirty_=true;
    }
    static I curve_distance(const Node& n,int j,SourcePosition src) {
        return math::square(I(n.curve_lower[j].x,n.curve_upper[j].x)-I(src.x))+
               math::square(I(n.curve_lower[j].y,n.curve_upper[j].y)-I(src.y));
    }
    void tighten(Node& n) {
        for (auto child:n.children) if (child) tighten(*child);
        if (!n.tube.resolved || !n.children[0] || !n.children[1] ||
            !n.child_masks_ready[0] || !n.child_masks_ready[1]) return;
        for (int j=0;j<4;++j) {
            int matched[2]={-1,-1};
            for (int c=0;c<2;++c) {
                if (!n.children[c]->tube.resolved) continue;
                for (int k=0;k<4;++k) if (n.child_parent[c][k]==j) matched[c]=k;
            }
            if (matched[0]<0 || matched[1]<0) continue;
            const auto a=n.children[0],b=n.children[1];
            n.curve_lower[j].x=std::max(n.curve_lower[j].x,std::min(a->curve_lower[matched[0]].x,b->curve_lower[matched[1]].x));
            n.curve_lower[j].y=std::max(n.curve_lower[j].y,std::min(a->curve_lower[matched[0]].y,b->curve_lower[matched[1]].y));
            n.curve_upper[j].x=std::min(n.curve_upper[j].x,std::max(a->curve_upper[matched[0]].x,b->curve_upper[matched[1]].x));
            n.curve_upper[j].y=std::min(n.curve_upper[j].y,std::max(a->curve_upper[matched[0]].y,b->curve_upper[matched[1]].y));
        }
    }
    static I distance(const CriticalPhaseEnclosure& p,int j,SourcePosition src) {
        return math::square(I(p.source_lower[j].x,p.source_upper[j].x)-I(src.x))+
               math::square(I(p.source_lower[j].y,p.source_upper[j].y)-I(src.y));
    }
    static std::size_t root(CriticalSupportCertificate& out,std::size_t i) {
        while (out.witness_parent[i]!=i) {
            out.witness_parent[i]=out.witness_parent[out.witness_parent[i]];i=out.witness_parent[i];
        }
        return i;
    }
    static void attach(CriticalSupportCertificate& out,int index,int bins,int j,std::size_t witness) {
        index%=bins;
        while (bins>64 && index%2==0) {index/=2;bins/=2;}
        const auto key=std::array<int,3>{index,bins,j};
        const auto inserted=out.vertex_witness.emplace(key,witness);
        if (!inserted.second) out.witness_parent[root(out,witness)]=root(out,inserted.first->second);
    }
    int next_root(Node& n,int j,int index,int bins,CriticalSupportCertificate& out) {
        if (n.next_roots[j]!=-2) return n.next_roots[j];
        Node& end=n.next ? *n.next : *(n.next=&node(index+1,bins,out));
        n.next_roots[j]=-1;
        if (end.point.resolved) for (int k=0;k<4;++k) {
            const double gap=critical_enclosure_detail::magnitude({
                I(end.roots[k].x)-I(n.roots[j].x),I(end.roots[k].y)-I(n.roots[j].y)}).hi;
            if ((I(gap)+I(end.point.radii[k])).hi<n.tube.radii[j]) n.next_roots[j]=k;
        }
        return n.next_roots[j];
    }
    void retain(Node& n,int j,int index,int bins,SourcePosition source,I radius2,
                CriticalSupportWitness witness,bool all_inside,CriticalSupportCertificate& out) {
        const auto id=out.witnesses.size();out.witnesses.push_back(witness);out.witness_parent.push_back(id);
        if (all_inside || (n.point.resolved && distance(n.point,j,source).hi<radius2.lo))
            attach(out,index,bins,j,id);
        const int k=next_root(n,j,index,bins,out);
        if (k>=0 && (all_inside || distance(n.next->point,k,source).hi<radius2.lo))
            attach(out,index+1,bins,k,id);
    }
    I radial_slope(Node& n,int j,int index,int bins,SourcePosition source) {
        if (n.slope_ready[j]) {
            const C half=n.half_phase[j];
            const I fx=I(n.curve_lower[j].x,n.curve_upper[j].x)-I(source.x);
            const I fy=I(n.curve_lower[j].y,n.curve_upper[j].y)-I(source.y);
            return n.amplitude[j]*(fx*half.x-fy*half.y);
        }
        using namespace critical_enclosure_detail;
        const double qi=std::abs(q_);
        const I q=qi<1 ? I(qi) : I(1)/I(qi),m1=I(1)/(I(1)+q),m2=q*m1;
        const double a=qi<1 ? -std::abs(s_) : std::abs(s_);
        const double r=n.tube.radii[j];
        const C z{I(n.roots[j].x)+I(-r,r),I(n.roots[j].y)+I(-r,r)};
        const C u=reciprocal(subtract(z,{I(a),I(0)})),v=reciprocal(z);
        const C p=scale(multiply(u,u),m1),t=scale(multiply(v,v),m2);
        const C derivative=scale(add(multiply(p,u),multiply(t,v)),I(-2));
        const double reach=(I(2)*pi()/I(bins)).hi;
        const C phase=add(unit_phase(index,bins),{I(-reach,reach),I(-reach,reach)});
        const C dz=multiply({-phase.y,phase.x},reciprocal(derivative));
        const double half_reach=(pi()/I(bins)).hi;
        const C half=add(unit_phase(index,2*bins),{I(-half_reach,half_reach),I(-half_reach,half_reach)});
        const I fx=I(n.curve_lower[j].x,n.curve_upper[j].x)-I(source.x);
        const I fy=I(n.curve_lower[j].y,n.curve_upper[j].y)-I(source.y);
        // dzeta = 2 Re(dz exp(i phi/2)) exp(-i phi/2).
        // Factoring the two real projections avoids subtracting two broad
        // complex rectangles near a cusp.
        n.slope_ready[j]=true;n.half_phase[j]=half;n.amplitude[j]=I(4)*(dz.x*half.x-dz.y*half.y);
        return n.amplitude[j]*(fx*half.x-fy*half.y);
    }
    void visit(int index,int bins,int depth,int mask,SourcePosition source,double rho,
               CriticalSupportCertificate& out,Node* existing=nullptr) {
        ++out.source_visits;
        Node& n=existing ? *existing : node(index,bins,out);
        const I radius2=math::square(I(rho));
        if (!n.tube.resolved) {
            if (!n.clusters_ready) {
                n.clusters_ready=true;n.clusters=binary_critical_cluster_enclosure(s_,q_,index,bins,n.roots);
            }
            bool classified=!n.clusters.empty();
            for (const auto& c:n.clusters) {
                const I d=math::square(I(c.source_lower.x,c.source_upper.x)-I(source.x))+
                          math::square(I(c.source_lower.y,c.source_upper.y)-I(source.y));
                if (d.lo>radius2.hi) continue;
                if (d.hi<radius2.lo) {
                    out.witness_parent.push_back(out.witnesses.size());out.witnesses.push_back({c.centre,c.radius});
                }
                else classified=false;
            }
            if (classified) return;
        }
        int pending=0;
        for (int j=0;j<4;++j) if (mask&(1<<j)) {
            if (!n.tube.resolved) {pending|=1<<j;continue;}
            I d=curve_distance(n,j,source);
            if (!(d.lo>radius2.hi || d.hi<radius2.lo)) {
                enclose_curve(n,j,index,bins);d=curve_distance(n,j,source);
            }
            if (d.lo>radius2.hi) {++out.outside_roots;continue;}
            if (d.hi<radius2.lo) {
                if (n.point.resolved && distance(n.point,j,source).hi<radius2.lo) {
                    ++out.inside_roots;retain(n,j,index,bins,source,radius2,{n.roots[j],n.point.radii[j]},true,out);continue;
                }
                if (distance(n.tube,j,source).hi<radius2.lo) {
                    ++out.inside_roots;retain(n,j,index,bins,source,radius2,{n.roots[j],n.tube.radii[j]},true,out);continue;
                }
            }
            const I slope=radial_slope(n,j,index,bins,source);
            if (!slope.finite() || slope.contains_zero() || !n.point.resolved) {
                pending|=1<<j;continue;
            }
            // The left endpoint needs no correspondence guess: its tiny root
            // disk is contained in this phase tube by the strict disk test.
            if (!(n.point.radii[j]<n.tube.radii[j])) {pending|=1<<j;continue;}
            const I left=distance(n.point,j,source);
            if (slope.lo>0 && left.lo>radius2.hi) {++out.monotone_roots;continue;}
            if (left.hi<radius2.lo) {
                // A monotone radial function has one contiguous inside arc,
                // attached to this certified inside endpoint.
                ++out.monotone_roots;retain(n,j,index,bins,source,radius2,{n.roots[j],n.point.radii[j]},false,out);continue;
            }
            const int k=next_root(n,j,index,bins,out);
            bool attached=false;
            if (k>=0) {
                Node& end=*n.next;
                const I right=distance(end.point,k,source);
                if (slope.hi<0 && right.lo>radius2.hi) attached=true;
                if (right.hi<radius2.lo) {
                    retain(n,j,index,bins,source,radius2,{end.roots[k],end.point.radii[k]},false,out);attached=true;
                }
            }
            if (attached) {++out.monotone_roots;continue;}
            pending|=1<<j;
        }
        if (!pending) return;
        if (depth>=14 || out.source_visits>=8192) {
            out.resolved=false;out.unresolved.push_back({index,bins,pending});return;
        }
        // Match child root disks to the parent tube by strict containment.
        // A failed match keeps the entire child mask: never guess labels.
        for (int child=0;child<2;++child) {
            Node& c=n.children[child] ? *n.children[child] : *(n.children[child]=&node(2*index+child,2*bins,out));
            if (!n.child_masks_ready[child]) {
                n.child_masks_ready[child]=true;
                std::array<int,4> correspondence{};correspondence.fill(-1);
                if (n.tube.resolved && c.point.resolved) for (int k=0;k<4;++k)
                    for (int j=0;j<4;++j) {
                        const I dx=I(c.roots[k].x)-I(n.roots[j].x),dy=I(c.roots[k].y)-I(n.roots[j].y);
                        const double gap=critical_enclosure_detail::magnitude({dx,dy}).hi;
                        if ((I(gap)+I(c.point.radii[k])).hi<n.tube.radii[j]) correspondence[k]=j;
                    }
                n.child_parent[child]=correspondence;geometry_dirty_=true;
                for (int pending_mask=0;pending_mask<16;++pending_mask) {
                    int child_mask=0;
                    for (int k=0;k<4;++k)
                        if (correspondence[k]<0 || (pending_mask&(1<<correspondence[k]))) child_mask|=1<<k;
                    n.child_masks[child][pending_mask]=child_mask;
                }
            }
            const int child_mask=n.child_masks[child][pending];
            if (child_mask) visit(2*index+child,2*bins,depth+1,child_mask,source,rho,out,&c);
        }
    }
public:
    CriticalSupportCache(double s,double q):s_(s),q_(q) {}
    std::vector<CriticalSupportGeometryTube> audit_tubes(std::size_t maximum=256) const {
        std::vector<CriticalSupportGeometryTube> result;
        const std::size_t stride=std::max<std::size_t>(1,nodes_.size()/std::max<std::size_t>(1,maximum));
        std::size_t ordinal=0;
        for (const auto& entry:nodes_) {
            const auto& n=entry.second;
            if (ordinal++%stride || !n.tube.resolved) continue;
            for (int j=0;j<4;++j) result.push_back({entry.first.first,entry.first.second,j,n.roots[j],n.curve_lower[j],n.curve_upper[j],n.tube.radii[j]});
        }
        return result;
    }
    CriticalSupportCertificate classify(SourcePosition source,double rho) {
        CriticalSupportCertificate out;
        if (!critical_enclosure_detail::supported_arithmetic() ||
            !std::isfinite(s_) || !std::isfinite(q_) || s_==0 || q_==0 ||
            !(std::isfinite(source.x) && std::isfinite(source.y) && std::isfinite(rho) && rho>0)) {
            out.resolved=false;return out;
        }
        for (int i=0;i<64;++i) {
            try {visit(i,64,0,15,source,rho,out);}
            catch (const CacheLimit&) {out.resolved=false;out.unresolved.push_back({i,64,15});}
        }
        if (geometry_dirty_) {
            for (int i=0;i<64;++i) {auto found=nodes_.find({i,64});if(found!=nodes_.end()) tighten(found->second);}
            geometry_dirty_=false;
        }
        // Inside disks are connected subsets of the physical image support.
        // Proved overlap permits one witness for their union, without roots.
        std::map<std::pair<double,double>,CriticalSupportWitness> unique;
        for (std::size_t i=0;i<out.witnesses.size();++i) {
            if (root(out,i)!=i) continue;
            const auto& w=out.witnesses[i];
            auto& existing=unique[{w.position.x,w.position.y}];
            if (w.inside_radius>=existing.inside_radius) existing=w;
        }
        out.witnesses.clear();
        for (const auto& entry:unique) out.witnesses.push_back(entry.second);
        const auto n=out.witnesses.size();std::vector<std::size_t> parent(n);
        for (std::size_t i=0;i<n;++i) parent[i]=i;
        const auto root=[&](std::size_t i) {while(parent[i]!=i) i=parent[i];return i;};
        for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<i;++j) {
            const auto& a=out.witnesses[i];const auto& b=out.witnesses[j];
            if (root(i)==root(j)) continue;
            // This rejection is only a work gate. A false rejection retains
            // an extra seed; every accepted merge still uses interval bounds.
            if (std::hypot(a.position.x-b.position.x,a.position.y-b.position.y)>
                (a.inside_radius+b.inside_radius)*1.001) continue;
            const double gap=critical_enclosure_detail::magnitude({I(a.position.x)-I(b.position.x),I(a.position.y)-I(b.position.y)}).hi;
            if (gap<(I(a.inside_radius)+I(b.inside_radius)).lo) parent[root(i)]=root(j);
        }
        std::vector<CriticalSupportWitness> reduced;
        for (std::size_t i=0;i<n;++i) if (root(i)==i) reduced.push_back(out.witnesses[i]);
        out.witnesses=std::move(reduced);return out;
    }
};
inline CriticalSupportCache& cached_critical_support_geometry(double s,double q) {
    thread_local std::unique_ptr<CriticalSupportCache> cache;
    thread_local double last_s=0,last_q=0;
    if (!cache || s!=last_s || q!=last_q) {cache=std::make_unique<CriticalSupportCache>(s,q);last_s=s;last_q=q;}
    return *cache;
}
inline CriticalSupportCertificate cached_critical_support_certificate(double s,double q,SourcePosition source,double rho) {
    return cached_critical_support_geometry(s,q).classify(source,rho);
}
} // namespace lcbinint::magnification

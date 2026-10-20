#pragma once
#include "lcbinint/magnification/binary_caustic_tree.hpp"

namespace lcbinint::magnification::detail {

struct BinaryLocalSeedResult {
    std::vector<SourcePosition> seeds;
    bool completed = false;
    int nodes = 0, monotone_arcs = 0, interior_arcs = 0;
};

// Enumerate critical-curve arcs inside the source disk without constructing
// global caustic polylines. Unknown enclosures or turning points fail closed.
inline BinaryLocalSeedResult binary_local_image_seeds(
    double separation, double mass_ratio, SourcePosition source, double rho,
    int maximum_nodes = 192, int maximum_depth = 18)
{
    BinaryLocalSeedResult out;
    if(!(separation>0 && mass_ratio>0 && rho>0) ||
       !std::isfinite(separation+mass_ratio+rho+source.x+source.y))return out;
    const double q=std::min(mass_ratio,1.0/mass_ratio);
    const double a=mass_ratio<1 ? -separation : separation;
    const double m1=1/(1+q),m2=q*m1;
    constexpr double pi=3.14159265358979323846;
    constexpr double eps=2048*std::numeric_limits<double>::epsilon();
    const Complex target(source.x,source.y);
    struct Node {double phase,half;int depth;};
    std::vector<Node> pending;
    for(int k=15;k>=0;--k)pending.push_back({pi*(k+.5)/16,pi/32,0});
    auto inside=[&](const CausticEnclosureDisk& d) {
        return std::hypot(d.first.x-source.x,d.first.y-source.y)+d.second <
            rho-eps*(1+rho+std::abs(source.x)+std::abs(source.y));
    };
    auto outside=[&](const CausticEnclosureDisk& d) {
        return std::hypot(d.first.x-source.x,d.first.y-source.y)-d.second >
            rho+eps*(1+rho+std::abs(source.x)+std::abs(source.y));
    };
    auto append=[&](Complex z) {
        // Return an image in the point solver's barycentric frame.
        out.seeds.push_back({z.real(),z.imag()});
    };
    while(!pending.empty()) {
        if(++out.nodes>maximum_nodes)return out;
        const Node node=pending.back();pending.pop_back();
        std::vector<Complex> roots;
        std::vector<double> radii;
        std::vector<CausticEnclosureDisk> disks;
        if(!binary_caustic_trees().exact.enclosure_at(
            separation,mass_ratio,node.phase,node.half,disks,roots,radii))
            disks=cover_binary_caustic_phase(separation,mass_ratio,
                node.phase,node.half,0,&roots,true,true,&radii);
        bool resolved=!disks.empty();
        std::vector<Complex> accepted;
        if(resolved)for(int side:{1,-1})for(int i=0;i<4;++i) {
            const int di=i+(side<0?4:0);
            if(outside(disks[di]))continue;
            const Complex z=side>0?roots[i]:std::conj(roots[i]);
            if(inside(disks[di])) {accepted.push_back(z);++out.interior_arcs;continue;}
            // k(z)=exp(i*phase), z'=i*exp(i*phase)/k'(z).
            // A root disk bounds k', then z', the mapped tangent, and g'.
            const double radius=radii[i];
            const double d1=std::abs(z-a),d2=std::abs(z);
            const Complex kprime=-2.0*m1/std::pow(z-a,3)-2.0*m2/std::pow(z,3);
            const double error=6*radius*(m1/std::pow(d1-radius,4)+m2/std::pow(d2-radius,4))+
                eps*(1+std::abs(kprime)+2*m1/std::pow(d1-radius,3)+2*m2/std::pow(d2-radius,3));
            const double lower=std::abs(kprime)-error;
            if(!(lower>0)) {resolved=false;break;}
            const Complex phase=std::polar(1.0,side*node.phase);
            const double phase_error=2*std::sin(node.half/2)+eps;
            const Complex velocity=Complex(0,1)*phase/kprime;
            const double velocity_error=phase_error/lower+error/(std::abs(kprime)*lower)+eps*(1+std::abs(velocity));
            const Complex tangent=velocity+std::conj(phase)*std::conj(velocity);
            const double tangent_error=2*velocity_error+phase_error*std::abs(velocity)+eps*(1+std::abs(tangent));
            const Complex mapped(disks[di].first.x,disks[di].first.y);
            const double derivative=2*std::real(std::conj(mapped-target)*tangent);
            const double derivative_error=2*(disks[di].second*std::abs(tangent)+
                (std::abs(mapped-target)+disks[di].second)*tangent_error)+eps*(1+std::abs(derivative));
            if(!(std::abs(derivative)>derivative_error)) {resolved=false;break;}
            // The minimum-distance endpoint decides whether this monotone
            // arc enters the disk. Certify that endpoint in the same root disk.
            const double endpoint=side*node.phase-(derivative>0?1:-1)*node.half;
            std::vector<Complex> end_roots;
            std::vector<double> end_radii;
            const auto end_disks=cover_binary_caustic_phase(separation,mass_ratio,
                endpoint,0,0,&end_roots,true,true,&end_radii);
            if(end_disks.empty()) {resolved=false;break;}
            int match=-1;
            for(int j=0;j<4;++j)
                if(std::abs(end_roots[j]-z)+end_radii[j]<radius) {
                    if(match!=-1){match=-2;break;}match=j;
                }
            if(match<0) {resolved=false;break;}
            if(inside(end_disks[match]))accepted.push_back(end_roots[match]);
            else if(!outside(end_disks[match])) {resolved=false;break;}
            ++out.monotone_arcs;
        }
        if(resolved) {for(auto z:accepted)append(z);continue;}
        if(node.depth>=maximum_depth)return out;
        const double half=node.half/2;
        pending.push_back({node.phase+half,half,node.depth+1});
        pending.push_back({node.phase-half,half,node.depth+1});
    }
    out.completed=true;
    return out;
}

} // namespace lcbinint::magnification::detail

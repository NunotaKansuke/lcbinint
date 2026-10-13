#pragma once
#include "lcbinint/magnification/critical_phase_enclosure.hpp"
#include <vector>

namespace lcbinint::magnification {
struct BinaryBoundaryEnclosure {
    bool resolved=false;
    double lower=0,upper=1;
    int iterations=0;
};

// A horizontal inside-to-outside segment. Certifies its unique limb root,
// not the area integral or completeness of image discovery.
inline BinaryBoundaryEnclosure binary_boundary_enclosure(
    double s,double qi,SourcePosition source,double rho,
    double inside_x,double outside_x,double y,double proposal)
{
    using namespace critical_enclosure_detail;
    BinaryBoundaryEnclosure result;
    for (double v : {s,qi,source.x,source.y,rho,inside_x,outside_x,y,proposal})
        if (!std::isfinite(v)) return result;
    if (qi==0 || rho<=0 || inside_x==outside_x ||
        std::fegetround()!=FE_TONEAREST || !std::numeric_limits<double>::is_iec559)
        return result;
    qi=std::abs(qi);
    const I q=qi<1 ? I(qi) : I(1)/I(qi);
    const I m1=I(1)/(I(1)+q),m2=q*m1;
    const double a=qi<1 ? -std::abs(s) : std::abs(s);
    const I step=I(outside_x)-I(inside_x);
    auto evaluate=[&](I t) {
        const C z{I(inside_x)+t*step,I(y)};
        const C u=reciprocal(subtract(z,{I(a),I(0)})),v=reciprocal(z);
        const C f=subtract(subtract(subtract(z,scale({u.x,-u.y},m1)),
                                  scale({v.x,-v.y},m2)),
                           {I(a)*m1+I(source.x),I(source.y)});
        const C k=add(scale(multiply(u,u),m1),scale(multiply(v,v),m2));
        const I g=math::square(f.x)+math::square(f.y)-math::square(I(rho));
        const I derivative=I(2)*(f.x*(I(1)+k.x)-f.y*k.y)*step;
        return std::array<I,2>{g,derivative};
    };
    const auto left=evaluate(I(0)),right=evaluate(I(1)),whole=evaluate(I(0,1));
    if (!left[0].finite() || !right[0].finite() || !whole[1].finite() ||
        !(left[0].hi<0 && right[0].lo>0 && whole[1].lo>0)) return result;
    // Endpoint signs establish existence; strict monotonicity establishes
    // uniqueness. Interval Newton then contracts the already proven bracket.
    I bracket(0,1);
    for (int n=0;n<6;++n) {
        const double p=n==0 ? std::clamp(proposal,bracket.lo,bracket.hi)
                            : bracket.lo+(bracket.hi-bracket.lo)*0.5;
        const auto point=evaluate(I(p)),range=evaluate(bracket);
        const I next=I(p)-point[0]/range[1];
        if (!next.finite()) break;
        const I contracted(std::max(bracket.lo,next.lo),std::min(bracket.hi,next.hi));
        if (contracted.lo>contracted.hi) return result;
        ++result.iterations;
        if (contracted.lo==bracket.lo && contracted.hi==bracket.hi) break;
        bracket=contracted;
    }
    result.lower=bracket.lo;result.upper=bracket.hi;result.resolved=true;
    return result;
}
struct BinaryBoundaryTube {
    bool resolved=false;
    double slope_lower=0,slope_upper=0;
    double curvature_lower=0,curvature_upper=0;
    double midpoint_area_error_upper=0;
};

// Uniform-brightness geometric contribution of ONE boundary graph x(y).
// Endpoint signs for every y and nonzero g_x prove a unique graph in the
// supplied rectangle. This excludes turning points and is not a global flux
// certificate or a remainder bound for Bennett's outer-coordinate stencil.
inline BinaryBoundaryTube binary_boundary_tube(
    double s,double qi,SourcePosition source,double rho,
    double xlo,double xhi,double ylo,double yhi)
{
    using namespace critical_enclosure_detail;
    BinaryBoundaryTube out;
    for (double v:{s,qi,source.x,source.y,rho,xlo,xhi,ylo,yhi})
        if (!std::isfinite(v)) return out;
    if (!supported_arithmetic() || qi==0 || rho<=0 || xlo>=xhi || ylo>=yhi) return out;
    qi=std::abs(qi);
    const I q=qi<1 ? I(qi) : I(1)/I(qi);
    const I m1=I(1)/(I(1)+q),m2=q*m1;
    const double a=qi<1 ? -std::abs(s) : std::abs(s);
    const auto dot=[](C u,C v) {return u.x*v.x+u.y*v.y;};
    const auto evaluate=[&](I x,I y) {
        const C z{x,y};
        const C u=reciprocal(subtract(z,{I(a),I(0)})),v=reciprocal(z);
        const C f=subtract(subtract(subtract(z,scale({u.x,-u.y},m1)),
            scale({v.x,-v.y},m2)),{I(a)*m1+I(source.x),I(source.y)});
        const C k=add(scale(multiply(u,u),m1),scale(multiply(v,v),m2));
        const C kp=scale(add(scale(multiply(multiply(u,u),u),m1),
            scale(multiply(multiply(v,v),v),m2)),I(-2));
        const C fx{I(1)+k.x,-k.y},fy{-k.y,I(1)-k.x};
        const C fxx{kp.x,-kp.y},fxy{-kp.y,-kp.x},fyy{-kp.x,kp.y};
        return std::array<I,6>{dot(f,f)-math::square(I(rho)),
            I(2)*dot(f,fx),I(2)*dot(f,fy),
            I(2)*(dot(fx,fx)+dot(f,fxx)),
            I(2)*(dot(fx,fy)+dot(f,fxy)),
            I(2)*(dot(fy,fy)+dot(f,fyy))};
    };
    const I ys(ylo,yhi);
    const auto left=evaluate(I(xlo),ys),right=evaluate(I(xhi),ys);
    const auto whole=evaluate(I(xlo,xhi),ys);
    for (auto value:whole) if (!value.finite()) return out;
    if (!left[0].finite() || !right[0].finite()) return out;
    const bool increasing=whole[1].lo>0 && left[0].hi<0 && right[0].lo>0;
    const bool decreasing=whole[1].hi<0 && left[0].lo>0 && right[0].hi<0;
    if (!increasing && !decreasing) return out;
    const I slope=-whole[2]/whole[1];
    const I curvature=-(whole[5]+I(2)*whole[4]*slope+whole[3]*math::square(slope))/whole[1];
    const I height=I(yhi)-I(ylo);
    const I error=height*height*height*I(curvature.magnitude())/I(24);
    if (!slope.finite() || !curvature.finite() || !error.finite()) return out;
    out.resolved=true;out.slope_lower=slope.lo;out.slope_upper=slope.hi;
    out.curvature_lower=curvature.lo;out.curvature_upper=curvature.hi;
    out.midpoint_area_error_upper=error.hi;
    return out;
}
struct BinaryBoundaryStrip {
    bool resolved=false;
    double area_lower=0,area_upper=0;
    int certified_substrips=0,unresolved_substrips=0;
};

// Integral between two supplied boundary graphs. Caller must establish that
// these are the only boundaries of the run; this local helper cannot discover
// missing images. Subdivision retains completed subintervals.
inline BinaryBoundaryStrip binary_boundary_strip(double s,double q,SourcePosition source,double rho,
    double left_lo,double left_hi,double right_lo,double right_hi,
    double ylo,double yhi,int maximum_depth=4)
{
    using namespace critical_enclosure_detail;
    BinaryBoundaryStrip out;
    if (!(left_lo<left_hi && left_hi<right_lo && right_lo<right_hi && ylo<yhi)) return out;
    maximum_depth=std::clamp(maximum_depth,0,8);
    struct Part {double lo,hi;int depth;};
    std::vector<Part> pending{{ylo,yhi,0}};
    I total(0);
    while (!pending.empty()) {
        const auto part=pending.back();pending.pop_back();
        const double mid=part.lo+(part.hi-part.lo)*0.5;
        const auto left=binary_boundary_tube(s,q,source,rho,left_lo,left_hi,part.lo,part.hi);
        const auto right=binary_boundary_tube(s,q,source,rho,right_lo,right_hi,part.lo,part.hi);
        const auto lp=left.resolved && right.resolved
            ? binary_boundary_enclosure(s,q,source,rho,left_hi,left_lo,mid,0.5) : BinaryBoundaryEnclosure{};
        const auto rp=left.resolved && right.resolved
            ? binary_boundary_enclosure(s,q,source,rho,right_lo,right_hi,mid,0.5) : BinaryBoundaryEnclosure{};
        if (left.resolved && right.resolved && lp.resolved && rp.resolved) {
            const I lx=I(left_hi)+I(lp.lower,lp.upper)*(I(left_lo)-I(left_hi));
            const I rx=I(right_lo)+I(rp.lower,rp.upper)*(I(right_hi)-I(right_lo));
            const I exact_mid=(I(part.lo)+I(part.hi))/I(2);
            const I midpoint_offset=I((I(mid)-exact_mid).magnitude());
            const double slope_bound=I(left.slope_lower,left.slope_upper).magnitude()
                + I(right.slope_lower,right.slope_upper).magnitude();
            const I error=I(left.midpoint_area_error_upper)+I(right.midpoint_area_error_upper)
                +(I(part.hi)-I(part.lo))*midpoint_offset*I(math::up(slope_bound));
            const I area=(rx-lx)*(I(part.hi)-I(part.lo));
            total=total+area+I(-error.hi,error.hi);
            ++out.certified_substrips;
        } else if (part.depth<maximum_depth && mid>part.lo && mid<part.hi) {
            pending.push_back({mid,part.hi,part.depth+1});
            pending.push_back({part.lo,mid,part.depth+1});
        } else ++out.unresolved_substrips;
    }
    out.resolved=out.unresolved_substrips==0 && total.finite();
    // Partial sums are diagnostic only whenever any substrip is unresolved.
    out.area_lower=total.lo;out.area_upper=total.hi;
    return out;
}
// Area of the actual source preimage clipped to a supplied rectangle.
// At regular horizontal turning points g_y stays nonzero: use vertical
// monotonicity instead of requiring a graph x(y). A finite, loose enclosure
// remains valid when monotonicity fails; it never means tolerance acceptance.
struct BinaryBoundaryCap {
    bool bounded=false;
    double area_lower=0,area_upper=0;
    int columns=0,monotone_columns=0,evaluations=0;
    bool extent_checked=false,lower_face_outside=false,left_face_outside=false,right_face_outside=false;
};
inline BinaryBoundaryCap binary_boundary_cap(double s,double qi,SourcePosition source,double rho,
    double xlo,double xhi,double ylo,double yhi,int maximum_columns=32,bool check_extent=false)
{
    using namespace critical_enclosure_detail;
    BinaryBoundaryCap out;
    for (double v:{s,qi,source.x,source.y,rho,xlo,xhi,ylo,yhi})
        if (!std::isfinite(v)) return out;
    if (!supported_arithmetic() || qi==0 || rho<=0 || xlo>=xhi || ylo>=yhi) return out;
    qi=std::abs(qi);
    const I q=qi<1 ? I(qi) : I(1)/I(qi);
    const I m1=I(1)/(I(1)+q),m2=q*m1;
    const double a=qi<1 ? -std::abs(s) : std::abs(s);
    auto evaluate=[&](I x,I y,bool derivative) {
        ++out.evaluations;
        const C z{x,y},u=reciprocal(subtract(z,{I(a),I(0)})),v=reciprocal(z);
        const C f=subtract(subtract(subtract(z,scale({u.x,-u.y},m1)),
            scale({v.x,-v.y},m2)),{I(a)*m1+I(source.x),I(source.y)});
        const I g=math::square(f.x)+math::square(f.y)-math::square(I(rho));
        I gy,gx;
        if (derivative) {
            const C k=add(scale(multiply(u,u),m1),scale(multiply(v,v),m2));
            gy=I(2)*(-f.x*k.y+f.y*(I(1)-k.x));
            gx=I(2)*(f.x*(I(1)+k.x)-f.y*k.y);
        }
        return std::array<I,3>{g,gy,gx};
    };
    struct Column {double lo,hi;I area;bool monotone;bool splittable=true;};
    const I height=I(yhi)-I(ylo);
    auto bound=[&](double lo,double hi) {
        const I width=I(hi)-I(lo),full=width*height;
        Column c{lo,hi,I(0,full.hi),false};
        const auto range=evaluate(I(lo,hi),I(ylo,yhi),true);
        if (range[0].finite() && range[0].hi<0) {c.area=full;c.splittable=false;return c;}
        if (range[0].finite() && range[0].lo>0) {c.area=I(0);c.splittable=false;return c;}
        if (!range[1].finite() || range[1].contains_zero()) return c;
        const double mid=ylo+(yhi-ylo)*0.5;
        const double xmid=lo+(hi-lo)*0.5;
        const I natural=evaluate(I(lo,hi),I(mid),false)[0];
        const I centre=evaluate(I(xmid),I(mid),false)[0];
        // Mean-value form cancels the large dependency error at a turning
        // point, where g_x is small. Intersect two independently valid forms.
        const I centred=centre+range[2]*(I(lo,hi)-I(xmid));
        const I gmid(std::max(natural.lo,centred.lo),std::min(natural.hi,centred.hi));
        if (gmid.lo>gmid.hi) return c;
        if (!gmid.finite()) return c;
        // Mean value theorem encloses every in-box transition in this
        // interval. If a column is wholly inside/outside, clipping the
        // interval to row edges also encloses its saturated transition:
        // the derivative lower bound forces the corresponding interval
        // endpoint beyond that edge. No existence of a root is assumed.
        const I roots=I(mid)-gmid/range[1];
        if (!roots.finite()) return c;
        const I clipped(std::clamp(roots.lo,ylo,yhi),std::clamp(roots.hi,ylo,yhi));
        const I thickness=range[1].lo>0 ? clipped-I(ylo) : I(yhi)-clipped;
        const I area=width*thickness;
        c.area=I(std::max(0.0,area.lo),std::min(full.hi,area.hi));
        c.monotone=true;return c;
    };
    maximum_columns=std::clamp(maximum_columns,1,256);
    std::vector<Column> leaves{bound(xlo,xhi)};
    while (static_cast<int>(leaves.size())<maximum_columns) {
        std::size_t largest=leaves.size();
        for (std::size_t j=0;j<leaves.size();++j)
            if (leaves[j].splittable && (largest==leaves.size() ||
                leaves[j].area.hi-leaves[j].area.lo > leaves[largest].area.hi-leaves[largest].area.lo))
                largest=j;
        if (largest==leaves.size()) break;
        const auto parent=leaves[largest];
        if (parent.area.hi-parent.area.lo <= 0) break;
        const double mid=parent.lo+(parent.hi-parent.lo)*0.5;
        if (!(mid>parent.lo && mid<parent.hi)) {leaves[largest].splittable=false;continue;}
        leaves[largest]=bound(parent.lo,mid);
        leaves.push_back(bound(mid,parent.hi));
    }
    I total(0);
    for (const auto& c:leaves) {total=total+c.area;out.monotone_columns+=c.monotone;}
    if (check_extent) {
        out.extent_checked=true;out.lower_face_outside=true;
        for (const auto& c:leaves) {
            const I face=evaluate(I(c.lo,c.hi),I(ylo),false)[0];
            if (!face.finite() || face.lo<=0) out.lower_face_outside=false;
        }
        const I left=evaluate(I(xlo),I(ylo,yhi),false)[0];
        const I right=evaluate(I(xhi),I(ylo,yhi),false)[0];
        out.left_face_outside=left.finite() && left.lo>0;
        out.right_face_outside=right.finite() && right.lo>0;
    }
    out.columns=static_cast<int>(leaves.size());out.bounded=total.finite();
    out.area_lower=std::max(0.0,total.lo);out.area_upper=total.hi;
    return out;
}
struct BoundaryPairError {
    bool bounded=false;
    double signed_lower=0,signed_upper=0;
    double absolute_upper=0,separate_absolute_upper=0;
};
// Conditional on valid input area intervals. Preserve signed residuals until
// after combining the two corrected contributions; no boundary re-evaluation.
inline BoundaryPairError boundary_pair_error(double first,double second,
    double first_lower,double first_upper,double second_lower,double second_upper)
{
    using namespace critical_enclosure_detail;
    BoundaryPairError out;
    for (double v:{first,second,first_lower,first_upper,second_lower,second_upper})
        if (!std::isfinite(v)) return out;
    if (!supported_arithmetic() || first_lower>first_upper || second_lower>second_upper) return out;
    const I a=I(first)-I(first_lower,first_upper);
    const I b=I(second)-I(second_lower,second_upper);
    const I sum=a+b,separate=I(a.magnitude())+I(b.magnitude());
    if (!sum.finite() || !separate.finite()) return out;
    out.bounded=true;out.signed_lower=sum.lo;out.signed_upper=sum.hi;
    out.absolute_upper=sum.magnitude();out.separate_absolute_upper=separate.hi;
    return out;
}
} // namespace lcbinint::magnification

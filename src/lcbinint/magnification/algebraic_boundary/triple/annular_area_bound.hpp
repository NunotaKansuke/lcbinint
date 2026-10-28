#pragma once
#include "angular_topology.hpp"
namespace lcbinint::algebraic_boundary::triple {
// Bound angular image coverage over an entire radial band. The radial lens
// map is Lipschitz away from poles: |df/dR| <= 1+sum(m/distance^2).
// Thus every original image angle is inside the inflated source at mid-R.
// All arithmetic that constructs the inflation is outward binary128.
inline double annular_support_fraction(double lo,double hi,const Geometry& p) {
    using Q=__float128;using I=interval_detail::Interval<Q>;
    using interval_detail::up;using interval_detail::down;
    if(!(hi>lo && lo>0))return 1;
    const std::array<I,3> mass{I(1)-I(p.m1)-I(p.m2),I(p.m1),I(p.m2)};
    const auto lenses=p.lenses();Q lipschitz=1;
    for(int j=0;j<3;++j) {
        // The constrained primary mass can straddle zero at a rounding
        // boundary. The triangle bound uses absolute masses even there.
        const Q magnitude=std::max(fabsq(mass[j].lo),fabsq(mass[j].hi));
        if(magnitude==0)continue;
        const I x(lenses[j].real()),y(lenses[j].imag()),norm=x*x+y*y;
        const Q rlo=down(sqrtq(std::max(Q(0),norm.lo))),rhi=up(sqrtq(norm.hi));
        Q distance=0;
        if(Q(lo)>rhi)distance=down(Q(lo)-rhi);
        else if(Q(hi)<rlo)distance=down(rlo-Q(hi));
        else return 1;
        const Q square=down(distance*distance);
        if(!(square>0))return 1;
        lipschitz=up(lipschitz+up(magnitude/square));
    }
    const Q enlarged=up(Q(p.rho)+up(lipschitz*up(Q(hi)-Q(lo))));
    Geometry cover=p;cover.rho=double(enlarged);
    if(Q(cover.rho)<enlarged)cover.rho=std::nextafter(cover.rho,INFINITY);
    if(!std::isfinite(cover.rho))return 1;
    AngularOptions options;options.maximum_boxes=2048;
    const auto topology=angular_topology(lo+(hi-lo)*.5,cover,options);
    if(!topology.completed)return 1;
    if(topology.arcs.empty())return 0;
    // Include rounded arc midpoints, circular-seam shifts and the sum of
    // bracket widths; these binary64 bookkeeping operations precede this
    // outward accumulation.
    I length(topology.angular_measure_error);
    length=length+I(256*std::numeric_limits<double>::epsilon()*(1+2*std::acos(-1.)));
    for(auto arc:topology.arcs)length=length+(I(arc.hi)-I(arc.lo));
    const Q pi=down(acosq(Q(-1)));
    const Q fraction=up(length.hi/(2*pi));
    double result=double(fraction);if(Q(result)<fraction)result=std::nextafter(result,INFINITY);
    return std::clamp(result,0.,1.);
}
}

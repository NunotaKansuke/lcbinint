#include "lcbinint/magnification/bennett_integration.hpp"
#include "lcbinint/magnification/binary_image_discriminant.hpp"
#include "lcbinint/magnification/point_source_magnifier.hpp"
#include "lcbinint/math/outward_interval.hpp"

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

using namespace lcbinint::magnification;

int main()
{
    using I=lcbinint::math::OutwardInterval;
    using Q=boost::multiprecision::cpp_bin_float_quad;
    std::mt19937 rng(20261012);
    std::uniform_real_distribution<double> random(-3.0,3.0);
    for (double value: {0.0, -0.0, std::numeric_limits<double>::denorm_min(),
         -std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max(),
         -std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()}) {
        if (lcbinint::math::up(value)!=std::nextafter(value,std::numeric_limits<double>::infinity())
            || lcbinint::math::down(value)!=std::nextafter(value,-std::numeric_limits<double>::infinity()))
            return 15;
    }
    // Independent higher precision arithmetic checks the binary64 enclosures.
    for (int i=0;i<500;++i) {
        const double a=random(rng), b=random(rng);
        if (lcbinint::math::up(a)!=std::nextafter(a,std::numeric_limits<double>::infinity())
            || lcbinint::math::down(a)!=std::nextafter(a,-std::numeric_limits<double>::infinity())) return 16;
        const I intervals[]={I(a)+I(b),I(a)-I(b),I(a)*I(b),I(a)/I(b)};
        const Q exact[]={Q(a)+Q(b),Q(a)-Q(b),Q(a)*Q(b),Q(a)/Q(b)};
        for (int j=0;j<4;++j)
            if (!(Q(intervals[j].lo)<=exact[j] && exact[j]<=Q(intervals[j].hi))) return 1;
    }
    if (binary_image_count_discriminant(1,1,{0,0}).status!=DiscriminantStatus::symmetry_axis)
        return 2;
    if (binary_image_count_discriminant(0,1,{1,1}).status!=DiscriminantStatus::invalid_geometry)
        return 3;
    const int old_round=std::fegetround();
    std::fesetround(FE_UPWARD);
    const auto unsupported=binary_image_count_discriminant(1,1,{1,1});
    std::fesetround(old_round);
    if (unsupported.status!=DiscriminantStatus::unsupported_arithmetic) return 4;
    // Equal binary's central source is inside its resonant caustic; far field
    // is outside. Check both signs against these geometric limits.
    if (binary_image_count_discriminant(1,1,{0,0.1}).image_count!=5) return 5;
    if (binary_image_count_discriminant(1,1,{2,1}).image_count!=3) return 6;

    PointSourceMagnifier roots;
    int resolved=0, unresolved=0, inside=0;
    for (double s: {0.6,1.0,1.3,2.0}) {
        for (double q: {1.0,0.1,0.001,0.00001,10.0}) {
            for (int i=0;i<100;++i) {
                const double scale=i<50 ? 1.0 : 0.05;
                const lcbinint::SourcePosition source{random(rng)*scale,random(rng)*scale};
                const auto count=binary_image_count_discriminant(s,q,source);
                if (count.image_count<0) { ++unresolved; continue; }
                ++resolved;
                inside += count.image_count==5;
                const int reference=roots.binary_mag0(s,q,source).image_count;
                if (count.image_count!=reference) {
                    std::cerr << "count mismatch: " << s << " " << q << " "
                              << source.x << " " << source.y << ": "
                              << count.image_count << " vs " << reference << "\n";
                    return 7;
                }
                if (binary_image_count_discriminant(s,q,{source.x,-source.y}).image_count
                    !=count.image_count) return 8;
            }
        }
    }
    if (resolved<100 || inside<10) return 9;

    using namespace lcbinint::magnification::detail;
    for (double delta: {0.01,0.05,0.15,0.25,0.9}) {
        for (double cutoff: {0.0,0.15,1.0}) {
            const double a=0.3,b=0.7,c=-0.2;
            const double f1=a+b*std::sqrt(delta)+c*delta;
            const double f2=a+b*std::sqrt(delta+1)+c*(delta+1);
            const auto model=bennett_boundary_model_residual(delta,f1,f2,a,cutoff);
            const auto w=bennett_boundary_weights(delta,cutoff);
            const double length=delta+0.5;
            const double integral=a*length+(2.0/3.0)*b*std::pow(length,1.5)+0.5*c*length*length;
            const double quadrature=w.limb*a+w.node*f1;
            if (!model.resolved || std::abs(model.signed_residual-(integral-quadrature))>2e-14)
                return 10;
        }
    }
    const auto exact_sqrt=bennett_boundary_model_residual(0.25,0.5,std::sqrt(1.25),0,0.15);
    if (!exact_sqrt.resolved || std::abs(exact_sqrt.signed_residual)>2e-15) return 11;
    const auto guarded=bennett_boundary_model_residual(0.05,std::sqrt(0.05),std::sqrt(1.05),0,0.15);
    if (!guarded.resolved || std::abs(guarded.signed_residual)<1e-3) return 12;
    if (bennett_boundary_model_residual(0,0,1,0,0.15).resolved) return 13;
    if (!bennett_boundary_model_residual(0,1,1,1,0.15).resolved) return 14;
    std::cout << "resolved=" << resolved << " unresolved=" << unresolved
              << " inside=" << inside << "; boundary model identities passed\n";
    return 0;
}

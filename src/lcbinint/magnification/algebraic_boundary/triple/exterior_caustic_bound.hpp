#pragma once
#include "geometry.hpp"
#include "interval.hpp"
#include <algorithm>
#include <utility>

#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math","fp-contract=off")
#endif
namespace lcbinint::algebraic_boundary::triple {
// Enclose ALL caustics, without finding critical-curve roots. Let delta be
// the minimum lens separation, M the total positive mass, and r the distance
// of a critical point to its nearest lens j. Since |sum m/(z-l)^2|=1,
// r<=sqrt(M). If r>=delta/2, the total deflection is <=2M/delta. Otherwise
// all other poles are >=delta/2 away and
//   m_j/r^2 <= 1+4(M-m_j)/delta^2.
// Thus the caustic lies in the union of lens-centred disks of radius max(
// sqrt(M)+2M/delta,
// min(sqrt(M),delta/2)+sqrt(m_j*(1+4(M-m_j)/delta^2))+2(M-m_j)/delta).
// All arithmetic below is outward. An escaping ray disjoint from these
// disks proves winding zero, hence four images, for a positive triple lens.
// A mere "outside all disks" test would not prove escape from a union's hole.
class ExteriorCausticBound {
    using I=interval_detail::Interval<double>;
    std::array<Complex,3> lenses_{};
    double radius_=INFINITY;
    static double sqrt_upper(I a) {
        if(!a.valid() || a.hi<0)return INFINITY;
        return interval_detail::up(std::sqrt(a.hi));
    }
    static double norm_lower(I x,I y) {
        const I n=x*x+y*y;
        if(!(n.lo>0))return 0;
        return std::max(0.,interval_detail::down(std::sqrt(n.lo)));
    }
    static double absolute_lower(I x) {
        return x.lo>0?x.lo:x.hi<0?-x.hi:0;
    }
public:
    ExteriorCausticBound()=default;
    ExteriorCausticBound(const std::array<Complex,3>& lenses,
                         const std::array<double,3>& masses):lenses_(lenses) {
        if(!interval_detail::environment_ok())return;
        I total;
        for(int j=0;j<3;++j) {
            if(!(masses[j]>0) || !std::isfinite(masses[j]) ||
               !std::isfinite(lenses[j].real()) || !std::isfinite(lenses[j].imag()))return;
            total=total+I(masses[j]);
        }
        double delta=INFINITY;
        for(int j=0;j<3;++j)for(int k=0;k<j;++k)
            delta=std::min(delta,norm_lower(I(lenses[j].real())-I(lenses[k].real()),
                                             I(lenses[j].imag())-I(lenses[k].imag())));
        if(!(delta>0) || !std::isfinite(delta) || !total.valid())return;
        const I inverse_delta(interval_detail::down(1/delta),interval_detail::up(1/delta));
        if(!inverse_delta.valid())return;
        const double root_mass=sqrt_upper(total);
        double bound=(I(root_mass)+I(2)*total*inverse_delta).hi;
        const double nearest=std::min(root_mass,interval_detail::up(delta*.5));
        for(int j=0;j<3;++j) {
            I other;
            for(int k=0;k<3;++k)if(k!=j)other=other+I(masses[k]);
            const I strength=I(masses[j])*(I(1)+I(4)*other*inverse_delta*inverse_delta);
            const I local=I(nearest)+I(sqrt_upper(strength))+I(2)*other*inverse_delta;
            bound=std::max(bound,local.hi);
        }
        if(std::isfinite(bound) && bound>0)radius_=bound;
    }
    bool valid()const{return std::isfinite(radius_);}
    double radius()const{return radius_;}
    std::pair<double,int> clearance(Complex source)const {
        if(!valid() || !interval_detail::environment_ok() ||
           !std::isfinite(source.real()) || !std::isfinite(source.imag()))return {0,-1};
        // Refusal only: avoid interval work for ordinary near-lens epochs.
        for(auto lens:lenses_)if(std::abs(source-lens)<=radius_)return {0,-1};
        std::array<I,3> x,y;
        std::array<double,3> distance;
        for(int j=0;j<3;++j) {
            x[j]=I(lenses_[j].real())-I(source.real());
            y[j]=I(lenses_[j].imag())-I(source.imag());
            distance[j]=norm_lower(x[j],y[j]);
        }
        double best=0;
        // Four exact unit directions avoid trigonometric/unit-vector error.
        for(int axis=0;axis<2;++axis)for(int sign:{-1,1}) {
            double nearest=INFINITY;
            for(int j=0;j<3;++j) {
                I along=axis?y[j]:x[j];
                if(sign<0)along=-along;
                const I across=axis?x[j]:y[j];
                // If the projection sign is uncertain, the perpendicular
                // lower bound is conservative on either half of the ray.
                nearest=std::min(nearest,along.hi<0?distance[j]:absolute_lower(across));
            }
            const double lower=interval_detail::down(nearest-radius_);
            if(std::isfinite(lower))best=std::max(best,lower);
        }
        return best>0?std::pair<double,int>{best,4}:std::pair<double,int>{0,-1};
    }
};
}
#if defined(__GNUC__)
#pragma GCC pop_options
#endif

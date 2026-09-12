#pragma once

// Research candidate for certifying a complete D14 root set without asking
// the simultaneous Aberth iteration to meet a global step threshold.  Every
// coefficient and Taylor-shift operation below uses outward-rounded
// binary128 intervals constructed directly from the binary64 PrimaryFrame.

#include <array>
#include <quadmath.h>

#include "lcbinint/magnification/holonomic/d14_real_sturm.hpp"
#include "lcbinint/magnification/holonomic/d14_structure.hpp"

namespace lcbinint::holonomic::re_detail {

using qf=__float128;
using D14RoucheInterval=positive_detail::Interval<__float128>;

struct D14RoucheComplex {
    D14RoucheInterval re{},im{};
};

inline D14RoucheComplex operator+(D14RoucheComplex a,D14RoucheComplex b) {
    return {a.re+b.re,a.im+b.im};
}
inline D14RoucheComplex operator-(D14RoucheComplex a,D14RoucheComplex b) {
    return {a.re-b.re,a.im-b.im};
}
inline D14RoucheComplex operator*(D14RoucheComplex a,D14RoucheComplex b) {
    return {a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re};
}

inline D14RoucheComplex d14_rouche_point(Cplx<qf> z) {
    return {positive_detail::point<D14RoucheInterval>(z.re),
            positive_detail::point<D14RoucheInterval>(z.im)};
}

inline qf d14_interval_min_abs(D14RoucheInterval x) {
    if(x.lo<=0&&x.hi>=0)return 0;
    return fminq(fabsq(x.lo),fabsq(x.hi));
}

inline qf d14_rouche_abs_lower(D14RoucheComplex z) {
    using positive_detail::down;
    const qf xr=d14_interval_min_abs(z.re),xi=d14_interval_min_abs(z.im);
    const qf square=down(down(xr*xr)+down(xi*xi));
    return down(sqrtq(fmaxq((qf)0,square)));
}

inline qf d14_rouche_abs_upper(D14RoucheComplex z) {
    using positive_detail::up;
    const qf xr=fmaxq(fabsq(z.re.lo),fabsq(z.re.hi));
    const qf xi=fmaxq(fabsq(z.im.lo),fabsq(z.im.hi));
    const qf square=up(up(xr*xr)+up(xi*xi));
    return up(sqrtq(square));
}

struct D14RoucheCertificate {
    bool certified=false;
    int isolated_disks=0;
    int failed_root=-1;
    std::array<bool,14> isolated{};
    std::array<qf,14> radius{};
    std::array<Cplx<qf>,14> center{};
    qf minimum_ratio=HUGE_VALQ;
};

// Reject-only point-arithmetic prefilter.  A pass has no correctness meaning;
// it merely avoids paying for interval coefficient construction when even
// the optimistic binary128 inequality cannot isolate all candidates.
inline bool d14_rouche_fast_screen(const std::vector<qf>& desc_v,
                                   const std::vector<Cplx<qf>>& roots) {
    if(desc_v.size()!=15||roots.size()!=14)return false;
    std::array<Cplx<qf>,14> centers{};
    for(size_t i=0;i<14;++i) {
        if(!finiteq(roots[i].re)||!finiteq(roots[i].im))return false;
        const qf real_cut=(qf)1e-8*((qf)1+fabsq(roots[i].re));
        centers[i]=fabsq(roots[i].im)<=real_cut?Cplx<qf>{roots[i].re,0}:roots[i];
    }
    for(size_t i=0;i<14;++i) {
        std::array<Cplx<qf>,15> shifted{};
        for(int k=0;k<=14;++k) {
            Cplx<qf> power{1,0};
            for(int exponent=k;exponent>=0;--exponent) {
                int choose=1;
                for(int j=1;j<=exponent;++j)choose=choose*(k-j+1)/j;
                shifted[exponent]=shifted[exponent]+power*Cplx<qf>{
                    desc_v[14-k]*(qf)choose,0};
                power=power*centers[i];
            }
        }
        qf separation=HUGE_VALQ;
        for(size_t j=0;j<14;++j)if(i!=j)
            separation=fminq(separation,cabs(centers[i]-centers[j]));
        const qf linear=cabs(shifted[1]);
        if(!finiteq(linear)||!(linear>0)||!finiteq(separation)||!(separation>0))
            return false;
        qf radius=fmaxq((qf)16*cabs(shifted[0])/linear,
                        (qf)1e-30*((qf)1+cabs(centers[i])));
        bool isolated=false;
        for(int attempt=0;attempt<24&&radius<separation/(qf)3;++attempt) {
            const qf lhs=linear*radius;
            qf rhs=cabs(shifted[0]),power=radius*radius;
            for(int k=2;k<=14;++k){rhs+=cabs(shifted[k])*power;power*=radius;}
            if(finiteq(lhs)&&finiteq(rhs)&&lhs>(qf)16*rhs){isolated=true;break;}
            radius*=2;
        }
        if(!isolated)return false;
    }
    return true;
}

inline std::array<D14RoucheComplex,15> d14_rouche_shift(
    const positive_detail::Poly<D14RoucheInterval>& polynomial,
    Cplx<qf> center) {
    std::array<D14RoucheComplex,15> shifted{};
    for(int k=0;k<=14;++k)shifted[k]={polynomial.c[k],{}};
    const auto c=d14_rouche_point(center);
    // In-place Taylor shift (Horner/Pascal form).  After pass k, one more
    // derivative coefficient has accumulated; degree 14 needs 105 complex
    // interval multiplies instead of explicitly forming every c^(j-k).
    for(int k=0;k<14;++k)
        for(int j=13;j>=k;--j)shifted[j]=shifted[j]+c*shifted[j+1];
    return shifted;
}

inline D14RoucheCertificate d14_rouche_certificate(
    const PrimaryFrame& pf,const std::vector<Cplx<qf>>& roots,
    bool collect_all=false) {
    using positive_detail::down;
    using positive_detail::up;
    D14RoucheCertificate out;
    if(roots.size()!=14||!positive_detail::environment_ok())return out;
    for(size_t i=0;i<roots.size();++i) {
        if(!finiteq(roots[i].re)||!finiteq(roots[i].im))return out;
        const qf real_cut=(qf)1e-8*((qf)1+fabsq(roots[i].re));
        out.center[i]=fabsq(roots[i].im)<=real_cut
            ? Cplx<qf>{roots[i].re,0}:roots[i];
    }
    const auto polynomial=positive_detail::polynomial<D14RoucheInterval>(pf);
    if(polynomial.degree!=14)return out;
    for(size_t i=0;i<roots.size();++i) {
        const auto shifted=d14_rouche_shift(polynomial,out.center[i]);
        qf separation=HUGE_VALQ;
        for(size_t j=0;j<roots.size();++j)if(i!=j) {
            const auto difference=d14_rouche_point(out.center[i])-
                                  d14_rouche_point(out.center[j]);
            separation=fminq(separation,d14_rouche_abs_lower(difference));
        }
        const qf linear_lower=d14_rouche_abs_lower(shifted[1]);
        if(!finiteq(linear_lower)||!(linear_lower>0)||
           !finiteq(separation)||!(separation>0)) {
            if(out.failed_root<0)out.failed_root=(int)i;
            if(!collect_all)return out;
            continue;
        }
        const qf constant_upper=d14_rouche_abs_upper(shifted[0]);
        qf radius=up(up((qf)32*constant_upper)/linear_lower);
        radius=fmaxq(radius,(qf)1e-30*((qf)1+cabs(out.center[i])));
        bool isolated=false;
        for(int attempt=0;attempt<24&&up((qf)3*radius)<separation;++attempt) {
            const qf lhs=down(linear_lower*radius);
            qf rhs=constant_upper,power=up(radius*radius);
            for(int k=2;k<=14;++k) {
                rhs=up(rhs+up(d14_rouche_abs_upper(shifted[k])*power));
                power=up(power*radius);
            }
            if(finiteq(lhs)&&finiteq(rhs)&&lhs>up((qf)16*rhs)) {
                isolated=true;out.radius[i]=radius;out.isolated[i]=true;
                out.minimum_ratio=fminq(out.minimum_ratio,down(lhs/rhs));
                break;
            }
            radius=up((qf)2*radius);
        }
        if(!isolated){
            out.radius[i]=radius;
            if(out.failed_root<0)out.failed_root=(int)i;
            if(!collect_all)return out;
            continue;
        }
        ++out.isolated_disks;
    }
    out.certified=out.isolated_disks==14;
    return out;
}

struct D14RoucheClusterCertificate {
    bool certified=false;
    int root_count=0;
    Cplx<qf> center{};
    qf radius=0;
    qf ratio=0;
};

inline D14RoucheClusterCertificate d14_rouche_cluster_certificate(
    const PrimaryFrame& pf,const std::vector<Cplx<qf>>& roots,
    const std::vector<int>& indices) {
    using positive_detail::down;
    using positive_detail::up;
    D14RoucheClusterCertificate out;
    out.root_count=(int)indices.size();
    if(indices.size()<2||indices.size()>14||roots.size()!=14||
       !positive_detail::environment_ok())return out;
    for(int i:indices) {
        if(i<0||i>=14||!finiteq(roots[i].re)||!finiteq(roots[i].im))return out;
        out.center=out.center+roots[i];
    }
    out.center.re=out.center.re/(qf)indices.size();
    out.center.im=out.center.im/(qf)indices.size();
    qf inner=0,outer=HUGE_VALQ;
    std::array<bool,14> member{};for(int i:indices)member[i]=true;
    for(int i=0;i<14;++i) {
        const qf distance=cabs(roots[i]-out.center);
        if(member[i])inner=fmaxq(inner,distance);
        else outer=fminq(outer,distance);
    }
    if(!finiteq(inner)||!finiteq(outer)||!(outer>inner))return out;
    const auto polynomial=positive_detail::polynomial<D14RoucheInterval>(pf);
    if(polynomial.degree!=14)return out;
    const auto shifted=d14_rouche_shift(polynomial,out.center);
    qf radius=positive_detail::up((qf)1.25*inner);
    for(int attempt=0;attempt<24&&positive_detail::up((qf)2*radius)<outer;
        ++attempt) {
        std::array<qf,15> power{};power[0]=1;
        for(int k=1;k<=14;++k)power[k]=up(power[k-1]*radius);
        const int degree=(int)indices.size();
        const qf lhs=down(d14_rouche_abs_lower(shifted[degree])*
                          positive_detail::down(power[degree]));
        qf rhs=0;
        for(int k=0;k<=14;++k)if(k!=degree)
            rhs=up(rhs+up(d14_rouche_abs_upper(shifted[k])*power[k]));
        if(finiteq(lhs)&&finiteq(rhs)&&lhs>up((qf)4*rhs)) {
            out.certified=true;out.radius=radius;
            out.ratio=down(lhs/rhs);return out;
        }
        radius=up((qf)1.5*radius);
    }
    return out;
}

inline bool d14_rouche_refine(const D14StructQf& sc,
                              const D14RoucheCertificate& certificate,
                              std::vector<Cplx<qf>>& roots) {
    if(!certificate.certified||roots.size()!=14)return false;
    for(size_t i=0;i<roots.size();++i) {
        Cplx<qf> z=certificate.center[i];
        for(int iteration=0;iteration<2;++iteration) {
            Cplx<qf> p,dp;d14_struct_eval(sc,z,p,dp);
            if(!(cabs(dp)>0)||!finiteq(cabs(p))||!finiteq(cabs(dp)))return false;
            const Cplx<qf> step=p/dp;
            if(!finiteq(step.re)||!finiteq(step.im)||
               cabs((z-step)-certificate.center[i])>=certificate.radius[i])
                return false;
            z=z-step;
        }
        roots[i]=z;
    }
    return true;
}

} // namespace lcbinint::holonomic::re_detail

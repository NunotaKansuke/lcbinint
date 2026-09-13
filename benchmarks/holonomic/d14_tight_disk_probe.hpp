#pragma once
// Research copy of the incumbent interval certificate. Strict outward
// lhs>rhs still proves one zero by Rouche; separation/3 is unchanged.
#include "lcbinint/magnification/holonomic/d14_rouche.hpp"
namespace lcbinint::holonomic::re_detail {
inline bool tight_fast_screen(const std::vector<qf>& desc_v,
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
        qf radius=fmaxq((qf)2*cabs(shifted[0])/linear,
                        (qf)1e-30*((qf)1+cabs(centers[i])));
        bool isolated=false;
        for(int attempt=0;attempt<24&&radius<separation/(qf)3;++attempt) {
            const qf lhs=linear*radius;
            qf rhs=cabs(shifted[0]),power=radius*radius;
            for(int k=2;k<=14;++k){rhs+=cabs(shifted[k])*power;power*=radius;}
            if(finiteq(lhs)&&finiteq(rhs)&&lhs>rhs){isolated=true;break;}
            radius*=2;
        }
        if(!isolated)return false;
    }
    return true;
}

// Reject-only native screen. A pass always requires the qf interval verifier.
inline bool native_tight_screen(const std::vector<qf>& desc,
                               const std::vector<Cplx<qf>>& roots) {
    if(desc.size()!=15 || roots.size()!=14)return false;
    using R=long double;using C=Cplx<R>;
    auto norm=[](C z){return std::hypot(z.re,z.im);};
    std::array<C,14> centers{};std::array<R,15> coeff{};
    for(int k=0;k<15;++k){coeff[k]=R(desc[k]);if(!std::isfinite(coeff[k]))return false;}
    for(int i=0;i<14;++i){
        const qf cut=(qf)1e-8*((qf)1+fabsq(roots[i].re));
        centers[i]={R(roots[i].re),fabsq(roots[i].im)<=cut?R(0):R(roots[i].im)};
        if(!std::isfinite(centers[i].re)||!std::isfinite(centers[i].im))return false;
    }
    for(int i=0;i<14;++i){
        // Triangular Taylor shift, sharing the polynomial's Horner work.
        std::array<C,15> b{};b[0]={coeff[0],0};
        for(int k=1;k<=14;++k){
            b[k]=b[k-1];
            for(int j=k-1;j>=1;--j)b[j]=b[j]*centers[i]+b[j-1];
            b[0]=b[0]*centers[i]+C(coeff[k],0);
        }
        R separation=INFINITY;
        for(int j=0;j<14;++j)if(i!=j)separation=std::min(separation,norm(centers[i]-centers[j]));
        const R linear=norm(b[1]);
        if(!(linear>0)||!std::isfinite(linear)||!(separation>0)||!std::isfinite(separation))return false;
        R radius=std::max(2*norm(b[0])/linear,R(1e-30L)*(1+norm(centers[i])));bool isolated=false;
        for(int attempt=0;attempt<24&&radius<separation/3;++attempt){
            R lhs=linear*radius,rhs=norm(b[0]),power=radius*radius;
            for(int k=2;k<=14;++k){rhs+=norm(b[k])*power;power*=radius;}
            if(std::isfinite(lhs)&&std::isfinite(rhs)&&lhs>rhs){isolated=true;break;}
            radius*=2;
        }
        if(!isolated)return false;
    }
    return true;
}

inline D14RoucheInterval point_times_interval(qf a,D14RoucheInterval b) {
    using positive_detail::down;using positive_detail::up;
    if(a==0 || b.zero())return {};
    // Preserve the incumbent behavior on invalid/unbounded operands.
    if(!finiteq(a)||!finiteq(b.lo)||!finiteq(b.hi)||b.lo>b.hi)
        return D14RoucheInterval(a,a)*b;
    if(a>0)return {down(a*b.lo),up(a*b.hi)};
    return {down(a*b.hi),up(a*b.lo)};
}
inline std::array<D14RoucheComplex,15> point_interval_shift(
    const positive_detail::Poly<D14RoucheInterval>& polynomial,Cplx<qf> center) {
    std::array<D14RoucheComplex,15> b{};
    for(int k=0;k<=14;++k)b[k]={polynomial.c[k],{}};
    for(int k=0;k<14;++k)for(int j=13;j>=k;--j){
        const auto x=b[j+1];D14RoucheComplex product;
        if(center.im==0){
            product={point_times_interval(center.re,x.re),point_times_interval(center.re,x.im)};
        } else {
            product={point_times_interval(center.re,x.re)-point_times_interval(center.im,x.im),
                     point_times_interval(center.re,x.im)+point_times_interval(center.im,x.re)};
        }
        b[j]=b[j]+product;
    }
    return b;
}

inline D14RoucheCertificate tight_certificate(
    const PrimaryFrame& pf,const std::vector<Cplx<qf>>& roots,
    bool collect_all=false,bool point_shift=false) {
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
        const auto shifted=point_shift?point_interval_shift(polynomial,out.center[i]):d14_rouche_shift(polynomial,out.center[i]);
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
        qf radius=up(up((qf)2*constant_upper)/linear_lower);
        radius=fmaxq(radius,(qf)1e-30*((qf)1+cabs(out.center[i])));
        bool isolated=false;
        for(int attempt=0;attempt<24&&up((qf)3*radius)<separation;++attempt) {
            const qf lhs=down(linear_lower*radius);
            qf rhs=constant_upper,power=up(radius*radius);
            for(int k=2;k<=14;++k) {
                rhs=up(rhs+up(d14_rouche_abs_upper(shifted[k])*power));
                power=up(power*radius);
            }
            if(finiteq(lhs)&&finiteq(rhs)&&lhs>rhs) {
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


}

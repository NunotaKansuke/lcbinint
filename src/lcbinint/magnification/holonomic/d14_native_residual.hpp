#pragma once
// Research-only L1 forward-error enclosure of the incumbent qf Horner result.
// Arithmetic audit first: this kernel does not replace a solver gate by itself.
#include <array>
#include <cfenv>
#include <limits>
#include "lcbinint/magnification/holonomic/poly_roots.hpp"
namespace lcbinint::holonomic::native_residual_detail {
using L=long double;
constexpr L eps=std::numeric_limits<L>::epsilon();
constexpr L inflate=1+16*eps;
inline L plus(L a,L b){return (a+b)*inflate;}
inline L times(L a,L b){return (a*b)*inflate;}
inline bool convert(__float128 q,L& x){
    x=static_cast<L>(q);
    if(x==0)return q==0;
    // With degree<=14 this range keeps products and error radii normal
    // in the binary80 exponent range, including cancellation/roundoff.
    return std::isfinite(x)&&std::fabs(x)>=1e-200L&&std::fabs(x)<=1e200L;
}
struct Bound {bool valid=false;L upper=0;std::array<L,14> root_upper{};unsigned mask=0;};
inline Bound evaluate(const __float128* c,int degree,
                      const std::vector<Cplx<__float128>>& roots,
                      __float128 scale){
#ifdef __FAST_MATH__
    return {};
#endif
    if(std::numeric_limits<L>::digits!=64 || std::numeric_limits<L>::max_exponent<16384 ||
       std::fegetround()!=FE_TONEAREST || degree<1||degree>14||roots.size()!=static_cast<size_t>(degree))return {};
#if defined(__i386__) || defined(__x86_64__)
    unsigned short control;__asm__ volatile("fnstcw %0":"=m"(control));
    if((control&0x0300)!=0x0300)return {}; // binary80 significand precision
#else
    return {};
#endif
    std::array<L,15> a{};L sc;
    if(!convert(scale,sc)||!(sc>0))return {};
    for(int k=0;k<=degree;++k)if(!convert(c[k],a[k]))return {};
    const L denominator=sc*(1-16*eps); // lower than exact qf scale; omits positive 1e-300
    Bound out;
    for(int i=0;i<degree;++i){
        const auto& z=roots[i];
        L xr,xi;if(!convert(z.re,xr)||!convert(z.im,xi))continue;
        const L X=plus(std::fabs(xr),std::fabs(xi));
        const L Xtrue=times(X,inflate);
        const L dx=times(eps,X);
        L br=a[0],bi=0,error=times(eps,std::fabs(a[0]));
        for(int k=1;k<=degree;++k){
            const L B=plus(std::fabs(br),std::fabs(bi));
            const L C=std::fabs(a[k]);
            // Covers both binary80 and qf multiply/add roundoff, plus the
            // coefficient conversion. No cancellation-based residual test.
            const L arithmetic=times(32*eps,plus(times(X,B),C));
            error=plus(plus(times(Xtrue,error),times(B,dx)),
                       plus(times(eps,C),arithmetic));
            const L nr=br*xr-bi*xi+a[k];
            const L ni=br*xi+bi*xr;
            br=nr;bi=ni;
        }
        // L1 bounds the qf Euclidean norm. Inflation also covers its norm
        // and normalization rounding (qf unit roundoff << binary80 eps).
        const L norm=plus(plus(std::fabs(br),std::fabs(bi)),error);
        if(norm>1e2400L || (norm>0 && norm<1e-2400L))continue; // exclude qf squared-norm range edges
        const L bound=(norm/denominator)*inflate;
        if(!std::isfinite(bound)||bound<0)continue;
        out.root_upper[i]=bound;out.mask|=1u<<i;
        out.upper=std::max(out.upper,bound);
    }
    out.valid=out.mask==((1u<<degree)-1);
    return out;
}
}
